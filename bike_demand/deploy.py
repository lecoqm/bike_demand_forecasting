"""Select an accepted model for the next API startup, or restore its predecessor."""

import argparse
import hashlib
import json
import logging
import math
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from bike_demand.features import make_features

LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL_PATH = Path("artifacts/run_01/model.joblib")
DEFAULT_STATE_PATH = Path("artifacts/deployment/state.json")


def model_version(model_bytes: bytes) -> str:
    """Identify the exact serialized model, as in API responses and request logs."""
    return "sha256:" + hashlib.sha256(model_bytes).hexdigest()


def snapshot_path(reference: dict, state_path: Path) -> Path:
    """Resolve only the content-addressed path recorded for this model version."""
    version = reference["model_version"]
    if not isinstance(version, str) or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", version
    ):
        raise ValueError("Invalid model version in deployment state.")
    relative = Path("models") / version.removeprefix("sha256:") / "model.joblib"
    if reference["model_path"] != relative.as_posix():
        raise ValueError("Invalid model snapshot path in deployment state.")
    return state_path.parent / relative


def read_state(state_path: Path) -> dict:
    """Fail visibly for damaged state instead of choosing another model silently."""
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if (
        state["schema_version"] != 1
        or not isinstance(state["previous_models"], list)
        or not isinstance(state["history"], list)
    ):
        raise ValueError("Unsupported deployment state.")
    for reference in [state["active"], *state["previous_models"]]:
        snapshot_path(reference, state_path)
    return state


def selected_model(
    state_path: Path, explicit_model_path: Path | None = None
) -> tuple[Path, str | None]:
    """Explicit model selection wins; otherwise use managed state when present."""
    if explicit_model_path is not None:
        return explicit_model_path, None
    if not state_path.exists():
        return DEFAULT_MODEL_PATH, None
    active = read_state(state_path)["active"]
    return snapshot_path(active, state_path), active["model_version"]


def checked_model_bytes(path: Path, expected_version: str | None = None) -> bytes:
    """Read once and reject a snapshot that differs from the recorded bytes."""
    model_bytes = path.read_bytes()
    if expected_version is not None and model_version(model_bytes) != expected_version:
        raise ValueError(f"Model checksum mismatch: {path}")
    return model_bytes


def check_model(model_bytes: bytes) -> None:
    """Check the serving features and one finite, nonnegative smoke prediction."""
    bundle = joblib.load(BytesIO(model_bytes))
    history = pd.DataFrame(
        {"cnt": [25] * 168 + [None]},
        index=pd.date_range(end="2012-12-01", periods=169, freq="h"),
    )
    features = make_features(history).iloc[[-1]]
    if bundle["feature_columns"] != features.columns.tolist():
        raise ValueError("Model features do not match the current feature builder.")
    with threadpool_limits(limits=1):
        prediction = np.asarray(bundle["estimator"].predict(features), dtype=float)
    if (
        prediction.shape != (1,)
        or not np.isfinite(prediction).all()
        or prediction[0] < 0
    ):
        raise ValueError("Model failed the serving smoke prediction.")


def validate_comparison(comparison_dir: Path) -> tuple[dict, bytes]:
    """Recheck the fixed rule and paired CSV scores before selecting a candidate."""
    # Keep MLflow and training imports out of API startup.
    from bike_demand.compare import ACCEPTANCE_RULE, candidate_decision

    report_bytes = (comparison_dir / "comparison.json").read_bytes()
    report = json.loads(report_bytes)
    plan = json.loads((comparison_dir / "plan.json").read_text(encoding="utf-8"))
    if report["completed"] is not True or plan["acceptance_rule"] != ACCEPTANCE_RULE:
        raise ValueError(
            "Require a completed comparison with the fixed acceptance rule."
        )
    scores = report["metrics"]
    for name in ["current", "candidate"]:
        for metric in ["mae", "rmse"]:
            value = scores[name][metric]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    "Comparison scores must be finite, nonnegative numbers."
                )
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    "Comparison scores must be finite, nonnegative numbers."
                )
    decision = candidate_decision(scores["current"], scores["candidate"])
    if not decision["accepted"] or report["decision"] != decision:
        raise ValueError("Candidate does not pass the recorded acceptance rule.")
    if plan["current_model_version"] != report["current"]["model_version"]:
        raise ValueError("Comparison plan and current model version disagree.")
    for name in ["current", "candidate"]:
        trained_until = pd.Timestamp(report[name]["training_target_end"])
        if not trained_until < pd.Timestamp(report["comparison_start"]):
            raise ValueError(
                "Comparison hours must follow both models' training targets."
            )

    predictions = pd.read_csv(
        comparison_dir / "comparison_predictions.csv",
        index_col="target_hour",
        parse_dates=True,
        dtype={"eligible": "boolean"},
    )
    scheduled = pd.date_range(
        report["comparison_start"],
        report["comparison_end_exclusive"],
        freq="h",
        inclusive="left",
        name="target_hour",
    )
    if not predictions.index.equals(scheduled) or predictions["eligible"].isna().any():
        raise ValueError(
            "Comparison CSV must contain each scheduled hour exactly once."
        )
    paired = predictions.loc[
        predictions["eligible"], ["actual", "current", "candidate"]
    ]
    if (
        paired.empty
        or not np.isfinite(paired.to_numpy()).all()
        or (paired < 0).any().any()
    ):
        raise ValueError("Comparison requires finite paired predictions and outcomes.")
    if report["coverage"]["paired_scored_hours"] != len(paired) or report["coverage"][
        "scheduled_hours"
    ] != len(scheduled):
        raise ValueError("Comparison coverage differs from its paired predictions.")
    for name in ["current", "candidate"]:
        errors = paired[name] - paired["actual"]
        measured = {
            "mae": float(errors.abs().mean()),
            "rmse": float(np.sqrt((errors**2).mean())),
        }
        if any(
            not math.isclose(
                measured[key], scores[name][key], rel_tol=1e-9, abs_tol=1e-9
            )
            for key in measured
        ):
            raise ValueError("Comparison scores differ from the paired predictions.")
    return report, report_bytes


def atomic_write(path: Path, contents: bytes) -> None:
    """Close and flush a temporary file before replacing it, including on Windows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".pending-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def deployment_lock(state_path: Path):
    """Allow only one local promotion or rollback to update state at a time."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = state_path.with_suffix(".lock")
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise FileExistsError(
            f"Deployment is locked: {lock_path}. See docs/deployment.md."
        ) from None
    try:
        os.close(descriptor)
        yield
    finally:
        lock_path.unlink()


def save_snapshot(
    model_bytes: bytes, source: Path, state_path: Path, run_id: str | None = None
) -> dict:
    """Keep a byte-identical local copy independent of the training directory."""
    version = model_version(model_bytes)
    reference = {
        "model_version": version,
        "model_path": f"models/{version.removeprefix('sha256:')}/model.joblib",
        "source_path": str(source.resolve()),
        "run_id": run_id,
    }
    path = snapshot_path(reference, state_path)
    if path.exists():
        checked_model_bytes(path, version)
    else:
        atomic_write(path, model_bytes)
    return reference


def save_state(
    state: dict, state_path: Path, action: str, previous: dict, evidence: dict
) -> dict:
    """Commit the selected version and its audit event in one file replacement."""
    state["history"].append(
        {
            "event_id": uuid4().hex,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "action": action,
            "from_version": previous["model_version"],
            "to_version": state["active"]["model_version"],
            **evidence,
        }
    )
    atomic_write(
        state_path,
        (json.dumps(state, indent=2, allow_nan=False) + "\n").encode("utf-8"),
    )
    return state


def promote(
    comparison_dir: Path,
    state_path: Path = DEFAULT_STATE_PATH,
    initial_model_path: Path = DEFAULT_MODEL_PATH,
) -> dict:
    """Select a validated candidate only if it was compared with the active version."""
    report, report_bytes = validate_comparison(comparison_dir)
    candidate_source = Path(report["candidate"]["model_path"])
    candidate_bytes = checked_model_bytes(
        candidate_source, report["candidate"]["model_version"]
    )
    check_model(candidate_bytes)
    with deployment_lock(state_path):
        if state_path.exists():
            state = read_state(state_path)
            current = state["active"]
            current_bytes = checked_model_bytes(
                snapshot_path(current, state_path), current["model_version"]
            )
        else:
            current_bytes = checked_model_bytes(
                initial_model_path, report["current"]["model_version"]
            )
            check_model(current_bytes)
            current = save_snapshot(current_bytes, initial_model_path, state_path)
            state = {
                "schema_version": 1,
                "active": current,
                "previous_models": [],
                "history": [],
            }
        if model_version(current_bytes) != report["current"]["model_version"]:
            raise ValueError(
                "Comparison is stale: its current model is not the active version."
            )
        if model_version(candidate_bytes) == current["model_version"]:
            raise ValueError("Candidate is already the current model.")
        candidate = save_snapshot(
            candidate_bytes, candidate_source, state_path, report["candidate"]["run_id"]
        )
        state["previous_models"].append(current)
        state["active"] = candidate
        return save_state(
            state,
            state_path,
            "promote",
            current,
            {
                "comparison_path": str((comparison_dir / "comparison.json").resolve()),
                "comparison_sha256": hashlib.sha256(report_bytes).hexdigest(),
                "candidate_run_id": report["candidate"]["run_id"],
            },
        )


def rollback(state_path: Path = DEFAULT_STATE_PATH) -> dict:
    """Restore the latest predecessor from a verified local snapshot."""
    with deployment_lock(state_path):
        state = read_state(state_path)
        if not state["previous_models"]:
            raise ValueError("No previous model is available for rollback.")
        previous = state["active"]
        restored = state["previous_models"].pop()
        model_bytes = checked_model_bytes(
            snapshot_path(restored, state_path), restored["model_version"]
        )
        check_model(model_bytes)
        state["active"] = restored
        return save_state(state, state_path, "rollback", previous, {})


def status(state_path: Path = DEFAULT_STATE_PATH) -> dict:
    """Describe selection for the next startup; query /health for a running API."""
    override = os.environ.get("BIKE_DEMAND_MODEL_PATH")
    state = read_state(state_path) if state_path.exists() else None
    if override:
        selection, path, expected = "environment", Path(override), None
    elif state is not None:
        selection = "managed"
        path = snapshot_path(state["active"], state_path)
        expected = state["active"]["model_version"]
    else:
        selection, path, expected = "initial", DEFAULT_MODEL_PATH, None
    model_bytes = checked_model_bytes(path, expected)
    result = {
        "selection": selection,
        "selected_model_path": str(path.resolve()),
        "selected_model_version": model_version(model_bytes),
        "state_path": str(state_path.resolve()),
    }
    if state is not None:
        result["managed_model_version"] = state["active"]["model_version"]
        result["rollback_versions"] = [
            item["model_version"] for item in state["previous_models"]
        ]
        result["history_events"] = len(state["history"])
    return result


def main() -> None:
    """Expose explicit local promote, rollback, and status operations."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["promote", "rollback", "status"]:
        command = commands.add_parser(name)
        command.add_argument(
            "--state-path",
            type=Path,
            default=Path(
                os.environ.get("BIKE_DEMAND_DEPLOYMENT_PATH", str(DEFAULT_STATE_PATH))
            ),
        )
        if name == "promote":
            command.add_argument(
                "--comparison-dir",
                type=Path,
                default=Path("artifacts/comparisons/october_update"),
            )
            command.add_argument(
                "--initial-model-path", type=Path, default=DEFAULT_MODEL_PATH
            )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        if args.command == "status":
            print(json.dumps(status(args.state_path), indent=2))
            return
        state = (
            promote(args.comparison_dir, args.state_path, args.initial_model_path)
            if args.command == "promote"
            else rollback(args.state_path)
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Deployment failed: {exc}\n")
    LOGGER.info("Selected model: %s", state["active"]["model_version"])
    LOGGER.info("Saved state and history to %s", args.state_path)
    LOGGER.info("Restart the API and check /health to verify the serving version.")
    if os.environ.get("BIKE_DEMAND_MODEL_PATH"):
        LOGGER.warning(
            "BIKE_DEMAND_MODEL_PATH is set and overrides managed selection; clear it in the API terminal before restarting."
        )


if __name__ == "__main__":
    main()
