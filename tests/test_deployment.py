"""Exercise accepted promotion, startup selection, audit history, and rollback."""

import json
import shutil
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyRegressor

from bike_demand import deploy
from bike_demand.api import create_app
from bike_demand.compare import run_comparison
from bike_demand.data import load_hourly_data
from bike_demand.replay import request_for_hour

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_DIR / "artifacts/run_01"


@pytest.fixture(scope="module")
def comparison_template(tmp_path_factory):
    root = tmp_path_factory.mktemp("deployment_comparison")
    previous_uri = mlflow.get_tracking_uri()
    try:
        result = run_comparison(
            PROJECT_DIR / "data/hour.csv",
            MODEL_DIR,
            root / "comparison",
            pd.Timestamp("2012-11-01"),
            pd.Timestamp("2012-12-01"),
            f"sqlite:///{(root / 'mlflow.db').as_posix()}",
        )
        assert result["decision"]["accepted"]
    finally:
        mlflow.end_run()
        mlflow.set_tracking_uri(previous_uri)
    return root / "comparison"


@pytest.fixture
def deployment(tmp_path, comparison_template, monkeypatch):
    monkeypatch.delenv("BIKE_DEMAND_MODEL_PATH", raising=False)
    monkeypatch.delenv("BIKE_DEMAND_DEPLOYMENT_PATH", raising=False)
    monkeypatch.chdir(PROJECT_DIR)
    comparison = tmp_path / "comparison"
    shutil.copytree(comparison_template, comparison)
    report_path = comparison / "comparison.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    initial = tmp_path / "initial" / "model.joblib"
    initial.parent.mkdir()
    initial.write_bytes((MODEL_DIR / "model.joblib").read_bytes())
    report["current"]["model_path"] = str(initial)
    report["candidate"]["model_path"] = str(
        comparison / "candidate" / report["candidate"]["run_id"] / "model.joblib"
    )
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return comparison, tmp_path / "deployment" / "state.json", initial, report


def test_promote_restart_and_rollback_use_copied_models_and_trace_requests(
    deployment, tmp_path
):
    comparison, state_path, initial, report = deployment
    log_path = tmp_path / "requests.jsonl"
    hourly = load_hourly_data(PROJECT_DIR / "data/hour.csv")
    payload = request_for_hour(hourly, pd.Timestamp("2012-11-01"))
    paired = pd.read_csv(comparison / "comparison_predictions.csv").iloc[0]
    source_hashes = {
        path: deploy.model_version(path.read_bytes())
        for path in [initial, Path(report["candidate"]["model_path"])]
    }

    with TestClient(create_app(log_path=log_path, state_path=state_path)) as client:
        assert (
            client.get("/health").json()["model_version"]
            == report["current"]["model_version"]
        )
        state = deploy.promote(comparison, state_path, initial)
        assert client.post("/predict", json=payload).json()[
            "prediction"
        ] == pytest.approx(paired["current"])

    for path, expected in source_hashes.items():
        assert deploy.model_version(path.read_bytes()) == expected
    assert (
        deploy.snapshot_path(state["active"], state_path).read_bytes()
        == Path(report["candidate"]["model_path"]).read_bytes()
    )
    assert (
        deploy.snapshot_path(state["previous_models"][-1], state_path).read_bytes()
        == initial.read_bytes()
    )
    assert state["active"]["run_id"] == report["candidate"]["run_id"]
    assert deploy.status(state_path)["selection"] == "managed"
    assert deploy.status(state_path)["rollback_versions"] == [
        report["current"]["model_version"]
    ]

    # Source cleanup cannot break serving or rollback once the snapshots exist.
    initial.unlink()
    shutil.rmtree(comparison / "candidate")
    with TestClient(create_app(log_path=log_path, state_path=state_path)) as client:
        assert (
            client.get("/health").json()["model_version"]
            == report["candidate"]["model_version"]
        )
        result = client.post("/predict", json=payload).json()
        assert result["prediction"] == pytest.approx(paired["candidate"])
        restored = deploy.rollback(state_path)
        assert (
            client.get("/health").json()["model_version"]
            == report["candidate"]["model_version"]
        )

    with TestClient(create_app(log_path=log_path, state_path=state_path)) as client:
        assert (
            client.get("/health").json()["model_version"]
            == report["current"]["model_version"]
        )
        assert client.post("/predict", json=payload).json()[
            "prediction"
        ] == pytest.approx(paired["current"])
    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [event["model_version"] for event in events] == [
        report[name]["model_version"] for name in ["current", "candidate", "current"]
    ]
    assert [event["action"] for event in restored["history"]] == ["promote", "rollback"]
    assert [event["to_version"] for event in restored["history"]] == [
        report["candidate"]["model_version"],
        report["current"]["model_version"],
    ]
    assert restored["history"][0]["candidate_run_id"] == report["candidate"]["run_id"]
    assert len({event["event_id"] for event in restored["history"]}) == 2
    assert all(
        datetime.fromisoformat(event["timestamp_utc"]).utcoffset().total_seconds() == 0
        for event in restored["history"]
    )
    assert restored["previous_models"] == []
    saved = state_path.read_bytes()
    with pytest.raises(ValueError, match="No previous model"):
        deploy.rollback(state_path)
    assert state_path.read_bytes() == saved


@pytest.mark.parametrize(
    "change", ["unfinished", "decision", "scores", "nan", "overlap", "plan"]
)
def test_promotion_rejects_invalid_comparison_records(deployment, change):
    comparison, state_path, initial, report = deployment
    if change == "unfinished":
        report["completed"] = False
    elif change == "decision":
        report["decision"]["accepted"] = False
    elif change == "scores":
        report["metrics"]["candidate"]["mae"] = report["metrics"]["current"]["mae"]
    elif change == "nan":
        report["metrics"]["candidate"]["mae"] = float("nan")
    elif change == "overlap":
        report["candidate"]["training_target_end"] = report["comparison_start"]
    else:
        plan_path = comparison / "plan.json"
        plan = json.loads(plan_path.read_text())
        plan["current_model_version"] = "sha256:" + "0" * 64
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
    (comparison / "comparison.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        deploy.promote(comparison, state_path, initial)
    assert not state_path.exists()


@pytest.mark.parametrize(
    "change", ["score", "duplicate-hour", "missing-hour", "coverage"]
)
def test_promotion_rechecks_paired_predictions(deployment, change):
    comparison, state_path, initial, report = deployment
    path = comparison / "comparison_predictions.csv"
    predictions = pd.read_csv(path)
    if change == "score":
        predictions.loc[0, "candidate"] += 100
    elif change == "duplicate-hour":
        predictions.loc[1, "target_hour"] = predictions.loc[0, "target_hour"]
    elif change == "missing-hour":
        predictions = predictions.iloc[1:]
    else:
        report["coverage"]["paired_scored_hours"] += 1
        (comparison / "comparison.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
    predictions.to_csv(path, index=False)
    with pytest.raises(ValueError):
        deploy.promote(comparison, state_path, initial)
    assert not state_path.exists()


@pytest.mark.parametrize("which", ["current", "candidate"])
def test_changed_model_bytes_prevent_promotion(deployment, which):
    comparison, state_path, initial, report = deployment
    path = initial if which == "current" else Path(report["candidate"]["model_path"])
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        deploy.promote(comparison, state_path, initial)
    assert not state_path.exists()
    assert not state_path.with_suffix(".lock").exists()


@pytest.mark.parametrize("problem", ["features", "negative-prediction"])
def test_incompatible_or_broken_candidate_cannot_be_selected(deployment, problem):
    comparison, state_path, initial, report = deployment
    path = Path(report["candidate"]["model_path"])
    bundle = joblib.load(path)
    if problem == "features":
        bundle["feature_columns"] = bundle["feature_columns"][:-1]
    else:
        columns = bundle["feature_columns"]
        bundle["estimator"] = DummyRegressor(strategy="constant", constant=-1).fit(
            pd.DataFrame([[0] * len(columns)], columns=columns), [1]
        )
    joblib.dump(bundle, path)
    report["candidate"]["model_version"] = deploy.model_version(path.read_bytes())
    (comparison / "comparison.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="features|smoke prediction"):
        deploy.promote(comparison, state_path, initial)
    assert not state_path.exists()


def test_comparison_for_another_active_model_is_rejected(deployment):
    comparison, state_path, initial, _ = deployment
    deploy.promote(comparison, state_path, initial)
    saved = state_path.read_bytes()
    with pytest.raises(ValueError, match="stale"):
        deploy.promote(comparison, state_path, initial)
    assert state_path.read_bytes() == saved


@pytest.mark.parametrize("source", ["explicit", "environment"])
def test_model_override_precedes_managed_selection(
    deployment, tmp_path, monkeypatch, source
):
    comparison, state_path, initial, report = deployment
    deploy.promote(comparison, state_path, initial)
    if source == "environment":
        monkeypatch.setenv("BIKE_DEMAND_MODEL_PATH", str(initial))
        app = create_app(log_path=tmp_path / "requests.jsonl", state_path=state_path)
        assert deploy.status(state_path)["selection"] == "environment"
    else:
        app = create_app(
            initial, log_path=tmp_path / "requests.jsonl", state_path=state_path
        )
    with TestClient(app) as client:
        assert (
            client.get("/health").json()["model_version"]
            == report["current"]["model_version"]
        )


def test_configured_state_path_selects_the_promoted_model(
    deployment, tmp_path, monkeypatch
):
    comparison, state_path, initial, report = deployment
    deploy.promote(comparison, state_path, initial)
    monkeypatch.setenv("BIKE_DEMAND_DEPLOYMENT_PATH", str(state_path))
    with TestClient(create_app(log_path=tmp_path / "requests.jsonl")) as client:
        assert (
            client.get("/health").json()["model_version"]
            == report["candidate"]["model_version"]
        )


@pytest.mark.parametrize("damage", ["json", "snapshot"])
def test_corrupt_deployment_prevents_api_startup(deployment, tmp_path, damage):
    comparison, state_path, initial, _ = deployment
    state = deploy.promote(comparison, state_path, initial)
    if damage == "json":
        state_path.write_text("{", encoding="utf-8")
    else:
        deploy.snapshot_path(state["active"], state_path).write_bytes(b"damaged")
    with (
        pytest.raises(ValueError),
        TestClient(
            create_app(log_path=tmp_path / "requests.jsonl", state_path=state_path)
        ),
    ):
        pass


def test_corrupt_rollback_snapshot_does_not_change_state(deployment):
    comparison, state_path, initial, _ = deployment
    state = deploy.promote(comparison, state_path, initial)
    saved = state_path.read_bytes()
    deploy.snapshot_path(state["previous_models"][-1], state_path).write_bytes(
        b"damaged"
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        deploy.rollback(state_path)
    assert state_path.read_bytes() == saved


def test_failed_state_replacement_preserves_selection_and_releases_lock(
    deployment, monkeypatch
):
    comparison, state_path, initial, _ = deployment
    deploy.promote(comparison, state_path, initial)
    saved = state_path.read_bytes()
    original_replace = deploy.os.replace

    def failed_replace(source, destination):
        if Path(destination) == state_path:
            raise OSError("simulated state replacement failure")
        original_replace(source, destination)

    monkeypatch.setattr(deploy.os, "replace", failed_replace)
    with pytest.raises(OSError, match="simulated"):
        deploy.rollback(state_path)
    assert state_path.read_bytes() == saved
    assert not state_path.with_suffix(".lock").exists()
    assert list(state_path.parent.glob(".pending-*")) == []


def test_existing_lock_blocks_a_competing_change(deployment):
    comparison, state_path, initial, _ = deployment
    deploy.promote(comparison, state_path, initial)
    saved = state_path.read_bytes()
    lock = state_path.with_suffix(".lock")
    lock.write_text("other command", encoding="utf-8")
    with pytest.raises(FileExistsError, match="locked"):
        deploy.rollback(state_path)
    assert state_path.read_bytes() == saved
    assert lock.read_text() == "other command"


def test_status_before_first_promotion_does_not_create_state(deployment):
    _, state_path, _, report = deployment
    result = deploy.status(state_path)
    assert result["selection"] == "initial"
    assert result["selected_model_version"] == report["current"]["model_version"]
    assert not state_path.exists()
