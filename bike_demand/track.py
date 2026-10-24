"""Record a training run and its saved files in a local MLflow experiment."""

import argparse
import json
import logging
import subprocess
from pathlib import Path

import mlflow
import pandas as pd

from bike_demand.train import run_training

LOGGER = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parents[1]


def git_tags() -> dict[str, str]:
    """Identify the code revision and report any uncommitted changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_DIR,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        changes = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_DIR,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"git_commit": "unknown", "git_dirty": "unknown"}
    return {"git_commit": commit, "git_dirty": str(bool(changes)).lower()}


def run_tracked_training(
    data_path: Path,
    output_dir: Path,
    validation_start: pd.Timestamp,
    test_start: pd.Timestamp,
    tracking_uri: str = "sqlite:///mlflow.db",
    run_name: str | None = None,
    test_end: pd.Timestamp | None = None,
) -> str:
    """Train once, save files under a unique run ID, and return that ID."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("bike-demand")

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        LOGGER.info("MLflow run: %s", run_id)
        mlflow.set_tags(git_tags())
        mlflow.log_params(
            {
                "validation_start": str(validation_start),
                "test_start": str(test_start),
                "mlflow_version": mlflow.__version__,
            }
        )
        if test_end is not None:
            mlflow.log_param("test_end_exclusive", str(test_end))
        run_dir = output_dir / run_id
        metrics = run_training(
            data_path, run_dir, validation_start, test_start, test_end
        )
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        mlflow.log_params(metadata["model_parameters"])
        mlflow.log_params(
            {
                key: metadata[key]
                for key in [
                    "data_sha256",
                    "training_target_end",
                    "forecast_horizon_hours",
                    "python_version",
                ]
            }
        )
        for split in ["validation", "test"]:
            for model_name, scores in metrics[split].items():
                mlflow.log_metrics(
                    {
                        f"{split}.{model_name}.{name}": value
                        for name, value in scores.items()
                    }
                )
        mlflow.log_metrics(
            {key: metrics[key] for key in ["test_forecast_hours", "test_coverage_pct"]}
        )
        for filename in [
            "model.joblib",
            "metrics.json",
            "metadata.json",
            "test_predictions.csv",
        ]:
            mlflow.log_artifact(str(run_dir / filename), artifact_path="training")
        mlflow.log_artifact(
            str(PROJECT_DIR / "requirements.txt"), artifact_path="training"
        )

    return run_id


def main() -> None:
    """Use the same data and split defaults as the plain training command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path("data/hour.csv"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Parent directory for training files, grouped by MLflow run ID.",
    )
    parser.add_argument("--validation-start", type=pd.Timestamp, default="2012-07-01")
    parser.add_argument("--test-start", type=pd.Timestamp, default="2012-10-01")
    parser.add_argument(
        "--test-end",
        type=pd.Timestamp,
        default=None,
        help="Optional exclusive evaluation end.",
    )
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_tracked_training(
        args.data_path,
        args.output_dir,
        args.validation_start,
        args.test_start,
        args.tracking_uri,
        args.run_name,
        args.test_end,
    )


if __name__ == "__main__":
    main()
