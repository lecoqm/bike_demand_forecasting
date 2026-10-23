"""Check that tracked runs retain reproducible results and failure status."""

import hashlib
import json
import subprocess
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient
from threadpoolctl import threadpool_limits

from bike_demand.data import load_hourly_data
from bike_demand.features import make_features
from bike_demand.track import run_tracked_training

PROJECT_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture
def tracking_store(tmp_path, monkeypatch):
    """Keep the database and artifact store outside the real project."""
    previous_uri = mlflow.get_tracking_uri()
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    yield uri, MlflowClient(tracking_uri=uri)
    mlflow.end_run()
    mlflow.set_tracking_uri(previous_uri)


def test_run_retains_metrics_provenance_and_reloadable_model(tmp_path, tracking_store):
    uri, client = tracking_store
    data_path = PROJECT_DIR / "data/hour.csv"
    test_start = pd.Timestamp("2012-10-01")
    run_id = run_tracked_training(
        data_path,
        tmp_path / "outputs",
        pd.Timestamp("2012-07-01"),
        test_start,
        tracking_uri=uri,
        run_name="integration-check",
    )

    run = client.get_run(run_id)
    run_dir = tmp_path / "outputs" / run_id
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert run.info.status == "FINISHED"
    assert (
        run.data.params["data_sha256"]
        == hashlib.sha256(data_path.read_bytes()).hexdigest()
    )
    assert run.data.params["test_start"] == str(test_start)
    assert run.data.params["max_iter"] == "250"
    assert (
        run.data.metrics["test.gradient_boosting.mae"]
        == metrics["test"]["gradient_boosting"]["mae"]
    )
    assert (
        run.data.metrics["validation.weekly_baseline.rmse"]
        == metrics["validation"]["weekly_baseline"]["rmse"]
    )
    assert run.data.metrics["test_forecast_hours"] == metrics["test_forecast_hours"]
    expected_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_DIR,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert run.data.tags["git_commit"] == expected_commit
    assert run.data.tags["git_dirty"] in {"true", "false"}
    assert {item.path for item in client.list_artifacts(run_id, "training")} == {
        "training/model.joblib",
        "training/metrics.json",
        "training/metadata.json",
        "training/test_predictions.csv",
        "training/requirements.txt",
    }

    restored_path = client.download_artifacts(run_id, "training/model.joblib")
    restored = joblib.load(restored_path)
    predictions = pd.read_csv(
        run_dir / "test_predictions.csv", index_col="target_hour", parse_dates=True
    )
    features = make_features(load_hourly_data(data_path)).loc[predictions.index]
    with threadpool_limits(limits=2):
        actual = restored["estimator"].predict(features[restored["feature_columns"]])
    np.testing.assert_allclose(
        actual, predictions["gradient_boosting"], rtol=0, atol=1e-10
    )
    assert mlflow.active_run() is None


def test_training_error_marks_run_failed(tracking_store, tmp_path):
    uri, client = tracking_store
    with pytest.raises(ValueError, match="Validation must start before"):
        run_tracked_training(
            PROJECT_DIR / "data/hour.csv",
            tmp_path / "outputs",
            pd.Timestamp("2012-10-01"),
            pd.Timestamp("2012-07-01"),
            tracking_uri=uri,
        )

    experiment = client.get_experiment_by_name("bike-demand")
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert runs[0].info.status == "FAILED"
    assert not client.list_artifacts(runs[0].info.run_id)
    assert mlflow.active_run() is None
