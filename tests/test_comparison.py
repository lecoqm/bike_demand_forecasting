"""Check added-data fitting, shared evaluation hours, and the acceptance rule."""

import hashlib
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient
from sklearn.ensemble import HistGradientBoostingRegressor

from bike_demand.compare import (
    ACCEPTANCE_RULE,
    candidate_decision,
    run_comparison,
    validate_schedule,
)
from bike_demand.train import run_training

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data/hour.csv"
MODEL_DIR = PROJECT_DIR / "artifacts/run_01"
START = pd.Timestamp("2012-11-01")
END = pd.Timestamp("2012-12-01")


@pytest.fixture
def tracking_store(tmp_path, monkeypatch):
    previous_uri = mlflow.get_tracking_uri()
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    yield uri, MlflowClient(tracking_uri=uri)
    mlflow.end_run()
    mlflow.set_tracking_uri(previous_uri)


@pytest.mark.parametrize(
    ("current", "candidate", "accepted"),
    [
        ({"mae": 10, "rmse": 20}, {"mae": 9, "rmse": 19}, True),
        ({"mae": 10, "rmse": 20}, {"mae": 9, "rmse": 21}, False),
        ({"mae": 10, "rmse": 20}, {"mae": 10, "rmse": 19}, False),
        ({"mae": 10, "rmse": 20}, {"mae": 11, "rmse": 18}, False),
        ({"mae": 0, "rmse": 0}, {"mae": 0, "rmse": 0}, False),
    ],
)
def test_acceptance_rule_is_fixed_and_requires_both_checks(
    current, candidate, accepted
):
    decision = candidate_decision(current, candidate)
    assert decision["accepted"] is accepted
    assert decision["recommendation"] == ("candidate" if accepted else "current")
    assert decision["rule"] == ACCEPTANCE_RULE
    if current["mae"] == 0:
        assert decision["mae_improvement_pct"] is None


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("2012-10-01", "2012-11-01"),
        ("2012-09-01", "2012-10-01"),
        ("2012-11-01T00:30:00", "2012-12-01"),
        ("2012-11-01T00:00:00Z", "2012-12-01"),
        ("2012-11-01", "2013-01-02"),
    ],
)
def test_training_overlap_and_invalid_comparison_dates_are_rejected(start, end):
    metadata = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
    with pytest.raises(ValueError):
        validate_schedule(
            metadata,
            pd.Timestamp(start),
            pd.Timestamp(end),
            pd.Timestamp("2012-12-31T23:00:00"),
        )


def test_training_rejects_an_end_before_its_test_start(tmp_path):
    output = tmp_path / "training"
    with pytest.raises(ValueError, match="Test end must be after"):
        run_training(DATA_PATH, output, pd.Timestamp("2012-10-01"), START, START)
    assert not output.exists()


def test_comparison_fits_only_earlier_targets_and_logs_shared_scores(
    tmp_path, tracking_store, monkeypatch
):
    uri, client = tracking_store
    output = tmp_path / "comparison"
    original_hashes = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [DATA_PATH, MODEL_DIR / "model.joblib", MODEL_DIR / "metadata.json"]
    }
    fit_indexes = []
    plans = []
    original_fit = HistGradientBoostingRegressor.fit

    def checked_fit(estimator, features, targets, *args, **kwargs):
        plan_path = output / "plan.json"
        assert plan_path.is_file()
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        assert plan["acceptance_rule"] == ACCEPTANCE_RULE
        plans.append(plan_path.read_bytes())
        fit_indexes.append(features.index.copy())
        assert features.index.max() < START
        return original_fit(estimator, features, targets, *args, **kwargs)

    monkeypatch.setattr(HistGradientBoostingRegressor, "fit", checked_fit)
    result = run_comparison(DATA_PATH, MODEL_DIR, output, START, END, uri)

    assert len(fit_indexes) == 2
    assert fit_indexes[0].max() < pd.Timestamp("2012-10-01")
    assert fit_indexes[1].max() < START
    assert (fit_indexes[1] >= pd.Timestamp("2012-10-01")).any()
    assert len(fit_indexes[1]) > len(fit_indexes[0])
    assert result["added_training_hours"] == len(fit_indexes[1]) - len(fit_indexes[0])
    assert plans[0] == plans[1] == (output / "plan.json").read_bytes()
    predictions = pd.read_csv(
        output / "comparison_predictions.csv", index_col="target_hour", parse_dates=True
    )
    assert len(predictions) == 30 * 24
    assert predictions.index.min() == START
    assert predictions.index.max() < END
    scored = predictions.loc[predictions["eligible"]]
    reference = pd.read_csv(
        MODEL_DIR / "test_predictions.csv", index_col="target_hour", parse_dates=True
    )
    reference = reference.loc[(reference.index >= START) & (reference.index < END)]
    pd.testing.assert_index_equal(scored.index, reference.index)
    np.testing.assert_allclose(
        scored["current"], reference["gradient_boosting"], rtol=0, atol=1e-9
    )
    np.testing.assert_array_equal(scored["actual"], reference["actual"])
    assert (
        predictions.loc[~predictions["eligible"], ["current", "candidate"]]
        .isna()
        .all()
        .all()
    )
    for name in ["current", "candidate", "weekly_baseline"]:
        errors = scored[name] - scored["actual"]
        assert result["metrics"][name]["mae"] == pytest.approx(errors.abs().mean())
        assert result["metrics"][name]["rmse"] == pytest.approx(
            np.sqrt((errors**2).mean())
        )
    assert result["coverage"]["paired_scored_hours"] == len(scored)
    assert result["coverage"]["scoring_coverage_pct"] == pytest.approx(
        100 * len(scored) / 720
    )

    candidate_dir = Path(result["candidate"]["model_path"]).parent
    metadata = json.loads((candidate_dir / "metadata.json").read_text(encoding="utf-8"))
    candidate_predictions = pd.read_csv(
        candidate_dir / "test_predictions.csv",
        index_col="target_hour",
        parse_dates=True,
    )
    assert metadata["test_end_exclusive"] == str(END)
    assert metadata["training_target_count"] == len(fit_indexes[1])
    assert pd.Timestamp(metadata["training_target_end"]) < START
    assert candidate_predictions.index.max() < END
    np.testing.assert_allclose(
        candidate_predictions["gradient_boosting"],
        scored["candidate"],
        rtol=0,
        atol=1e-9,
    )
    assert (
        result["candidate"]["model_version"]
        == "sha256:"
        + hashlib.sha256((candidate_dir / "model.joblib").read_bytes()).hexdigest()
    )
    assert result["decision"] == candidate_decision(
        result["metrics"]["current"], result["metrics"]["candidate"]
    )
    run = client.get_run(result["candidate"]["run_id"])
    assert run.info.status == "FINISHED"
    assert run.data.tags["model_role"] == "candidate"
    assert (
        run.data.tags["candidate_accepted"]
        == str(result["decision"]["accepted"]).lower()
    )
    for name in ["current", "candidate", "weekly_baseline"]:
        assert (
            run.data.metrics[f"comparison.{name}.mae"] == result["metrics"][name]["mae"]
        )
    assert run.data.params["test_end_exclusive"] == str(END)
    assert (
        run.data.params["current_model_version"] == result["current"]["model_version"]
    )
    assert {
        item.path for item in client.list_artifacts(run.info.run_id, "comparison")
    } == {
        f"comparison/{name}"
        for name in [
            "plan.json",
            "comparison.json",
            "comparison.html",
            "scores.csv",
            "comparison_predictions.csv",
        ]
    }
    assert (
        json.loads((output / "comparison.json").read_text(encoding="utf-8")) == result
    )
    html = (output / "comparison.html").read_text(encoding="utf-8")
    assert "RETROSPECTIVE MODEL UPDATE SIMULATION" in html
    assert result["candidate"]["run_id"] in html
    for path, digest in original_hashes.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert mlflow.active_run() is None
    with pytest.raises(FileExistsError):
        run_comparison(DATA_PATH, MODEL_DIR, output, START, END, uri)
