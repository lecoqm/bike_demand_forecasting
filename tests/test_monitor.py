"""Check report denominators, paired model scores, and standalone outputs."""

import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bike_demand import monitor
from bike_demand.data import load_hourly_data

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data/hour.csv"
MODEL_VERSION = "sha256:example"


@pytest.fixture
def replay_files(tmp_path):
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    records = pd.DataFrame(
        {
            "target_hour": pd.date_range("2012-10-01", periods=6, freq="h"),
            "request_id": [f"request-{i}" for i in range(6)],
            "model_version": [MODEL_VERSION] * 6,
            "status_code": [200, 200, 422, None, 200, 200],
            "client_latency_ms": [10, 20, 30, 40, 50, 60],
            "prediction": [12, 16, None, None, 25, 8],
            "actual": [10, 20, 30, 40, None, 8],
            "weekly_baseline": [14, 12, 25, 35, 24, None],
            "missing_count": [0, 2, 1, 0, 0, 0],
            "error": [None, None, "validation error", "ReadTimeout", None, None],
        }
    )
    metadata = {
        "simulation": True,
        "completed": True,
        "start": "2012-10-01T00:00:00",
        "end_exclusive": "2012-10-01T06:00:00",
        "planned_hours": 6,
        "model_version": MODEL_VERSION,
        "training_target_end": "2012-09-30T23:00:00",
    }
    write_replay(replay_dir, records, metadata)
    return replay_dir, records, metadata


def write_replay(replay_dir, records, metadata):
    records.to_csv(replay_dir / "predictions.csv", index=False)
    (replay_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_scores_use_identical_hours_and_keep_failures_in_coverage(replay_files):
    replay_dir, _, _ = replay_files
    records, metadata = monitor.load_replay(replay_dir)
    before = records.copy(deep=True)
    summary, metrics, hourly = monitor.summarize(records, metadata)

    assert summary["coverage"] == {
        "scheduled_hours": 6,
        "attempted_hours": 6,
        "predictions": 4,
        "prediction_coverage_pct": pytest.approx(400 / 6),
        "known_outcomes": 5,
        "scored_predictions": 3,
        "paired_scored_hours": 2,
    }
    scores = metrics.set_index("model")
    assert scores.loc["gradient_boosting", "mae"] == 3
    assert scores.loc["gradient_boosting", "rmse"] == pytest.approx(np.sqrt(10))
    assert scores.loc["weekly_baseline", "mae"] == 6
    assert scores.loc["weekly_baseline", "rmse"] == pytest.approx(np.sqrt(40))
    assert scores["scored_hours"].tolist() == [2, 2]
    assert summary["performance"]["mae_improvement_pct"] == 50
    assert summary["service"]["rejected_requests"] == 1
    assert summary["service"]["failed_requests"] == 1
    assert summary["service"]["error_rate_pct"] == pytest.approx(200 / 6)
    assert summary["service"]["client_latency_ms"]["median"] == 35
    assert summary["service"]["client_latency_ms"]["p95"] == pytest.approx(57.5)
    assert summary["data_quality"]["requests_with_missing_history"] == 2
    assert summary["data_quality"]["missing_outcomes"] == 1
    assert hourly["scored_hours"].sum() == 2
    assert hourly.loc[0, "gradient_boosting_mae"] == 2
    assert pd.isna(hourly.loc[2, "gradient_boosting_mae"])
    pd.testing.assert_frame_equal(records, before)


def test_partial_replay_uses_all_scheduled_hours(replay_files):
    replay_dir, records, metadata = replay_files
    metadata.update(
        completed=False, planned_hours=8, end_exclusive="2012-10-01T08:00:00"
    )
    write_replay(replay_dir, records, metadata)
    loaded, metadata = monitor.load_replay(replay_dir)
    summary, _, _ = monitor.summarize(loaded, metadata)
    assert summary["replay_completed"] is False
    assert summary["coverage"]["prediction_coverage_pct"] == 50


@pytest.mark.parametrize("failure", ["wrong-model", "invalid-response", "server-error"])
def test_invalid_success_rows_cannot_improve_scores(replay_files, failure):
    replay_dir, records, metadata = replay_files
    if failure == "wrong-model":
        records.loc[0, "model_version"] = "sha256:different"
    elif failure == "invalid-response":
        records.loc[0, "error"] = "Invalid API response"
    else:
        records.loc[0, "status_code"] = 500
    write_replay(replay_dir, records, metadata)
    loaded, metadata = monitor.load_replay(replay_dir)
    summary, _, _ = monitor.summarize(loaded, metadata)
    assert summary["coverage"]["paired_scored_hours"] == 1
    assert summary["performance"]["gradient_boosting"]["mae"] == 4
    assert summary["service"]["failed_requests"] == 2


def test_zero_baseline_error_has_no_percentage_improvement(replay_files):
    replay_dir, records, metadata = replay_files
    records.loc[[0, 1], "weekly_baseline"] = records.loc[[0, 1], "actual"]
    write_replay(replay_dir, records, metadata)
    loaded, metadata = monitor.load_replay(replay_dir)
    summary, _, _ = monitor.summarize(loaded, metadata)
    assert summary["performance"]["weekly_baseline"]["mae"] == 0
    assert summary["performance"]["mae_improvement_pct"] is None


def test_training_reference_uses_only_eligible_earlier_targets():
    cutoff = pd.Timestamp("2012-09-30T23:00:00")
    reference = monitor.training_reference(DATA_PATH, str(cutoff))
    hourly = load_hourly_data(DATA_PATH)
    assert reference.index.max() <= cutoff
    np.testing.assert_array_equal(
        reference, hourly["cnt"].shift(168).loc[reference.index]
    )
    assert hourly.loc[reference.index, "cnt"].notna().all()
    assert hourly["cnt"].shift(1).loc[reference.index].notna().all()


def test_report_is_self_contained_and_preserves_source_files(replay_files, tmp_path):
    replay_dir, _, _ = replay_files
    before = {path.name: path.read_bytes() for path in replay_dir.iterdir()}
    output_dir = tmp_path / "monitoring"
    summary = monitor.run_monitoring(replay_dir, DATA_PATH, output_dir)
    assert {path.name for path in output_dir.iterdir()} == {
        "summary.json",
        "metrics.csv",
        "by_hour.csv",
        "input_distribution.csv",
        "monitoring.png",
        "report.html",
    }
    html = (output_dir / "report.html").read_text(encoding="utf-8")
    assert "HISTORICAL REPLAY SIMULATION" in html
    assert "same 2 hours" in html
    image = html.split('src="data:image/png;base64,', 1)[1].split('"', 1)[0]
    assert base64.b64decode(image) == (output_dir / "monitoring.png").read_bytes()
    assert (output_dir / "monitoring.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (
        json.loads((output_dir / "summary.json").read_text(encoding="utf-8")) == summary
    )
    for name, contents in before.items():
        assert (replay_dir / name).read_bytes() == contents
    assert (
        summary["source_sha256"]["predictions"]
        == hashlib.sha256(before["predictions.csv"]).hexdigest()
    )


@pytest.mark.parametrize("empty", [False, True])
def test_no_scorable_results_produce_null_metrics(replay_files, tmp_path, empty):
    replay_dir, records, metadata = replay_files
    records["prediction"] = np.nan
    records["weekly_baseline"] = np.nan
    records["client_latency_ms"] = np.nan
    records["missing_count"] = np.nan
    if empty:
        records = records.iloc[:0]
        metadata["completed"] = False
    write_replay(replay_dir, records, metadata)
    summary = monitor.run_monitoring(replay_dir, DATA_PATH, tmp_path / "report")
    assert summary["performance"]["gradient_boosting"]["mae"] is None
    assert summary["performance"]["weekly_baseline"]["rmse"] is None
    assert summary["performance"]["mae_improvement_pct"] is None
    assert summary["service"]["client_latency_ms"]["median"] is None
    assert summary["input_distribution"][1]["mean"] is None
    assert summary["coverage"]["prediction_coverage_pct"] == 0
    json.dumps(summary, allow_nan=False)
    html = (tmp_path / "report/report.html").read_text(encoding="utf-8")
    assert "n/a" in html
    assert "<td>None</td>" not in html
    if empty:
        assert "Partial replay" in html
        assert summary["service"]["error_rate_pct"] is None


@pytest.mark.parametrize(
    "invalid",
    [
        "missing-column",
        "duplicate-hour",
        "infinite-latency",
        "inconsistent-period",
        "missing-completed-hour",
    ],
)
def test_malformed_inputs_are_rejected(replay_files, invalid):
    replay_dir, records, metadata = replay_files
    if invalid == "missing-column":
        records = records.drop(columns="actual")
    elif invalid == "duplicate-hour":
        records.loc[1, "target_hour"] = records.loc[0, "target_hour"]
    elif invalid == "infinite-latency":
        records["client_latency_ms"] = np.inf
    elif invalid == "inconsistent-period":
        metadata["planned_hours"] = 100
    else:
        records = records.iloc[:-1]
    write_replay(replay_dir, records, metadata)
    with pytest.raises(ValueError):
        monitor.load_replay(replay_dir)
