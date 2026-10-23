"""Check causal HTTP replay, training cutoffs, and retained failure records."""

import hashlib
import json
from pathlib import Path

import httpx2 as httpx
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from bike_demand import replay
from bike_demand.api import create_app
from bike_demand.data import load_hourly_data

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data/hour.csv"
MODEL_DIR = PROJECT_DIR / "artifacts/run_01"
START = pd.Timestamp("2012-10-01")
MODEL_VERSION = (
    "sha256:" + hashlib.sha256((MODEL_DIR / "model.joblib").read_bytes()).hexdigest()
)


def read_records(output_dir):
    return pd.read_csv(output_dir / "predictions.csv", parse_dates=["target_hour"])


def test_request_excludes_current_and_future_counts_and_preserves_gaps():
    index = pd.date_range("2012-09-23", periods=240, freq="h")
    hourly = pd.DataFrame({"cnt": np.arange(len(index), dtype=float)}, index=index)
    target_hour = index[200]
    hourly.loc[index[40], "cnt"] = np.nan
    payload = replay.request_for_hour(hourly, target_hour)
    changed = hourly.copy()
    changed.loc[target_hour:, "cnt"] = 999999

    assert replay.request_for_hour(changed, target_hour) == payload
    assert len(payload["recent_counts"]) == 168
    assert payload["recent_counts"][0] == 32
    assert payload["recent_counts"][-1] == 199
    assert payload["recent_counts"][8] is None
    assert set(payload) == {"target_hour", "recent_counts"}


def test_replay_matches_saved_predictions_and_links_to_server_logs(tmp_path):
    output_dir = tmp_path / "replay"
    log_path = tmp_path / "requests.jsonl"
    end = START + pd.Timedelta(days=1)
    with TestClient(create_app(MODEL_DIR / "model.joblib", log_path)) as client:
        summary = replay.run_replay(
            DATA_PATH, MODEL_DIR, output_dir, START, end, client
        )

    records = read_records(output_dir)
    scored = records.dropna(subset=["prediction", "actual"]).set_index("target_hour")
    reference = pd.read_csv(
        MODEL_DIR / "test_predictions.csv", index_col="target_hour", parse_dates=True
    )
    reference = reference.loc[(reference.index >= START) & (reference.index < end)]
    pd.testing.assert_index_equal(scored.index, reference.index)
    np.testing.assert_allclose(
        scored["prediction"], reference["gradient_boosting"], rtol=0, atol=1e-9
    )
    np.testing.assert_array_equal(scored["actual"], reference["actual"])
    np.testing.assert_array_equal(
        scored["weekly_baseline"], reference["weekly_baseline"]
    )
    events = [
        json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records["request_id"].tolist() == [event["request_id"] for event in events]
    assert summary["requests"] == len(records) == 24
    assert summary["scored_predictions"] == len(reference)
    assert summary["failed_requests"] == 0
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["simulation"] is metadata["completed"] is True
    assert metadata["model_version"] == MODEL_VERSION
    assert metadata["summary"] == summary
    assert (records["client_latency_ms"] >= 0).all()


def test_replay_keeps_rejections_and_predictions_without_outcomes(
    tmp_path, monkeypatch
):
    hourly = load_hourly_data(DATA_PATH)
    hourly.loc[
        START - pd.Timedelta(hours=168) : START + pd.Timedelta(hours=3), "cnt"
    ] = 50
    hourly.loc[START - pd.Timedelta(hours=168), "cnt"] = np.nan
    hourly.loc[START + pd.Timedelta(hours=1), "cnt"] = np.nan
    monkeypatch.setattr(replay, "load_hourly_data", lambda path: hourly)
    output_dir = tmp_path / "replay"
    with TestClient(
        create_app(MODEL_DIR / "model.joblib", tmp_path / "requests.jsonl")
    ) as client:
        summary = replay.run_replay(
            DATA_PATH,
            MODEL_DIR,
            output_dir,
            START,
            START + pd.Timedelta(hours=4),
            client,
        )

    records = read_records(output_dir)
    assert records["status_code"].tolist() == [422, 200, 422, 200]
    assert pd.isna(records.loc[1, "actual"])
    assert pd.notna(records.loc[1, "prediction"])
    assert pd.isna(records.loc[0, "weekly_baseline"])
    assert records["error"].notna().tolist() == [True, False, True, False]
    assert summary == {
        "requests": 4,
        "predictions": 2,
        "rejected_requests": 2,
        "failed_requests": 0,
        "missing_outcomes": 1,
        "scored_predictions": 1,
    }


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        ("2012-09-30T23:00:00", "2012-10-02", "after the model"),
        ("2012-10-01", "2012-10-01", "before its exclusive end"),
        ("2012-10-01T00:30:00", "2012-10-02", "whole hours"),
        ("2012-10-01T00:00:00Z", "2012-10-02", "timezone-naive"),
        ("2012-12-31", "2013-01-02", "outside the dataset"),
    ],
)
def test_invalid_replay_periods_are_rejected(start, end, message):
    with pytest.raises(ValueError, match=message):
        replay.validate_period(
            load_hourly_data(DATA_PATH),
            pd.Timestamp(start),
            pd.Timestamp(end),
            START - pd.Timedelta(hours=1),
        )


def test_replay_requires_the_full_history_window():
    hourly = load_hourly_data(DATA_PATH)
    start = hourly.index.min() + pd.Timedelta(hours=167)
    with pytest.raises(ValueError, match="all 168 history hours"):
        replay.validate_period(
            hourly, start, start + pd.Timedelta(hours=1), hourly.index.min()
        )


def test_model_mismatch_stops_before_creating_outputs(tmp_path):
    def handler(request):
        assert request.url.path == "/health"
        return httpx.Response(200, json={"model_version": "sha256:different"})

    output_dir = tmp_path / "replay"
    with (
        httpx.Client(
            base_url="http://test", transport=httpx.MockTransport(handler)
        ) as client,
        pytest.raises(ValueError, match="API model differs"),
    ):
        replay.run_replay(
            DATA_PATH,
            MODEL_DIR,
            output_dir,
            START,
            START + pd.Timedelta(hours=1),
            client,
        )
    assert not output_dir.exists()


def test_replay_records_transport_errors_server_errors_and_changed_models(tmp_path):
    attempts = []

    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"model_version": MODEL_VERSION})
        payload = json.loads(request.content)
        attempts.append(payload)
        if len(attempts) == 1:
            raise httpx.ReadTimeout("simulated timeout", request=request)
        if len(attempts) == 2:
            return httpx.Response(500, text="Internal Server Error")
        return httpx.Response(
            200,
            json={
                "target_hour": payload["target_hour"],
                "prediction": 42,
                "model_version": MODEL_VERSION
                if len(attempts) == 3
                else "sha256:changed",
            },
            headers={"X-Request-ID": f"request-{len(attempts)}"},
        )

    output_dir = tmp_path / "replay"
    with httpx.Client(
        base_url="http://test", transport=httpx.MockTransport(handler)
    ) as client:
        summary = replay.run_replay(
            DATA_PATH,
            MODEL_DIR,
            output_dir,
            START,
            START + pd.Timedelta(hours=4),
            client,
        )

    records = read_records(output_dir)
    assert len(attempts) == 4
    assert "ReadTimeout" in records.loc[0, "error"]
    assert records.loc[1, "status_code"] == 500
    assert records.loc[2, "prediction"] == 42
    assert "model changed" in records.loc[3, "error"]
    assert pd.isna(records.loc[3, "prediction"])
    assert summary["failed_requests"] == 3
    assert summary["predictions"] == summary["scored_predictions"] == 1


def test_existing_outputs_are_preserved(tmp_path):
    sentinel = tmp_path / "predictions.csv"
    sentinel.write_text("previous replay\n", encoding="utf-8")
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"model_version": MODEL_VERSION})
    )
    with (
        httpx.Client(base_url="http://test", transport=transport) as client,
        pytest.raises(FileExistsError),
    ):
        replay.run_replay(
            DATA_PATH,
            MODEL_DIR,
            tmp_path,
            START,
            START + pd.Timedelta(hours=1),
            client,
        )
    assert sentinel.read_text(encoding="utf-8") == "previous replay\n"
