"""Check the HTTP contract and consistency with offline predictions."""

import hashlib
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from bike_demand.api import create_app
from bike_demand.data import load_hourly_data

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "artifacts/run_01/model.joblib"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    log_path = tmp_path_factory.mktemp("api_logs") / "requests.jsonl"
    with TestClient(create_app(MODEL_PATH, log_path=log_path)) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def reference_data():
    hourly = load_hourly_data(PROJECT_DIR / "data/hour.csv")
    predictions = pd.read_csv(
        PROJECT_DIR / "artifacts/run_01/test_predictions.csv",
        index_col="target_hour",
        parse_dates=True,
    )
    return hourly, predictions


def request_for_hour(hourly, target_hour):
    counts = hourly.loc[
        target_hour - pd.Timedelta(hours=168) : target_hour - pd.Timedelta(hours=1),
        "cnt",
    ]
    return {
        "target_hour": target_hour.isoformat(),
        "recent_counts": [int(value) if pd.notna(value) else None for value in counts],
    }


def test_health_reports_loaded_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_version": "sha256:"
        + hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
    }


@pytest.mark.parametrize("position", [0, 1000, -1])
def test_api_matches_saved_offline_predictions(client, reference_data, position):
    hourly, predictions = reference_data
    target_hour = predictions.index[position]
    response = client.post("/predict", json=request_for_hour(hourly, target_hour))

    assert response.status_code == 200
    result = response.json()
    assert pd.Timestamp(result["target_hour"]) == target_hour
    assert result["prediction"] == pytest.approx(
        predictions.loc[target_hour, "gradient_boosting"], abs=1e-9, rel=0
    )
    assert result["model_version"] == client.get("/health").json()["model_version"]


@pytest.mark.parametrize(
    "change",
    [
        {"recent_counts": [10] * 167},
        {"recent_counts": [10] * 169},
        {"recent_counts": [-1] + [10] * 167},
        {"recent_counts": [True] + [10] * 167},
        {"recent_counts": [10.5] + [10] * 167},
        {"recent_counts": [None] + [10] * 167},
        {"recent_counts": [10] * 167 + [None]},
        {"target_hour": "2012-10-01T00:30:00"},
        {"target_hour": "2012-10-01T00:00:00Z"},
        {"target_hour": "0001-01-01T00:00:00"},
        {"target_count": 100},
    ],
    ids=[
        "short-history",
        "long-history",
        "negative-count",
        "boolean-count",
        "fractional-count",
        "missing-weekly-count",
        "missing-latest-count",
        "partial-hour",
        "timezone",
        "unsupported-date",
        "unknown-field",
    ],
)
def test_invalid_inputs_are_rejected(client, change):
    payload = {"target_hour": "2012-10-01T00:00:00", "recent_counts": [10] * 168}
    payload.update(change)
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_model_remains_loaded_between_requests(tmp_path, reference_data):
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(MODEL_PATH.read_bytes())
    hourly, predictions = reference_data
    payload = request_for_hour(hourly, predictions.index[0])

    with TestClient(
        create_app(model_path, log_path=tmp_path / "requests.jsonl")
    ) as client:
        model_path.unlink()
        first = client.post("/predict", json=payload)
        second = client.post("/predict", json=payload)

    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()


@pytest.mark.parametrize("source", ["explicit", "environment"])
def test_missing_model_prevents_startup(tmp_path, monkeypatch, source):
    model_path = tmp_path / "missing.joblib"
    if source == "environment":
        monkeypatch.setenv("BIKE_DEMAND_MODEL_PATH", str(model_path))
        app = create_app()
    else:
        app = create_app(model_path)
    with (
        pytest.raises(FileNotFoundError, match="missing.joblib"),
        TestClient(app),
    ):
        pass


def test_incompatible_features_prevent_startup(tmp_path):
    bundle = joblib.load(MODEL_PATH)
    bundle["feature_columns"] = bundle["feature_columns"][:-1]
    model_path = tmp_path / "model.joblib"
    joblib.dump(bundle, model_path)

    with (
        pytest.raises(ValueError, match="Model features do not match"),
        TestClient(create_app(model_path)),
    ):
        pass
