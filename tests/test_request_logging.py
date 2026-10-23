"""Verify complete, isolated request records across responses and restarts."""

import json
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from bike_demand.api import create_app

MODEL_PATH = Path(__file__).resolve().parents[1] / "artifacts/run_01/model.joblib"


@pytest.fixture
def logged_client(tmp_path):
    log_path = tmp_path / "logs" / "requests.jsonl"
    with TestClient(
        create_app(MODEL_PATH, log_path=log_path), raise_server_exceptions=False
    ) as client:
        yield client, log_path


def payload():
    return {"target_hour": "2012-10-01T00:00:00", "recent_counts": [25] * 168}


def read_events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_successful_prediction_is_traceable(logged_client):
    client, log_path = logged_client
    client.get("/health")
    client.get("/docs")
    request = payload()
    request["recent_counts"][84] = None
    before = datetime.now(UTC)
    response = client.post("/predict", json=request)
    after = datetime.now(UTC)

    assert response.status_code == 200
    events = read_events(log_path)
    assert len(events) == 1
    event = events[0]
    assert UUID(event["request_id"]).hex == response.headers["X-Request-ID"]
    assert before <= datetime.fromisoformat(event["received_at"]) <= after
    assert event["target_hour"] == response.json()["target_hour"]
    assert event["model_version"] == response.json()["model_version"]
    assert event["prediction"] == response.json()["prediction"]
    assert event["status_code"] == 200
    assert event["history_hours"] == 168
    assert event["missing_count"] == 1
    assert math.isfinite(event["latency_ms"]) and event["latency_ms"] >= 0


@pytest.mark.parametrize("counts", [[25] * 167, [None] + [25] * 167])
def test_rejected_requests_keep_input_summaries(logged_client, counts):
    client, log_path = logged_client
    request = payload()
    request["recent_counts"] = counts
    response = client.post("/predict", json=request)

    assert response.status_code == 422
    events = read_events(log_path)
    assert len(events) == 1
    event = events[0]
    assert event["request_id"] == response.headers["X-Request-ID"]
    assert event["status_code"] == 422
    assert event["target_hour"] == request["target_hour"]
    assert event["history_hours"] == len(counts)
    assert event["missing_count"] == sum(value is None for value in counts)
    assert event["prediction"] is None


@pytest.mark.parametrize("content", ["{", "[]"])
def test_invalid_json_bodies_are_logged(logged_client, content):
    client, log_path = logged_client
    response = client.post(
        "/predict", content=content, headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 422
    events = read_events(log_path)
    assert len(events) == 1
    event = events[0]
    assert event["status_code"] == 422
    assert event["request_id"] == response.headers["X-Request-ID"]
    assert event["target_hour"] is None
    assert event["history_hours"] is None
    assert event["missing_count"] is None


def test_server_error_is_logged_once(logged_client, monkeypatch):
    client, log_path = logged_client

    def broken_predict(features):
        raise RuntimeError("Prediction failed")

    monkeypatch.setattr(client.app.state.estimator, "predict", broken_predict)
    response = client.post("/predict", json=payload())

    assert response.status_code == 500
    events = read_events(log_path)
    assert len(events) == 1
    event = events[0]
    assert event["status_code"] == 500
    assert event["request_id"] == response.headers["X-Request-ID"]
    assert event["target_hour"] == payload()["target_hour"]
    assert event["prediction"] is None


def test_logs_append_across_restarts(tmp_path):
    log_path = tmp_path / "requests.jsonl"
    for _ in range(2):
        with TestClient(create_app(MODEL_PATH, log_path=log_path)) as client:
            assert client.post("/predict", json=payload()).status_code == 200

    events = read_events(log_path)
    assert len(events) == 2
    assert len({event["request_id"] for event in events}) == 2


def test_concurrent_requests_keep_their_own_details(logged_client):
    client, log_path = logged_client
    requests = []
    for hour in range(4):
        request = payload()
        request["target_hour"] = f"2012-10-01T{hour:02}:00:00"
        request["recent_counts"][1 : 1 + hour] = [None] * hour
        requests.append(request)

    def send(request):
        return client.post("/predict", json=request)

    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(send, requests))

    events = read_events(log_path)
    assert len(events) == 4
    by_id = {event["request_id"]: event for event in events}
    assert len(by_id) == 4
    for hour, response in enumerate(responses):
        assert response.status_code == 200
        event = by_id[response.headers["X-Request-ID"]]
        assert event["target_hour"] == requests[hour]["target_hour"]
        assert event["missing_count"] == hour
        assert event["prediction"] == response.json()["prediction"]
