"""Check the real HTTP lifecycle and isolation of the final rehearsal."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from bike_demand.demo import run_demo

PROJECT_DIR = Path(__file__).resolve().parents[1]


def test_demo_refuses_to_overwrite_an_existing_run(tmp_path):
    marker = tmp_path / "keep.txt"
    marker.write_text("existing results", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_demo(tmp_path)
    assert marker.read_text() == "existing results"
    assert not (tmp_path / "manifest.json").exists()


def test_real_http_demo_replays_monitors_and_restores_the_original_model(tmp_path):
    output = tmp_path / "demo"
    external_state = tmp_path / "other_state.json"
    external_log = tmp_path / "other_requests.jsonl"
    external_state.write_text("existing deployment", encoding="utf-8")
    external_log.write_text("existing logs", encoding="utf-8")
    protected = [
        PROJECT_DIR / "data/hour.csv",
        PROJECT_DIR / "artifacts/run_01/model.joblib",
    ]
    original_hashes = {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected
    }
    environment = os.environ.copy()
    environment.update(
        {
            "MLFLOW_DISABLE_TELEMETRY": "true",
            "BIKE_DEMAND_MODEL_PATH": str(tmp_path / "missing_override.joblib"),
            "BIKE_DEMAND_DEPLOYMENT_PATH": str(external_state),
            "BIKE_DEMAND_LOG_PATH": str(external_log),
        }
    )
    completed = subprocess.run(
        [sys.executable, "-m", "bike_demand.demo", "--output-dir", str(output)],
        check=False,
        cwd=PROJECT_DIR,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["completed"] is True
    assert manifest["comparison_decision"]["accepted"] is True
    candidate = manifest["candidate_health"]["model_version"]
    original = manifest["restored_health"]["model_version"]
    assert candidate != original
    assert manifest["before_restart_health"]["model_version"] == candidate
    assert manifest["rollback_prediction"]["model_version"] == original
    assert manifest["replay"]["requests"] == manifest["replay"]["predictions"] == 24
    assert manifest["replay"]["failed_requests"] == 0
    records = pd.read_csv(output / "replay/predictions.csv")
    assert records["model_version"].eq(candidate).all()
    errors = records["prediction"] - records["actual"]
    scores = manifest["monitoring"]["performance"]["gradient_boosting"]
    assert scores["mae"] == pytest.approx(errors.abs().mean())
    assert scores["rmse"] == pytest.approx((errors**2).mean() ** 0.5)
    state = json.loads((output / "deployment/state.json").read_text())
    assert state["active"]["model_version"] == original
    assert state["previous_models"] == []
    assert [event["action"] for event in state["history"]] == ["promote", "rollback"]
    events = [
        json.loads(line)
        for line in (output / "api_requests.jsonl").read_text().splitlines()
    ]
    assert [event["model_version"] for event in events] == [candidate] * 24 + [original]
    assert [event["request_id"] for event in events[:24]] == records[
        "request_id"
    ].tolist()
    assert events[-1]["request_id"] == manifest["rollback_request_id"]
    assert (output / "mlflow.db").is_file()
    assert (
        (output / "replay/monitoring/monitoring.png")
        .read_bytes()
        .startswith(b"\x89PNG")
    )
    assert candidate in (output / "summary.html").read_text(encoding="utf-8")
    assert external_state.read_text() == "existing deployment"
    assert external_log.read_text() == "existing logs"
    for path, digest in original_hashes.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
