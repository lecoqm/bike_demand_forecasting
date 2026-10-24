"""Rehearse candidate promotion, December HTTP replay, monitoring, and rollback."""

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from time import monotonic, sleep

import httpx2 as httpx
import mlflow
import pandas as pd
from mlflow import MlflowClient

from bike_demand.api import ForecastResponse
from bike_demand.compare import run_comparison
from bike_demand.data import load_hourly_data
from bike_demand.deploy import promote, rollback
from bike_demand.monitor import run_monitoring
from bike_demand.replay import request_for_hour, run_replay
from bike_demand.track import git_tags

LOGGER = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data/hour.csv"
MODEL_DIR = PROJECT_DIR / "artifacts/run_01"


@contextmanager
def running_api(state_path: Path, output_dir: Path):
    """Start one owned subprocess on a free local port and always stop it."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    environment = os.environ.copy()
    environment.pop("BIKE_DEMAND_MODEL_PATH", None)
    environment["BIKE_DEMAND_DEPLOYMENT_PATH"] = str(state_path)
    environment["BIKE_DEMAND_LOG_PATH"] = str(output_dir / "api_requests.jsonl")
    with (output_dir / "server.log").open("ab") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "bike_demand.api:create_app",
                "--factory",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=PROJECT_DIR,
            env=environment,
            stdout=log,
            stderr=log,
        )
        try:
            with httpx.Client(
                base_url=f"http://127.0.0.1:{port}", timeout=30, trust_env=False
            ) as client:
                deadline = monotonic() + 30
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(
                            f"Demo API exited; see {output_dir / 'server.log'}"
                        )
                    try:
                        response = client.get("/health", timeout=1)
                        response.raise_for_status()
                        break
                    except httpx.HTTPError:
                        if monotonic() >= deadline:
                            raise TimeoutError(
                                "Demo API was not ready within 30 seconds."
                            ) from None
                        sleep(0.1)
                yield client
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


def verified_health(client: httpx.Client, expected: str) -> dict:
    """Check the loaded version rather than inferring it from deployment state."""
    response = client.get("/health")
    response.raise_for_status()
    health = response.json()
    if health.get("status") != "ok" or health.get("model_version") != expected:
        raise ValueError("The demo API is serving an unexpected model version.")
    return health


def write_manifest(output_dir: Path, result: dict) -> None:
    """Retain incomplete-run evidence as well as the final verification record."""
    (output_dir / "manifest.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def write_summary(output_dir: Path, result: dict) -> None:
    """Link the comparison and monitoring reports with the verified versions."""
    rows = "".join(
        f"<tr><td>{label}</td><td><code>{escape(result[key]['model_version'])}</code></td></tr>"
        for label, key in [
            ("Promoted API", "candidate_health"),
            ("After rollback, before restart", "before_restart_health"),
            ("Restored API after restart", "restored_health"),
        ]
    )
    quality = result["monitoring"]["performance"]["gradient_boosting"]
    coverage = result["monitoring"]["coverage"]
    html = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bike demand end-to-end demonstration</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;max-width:1050px;margin:36px auto;padding:0 20px;color:#172033;background:#f4f6f8}}
section{{background:white;border:1px solid #dce3eb;border-radius:12px;padding:24px;margin:24px 0}}
table{{width:100%;border-collapse:collapse}}td{{padding:10px;border-bottom:1px solid #dce3eb}}code{{font-size:12px;overflow-wrap:anywhere}}a{{color:#1857a4}}</style>
<p>RETROSPECTIVE ENGINEERING DEMONSTRATION</p><h1>Promotion, replay, monitoring, rollback</h1>
<p>Candidate fitted through October, selected on November, and replayed on December 1, 2012.</p>
<section><h2>Verified December replay</h2><p>{coverage["predictions"]} predictions / {coverage["scheduled_hours"]} scheduled hours.
MAE: {quality["mae"]:.2f}; RMSE: {quality["rmse"]:.2f} rentals per hour.</p>
<p><a href="comparison/comparison.html">November candidate comparison</a> · <a href="replay/monitoring/report.html">December monitoring report</a> · <a href="manifest.json">Verification record</a></p></section>
<section><h2>Loaded model checks</h2><table>{rows}</table>
<p>The isolated deployment ends on the original model after rollback and restart. Both demo API processes have stopped.</p></section>
<section><h2>Scope</h2><p>This is a one-day historical replay using an already explored dataset. Latency depends on the machine.
This demonstrates the local model lifecycle; it does not establish future forecasting performance.</p></section></html>"""
    (output_dir / "summary.html").write_text(html, encoding="utf-8")


def run_demo(output_dir: Path) -> dict:
    """Run the complete rehearsal using its own models, tracking DB, and state."""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "simulation": True,
        "completed": False,
        "started_at": datetime.now(UTC).isoformat(),
        "git": git_tags(),
        "start": "2012-12-01T00:00:00",
        "end_exclusive": "2012-12-02T00:00:00",
    }
    write_manifest(output_dir, result)
    previous_uri = mlflow.get_tracking_uri()
    try:
        tracking_uri = f"sqlite:///{(output_dir / 'mlflow.db').as_posix()}"
        MlflowClient(tracking_uri=tracking_uri).create_experiment(
            "bike-demand", artifact_location=(output_dir / "mlruns").as_uri()
        )
        comparison_dir = output_dir / "comparison"
        comparison = run_comparison(
            DATA_PATH,
            MODEL_DIR,
            comparison_dir,
            pd.Timestamp("2012-11-01"),
            pd.Timestamp("2012-12-01"),
            tracking_uri,
        )
        state_path = output_dir / "deployment/state.json"
        promote(comparison_dir, state_path, MODEL_DIR / "model.joblib")
        candidate = comparison["candidate"]
        original = comparison["current"]
        result["candidate_run_id"] = candidate["run_id"]
        result["comparison_decision"] = comparison["decision"]
        replay_dir = output_dir / "replay"
        with running_api(state_path, output_dir) as client:
            result["candidate_health"] = verified_health(
                client, candidate["model_version"]
            )
            result["replay"] = run_replay(
                DATA_PATH,
                Path(candidate["model_path"]).parent,
                replay_dir,
                pd.Timestamp(result["start"]),
                pd.Timestamp(result["end_exclusive"]),
                client,
            )
            if result["replay"]["failed_requests"]:
                raise ValueError("December replay contains unexpected failed requests.")
            result["monitoring"] = run_monitoring(
                replay_dir, DATA_PATH, replay_dir / "monitoring"
            )
            if result["monitoring"]["coverage"]["paired_scored_hours"] == 0:
                raise ValueError("December replay has no scored predictions.")
            rollback(state_path)
            result["before_restart_health"] = verified_health(
                client, candidate["model_version"]
            )
        with running_api(state_path, output_dir) as client:
            result["restored_health"] = verified_health(
                client, original["model_version"]
            )
            records = pd.read_csv(
                replay_dir / "predictions.csv", parse_dates=["target_hour"]
            )
            target = records.loc[records["prediction"].notna(), "target_hour"].iloc[0]
            payload = request_for_hour(load_hourly_data(DATA_PATH), target)
            response = client.post("/predict", json=payload)
            response.raise_for_status()
            forecast = ForecastResponse.model_validate(response.json())
            if forecast.model_version != original["model_version"]:
                raise ValueError("Rollback prediction used an unexpected model.")
            result["rollback_prediction"] = forecast.model_dump(mode="json")
            result["rollback_request_id"] = response.headers["X-Request-ID"]
        result.update(completed=True, finished_at=datetime.now(UTC).isoformat())
        write_summary(output_dir, result)
    except Exception as exc:
        result.update(
            completed=False, error=str(exc), finished_at=datetime.now(UTC).isoformat()
        )
        raise
    finally:
        mlflow.set_tracking_uri(previous_uri)
        write_manifest(output_dir, result)
    LOGGER.info("Completed December demonstration: %s", output_dir / "summary.html")
    return result


def main() -> None:
    """Train and compare, start real HTTP services, and stop them before exiting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/demos/december_day_01")
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    try:
        run_demo(args.output_dir)
    except (OSError, ValueError, KeyError, RuntimeError, httpx.HTTPError) as exc:
        parser.exit(1, f"Demonstration failed: {exc}\n")


if __name__ == "__main__":
    main()
