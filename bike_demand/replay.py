"""Replay historical hours through the API as a local serving simulation."""

import argparse
import csv
import hashlib
import json
import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import httpx2 as httpx
import numpy as np
import pandas as pd

from bike_demand.api import HISTORY_HOURS, ForecastResponse
from bike_demand.data import load_hourly_data

LOGGER = logging.getLogger(__name__)
RECORD_COLUMNS = [
    "target_hour",
    "request_id",
    "model_version",
    "status_code",
    "client_latency_ms",
    "prediction",
    "actual",
    "weekly_baseline",
    "latest_count",
    "missing_count",
    "error",
]


@dataclass(frozen=True)
class ReplayScenario:
    """Parameters for deterministic fault injection into copies of the data."""

    name: str = "control"
    seed: int = 42
    missing_rate: float = 0.1
    invalid_rate: float = 0.2
    demand_multiplier: float = 1.5
    shift_start: pd.Timestamp | None = None

    def __post_init__(self) -> None:
        if self.name not in {
            "control",
            "missing_history",
            "invalid_requests",
            "demand_shift",
        }:
            raise ValueError(f"Unknown replay scenario: {self.name}")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("Scenario seed must be a nonnegative integer.")
        for name, value in [
            ("missing_rate", self.missing_rate),
            ("invalid_rate", self.invalid_rate),
        ]:
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite fraction between 0 and 1.")
        if not math.isfinite(self.demand_multiplier) or self.demand_multiplier <= 0:
            raise ValueError("demand_multiplier must be finite and positive.")

    def shift_hour(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.Timestamp:
        """Place the default change halfway through the replay, on a whole hour."""
        hours = int((end - start) / pd.Timedelta(hours=1))
        value = (
            self.shift_start
            if self.shift_start is not None
            else start + pd.Timedelta(hours=hours // 2)
        )
        if (
            pd.isna(value)
            or value.tz is not None
            or value != value.floor("h")
            or not start <= value < end
        ):
            raise ValueError(
                "shift_start must be a timezone-naive whole hour inside the replay period."
            )
        return value


def prepare_scenario(
    hourly: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    scenario: ReplayScenario,
) -> tuple[pd.DataFrame, pd.Series, set[pd.Timestamp], dict]:
    """Separate available history from outcomes, preserving the original dataset."""
    history = hourly.copy(deep=True)
    outcomes = hourly["cnt"].copy(deep=True)
    invalid_hours = set()
    settings = {"name": scenario.name, "seed": scenario.seed}
    rng = np.random.default_rng(scenario.seed)
    if scenario.name == "missing_history":
        # One missing observation stays missing in every overlapping request.
        candidates = history.index[
            (history.index >= start - pd.Timedelta(hours=HISTORY_HOURS))
            & (history.index < end - pd.Timedelta(hours=1))
            & history["cnt"].notna()
        ]
        count = round(scenario.missing_rate * len(candidates))
        selected = candidates[
            np.sort(rng.choice(len(candidates), size=count, replace=False))
        ]
        history.loc[selected, "cnt"] = np.nan
        settings.update(
            missing_rate=scenario.missing_rate,
            candidate_observations=len(candidates),
            injected_observation_hours=[value.isoformat() for value in selected],
            outcomes="Original recorded counts; only the request history is masked.",
        )
    elif scenario.name == "invalid_requests":
        hours = pd.date_range(start, end, freq="h", inclusive="left")
        count = round(scenario.invalid_rate * len(hours))
        selected = hours[np.sort(rng.choice(len(hours), size=count, replace=False))]
        invalid_hours = set(selected)
        settings.update(
            invalid_rate=scenario.invalid_rate,
            invalid_target_hours=[value.isoformat() for value in selected],
            corruption="Replace the most recent count with -1.",
        )
    elif scenario.name == "demand_shift":
        shift_start = scenario.shift_hour(start, end)
        affected = (history.index >= shift_start) & (history.index < end)
        values = history.loc[affected, "cnt"]
        if (
            values.notna().any()
            and float(values.max()) * scenario.demand_multiplier > 2**53 - 1
        ):
            raise ValueError("Demand multiplier would exceed the API count limit.")
        shifted = np.rint(values * scenario.demand_multiplier)
        history.loc[affected, "cnt"] = shifted
        outcomes.loc[affected] = shifted
        settings.update(
            demand_multiplier=scenario.demand_multiplier,
            shift_start=shift_start.isoformat(),
            rounding="Nearest integer, ties to even (numpy.rint).",
            outcomes="Synthetic counts after shift_start; missing outcomes stay missing.",
        )
    return history, outcomes, invalid_hours, settings


def request_for_hour(hourly: pd.DataFrame, target_hour: pd.Timestamp) -> dict:
    """Send only earlier counts, keeping every clock hour in its original slot."""
    history_index = pd.date_range(
        end=target_hour - pd.Timedelta(hours=1), periods=HISTORY_HOURS, freq="h"
    )
    counts = hourly["cnt"].reindex(history_index)
    return {
        "target_hour": target_hour.isoformat(),
        "recent_counts": [int(value) if pd.notna(value) else None for value in counts],
    }


def validate_period(
    hourly: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    training_target_end: pd.Timestamp,
) -> None:
    """Keep the replay within the dataset and strictly after model training."""
    for value in [start, end, training_target_end]:
        if pd.isna(value) or value.tz is not None or value != value.floor("h"):
            raise ValueError(
                "Replay and training dates must be timezone-naive whole hours."
            )
    if start >= end:
        raise ValueError("Replay start must be before its exclusive end.")
    if start <= training_target_end:
        raise ValueError("Replay must start after the model's last training target.")
    if start < hourly.index.min() + pd.Timedelta(hours=HISTORY_HOURS):
        raise ValueError("The dataset must cover all 168 history hours before replay.")
    if end > hourly.index.max() + pd.Timedelta(hours=1):
        raise ValueError("Replay end is outside the dataset.")


def predict_hour(client: httpx.Client, payload: dict, model_version: str) -> dict:
    """Retain HTTP errors and transport failures without retrying a target hour."""
    event = {
        "target_hour": payload["target_hour"],
        "request_id": None,
        "model_version": model_version,
        "status_code": None,
        "prediction": None,
        "weekly_baseline": payload["recent_counts"][0],
        "latest_count": payload["recent_counts"][-1],
        "missing_count": sum(value is None for value in payload["recent_counts"]),
        "error": None,
    }
    started = perf_counter()
    try:
        response = client.post("/predict", json=payload)
        event["status_code"] = response.status_code
        event["request_id"] = response.headers.get("X-Request-ID")
        if response.status_code != 200:
            event["error"] = response.text
        else:
            try:
                result = ForecastResponse.model_validate(response.json())
                event["model_version"] = result.model_version
                if result.model_version != model_version:
                    raise ValueError("The API model changed during replay.")
                if result.target_hour.isoformat() != payload["target_hour"]:
                    raise ValueError("The API returned a different target hour.")
                event["prediction"] = result.prediction
            except ValueError as exc:
                event["error"] = f"Invalid API response: {exc}"
    except httpx.RequestError as exc:
        event["error"] = f"{type(exc).__name__}: {exc}"
    event["client_latency_ms"] = round((perf_counter() - started) * 1000, 3)
    return event


def run_replay(
    data_path: Path,
    model_dir: Path,
    output_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    client: httpx.Client,
    scenario: ReplayScenario | None = None,
) -> dict:
    """Write one row per clock hour, reading its outcome after the API call."""
    hourly = load_hourly_data(data_path)
    model_metadata = json.loads(
        (model_dir / "metadata.json").read_text(encoding="utf-8")
    )
    training_end = pd.Timestamp(model_metadata["training_target_end"])
    validate_period(hourly, start, end, training_end)
    history, outcomes, invalid_hours, scenario_settings = prepare_scenario(
        hourly, start, end, scenario or ReplayScenario()
    )
    model_version = (
        "sha256:"
        + hashlib.sha256((model_dir / "model.joblib").read_bytes()).hexdigest()
    )
    health = client.get("/health")
    health.raise_for_status()
    if health.json().get("model_version") != model_version:
        raise ValueError(
            "API model differs from --model-dir; restart it with that model."
        )

    hours = pd.date_range(start, end, freq="h", inclusive="left")
    metadata = {
        "simulation": True,
        "scenario": scenario_settings,
        "completed": False,
        "started_at": datetime.now(UTC).isoformat(),
        "start": start.isoformat(),
        "end_exclusive": end.isoformat(),
        "planned_hours": len(hours),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "model_version": model_version,
        "training_target_end": training_end.isoformat(),
        "forecast_horizon_hours": 1,
        "api_url": str(client.base_url),
    }
    # Refuse to overwrite earlier runs; flush each row so interruptions retain work.
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    summary = {
        "requests": 0,
        "predictions": 0,
        "rejected_requests": 0,
        "failed_requests": 0,
        "missing_outcomes": 0,
        "scored_predictions": 0,
    }
    with (output_dir / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=RECORD_COLUMNS)
        writer.writeheader()
        for target_hour in hours:
            payload = request_for_hour(history, target_hour)
            if target_hour in invalid_hours:
                payload["recent_counts"][-1] = -1
            event = predict_hour(client, payload, model_version)
            # Reveal this outcome only after its prediction attempt has finished.
            actual = outcomes.at[target_hour]
            event["actual"] = int(actual) if pd.notna(actual) else None
            writer.writerow(event)
            stream.flush()
            predicted = event["prediction"] is not None
            rejected = (
                event["status_code"] is not None and 400 <= event["status_code"] < 500
            )
            summary["requests"] += 1
            summary["predictions"] += int(predicted)
            summary["rejected_requests"] += int(rejected)
            summary["failed_requests"] += int(not predicted and not rejected)
            summary["missing_outcomes"] += int(event["actual"] is None)
            summary["scored_predictions"] += int(
                predicted and event["actual"] is not None
            )
            if summary["requests"] % 24 == 0 or target_hour == hours[-1]:
                LOGGER.info("Replayed %s/%s hours", summary["requests"], len(hours))

    metadata.update(
        completed=True, finished_at=datetime.now(UTC).isoformat(), summary=summary
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Replay summary: %s", summary)
    LOGGER.info("Saved replay to %s", output_dir)
    return summary


def main() -> None:
    """Run a one-day smoke replay by default; end dates are exclusive."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path("data/hour.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/run_01"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/replays/october_day_01")
    )
    parser.add_argument("--start", type=pd.Timestamp, default="2012-10-01")
    parser.add_argument(
        "--end", type=pd.Timestamp, default="2012-10-02", help="Exclusive end."
    )
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--timeout", type=float, default=30.0, help="HTTP timeout in seconds."
    )
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be a finite positive number.")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    try:
        with httpx.Client(
            base_url=args.api_url, timeout=args.timeout, trust_env=False
        ) as client:
            run_replay(
                args.data_path,
                args.model_dir,
                args.output_dir,
                args.start,
                args.end,
                client,
            )
    except (OSError, ValueError, httpx.HTTPError) as exc:
        parser.exit(1, f"Replay failed: {exc}\n")


if __name__ == "__main__":
    main()
