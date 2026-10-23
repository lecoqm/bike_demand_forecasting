"""Check reproducible faults, causal demand changes, and their API reports."""

import hashlib
import json
from pathlib import Path

import httpx2 as httpx
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from bike_demand.api import create_app
from bike_demand.data import load_hourly_data
from bike_demand.monitor import load_replay, run_monitoring
from bike_demand.replay import ReplayScenario, prepare_scenario, request_for_hour
from bike_demand.scenarios import SCENARIO_NAMES, run_scenarios

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data/hour.csv"
MODEL_DIR = PROJECT_DIR / "artifacts/run_01"
START = pd.Timestamp("2012-10-01")
END = pd.Timestamp("2012-10-02")


@pytest.fixture(scope="module")
def hourly():
    return load_hourly_data(DATA_PATH)


def test_control_leaves_history_and_outcomes_unchanged(hourly):
    history, outcomes, invalid, settings = prepare_scenario(
        hourly, START, END, ReplayScenario()
    )
    pd.testing.assert_frame_equal(history, hourly)
    pd.testing.assert_series_equal(outcomes, hourly["cnt"])
    assert invalid == set()
    assert settings == {"name": "control", "seed": 42}


def test_missing_history_is_reproducible_and_preserves_outcomes(hourly):
    original = hourly.copy(deep=True)
    scenario = ReplayScenario(name="missing_history", seed=42, missing_rate=0.1)
    history, outcomes, _, settings = prepare_scenario(hourly, START, END, scenario)
    repeated, _, _, repeated_settings = prepare_scenario(hourly, START, END, scenario)
    other, _, _, other_settings = prepare_scenario(
        hourly, START, END, ReplayScenario(name="missing_history", seed=43)
    )

    pd.testing.assert_frame_equal(hourly, original)
    pd.testing.assert_frame_equal(history, repeated)
    pd.testing.assert_series_equal(outcomes, hourly["cnt"])
    assert settings == repeated_settings
    assert (
        settings["injected_observation_hours"]
        != other_settings["injected_observation_hours"]
    )
    assert not history.equals(other)
    selected = pd.DatetimeIndex(settings["injected_observation_hours"])
    assert len(selected) == round(0.1 * settings["candidate_observations"])
    assert hourly.loc[selected, "cnt"].notna().all()
    assert history.loc[selected, "cnt"].isna().all()
    # Check one masked observation in two overlapping request windows.
    eligible = selected[
        (selected > START - pd.Timedelta(hours=168))
        & (selected < END - pd.Timedelta(hours=2))
    ]
    observation = eligible[0]
    first_target = max(START, observation + pd.Timedelta(hours=1))
    for target in [first_target, first_target + pd.Timedelta(hours=1)]:
        payload = request_for_hour(history, target)
        position = int(
            (observation - (target - pd.Timedelta(hours=168))) / pd.Timedelta(hours=1)
        )
        assert payload["recent_counts"][position] is None


def test_demand_shift_reaches_history_only_after_the_outcome_hour(hourly):
    original = hourly.copy(deep=True)
    shift = START + pd.Timedelta(hours=12)
    history, outcomes, _, settings = prepare_scenario(
        hourly, START, END, ReplayScenario(name="demand_shift", demand_multiplier=1.5)
    )
    assert settings["shift_start"] == shift.isoformat()
    assert request_for_hour(history, shift) == request_for_hour(hourly, shift)
    expected = int(np.rint(hourly.at[shift, "cnt"] * 1.5))
    assert outcomes.at[shift] == expected
    assert (
        request_for_hour(history, shift + pd.Timedelta(hours=1))["recent_counts"][-1]
        == expected
    )
    pd.testing.assert_series_equal(
        outcomes.loc[: shift - pd.Timedelta(hours=1)],
        hourly.loc[: shift - pd.Timedelta(hours=1), "cnt"],
    )
    pd.testing.assert_frame_equal(hourly, original)


@pytest.mark.parametrize(
    "options",
    [
        {"missing_rate": 1.1},
        {"invalid_rate": float("nan")},
        {"demand_multiplier": 0},
        {"seed": -1},
        {"name": "unknown"},
    ],
)
def test_invalid_scenario_settings_are_rejected(options):
    with pytest.raises(ValueError):
        ReplayScenario(**options)


@pytest.mark.parametrize(
    "shift", ["2012-10-01T00:30:00", "2012-10-02T00:00:00", "2012-10-01T12:00:00Z"]
)
def test_demand_shift_must_be_a_whole_hour_inside_replay(hourly, shift):
    with pytest.raises(ValueError, match="inside the replay"):
        prepare_scenario(
            hourly,
            START,
            END,
            ReplayScenario(name="demand_shift", shift_start=pd.Timestamp(shift)),
        )


def test_unavailable_api_does_not_create_a_suite_directory(tmp_path):
    def unavailable(request):
        raise httpx.ConnectError("simulated offline API", request=request)

    output = tmp_path / "scenarios"
    with (
        httpx.Client(
            base_url="http://test", transport=httpx.MockTransport(unavailable)
        ) as client,
        pytest.raises(httpx.ConnectError),
    ):
        run_scenarios(DATA_PATH, MODEL_DIR, output, START, END, client)
    assert not output.exists()


def test_four_scenarios_produce_comparable_reports_without_modifying_sources(tmp_path):
    source_hashes = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [DATA_PATH, MODEL_DIR / "model.joblib"]
    }
    output = tmp_path / "scenarios"
    options = ReplayScenario(seed=42, invalid_rate=0.25)
    with TestClient(
        create_app(MODEL_DIR / "model.joblib", tmp_path / "requests.jsonl")
    ) as client:
        comparison = run_scenarios(
            DATA_PATH, MODEL_DIR, output, START, END, client, options
        )
        with pytest.raises(FileExistsError):
            run_scenarios(DATA_PATH, MODEL_DIR, output, START, END, client, options)

    result = comparison.set_index("scenario")
    assert result.index.tolist() == SCENARIO_NAMES
    assert result["requests"].tolist() == [24] * 4
    assert result.loc["control", "predictions"] == 24
    assert result.loc["invalid_requests", "rejected"] == 6
    assert result.loc["invalid_requests", "predictions"] == 18
    assert result.loc["missing_history", "coverage_pct"] < 100
    assert (
        result.loc["missing_history", "mean_missing_history_hours"]
        > result.loc["control", "mean_missing_history_hours"]
    )
    assert (
        result.loc["demand_shift", "input_mean"] > result.loc["control", "input_mean"]
    )
    assert result["missing_outcomes"].tolist() == [0] * 4
    control = pd.read_csv(output / "control/predictions.csv")
    missing = pd.read_csv(output / "missing_history/predictions.csv")
    shifted = pd.read_csv(output / "demand_shift/predictions.csv")
    assert missing["actual"].tolist() == control["actual"].tolist()
    np.testing.assert_array_equal(
        shifted.loc[:11, "actual"], control.loc[:11, "actual"]
    )
    np.testing.assert_array_equal(
        shifted.loc[:12, "prediction"], control.loc[:12, "prediction"]
    )
    np.testing.assert_array_equal(
        shifted.loc[12:, "actual"], np.rint(control.loc[12:, "actual"] * 1.5)
    )

    versions = set()
    for name in SCENARIO_NAMES:
        records, metadata = load_replay(output / name)
        versions.add(metadata["model_version"])
        assert metadata["scenario"]["name"] == name
        assert metadata["scenario"]["seed"] == 42
        summary = json.loads(
            (output / name / "monitoring/summary.json").read_text(encoding="utf-8")
        )
        assert summary["input_feature"] == "lag_1h"
        assert summary["scenario"] == metadata["scenario"]
        html = (output / name / "monitoring/report.html").read_text(encoding="utf-8")
        assert f"<strong>{name}</strong>" in html
        if name == "invalid_requests":
            assert summary["data_quality"]["invalid_recent_count_requests"] == 6
            assert (
                records.loc[records["status_code"].eq(422), "latest_count"].eq(-1).all()
            )
    assert len(versions) == 1
    for path, digest in source_hashes.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert (output / "comparison.csv").is_file()
    html = (output / "comparison.html").read_text(encoding="utf-8")
    assert "FAULT INJECTION SIMULATION" in html
    for name in SCENARIO_NAMES:
        assert f"{name}/monitoring/report.html" in html
    assert (
        json.loads((output / "suite.json").read_text(encoding="utf-8"))["completed"]
        is True
    )


def test_monitoring_records_but_excludes_negative_recent_inputs(tmp_path):
    # Reuse a normal replay fixture shape, with one intentionally invalid input.
    folder = tmp_path / "replay"
    folder.mkdir()
    pd.DataFrame(
        {
            "target_hour": [START, START + pd.Timedelta(hours=1)],
            "model_version": ["sha256:test"] * 2,
            "status_code": [200, 422],
            "client_latency_ms": [1, 2],
            "prediction": [12, None],
            "actual": [10, 20],
            "weekly_baseline": [14, 12],
            "latest_count": [30, -1],
            "missing_count": [0, 0],
            "error": [None, "invalid count"],
        }
    ).to_csv(folder / "predictions.csv", index=False)
    metadata = {
        "simulation": True,
        "completed": True,
        "start": START.isoformat(),
        "end_exclusive": (START + pd.Timedelta(hours=2)).isoformat(),
        "planned_hours": 2,
        "model_version": "sha256:test",
        "training_target_end": "2012-09-30T23:00:00",
    }
    (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    summary = run_monitoring(folder, DATA_PATH, folder / "monitoring")
    assert summary["input_feature"] == "lag_1h"
    assert summary["input_distribution"][1]["count"] == 1
    assert summary["input_distribution"][1]["mean"] == 30
    assert summary["data_quality"]["invalid_recent_count_requests"] == 1
