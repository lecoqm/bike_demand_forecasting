"""Replay a control and three reproducible failure scenarios through the API."""

import argparse
import json
import logging
from dataclasses import replace
from html import escape
from pathlib import Path

import httpx2 as httpx
import pandas as pd

from bike_demand.monitor import run_monitoring
from bike_demand.replay import ReplayScenario, run_replay

LOGGER = logging.getLogger(__name__)
SCENARIO_NAMES = ["control", "missing_history", "invalid_requests", "demand_shift"]


def write_comparison(
    comparison: pd.DataFrame, output_dir: Path, start: pd.Timestamp, end: pd.Timestamp
) -> None:
    """Keep coverage beside error scores when scenarios reject different hours."""

    def table(columns):
        labels = {
            "scenario": "Scenario",
            "predictions": "Predictions",
            "coverage_pct": "Coverage (%)",
            "rejected": "Rejected",
            "failed": "Other failures",
            "mean_missing_history_hours": "Mean missing hours",
            "missing_outcomes": "Missing outcomes",
            "scored_hours": "Scored hours",
            "model_mae": "Model MAE",
            "baseline_mae": "Baseline MAE",
            "client_p95_ms": "P95 client (ms)",
            "input_mean": "Mean recent count",
        }
        return (
            comparison[columns]
            .rename(columns=labels)
            .to_html(
                index=False,
                na_rep="n/a",
                float_format=lambda value: f"{value:,.2f}",
                border=0,
            )
        )

    service = table(
        [
            "scenario",
            "predictions",
            "coverage_pct",
            "rejected",
            "failed",
            "mean_missing_history_hours",
            "missing_outcomes",
        ]
    )
    performance = table(
        [
            "scenario",
            "scored_hours",
            "model_mae",
            "baseline_mae",
            "client_p95_ms",
            "input_mean",
        ]
    )
    links = "".join(
        f'<li><a href="{name}/monitoring/report.html">{name}</a></li>'
        for name in SCENARIO_NAMES
    )
    html = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bike demand failure scenarios</title>
<style>body{{font:15px/1.6 system-ui,sans-serif;max-width:1160px;margin:40px auto;padding:0 20px;color:#172033;background:#f4f6f8}}
section{{background:white;padding:24px;margin:24px 0;border:1px solid #dce3eb;border-radius:12px;overflow-x:auto}}
table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #e5e9ef}}
h1{{line-height:1.2}}h2{{font-size:21px}}a{{color:#1857a4}}</style>
<p>HISTORICAL REPLAY · FAULT INJECTION SIMULATION</p><h1>Failure scenario comparison</h1>
<p>{escape(start.isoformat())} to {escape(end.isoformat())} (end exclusive). All cases use the same model and target hours.</p>
<section><h2>Service and data</h2>{service}<p>Coverage is a percentage of scheduled hours. Mean missing history is the number of unavailable observations per 168-hour request.</p></section>
<section><h2>Prediction quality</h2>{performance}
<p>Each case scores the model and weekly baseline on its own shared eligible hours. Faults can change coverage and the evaluated hours. These scores describe the simulation and do not select a model for promotion.</p>
<p>In demand_shift, outcomes and subsequent observations are synthetic after the recorded change time. Demand before the change and all training data remain original.</p></section>
<p>MAE is measured in rentals per hour. Input mean uses available nonnegative counts from one hour earlier. Negative inputs in invalid_requests are counted as invalid and excluded from the input distribution.</p>
<section><h2>Individual reports</h2><ul>{links}</ul>
<p>Each report records the seed and injected hours. The control has no injected faults. Rerunning with the same inputs, seed, and parameters reproduces fault placement and predictions; wall-clock latency can vary.</p></section></html>"""
    (output_dir / "comparison.html").write_text(html, encoding="utf-8")


def run_scenarios(
    data_path: Path,
    model_dir: Path,
    output_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    client: httpx.Client,
    options: ReplayScenario | None = None,
) -> pd.DataFrame:
    """Save each replay and report separately, followed by a comparison."""
    options = options or ReplayScenario()
    options.shift_hour(start, end)
    if output_dir.exists():
        raise FileExistsError(
            f"{output_dir} already exists. Choose a new --output-dir."
        )
    rows, versions, settings = [], set(), []
    for name in SCENARIO_NAMES:
        case = replace(options, name=name)
        run_dir = output_dir / name
        LOGGER.info("Starting scenario: %s", name)
        # The first replay creates the parent only after model and API checks.
        run_replay(data_path, model_dir, run_dir, start, end, client, scenario=case)
        report = run_monitoring(run_dir, data_path, run_dir / "monitoring")
        coverage, service = report["coverage"], report["service"]
        performance = report["performance"]
        versions.add(report["model_version"])
        settings.append(report["scenario"])
        rows.append(
            {
                "scenario": name,
                "requests": coverage["attempted_hours"],
                "predictions": coverage["predictions"],
                "coverage_pct": coverage["prediction_coverage_pct"],
                "rejected": service["rejected_requests"],
                "failed": service["failed_requests"],
                "missing_history_requests": report["data_quality"][
                    "requests_with_missing_history"
                ],
                "mean_missing_history_hours": report["data_quality"][
                    "mean_missing_history_hours"
                ],
                "missing_outcomes": report["data_quality"]["missing_outcomes"],
                "scored_hours": coverage["paired_scored_hours"],
                "model_mae": performance["gradient_boosting"]["mae"],
                "baseline_mae": performance["weekly_baseline"]["mae"],
                "client_p95_ms": service["client_latency_ms"]["p95"],
                "input_mean": report["input_distribution"][1]["mean"],
            }
        )
    if len(versions) != 1:
        raise ValueError("All scenarios must use the same model version.")
    comparison = pd.DataFrame(rows).astype(
        {
            "model_mae": float,
            "baseline_mae": float,
            "client_p95_ms": float,
            "input_mean": float,
            "mean_missing_history_hours": float,
        }
    )
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    manifest = {
        "simulation": True,
        "completed": True,
        "start": start.isoformat(),
        "end_exclusive": end.isoformat(),
        "model_version": next(iter(versions)),
        "scenarios": settings,
    }
    (output_dir / "suite.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_comparison(comparison, output_dir, start, end)
    LOGGER.info("Saved scenario comparison to %s", output_dir / "comparison.html")
    return comparison


def main() -> None:
    """Run four one-day replays by default, with a model already served by the API."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path("data/hour.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/run_01"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/scenarios/october_day_01")
    )
    parser.add_argument("--start", type=pd.Timestamp, default="2012-10-01")
    parser.add_argument("--end", type=pd.Timestamp, default="2012-10-02")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--missing-rate", type=float, default=0.1)
    parser.add_argument("--invalid-rate", type=float, default=0.2)
    parser.add_argument("--demand-multiplier", type=float, default=1.5)
    parser.add_argument("--shift-start", type=pd.Timestamp, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    try:
        options = ReplayScenario(
            seed=args.seed,
            missing_rate=args.missing_rate,
            invalid_rate=args.invalid_rate,
            demand_multiplier=args.demand_multiplier,
            shift_start=args.shift_start,
        )
        with httpx.Client(base_url=args.api_url, timeout=30, trust_env=False) as client:
            run_scenarios(
                args.data_path,
                args.model_dir,
                args.output_dir,
                args.start,
                args.end,
                client,
                options,
            )
    except (OSError, ValueError, KeyError, httpx.HTTPError) as exc:
        parser.exit(1, f"Scenarios failed: {exc}\n")


if __name__ == "__main__":
    main()
