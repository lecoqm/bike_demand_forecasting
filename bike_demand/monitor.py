"""Build an offline monitoring report from a chronological API replay."""

import argparse
import base64
import hashlib
import json
import logging
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import dates as mdates
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from bike_demand.data import load_hourly_data

LOGGER = logging.getLogger(__name__)
NUMERIC_COLUMNS = [
    "status_code",
    "client_latency_ms",
    "prediction",
    "actual",
    "weekly_baseline",
    "missing_count",
]


def load_replay(replay_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Validate the replay files before using their rows as metric denominators."""
    metadata = json.loads((replay_dir / "metadata.json").read_text(encoding="utf-8"))
    records = pd.read_csv(replay_dir / "predictions.csv")
    required = {"target_hour", "model_version", "error", *NUMERIC_COLUMNS}
    if not required.issubset(records.columns):
        raise ValueError(
            f"Replay CSV is missing columns: {sorted(required - set(records.columns))}"
        )
    records["target_hour"] = pd.to_datetime(records["target_hour"], errors="raise")
    hours = records["target_hour"]
    if (
        hours.isna().any()
        or hours.dt.tz is not None
        or hours.duplicated().any()
        or not hours.is_monotonic_increasing
        or not hours.eq(hours.dt.floor("h")).all()
    ):
        raise ValueError(
            "Replay target hours must be sorted, unique, timezone-naive whole hours."
        )
    for name in NUMERIC_COLUMNS:
        values = pd.to_numeric(records[name], errors="raise")
        if ((values.notna() & ~np.isfinite(values)) | values.lt(0)).any():
            raise ValueError(
                f"{name} must contain finite nonnegative values or missing values."
            )
        records[name] = values
    if "latest_count" in records:
        values = pd.to_numeric(records["latest_count"], errors="raise")
        if (values.notna() & ~np.isfinite(values)).any():
            raise ValueError("latest_count must be finite or missing.")
        # Negative inputs are retained for diagnosing rejected requests.
        records["latest_count"] = values
    for name, maximum in [("status_code", 599), ("missing_count", 168)]:
        values = records[name].dropna()
        if (values.gt(maximum) | values.ne(values.round())).any():
            raise ValueError(f"Invalid {name} in replay CSV.")
    if records["status_code"].dropna().lt(100).any():
        raise ValueError("Invalid status_code in replay CSV.")
    start, end = (
        pd.Timestamp(metadata["start"]),
        pd.Timestamp(metadata["end_exclusive"]),
    )
    if (
        any(
            pd.isna(value) or value.tz is not None or value != value.floor("h")
            for value in [start, end]
        )
        or start >= end
    ):
        raise ValueError(
            "Replay metadata must describe an ordered period of whole hours."
        )
    expected = pd.date_range(start, end, freq="h", inclusive="left")
    if metadata["planned_hours"] != len(expected) or not hours.isin(expected).all():
        raise ValueError(
            "Replay rows or planned_hours disagree with the metadata period."
        )
    if metadata["completed"] and len(records) != len(expected):
        raise ValueError("A completed replay must contain every scheduled hour.")
    return records, metadata


def percentage(count: int, total: int) -> float | None:
    """Keep rates undefined when their denominator is zero."""
    return 100 * count / total if total else None


def distribution(values: pd.Series) -> dict:
    """Describe available values; do not turn missing observations into zeros."""
    values = values.dropna()
    return {
        "count": len(values),
        "mean": float(values.mean()) if len(values) else None,
        "median": float(values.median()) if len(values) else None,
        "p95": float(values.quantile(0.95, interpolation="linear"))
        if len(values)
        else None,
    }


def summarize(
    records: pd.DataFrame, metadata: dict
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Count all attempts and compare both models on exactly the same outcomes."""
    valid = (
        records["status_code"].eq(200)
        & records["prediction"].notna()
        & records["error"].fillna("").eq("")
        & records["model_version"].eq(metadata["model_version"])
    )
    rejected = records["status_code"].between(400, 499)
    scored = valid & records["actual"].notna()
    paired = records.loc[scored & records["weekly_baseline"].notna()].copy()
    metrics = []
    for name, column in [
        ("gradient_boosting", "prediction"),
        ("weekly_baseline", "weekly_baseline"),
    ]:
        error = paired[column] - paired["actual"]
        metrics.append(
            {
                "model": name,
                "scored_hours": len(paired),
                "mae": float(error.abs().mean()) if len(paired) else None,
                "rmse": float(np.sqrt((error**2).mean())) if len(paired) else None,
            }
        )
    scores = {
        item["model"]: {key: value for key, value in item.items() if key != "model"}
        for item in metrics
    }
    baseline_mae = scores["weekly_baseline"]["mae"]
    improvement = (
        100 * (baseline_mae - scores["gradient_boosting"]["mae"]) / baseline_mae
        if baseline_mae
        else None
    )
    missing = records["missing_count"]
    summary = {
        "simulation": metadata["simulation"],
        "scenario": metadata.get("scenario", {"name": "control"}),
        "replay_completed": metadata["completed"],
        "start": metadata["start"],
        "end_exclusive": metadata["end_exclusive"],
        "model_version": metadata["model_version"],
        "training_target_end": metadata["training_target_end"],
        "coverage": {
            "scheduled_hours": metadata["planned_hours"],
            "attempted_hours": len(records),
            "predictions": int(valid.sum()),
            "prediction_coverage_pct": percentage(
                int(valid.sum()), metadata["planned_hours"]
            ),
            "known_outcomes": int(records["actual"].notna().sum()),
            "scored_predictions": int(scored.sum()),
            "paired_scored_hours": len(paired),
        },
        "service": {
            "rejected_requests": int(rejected.sum()),
            "failed_requests": int((~valid & ~rejected).sum()),
            "error_rate_pct": percentage(int((~valid).sum()), len(records)),
            "client_latency_ms": distribution(records["client_latency_ms"]),
        },
        "data_quality": {
            "invalid_recent_count_requests": int(records["latest_count"].lt(0).sum())
            if "latest_count" in records
            else None,
            "missing_outcomes": int(records["actual"].isna().sum()),
            "history_count_known_requests": int(missing.notna().sum()),
            "requests_with_missing_history": int(missing.gt(0).sum()),
            "requests_with_missing_history_pct": percentage(
                int(missing.gt(0).sum()), int(missing.notna().sum())
            ),
            "mean_missing_history_hours": float(missing.mean())
            if missing.notna().any()
            else None,
            "unexpected_model_versions": int(
                (
                    records["model_version"].notna()
                    & records["model_version"].ne(metadata["model_version"])
                ).sum()
            ),
        },
        "performance": {**scores, "mae_improvement_pct": improvement},
    }
    hourly = []
    for hour in range(24):
        group = paired.loc[paired["target_hour"].dt.hour.eq(hour)]
        hourly.append(
            {
                "hour_of_day": hour,
                "scored_hours": len(group),
                "gradient_boosting_mae": float(
                    (group["prediction"] - group["actual"]).abs().mean()
                )
                if len(group)
                else None,
                "weekly_baseline_mae": float(
                    (group["weekly_baseline"] - group["actual"]).abs().mean()
                )
                if len(group)
                else None,
            }
        )
    metrics_table = pd.DataFrame(metrics).astype({"mae": float, "rmse": float})
    return summary, metrics_table, pd.DataFrame(hourly)


def training_reference(
    data_path: Path, training_target_end: str, lag_hours: int = 168
) -> pd.Series:
    """Reconstruct a lag input for the final model's eligible fit hours."""
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Reference CSV not found: {data_path}. Set --data-path to the existing UCI CSV."
        )
    hourly = load_hourly_data(data_path)
    demand = hourly["cnt"]
    weekly = demand.shift(168)
    eligible = (
        (hourly.index <= pd.Timestamp(training_target_end))
        & demand.notna()
        & demand.shift(1).notna()
        & weekly.notna()
    )
    reference = demand.shift(lag_hours).loc[eligible]
    if reference.empty:
        raise ValueError("No eligible training inputs before training_target_end.")
    return reference


def make_plot(
    records: pd.DataFrame,
    hourly: pd.DataFrame,
    reference: pd.Series,
    output_path: Path,
    scenario_name: str = "control",
    input_lag_hours: int = 168,
) -> None:
    """Write four static panels without requiring a desktop or a running API."""
    figure = Figure(figsize=(12, 8), layout="constrained")
    qualifier = " · synthetic outcomes" if scenario_name == "demand_shift" else ""
    figure.suptitle(
        f"Historical replay simulation · {scenario_name}{qualifier}", fontsize=13
    )
    FigureCanvasAgg(figure)
    forecast, error, latency, inputs = figure.subplots(2, 2).flat
    for column, label, color in [
        (
            "actual",
            "Scenario outcome" if scenario_name == "demand_shift" else "Actual",
            "#111827",
        ),
        ("prediction", "Model", "#2563eb"),
        ("weekly_baseline", "Weekly baseline", "#d97706"),
    ]:
        forecast.plot(
            records["target_hour"],
            records[column],
            label=label,
            color=color,
            linewidth=1.5,
        )
    locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
    forecast.xaxis.set_major_locator(locator)
    forecast.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    forecast.set(title="Demand and forecasts", ylabel="Rentals per hour")
    forecast.legend(fontsize=8)

    x = hourly["hour_of_day"].to_numpy()
    if hourly["scored_hours"].sum():
        error.bar(
            x - 0.2,
            hourly["gradient_boosting_mae"],
            width=0.4,
            label="Model",
            color="#2563eb",
        )
        error.bar(
            x + 0.2,
            hourly["weekly_baseline_mae"],
            width=0.4,
            label="Weekly baseline",
            color="#d97706",
        )
        error.legend(fontsize=8)
    else:
        error.text(
            0.5,
            0.5,
            "No paired outcomes to score",
            ha="center",
            transform=error.transAxes,
        )
    error.set(
        title="MAE by target hour · paired outcomes",
        xlabel="Hour of day",
        ylabel="MAE (rentals per hour)",
        xticks=range(0, 24, 3),
    )

    durations = records["client_latency_ms"].dropna()
    if len(durations):
        latency.hist(
            durations,
            bins=min(20, max(1, int(np.sqrt(len(durations))))),
            color="#2563eb",
            alpha=0.7,
        )
        for quantile, label, color in [
            (0.5, "Median", "#111827"),
            (0.95, "P95", "#d97706"),
        ]:
            value = durations.quantile(quantile, interpolation="linear")
            latency.axvline(
                value, label=f"{label}: {value:.1f} ms", color=color, linestyle="--"
            )
        latency.legend(fontsize=8)
    else:
        latency.text(
            0.5, 0.5, "No recorded timings", ha="center", transform=latency.transAxes
        )
    latency.set(
        title="Client round-trip latency · all attempts",
        xlabel="Milliseconds",
        ylabel="Requests",
    )

    input_column = "latest_count" if input_lag_hours == 1 else "weekly_baseline"
    current = records[input_column].where(records[input_column].ge(0)).dropna()
    bins = np.histogram_bin_edges(
        np.concatenate([reference.to_numpy(), current.to_numpy()]), bins=20
    )
    for values, label, color in [
        (reference, "Training reference", "#0f766e"),
        (current, "Replay", "#2563eb"),
    ]:
        if len(values):
            inputs.hist(
                values,
                bins=bins,
                weights=np.full(len(values), 100 / len(values)),
                histtype="step",
                linewidth=1.8,
                label=label,
                color=color,
            )
    inputs.set(
        title="Demand input · descriptive comparison",
        xlabel=f"Count {input_lag_hours} hour(s) earlier",
        ylabel="Observations (%)",
    )
    inputs.legend(fontsize=8)
    for axis in [forecast, error, latency, inputs]:
        axis.grid(axis="y", alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output_path, dpi=150)


def write_html(
    summary: dict, metrics: pd.DataFrame, inputs: pd.DataFrame, output_dir: Path
) -> None:
    """Embed the plot so the HTML remains readable when copied by itself."""

    def number(value, suffix=""):
        return "n/a" if value is None else f"{value:,.2f}{suffix}"

    def table(rows):
        return pd.DataFrame(rows, columns=["Measure", "Value"]).to_html(
            index=False, escape=True, border=0
        )

    coverage, service, quality = (
        summary[name] for name in ["coverage", "service", "data_quality"]
    )
    duration = service["client_latency_ms"]
    status = (
        "Completed replay"
        if summary["replay_completed"]
        else "Partial replay — scheduled hours remain in the coverage denominator"
    )
    scenario = summary["scenario"]
    scenario_settings = escape(json.dumps(scenario, indent=2, allow_nan=False))
    image = base64.b64encode((output_dir / "monitoring.png").read_bytes()).decode(
        "ascii"
    )
    score_table = metrics.rename(
        columns={
            "model": "Model",
            "scored_hours": "Hours",
            "mae": "MAE",
            "rmse": "RMSE",
        }
    ).to_html(
        index=False, float_format=lambda value: f"{value:,.2f}", na_rep="n/a", border=0
    )
    input_table = inputs.rename(
        columns={
            "population": "Population",
            "count": "Hours",
            "mean": "Mean",
            "median": "Median",
            "p95": "P95",
        }
    ).to_html(
        index=False, float_format=lambda value: f"{value:,.2f}", na_rep="n/a", border=0
    )
    service_table = table(
        [
            ("Rejected requests (HTTP 4xx)", service["rejected_requests"]),
            ("Other failed attempts", service["failed_requests"]),
            ("Error rate / attempted hours", number(service["error_rate_pct"], "%")),
            ("Median client latency", number(duration["median"], " ms")),
            ("P95 client latency", number(duration["p95"], " ms")),
            ("Attempts with recorded latency", duration["count"]),
        ]
    )
    quality_table = table(
        [
            (
                "Known outcomes / attempted hours",
                f"{coverage['known_outcomes']} / {coverage['attempted_hours']}",
            ),
            ("Missing outcomes", quality["missing_outcomes"]),
            ("Predictions with known outcomes", coverage["scored_predictions"]),
            ("Requests with missing history", quality["requests_with_missing_history"]),
            (
                "Requests with a known history-missing count",
                quality["history_count_known_requests"],
            ),
            ("Unexpected model versions", quality["unexpected_model_versions"]),
            (
                "Invalid recent counts",
                quality["invalid_recent_count_requests"]
                if quality["invalid_recent_count_requests"] is not None
                else "n/a",
            ),
        ]
    )
    html = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bike demand monitoring report</title>
<style>
body{{font:15px/1.55 system-ui,sans-serif;background:#f4f6f8;color:#172033;margin:0}}
main{{max-width:1120px;margin:36px auto;padding:0 24px 40px}}h1{{font-size:32px;line-height:1.2;margin:8px 0}}
h2{{font-size:20px;margin:0 0 12px}}p{{margin:8px 0}}.muted{{color:#526174}}.badge{{color:#1857a4;font-weight:650}}
.cards,.columns{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px;margin:24px 0}}.columns{{grid-template-columns:1fr 1fr}}
section,.card{{background:white;border:1px solid #dce3eb;border-radius:12px;padding:20px;margin:20px 0}}
.card{{margin:0}}.card strong{{display:block;font-size:27px;margin-top:8px}}.columns section{{margin:0}}
table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}th,td{{text-align:left;padding:9px 8px;border-bottom:1px solid #e5e9ef}}
th{{color:#526174;font-weight:600}}img{{display:block;width:100%;height:auto}}code{{overflow-wrap:anywhere;font-size:12px}}
pre{{font-size:12px;max-height:240px;overflow:auto}}summary{{cursor:pointer}}
@media(max-width:760px){{.cards,.columns{{grid-template-columns:1fr 1fr}}.columns{{grid-template-columns:1fr}}main{{padding:0 12px}}}}
</style><main>
<p class="badge">HISTORICAL REPLAY SIMULATION</p><h1>Bike demand monitoring</h1>
<p class="muted">{escape(str(summary["start"]))} to {escape(str(summary["end_exclusive"]))} (end exclusive)</p>
<p>{escape(status)}</p>
<p>Scenario: <strong>{escape(str(scenario["name"]))}</strong>. Injected faults are simulated; demand_shift also changes outcomes after its recorded change time.</p>
<details><summary>Scenario settings</summary><pre>{scenario_settings}</pre></details>
<div class="cards">
<div class="card">Attempted / scheduled hours<strong>{coverage["attempted_hours"]} / {coverage["scheduled_hours"]}</strong></div>
<div class="card">Prediction coverage<strong>{number(coverage["prediction_coverage_pct"], "%")}</strong></div>
<div class="card">Model MAE · paired hours<strong>{number(summary["performance"]["gradient_boosting"]["mae"])}</strong></div>
<div class="card">P95 client latency<strong>{number(duration["p95"], " ms")}</strong></div>
</div>
<section><h2>Prediction quality</h2>{score_table}
<p class="muted">Both models use the same {coverage["paired_scored_hours"]} hours with a valid API prediction, a known outcome, and a weekly baseline. MAE and RMSE are in rentals per hour.</p>
<p>MAE reduction versus baseline: {number(summary["performance"]["mae_improvement_pct"], "%")}.</p></section>
<div class="columns"><section><h2>Service</h2>{service_table}</section><section><h2>Data quality</h2>{quality_table}</section></div>
<section><h2>Replay plots</h2><img src="data:image/png;base64,{image}" alt="Demand forecasts, error by hour, client latency, and demand input distributions"></section>
<section><h2>Demand input · {escape(summary["input_feature"])}</h2>{input_table}
<p class="muted">The reference uses eligible training targets through {escape(str(summary["training_target_end"]))}. Replay statistics include attempts with a known nonnegative input count. The comparison is descriptive; different calendar mixes can change this distribution.</p></section>
<section><h2>How to read this report</h2>
<p>Prediction coverage is valid predictions divided by all scheduled hours, including unattempted hours in a partial replay. The error rate includes rejections, transport errors, server errors, and invalid responses.</p>
<p>Latency measures the client HTTP round trip for all attempts, including failures. It differs from server processing time. Missing values and undefined scores are shown as n/a.</p>
<p>A one-day replay is a smoke test. Longer periods are needed to examine weekday patterns and changes over time. Input comparisons do not establish concept drift or trigger model promotion.</p>
<p class="muted">Model: <code>{escape(str(summary["model_version"]))}</code></p></section>
</main></html>"""
    (output_dir / "report.html").write_text(html, encoding="utf-8")


def run_monitoring(replay_dir: Path, data_path: Path, output_dir: Path) -> dict:
    """Generate HTML, a four-panel PNG, and machine-readable summaries."""
    records, metadata = load_replay(replay_dir)
    summary, metrics, hourly = summarize(records, metadata)
    lag_hours = 1 if "latest_count" in records else 168
    input_column = "latest_count" if lag_hours == 1 else "weekly_baseline"
    summary["input_feature"] = f"lag_{lag_hours}h"
    reference = training_reference(
        data_path, metadata["training_target_end"], lag_hours
    )
    input_rows = [
        {"population": "Training reference", **distribution(reference)},
        {
            "population": "Replay",
            **distribution(records[input_column].where(records[input_column].ge(0))),
        },
    ]
    inputs = pd.DataFrame(input_rows)
    summary["input_distribution"] = input_rows
    summary["source_sha256"] = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in [
            ("predictions", replay_dir / "predictions.csv"),
            ("replay_metadata", replay_dir / "metadata.json"),
            ("reference_data", data_path),
        ]
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    hourly.to_csv(output_dir / "by_hour.csv", index=False)
    inputs.to_csv(output_dir / "input_distribution.csv", index=False)
    # Hide any prediction rejected by the same validity rules as the scores.
    plot_records = records.copy()
    invalid = (
        plot_records["status_code"].ne(200)
        | plot_records["error"].fillna("").ne("")
        | plot_records["model_version"].ne(metadata["model_version"])
    )
    plot_records.loc[invalid, "prediction"] = np.nan
    make_plot(
        plot_records,
        hourly,
        reference,
        output_dir / "monitoring.png",
        summary["scenario"]["name"],
        lag_hours,
    )
    write_html(summary, metrics, inputs, output_dir)
    LOGGER.info(
        "Coverage: %s predictions / %s scheduled hours",
        summary["coverage"]["predictions"],
        metadata["planned_hours"],
    )
    LOGGER.info("Saved monitoring report to %s", output_dir / "report.html")
    return summary


def main() -> None:
    """Read existing files; no API or MLflow server is needed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay-dir", type=Path, default=Path("artifacts/replays/october_day_01")
    )
    parser.add_argument("--data-path", type=Path, default=Path("data/hour.csv"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        run_monitoring(
            args.replay_dir,
            args.data_path,
            args.output_dir or args.replay_dir / "monitoring",
        )
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Monitoring failed: {exc}\n")


if __name__ == "__main__":
    main()
