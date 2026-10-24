"""Train a candidate on added observations and compare fixed models chronologically."""

import argparse
import hashlib
import json
import logging
from html import escape
from io import BytesIO
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from threadpoolctl import threadpool_limits

from bike_demand.data import ALLOWED_SHA256, load_hourly_data
from bike_demand.features import make_features
from bike_demand.track import git_tags, run_tracked_training
from bike_demand.train import evaluate, make_model

LOGGER = logging.getLogger(__name__)
ACCEPTANCE_RULE = {
    "mae": "Candidate MAE must be strictly lower than current MAE.",
    "rmse": "Candidate RMSE must be no higher than current RMSE.",
}


def candidate_decision(current: dict, candidate: dict) -> dict:
    """Apply the fixed rule, including rejection of tied MAE scores."""
    checks = {
        "lower_mae": candidate["mae"] < current["mae"],
        "no_rmse_increase": candidate["rmse"] <= current["rmse"],
    }
    accepted = all(checks.values())
    return {
        "accepted": accepted,
        "recommendation": "candidate" if accepted else "current",
        "checks": checks,
        "rule": ACCEPTANCE_RULE,
        "mae_improvement_pct": 100
        * (current["mae"] - candidate["mae"])
        / current["mae"]
        if current["mae"]
        else None,
    }


def validate_schedule(
    metadata: dict, start: pd.Timestamp, end: pd.Timestamp, last_hour: pd.Timestamp
) -> pd.Timestamp:
    """Use the current model's former test boundary as the added-data boundary."""
    current_end = pd.Timestamp(metadata["training_target_end"])
    update_start = pd.Timestamp(metadata["test_start"])
    for value in [current_end, update_start, start, end]:
        if pd.isna(value) or value.tz is not None or value != value.floor("h"):
            raise ValueError(
                "Model and comparison dates must be timezone-naive whole hours."
            )
    if not current_end < update_start < start < end:
        raise ValueError(
            "Require current training end < added-data start < comparison start < comparison end."
        )
    if end > last_hour + pd.Timedelta(hours=1):
        raise ValueError("Comparison end is outside the dataset.")
    return update_start


def write_report(result: dict, scores: pd.DataFrame, output_dir: Path) -> None:
    """Present scoring coverage, provenance, and the recommendation together."""
    decision = result["decision"]
    message = (
        "Candidate passes the acceptance rule"
        if decision["accepted"]
        else "Keep the current model"
    )
    table = scores.rename(
        columns={
            "model": "Model",
            "scored_hours": "Hours",
            "mae": "MAE",
            "rmse": "RMSE",
        }
    ).to_html(index=False, float_format=lambda value: f"{value:,.2f}", border=0)
    checks = "".join(
        f"<li>{escape(ACCEPTANCE_RULE[name])} <strong>{'PASS' if passed else 'FAIL'}</strong></li>"
        for name, passed in [
            ("mae", decision["checks"]["lower_mae"]),
            ("rmse", decision["checks"]["no_rmse_increase"]),
        ]
    )
    coverage = result["coverage"]
    current, candidate = result["current"], result["candidate"]
    html = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bike demand candidate comparison</title>
<style>body{{font:15px/1.6 system-ui,sans-serif;max-width:1000px;margin:40px auto;padding:0 20px;background:#f4f6f8;color:#172033}}
section{{background:white;border:1px solid #dce3eb;border-radius:12px;padding:24px;margin:24px 0}}h1{{line-height:1.2}}h2{{font-size:21px}}
table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #e5e9ef}}
code{{font-size:12px;overflow-wrap:anywhere}}.muted{{color:#526174}}</style>
<p>RETROSPECTIVE MODEL UPDATE SIMULATION</p><h1>Candidate comparison</h1>
<p>Comparison: {escape(result["comparison_start"])} to {escape(result["comparison_end_exclusive"])} (end exclusive).</p>
<section><h2>{message}</h2><ul>{checks}</ul><p>This recommendation is recorded for review. Model promotion is a separate operation.</p></section>
<section><h2>Scores on identical target hours</h2>{table}
<p>MAE and RMSE are in rentals per hour. Both fixed models and the weekly baseline use the same {coverage["paired_scored_hours"]} eligible hours.</p>
<p>Scoring coverage: {coverage["scoring_coverage_pct"]:.2f}% of {coverage["scheduled_hours"]} scheduled hours; {coverage["known_outcomes"]} outcomes are recorded. Hours without outcomes or required lags remain in the CSV and coverage counts.</p></section>
<section><h2>Training chronology</h2>
<p>Current model: trained through {escape(current["training_target_end"])}.</p>
<p>Added observations: {escape(result["added_data_start"])} to {escape(result["comparison_start"])} (end exclusive), {result["added_training_hours"]} eligible training targets.</p>
<p>Candidate: trained through {escape(candidate["training_target_end"])}, with the same model configuration. Evaluation outcomes do not enter fitting. Earlier evaluation outcomes become history for subsequent one-hour forecasts.</p>
<p>The inputs are the original UCI observations. This is a retrospective engineering demonstration using periods already available in the project.</p></section>
<section><h2>Traceability</h2>
<p>Current model: <code>{escape(current["model_version"])}</code></p>
<p>Candidate model: <code>{escape(candidate["model_version"])}</code></p>
<p>Candidate MLflow run: <code>{escape(candidate["run_id"])}</code></p>
<p class="muted">The acceptance rule was written to plan.json before candidate training. comparison.json records the dates, model paths, checksums, scores, and recommendation.</p></section></html>"""
    (output_dir / "comparison.html").write_text(html, encoding="utf-8")


def run_comparison(
    data_path: Path,
    current_model_dir: Path,
    output_dir: Path,
    comparison_start: pd.Timestamp,
    comparison_end: pd.Timestamp,
    tracking_uri: str = "sqlite:///mlflow.db",
) -> dict:
    """Refit on earlier targets, then score both models on one shared period."""
    metadata_bytes = (current_model_dir / "metadata.json").read_bytes()
    current_metadata = json.loads(metadata_bytes)
    model_bytes = (current_model_dir / "model.joblib").read_bytes()
    current = joblib.load(BytesIO(model_bytes))
    hourly = load_hourly_data(data_path)
    update_start = validate_schedule(
        current_metadata, comparison_start, comparison_end, hourly.index.max()
    )
    features = make_features(hourly)
    columns = features.columns.tolist()
    if (
        current["feature_columns"] != columns
        or current_metadata["feature_columns"] != columns
    ):
        raise ValueError("Current model features differ from the feature builder.")
    if current["estimator"].get_params() != make_model().get_params():
        raise ValueError(
            "Current model parameters differ from the fixed training configuration."
        )
    if (
        current_metadata["forecast_horizon_hours"] != 1
        or current_metadata["data_sha256"] not in ALLOWED_SHA256
    ):
        raise ValueError(
            "Current model must use the original UCI data and one-hour horizon."
        )
    data = features.assign(target=hourly["cnt"])
    eligible = data.dropna(subset=["target", "lag_1h", "lag_168h"])
    added = eligible.loc[
        (eligible.index >= update_start) & (eligible.index < comparison_start)
    ]
    evaluation = eligible.loc[
        (eligible.index >= comparison_start) & (eligible.index < comparison_end)
    ]
    if added.empty or evaluation.empty:
        raise ValueError(
            "Added-data and comparison periods must both contain eligible targets."
        )
    if output_dir.exists():
        raise FileExistsError(
            f"{output_dir} already exists. Choose a new --output-dir."
        )
    plan = {
        "simulation": True,
        "git": git_tags(),
        "current_model_version": "sha256:" + hashlib.sha256(model_bytes).hexdigest(),
        "current_metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "current_training_target_end": current_metadata["training_target_end"],
        "added_data_start": update_start.isoformat(),
        "added_training_hours": len(added),
        "candidate_training_before": comparison_start.isoformat(),
        "comparison_start": comparison_start.isoformat(),
        "comparison_end_exclusive": comparison_end.isoformat(),
        "acceptance_rule": ACCEPTANCE_RULE,
    }
    output_dir.mkdir(parents=True)
    (output_dir / "plan.json").write_text(
        json.dumps(plan, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    # The existing training helper fits only targets before comparison_start.
    run_id = run_tracked_training(
        data_path,
        output_dir / "candidate",
        update_start,
        comparison_start,
        tracking_uri=tracking_uri,
        run_name="candidate-comparison",
        test_end=comparison_end,
    )
    with mlflow.start_run(run_id=run_id):
        candidate_dir = output_dir / "candidate" / run_id
        candidate_bytes = (candidate_dir / "model.joblib").read_bytes()
        candidate = joblib.load(BytesIO(candidate_bytes))
        candidate_metadata = json.loads(
            (candidate_dir / "metadata.json").read_text(encoding="utf-8")
        )
        with threadpool_limits(limits=2):
            current_scores, current_predictions = evaluate(
                current["estimator"], evaluation, columns
            )
            candidate_scores, candidate_predictions = evaluate(
                candidate["estimator"], evaluation, columns
            )
        metrics = {
            "current": current_scores["gradient_boosting"],
            "candidate": candidate_scores["gradient_boosting"],
            "weekly_baseline": current_scores["weekly_baseline"],
        }
        decision = candidate_decision(metrics["current"], metrics["candidate"])
        scheduled = hourly.loc[
            (hourly.index >= comparison_start) & (hourly.index < comparison_end)
        ]
        predictions = pd.DataFrame(
            {
                "actual": scheduled["cnt"],
                "weekly_baseline": features.loc[scheduled.index, "lag_168h"],
                "eligible": scheduled.index.isin(evaluation.index),
                "current": current_predictions["gradient_boosting"],
                "candidate": candidate_predictions["gradient_boosting"],
            },
            index=scheduled.index,
        )
        coverage = {
            "scheduled_hours": len(scheduled),
            "known_outcomes": int(scheduled["cnt"].notna().sum()),
            "paired_scored_hours": len(evaluation),
            "scoring_coverage_pct": 100 * len(evaluation) / len(scheduled),
        }
        result = {
            "simulation": True,
            "completed": True,
            "comparison_start": comparison_start.isoformat(),
            "comparison_end_exclusive": comparison_end.isoformat(),
            "added_data_start": update_start.isoformat(),
            "added_training_hours": len(added),
            "data_sha256": plan["data_sha256"],
            "git": plan["git"],
            "current": {
                "model_path": str(current_model_dir / "model.joblib"),
                "model_version": plan["current_model_version"],
                "training_target_end": current_metadata["training_target_end"],
            },
            "candidate": {
                "model_path": str(candidate_dir / "model.joblib"),
                "model_version": "sha256:"
                + hashlib.sha256(candidate_bytes).hexdigest(),
                "training_target_end": candidate_metadata["training_target_end"],
                "training_target_count": candidate_metadata["training_target_count"],
                "run_id": run_id,
            },
            "coverage": coverage,
            "metrics": metrics,
            "decision": decision,
        }
        scores = pd.DataFrame(
            [
                {"model": name, "scored_hours": len(evaluation), **values}
                for name, values in metrics.items()
            ]
        )
        scores.to_csv(output_dir / "scores.csv", index=False)
        predictions.to_csv(
            output_dir / "comparison_predictions.csv", index_label="target_hour"
        )
        (output_dir / "comparison.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        write_report(result, scores, output_dir)
        mlflow.set_tags(
            {
                "model_role": "candidate",
                "candidate_accepted": str(decision["accepted"]).lower(),
            }
        )
        mlflow.log_params(
            {
                "current_model_version": plan["current_model_version"],
                "comparison_start": comparison_start.isoformat(),
                "comparison_end_exclusive": comparison_end.isoformat(),
                "acceptance_rule": json.dumps(ACCEPTANCE_RULE, sort_keys=True),
            }
        )
        for name, values in metrics.items():
            mlflow.log_metrics(
                {f"comparison.{name}.{key}": value for key, value in values.items()}
            )
        mlflow.log_metrics(
            {
                "comparison.paired_scored_hours": len(evaluation),
                "comparison.scoring_coverage_pct": coverage["scoring_coverage_pct"],
                "added_training_hours": len(added),
            }
        )
        for name in [
            "plan.json",
            "comparison.json",
            "scores.csv",
            "comparison_predictions.csv",
            "comparison.html",
        ]:
            mlflow.log_artifact(str(output_dir / name), artifact_path="comparison")
    LOGGER.info(
        "Current MAE: %.2f; candidate MAE: %.2f",
        metrics["current"]["mae"],
        metrics["candidate"]["mae"],
    )
    LOGGER.info("Recommendation: %s", decision["recommendation"])
    LOGGER.info("Saved comparison to %s", output_dir / "comparison.html")
    return result


def main() -> None:
    """Use October as added training data and November as the comparison period."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path("data/hour.csv"))
    parser.add_argument(
        "--current-model-dir", type=Path, default=Path("artifacts/run_01")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/comparisons/october_update")
    )
    parser.add_argument("--comparison-start", type=pd.Timestamp, default="2012-11-01")
    parser.add_argument("--comparison-end", type=pd.Timestamp, default="2012-12-01")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        run_comparison(
            args.data_path,
            args.current_model_dir,
            args.output_dir,
            args.comparison_start,
            args.comparison_end,
            args.tracking_uri,
        )
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Comparison failed: {exc}\n")


if __name__ == "__main__":
    main()
