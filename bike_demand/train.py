"""Train and evaluate the notebook's fixed gradient-boosting configuration."""

import argparse
import hashlib
import json
import logging
import platform
from importlib.metadata import version
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from threadpoolctl import threadpool_limits

from bike_demand.data import load_hourly_data
from bike_demand.features import make_features

LOGGER = logging.getLogger(__name__)


def split_data(
    data: pd.DataFrame, validation_start: pd.Timestamp, test_start: pd.Timestamp
) -> dict[str, pd.DataFrame]:
    """Split target hours into three ordered, nonempty periods."""
    if validation_start >= test_start:
        raise ValueError("Validation must start before the test period.")
    splits = {
        "train": data.loc[data.index < validation_start],
        "validation": data.loc[
            (data.index >= validation_start) & (data.index < test_start)
        ],
        "test": data.loc[data.index >= test_start],
    }
    if any(part.empty for part in splits.values()):
        raise ValueError(
            "Each split must contain eligible target hours; check the dates."
        )
    return splits


def make_model() -> HistGradientBoostingRegressor:
    """Use the configuration selected on validation data in the notebook."""
    return HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.06,
        max_iter=250,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=42,
    )


def evaluate(
    model: HistGradientBoostingRegressor, data: pd.DataFrame, columns: list[str]
) -> tuple[dict, pd.DataFrame]:
    """Score the model and weekly baseline on exactly the same target hours."""
    predictions = pd.DataFrame(
        {
            "actual": data["target"],
            "weekly_baseline": data["lag_168h"],
            "gradient_boosting": model.predict(data[columns]),
        },
        index=data.index,
    )
    scores = {
        name: {
            "mae": float(mean_absolute_error(predictions["actual"], predictions[name])),
            "rmse": float(
                root_mean_squared_error(predictions["actual"], predictions[name])
            ),
        }
        for name in ["weekly_baseline", "gradient_boosting"]
    }
    return scores, predictions


def run_training(
    data_path: Path,
    output_dir: Path,
    validation_start: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp | None = None,
) -> dict:
    """Prepare data, evaluate chronologically, and save the fitted model."""
    if test_end is not None and test_end <= test_start:
        raise ValueError("Test end must be after test start.")
    hourly = load_hourly_data(data_path)
    if test_end is not None:
        hourly = hourly.loc[hourly.index < test_end]
    data_sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest()
    features = make_features(hourly)
    columns = features.columns.tolist()
    data = features.assign(target=hourly["cnt"])
    # Require the same recent history and weekly reference as in the notebook.
    data = data.dropna(subset=["target", "lag_1h", "lag_168h"])
    splits = split_data(data, validation_start, test_start)
    LOGGER.info("Eligible target hours: %s", {k: len(v) for k, v in splits.items()})

    with threadpool_limits(limits=2):
        model = make_model()
        model.fit(splits["train"][columns], splits["train"]["target"])
        validation_scores, _ = evaluate(model, splits["validation"], columns)

        # Refit using earlier targets only; keep weights fixed during the test replay.
        fit_data = pd.concat([splits["train"], splits["validation"]])
        final_model = make_model()
        final_model.fit(fit_data[columns], fit_data["target"])
        test_scores, test_predictions = evaluate(final_model, splits["test"], columns)

    recorded_test_hours = int(
        hourly.loc[hourly.index >= test_start, "cnt"].notna().sum()
    )
    metrics = {
        "validation": validation_scores,
        "test": test_scores,
        "test_forecast_hours": len(splits["test"]),
        "test_coverage_pct": 100 * len(splits["test"]) / recorded_test_hours,
    }
    metadata = {
        "data_sha256": data_sha256,
        "validation_start": str(validation_start),
        "test_start": str(test_start),
        "training_target_end": str(fit_data.index.max()),
        "training_target_count": len(fit_data),
        "forecast_horizon_hours": 1,
        "feature_columns": columns,
        "model_parameters": final_model.get_params(),
        "python_version": platform.python_version(),
        "package_versions": {
            name: version(name)
            for name in ["numpy", "pandas", "scikit-learn", "joblib", "threadpoolctl"]
        },
    }
    if test_end is not None:
        metadata["test_end_exclusive"] = str(test_end)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"estimator": final_model, "feature_columns": columns},
        output_dir / "model.joblib",
    )
    for filename, result in [("metrics.json", metrics), ("metadata.json", metadata)]:
        (output_dir / filename).write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    test_predictions.to_csv(
        output_dir / "test_predictions.csv", index_label="target_hour"
    )
    LOGGER.info(
        "Test MAE: %.2f; weekly baseline: %.2f",
        test_scores["gradient_boosting"]["mae"],
        test_scores["weekly_baseline"]["mae"],
    )
    LOGGER.info("Saved model, metrics, metadata and test predictions to %s", output_dir)
    return metrics


def main() -> None:
    """Read paths and split dates from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path("data/hour.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--validation-start", type=pd.Timestamp, default="2012-07-01")
    parser.add_argument("--test-start", type=pd.Timestamp, default="2012-10-01")
    parser.add_argument(
        "--test-end",
        type=pd.Timestamp,
        default=None,
        help="Optional exclusive evaluation end.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_training(
        args.data_path,
        args.output_dir,
        args.validation_start,
        args.test_start,
        args.test_end,
    )


if __name__ == "__main__":
    main()
