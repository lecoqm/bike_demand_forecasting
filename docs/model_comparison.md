# Train and compare a candidate

This step refits the fixed model configuration with additional original UCI
observations, then evaluates the current model and candidate on identical later
hours. It records a recommendation for review. Promotion follows separately.

## Default chronology

| Phase | Period |
| --- | --- |
| Current model | `artifacts/run_01`, trained through September 30, 2012 |
| Added observations | October 2012 |
| Candidate fitting | All eligible targets before November 1, including October |
| Shared comparison | November 1 to December 1, 2012, end exclusive |
| Subsequent demonstration | December, excluded from this fitting and comparison |

The command uses original observations; the failure-scenario outputs do not
enter training. Model settings remain fixed. The acceptance rule is recorded
in `plan.json` before training starts:

1. Candidate MAE must be strictly lower than current MAE.
2. Candidate RMSE must be no higher than current RMSE.

Both checks must pass. Tied MAE scores keep the current model. This is a
retrospective engineering demonstration using a dataset already explored in the
project; the comparison period is not a new blind research test.

## Run

From the repository root in the project environment:

```powershell
$env:MLFLOW_DISABLE_TELEMETRY = "true"
python -m ruff check bike_demand tests
python -m pytest
```

Commit the code before recording the experiment, so the Git revision in MLflow
identifies that committed version. Then run:

```powershell
python -m bike_demand.compare
Start-Process "artifacts/comparisons/october_update/comparison.html"
```

Training writes directly to the local SQLite tracking database. The environment
setting above disables MLflow's optional usage telemetry; experiment logging
still works. The API and MLflow web server are optional. Start the MLflow
interface later with the same repository-root command used previously.

## Outputs

`artifacts/comparisons/october_update/` contains:

- `plan.json`: dates, added training count, fixed acceptance rule, current model
  identity, data checksum, and Git revision/status, written before fitting.
- `candidate/<run_id>/`: candidate model, metadata, metrics and predictions.
- `comparison_predictions.csv`: every scheduled hour, its outcome and weekly
  baseline, a shared scoring-eligibility flag, and the two model predictions.
- `scores.csv`: MAE/RMSE for current model, candidate, and weekly baseline, with
  an identical scored-hour count.
- `comparison.json`: model paths and checksums, candidate MLflow run ID,
  training/evaluation dates, coverage, scores, and acceptance checks.
- `comparison.html`: the readable report.

The candidate's MLflow run contains its usual `training/` artifacts plus the
`comparison/` files, comparison metrics, and the `candidate_accepted` tag.
The same current model file is preserved for the service and later rollback.

All models are scored on the same hours with a known outcome, latest count,
and weekly reference. Missing internal history values stay missing, matching
training and serving. Excluded hours remain in the comparison CSV. Scoring
coverage uses all scheduled clock hours as its denominator.

Both models keep fixed weights during evaluation. As in the earlier one-hour
replay, outcomes from earlier evaluation hours may become inputs to later
forecasts. Evaluation targets never enter fitting. The training command's new
optional `--test-end` prevents December from entering this run's evaluation;
omitting it preserves the earlier training command's date behavior.

To repeat the experiment, choose a fresh output directory:

```powershell
python -m bike_demand.compare --output-dir artifacts/comparisons/october_update_repeat
```

`--current-model-dir`, `--data-path`, `--comparison-start`, `--comparison-end`,
and `--tracking-uri` are configurable. The added-data period begins at the
current model metadata's `test_start` and ends at the exclusive candidate
fitting boundary, `--comparison-start`. Each period must be nonempty and
chronologically ordered. Existing comparison directories are never overwritten.
