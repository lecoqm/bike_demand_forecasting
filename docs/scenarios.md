# Reproducible failure scenarios

This step sends simulated faults through the existing API and generates a
monitoring report for each case. It uses a fixed saved model and does not retrain
or promote it. The UCI CSV and the saved model remain unchanged.

## Run the demonstration

Start the API in one PowerShell terminal from the repository root:

```powershell
$env:BIKE_DEMAND_MODEL_PATH = "artifacts/run_01/model.joblib"
python -m uvicorn bike_demand.api:create_app --factory --host 127.0.0.1 --port 8000
```

In another terminal, activate the same environment and run:

```powershell
python -m bike_demand.scenarios
Start-Process "artifacts/scenarios/october_day_01/comparison.html"
```

The default run sends 96 requests: the same 24 target hours from October 1,
2012, replayed once for each case. The end date is exclusive.

| Case | Default injection | What to inspect |
| --- | --- | --- |
| `control` | No added fault | Reference coverage and scores |
| `missing_history` | Mask 10% of available observations used in history windows | Missing-input counts, rejection rate, coverage, and model error |
| `invalid_requests` | Replace the latest count with `-1` in 20% of requests | HTTP 422 responses and invalid recent counts |
| `demand_shift` | Multiply counts by 1.5 from the midpoint of the replay | Demand, recent-input distribution, and prediction error |

The missing and invalid injections select a fixed number of observations or
requests, rounded to the nearest whole count, without replacement. The default
seed is 42. At 24 hours, the invalid scenario injects five rejected requests.
Existing missing observations are preserved.

For missing history, a selected clock hour stays missing in all overlapping
requests. Its recorded outcome remains available for evaluation. For demand
shift, both outcomes and observations are synthetic from the change time;
scaled counts are rounded to the nearest integer with ties to even. Missing
outcomes stay missing. A changed outcome enters request history only in later
hours, after that outcome has been revealed.

## Outputs and reproduction

The default parent is `artifacts/scenarios/october_day_01/`. Each case contains
its own replay files and `monitoring/` report. The parent also contains:

- `comparison.html`: coverage, service, data-quality and model-score comparison,
  with links to the individual reports.
- `comparison.csv`: the same comparison in machine-readable form.
- `suite.json`: completion status, dates, model version, seed, and scenario settings.

Each replay's `metadata.json` records its scenario, seed, exact injected hours,
and active parameters. The demand case records its change time and rounding
rule. The source-data checksum still identifies the original UCI CSV.
The scenario label also appears in each HTML report and PNG dashboard.

For another deterministic run, select a new output directory:

```powershell
python -m bike_demand.scenarios --seed 7 --missing-rate 0.05 --invalid-rate 0.1 --demand-multiplier 2 --shift-start 2012-10-01T12:00:00 --output-dir artifacts/scenarios/seed_7
```

`--start`, `--end`, `--data-path`, `--model-dir`, and `--api-url` work like the
replay command. The API must serve the model selected by `--model-dir`, and
replay dates must follow that model's training cutoff. Existing scenario
directories are never overwritten. After an interrupted run, use a new directory.
The same data, model, seed, and parameters reproduce fault placement and
predictions; measured latency can vary.

## Read the comparison carefully

Within each case, the model and weekly baseline are scored on identical eligible
hours. Across cases, rejected requests can change the evaluated set; interpret
MAE alongside coverage and scored-hour counts. This comparison describes fault
effects and does not decide model promotion.

New replay files record `latest_count`. Reports compare available nonnegative
counts from one hour earlier (`lag_1h`) with eligible training inputs, making a
demand change visible within one day. Negative request values are counted as
invalid and excluded from that distribution. Older replay files continue to use
their weekly input (`lag_168h`). Distribution comparisons are descriptive.
