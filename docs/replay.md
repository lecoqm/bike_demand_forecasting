# Chronological API replay

This is a retrospective simulation using the UCI dataset. It sends requests
in historical order, as quickly as the API responds. Wall-clock latency is
measured on your machine; these are not requests from real users.

## First run

From the repository root, start the API in one PowerShell terminal:

```powershell
$env:BIKE_DEMAND_MODEL_PATH = "artifacts/run_01/model.joblib"
python -m uvicorn bike_demand.api:create_app --factory --host 127.0.0.1 --port 8000
```

In a second terminal, activate the same environment and run:

```powershell
python -m bike_demand.replay
```

The defaults replay the 24 hours of October 1, 2012. The model in `run_01`
was last trained on September 30, so the replay follows its training period.
The end date is exclusive. Each request contains the preceding 168 hourly
counts, oldest first, with missing observations retained as `null`. The current
outcome is read after the request and becomes available to later requests.
The model remains fixed throughout the replay.

The command checks that the API model's SHA-256 matches `--model-dir` and that
replay begins after the last training target recorded in that directory's
`metadata.json`. To use an MLflow run, set the API model path to its
`model.joblib` and pass that same run directory as `--model-dir`.

## Outputs

By default, outputs are saved in `artifacts/replays/october_day_01/`:

- `predictions.csv`: one row per requested clock hour, including rejected
  requests, transport failures, and hours without recorded outcomes. Successful
  rows include the prediction; the weekly baseline uses the count 168 hours
  earlier. Missing values stay empty in the CSV.
- `metadata.json`: replay dates, data checksum, model version, training cutoff,
  completion status, and counts of predictions, rejections, failures, missing
  outcomes, and predictions with known outcomes.

`request_id` links each HTTP response to `artifacts/api_requests.jsonl`.
`client_latency_ms` measures the HTTP round trip, including server processing;
it differs from the API log's server-side `latency_ms`. A transport failure can
lack a request ID even if the server received the request.

There are no retries. CSV rows are flushed after every request; an interrupted
run leaves `completed: false`. Existing output directories are never overwritten.
Choose another `--output-dir` for each new replay.

After checking the one-day replay, you can replay the rest of October:

```powershell
python -m bike_demand.replay --start 2012-10-02 --end 2012-11-01 --output-dir artifacts/replays/october_rest
```

`--api-url` and `--timeout` configure the API address and the HTTP timeout in
seconds (default 30). HTTP failures are recorded in the CSV and summarized in
the terminal. A completed replay means every hour was attempted; inspect its
failure and rejection counts. Performance metrics and plots are a separate step.
