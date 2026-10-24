# Promote a candidate and roll back

This step adds explicit local model selection. `promote` selects an accepted
candidate for the next API startup. `rollback` restores its predecessor.
`status` describes the next startup; the running API's `/health` endpoint
reports the version actually serving requests.

## Check and commit

From the repository root in the project environment:

```powershell
$env:MLFLOW_DISABLE_TELEMETRY = "true"
python -m ruff check bike_demand tests
python -m pytest
```

Commit the code after the checks pass. The existing comparison under
`artifacts/comparisons/october_update/` is the input to promotion. Training
does not need to run again. Neither command needs an MLflow server.

## Promote

The earlier replay commands pinned the API to the original model with
`BIKE_DEMAND_MODEL_PATH`. Clear that setting in the terminal used to start
the API so it follows managed selection. Environment changes affect only
that terminal and processes started from it.

Stop the running API with Ctrl+C, then run in that terminal:

```powershell
$env:MLFLOW_DISABLE_TELEMETRY = "true"
Remove-Item Env:BIKE_DEMAND_MODEL_PATH -ErrorAction SilentlyContinue
python -m bike_demand.deploy status
python -m bike_demand.deploy promote
python -m bike_demand.deploy status
python -m uvicorn bike_demand.api:create_app --factory --host 127.0.0.1 --port 8000
```

From a second terminal, verify readiness and the loaded model version:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Its `model_version` must equal `selected_model_version` from the managed
status output and the candidate version in `comparison.json`. Every
prediction response and request log retains that exact model version.
The API loads once at startup, so changing selection while it is running
takes effect after a restart. Use the usual single-process command above
for this local demonstration.

Promotion checks all of the following:

- The comparison completed and records the fixed acceptance rule.
- Recomputing the rule gives lower candidate MAE and no increase in RMSE.
- Scores agree with finite predictions on the shared eligible CSV rows,
  and the CSV covers each scheduled hour exactly once.
- The current model in the comparison is still the managed active version,
  or the original model before the first promotion.
- Both selected artifacts have their recorded SHA-256 checksums. The
  candidate also matches the serving features and produces a finite,
  nonnegative smoke prediction.

A comparison against an older active model cannot promote another candidate.
Run a new comparison with the current model as its baseline when preparing
a subsequent update. Existing model directories are preserved.

## Roll back

Stop the API with Ctrl+C. In the API terminal, restore the predecessor and
restart:

```powershell
python -m bike_demand.deploy rollback
python -m bike_demand.deploy status
python -m uvicorn bike_demand.api:create_app --factory --host 127.0.0.1 --port 8000
```

Check `/health` again from the second terminal. After the first candidate
update, rollback restores the original model version. Each rollback consumes
the latest predecessor. With several promotions, further rollbacks walk back
through those predecessors; with none left, the command fails without
changing state.

## Saved state and overrides

`artifacts/deployment/state.json` records the active model, predecessor stack,
and promotion/rollback history. Each event includes a unique ID, UTC time,
action and source/destination model versions. Promotion events also record
the candidate MLflow run ID and the comparison report path/checksum.

`artifacts/deployment/models/<sha256>/model.joblib` contains byte-identical
copies used for serving and rollback. Keep the comparison directory for its
full evaluation evidence. The model snapshots allow serving and rollback
even if the original training files are later unavailable.

State and its history are replaced together after snapshots are ready. Failed
checks or a failed state replacement preserve the previously saved selection.
A damaged or missing selected snapshot prevents API startup visibly.

The API chooses its model in this order:

1. An explicit `create_app(model_path=...)` argument.
2. `BIKE_DEMAND_MODEL_PATH`, when set.
3. The active deployment snapshot, when a state file exists.
4. The original `artifacts/run_01/model.joblib` before the first promotion.

Use `BIKE_DEMAND_DEPLOYMENT_PATH` to configure another state file for both
the API and CLI. CLI operations also accept `--state-path`; promotion accepts
`--comparison-dir` and `--initial-model-path`. Run commands from the repository
root, matching the paths recorded by the comparison command.

One exclusive local lock prevents two changes from replacing state at once.
Normal errors release it. If a process is forcibly terminated, confirm no
promotion or rollback is running before manually removing the corresponding
`state.lock` and retrying. This mechanism is intended for the local filesystem
demonstration, not coordinated deployment across machines.
