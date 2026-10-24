# December end-to-end demonstration

Run this command from the repository root in the Python 3.12 environment:

```powershell
$env:MLFLOW_DISABLE_TELEMETRY = "true"
python -m bike_demand.demo
Start-Process "artifacts/demos/december_day_01/summary.html"
```

The command performs the complete local rehearsal:

1. Refit the fixed configuration with October observations and compare it
   against the original model on the same November hours.
2. Apply the existing promotion checks and save model snapshots in the demo's
   deployment directory.
3. Start its own Uvicorn subprocess on a free localhost port and verify the
   candidate version through `/health`.
4. Replay the 24 hours of December 1 through HTTP and generate monitoring.
5. Roll back, confirm the running API still has the candidate loaded, stop it,
   and start another process that loads the original version.
6. Verify `/health` and one forecast after rollback, then stop that process.

Each process receives the demo's deployment and request-log paths. A model
override in the calling terminal is cleared for these child processes so it
cannot pin the rehearsal to another artifact. The regular deployment state
and API can be used independently.

All demo files live under `artifacts/demos/december_day_01/`:

| Path | Contents |
| --- | --- |
| `summary.html` | Entry point linking the reports and version checks |
| `manifest.json` | Completion status, Git provenance, API identities, replay/monitoring results |
| `comparison/` | Candidate, November paired predictions, scores, acceptance decision |
| `replay/` | December request records and outcomes |
| `replay/monitoring/` | Standalone HTML report, plots and metric tables |
| `deployment/` | Copied model artifacts and promotion/rollback history |
| `api_requests.jsonl` | Candidate requests followed by the restored-model smoke request |
| `server.log` | Output from the two owned API processes |
| `mlflow.db`, `mlruns/` | Isolated experiment database and MLflow artifacts |

The rehearsal ends with its selected model restored to the original version.
`manifest.json` is marked complete only after the restored model successfully
serves a forecast. On failure, the partial record retains `completed: false`
and an error message. API processes are cleaned up when their context exits.

Existing output directories are refused. For another rehearsal, use:

```powershell
python -m bike_demand.demo --output-dir artifacts/demos/december_day_01_repeat
```

To inspect this demo's MLflow experiment, run from the repository root:

```powershell
python -m mlflow ui --backend-store-uri sqlite:///artifacts/demos/december_day_01/mlflow.db --host 127.0.0.1 --port 5000
```

This is a retrospective engineering demonstration. December was excluded
from this candidate's fitting and November selection, but the dataset has
already been explored in the project. One day is a smoke replay; metrics and
latency do not establish future or production performance.
