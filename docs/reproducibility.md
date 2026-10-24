# Reproducibility checks

The repository includes the hourly UCI CSV, a fitted original model, its
metadata, and saved test predictions. These let tests check numerical
agreement independently of another training run. The loader accepts the
original UCI file or the same CSV with Git-normalized line endings, verifying
their known SHA-256 checksums.

## Fresh-clone procedure

After pushing the final changes, clone into a new directory and create a new
Python 3.12 environment. Install only `requirements.txt`, then run:

```powershell
$env:MLFLOW_DISABLE_TELEMETRY = "true"
python -m pip check
python -m ruff check bike_demand tests
python -m pytest
python -m bike_demand.demo
```

On a POSIX shell, use `export MLFLOW_DISABLE_TELEMETRY=true`. The README has
the complete environment setup for each platform.

The test suite includes an integration test that invokes the demonstration
in a child process. It trains the candidate, starts real Uvicorn processes,
replays December through HTTP, checks monitoring metrics independently,
verifies request IDs and model versions, and tests rollback after restart.
Temporary test data lives in pytest's test directories. The standalone demo
creates its evidence under `artifacts/demos/december_day_01/`.

Review `manifest.json` and `deployment/state.json` there. A completed rehearsal
has 24 December requests, a candidate served before restart, an original model
served after rollback/restart, and promotion followed by rollback in history.
Both owned API processes stop when the rehearsal completes.

## GitHub Actions

Push the commits and open [the Tests workflow](https://github.com/lecoqm/bike_demand_forecasting/actions/workflows/tests.yml).
Select the run for the exact commit you pushed. Both jobs must finish with
success:

- `Python 3.12 (ubuntu-latest)`
- `Python 3.12 (windows-latest)`

Each job starts from checkout, installs requirements, runs `pip check`, runs
Ruff, and executes the complete pytest suite including the HTTP rehearsal.
MLflow usage tracking is disabled in the workflow environment. Hosted Windows
results verify Windows behavior; a local Linux run alone cannot establish it.

## What is reproducible

Chronological boundaries, feature definitions, model configuration, random
seeds, paired scoring hours, fault scenarios, and the acceptance rule are
recorded explicitly. MLflow and local artifacts retain data/model hashes,
Git provenance, parameters, metrics, and evaluation outputs.

Top-level dependencies are pinned rather than every transitive dependency.
Package metadata, native libraries, hardware, run IDs, timestamps, localhost
ports and latency may differ. Serialized artifact hashes may also differ
between environments. Tests verify predictions within stated numerical
tolerances instead of requiring identical serialized files.

Use a committed source revision for a recorded experiment. A dirty working
tree is reported in provenance; it is not silently presented as a clean run.
The dataset is retrospective and already explored, so reproducible scores do
not establish performance on an unseen future production period.
