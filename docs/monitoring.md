# Replay monitoring report

Generate a report from the saved replay. The API and MLflow server can be stopped.
Run these commands from the repository root in your virtual environment:

```powershell
python -m pip install -r requirements.txt
python -m bike_demand.monitor
Start-Process artifacts/replays/october_day_01/monitoring/report.html
```

The report contains model and weekly-baseline scores, service statistics,
coverage, missing-data counts, and four plots: demand and forecasts, MAE by
hour, client latency, and a demand-input distribution comparison.

## Outputs

The default directory is `artifacts/replays/october_day_01/monitoring/`:

- `report.html`: a standalone HTML report with its plot embedded.
- `monitoring.png`: the four plots, ready to reuse in a project presentation.
- `summary.json`: metrics, counts, model version, replay dates, and source hashes.
- `metrics.csv`: model and baseline MAE and RMSE, with their shared sample count.
- `by_hour.csv`: MAE and sample counts for each target hour of the day.
- `input_distribution.csv`: count, mean, median, and P95 of the selected
  demand input in training and replay (`input_feature` in `summary.json`).

The command reads `predictions.csv` and `metadata.json` from the replay directory.
It also reads the existing UCI CSV at `data/hour.csv` to reconstruct the selected
lag input for eligible training targets through the recorded training cutoff. It
does not download data. Use `--data-path` if the CSV is stored elsewhere.

For another replay:

```powershell
python -m bike_demand.monitor --replay-dir artifacts/replays/october_rest
```

`--output-dir` overrides the destination. Re-running the command refreshes the
generated report files; the replay files remain unchanged.

## Metric definitions

- **Prediction coverage:** valid predictions divided by all scheduled hours.
  An interrupted replay keeps unattempted hours in this denominator.
- **Error rate:** rejected or failed attempts divided by attempted hours.
  Transport errors and invalid HTTP 200 responses count as failures.
- **MAE and RMSE:** both models use exactly the same hours with a valid API
  prediction, a known outcome, and a known weekly baseline. Other rows still
  contribute to the relevant coverage, error, latency, and missing-data counts.
- **Client latency:** HTTP round-trip time for all attempts, including failures.
  Median and P95 use the available timings with linear percentile interpolation.
  These values differ from the server processing times in API logs.
- **Missing history:** counts remain missing rather than becoming zero. The
  percentage uses attempts whose missing-history count is known.
- **Input comparison:** new replay files record `latest_count`, so the report
  compares the count from one hour earlier (`lag_1h`) with that input in training.
  Older files without this column use the weekly count (`lag_168h`). Negative
  recent counts are reported as invalid and excluded from the distribution.
  Calendar composition can explain changes; this plot is not an automatic drift alert.

Injected scenarios and their settings appear in the report. Demand-shift outcomes
are synthetic after the recorded change time; see [scenarios.md](scenarios.md).

Undefined values, including metrics without eligible outcomes and percentage
improvement over a zero-error baseline, are `null` in JSON and `n/a` in HTML.
A completed replay must contain every scheduled hour; partial replays are
labelled explicitly. One day demonstrates the mechanics; longer replays are
needed to examine weekday patterns and changes over time.
