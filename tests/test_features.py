"""Check the temporal assumptions that training and serving must share."""

import numpy as np
import pandas as pd
import pytest

from bike_demand.features import make_features


@pytest.fixture
def history() -> pd.DataFrame:
    index = pd.date_range("2012-01-01", periods=400, freq="h", name="timestamp")
    return pd.DataFrame({"cnt": np.arange(400, dtype=float)}, index=index)


def test_future_counts_do_not_change_current_features(history):
    cutoff = history.index[300]
    changed = history.copy()
    changed.loc[cutoff:, "cnt"] = 99999.0

    original_features = make_features(history)
    changed_features = make_features(changed)

    pd.testing.assert_frame_equal(
        original_features.loc[:cutoff], changed_features.loc[:cutoff]
    )
    assert changed_features.loc[cutoff + pd.Timedelta(hours=1), "lag_1h"] == 99999


def test_missing_count_keeps_its_clock_hour(history):
    missing_hour = history.index[200]
    history.loc[missing_hour, "cnt"] = np.nan
    features = make_features(history)

    assert pd.isna(features.loc[missing_hour + pd.Timedelta(hours=1), "lag_1h"])
    assert pd.isna(features.loc[missing_hour + pd.Timedelta(hours=24), "lag_24h"])
    assert features.loc[missing_hour + pd.Timedelta(hours=2), "lag_1h"] == 201


def test_irregular_hourly_grid_is_rejected(history):
    with pytest.raises(ValueError, match="complete hourly grid"):
        make_features(history.drop(history.index[200]))


def test_history_snapshot_matches_batch_features(history):
    target_hour = history.index[300]
    available = history.loc[history.index < target_hour].copy()
    available = available.reindex(history.index[:301])

    pd.testing.assert_series_equal(
        make_features(available).loc[target_hour],
        make_features(history).loc[target_hour],
    )
