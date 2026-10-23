"""Check split boundaries before relying on holdout scores."""

import pandas as pd
import pytest

from bike_demand.train import split_data


def sample_data() -> pd.DataFrame:
    index = pd.date_range("2012-06-30", periods=72, freq="h")
    return pd.DataFrame({"target": range(72)}, index=index)


def test_target_hours_stay_in_their_periods():
    validation_start = pd.Timestamp("2012-07-01")
    test_start = pd.Timestamp("2012-07-02")
    splits = split_data(sample_data(), validation_start, test_start)

    assert splits["train"].index.max() < validation_start
    assert splits["validation"].index.min() == validation_start
    assert splits["validation"].index.max() < test_start
    assert splits["test"].index.min() == test_start
    assert sum(len(part) for part in splits.values()) == 72


def test_reversed_dates_are_rejected():
    with pytest.raises(ValueError, match="Validation must start before"):
        split_data(
            sample_data(), pd.Timestamp("2012-07-02"), pd.Timestamp("2012-07-01")
        )


def test_empty_period_is_rejected():
    with pytest.raises(ValueError, match="Each split"):
        split_data(
            sample_data(), pd.Timestamp("2013-01-01"), pd.Timestamp("2013-02-01")
        )
