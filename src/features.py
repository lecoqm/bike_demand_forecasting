"""Calendar and demand features available before each target hour."""

import pandas as pd


def make_features(history: pd.DataFrame) -> pd.DataFrame:
    """Build features for hour h using counts strictly before h.

    Input must be a complete hourly grid. Unrecorded counts stay missing.
    """
    index = history.index
    if (
        history.empty
        or not isinstance(index, pd.DatetimeIndex)
        or not index.is_unique
        or not index.is_monotonic_increasing
    ):
        raise ValueError("History must have a nonempty, sorted, unique DatetimeIndex.")
    calendar = pd.date_range(index.min(), index.max(), freq="h", name=index.name)
    if not index.equals(calendar):
        raise ValueError(
            "Reindex history to a complete hourly grid before making lags."
        )

    features = pd.DataFrame(index=index)
    features["hour"] = index.hour
    features["day_of_week"] = index.dayofweek
    features["is_weekend"] = (index.dayofweek >= 5).astype(int)
    features["month"] = index.month
    features["year"] = index.year
    features["day_of_year"] = index.dayofyear

    demand = history["cnt"].astype(float)
    for lag in [1, 2, 3, 24, 48, 168]:
        features[f"lag_{lag}h"] = demand.shift(lag)

    # Shift first so the target hour cannot enter its own rolling statistics.
    past = demand.shift(1)
    for window in [3, 24, 168]:
        features[f"rolling_mean_{window}h"] = past.rolling(
            window, min_periods=window
        ).mean()
    features["rolling_std_24h"] = past.rolling(24, min_periods=24).std()
    features["observations_last_24h"] = past.rolling(24, min_periods=1).count()
    return features
