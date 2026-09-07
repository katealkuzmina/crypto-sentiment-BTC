import pandas as pd

HORIZONS_HOURS = {"1h": 1, "24h": 24, "7d": 24 * 7, "14d": 24 * 14}


def _validate_gapless_hourly_index(price: pd.Series) -> None:
    expected_index = pd.date_range(price.index[0], price.index[-1], freq="h")
    if not price.index.equals(expected_index):
        raise ValueError(
            "price index must be a complete, gapless hourly range — "
            "found gaps or duplicate/out-of-order timestamps"
        )


def compute_forward_returns(
    price: pd.Series, horizons: dict[str, int] = HORIZONS_HOURS
) -> pd.DataFrame:
    """price: pandas Series of close prices, indexed by an hourly UTC
    DatetimeIndex with no gaps (exactly one row per hour, ascending).
    Raises ValueError if the index isn't a complete, gapless hourly range —
    a horizon like "24h" is implemented as a 24-row shift, which is only
    equivalent to a 24-clock-hour shift when there are no missing hours.

    Returns a DataFrame aligned to `price`'s index, one column per horizon
    key: price[t + horizon] / price[t] - 1. NaN near the end of the series,
    where t + horizon falls past the last available price.
    """
    _validate_gapless_hourly_index(price)
    out = {}
    for label, hours in horizons.items():
        shifted = price.shift(-hours)
        out[label] = shifted / price - 1
    return pd.DataFrame(out, index=price.index)


def compute_trailing_returns(
    price: pd.Series, horizons: dict[str, int] = HORIZONS_HOURS
) -> pd.DataFrame:
    """Same index contract as compute_forward_returns.

    Returns a DataFrame aligned to `price`'s index, one column per horizon
    key: price[t] / price[t - horizon] - 1 — the return over the window
    ENDING at t, entirely in the past relative to t. NaN near the start of
    the series, where t - horizon falls before the first available price.

    Use this (not compute_forward_returns) whenever price is the leading
    variable and something else is the response — e.g. testing whether a
    past price move predicts today's sentiment. A forward return at lag 1
    is not "the past": when the row spacing equals the horizon (as it does
    for 7d/14d on a weekly-sampled daily series), shifting a forward return
    by one row lands almost exactly on the trailing return ending at the
    current row — a contemporaneous quantity dressed up as a lagged one.
    """
    _validate_gapless_hourly_index(price)
    out = {}
    for label, hours in horizons.items():
        shifted = price.shift(hours)
        out[label] = price / shifted - 1
    return pd.DataFrame(out, index=price.index)
