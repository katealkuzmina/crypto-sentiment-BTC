import pandas as pd

HORIZONS_HOURS = {"1h": 1, "24h": 24, "7d": 24 * 7, "14d": 24 * 14}


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
    expected_index = pd.date_range(price.index[0], price.index[-1], freq="h")
    if not price.index.equals(expected_index):
        raise ValueError(
            "price index must be a complete, gapless hourly range — "
            "found gaps or duplicate/out-of-order timestamps"
        )

    out = {}
    for label, hours in horizons.items():
        shifted = price.shift(-hours)
        out[label] = shifted / price - 1
    return pd.DataFrame(out, index=price.index)
