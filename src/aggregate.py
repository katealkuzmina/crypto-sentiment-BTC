import pandas as pd

from src.returns import HORIZONS_HOURS, compute_forward_returns


def build_sentiment_index(
    labeled: pd.DataFrame, freq: str = "1h", anchor: str | None = None
) -> pd.DataFrame:
    """labeled: DataFrame with columns published_on (Unix seconds),
    sentiment_score (float, -1..1), confidence (float, 0..1).

    Buckets articles into `freq`-sized time bins (based on published_on,
    interpreted as UTC) and returns a DataFrame indexed by bin start,
    with columns:
      - mean_sentiment: unweighted mean sentiment_score in the bin
      - weighted_sentiment: confidence-weighted mean sentiment_score
      - article_count: number of articles in the bin

    anchor: optional pandas-Timedelta-parseable offset (e.g. "18h") added
    to each floored bin timestamp. Use this when `freq` is coarser than
    how the underlying events cluster in time — e.g. daily bins for
    articles that actually publish in the evening. Without it, a "24h
    forward return" computed from a midnight-floored daily bin is mostly
    a return that already happened by the time a typical article in that
    bin was published.

    Bins with zero articles are NOT included in the output — the caller
    is responsible for reindexing/filling against a complete time grid
    if needed."""
    df = labeled.copy()
    df["timestamp"] = pd.to_datetime(df["published_on"].astype(int), unit="s", utc=True)
    df["bin"] = df["timestamp"].dt.floor(freq)
    if anchor is not None:
        df["bin"] = df["bin"] + pd.Timedelta(anchor)

    def weighted_mean(group: pd.DataFrame) -> float:
        valid = group.dropna(subset=["sentiment_score"])
        weights = valid["confidence"]
        if weights.sum() == 0:
            return valid["sentiment_score"].mean()
        return (valid["sentiment_score"] * weights).sum() / weights.sum()

    grouped = df.groupby("bin")
    result = pd.DataFrame({
        "mean_sentiment": grouped["sentiment_score"].mean(),
        "weighted_sentiment": grouped.apply(weighted_mean, include_groups=False),
        "article_count": grouped.size(),
    })
    return result


def join_sentiment_and_returns(
    sentiment_index: pd.DataFrame,
    price: pd.Series,
    horizons: dict[str, int] = HORIZONS_HOURS,
) -> pd.DataFrame:
    """Joins an hourly sentiment_index (from build_sentiment_index) with
    forward returns computed from `price` (a gapless hourly close-price
    Series). Only hours present in BOTH sentiment_index and the returns
    output are kept (inner join) — hours with no news activity are
    dropped, since there's no sentiment value to correlate against a
    return for them."""
    returns = compute_forward_returns(price, horizons=horizons)
    return sentiment_index.join(returns, how="inner")
