import pandas as pd
import pytest

from src.aggregate import build_sentiment_index, join_sentiment_and_returns


def test_build_sentiment_index_buckets_by_hour():
    labeled = pd.DataFrame({
        "published_on": [1704067200, 1704067800, 1704070800],  # 2024-01-01 00:00, 00:10, 01:00 UTC
        "sentiment_score": [0.5, -0.5, 0.8],
        "confidence": [1.0, 1.0, 1.0],
    })

    index = build_sentiment_index(labeled)

    assert len(index) == 2
    first_hour = index.iloc[0]
    assert first_hour["article_count"] == 2
    assert first_hour["mean_sentiment"] == pytest.approx(0.0)


def test_build_sentiment_index_weighted_mean_respects_confidence():
    labeled = pd.DataFrame({
        "published_on": [1704067200, 1704067800],
        "sentiment_score": [1.0, -1.0],
        "confidence": [0.9, 0.1],
    })

    index = build_sentiment_index(labeled)

    assert index.iloc[0]["weighted_sentiment"] == pytest.approx((1.0 * 0.9 + -1.0 * 0.1) / 1.0)


def test_join_sentiment_and_returns_inner_join_drops_newsless_hours():
    idx = pd.date_range("2024-01-01", periods=5, freq="h")
    price = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=idx)

    sentiment_index = pd.DataFrame(
        {"mean_sentiment": [0.5], "weighted_sentiment": [0.5], "article_count": [1]},
        index=[idx[0]],
    )

    joined = join_sentiment_and_returns(sentiment_index, price, horizons={"1h": 1})

    assert len(joined) == 1
    assert joined.iloc[0]["1h"] == pytest.approx((101.0 - 100.0) / 100.0)
