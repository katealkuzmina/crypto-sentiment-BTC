import numpy as np
import pandas as pd
import pytest

from src.returns import compute_forward_returns


def test_compute_forward_returns_basic():
    idx = pd.date_range("2024-01-01", periods=30, freq="h")
    price = pd.Series(np.arange(100, 130, dtype=float), index=idx)

    result = compute_forward_returns(price, horizons={"1h": 1, "5h": 5})

    assert result["1h"].iloc[0] == pytest.approx((101 - 100) / 100)
    assert result["5h"].iloc[0] == pytest.approx((105 - 100) / 100)


def test_compute_forward_returns_nan_near_tail():
    idx = pd.date_range("2024-01-01", periods=10, freq="h")
    price = pd.Series(np.arange(10, dtype=float), index=idx)

    result = compute_forward_returns(price, horizons={"5h": 5})

    assert result["5h"].iloc[-5:].isna().all()
    assert result["5h"].iloc[:-5].notna().all()


def test_compute_forward_returns_rejects_gappy_index():
    idx = pd.date_range("2024-01-01", periods=10, freq="h").delete(3)
    price = pd.Series(np.arange(9, dtype=float), index=idx)

    with pytest.raises(ValueError, match="gapless"):
        compute_forward_returns(price, horizons={"1h": 1})
