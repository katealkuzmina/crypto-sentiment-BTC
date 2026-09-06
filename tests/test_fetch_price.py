from unittest.mock import MagicMock

import pytest

from src.fetch_price import HOUR_MS, MAX_KLINES_PER_REQUEST, fetch_klines_range


def _kline_row(open_time_ms, close_price):
    # Binance kline row shape: [open_time, open, high, low, close, volume,
    # close_time, quote_volume, trades, taker_buy_base, taker_buy_quote, ignore]
    return [
        open_time_ms, "100.0", "101.0", "99.0", str(close_price), "10.0",
        open_time_ms + HOUR_MS - 1, "1000.0", 5, "5.0", "500.0", "0",
    ]


def _make_response(rows):
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = rows
    return response


def test_fetch_klines_range_single_batch():
    session = MagicMock()
    start_ms = 0
    rows = [_kline_row(start_ms + i * HOUR_MS, 100 + i) for i in range(3)]
    session.get.return_value = _make_response(rows)

    df = fetch_klines_range("BTCUSDT", start_ms, start_ms + 2 * HOUR_MS, session=session)

    assert len(df) == 3
    assert list(df["close"]) == [100.0, 101.0, 102.0]
    assert session.get.call_count == 1


def test_fetch_klines_range_paginates_across_chunks(monkeypatch):
    session = MagicMock()
    start_ms = 0

    first_batch = [_kline_row(start_ms + i * HOUR_MS, 100 + i) for i in range(MAX_KLINES_PER_REQUEST)]
    second_start = start_ms + MAX_KLINES_PER_REQUEST * HOUR_MS
    second_batch = [_kline_row(second_start + i * HOUR_MS, 200 + i) for i in range(5)]

    session.get.side_effect = [_make_response(first_batch), _make_response(second_batch)]
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    end_ms = second_start + 4 * HOUR_MS
    df = fetch_klines_range("BTCUSDT", start_ms, end_ms, session=session)

    assert session.get.call_count == 2
    assert len(df) == MAX_KLINES_PER_REQUEST + 5
    assert df.index.is_monotonic_increasing
    assert not df.index.duplicated().any()


def test_fetch_klines_range_empty_result():
    session = MagicMock()
    session.get.return_value = _make_response([])

    df = fetch_klines_range("BTCUSDT", 0, HOUR_MS, session=session)

    assert len(df) == 0
