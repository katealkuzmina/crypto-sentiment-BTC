import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

KLINES_URL = "https://api.binance.com/api/v3/klines"
MAX_KLINES_PER_REQUEST = 1000
HOUR_MS = 60 * 60 * 1000


def fetch_klines_range(
    symbol: str,
    start_ms: int,
    end_ms: int,
    session: requests.Session | None = None,
    sleep_between_requests: float = 0.5,
) -> pd.DataFrame:
    """Fetches hourly klines for `symbol` between start_ms and end_ms
    (inclusive, Unix milliseconds), paginating in chunks of
    MAX_KLINES_PER_REQUEST since Binance caps each request. Returns a
    DataFrame with columns open, high, low, close, volume, indexed by a
    UTC DatetimeIndex built from open_time, deduplicated and sorted
    ascending."""
    session = session or requests.Session()
    rows = []
    cursor = start_ms

    while cursor <= end_ms:
        params = {
            "symbol": symbol,
            "interval": "1h",
            "startTime": cursor,
            "endTime": end_ms,
            "limit": MAX_KLINES_PER_REQUEST,
        }
        response = session.get(KLINES_URL, params=params, timeout=15)
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break

        rows.extend(batch)
        last_open_time = batch[-1][0]
        next_cursor = last_open_time + HOUR_MS
        if next_cursor <= cursor:
            break  # safety net against an infinite loop on unexpected data

        if len(batch) < MAX_KLINES_PER_REQUEST:
            break  # short batch means we've reached the end of available data

        cursor = next_cursor
        time.sleep(sleep_between_requests)

    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])
    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="open_time").sort_values("open_time").set_index("open_time")
    return df


if __name__ == "__main__":
    start = datetime(2024, 1, 1, tzinfo=timezone.utc) - timedelta(days=14)
    end = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=14)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    df = fetch_klines_range("BTCUSDT", start_ms, end_ms)
    output_path = Path("data/btc_price_hourly.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
