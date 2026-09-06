import csv
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from src.config import get_env

NEWS_API_URL = "https://min-api.cryptocompare.com/data/v2/news/"
CSV_FIELDS = ["id", "published_on", "title", "body", "url", "source", "tags", "categories"]


def week_boundaries(year: int) -> list[tuple[datetime, datetime]]:
    """Non-overlapping, contiguous 7-day (week_start, week_end) UTC blocks
    covering the full calendar year, starting Jan 1 00:00:00 UTC. The
    final block may be shorter than 7 days; its week_end is always
    Dec 31 23:59:59 UTC of that year."""
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    year_end = datetime(year + 1, 1, 1, tzinfo=timezone.utc) - timedelta(seconds=1)

    weeks = []
    cursor = start
    while cursor <= year_end:
        week_end = min(cursor + timedelta(days=7) - timedelta(seconds=1), year_end)
        weeks.append((cursor, week_end))
        cursor = week_end + timedelta(seconds=1)
    return weeks


def lts_cutoff(week_end: datetime) -> int:
    """The CryptoCompare/CoinDesk News API's `lTs` parameter (Unix
    seconds) to request articles strictly before. Anchored one second
    after week_end, so an article published in the final second of the
    week is still included."""
    return int((week_end + timedelta(seconds=1)).timestamp())


def fetch_week_articles(
    week_end: datetime,
    api_key: str,
    session: requests.Session,
    categories: str = "BTC",
    max_retries: int = 3,
    retry_delay_seconds: float = 2.0,
) -> list[dict]:
    """Fetches up to 50 articles published before the given week's end
    boundary, for the given category. Retries transient failures
    (network errors, bad status codes) up to max_retries times with a
    fixed delay; raises the last exception if all retries are
    exhausted."""
    params = {"lang": "EN", "categories": categories, "lTs": lts_cutoff(week_end)}
    headers = {"authorization": f"Apikey {api_key}"}

    last_error = None
    for attempt in range(max_retries):
        try:
            response = session.get(NEWS_API_URL, params=params, headers=headers, timeout=15)
            response.raise_for_status()
            payload = response.json()
            data = payload.get("Data")
            if not isinstance(data, list):
                raise ValueError(f"API error response: {payload.get('Message', payload)!r}")
            return data
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(retry_delay_seconds)
    raise last_error


def load_already_fetched_week_ends(csv_path: Path) -> set[str]:
    """Reads the progress-tracking column of an existing output CSV (if
    any) to find which week_end values have already been fetched, so a
    resumed run can skip them."""
    if not csv_path.exists():
        return set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["_source_week_end"] for row in reader}


def fetch_all_weeks(
    year: int,
    api_key: str,
    csv_path: Path,
    session: requests.Session | None = None,
    sleep_between_requests: float = 2.0,
) -> None:
    """Fetches all weekly strata for `year` not already present in
    `csv_path`, appending each week's articles as soon as they're
    fetched (not batched at the end) so an interrupted run doesn't lose
    already-spent API quota."""
    session = session or requests.Session()
    already_fetched = load_already_fetched_week_ends(csv_path)
    file_exists = csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS + ["_source_week_end"])
        if not file_exists:
            writer.writeheader()

        for week_start, week_end in week_boundaries(year):
            week_end_key = week_end.isoformat()
            if week_end_key in already_fetched:
                continue

            articles = fetch_week_articles(week_end, api_key, session)
            week_start_ts = week_start.timestamp()
            week_end_ts = week_end.timestamp()
            articles = [
                article for article in articles
                if week_start_ts <= (article.get("published_on") or 0) <= week_end_ts
            ]
            if articles:
                for article in articles:
                    writer.writerow({
                        "id": article.get("id"),
                        "published_on": article.get("published_on"),
                        "title": article.get("title"),
                        "body": article.get("body"),
                        "url": article.get("url"),
                        "source": article.get("source"),
                        "tags": article.get("tags"),
                        "categories": article.get("categories"),
                        "_source_week_end": week_end_key,
                    })
            else:
                # Write sentinel row for empty weeks so they're tracked as fetched
                writer.writerow({"_source_week_end": week_end_key})
            f.flush()
            time.sleep(sleep_between_requests)


if __name__ == "__main__":
    fetch_all_weeks(
        year=2024,
        api_key=get_env("CRYPTOCOMPARE_API_KEY"),
        csv_path=Path("data/news_raw.csv"),
    )
