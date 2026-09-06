import csv
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
import requests

from src.fetch_news import (
    fetch_all_weeks,
    fetch_week_articles,
    load_already_fetched_week_ends,
    lts_cutoff,
    week_boundaries,
)


def test_week_boundaries_count_and_coverage():
    weeks = week_boundaries(2024)

    # 2024 is a leap year: 366 days / 7 = 52 full weeks + 1 partial (2 days)
    assert len(weeks) == 53
    assert weeks[0][0] == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert weeks[-1][1] == datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)


def test_week_boundaries_are_contiguous_no_gaps_no_overlap():
    weeks = week_boundaries(2024)
    for (_, prev_end), (next_start, _) in zip(weeks, weeks[1:]):
        assert next_start == prev_end + timedelta(seconds=1)


def test_lts_cutoff_is_one_second_after_week_end():
    week_end = datetime(2024, 1, 7, 23, 59, 59, tzinfo=timezone.utc)
    assert lts_cutoff(week_end) == int(
        datetime(2024, 1, 8, tzinfo=timezone.utc).timestamp()
    )


def _make_response(json_data, status=200):
    response = MagicMock()
    response.status_code = status
    response.json.return_value = json_data
    response.raise_for_status = MagicMock()
    if status >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(f"status {status}")
    return response


def test_fetch_week_articles_returns_data_list():
    session = MagicMock()
    session.get.return_value = _make_response({"Response": "Success", "Data": [{"id": "1"}, {"id": "2"}]})

    week_end = datetime(2024, 1, 7, 23, 59, 59, tzinfo=timezone.utc)
    result = fetch_week_articles(week_end, "fake-key", session)

    assert result == [{"id": "1"}, {"id": "2"}]
    session.get.assert_called_once()


def test_fetch_week_articles_retries_then_succeeds(monkeypatch):
    session = MagicMock()
    session.get.side_effect = [
        requests.ConnectionError("boom"),
        _make_response({"Response": "Success", "Data": [{"id": "1"}]}),
    ]
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    week_end = datetime(2024, 1, 7, 23, 59, 59, tzinfo=timezone.utc)
    result = fetch_week_articles(
        week_end, "fake-key", session, max_retries=2, retry_delay_seconds=0
    )

    assert result == [{"id": "1"}]
    assert session.get.call_count == 2


def test_fetch_week_articles_raises_after_exhausting_retries(monkeypatch):
    session = MagicMock()
    session.get.side_effect = requests.ConnectionError("boom")
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    week_end = datetime(2024, 1, 7, 23, 59, 59, tzinfo=timezone.utc)
    with pytest.raises(requests.ConnectionError):
        fetch_week_articles(
            week_end, "fake-key", session, max_retries=2, retry_delay_seconds=0
        )

    assert session.get.call_count == 2


def test_fetch_week_articles_raises_on_api_error_response_with_200_status(monkeypatch):
    # CryptoCompare/CoinDesk returns HTTP 200 with {"Response": "Error", ...}
    # on rate-limit/param errors. raise_for_status() won't catch this, so the
    # payload shape itself must be validated.
    session = MagicMock()
    session.get.return_value = _make_response({"Response": "Error", "Message": "rate limit"})
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    week_end = datetime(2024, 1, 7, 23, 59, 59, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="rate limit"):
        fetch_week_articles(week_end, "fake-key", session, max_retries=2, retry_delay_seconds=0)

    # Should have retried (integrates with the existing retry loop) not
    # silently returned [] on the first bad response.
    assert session.get.call_count == 2


def test_load_already_fetched_week_ends_empty_when_no_file(tmp_path):
    csv_path = tmp_path / "news_raw.csv"
    assert load_already_fetched_week_ends(csv_path) == set()


def test_fetch_all_weeks_skips_already_fetched_weeks(monkeypatch, tmp_path):
    csv_path = tmp_path / "news_raw.csv"
    call_count = {"n": 0}

    def fake_fetch_week_articles(week_end, api_key, session, **kwargs):
        call_count["n"] += 1
        return [{
            "id": str(call_count["n"]),
            "published_on": 1,
            "title": "t",
            "body": "b",
            "url": "u",
            "source": "s",
            "tags": "tag",
            "categories": "BTC",
        }]

    monkeypatch.setattr("src.fetch_news.fetch_week_articles", fake_fetch_week_articles)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    fake_weeks = [
        (datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 1, 7, 23, 59, 59, tzinfo=timezone.utc)),
        (datetime(2024, 1, 8, tzinfo=timezone.utc), datetime(2024, 1, 14, 23, 59, 59, tzinfo=timezone.utc)),
    ]
    monkeypatch.setattr("src.fetch_news.week_boundaries", lambda year: fake_weeks)

    fetch_all_weeks(year=2024, api_key="fake-key", csv_path=csv_path)
    assert call_count["n"] == 2

    fetch_all_weeks(year=2024, api_key="fake-key", csv_path=csv_path)
    assert call_count["n"] == 2  # resumed run: nothing new fetched

    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2


def test_fetch_week_articles_locks_http_request_shape():
    session = MagicMock()
    session.get.return_value = _make_response({"Response": "Success", "Data": [{"id": "1"}]})

    week_end = datetime(2024, 1, 7, 23, 59, 59, tzinfo=timezone.utc)
    fetch_week_articles(week_end, "test-api-key", session)

    # Verify the exact HTTP request shape
    call_args, call_kwargs = session.get.call_args
    assert call_args == ("https://min-api.cryptocompare.com/data/v2/news/",)
    assert call_kwargs["params"]["lang"] == "EN"
    assert call_kwargs["params"]["categories"] == "BTC"
    assert call_kwargs["params"]["lTs"] == int(
        datetime(2024, 1, 8, tzinfo=timezone.utc).timestamp()
    )
    assert call_kwargs["headers"]["authorization"] == "Apikey test-api-key"
    assert call_kwargs["timeout"] == 15


def test_fetch_all_weeks_records_empty_week_as_fetched(monkeypatch, tmp_path):
    csv_path = tmp_path / "news_raw.csv"
    call_count = {"n": 0}

    def fake_fetch_week_articles(week_end, api_key, session, **kwargs):
        call_count["n"] += 1
        # First week returns empty (low-news week), second returns one article
        if call_count["n"] == 1:
            return []
        return [{
            "id": "2",
            "published_on": 2,
            "title": "t2",
            "body": "b2",
            "url": "u2",
            "source": "s2",
            "tags": "tag2",
            "categories": "BTC",
        }]

    monkeypatch.setattr("src.fetch_news.fetch_week_articles", fake_fetch_week_articles)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    fake_weeks = [
        (datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 1, 7, 23, 59, 59, tzinfo=timezone.utc)),
        (datetime(2024, 1, 8, tzinfo=timezone.utc), datetime(2024, 1, 14, 23, 59, 59, tzinfo=timezone.utc)),
    ]
    monkeypatch.setattr("src.fetch_news.week_boundaries", lambda year: fake_weeks)

    # First run: fetch both weeks (first returns 0 articles, second returns 1)
    fetch_all_weeks(year=2024, api_key="fake-key", csv_path=csv_path)
    assert call_count["n"] == 2

    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # Should have 2 rows: sentinel for empty week, and 1 article from second week
    assert len(rows) == 2

    # Second run: should skip both weeks (even though first had 0 articles)
    fetch_all_weeks(year=2024, api_key="fake-key", csv_path=csv_path)
    assert call_count["n"] == 2  # resumed run: no new calls (empty week is tracked)


def test_fetch_all_weeks_filters_articles_outside_week_bounds(monkeypatch, tmp_path):
    csv_path = tmp_path / "news_raw.csv"

    week_start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    week_end = datetime(2024, 1, 7, 23, 59, 59, tzinfo=timezone.utc)

    in_bounds_ts = int((week_start + timedelta(days=2)).timestamp())
    # Leaked article from a prior week/year (no lower bound on the API's lTs param)
    out_of_bounds_ts = int((week_start - timedelta(days=10)).timestamp())

    def fake_fetch_week_articles(week_end, api_key, session, **kwargs):
        return [
            {
                "id": "in-bounds",
                "published_on": in_bounds_ts,
                "title": "t1", "body": "b1", "url": "u1",
                "source": "s1", "tags": "tag1", "categories": "BTC",
            },
            {
                "id": "out-of-bounds",
                "published_on": out_of_bounds_ts,
                "title": "t2", "body": "b2", "url": "u2",
                "source": "s2", "tags": "tag2", "categories": "BTC",
            },
        ]

    monkeypatch.setattr("src.fetch_news.fetch_week_articles", fake_fetch_week_articles)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    monkeypatch.setattr("src.fetch_news.week_boundaries", lambda year: [(week_start, week_end)])

    fetch_all_weeks(year=2024, api_key="fake-key", csv_path=csv_path)

    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    ids = {row["id"] for row in rows}
    assert "in-bounds" in ids
    assert "out-of-bounds" not in ids
