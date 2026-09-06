import csv
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.label_sentiment import (
    label_all_articles,
    label_article,
    load_already_labeled_ids,
)


def _make_tool_use_message(input_dict):
    tool_use_block = SimpleNamespace(type="tool_use", input=input_dict)
    return SimpleNamespace(content=[tool_use_block])


def test_label_article_returns_valid_result():
    client = MagicMock()
    client.messages.create.return_value = _make_tool_use_message({
        "sentiment_category": "positive",
        "confidence": 0.9,
        "sentiment_score": 0.7,
    })

    result = label_article(client, "Bitcoin surges", "Bitcoin price rose sharply today.")

    assert result == {
        "sentiment_category": "positive",
        "confidence": 0.9,
        "sentiment_score": 0.7,
    }


def test_label_article_raises_on_missing_tool_use():
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(content=[])

    with pytest.raises(ValueError, match="tool_use"):
        label_article(client, "title", "body")


def test_label_article_raises_on_invalid_category():
    client = MagicMock()
    client.messages.create.return_value = _make_tool_use_message({
        "sentiment_category": "bullish",
        "confidence": 0.9,
        "sentiment_score": 0.7,
    })

    with pytest.raises(ValueError, match="sentiment_category"):
        label_article(client, "title", "body")


def test_label_article_raises_on_out_of_range_score():
    client = MagicMock()
    client.messages.create.return_value = _make_tool_use_message({
        "sentiment_category": "positive",
        "confidence": 0.9,
        "sentiment_score": 5.0,
    })

    with pytest.raises(ValueError, match="sentiment_score"):
        label_article(client, "title", "body")


def test_load_already_labeled_ids_empty_when_no_file(tmp_path):
    assert load_already_labeled_ids(tmp_path / "missing.csv") == set()


def test_label_all_articles_skips_already_labeled(tmp_path, monkeypatch):
    raw_csv = tmp_path / "news_raw.csv"
    labeled_csv = tmp_path / "news_labeled.csv"

    with raw_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "published_on", "title", "body"])
        writer.writeheader()
        writer.writerow({"id": "1", "published_on": "100", "title": "A", "body": "body A"})
        writer.writerow({"id": "2", "published_on": "200", "title": "B", "body": "body B"})

    call_count = {"n": 0}

    def fake_label_article(client, title, body):
        call_count["n"] += 1
        return {"sentiment_category": "neutral", "confidence": 0.5, "sentiment_score": 0.0}

    monkeypatch.setattr("src.label_sentiment.label_article", fake_label_article)

    label_all_articles(raw_csv, labeled_csv, client=MagicMock())
    assert call_count["n"] == 2

    label_all_articles(raw_csv, labeled_csv, client=MagicMock())
    assert call_count["n"] == 2  # resumed run: nothing new labeled

    with labeled_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2


def test_label_all_articles_skips_empty_id_sentinel_row(tmp_path, monkeypatch):
    # fetch_news.py writes a sentinel row (blank id/title/body) for
    # genuinely empty weeks so they're tracked as fetched. That sentinel
    # must never reach label_article (it would waste an API call and
    # later crash the aggregator on a blank published_on).
    raw_csv = tmp_path / "news_raw.csv"
    labeled_csv = tmp_path / "news_labeled.csv"

    with raw_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "published_on", "title", "body", "_source_week_end"]
        )
        writer.writeheader()
        writer.writerow({
            "id": "", "published_on": "", "title": "", "body": "",
            "_source_week_end": "2024-01-07T23:59:59+00:00",
        })
        writer.writerow({
            "id": "1", "published_on": "100", "title": "A", "body": "body A",
            "_source_week_end": "2024-01-07T23:59:59+00:00",
        })

    call_count = {"n": 0}

    def fake_label_article(client, title, body):
        call_count["n"] += 1
        return {"sentiment_category": "neutral", "confidence": 0.5, "sentiment_score": 0.0}

    monkeypatch.setattr("src.label_sentiment.label_article", fake_label_article)

    label_all_articles(raw_csv, labeled_csv, client=MagicMock())

    assert call_count["n"] == 1

    with labeled_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["id"] == "1"


def test_label_all_articles_dedupes_same_id_within_one_run(tmp_path, monkeypatch):
    # If the same article id appears twice in the raw CSV within a single
    # run (e.g. duplicate across weeks), it should only be labeled once —
    # already_labeled must be updated as the run progresses, not just
    # snapshotted at the start.
    raw_csv = tmp_path / "news_raw.csv"
    labeled_csv = tmp_path / "news_labeled.csv"

    with raw_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "published_on", "title", "body"])
        writer.writeheader()
        writer.writerow({"id": "1", "published_on": "100", "title": "A", "body": "body A"})
        writer.writerow({"id": "1", "published_on": "100", "title": "A", "body": "body A"})

    call_count = {"n": 0}

    def fake_label_article(client, title, body):
        call_count["n"] += 1
        return {"sentiment_category": "neutral", "confidence": 0.5, "sentiment_score": 0.0}

    monkeypatch.setattr("src.label_sentiment.label_article", fake_label_article)

    label_all_articles(raw_csv, labeled_csv, client=MagicMock())

    assert call_count["n"] == 1

    with labeled_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1


def test_label_all_articles_abort_mid_batch_with_article_context(tmp_path, monkeypatch):
    raw_csv = tmp_path / "news_raw.csv"
    labeled_csv = tmp_path / "news_labeled.csv"

    with raw_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "published_on", "title", "body"])
        writer.writeheader()
        writer.writerow({"id": "1", "published_on": "100", "title": "A", "body": "body A"})
        writer.writerow({"id": "2", "published_on": "200", "title": "B", "body": "body B"})
        writer.writerow({"id": "3", "published_on": "300", "title": "C", "body": "body C"})

    def fake_label_article(client, title, body):
        if title == "B":
            raise ValueError("invalid sentiment_category: 'bullish'")
        return {"sentiment_category": "neutral", "confidence": 0.5, "sentiment_score": 0.0}

    monkeypatch.setattr("src.label_sentiment.label_article", fake_label_article)

    # First run should abort on article 2 with article context in error message
    with pytest.raises(ValueError, match="failed to label article id=2"):
        label_all_articles(raw_csv, labeled_csv, client=MagicMock())

    # Article 1 should be written before the failure
    with labeled_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["id"] == "1"

    # Article 2 should NOT be in the labeled ids (so resumed run retries it)
    already_labeled = load_already_labeled_ids(labeled_csv)
    assert "2" not in already_labeled
    assert "1" in already_labeled

    # No row should exist for article 2
    row_ids = {row["id"] for row in rows}
    assert "2" not in row_ids
