import csv
from pathlib import Path

from anthropic import Anthropic

from src.config import get_env

MODEL = "claude-haiku-4-5-20251001"
MAX_BODY_CHARS = 2000

SENTIMENT_TOOL = {
    "name": "record_sentiment",
    "description": "Record the sentiment assessment of a crypto news article.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "sentiment_category": {
                "type": "string",
                "enum": ["positive", "neutral", "negative"],
            },
            # strict:true tool schemas reject "minimum"/"maximum" on number
            # properties (confirmed via a live 400 response) — range is
            # enforced in label_article()'s own validation instead.
            "confidence": {"type": "number"},
            "sentiment_score": {"type": "number"},
        },
        "required": ["sentiment_category", "confidence", "sentiment_score"],
        "additionalProperties": False,
    },
}

VALID_CATEGORIES = {"positive", "neutral", "negative"}


def label_article(client: Anthropic, title: str, body: str) -> dict:
    """Calls Claude once for a single article, requesting a structured
    sentiment assessment via tool use. Returns a dict with keys
    sentiment_category, confidence, sentiment_score. Raises ValueError if
    the model doesn't return a valid tool call, or if the returned values
    fail basic range/enum validation."""
    truncated_body = (body or "")[:MAX_BODY_CHARS]
    message = client.messages.create(
        model=MODEL,
        max_tokens=200,
        tools=[SENTIMENT_TOOL],
        tool_choice={"type": "tool", "name": "record_sentiment"},
        messages=[{
            "role": "user",
            "content": (
                "Assess the sentiment of this crypto news article toward "
                "Bitcoin's price outlook.\n\n"
                f"Title: {title}\n\nBody: {truncated_body}"
            ),
        }],
    )

    tool_use_blocks = [b for b in message.content if b.type == "tool_use"]
    if not tool_use_blocks:
        raise ValueError("model did not return a tool_use block")

    result = tool_use_blocks[0].input

    if result.get("sentiment_category") not in VALID_CATEGORIES:
        raise ValueError(f"invalid sentiment_category: {result.get('sentiment_category')!r}")
    if not (0 <= result.get("confidence", -1) <= 1):
        raise ValueError(f"confidence out of range: {result.get('confidence')!r}")
    if not (-1 <= result.get("sentiment_score", -2) <= 1):
        raise ValueError(f"sentiment_score out of range: {result.get('sentiment_score')!r}")

    return {
        "sentiment_category": result["sentiment_category"],
        "confidence": result["confidence"],
        "sentiment_score": result["sentiment_score"],
    }


def load_already_labeled_ids(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["id"] for row in reader}


def label_all_articles(
    raw_csv_path: Path,
    labeled_csv_path: Path,
    client: Anthropic | None = None,
) -> None:
    """Reads articles from raw_csv_path, labels each one not already
    present in labeled_csv_path, and appends results immediately after
    each successful call — an interrupted run doesn't re-pay for
    already-labeled articles on resume."""
    client = client or Anthropic(api_key=get_env("ANTHROPIC_API_KEY"))
    already_labeled = load_already_labeled_ids(labeled_csv_path)
    file_exists = labeled_csv_path.exists()
    labeled_csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["id", "published_on", "title", "sentiment_category", "confidence", "sentiment_score"]

    with raw_csv_path.open(newline="", encoding="utf-8") as raw_f, \
         labeled_csv_path.open("a", newline="", encoding="utf-8") as out_f:
        reader = csv.DictReader(raw_f)
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for row in reader:
            if not row["id"]:
                continue  # empty-week sentinel row from fetch_news, not a real article

            if row["id"] in already_labeled:
                continue

            try:
                label = label_article(client, row["title"], row["body"])
            except ValueError as exc:
                raise ValueError(f"failed to label article id={row['id']}: {exc}") from exc
            writer.writerow({
                "id": row["id"],
                "published_on": row["published_on"],
                "title": row["title"],
                **label,
            })
            out_f.flush()
            already_labeled.add(row["id"])


if __name__ == "__main__":
    label_all_articles(
        raw_csv_path=Path("data/news_raw.csv"),
        labeled_csv_path=Path("data/news_labeled.csv"),
    )
