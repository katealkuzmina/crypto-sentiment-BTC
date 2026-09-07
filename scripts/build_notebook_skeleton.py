import nbformat as nbf


def build_skeleton_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Crypto News Sentiment vs. BTC Price Movement\n\n"
            "Hypothesis test: is there a statistically detectable relationship "
            "between BTC-related news sentiment and BTC price movement over "
            "2024? Grounded in real news-aggregation experience (TheCoinZone) "
            "and GenAI-based sentiment labeling."
        ),
        nbf.v4.new_markdown_cell("## Setup"),
        nbf.v4.new_code_cell(
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "import statsmodels.api as sm\n"
            "from statsmodels.tsa.stattools import grangercausalitytests\n"
            "from scipy import stats\n\n"
            "from src.returns import HORIZONS_HOURS, compute_forward_returns\n"
            "from src.aggregate import build_sentiment_index, join_sentiment_and_returns\n\n"
            "pd.set_option(\"display.max_columns\", 60)\n"
            "sns.set_theme(style=\"whitegrid\")"
        ),
        nbf.v4.new_code_cell(
            "labeled = pd.read_csv(\"data/news_labeled.csv\")\n"
            "price = pd.read_csv(\"data/btc_price_hourly.csv\", index_col=0, parse_dates=True)[\"close\"]\n\n"
            "print(labeled.shape, price.shape)\n"
            "labeled.head()"
        ),
        nbf.v4.new_markdown_cell("## 1. EDA"),
        nbf.v4.new_code_cell("# your code here"),
        nbf.v4.new_markdown_cell("## 2. Sentiment index and returns"),
        nbf.v4.new_code_cell(
            "sentiment_index = build_sentiment_index(labeled)\n"
            "joined = join_sentiment_and_returns(sentiment_index, price)\n"
            "joined.head()"
        ),
        nbf.v4.new_markdown_cell("## 3. Hypothesis tests"),
        nbf.v4.new_code_cell("# correlation, OLS with lags, Granger causality"),
        nbf.v4.new_markdown_cell("## 4. Held-out validation"),
        nbf.v4.new_code_cell("# split: first ~10 months vs. last ~2 months of 2024"),
        nbf.v4.new_markdown_cell("## 5. Write-up"),
        nbf.v4.new_markdown_cell("*(results and conclusions here)*"),
    ]
    return nb


import argparse
from pathlib import Path


def write_skeleton_notebook(path: Path, overwrite: bool = False) -> None:
    """Writes the skeleton notebook to `path`. Refuses if `path` already
    exists unless `overwrite=True` — a finished, executed notebook is
    indistinguishable from a stale skeleton at the filesystem level, and
    this script has no way to tell an in-progress analysis from one safe
    to discard."""
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists — refusing to overwrite. "
            "Pass overwrite=True (or --force on the CLI) if this is intentional."
        )
    nb = build_skeleton_notebook()
    with path.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Overwrite an existing notebook")
    args = parser.parse_args()
    write_skeleton_notebook(Path("crypto_sentiment_btc.ipynb"), overwrite=args.force)
