# Crypto News Sentiment vs. BTC Price Movement

A hypothesis test — is there a statistically detectable relationship
between BTC-related news sentiment and BTC price movement — grounded in
real news-aggregation experience at TheCoinZone (CryptoPanic) and GenAI
sentiment labeling. 

**This is a hypothesis test, not a trading model.** No predictive up/down
classifier or trading-strategy claim appears anywhere in this repo.

## Problem

Does BTC-related news sentiment carry a statistically detectable
relationship with BTC's subsequent price movement, across several time
horizons? Answered as a hypothesis test against real 2024 market data, not
as a production trading signal.

## Data

- **News:** CryptoCompare/CoinDesk News API (`min-api.cryptocompare.com`),
  BTC category, English language. 
- **Sampling:** stratified by calendar week across 2024 (53 weeks), one API
  call per week anchored at week-end, up to 50 most-recent articles per
  week — fits the free tier's 100-requests/month cap. In practice this
  collapsed further than expected: the BTC news category is high-volume
  enough that every week's "50 most recent" query is satisfied entirely by
  articles published within about a day of the cutoff, so all 2650
  articles landed on exactly 53 single calendar dates (one per sampled
  week), not spread across the year. Documented as a limitation, not
  hidden — see Results/Limitations below for how it shaped the analysis.
- **Price:** Binance public klines API (`api.binance.com`), `BTCUSDT`,
  hourly candles, no API key required. 2024 plus a 14+ day buffer on both
  ends (needed for the 14-day return horizon near year boundaries).

## Methodology

1. **Sentiment labeling** — one Claude API call per article (small/fast
   model — a bounded classification task doesn't need a top-tier model),
   returning a category (positive/neutral/negative), a confidence score,
   and a continuous -1..+1 sentiment score in one structured response.
2. **Aggregation** — articles bucketed into an hourly sentiment index
   (mean and confidence-weighted mean sentiment, plus article count as a
   volume signal).
3. **Hypothesis test**, in increasing order of rigor:
   - Pearson/Spearman correlation of the sentiment index against BTC
     returns at 1h/24h/7d/14d horizons.
   - OLS regression of returns on the sentiment index plus lagged values —
     distinguishes "sentiment leads price" from "sentiment reacts to
     price."
   - Granger causality test (`statsmodels`), both directions.
   - Bonferroni correction across the 4 horizons × multiple tests.
4. **Held-out evaluation** — relationship characterized on the first ~10
   months of 2024; the last ~2 months held out to check whether it
   replicates out-of-sample.

## Results

**Partial relationship, and in the opposite direction from a trading-signal
framing.** Full detail and code in `crypto_sentiment_btc.ipynb`.

- **No evidence that sentiment predicts future price moves.** Across all 4
  horizons, in the correlation, OLS-with-lags, and Granger-causality tests,
  `sentiment -> returns` never survives Bonferroni correction — a checked
  null result, not just an absence of looking.
- **One relationship is real:** daily-sampled sentiment moves together with
  the *contemporaneous* 24h return (Pearson r≈0.47, survives correction in
  two independent test families). It holds directionally on a held-out
  final two months (r≈0.59) but that split (n=10) is too small to
  reconfirm significance on its own.
- **The strongest, most robust finding runs the other way:** Granger
  causality shows `returns -> sentiment` at the 7-day and 14-day horizons
  (p < 0.002, both lags, comfortably survives correction). BTC's price
  trend over the prior week or two Granger-causes that period's news
  sentiment — news reads as a lagging narrative on price, not a leading
  indicator of it.
- **Biggest caveat:** the sampling collapse described above means every
  test above ultimately runs on 53 independent weekly observations, not
  the ~2600-article or 490-hour sample size it might look like at first —
  this shaped both which aggregation level (daily, primary) and which
  Bonferroni correction scheme the notebook used.

Consistent with this project's opening framing: crypto markets are close
to efficient, and a specific, honestly-null result in the "predictive"
direction is a legitimate, useful portfolio outcome here — not a
disappointing one.

## How to reproduce

```bash
uv sync
uv run pytest                                  # unit tests (all mocked, no live API calls)
cp .env.example .env                           # then fill in CRYPTOCOMPARE_API_KEY, ANTHROPIC_API_KEY
uv run python -m src.fetch_price               # pulls data/btc_price_hourly.csv (no key needed)
uv run python -m src.fetch_news                # pulls data/news_raw.csv (spends real API quota)
uv run python -m src.label_sentiment           # pulls data/news_labeled.csv (spends real API budget)
uv run jupyter notebook crypto_sentiment_btc.ipynb
```

`crypto_sentiment_btc.ipynb` is the finished analysis (executed, real
results).

## Limitations

- **Sampling collapsed to 53 single days, not a year-round sample** — see
  Data/Sampling above. The effective sample size for every hypothesis test
  is 53 weekly observations, not ~2600 articles or 490 news-hours; this is
  the dominant limitation on statistical power throughout the analysis. A
  rerun with pagination within each week (rather than "50 most recent")
  would fix this without needing more calendar time.
- Single asset (BTC), single year (2024) — findings don't automatically
  generalize without rerunning the pipeline.
- LLM sentiment labels aren't validated against human-annotated ground
  truth.
- Possible endogeneity (sentiment may react to price rather than predict
  it) — this is exactly what the Granger causality test checks, and the
  write-up reports whichever direction the data actually shows.
- Free-tier API quota — reusing this pipeline for a different year or
  asset requires a fresh month of CryptoCompare/CoinDesk quota.
- CryptoPanic itself was never used as a data source (its domain returns
  HTTP 403 from Cloudflare on every path, including the API, confirmed
  from two independent networks) — findings are grounded in a different,
  though overlapping, news source.
