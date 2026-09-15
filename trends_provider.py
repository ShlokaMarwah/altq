"""Real (non-mock) data provider: Google search-attention shocks -> equity returns.

Hypothesis under test: an abnormal spike in Google search interest for a
company's consumer-facing brand name - measured relative to that name's own
trailing baseline, not its raw level - proxies a retail-investor attention
shock. Per the retail-attention literature (Da, Engelberg & Gao, "In Search of
Attention," Journal of Finance, 2011), attention shocks are associated with
short-horizon price pressure. That prior result does not automatically transfer
to this universe/window: the Skeptic gate is what decides whether it survives
here, not the literature citation.

This is a genuinely independent second signal from edgar_provider.py - different
data source (Google Trends vs. SEC filings), different economic mechanism
(retail attention vs. disclosure-driven uncertainty), different universe
(consumer/retail brand names vs. a general large-cap mix) - so Discovery has two
real, uncorrelated feeds to draw hypotheses from instead of one.

Caveats (read before trusting results):
- pytrends is an UNOFFICIAL scraping wrapper around Google Trends, not a
  supported API. It rate-limits aggressively (HTTP 429) and can change behavior
  without notice - treat this as a research prototype, not a production feed.
- Google Trends values are RELATIVE (0-100, independently rescaled per query
  batch), not absolute search volume, and are retroactively re-normalized
  whenever Google refreshes its index - a real restatement-leakage risk, not
  just a theoretical one. List it in any hypothesis's `confounders` field.
- Search terms are bare brand names, not disambiguated entities. Only names that
  are overwhelmingly brand-dominant as a plain search query are included below
  (no common English words, no multi-meaning collisions); even so, treat this as
  approximate.
- Google Trends returns weekly (not daily) granularity for multi-year windows -
  the signal changes once per week; this is handled generically (rolling
  window operates on whatever periodicity comes back) but means the effective
  number of independent observations is far lower than the trading-day count.
"""
from __future__ import annotations
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pit_universe import mask_pre_inclusion

CACHE_DIR = Path(__file__).parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)

# Ticker -> Google Trends search query. Picked for being overwhelmingly
# brand-dominant as a bare search term - deliberately excludes ambiguous names
# like "Target" (the word), "Apple" (the fruit), or "Meta" (the common word).
TICKER_TO_QUERY = {
    "TSLA": "Tesla", "NVDA": "Nvidia", "NFLX": "Netflix", "SBUX": "Starbucks",
    "NKE": "Nike", "BA": "Boeing", "DIS": "Disney", "COST": "Costco",
    "UBER": "Uber", "PYPL": "PayPal", "ABNB": "Airbnb", "PLTR": "Palantir",
}

_BATCH = 5                      # pytrends' max keywords per request
_SLEEP_BETWEEN_BATCHES = 5.0    # unofficial endpoint - stay well clear of 429s
_CACHE_MAX_AGE_DAYS = 1.0
_MAX_RETRIES = 4
_BACKOFF_BASE_SECONDS = 20.0    # 20s, 40s, 80s, 160s - Google's 429 cooldown is minutes-scale


def _fetch_batch_with_retry(pt, queries: list[str], timeframe: str) -> pd.DataFrame:
    from pytrends.exceptions import TooManyRequestsError
    for attempt in range(_MAX_RETRIES):
        try:
            pt.build_payload(queries, timeframe=timeframe)
            return pt.interest_over_time()
        except TooManyRequestsError:
            if attempt == _MAX_RETRIES - 1:
                raise
            wait = _BACKOFF_BASE_SECONDS * (2 ** attempt)
            print(f"[trends_provider] 429 from Google Trends on {queries}, "
                  f"backing off {wait:.0f}s (attempt {attempt + 1}/{_MAX_RETRIES})")
            time.sleep(wait)
    return pd.DataFrame()


def _cached_trends(tickers: list[str], timeframe: str) -> pd.DataFrame:
    cache_path = CACHE_DIR / f"trends_{'_'.join(sorted(tickers))}_{timeframe.replace(' ', '_')}.json"
    if cache_path.exists():
        age_days = (time.time() - cache_path.stat().st_mtime) / 86400
        if age_days < _CACHE_MAX_AGE_DAYS:
            return pd.read_json(cache_path)

    from pytrends.request import TrendReq
    pt = TrendReq(hl="en-US", tz=360)
    frames = []
    for i in range(0, len(tickers), _BATCH):
        batch = tickers[i:i + _BATCH]
        queries = [TICKER_TO_QUERY[t] for t in batch]
        df = _fetch_batch_with_retry(pt, queries, timeframe)
        if not df.empty:
            df = df.drop(columns=["isPartial"], errors="ignore")
            df.columns = batch
            frames.append(df)
        time.sleep(_SLEEP_BETWEEN_BATCHES)
    out = pd.concat(frames, axis=1) if frames else pd.DataFrame()
    out.to_json(cache_path)
    return out


class GoogleTrendsAttentionProvider:
    """Real-data DataProvider: retail search-attention shocks -> equity returns.

    load(hyp) -> (signal_df, returns_df), same interface as the mock DataProvider
    and EdgarFilingCadenceProvider in graph.py - no other code needs to change.
    """

    def __init__(self, universe: list[str] | None = None, lookback_years: float = 3.0,
                 baseline_window_periods: int = 8):
        self.universe = universe or list(TICKER_TO_QUERY)
        self.lookback_years = lookback_years
        self.baseline_window_periods = baseline_window_periods
        self._cache: tuple[pd.DataFrame, pd.DataFrame] | None = None

    def load(self, hyp) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self._cache is not None:
            return self._cache
        import yfinance as yf

        start = (pd.Timestamp.today() - pd.Timedelta(days=self.lookback_years * 365.25)).date()
        end = pd.Timestamp.today().date()

        prices = yf.download(self.universe, start=start, auto_adjust=True,
                             progress=False)["Close"]
        prices = prices.dropna(axis=1, how="all")
        rets = prices.pct_change().dropna(how="all")

        timeframe = f"{start.isoformat()} {end.isoformat()}"
        trends = _cached_trends([t for t in self.universe if t in prices.columns], timeframe)
        trends.index = pd.to_datetime(trends.index)

        # Abnormal attention = current interest minus its own trailing baseline,
        # in units of that name's own trailing std - the shock, not the level.
        baseline = trends.rolling(self.baseline_window_periods).mean()
        vol = trends.rolling(self.baseline_window_periods).std()
        shock = (trends - baseline) / vol.replace(0, np.nan)

        sig = shock.reindex(rets.index, method="ffill")
        common = [c for c in rets.columns if c in sig.columns]
        rets, sig = rets[common].fillna(0.0), sig[common].fillna(0.0)
        sig, rets = mask_pre_inclusion(sig, rets)   # forward-survivorship fix - see pit_universe.py
        self._cache = (sig, rets)
        return self._cache
