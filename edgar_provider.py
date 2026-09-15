"""Real (non-mock) data provider: SEC EDGAR 8-K filing cadence -> equity returns.

Hypothesis under test: elevated unscheduled material-event disclosure frequency
(8-K filing rate, trailing window) proxies operational/financial uncertainty.
Information diffuses slowly to less-attentive investors, producing a short-horizon
negative drift in names with abnormally high recent 8-K cadence relative to their
own history and peers. This is a real, testable hypothesis in the post-disclosure
drift literature - the pipeline's job is to find out whether it survives the
Skeptic's DSR/CSCV gate on this specific universe and window, not to assume it does.

Point-in-time note: an SEC filing's `filingDate` is fixed at submission and never
restated (unlike, say, reported financials), so the signal construction itself has
no look-ahead. This module does NOT enforce point-in-time *universe* membership -
today's ticker list is used across all history, which is a survivorship-bias risk
for a real study. See README.md -> Limitations.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import pandas as pd
import requests

# SEC requires a descriptive User-Agent identifying the requester per its
# fair-access policy: https://www.sec.gov/os/webmaster-faq#developers
# Replace this with your own name/contact before running anything beyond a
# quick local test - do not ship the placeholder.
SEC_USER_AGENT = "altq-research (replace-with-your-contact@example.com)"

CACHE_DIR = Path(__file__).parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)

# Liquid, sector-diverse large caps - gives cross-sectional dispersion for
# quintile ranking. Swap freely; more names = better-powered CSCV blocks.
DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "BAC", "WFC",
    "XOM", "CVX", "JNJ", "PFE", "UNH", "PG", "KO", "WMT", "HD", "DIS",
    "CAT", "BA", "GE", "UPS", "NKE",
]

_SLEEP_BETWEEN_REQUESTS = 0.15  # stay well under SEC's 10 req/sec fair-access limit


def _cached_get(url: str, cache_path: Path, max_age_days: float = 1.0) -> dict:
    if cache_path.exists():
        age_days = (time.time() - cache_path.stat().st_mtime) / 86400
        if age_days < max_age_days:
            return json.loads(cache_path.read_text())
    resp = requests.get(url, headers={"User-Agent": SEC_USER_AGENT}, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    cache_path.write_text(json.dumps(data))
    time.sleep(_SLEEP_BETWEEN_REQUESTS)
    return data


def _cik_map() -> dict[str, str]:
    data = _cached_get(
        "https://www.sec.gov/files/company_tickers.json",
        CACHE_DIR / "company_tickers.json", max_age_days=7,
    )
    return {row["ticker"]: str(row["cik_str"]).zfill(10) for row in data.values()}


def _eight_k_dates(ticker: str, cik10: str) -> list[str]:
    data = _cached_get(
        f"https://data.sec.gov/submissions/CIK{cik10}.json",
        CACHE_DIR / f"submissions_{ticker}.json", max_age_days=1.0,
    )
    recent = data["filings"]["recent"]
    return [d for f, d in zip(recent["form"], recent["filingDate"]) if f.startswith("8-K")]


def _filing_cadence_signal(universe: list[str], calendar_index: pd.DatetimeIndex,
                            window_days: int = 90) -> pd.DataFrame:
    """Trailing-N-calendar-day count of 8-K filings per ticker, as of each day."""
    cik = _cik_map()
    cols = {}
    for t in universe:
        if t not in cik:
            continue
        dates = pd.to_datetime(_eight_k_dates(t, cik[t])).normalize()
        counts = pd.Series(1.0, index=dates).groupby(level=0).sum()
        daily = counts.reindex(calendar_index, fill_value=0.0)
        cols[t] = daily.rolling(f"{window_days}D").sum()
    return pd.DataFrame(cols)


class EdgarFilingCadenceProvider:
    """Real-data drop-in replacement for the mock DataProvider in graph.py.

    load(hyp) -> (signal_df, returns_df), both indexed by trading day with
    matching tickers as columns - the exact interface _variant_returns() and
    the mock DataProvider already expect, so no other code needs to change.
    Results are memoized in-process (one network round-trip per process run);
    raw API responses are cached to disk under data_cache/ across runs.
    """

    def __init__(self, universe: list[str] | None = None, lookback_years: float = 3.0,
                 filing_window_days: int = 90):
        self.universe = universe or DEFAULT_UNIVERSE
        self.lookback_years = lookback_years
        self.filing_window_days = filing_window_days
        self._cache: tuple[pd.DataFrame, pd.DataFrame] | None = None

    def load(self, hyp) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self._cache is not None:
            return self._cache
        import yfinance as yf

        start = (pd.Timestamp.today() - pd.Timedelta(days=self.lookback_years * 365.25)).date()
        prices = yf.download(self.universe, start=start, auto_adjust=True,
                             progress=False)["Close"]
        prices = prices.dropna(axis=1, how="all")
        rets = prices.pct_change().dropna(how="all")

        calendar_index = pd.date_range(rets.index.min(), rets.index.max(), freq="D")
        sig_calendar = _filing_cadence_signal(list(prices.columns), calendar_index,
                                              self.filing_window_days)
        sig = sig_calendar.reindex(rets.index, method="ffill")

        common = [c for c in rets.columns if c in sig.columns]
        rets, sig = rets[common].fillna(0.0), sig[common].fillna(0.0)
        self._cache = (sig, rets)
        return self._cache
