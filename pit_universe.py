"""Point-in-time universe membership: fixes forward-survivorship bias only.

What this fixes: any fixed ticker list - whether hand-picked (trends_provider.py's
TICKER_TO_QUERY) or systematic (sp500_tickers() below) - implicitly claims each
name was a large, liquid,
well-covered index constituent for the *entire* backtest window. For a name
added to the S&P 500 partway through that window (Uber, Airbnb, and Palantir
were all added within the last ~2-3 years as of writing), that claim is false
for the pre-addition period - the stock existed and traded, but its ownership
base, coverage, and liquidity profile were meaningfully different before index
inclusion. `mask_pre_inclusion` NaNs out signal/return cells before each name's
actual addition date, sourced from Wikipedia's current-membership table
(free, no key, cached to data_cache/).

What this does NOT fix: backward survivorship bias. A company removed from the
S&P 500 before today - via bankruptcy, acquisition, or delisting - is simply
absent from Wikipedia's *current* membership table and can never appear in a
universe built this way. There is no free source for full point-in-time index
history; fixing this properly requires a licensed dataset (WRDS/CRSP, or a paid
Nasdaq Data Link constituents table). Any DSR/CSCV result from a universe built
this way still overstates achievable Sharpe to the extent failed/acquired names
would have dragged on it - state this caveat alongside any result, don't treat
the forward-bias fix as a complete fix.

Tickers never in the current S&P 500 (e.g. Rivian, Shopify, Roku as of writing)
have no entry here and are left untouched - this module makes no claim about
them either way.
"""
from __future__ import annotations
import io
import json
import time
from pathlib import Path

import pandas as pd
import requests

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# Wikipedia blocks requests with no identifying User-Agent (403) - same fair-use
# norm as SEC EDGAR. Any descriptive string works; no account needed.
WIKI_USER_AGENT = "altq-research (replace-with-your-contact@example.com)"

CACHE_DIR = Path(__file__).parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)
_TABLE_CACHE_FILE = CACHE_DIR / "sp500_table.json"
_CACHE_FILE = CACHE_DIR / "sp500_addition_dates.json"
_CACHE_MAX_AGE_DAYS = 7


def _sp500_table() -> pd.DataFrame | None:
    """Raw current S&P 500 constituent table from Wikipedia (Symbol, Date added, ...).

    Shared fetch behind both sp500_addition_dates() and sp500_tickers() so a
    membership run only hits Wikipedia once per cache window. Returns None
    (with a printed warning) on any failure - callers decide the fallback.
    """
    if _TABLE_CACHE_FILE.exists():
        age_days = (time.time() - _TABLE_CACHE_FILE.stat().st_mtime) / 86400
        if age_days < _CACHE_MAX_AGE_DAYS:
            return pd.read_json(_TABLE_CACHE_FILE)
    try:
        resp = requests.get(WIKI_URL, headers={"User-Agent": WIKI_USER_AGENT}, timeout=20)
        resp.raise_for_status()
        table = pd.read_html(io.StringIO(resp.text))[0]
        table.to_json(_TABLE_CACHE_FILE)
        return table
    except Exception as exc:  # network hiccup, Wikipedia layout change, etc.
        print(f"[pit_universe] WARNING: could not fetch the S&P 500 table ({exc}).")
        return None


def sp500_addition_dates() -> dict[str, pd.Timestamp]:
    """ticker -> the date it was added to the S&P 500, for CURRENT members only.

    Returns {} (with a printed warning) if Wikipedia is unreachable, rather than
    raising - callers should treat a PIT mask as best-effort, not load-bearing.
    """
    if _CACHE_FILE.exists():
        age_days = (time.time() - _CACHE_FILE.stat().st_mtime) / 86400
        if age_days < _CACHE_MAX_AGE_DAYS:
            raw = json.loads(_CACHE_FILE.read_text())
            return {k: pd.Timestamp(v) for k, v in raw.items()}

    table = _sp500_table()
    if table is None:
        print("[pit_universe] proceeding with NO point-in-time mask applied.")
        return {}
    dates = pd.to_datetime(table["Date added"], errors="coerce")
    out = {sym: dt for sym, dt in zip(table["Symbol"], dates) if pd.notna(dt)}
    _CACHE_FILE.write_text(json.dumps({k: v.isoformat() for k, v in out.items()}))
    return out


def sp500_tickers(yahoo_format: bool = True) -> list[str] | None:
    """The full list of current S&P 500 constituent tickers (~500 names).

    This is the systematic replacement for a hand-picked ticker list: instead
    of an analyst choosing 25 "liquid, sector-diverse" names, the universe is
    every name in the actual index today. It's still forward-survivorship-
    biased on its own (see module docstring) - always run it through
    mask_pre_inclusion() - and it still says nothing about names removed from
    the index before today (backward survivorship, unfixed).

    yahoo_format=True rewrites Wikipedia's dotted share-class tickers (BRK.B)
    to Yahoo Finance's hyphenated form (BRK-B), since that's what yfinance and
    every provider in this project expect. Returns None (caller should fall
    back to a fixed list) if Wikipedia is unreachable.
    """
    table = _sp500_table()
    if table is None:
        return None
    symbols = table["Symbol"].astype(str).tolist()
    if yahoo_format:
        symbols = [s.replace(".", "-") for s in symbols]
    return symbols


def mask_pre_inclusion(signal: pd.DataFrame, returns: pd.DataFrame,
                        addition_dates: dict[str, pd.Timestamp] | None = None,
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """NaN out rows before each ticker's S&P 500 addition date, in both frames.

    Tickers with no known addition date (never an S&P 500 member, or the lookup
    failed) are returned unchanged - see module docstring for what that does and
    does not imply. NaN (not 0.0) is used deliberately: downstream cross-sectional
    ranking already treats NaN as "this name doesn't participate today," which is
    the correct point-in-time behavior - zero-filling would fabricate a real
    observation for a period the name shouldn't be in the universe at all.
    """
    addition_dates = sp500_addition_dates() if addition_dates is None else addition_dates
    signal, returns = signal.copy(), returns.copy()
    for col in set(signal.columns) & set(addition_dates):
        pre = signal.index < addition_dates[col]
        signal.loc[pre, col] = float("nan")
    for col in set(returns.columns) & set(addition_dates):
        pre = returns.index < addition_dates[col]
        returns.loc[pre, col] = float("nan")
    return signal, returns
