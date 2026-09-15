"""Real (non-mock) data provider: Wikipedia pageview attention shocks -> equity returns.

Third independent signal, after edgar_provider.py (SEC EDGAR filings) and
trends_provider.py (Google Trends search interest). Same underlying mechanism
as Trends - an abnormal spike in public attention to a company proxies a
retail-investor attention shock (Da, Engelberg & Gao, "In Search of
Attention," Journal of Finance, 2011) - but from a genuinely different data
source and API:

- Wikimedia's Pageviews API (wikimedia.org/api/rest_v1/metrics/pageviews) is an
  OFFICIAL, documented REST API, not an unofficial scraper like pytrends - no
  429 backoff dance needed.
- It returns absolute daily view counts, not a 0-100 relative index that gets
  retroactively renormalized whenever Google refreshes its Trends index.
- Daily granularity holds over multi-year windows (Trends degrades to weekly
  past a few months), which is a real statistical-power improvement.

Universe construction is systematic, not hand-picked, and that's the point of
building this as a third signal rather than just a third copy of Trends: for
every ticker in the current S&P 500 (pit_universe.sp500_tickers()), the
SEC-registered legal name (edgar_provider.ticker_to_legal_name(), from the
same company_tickers.json edgar_provider.py already caches) is resolved to a
Wikipedia article title in two passes: an exact `action=query&redirects`
lookup first (fast, batched, handles correctly-cased names like "Apple Inc."
straight to their canonical title), then a plain full-text search for
whatever's left, mostly SEC's inconsistently-cased legal names ("MICROSOFT
CORP", "AMAZON COM INC"), which a case-sensitive exact match can't reach. Two
naive approaches were tried and rejected for silently mapping to the wrong
page - Wikipedia's opensearch (fuzzy prefix search) sent "JPMorgan Chase &
Co." to a near-dead redirect stub at ~2 views/day instead of the real
"JPMorgan Chase" article at ~3,000/day, and an `intitle:"..."` search sent
"BOEING CO" to the "Boeing Model 42" article and "MICROSOFT CORP" to the
"United States v. Microsoft Corp." court-case article, both because literal
title-substring completeness outranked being the actual company page. Plain
full-text search - whose default relevance ranking factors in a page's
incoming links and traffic - got every one of those cases right. A ticker is
dropped, not guessed, if nothing resolves.

Caveats (read before trusting results):
- This resolution strategy was validated by hand against ~15 names spanning
  clean and messy SEC casing, not all ~500 - a wrong mapping for some
  obscure or generically-named company is still possible.
  `data_cache/wiki_article_titles.json` is a plain ticker->title JSON file;
  spot-check it if you extend this beyond a research prototype.
- Pageview spikes reflect ANY reason someone might read a company's Wikipedia
  page - a news event, a product launch, a scandal, historical curiosity, a
  homework assignment - not specifically trading intent. List this in any
  hypothesis's `confounders` field.
- Wikimedia's pageview data starts July 2015; irrelevant at the lookback
  windows this project uses, but worth knowing if you extend lookback_years.
- Same forward-survivorship caveat as the other two providers: mask_pre_inclusion()
  is applied here too; backward survivorship is not fixed (see pit_universe.py).
"""
from __future__ import annotations
import json
import re
import time
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from pit_universe import mask_pre_inclusion, sp500_tickers, WIKI_USER_AGENT
from edgar_provider import ticker_to_legal_name, FALLBACK_UNIVERSE

CACHE_DIR = Path(__file__).parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)

PAGEVIEWS_BASE = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents"
_ARTICLE_CACHE_FILE = CACHE_DIR / "wiki_article_titles.json"
_ARTICLE_CACHE_MAX_AGE_DAYS = 30
_SLEEP_BETWEEN_REQUESTS = 0.05  # Wikimedia's REST API has generous limits; this is politeness, not compliance


_QUERY_BATCH = 50  # MediaWiki's action=query allows up to 50 titles per request unauthenticated

# ~8% of S&P 500 legal names in SEC's data carry a trailing state-of-incorporation
# tag in varying, inconsistent formats: "CORP /DE/", "INC/MN", "CORP/OH/", "INC /
# MA". Left in, this isn't harmless noise - it measurably breaks search relevance:
# "IDEX CORP /DE/" (ticker IEX) matched an unrelated article ("Paramount
# Skydance") because stray tokens like the fragment "de" diluted the query into
# a near-generic 265-hit search. Stripped before every lookup below.
_STATE_SUFFIX_RE = re.compile(r"\s*/\s*[A-Za-z]{0,4}\s*/?\s*$")


def _clean_legal_name(name: str) -> str:
    return _STATE_SUFFIX_RE.sub("", name).strip()


def _search_fulltext(name: str) -> str | None:
    """Case-insensitive fallback for SEC legal names action=query can't match exactly.

    SEC's company_tickers.json is inconsistently cased - "Apple Inc." next to
    "MICROSOFT CORP" and "AMAZON COM INC" - and MediaWiki titles only
    auto-capitalize the first character, so an all-caps legal name never
    exact-matches a mixed-case article title.

    Plain full-text search (not `intitle:`-restricted) is used deliberately:
    CirrusSearch's default relevance ranking factors in a page's incoming
    links and traffic, which reliably surfaces the flagship company article
    first. An `intitle:"..."` variant was tried first and was worse, not
    better: it ranks literal title-substring completeness above everything
    else, so "BOEING CO" matched "Boeing Model 42" (whose title happens to
    contain that exact substring) over the real "Boeing" article, and
    "MICROSOFT CORP" matched the "United States v. Microsoft Corp." court-case
    article over "Microsoft" itself. Plain search got both of those - and
    every other case spot-checked by hand - right.
    """
    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": name,
                    "srlimit": 1, "format": "json"},
            headers={"User-Agent": WIKI_USER_AGENT}, timeout=10,
        )
        resp.raise_for_status()
        hits = resp.json()["query"]["search"]
        return hits[0]["title"] if hits else None
    except Exception:
        return None


def _resolve_article_titles(ticker_to_name: dict[str, str]) -> dict[str, str]:
    """ticker -> canonical Wikipedia article title, via exact-title + redirect resolution.

    Two passes: batched exact action=query&redirects (fast, handles correctly-
    cased legal names like "Apple Inc." straight to their canonical/redirected
    title), then _search_fulltext() one at a time for whatever didn't resolve
    (mostly the all-caps SEC names). A name that fails both is dropped, never
    guessed at.
    """
    if _ARTICLE_CACHE_FILE.exists():
        age_days = (time.time() - _ARTICLE_CACHE_FILE.stat().st_mtime) / 86400
        if age_days < _ARTICLE_CACHE_MAX_AGE_DAYS:
            return json.loads(_ARTICLE_CACHE_FILE.read_text())

    names = sorted({_clean_legal_name(n) for n in ticker_to_name.values()})
    resolved: dict[str, str] = {}  # legal name -> canonical article title
    for i in range(0, len(names), _QUERY_BATCH):
        batch = names[i:i + _QUERY_BATCH]
        try:
            resp = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={"action": "query", "titles": "|".join(batch), "redirects": "", "format": "json"},
                headers={"User-Agent": WIKI_USER_AGENT}, timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()["query"]
        except Exception:
            time.sleep(_SLEEP_BETWEEN_REQUESTS)
            continue
        norm_map = {e["from"]: e["to"] for e in data.get("normalized", [])}
        redir_map = {e["from"]: e["to"] for e in data.get("redirects", [])}
        valid_titles = {p["title"] for p in data.get("pages", {}).values() if "missing" not in p}
        for name in batch:
            final = redir_map.get(norm_map.get(name, name), norm_map.get(name, name))
            if final in valid_titles:
                resolved[name] = final
        time.sleep(_SLEEP_BETWEEN_REQUESTS)

    for name in names:
        if name not in resolved:
            title = _search_fulltext(name)
            if title:
                resolved[name] = title
            time.sleep(_SLEEP_BETWEEN_REQUESTS)

    out = {ticker: resolved[_clean_legal_name(name)] for ticker, name in ticker_to_name.items()
           if _clean_legal_name(name) in resolved}
    _ARTICLE_CACHE_FILE.write_text(json.dumps(out))
    return out


def _cached_pageviews(ticker: str, article_title: str, start: str, end: str) -> pd.Series:
    cache_path = CACHE_DIR / f"wikipv_{ticker}.json"
    if cache_path.exists():
        age_days = (time.time() - cache_path.stat().st_mtime) / 86400
        if age_days < 1.0:
            raw = json.loads(cache_path.read_text())
            return pd.Series(raw["views"], index=pd.to_datetime(raw["dates"]))

    safe_title = urllib.parse.quote(article_title.replace(" ", "_"), safe=",()'")
    url = f"{PAGEVIEWS_BASE}/{safe_title}/daily/{start}/{end}"
    resp = requests.get(url, headers={"User-Agent": WIKI_USER_AGENT}, timeout=20)
    time.sleep(_SLEEP_BETWEEN_REQUESTS)
    if resp.status_code == 404:
        return pd.Series(dtype=float)
    resp.raise_for_status()
    items = resp.json()["items"]
    dates = [it["timestamp"][:8] for it in items]
    views = [float(it["views"]) for it in items]
    cache_path.write_text(json.dumps({"dates": dates, "views": views}))
    return pd.Series(views, index=pd.to_datetime(dates))


class WikipediaAttentionProvider:
    """Real-data DataProvider: Wikipedia pageview attention shocks -> equity returns.

    load(hyp) -> (signal_df, returns_df), same interface as every other
    provider in this project - no other code needs to change.
    """

    def __init__(self, universe: list[str] | None = None, lookback_years: float = 3.0,
                 baseline_window_periods: int = 8):
        self.universe = universe
        self.lookback_years = lookback_years
        self.baseline_window_periods = baseline_window_periods
        self._cache: tuple[pd.DataFrame, pd.DataFrame] | None = None

    def load(self, hyp) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self._cache is not None:
            return self._cache
        import yfinance as yf

        universe = self.universe
        if universe is None:
            universe = sp500_tickers()
            if universe is None:
                print("[wiki_provider] WARNING: could not fetch the systematic S&P 500 "
                      "universe; falling back to edgar_provider.FALLBACK_UNIVERSE.")
                universe = FALLBACK_UNIVERSE

        start = (pd.Timestamp.today() - pd.Timedelta(days=self.lookback_years * 365.25)).date()
        end = pd.Timestamp.today().date()

        prices = yf.download(universe, start=start, auto_adjust=True, progress=False)["Close"]
        prices = prices.dropna(axis=1, how="all")
        rets = prices.pct_change().dropna(how="all")

        name_map = ticker_to_legal_name()
        candidates = {t: name_map[t] for t in prices.columns if t in name_map}
        article_titles = _resolve_article_titles(candidates)

        series = {}
        for ticker, title in article_titles.items():
            s = _cached_pageviews(ticker, title, start.strftime("%Y%m%d"), end.strftime("%Y%m%d"))
            if not s.empty:
                series[ticker] = s
        pv = pd.DataFrame(series)

        # Abnormal attention = current pageviews minus this article's own trailing
        # baseline, in units of its own trailing std - the shock, not the level
        # (a company's baseline traffic scales with brand size, not tradability).
        baseline = pv.rolling(self.baseline_window_periods).mean()
        vol = pv.rolling(self.baseline_window_periods).std()
        shock = (pv - baseline) / vol.replace(0, np.nan)

        sig = shock.reindex(rets.index, method="ffill")
        common = [c for c in rets.columns if c in sig.columns]
        rets, sig = rets[common].fillna(0.0), sig[common].fillna(0.0)
        sig, rets = mask_pre_inclusion(sig, rets)   # forward-survivorship fix - see pit_universe.py
        self._cache = (sig, rets)
        return self._cache
