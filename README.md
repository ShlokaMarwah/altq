# altq — Alternative-Data Quant Research Pipeline

An autonomous, multi-agent pipeline that proposes, tests, and gates unconventional
alternative-data correlations with equity returns before they ever reach a portfolio.
The design is built around one idea: **most backtested "alpha" is multiple-testing
noise**, so the pipeline treats every hypothesis as guilty until it survives an
adversarial statistical review.

## Why this exists

A naive research loop — try an idea, backtest it, keep it if the Sharpe ratio looks
good — will eventually produce a great-looking Sharpe ratio out of pure noise if you
try enough ideas. This is the data-mining / multiple-testing problem, and it is the
single biggest reason backtested strategies fail in live trading. This pipeline
addresses it structurally:

- Every hypothesis and every parameter variant of that hypothesis is counted as a
  "trial."
- The trial count compounds every time a hypothesis is rejected and resubmitted, so
  an agent cannot quietly re-run the same idea with a tweaked lag or window to get a
  better-looking result for free.
- A hypothesis is only accepted if its **Deflated Sharpe Ratio** and **Combinatorially
  Symmetric Cross-Validation (CSCV)** overfitting probability both clear a strict bar
  *after* that inflated trial count is applied.

## Architecture

Four agents/stages wired as a LangGraph state machine, with a conditional edge that
sends rejected hypotheses back to the start with a penalty attached:

```
        ┌─────────────┐      ┌──────────────┐      ┌───────────────┐
   ┌───▶│  Discovery  │─────▶│  ETL/Feature │─────▶│    Skeptic    │
   │    │    Agent    │      │    Agent     │      │     Agent     │
   │    └─────────────┘      └──────────────┘      └───────┬───────┘
   │                                                        │
   │        REJECT / REVISE (trial penalty × 1.5)           │  ACCEPT
   └────────────────────────────────────────────────────────┤
                                                              ▼
                                              ┌───────────────────────┐
                                              │   Portfolio Agent     │
                                              │ (sizing, kill-switch) │
                                              └───────────────────────┘

   trial budget exhausted ──▶ ABORT (no trade)
```

| Stage | Role |
|---|---|
| **Discovery** | Proposes a falsifiable alt-data → equity hypothesis: data source, transmission mechanism, target universe, lag structure, signal construction, a null prediction that would reject it, and known confounders. |
| **ETL / Feature** | Pulls the (point-in-time) signal and return data, builds the actual return series for every parameter variant declared (lag × window grid), and hands a full trial matrix to the Skeptic — not just the best-looking column. |
| **Skeptic** | The adversary. Computes Deflated Sharpe Ratio and CSCV probability-of-overfitting on the *entire* trial matrix, with the honest (and penalty-inflated) trial count. Defaults to REJECT; only accepts if every check passes explicitly. |
| **Portfolio** | Sizes the accepted strategy to a target volatility, attaches a kill-switch (rolling Deflated Sharpe Ratio floor, max drawdown), and emits the execution payload. |

## Files

| File | Contents |
|---|---|
| `graph.py` | State schema (`PipelineState`), all four agent nodes, the mock point-in-time `DataProvider`, and the LangGraph wiring including the reject → Discovery feedback edge. |
| `skeptic_stats.py` | Pure `numpy` + stdlib implementation of the Deflated Sharpe Ratio (Bailey & López de Prado, 2014) and CSCV / probability of backtest overfitting (Bailey, Borwein, López de Prado & Zhu, 2017). No `scipy` dependency — see [Notes](#notes) below. |
| `prompts.py` | Condensed system prompts for the Discovery and Skeptic agents. |
| `run.py` | Production entry point: initializes state, streams node transitions to stdout, prints the final execution payload. |

## The statistical gate, in plain terms

**Deflated Sharpe Ratio (DSR)** asks: given that you tried `N` variants and picked
the best one, what's the probability the *true* Sharpe ratio is actually positive,
once you account for (a) how much luck you'd expect from trying `N` things and
(b) the non-normality (skew, fat tails) of the actual return series? A raw Sharpe
ratio of 2.0 selected from 50 trials can have a DSR near zero — the pipeline gates
on DSR ≥ 0.95, not on the raw Sharpe ratio.

**CSCV / Probability of Backtest Overfitting (PBO)** asks a different question:
if you split history into blocks and repeatedly pick the best-in-sample variant,
how often does that pick turn out to be below-median out-of-sample? A strategy that
only looks good in-sample will rank poorly out-of-sample most of the time. The
pipeline gates on PBO ≤ 0.20.

Both gates were validated against synthetic panels with a known ground truth before
being wired into the graph:

| Panel | Raw annualized Sharpe | DSR | PBO | Gate result |
|---|---|---|---|---|
| Pure noise, 50 trials, best-of-N | 1.24 | 0.46 | 0.36 | REJECT (correct — no real signal) |
| Injected alpha in 1 of 50 columns | 2.07 | 0.89 | 0.10 | REJECT (correct — DSR still short of 0.95 threshold at N=50) |

## Running it

```powershell
cd C:\Users\marwa\projects\altq
.\venv\Scripts\Activate.ps1

# Dry run — no LLM calls, deterministic stub hypothesis, mock data provider
$env:ALTQ_DRY_RUN = "1"
$env:ALTQ_TRUE_BETA = "0.4"     # 0.0 = pure noise signal, >0 = injected true alpha
python run.py 3                  # arg = trial budget before ABORT
```

With `ALTQ_TRUE_BETA=0.0` the mock signal carries no real information, so all three
trials are REJECTed and the run ends in ABORT. With `ALTQ_TRUE_BETA=0.4` a real
signal is injected and the first trial is ACCEPTed, producing a sized execution
payload with a kill-switch attached.

To use live Claude API calls for Discovery and the Skeptic (`claude-opus-5`,
adaptive thinking, structured Pydantic outputs) instead of the deterministic stubs:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
$env:ALTQ_DRY_RUN = "0"
python run.py 5
```

### Running it against real data (SEC EDGAR + Yahoo Finance)

`edgar_provider.py` is a real, non-mock `DataProvider` implementation. It tests one
concrete hypothesis: does an issuer's trailing 90-day count of 8-K filings (a proxy
for unscheduled-disclosure / operational-uncertainty intensity) cross-sectionally
predict short-horizon returns, across the **full, systematic current S&P 500
constituent list** (503 names, fetched live from Wikipedia via
`pit_universe.sp500_tickers()` — not a hand-picked subset). If that fetch fails,
it falls back to a small hand-picked `FALLBACK_UNIVERSE` so the provider still
runs. Both data sources are free and require no account:

- **Signal**: SEC EDGAR's `submissions` API (`data.sec.gov/submissions/CIK...json`),
  filtered to `8-K` filings. A filing's `filingDate` is fixed at submission and
  never restated, so there is no look-ahead in the signal itself.
- **Returns**: Yahoo Finance daily closes via `yfinance`.

Before running it, open `edgar_provider.py` and replace the `SEC_USER_AGENT`
placeholder with your own name and email — SEC's fair-access policy requires a
real, identifying User-Agent on every request and can rate-limit or block generic
ones.

```powershell
pip install yfinance requests   # if not already installed
$env:ALTQ_DRY_RUN = "1"
$env:ALTQ_DATA_SOURCE = "edgar"
python run.py 3
```

Raw API responses are cached to `data_cache/` (git-ignored) so repeated runs don't
re-hit SEC/Yahoo every time — expect the first run against the full 503-name
universe to take several minutes (SEC fair-access sleep + ~500 filing-history
fetches), and be near-instant afterward while the cache is warm (SEC submissions
cache for 1 day, the ticker map and S&P 500 table for 7 days).

Against the earlier, hand-picked 25-name universe, this hypothesis came back
**REJECTed**: a raw annualized Sharpe of 1.48 but a Deflated Sharpe Ratio of 0.91 —
just under the 0.95 gate. Re-run against the full, systematic 503-name universe,
it's rejected far more decisively: annualized Sharpe of **-0.62**, DSR **0.014**,
PBO **0.4**. The hand-picked list happened to include several liquid, high-beta
names that flattered this particular signal; the systematic universe removes that
selection bias and shows the effect for what it is. That's the pipeline — and this
fix — doing exactly what they're for: most first-pass alt-data hypotheses don't
survive contact with a multiple-testing-aware statistical review, and a
non-cherry-picked universe makes that verdict harder to argue with.

### A second, independent real signal (Google Trends attention)

`trends_provider.py` tests a different hypothesis, from a different data source,
with a different economic mechanism, so Discovery has more than one real feed to
draw from:

- **Signal**: Google Trends search-interest index via `pytrends` (unofficial),
  turned into an *attention shock* — current interest minus the name's own
  trailing baseline, in units of its own trailing standard deviation — for 12
  consumer-facing brand names with unambiguous search terms (Tesla, Nvidia,
  Netflix, Starbucks, Nike, Boeing, Disney, Costco, Uber, PayPal, Airbnb,
  Palantir).
- **Returns**: Yahoo Finance daily closes via `yfinance`, same as the EDGAR provider.
- **Mechanism**: a search-interest spike proxies a retail-attention shock, which
  the attention literature (Da, Engelberg & Gao, 2011) associates with
  short-horizon price pressure.

```powershell
pip install pytrends lxml   # if not already installed
$env:ALTQ_DRY_RUN = "1"
$env:ALTQ_DATA_SOURCE = "trends"
python run.py 3
```

`pytrends` is an **unofficial** scraper, not a supported API — it rate-limits
aggressively (HTTP 429) under repeated testing, and the provider retries with
exponential backoff (up to ~3 minutes total) rather than failing outright. If it
still fails, wait a few minutes before rerunning; this is a known limitation of
the free tier, documented in the module docstring, not a bug to chase.

On a first real run (default 12-name universe, 3-year lookback), this hypothesis
also came back **REJECTed** — an annualized Sharpe of 0.56 and a Deflated Sharpe
Ratio of 0.24, well short of the 0.95 gate. Two-for-two real-data rejections is
the expected base rate for first-pass alt-data ideas, not a sign anything is
broken — see the Skeptic's whole design intent above.

### A third, independent real signal (Wikipedia pageview attention)

`wiki_provider.py` tests the same attention-shock mechanism as Trends, but
from a genuinely different, official data source — Wikimedia's Pageviews REST
API, not an unofficial scraper — and across the full systematic S&P 500
universe rather than a hand-picked list:

- **Signal**: daily Wikipedia article pageviews (`wikimedia.org/api/rest_v1/metrics/pageviews`,
  free, no key), turned into an attention shock the same way as Trends — current
  views minus the article's own trailing 8-period mean, in units of its own
  trailing standard deviation.
- **Universe**: every current S&P 500 ticker, resolved to its Wikipedia article
  automatically from the same SEC legal names `edgar_provider.py` already
  caches (`edgar_provider.ticker_to_legal_name()`) — no hand-curated query list,
  unlike Trends. Getting this resolution right took real iteration: an
  opensearch-based first attempt silently mapped "JPMorgan Chase & Co." to a
  near-dead redirect stub instead of the real article, an `intitle:`-restricted
  search sent "BOEING CO" to the "Boeing Model 42" article, and SEC's own data
  quality added a third failure mode — ~8% of legal names carry a junk trailing
  state-of-incorporation tag ("IDEX CORP /DE/") that diluted search relevance
  enough to match a wholly unrelated page. All three are fixed in
  `wiki_provider.py`; read its module docstring if you extend the resolution
  logic further.
- **Returns**: Yahoo Finance daily closes via `yfinance`, same as the other two providers.

```powershell
$env:ALTQ_DRY_RUN = "1"
$env:ALTQ_DATA_SOURCE = "wiki"
python run.py 3
```

502 of 503 tickers resolved correctly on a full run. This hypothesis also came
back **REJECTed** — annualized Sharpe of 0.65, DSR 0.25, PBO 0.63 — the third
real hypothesis in a row to fail the gate, and, like the systematic-universe
result above, a more decisive rejection than a hand-picked list would likely
have produced.

### Fixing forward-survivorship bias (`pit_universe.py`)

All three real providers now mask out a ticker's history from before it
actually joined the S&P 500 — sourced from Wikipedia's current-membership table
(free, no key). This matters concretely for the Trends universe: Palantir
(added 2024-09-23) and Uber (added 2023-12-18) both fall inside the default
3-year lookback, so without the mask the pipeline would credit them with
"large, liquid, index-covered" status for a stretch of history when that
wasn't true yet.

**What this does not fix**: backward survivorship. A company removed from the
S&P 500 before today (bankruptcy, acquisition, delisting) is absent from
Wikipedia's *current* membership table and can never appear in any of these
universes —
there's no free source for full point-in-time index history. Any DSR/CSCV result
here still overstates achievable Sharpe to the extent failed or acquired names
would have dragged on it. Fixing that properly needs a licensed dataset (WRDS/CRSP
or a paid Nasdaq Data Link constituents table) — out of scope for a free-data
prototype, and flagged as an open item below rather than glossed over.

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `ALTQ_DRY_RUN` | `1` | `1` = stub hypothesis + rule-based Skeptic verdict, no API cost. `0` = live `claude-opus-5` calls. |
| `ALTQ_DATA_SOURCE` | `mock` | `mock` = synthetic `DataProvider`. `edgar` = SEC EDGAR + Yahoo Finance (`EdgarFilingCadenceProvider`). `trends` = Google Trends + Yahoo Finance (`GoogleTrendsAttentionProvider`). `wiki` = Wikipedia pageviews + Yahoo Finance (`WikipediaAttentionProvider`). |
| `ALTQ_TRUE_BETA` | `0.0` | Strength of the true signal baked into the mock `DataProvider`. Only affects `ALTQ_DATA_SOURCE=mock` runs. |
| `ANTHROPIC_API_KEY` | — | Required when `ALTQ_DRY_RUN=0`. |

`run.py <max_trials>` — trial budget before the graph routes to `abort` instead of
back to `discovery`. Defaults to 3.

## Current state vs. production

This is a working, statistically-verified scaffold, not a production trading system.
What's real:

- The graph wiring, conditional routing, and trial-penalty feedback loop.
- The DSR and CSCV math, verified against synthetic ground-truth panels.
- The Pydantic schemas and prompts for both LLM agents.

What's still a placeholder:

- **Backward survivorship bias is not fixed, and can't be with free data.**
  `pit_universe.py` masks out a ticker's history before it joined the S&P 500
  (forward bias — fixed), but a company removed from the index before today
  (bankruptcy, acquisition, delisting) is simply absent from all three real
  providers' universes and always will be without a licensed historical-
  constituents dataset. Treat every real DSR/PBO result above as an upper bound
  on what a truly survivorship-free study would show, not the final word.
- `edgar_provider.py` and `wiki_provider.py` both use a systematic universe (the
  full current S&P 500, ~503 names via `pit_universe.sp500_tickers()`) instead of
  a hand-picked list. `trends_provider.py` still can't do this: it needs a
  curated, unambiguous brand-name search query per ticker (`TICKER_TO_QUERY`),
  since Google Trends has no equivalent of Wikipedia's exact-title/redirect
  lookup to disambiguate a bare brand name automatically — the 12-name universe
  there stays a deliberate, disclosed limitation, not a placeholder to "fix" later.
- None of the three universes is a *liquidity-floor-adjusted* research
  universe — "all current S&P 500 members" and "12 unambiguous brand names" are
  both blunter than a real desk would use, just no longer hand-tuned to flatter
  any one hypothesis.
- The default `DataProvider` in `graph.py` (used when `ALTQ_DATA_SOURCE=mock`) is
  still fully synthetic — useful for fast iteration on the graph/statistics
  themselves, not for research conclusions.
- No persistence/checkpointing between runs (LangGraph supports a `SqliteSaver`
  checkpointer; not yet wired in).
- No live kill-switch monitor — the kill-switch thresholds are attached to the
  execution payload but nothing currently watches live P&L against them.
- No paper-trading or execution/OMS integration.

### Path to production, in order

1. ~~Replace `DataProvider.load()` with a real data source~~ — done for one
   hypothesis via `edgar_provider.py` (SEC EDGAR + Yahoo Finance).
   ~~Fix forward-survivorship bias~~ — done via `pit_universe.py`, wired into all
   real providers. ~~Wire in a second, independent real signal~~ — done via
   `trends_provider.py` (Google Trends attention shocks). ~~Systematic,
   historically-accurate universe~~ — done for EDGAR via
   `pit_universe.sp500_tickers()` (full 503-name S&P 500 instead of a hand-picked
   25); re-run against that universe, the 8-K-cadence hypothesis is REJECTed far
   more decisively (SR_ann -0.62, DSR 0.014) than it was on the hand-picked list
   (SR_ann 1.48, DSR 0.91) — a concrete demonstration of how much a hand-picked
   universe can flatter a result. ~~Wire in a third, independent real signal~~ —
   done via `wiki_provider.py` (Wikipedia pageview attention shocks). AIS
   shipping and workforce data were investigated first and found to be
   paid/enterprise-only (see project notes); Wikipedia pageviews turned out to
   be a genuinely free, official-API alternative that also let the universe be
   systematic rather than hand-picked, unlike Trends. All three real hypotheses
   tested so far have been correctly REJECTed by the Skeptic gate. Trends'
   universe stays hand-picked by necessity (see above). Next: a fourth signal if
   one turns up, or move on to the items below — three independent real,
   statistically-gated hypotheses is a reasonable place to call this phase done.
2. Re-validate DSR/PBO against synthetic panels built from *that* data source's
   actual noise characteristics.
3. Turn on live LLM calls (`ALTQ_DRY_RUN=0`) and add server-side refusal fallbacks
   (`fallbacks="default"` on `claude-fable-5-1`/`claude-opus-5` calls) so a policy
   decline doesn't hard-stop the loop.
4. Add a LangGraph checkpointer for durability and an audit trail of every rejected
   hypothesis.
5. Build the live kill-switch monitor (scheduled job that recomputes rolling DSR
   against live returns).
6. Paper-trade for at least as long as the hypothesis's stated decay horizon before
   committing capital.

## Notes

- `scipy` was intentionally avoided: on this machine, a Windows Application Control
  policy blocks `scipy`'s compiled `_ansari_swilk_statistics` DLL when installed
  under a temp directory. `skeptic_stats.py` reimplements skewness, kurtosis, and
  the normal CDF/inverse-CDF using `numpy` and the stdlib `statistics.NormalDist`
  instead, with no loss of accuracy for this use case.
- The project was originally scaffolded in a session temp directory and moved to
  `C:\Users\marwa\projects\altq` for persistence; the git history starts from that
  move.
