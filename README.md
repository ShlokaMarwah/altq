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
predict short-horizon returns, across a fixed 25-name liquid large-cap universe.
Both data sources are free and require no account:

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
re-hit SEC/Yahoo every time. On a first real run against the default universe and a
3-year lookback, this hypothesis came back **REJECTed**: a raw annualized Sharpe of
1.48 but a Deflated Sharpe Ratio of 0.91 — just under the 0.95 gate once the honest
trial count is applied. That's the pipeline doing exactly what it's for: most
first-pass alt-data hypotheses, even ones with a plausible economic story, don't
survive contact with a multiple-testing-aware statistical review.

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

### Fixing forward-survivorship bias (`pit_universe.py`)

Both real providers now mask out a ticker's history from before it actually
joined the S&P 500 — sourced from Wikipedia's current-membership table (free, no
key). This matters concretely for the Trends universe: Palantir (added
2024-09-23) and Uber (added 2023-12-18) both fall inside the default 3-year
lookback, so without the mask the pipeline would credit them with "large,
liquid, index-covered" status for a stretch of history when that wasn't true yet.

**What this does not fix**: backward survivorship. A company removed from the
S&P 500 before today (bankruptcy, acquisition, delisting) is absent from
Wikipedia's *current* membership table and can never appear in either universe —
there's no free source for full point-in-time index history. Any DSR/CSCV result
here still overstates achievable Sharpe to the extent failed or acquired names
would have dragged on it. Fixing that properly needs a licensed dataset (WRDS/CRSP
or a paid Nasdaq Data Link constituents table) — out of scope for a free-data
prototype, and flagged as an open item below rather than glossed over.

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `ALTQ_DRY_RUN` | `1` | `1` = stub hypothesis + rule-based Skeptic verdict, no API cost. `0` = live `claude-opus-5` calls. |
| `ALTQ_DATA_SOURCE` | `mock` | `mock` = synthetic `DataProvider`. `edgar` = SEC EDGAR + Yahoo Finance (`EdgarFilingCadenceProvider`). `trends` = Google Trends + Yahoo Finance (`GoogleTrendsAttentionProvider`). |
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
  (bankruptcy, acquisition, delisting) is simply absent from both real providers'
  universes and always will be without a licensed historical-constituents dataset.
  Treat both real DSR/PBO results above as an upper bound on what a truly
  survivorship-free study would show, not the final word.
- Both real providers still use a small, hand-picked, fixed ticker list rather
  than a systematically-defined universe (e.g. "all S&P 500 members, historically
  accurate, above a liquidity floor") — fine for demonstrating the pipeline works
  on real data, not yet a real research universe.
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
   ~~Fix forward-survivorship bias~~ — done via `pit_universe.py`, wired into both
   real providers. ~~Wire in a second, independent real signal~~ — done via
   `trends_provider.py` (Google Trends attention shocks). Both real hypotheses
   tested so far were correctly REJECTed by the Skeptic gate. Next: a systematic,
   historically-accurate universe (fixes the hand-picked-list gap above), and/or
   a third signal from the data-sources list (AIS shipping, workforce data — see
   project notes) once you want more than two real feeds for Discovery to draw on.
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
