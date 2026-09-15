"""LangGraph state machine: Discovery -> ETL/Feature -> Skeptic -> Portfolio, with penalized feedback."""
from __future__ import annotations
import json, os
from typing import Annotated, Literal, TypedDict
from operator import add
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from skeptic_stats import deflated_sharpe_ratio, cscv_pbo, haircut_trials
from prompts import DISCOVERY_SYSTEM, SKEPTIC_SYSTEM

MODEL = "claude-opus-5"
DRY_RUN = os.environ.get("ALTQ_DRY_RUN", "1") == "1"   # 1 = no API calls, deterministic stubs
DATA_SOURCE = os.environ.get("ALTQ_DATA_SOURCE", "mock")  # "mock" | "edgar"


# ---------- schemas ----------
class Hypothesis(BaseModel):
    id: str
    data_source: str
    transmission_mechanism: str
    target_universe: str
    lag_days: int = Field(ge=0, le=60)
    decay_days: int = Field(ge=1, le=250)
    signal_construction: str
    null_prediction: str
    confounders: list[str]
    lag_grid: list[int] = Field(default_factory=lambda: [1, 3, 5])
    window_grid: list[int] = Field(default_factory=lambda: [20, 60])


class HypothesisBatch(BaseModel):
    hypotheses: list[Hypothesis]
    declared_n_trials: int = Field(ge=1, description="Every variant the agent considered, incl. discarded")


class SkepticVerdict(BaseModel):
    verdict: Literal["ACCEPT", "REJECT", "REVISE"]
    fatal_flaw: str | None
    corrected_n_trials: int
    rejection_reason: str


class PipelineState(TypedDict, total=False):
    trial: int
    max_trials: int
    penalty: float                     # multiplier applied to trial count on each feedback loop
    n_rejections: int
    n_trials_total: int                # cumulative honest trial count (feeds DSR)
    sr_history: Annotated[list[float], add]   # per-period SR of every variant ever run
    rejected: Annotated[list[dict], add]
    hypotheses: list[dict]
    features: dict                     # {hyp_id: {"returns_matrix": [[...]], "variants": [...]}}
    validation: dict
    verdict: str
    portfolio: dict


# ---------- LLM plumbing ----------
def _llm(schema, system: str, user: str):
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.parse(
        model=MODEL, max_tokens=16000, thinking={"type": "adaptive"},
        system=system, messages=[{"role": "user", "content": user}], output_format=schema,
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("model refused")
    return resp.parsed_output


# ---------- data provider (swap for real vendor) ----------
class DataProvider:
    """Point-in-time interface. Mock: 20 names, 1500 days, alt-signal with tunable true alpha."""
    def __init__(self, n_names=20, T=1500, true_beta=float(os.environ.get("ALTQ_TRUE_BETA", "0.0")), seed=7):
        rng = np.random.default_rng(seed)
        self.rets = pd.DataFrame(rng.normal(0, 0.015, (T, n_names)),
                                 columns=[f"TK{i:02d}" for i in range(n_names)])
        fut = self.rets.shift(-3).fillna(0)                          # signal leads returns by 3d
        sig = true_beta * fut.values / 0.015 + rng.normal(0, 1.0, (T, n_names))
        self.signal = pd.DataFrame(sig, columns=self.rets.columns)

    def load(self, hyp: Hypothesis) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.signal, self.rets


def _build_provider():
    if DATA_SOURCE == "edgar":
        from edgar_provider import EdgarFilingCadenceProvider
        return EdgarFilingCadenceProvider()
    return DataProvider()


PROVIDER = _build_provider()


def _variant_returns(sig: pd.DataFrame, rets: pd.DataFrame, lag: int, window: int) -> np.ndarray:
    z = (sig - sig.rolling(window).mean()) / sig.rolling(window).std()
    rank = z.rank(axis=1, pct=True)
    w = ((rank > 0.8).astype(float) - (rank < 0.2).astype(float))
    w = w.div(w.abs().sum(axis=1), axis=0).fillna(0)
    pnl = (w.shift(lag) * rets).sum(axis=1)                       # signal known at t, traded t+lag
    return pnl.iloc[window + lag:].to_numpy()


# ---------- nodes ----------
def discovery_node(state: PipelineState) -> dict:
    trial = state.get("trial", 0) + 1
    if DRY_RUN and DATA_SOURCE == "edgar":
        batch = HypothesisBatch(hypotheses=[Hypothesis(
            id=f"H{trial}",
            data_source="SEC EDGAR submissions API: 90d trailing count of 8-K filings per issuer, free, updated on filing",
            transmission_mechanism="Elevated unscheduled 8-K disclosure cadence signals operational/financial uncertainty -> slow diffusion to inattentive investors -> short-horizon negative drift",
            target_universe="Liquid large-cap US equities (fixed 25-name universe, see edgar_provider.DEFAULT_UNIVERSE)",
            lag_days=3, decay_days=20,
            signal_construction="90d trailing count of 8-K filings, cross-sectional z-score/quintile rank",
            null_prediction="IC(3d) <= 0.01 or DSR < 0.95 => reject",
            confounders=["earnings-announcement clustering", "size", "volatility regime"])],
            declared_n_trials=6)
    elif DRY_RUN:
        batch = HypothesisBatch(hypotheses=[Hypothesis(
            id=f"H{trial}", data_source="Mock: port-berth AIS dwell times, daily, 2018-",
            transmission_mechanism="Berth congestion -> delayed COGS recognition -> negative EPS surprise for importers",
            target_universe="US-listed importers, ADV > $20M", lag_days=3, decay_days=20,
            signal_construction="60d z-score of dwell time, cross-sectional quintile L/S",
            null_prediction="IC(3d) <= 0.01 or DSR < 0.95 => reject",
            confounders=["oil beta", "size"])], declared_n_trials=6)
    else:
        user = json.dumps({"rejected": state.get("rejected", []),
                           "trials_remaining": state["max_trials"] - trial + 1,
                           "penalty_multiplier": state.get("penalty", 1.0)})
        batch = _llm(HypothesisBatch, DISCOVERY_SYSTEM, user)
    return {"trial": trial, "hypotheses": [h.model_dump() for h in batch.hypotheses],
            "n_trials_total": state.get("n_trials_total", 0) + batch.declared_n_trials}


def etl_node(state: PipelineState) -> dict:
    feats, srs = {}, []
    for hd in state["hypotheses"]:
        h = Hypothesis(**hd)
        sig, rets = PROVIDER.load(h)
        variants, cols = [], []
        for lag in h.lag_grid:
            for win in h.window_grid:
                r = _variant_returns(sig, rets, lag, win)
                variants.append({"lag": lag, "window": win}); cols.append(r)
        L = min(len(c) for c in cols)
        M = np.column_stack([c[-L:] for c in cols])
        srs += list(M.mean(0) / M.std(0, ddof=1))
        feats[h.id] = {"returns_matrix": M.tolist(), "variants": variants}
    return {"features": feats, "sr_history": srs}


def skeptic_node(state: PipelineState) -> dict:
    hid, f = next(iter(state["features"].items()))
    M = np.asarray(f["returns_matrix"])
    n_trials = haircut_trials(state["n_trials_total"] + M.shape[1],
                              state.get("n_rejections", 0), state.get("penalty", 1.0))
    var_sr = float(np.var(state["sr_history"], ddof=1)) if len(state["sr_history"]) > 1 else 1e-4
    best = int(np.argmax(M.mean(0) / M.std(0, ddof=1)))
    dsr = deflated_sharpe_ratio(M[:, best], n_trials, var_sr)
    cscv = cscv_pbo(M, S=8)
    stats = {"hypothesis": hid, "best_variant": f["variants"][best], "n_trials_effective": n_trials,
             "sharpe_per_period": dsr.sharpe_hat, "sharpe_annualized": dsr.sharpe_hat * np.sqrt(252),
             "sr_benchmark_under_H0": dsr.sharpe_benchmark, "dsr": dsr.dsr, "pbo": cscv.pbo,
             "oos_degradation_slope": cscv.oos_perf_degradation,
             "stat_pass": bool(dsr.passed and cscv.passed)}
    if DRY_RUN or not stats["stat_pass"]:
        v = SkepticVerdict(verdict="ACCEPT" if stats["stat_pass"] else "REJECT",
                           fatal_flaw=None if stats["stat_pass"] else f"DSR={dsr.dsr:.2f}, PBO={cscv.pbo:.2f}",
                           corrected_n_trials=n_trials,
                           rejection_reason="Fails DSR/PBO gate; effect indistinguishable from search noise.")
    else:
        v = _llm(SkepticVerdict, SKEPTIC_SYSTEM,
                 json.dumps({"hypothesis": state["hypotheses"][0], "stats": stats}))
    out = {"validation": {"stats": stats, "skeptic": v.model_dump()}, "verdict": v.verdict}
    if v.verdict != "ACCEPT":
        out.update({"rejected": [{"id": hid, "reason": v.rejection_reason, "flaw": v.fatal_flaw}],
                    "n_rejections": state.get("n_rejections", 0) + 1,
                    "penalty": state.get("penalty", 1.0) * 1.5,
                    "n_trials_total": max(state["n_trials_total"], v.corrected_n_trials)})
    return out


def portfolio_node(state: PipelineState) -> dict:
    s = state["validation"]["stats"]
    M = np.asarray(state["features"][s["hypothesis"]]["returns_matrix"])
    r = M[:, int(np.argmax(M.mean(0) / M.std(0, ddof=1)))]
    target_vol, realized = 0.10, float(r.std(ddof=1) * np.sqrt(252))
    lev = min(target_vol / max(realized, 1e-6), 3.0)
    return {"portfolio": {"strategy_id": s["hypothesis"], "variant": s["best_variant"],
                          "gross_leverage": round(lev, 3), "target_vol": target_vol,
                          "rebalance": "daily", "max_name_weight": 0.05,
                          "kill_switch": {"rolling_60d_dsr_below": 0.5, "drawdown": -0.08},
                          "evidence": {k: s[k] for k in ("dsr", "pbo", "n_trials_effective")}}}


# ---------- routing ----------
def route_after_skeptic(state: PipelineState) -> Literal["portfolio", "discovery", "abort"]:
    if state["verdict"] == "ACCEPT":
        return "portfolio"
    return "discovery" if state["trial"] < state["max_trials"] else "abort"


def abort_node(state: PipelineState) -> dict:
    return {"verdict": "ABORT", "portfolio": {"action": "NONE", "reason": "trial budget exhausted"}}


def build_graph():
    g = StateGraph(PipelineState)
    g.add_node("discovery", discovery_node)
    g.add_node("etl", etl_node)
    g.add_node("skeptic", skeptic_node)
    g.add_node("portfolio", portfolio_node)
    g.add_node("abort", abort_node)
    g.add_edge(START, "discovery")
    g.add_edge("discovery", "etl")
    g.add_edge("etl", "skeptic")
    g.add_conditional_edges("skeptic", route_after_skeptic,
                            {"portfolio": "portfolio", "discovery": "discovery", "abort": "abort"})
    g.add_edge("portfolio", END)
    g.add_edge("abort", END)
    return g.compile()
