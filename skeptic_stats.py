"""Multiple-testing-aware validation: Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)
and CSCV / Probability of Backtest Overfitting (Bailey, Borwein, Lopez de Prado, Zhu 2017)."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
import numpy as np
from statistics import NormalDist

_N = NormalDist()


def _skew_kurt(r: np.ndarray) -> tuple[float, float]:
    d = r - r.mean(); m2 = (d ** 2).mean()
    return float((d ** 3).mean() / m2 ** 1.5), float((d ** 4).mean() / m2 ** 2)   # kurt non-excess

EULER_GAMMA = 0.5772156649015329


@dataclass(frozen=True)
class DSRResult:
    sharpe_hat: float          # per-period SR of selected strategy
    sharpe_benchmark: float    # SR0: expected max SR under H0 across n_trials
    dsr: float                 # P[true SR > 0 | n_trials, non-normality]
    n_trials: int
    passed: bool


def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int,
    var_sharpe_trials: float,
    threshold: float = 0.95,
) -> DSRResult:
    """returns: 1-D per-period returns of the *selected* strategy.
    n_trials: total strategies/hypotheses tried (incl. discarded). Never under-count.
    var_sharpe_trials: variance of per-period SR across all trials."""
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    T = len(r)
    if T < 30 or r.std(ddof=1) == 0:
        return DSRResult(0.0, np.inf, 0.0, n_trials, False)
    sr = r.mean() / r.std(ddof=1)
    g3, g4 = _skew_kurt(r)
    N = max(int(n_trials), 1)
    sd_trials = np.sqrt(max(var_sharpe_trials, 1e-12))
    if N == 1:
        sr0 = 0.0
    else:
        sr0 = sd_trials * ((1 - EULER_GAMMA) * _N.inv_cdf(1 - 1 / N)
                           + EULER_GAMMA * _N.inv_cdf(1 - 1 / (N * np.e)))
    denom = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    z = (sr - sr0) * np.sqrt(T - 1) / denom
    dsr = float(_N.cdf(float(z)))
    return DSRResult(float(sr), float(sr0), dsr, N, dsr >= threshold)


@dataclass(frozen=True)
class CSCVResult:
    pbo: float                    # probability of backtest overfitting
    logits: np.ndarray            # lambda_c per combination
    oos_perf_degradation: float   # slope of OOS vs IS performance
    n_combinations: int
    passed: bool


def _sharpe_cols(M: np.ndarray) -> np.ndarray:
    mu, sd = M.mean(0), M.std(0, ddof=1)
    return np.where(sd > 0, mu / np.where(sd > 0, sd, 1), -np.inf)


def cscv_pbo(returns_matrix: np.ndarray, S: int = 16, pbo_max: float = 0.20) -> CSCVResult:
    """returns_matrix: T x N, column n = per-period returns of trial n (ALL trials, not survivors).
    S: number of contiguous blocks (even). C(S, S/2) train/test partitions."""
    M = np.asarray(returns_matrix, float)
    T, N = M.shape
    assert S % 2 == 0 and N >= 2 and T >= S * 5, "need even S, >=2 trials, >=5 obs/block"
    M = M[: (T // S) * S]
    blocks = np.array_split(M, S, axis=0)
    idx = range(S)
    logits, is_perf, oos_perf = [], [], []
    for train in combinations(idx, S // 2):
        test = tuple(i for i in idx if i not in train)
        J = np.vstack([blocks[i] for i in train])
        Jc = np.vstack([blocks[i] for i in test])
        sr_is, sr_oos = _sharpe_cols(J), _sharpe_cols(Jc)
        n_star = int(np.argmax(sr_is))
        rank = (sr_oos < sr_oos[n_star]).sum() + 1          # 1..N, 1 = worst
        omega = rank / (N + 1)
        logits.append(np.log(omega / (1 - omega)))
        is_perf.append(sr_is[n_star]); oos_perf.append(sr_oos[n_star])
    lam = np.array(logits)
    pbo = float((lam <= 0).mean())
    slope = float(np.polyfit(is_perf, oos_perf, 1)[0]) if len(is_perf) > 2 else np.nan
    return CSCVResult(pbo, lam, slope, len(lam), pbo <= pbo_max)


def haircut_trials(n_trials_declared: int, n_rejected_feedback: int, penalty: float) -> int:
    """Trial count grows with every Discovery re-entry; penalty >= 1 inflates it super-linearly
    so a looping Discovery agent cannot 'launder' its search via re-labeling."""
    return int(np.ceil(n_trials_declared * (1 + penalty * n_rejected_feedback)))
