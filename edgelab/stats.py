"""Statistics for deciding whether a result is evidence or noise."""
from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import numpy as np
from scipy import stats as st

EULER_GAMMA = 0.5772156649015329


def newey_west_tstat(x: Sequence[float], lags: int = 3) -> Tuple[float, float]:
    """Mean and HAC (Bartlett kernel) t-statistic for H0: mean = 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < lags + 2:
        raise ValueError("not enough observations")
    mu = x.mean()
    e = x - mu
    lrv = e @ e / n
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        lrv += 2 * w * (e[k:] @ e[:-k]) / n
    se = math.sqrt(lrv / n) if lrv > 0 else float("nan")
    return float(mu), float(mu / se) if se and se > 0 else float("nan")


def sharpe(x: Sequence[float]) -> float:
    """Per-period Sharpe ratio (mean / std, ddof=1). Not annualised."""
    x = np.asarray(x, dtype=float)
    sd = x.std(ddof=1)
    return float(x.mean() / sd) if sd > 0 else float("nan")


def probabilistic_sharpe(x: Sequence[float], sr_benchmark: float = 0.0) -> float:
    """PSR (Bailey & Lopez de Prado): P(true per-period SR > benchmark), adjusting for skew/kurtosis."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    sr = sharpe(x)
    g3 = st.skew(x)
    g4 = st.kurtosis(x, fisher=False)
    denom = math.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    return float(st.norm.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / denom))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """Expected maximum per-period SR among n_trials skill-less strategies (False Strategy Theorem)."""
    if n_trials < 1:
        raise ValueError("n_trials >= 1")
    if n_trials == 1:
        return 0.0
    z1 = st.norm.ppf(1 - 1 / n_trials)
    z2 = st.norm.ppf(1 - 1 / (n_trials * math.e))
    return float(math.sqrt(var_sr) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe(x: Sequence[float], n_trials: int, var_sr_across_trials: float) -> float:
    """DSR: PSR against the SR that the best of n_trials noise strategies would be expected to show."""
    return probabilistic_sharpe(x, expected_max_sharpe(n_trials, var_sr_across_trials))


def stationary_bootstrap_ci(x: Sequence[float], stat=np.mean, n_boot: int = 5000,
                            mean_block: float = 5.0, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    """Politis-Romano stationary bootstrap CI (keeps serial dependence)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    rng = np.random.default_rng(seed)
    p = 1 / mean_block
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n, dtype=int)
        i = rng.integers(n)
        for t in range(n):
            if t > 0 and rng.random() < p:
                i = rng.integers(n)
            idx[t] = i
            i = (i + 1) % n
        out[b] = stat(x[idx])
    return float(np.quantile(out, alpha / 2)), float(np.quantile(out, 1 - alpha / 2))


def concentration(x: Sequence[float], k: int = 3) -> Dict[str, float]:
    """How much the total depends on the k best and k worst observations."""
    x = np.sort(np.asarray(x, dtype=float))
    total = x.sum()
    return {
        "total": float(total),
        "total_without_best_k": float(x[:-k].sum()),
        "total_without_worst_k": float(x[k:].sum()),
        "worst": float(x[0]),
        "best": float(x[-1]),
    }
