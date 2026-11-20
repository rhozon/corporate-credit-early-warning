"""Inference for model comparisons: DeLong test, firm-clustered bootstrap, Holm correction."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata


def _components(y: np.ndarray, p: np.ndarray):
    """DeLong structural components from midranks (Sun and Xu, 2014)."""
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    r_all = rankdata(p)
    r_pos = rankdata(p[pos])
    r_neg = rankdata(p[neg])
    v10 = (r_all[pos] - r_pos) / n
    v01 = 1 - (r_all[neg] - r_neg) / m
    return v10.mean(), v10, v01


def delong(y, p1, p2) -> tuple[float, float, float]:
    """AUC difference, z statistic and one-sided p-value for H1: AUC(p1) > AUC(p2)."""
    y, p1, p2 = np.asarray(y), np.asarray(p1, float), np.asarray(p2, float)
    a1, v10_1, v01_1 = _components(y, p1)
    a2, v10_2, v01_2 = _components(y, p2)
    m, n = len(v10_1), len(v01_1)
    s10 = np.cov(np.vstack([v10_1, v10_2]))
    s01 = np.cov(np.vstack([v01_1, v01_2]))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n
    z = (a1 - a2) / np.sqrt(var) if var > 0 else 0.0
    return float(a1 - a2), float(z), float(1 - norm.cdf(z))


def holm(pvals) -> list[float]:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    k = len(p)
    adj = np.empty(k)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (k - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def cluster_bootstrap(fn: Callable[[pd.DataFrame], float], df: pd.DataFrame, by: str = "cd_cvm",
                      B: int = 1000, seed: int = 2026, level: float = 0.95) -> tuple[float, float, float]:
    """Percentile interval resampling whole firms, so a firm's quarters stay together."""
    est = float(fn(df))
    groups = df.groupby(by).indices
    keys = np.array(list(groups))
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(B):
        pick = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([groups[k] for k in pick])
        try:
            v = float(fn(df.iloc[idx]))
        except ValueError:
            continue
        if np.isfinite(v):
            stats.append(v)
    a = (1 - level) / 2
    lo, hi = np.percentile(stats, [100 * a, 100 * (1 - a)]) if stats else (np.nan, np.nan)
    return est, float(lo), float(hi)
