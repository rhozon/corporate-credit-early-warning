"""Shared pieces of the model layer.

Every model answers two questions about a panel row: how risky is this firm
relative to others (``risk``), and what is the probability it files within 12, 24
and 36 months (``pd``). Survival models are trained on stacked landmarks: each
firm-quarter is a starting point whose outcome is the time to the next event,
censored at cancellation, at the end of the training information, or at 36 months.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (12, 24, 36)
MONTH = 30.44


def surv_target(rows: pd.DataFrame, horizon_end: pd.Timestamp, cap: float = 36.0) -> tuple[np.ndarray, np.ndarray]:
    q = pd.to_datetime(rows["q"])
    to_end = (pd.Timestamp(horizon_end) - q).dt.days / MONTH
    cancel = pd.to_datetime(rows["cancel_date"])
    to_cancel = ((cancel - q).dt.days / MONTH).fillna(np.inf)
    censor = np.minimum(np.minimum(to_end.values, to_cancel.values), cap)
    t_ev = rows["t_to_event"].astype(float).fillna(np.inf).values
    event = (t_ev <= censor).astype(int)
    duration = np.where(event == 1, t_ev, censor)
    return np.maximum(duration, 0.01), event


class Prep:
    """Winsorise at the 1st and 99th percentiles, impute the median, standardise. Fit on train only."""

    def fit(self, X: pd.DataFrame) -> "Prep":
        X = X.astype(float)
        self.lo, self.hi = X.quantile(0.01), X.quantile(0.99)
        Xc = X.clip(self.lo, self.hi, axis=1)
        self.med = Xc.median()
        Xc = Xc.fillna(self.med)
        self.mu, self.sd = Xc.mean(), Xc.std().replace(0, 1).fillna(1)
        return self

    def transform(self, X: pd.DataFrame, scale: bool = True) -> np.ndarray:
        Xc = X.astype(float).clip(self.lo, self.hi, axis=1).fillna(self.med)
        if scale:
            Xc = (Xc - self.mu) / self.sd
        return Xc.values.astype("float32")


def hazard_to_pd(p_quarter: np.ndarray, horizons=HORIZONS) -> pd.DataFrame:
    p = np.clip(np.asarray(p_quarter, float), 0, 1)
    return pd.DataFrame({f"pd{h}": 1 - (1 - p) ** (h / 3) for h in horizons})


def breslow(risk_train: np.ndarray, duration: np.ndarray, event: np.ndarray, times=HORIZONS) -> np.ndarray:
    """Baseline cumulative hazard at ``times`` for a log-risk score (Breslow estimator)."""
    order = np.argsort(duration)
    d, e, r = duration[order], event[order], np.exp(risk_train[order] - risk_train.max())
    at_risk = np.cumsum(r[::-1])[::-1]
    increments = np.where(e == 1, 1.0 / at_risk, 0.0)
    cum = np.cumsum(increments)
    out = []
    for t in times:
        i = np.searchsorted(d, t, side="right")
        out.append(cum[i - 1] if i else 0.0)
    return np.array(out) * np.exp(-risk_train.max())


class Model:
    name = "base"

    def __init__(self, features=None, seed: int = 2026, fast: bool = False):
        self.features, self.seed, self.fast = features, seed, fast

    def fit(self, train: pd.DataFrame, horizon_end: pd.Timestamp) -> "Model":
        raise NotImplementedError

    def risk(self, rows: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def pd(self, rows: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError
