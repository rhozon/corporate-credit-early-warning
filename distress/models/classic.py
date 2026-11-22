"""Benchmarks from the bankruptcy literature: Altman's score and Shumway's hazard logit."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .base import Model, Prep, hazard_to_pd

RATIOS = ["wc_ta", "re_ta", "ebit_ta", "eq_tl"]
Z2 = np.array([6.56, 3.26, 6.72, 1.05])


class Altman(Model):
    """Altman (2005) emerging-market score, mapped to a quarterly hazard by a one-variable logit."""
    name = "altman"

    def __init__(self, features=None, seed=2026, fast=False):
        super().__init__(RATIOS, seed, fast)

    def _z(self, rows):
        return 3.25 + self.prep.transform(rows[RATIOS], scale=False) @ Z2

    def fit(self, train, horizon_end):
        self.prep = Prep().fit(train[RATIOS])
        z = self._z(train).reshape(-1, 1)
        self.clf = LogisticRegression(C=1e6, max_iter=1000).fit(z, train["event"])
        return self

    def risk(self, rows):
        return -self._z(rows)

    def pd(self, rows):
        return hazard_to_pd(self.clf.predict_proba(self._z(rows).reshape(-1, 1))[:, 1])


class HazardLogit(Model):
    """Discrete-time hazard (Shumway 2001): logit of a filing in the next quarter."""
    name = "logit_hazard"
    grid = (0.01, 0.1, 1.0)

    def fit(self, train, horizon_end):
        self.prep = Prep(self.keep_tails).fit(train[self.features])
        X, y = self.prep.transform(train[self.features]), train["event"].values
        C = 1.0 if self.fast else _temporal_cv_C(train, X, y, self.grid)
        self.C = C
        self.clf = LogisticRegression(C=C, max_iter=5000).fit(X, y)
        return self

    def risk(self, rows):
        return self.clf.decision_function(self.prep.transform(rows[self.features]))

    def pd(self, rows):
        return hazard_to_pd(self.clf.predict_proba(self.prep.transform(rows[self.features]))[:, 1])


def _temporal_cv_C(train, X, y, grid):
    """Choose C by log-loss on each of the last three training years, fitting on the years before."""
    from sklearn.metrics import log_loss

    years = pd.to_datetime(train["q"]).dt.year.values
    scores = {c: [] for c in grid}
    for yv in sorted(set(years))[-3:]:
        tr, va = years < yv, years == yv
        if y[tr].sum() == 0 or y[va].sum() == 0:
            continue
        for c in grid:
            m = LogisticRegression(C=c, max_iter=5000).fit(X[tr], y[tr])
            scores[c].append(log_loss(y[va], m.predict_proba(X[va])[:, 1], labels=[0, 1]))
    means = {c: np.mean(v) for c, v in scores.items() if v}
    return min(means, key=means.get) if means else 1.0
