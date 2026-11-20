"""Fit every model on one sample, score all of them on another, in the same way.

``validation_split`` reproduces the test design inside the training years so that
choices can be made without looking at the test period: fit on quarters up to
2015, skip 2016 (embargo), validate on 2017 and 2018.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import evaluate, stats
from .models import MODELS
from .models.base import surv_target

ACCOUNTING = ["wc_ta", "re_ta", "ebit_ta", "eq_tl", "lev", "curr", "cash_ta", "int_cov", "margin", "sales_g",
              "log_size_real", "neg_eq", "loss_q4"]
DISCLOSURE = ["filing_delay_days", "n_restatements_4q", "info_age_days", "gc_flag", "prior_event"]
MACRO = ["selic", "ipca12", "ibc_yoy", "unemp", "pj_default", "spread"]
BASE = ACCOUNTING + DISCLOSURE + MACRO


def validation_split(train: pd.DataFrame):
    fit = train[train["q"] <= pd.Timestamp("2015-12-31")]
    val = train[(train["q"] >= pd.Timestamp("2017-01-01")) & (train["q"] <= pd.Timestamp("2018-12-31"))]
    return fit, val, pd.Timestamp("2016-12-31"), pd.Timestamp("2019-12-31")


def run(train: pd.DataFrame, test: pd.DataFrame, features: list[str], models: list[str], fit_end: pd.Timestamp,
        eval_end: pd.Timestamp, fast: bool = False, bootstrap: int = 0, seed: int = 2026):
    """Metrics per model on ``test`` and the predictions behind them.

    PD metrics use rows whose 12-month window is fully observed by ``eval_end``.
    """
    d_tr, e_tr = surv_target(train, fit_end)
    d_te, e_te = surv_target(test, eval_end)
    complete = (pd.to_datetime(test["q"]) + pd.DateOffset(months=12) <= eval_end).values
    rows, preds = [], []
    for name in models:
        m = MODELS[name](features=features, seed=seed, fast=fast).fit(train, horizon_end=fit_end)
        risk = m.risk(test)
        pdc = m.pd(test).reset_index(drop=True)
        row = {"model": name}
        row.update(evaluate.metrics(test["y12"].values[complete], pdc["pd12"].values[complete]))
        row.update(evaluate.surv_metrics(d_tr, e_tr, d_te, e_te, risk, pdc))
        if bootstrap:
            sub = test[complete].assign(_p=pdc["pd12"].values[complete])
            _, lo, hi = stats.cluster_bootstrap(lambda d: evaluate.metrics(d["y12"], d["_p"])["auc"], sub,
                                                B=bootstrap, seed=seed)
            row["auc_lo"], row["auc_hi"] = lo, hi
        rows.append(row)
        preds.append(pd.DataFrame({"model": name, "cd_cvm": test["cd_cvm"].values, "q": test["q"].values,
                                   "y12": test["y12"].values, "complete": complete, "risk": risk,
                                   **{c: pdc[c].values for c in ("pd12", "pd24", "pd36")}}))
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)
