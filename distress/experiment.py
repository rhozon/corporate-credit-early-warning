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
from .llm.signals import FEATURE_CATEGORIES
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
                                   "dur": d_te, "evt": e_te,
                                   **{c: pdc[c].values for c in ("pd12", "pd24", "pd36")}}))
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


REFERENCES = ("altman", "logit_hazard")
# Treated arm of H2, fixed before the test was opened: 12-month counts of each distress category read from
# material facts, and of any category. The company's own recovery announcements are never a feature.
SIGNALS = [f"sig_{c}_12m" for c in FEATURE_CATEGORIES] + ["sig_any_12m"]
PRIMARY_MODEL = "logit_hazard"


def reliable_signals(kappa: dict, threshold: float = 0.6) -> list[str]:
    """Signal columns of the categories on which the two language models agree (kappa at or above the
    threshold). The any-signal count is left out, since it mixes in the categories that are dropped."""
    return [f"sig_{c}_12m" for c in FEATURE_CATEGORIES if (kappa.get(c) or 0) >= threshold]


def _clustered(y, p1, p2, firms, B: int, seed: int) -> dict:
    """The firm-clustered paired test, reported beside DeLong (rows of one firm are not independent)."""
    c = stats.cluster_paired_auc_test(y, p1, p2, firms, B=B, seed=seed)
    return {"p_cluster": c["p_one_sided"], "diff_lo": c["lo"], "diff_hi": c["hi"]}


def ablation_table(base: pd.DataFrame, signals: pd.DataFrame, B: int = 1000, seed: int = 2026) -> pd.DataFrame:
    """Each model with and without the signals (one-sided DeLong, signals better). The primary model's
    row is the pre-registered test; Holm's correction covers the other, secondary, rows."""
    rows = []
    for m, g in base[base["complete"]].groupby("model", sort=False):
        s = signals[(signals["model"] == m) & signals["complete"]]
        diff, z, p = stats.delong(g["y12"].values, s["pd12"].values, g["pd12"].values)
        rows.append({"model": m, "role": "primary" if m == PRIMARY_MODEL else "secondary",
                     "auc_base": evaluate.metrics(g["y12"].values, g["pd12"].values)["auc"],
                     "auc_signals": evaluate.metrics(s["y12"].values, s["pd12"].values)["auc"],
                     "auc_diff": diff, "z": z, "p": p,
                     **_clustered(g["y12"].values, s["pd12"].values, g["pd12"].values, g["cd_cvm"].values, B, seed)})
    out = pd.DataFrame(rows)
    sec = out["role"] == "secondary"
    out["p_holm"], out["p_cluster_holm"] = np.nan, np.nan
    if sec.any():
        out.loc[sec, "p_holm"] = stats.holm(out.loc[sec, "p"])
        out.loc[sec, "p_cluster_holm"] = stats.holm(out.loc[sec, "p_cluster"])
    return out


def delong_table(preds: pd.DataFrame, references=REFERENCES, B: int = 1000, seed: int = 2026) -> pd.DataFrame:
    """Each model against each reference on the 12-month PD (one-sided: model better), Holm-adjusted.

    These are the secondary comparisons of H1; the rows of every model are the same test rows. DeLong
    treats rows as independent; the firm-clustered paired test beside it does not.
    """
    ok = preds[preds["complete"]]
    by = {m: g for m, g in ok.groupby("model", sort=False)}
    rows = []
    for m, g in by.items():
        if m in references:
            continue
        for ref in references:
            if ref not in by:
                continue
            diff, z, p = stats.delong(g["y12"].values, g["pd12"].values, by[ref]["pd12"].values)
            rows.append({"model": m, "versus": ref, "auc_diff": diff, "z": z, "p": p,
                         **_clustered(g["y12"].values, g["pd12"].values, by[ref]["pd12"].values,
                                      g["cd_cvm"].values, B, seed)})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = stats.holm(out["p"])
        out["p_cluster_holm"] = stats.holm(out["p_cluster"])
    return out


def h2_test(train: pd.DataFrame, test: pd.DataFrame, base: list[str], signals: list[str], fit_end: pd.Timestamp,
            eval_end: pd.Timestamp, model: str = "logit_hazard", seed: int = 2026, keep_tails=(),
            B: int = 1000) -> dict:
    """Pre-registered primary test of H2: the discrete-time hazard logit with and without the language-model
    signals, AUC of the 12-month PD on the out-of-time rows, one-sided DeLong test (signals better).
    The firm-clustered paired test is reported beside it."""
    complete = (pd.to_datetime(test["q"]) + pd.DateOffset(months=12) <= eval_end).values
    y = test["y12"].values[complete]
    pds = {}
    for arm, feats in (("base", base), ("signals", base + signals)):
        m = MODELS[model](features=feats, seed=seed, fast=False, keep_tails=keep_tails).fit(train, horizon_end=fit_end)
        pds[arm] = m.pd(test)["pd12"].values[complete]
    diff, z, p = stats.delong(y, pds["signals"], pds["base"])
    c = _clustered(y, pds["signals"], pds["base"], test["cd_cvm"].values[complete], B, seed)
    return {"model": model, "n": int(complete.sum()), "events": int(y.sum()),
            "auc_base": evaluate.metrics(y, pds["base"])["auc"],
            "auc_signals": evaluate.metrics(y, pds["signals"])["auc"],
            "auc_diff": diff, "z": z, "p_one_sided": p, "p_cluster_one_sided": c["p_cluster"],
            "diff_lo": c["diff_lo"], "diff_hi": c["diff_hi"],
            "signals": list(signals), "keep_tails": list(keep_tails)}


def calibration_by_horizon(preds: pd.DataFrame, eval_end: pd.Timestamp, horizons=(12, 24, 36)) -> pd.DataFrame:
    """Observed default rate against the mean PD at each horizon (H3), on rows whose window has closed by
    ``eval_end``. Rows are chosen by date alone, never by outcome: keeping an open window only when it ended
    in a filing would inflate the observed rate. As in ``y12``, a firm that leaves the registry inside the
    window without a filing counts as a non-event."""
    rows = []
    for m, g in preds.groupby("model", sort=False):
        for h in horizons:
            dur, evt = g["dur"].values, g["evt"].values
            valid = (pd.to_datetime(g["q"]) + pd.DateOffset(months=h) <= eval_end).values
            y = ((evt == 1) & (dur <= h)).astype(int)[valid]
            p = g[f"pd{h}"].values[valid]
            row = {"model": m, "horizon": h, "n": int(valid.sum()), "events": int(y.sum()),
                   "observed": float(y.mean()) if len(y) else np.nan, "mean_pd": float(p.mean()) if len(p) else np.nan}
            try:
                met = evaluate.metrics(y, p)
                row.update({k: met[k] for k in ("brier", "calib_intercept", "calib_slope")})
            except Exception:
                row.update({"brier": np.nan, "calib_intercept": np.nan, "calib_slope": np.nan})
            rows.append(row)
    return pd.DataFrame(rows)


def rolling(p: pd.DataFrame, years, models: list[str], features: list[str], eval_end: pd.Timestamp,
            fast: bool = False, seed: int = 2026):
    """Annual refit with a moving origin (robustness): to score year Y, fit on quarters up to the end of
    Y-2 with outcomes observed to the end of Y-1 (the embargo year), and test on the quarters of Y."""
    q = pd.to_datetime(p["q"])
    rows, folds = [], []
    for y in years:
        train = p[q <= pd.Timestamp(f"{y - 2}-12-31")]
        test = p[(q >= pd.Timestamp(f"{y}-01-01")) & (q <= pd.Timestamp(f"{y}-12-31"))]
        res, _ = run(train, test, features, models, fit_end=pd.Timestamp(f"{y - 1}-12-31"), eval_end=eval_end,
                     fast=fast, seed=seed)
        rows.append(res.assign(year=y))
        folds.append({"year": y, "train_last_q": pd.to_datetime(train["q"]).max(),
                      "test_first_q": pd.to_datetime(test["q"]).min(), "n_train": len(train), "n_test": len(test),
                      "train_events": int(train["event"].sum()), "test_events": int(test["event"].sum())})
    res = pd.concat(rows, ignore_index=True)
    return res[["year"] + [c for c in res.columns if c != "year"]], pd.DataFrame(folds)
