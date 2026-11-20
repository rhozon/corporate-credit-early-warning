"""Out-of-time metrics.

For the 12-month PD: AUC, area under the precision-recall curve (events are rare),
KS (the statistic Brazilian credit desks report), Brier score, and calibration
slope and intercept from a logistic recalibration of the outcome on logit(PD).

For survival: Harrell's and Uno's concordance, time-dependent AUC at 12, 24 and 35
months, and the integrated Brier score. 35 rather than 36 because outcomes are
censored at 36 months and the estimators need a time inside the follow-up.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

TIMES = (12.0, 24.0, 35.0)


def metrics(y, p) -> dict:
    y, p = np.asarray(y).astype(int), np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    fpr, tpr, _ = roc_curve(y, p)
    out = {"n": int(len(y)), "events": int(y.sum()), "auc": roc_auc_score(y, p),
           "pr_auc": average_precision_score(y, p), "ks": float(np.max(tpr - fpr)),
           "brier": brier_score_loss(y, p)}
    import statsmodels.api as sm

    logit = np.log(p / (1 - p))
    try:
        fit = sm.Logit(y, sm.add_constant(logit)).fit(disp=False)
        out["calib_intercept"], out["calib_slope"] = float(fit.params[0]), float(fit.params[1])
    except Exception:
        out["calib_intercept"] = out["calib_slope"] = np.nan
    return out


def _surv(e, d):
    from sksurv.util import Surv

    return Surv.from_arrays(np.asarray(e).astype(bool), np.asarray(d, float))


def surv_metrics(d_train, e_train, d_test, e_test, risk, pd_curve: pd.DataFrame) -> dict:
    from sksurv.metrics import (concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc,
                                integrated_brier_score)

    ytr, yte = _surv(e_train, d_train), _surv(e_test, d_test)
    risk = np.asarray(risk, float)
    out = {"c_harrell": concordance_index_censored(yte["event"], yte["time"], risk)[0],
           "c_uno": concordance_index_ipcw(ytr, yte, risk, tau=TIMES[-1])[0]}
    pdm = pd_curve[["pd12", "pd24", "pd36"]].values.astype(float)
    # Only horizons inside the test follow-up can be scored.
    t_max, t_min = float(np.max(yte["time"])), float(np.min(yte["time"]))
    ok = [k for k, t in enumerate(TIMES) if t_min <= t < t_max]
    for k, h in enumerate((12, 24, 36)):
        out[f"auc_t{h}"] = np.nan
    if ok:
        aucs, _ = cumulative_dynamic_auc(ytr, yte, pdm[:, ok], [TIMES[k] for k in ok])
        for k, a in zip(ok, np.atleast_1d(aucs)):
            out[f"auc_t{(12, 24, 36)[k]}"] = float(a)
    out["ibs"] = (float(integrated_brier_score(ytr, yte, 1 - pdm[:, ok], [TIMES[k] for k in ok]))
                  if len(ok) >= 2 else np.nan)
    return out
