"""Survival models on stacked landmarks: Cox, gradient-boosted Cox, random survival forest."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import HORIZONS, Model, Prep, breslow, hazard_to_pd, surv_target


class Cox(Model):
    """Cox proportional hazards with a ridge penalty (lifelines)."""
    name = "cox"

    def fit(self, train, horizon_end):
        from lifelines import CoxPHFitter

        self.prep = Prep().fit(train[self.features])
        d, e = surv_target(train, horizon_end)
        X = self.prep.transform(train[self.features])
        # A column left constant by winsorising carries nothing and makes the Newton steps singular.
        self.keep = [i for i in range(X.shape[1]) if np.ptp(X[:, i]) > 0]
        df = pd.DataFrame(X[:, self.keep], columns=[self.features[i] for i in self.keep])
        df["duration"], df["event"] = d, e
        self.cph = CoxPHFitter(penalizer=0.1).fit(df, "duration", "event")
        return self

    def _X(self, rows):
        X = self.prep.transform(rows[self.features])[:, self.keep]
        return pd.DataFrame(X, columns=[self.features[i] for i in self.keep])

    def risk(self, rows):
        return np.log(self.cph.predict_partial_hazard(self._X(rows)).values)

    def pd(self, rows):
        s = self.cph.predict_survival_function(self._X(rows), times=list(HORIZONS)).T.values
        return pd.DataFrame(1 - s, columns=[f"pd{h}" for h in HORIZONS])


class XgbPD12(Model):
    """Gradient-boosted classifier of a filing within 12 months."""
    name = "xgb_pd12"

    def fit(self, train, horizon_end):
        from xgboost import XGBClassifier

        self.prep = Prep().fit(train[self.features])
        X = self.prep.transform(train[self.features], scale=False)
        y = train["y12"].values
        self.clf = XGBClassifier(n_estimators=150 if self.fast else 400, max_depth=3, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                                 scale_pos_weight=1.0, random_state=self.seed, n_jobs=1, tree_method="hist")
        self.clf.fit(X, y)
        return self

    def _p12(self, rows):
        return self.clf.predict_proba(self.prep.transform(rows[self.features], scale=False))[:, 1]

    def risk(self, rows):
        return self._p12(rows)

    def pd(self, rows):
        p12 = self._p12(rows)
        return hazard_to_pd(1 - (1 - p12) ** 0.25)


class XgbCox(Model):
    """Gradient-boosted Cox model (XGBoost ``survival:cox``) with a Breslow baseline."""
    name = "xgb_cox"

    def fit(self, train, horizon_end):
        from xgboost import XGBRegressor

        self.prep = Prep().fit(train[self.features])
        X = self.prep.transform(train[self.features], scale=False)
        d, e = surv_target(train, horizon_end)
        self.reg = XGBRegressor(objective="survival:cox", n_estimators=150 if self.fast else 400, max_depth=3,
                                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                                random_state=self.seed, n_jobs=1, tree_method="hist")
        self.reg.fit(X, np.where(e == 1, d, -d))
        self.H0 = breslow(self._margin(X), d, e)
        return self

    def _margin(self, X):
        return self.reg.predict(X, output_margin=True)

    def risk(self, rows):
        return self._margin(self.prep.transform(rows[self.features], scale=False))

    def pd(self, rows):
        r = np.exp(self.risk(rows))[:, None]
        return pd.DataFrame(1 - np.exp(-self.H0[None, :] * r), columns=[f"pd{h}" for h in HORIZONS])


class RSF(Model):
    """Random survival forest (Ishwaran et al. 2008), scikit-survival."""
    name = "rsf"

    def fit(self, train, horizon_end):
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.util import Surv

        self.prep = Prep().fit(train[self.features])
        X = self.prep.transform(train[self.features], scale=False)
        d, e = surv_target(train, horizon_end)
        self.rsf = RandomSurvivalForest(n_estimators=100 if self.fast else 300, min_samples_leaf=20,
                                        max_features="sqrt", random_state=self.seed, n_jobs=-1)
        self.rsf.fit(X, Surv.from_arrays(e.astype(bool), d))
        return self

    def _X(self, rows):
        return self.prep.transform(rows[self.features], scale=False)

    def risk(self, rows):
        return self.rsf.predict(self._X(rows))

    def pd(self, rows):
        fns = self.rsf.predict_survival_function(self._X(rows))
        return pd.DataFrame([[1 - f(min(h, f.x[-1])) for h in HORIZONS] for f in fns],
                            columns=[f"pd{h}" for h in HORIZONS])
