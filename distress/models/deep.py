"""Neural survival models: DeepSurv (Katzman et al. 2018) and DeepHit (Lee et al. 2018), via pycox."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import HORIZONS, Model, Prep, surv_target

CUTS = np.arange(0, 39, 3, dtype="float32")


def _seed(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _net(n_in: int, n_out: int):
    import torchtuples as tt

    return tt.practical.MLPVanilla(n_in, [32, 32], n_out, batch_norm=True, dropout=0.1)


def _split_val(n: int, seed: int, frac: float = 0.15):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    k = max(1, int(n * frac))
    return idx[k:], idx[:k]


class DeepSurv(Model):
    name = "deepsurv"

    def fit(self, train, horizon_end):
        import torchtuples as tt
        from pycox.models import CoxPH

        _seed(self.seed)
        self.prep = Prep().fit(train[self.features])
        X = self.prep.transform(train[self.features])
        d, e = surv_target(train, horizon_end)
        d, e = d.astype("float32"), e.astype("float32")
        tr, va = _split_val(len(X), self.seed)
        self.m = CoxPH(_net(X.shape[1], 1), tt.optim.Adam(lr=1e-3))
        self.m.fit(X[tr], (d[tr], e[tr]), batch_size=256, epochs=15 if self.fast else 60, verbose=False,
                   callbacks=[tt.callbacks.EarlyStopping(patience=8)], val_data=(X[va], (d[va], e[va])))
        self.m.compute_baseline_hazards(X, (d, e))
        return self

    def risk(self, rows):
        return self.m.predict(self.prep.transform(rows[self.features])).ravel()

    def pd(self, rows):
        s = self.m.predict_surv_df(self.prep.transform(rows[self.features]))
        return _at_horizons(s)


class DeepHit(Model):
    name = "deephit"

    def fit(self, train, horizon_end):
        import torchtuples as tt
        from pycox.models import DeepHitSingle

        _seed(self.seed)
        self.prep = Prep().fit(train[self.features])
        X = self.prep.transform(train[self.features])
        d, e = surv_target(train, horizon_end)
        self.lab = DeepHitSingle.label_transform(CUTS)
        y = self.lab.fit_transform(d.astype("float32"), e.astype("float32"))
        tr, va = _split_val(len(X), self.seed)
        self.m = DeepHitSingle(_net(X.shape[1], self.lab.out_features), tt.optim.Adam(lr=1e-3),
                               alpha=0.2, sigma=0.1, duration_index=self.lab.cuts)
        self.m.fit(X[tr], (y[0][tr], y[1][tr]), batch_size=256, epochs=15 if self.fast else 60, verbose=False,
                   callbacks=[tt.callbacks.EarlyStopping(patience=8)], val_data=(X[va], (y[0][va], y[1][va])))
        return self

    def risk(self, rows):
        return self.pd(rows)["pd36"].values

    def pd(self, rows):
        return _at_horizons(self.m.predict_surv_df(self.prep.transform(rows[self.features])))


def _at_horizons(surv: pd.DataFrame) -> pd.DataFrame:
    """Survival curves (index = time) to cumulative PD at 12, 24 and 36 months."""
    idx = surv.index.values.astype(float)
    out = {}
    for h in HORIZONS:
        i = np.searchsorted(idx, h, side="right") - 1
        s = surv.iloc[max(i, 0)].values if i >= 0 else np.ones(surv.shape[1])
        out[f"pd{h}"] = 1 - s
    df = pd.DataFrame(out)
    df["pd24"] = np.maximum(df["pd24"], df["pd12"])
    df["pd36"] = np.maximum(df["pd36"], df["pd24"])
    return df
