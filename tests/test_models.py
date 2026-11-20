import numpy as np
import pandas as pd
import pytest

from distress.models import MODELS
from distress.models.base import surv_target

FEATURES = ["x1", "x2"]


def synthetic_panel(n_firms=400, seed=0):
    """Quarterly rows; quarterly hazard sigmoid(-4.5 + 1.6 x1); x2 is noise."""
    rng = np.random.default_rng(seed)
    grid = pd.date_range("2011-03-31", "2018-12-31", freq="QE")
    rows = []
    for f in range(n_firms):
        base = rng.normal()
        event_q = None
        xs = []
        for i, q in enumerate(grid):
            x1 = base + rng.normal(scale=0.3)
            xs.append((q, x1, rng.normal()))
            if rng.random() < 1 / (1 + np.exp(4.5 - 1.6 * x1)):
                event_q = i
                break
        ev_date = grid[event_q] + pd.Timedelta(days=45) if event_q is not None else None
        for i, (q, x1, x2) in enumerate(xs):
            t = (ev_date - q).days / 30.44 if ev_date is not None else np.nan
            rows.append({"cd_cvm": f, "q": q, "x1": x1, "x2": x2, "t_to_event": t,
                         "event": int(ev_date is not None and t <= 3), "y12": int(ev_date is not None and t <= 12),
                         "cancel_date": pd.NaT})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def data():
    p = synthetic_panel()
    firms = p["cd_cvm"].unique()
    train, test = p[p["cd_cvm"].isin(firms[::2])], p[p["cd_cvm"].isin(firms[1::2])]
    return train, test


def c_index(risk, duration, event):
    from sksurv.metrics import concordance_index_censored
    return concordance_index_censored(event.astype(bool), duration, risk)[0]


def test_surv_target_censors_at_horizon_end_and_cap():
    rows = pd.DataFrame({"q": pd.to_datetime(["2018-03-31", "2018-03-31", "2015-03-31"]),
                         "t_to_event": [6.0, 30.0, 50.0], "cancel_date": [pd.NaT, pd.NaT, pd.NaT]})
    d, e = surv_target(rows, horizon_end=pd.Timestamp("2019-12-31"), cap=36)
    assert e.tolist() == [1, 0, 0]
    assert d[0] == pytest.approx(6.0)
    assert d[1] == pytest.approx(21.0, abs=0.1)  # censored at 2019-12-31
    assert d[2] == pytest.approx(36.0)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_model_ranks_risk_and_gives_monotone_pd(name, data):
    train, test = data
    m = MODELS[name](features=FEATURES, seed=2026, fast=True).fit(train, horizon_end=pd.Timestamp("2019-12-31"))
    risk = m.risk(test)
    d, e = surv_target(test, horizon_end=pd.Timestamp("2019-12-31"))
    assert c_index(risk, d, e) > 0.7
    pd_ = m.pd(test)
    assert list(pd_.columns) == ["pd12", "pd24", "pd36"]
    assert ((pd_.values >= 0) & (pd_.values <= 1)).all()
    assert (pd_["pd12"] <= pd_["pd24"] + 1e-9).all() and (pd_["pd24"] <= pd_["pd36"] + 1e-9).all()


@pytest.mark.parametrize("name", sorted(MODELS))
def test_model_is_deterministic_under_seed(name, data):
    train, test = data
    a = MODELS[name](features=FEATURES, seed=2026, fast=True).fit(train, horizon_end=pd.Timestamp("2019-12-31")).risk(test[:50])
    b = MODELS[name](features=FEATURES, seed=2026, fast=True).fit(train, horizon_end=pd.Timestamp("2019-12-31")).risk(test[:50])
    np.testing.assert_allclose(a, b, rtol=1e-6)


def test_altman_uses_its_own_ratios():
    assert MODELS["altman"](features=None).features == ["wc_ta", "re_ta", "ebit_ta", "eq_tl"]
