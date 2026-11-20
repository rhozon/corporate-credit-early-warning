import numpy as np
import pandas as pd
import pytest

from distress import macro, panel


def feats(cd, dates, x=1.0):
    return pd.DataFrame({"cd_cvm": cd, "as_of": pd.to_datetime(dates),
                         "dt_refer": pd.to_datetime(dates) - pd.Timedelta(days=40), "x": x})


def cad(cd, cancel=None, sector="Comércio"):
    return {"cd_cvm": cd, "cancel_date": pd.Timestamp(cancel) if cancel else pd.NaT, "sector": sector}


def ev(cd, date, end, prior=0, truncated=False):
    return {"cd_cvm": cd, "event_date": pd.Timestamp(date), "event_type": "RJ", "episode_end": pd.Timestamp(end),
            "n_prior_events": prior, "left_truncated": truncated}


QUARTERLY = pd.date_range("2014-05-10", "2019-11-10", freq="3MS") + pd.Timedelta(days=9)
MACRO = pd.DataFrame({"selic": np.arange(200.0)}, index=pd.date_range("2008-01-01", periods=200, freq="MS"))
EMPTY_EVENTS = pd.DataFrame(columns=["cd_cvm", "event_date", "event_type", "episode_end", "n_prior_events", "left_truncated"])


def build(f, e, c, **kw):
    return panel.build_panel(f, e, pd.DataFrame(c), MACRO, start="2015-03-31", end="2019-12-31", lags={"selic": 2}, **kw)


def test_features_are_never_newer_than_the_quarter():
    p = build(feats(1, QUARTERLY), EMPTY_EVENTS, [cad(1)])
    assert (p["feature_as_of"] <= p["q"]).all()
    assert p["info_age_days"].min() >= 0


def test_y12_marks_exactly_the_year_before_the_event():
    p = build(feats(1, QUARTERLY), pd.DataFrame([ev(1, "2018-08-15", "2019-06-30")]), [cad(1)])
    flagged = p.loc[p["y12"] == 1, "q"].dt.strftime("%Y-%m").tolist()
    assert flagged == ["2017-09", "2017-12", "2018-03", "2018-06"]
    assert p.loc[p["q"] == pd.Timestamp("2018-06-30"), "event"].item() == 1


def test_rows_inside_an_episode_are_dropped_and_resume_after_exit():
    p = build(feats(1, QUARTERLY), pd.DataFrame([ev(1, "2016-08-15", "2017-06-30")]), [cad(1)])
    qs = p["q"]
    assert not ((qs > pd.Timestamp("2016-08-15")) & (qs <= pd.Timestamp("2017-06-30"))).any()
    after = p[qs > pd.Timestamp("2017-06-30")]
    assert len(after) > 0 and (after["prior_event"] == 1).all()


def test_cancelled_firm_is_censored_at_cancellation():
    p = build(feats(2, QUARTERLY), EMPTY_EVENTS, [cad(2, cancel="2017-05-01")])
    assert p["q"].max() <= pd.Timestamp("2017-05-01")
    assert p.iloc[-1]["censored_reason"] == "cancelled"
    assert p["event"].sum() == 0
    assert (p["cancel_date"] == pd.Timestamp("2017-05-01")).all()


def test_left_truncated_and_excluded_firms_are_out():
    e = pd.DataFrame([ev(3, "2010-02-01", "2016-01-01", truncated=True)])
    p = build(pd.concat([feats(3, QUARTERLY), feats(4, QUARTERLY)]), e, [cad(3), cad(4)], exclude={4})
    assert p.empty


def test_macro_uses_publication_lag():
    p = build(feats(1, QUARTERLY), EMPTY_EVENTS, [cad(1)])
    row = p[p["q"] == pd.Timestamp("2016-06-30")].iloc[0]
    # June 2016 minus a 2-month lag is April 2016
    assert row["selic"] == MACRO.loc["2016-04-01", "selic"]


def test_counting_process_intervals():
    p = build(feats(1, QUARTERLY), EMPTY_EVENTS, [cad(1)])
    assert (p["stop"] - p["start"] == 3).all()
    assert p["start"].iloc[0] == 0


def test_split_embargoes_2019():
    q = pd.date_range("2015-03-31", "2024-12-31", freq="QE")
    p = pd.DataFrame({"q": q, "y12": 0})
    train, test = panel.split(p, data_end=pd.Timestamp("2026-11-20"))
    assert train["q"].max() == pd.Timestamp("2018-12-31")
    assert test["q"].min() == pd.Timestamp("2020-03-31")
    assert test["y12_complete"].all()


def test_build_macro_transforms_and_names():
    idx = pd.date_range("2008-01-01", periods=60, freq="MS")
    fake = {432: pd.Series(10.0, idx), 433: pd.Series(0.5, idx), 24363: pd.Series(np.linspace(100, 160, 60), idx),
            24369: pd.Series(8.0, idx), 21083: pd.Series(3.0, idx), 20783: pd.Series(20.0, idx)}
    m = macro.build_macro(lambda code, start: fake[code])
    assert list(m.columns) == ["selic", "ipca12", "ibc_yoy", "unemp", "pj_default", "spread"]
    assert m["ipca12"].iloc[-1] == pytest.approx((1.005 ** 12 - 1) * 100)
    assert m["ibc_yoy"].iloc[-1] == pytest.approx((160 / fake[24363].iloc[-13] - 1) * 100)
