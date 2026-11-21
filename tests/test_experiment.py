import pandas as pd

from distress import experiment
from test_models import synthetic_panel


def test_validation_split_mirrors_the_test_embargo():
    p = pd.DataFrame({"q": pd.date_range("2011-03-31", "2018-12-31", freq="QE")})
    fit, val, fit_end, val_end = experiment.validation_split(p)
    assert fit["q"].max() == pd.Timestamp("2015-12-31")
    assert val["q"].min() == pd.Timestamp("2017-03-31") and val["q"].max() == pd.Timestamp("2018-12-31")
    assert fit_end == pd.Timestamp("2016-12-31") and val_end == pd.Timestamp("2019-12-31")


def test_run_scores_every_model_on_the_same_rows():
    p = synthetic_panel(n_firms=200, seed=3)
    firms = p["cd_cvm"].unique()
    train, test = p[p["cd_cvm"].isin(firms[::2])], p[p["cd_cvm"].isin(firms[1::2])]
    models = ["logit_hazard", "cox", "xgb_pd12"]
    res, preds = experiment.run(train, test, ["x1", "x2"], models, fit_end=pd.Timestamp("2019-12-31"),
                                eval_end=pd.Timestamp("2019-12-31"), fast=True, bootstrap=0)
    assert res["model"].tolist() == models
    assert {"auc", "pr_auc", "ks", "brier", "c_harrell", "c_uno", "auc_t12", "ibs"} <= set(res.columns)
    assert (res["auc"] > 0.7).all()
    assert set(preds["model"]) == set(models) and len(preds) == 3 * len(test)


def fake_preds(seed=0, n=600):
    import numpy as np

    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.1).astype(int)
    good = y * 1.5 + rng.normal(size=n)
    weak = y * 0.3 + rng.normal(size=n)
    out = []
    for name, s in [("altman", weak), ("logit_hazard", weak + rng.normal(scale=0.5, size=n)), ("xgb_pd12", good),
                    ("cox", good + rng.normal(scale=0.3, size=n))]:
        out.append(pd.DataFrame({"model": name, "cd_cvm": np.arange(n) % 150, "y12": y, "complete": True,
                                 "pd12": 1 / (1 + np.exp(-s))}))
    return pd.concat(out, ignore_index=True)


def test_delong_table_compares_every_other_model_with_both_references():
    t = experiment.delong_table(fake_preds())
    assert len(t) == 4 and set(t["versus"]) == {"altman", "logit_hazard"}
    assert set(t["model"]) == {"xgb_pd12", "cox"}
    assert (t["p_holm"] >= t["p"] - 1e-12).all()
    assert (t.loc[t["model"] == "xgb_pd12", "auc_diff"] > 0).all()


def test_primary_h2_test_finds_an_informative_signal_and_not_a_noise_one():
    import numpy as np

    p = synthetic_panel(n_firms=300, seed=5)
    rng = np.random.default_rng(1)
    p["sig_good"] = (p["y12"] * rng.poisson(2, len(p))) + rng.poisson(0.2, len(p))
    p["sig_noise"] = rng.poisson(0.5, len(p))
    firms = p["cd_cvm"].unique()
    train, test = p[p["cd_cvm"].isin(firms[::2])], p[p["cd_cvm"].isin(firms[1::2])]
    end = pd.Timestamp("2019-12-31")
    good = experiment.h2_test(train, test, ["x2"], ["sig_good"], end, end)
    noise = experiment.h2_test(train, test, ["x2"], ["sig_noise"], end, end)
    assert good["model"] == "logit_hazard" and good["auc_signals"] > good["auc_base"]
    assert good["p_one_sided"] < 0.05
    assert noise["p_one_sided"] > 0.05


def test_calibration_by_horizon_uses_only_windows_that_are_observed():
    import numpy as np

    preds = pd.DataFrame({"model": "m", "pd12": [0.1, 0.2, 0.3, 0.4], "pd24": [0.2, 0.3, 0.4, 0.5],
                          "pd36": [0.3, 0.4, 0.5, 0.6], "dur": [5.0, 30.0, 14.0, 40.0], "evt": [1, 0, 0, 0]})
    t = experiment.calibration_by_horizon(preds).set_index("horizon")
    assert t.loc[12, "n"] == 4 and t.loc[12, "observed"] == 0.25          # row 3 survived 14 > 12 months
    assert t.loc[24, "n"] == 3                                            # row 3 censored at 14 < 24: dropped
    assert t.loc[36, "n"] == 2 and np.isclose(t.loc[36, "mean_pd"], 0.45)


def test_run_keeps_duration_and_event_for_calibration():
    p = synthetic_panel(n_firms=60, seed=2)
    firms = p["cd_cvm"].unique()
    train, test = p[p["cd_cvm"].isin(firms[::2])], p[p["cd_cvm"].isin(firms[1::2])]
    _, preds = experiment.run(train, test, ["x1"], ["logit_hazard"], fit_end=pd.Timestamp("2019-12-31"),
                              eval_end=pd.Timestamp("2019-12-31"), fast=True)
    assert {"dur", "evt"} <= set(preds.columns)


def test_signal_set_is_the_12_month_counts_without_the_own_event():
    assert "sig_any_12m" in experiment.SIGNALS
    assert all(s.endswith("_12m") for s in experiment.SIGNALS)
    assert not any("own_recovery" in s for s in experiment.SIGNALS)
    assert len(experiment.SIGNALS) == 11


def test_ablation_keeps_the_primary_model_out_of_the_holm_family():
    base = fake_preds(seed=1)
    sig = base.copy()
    sig.loc[sig["model"] == "cox", "pd12"] = 0.9 * sig.loc[sig["model"] == "cox", "y12"] + 0.05
    t = experiment.ablation_table(base, sig).set_index("model")
    assert t.loc["logit_hazard", "role"] == "primary" and pd.isna(t.loc["logit_hazard", "p_holm"])
    assert (t.drop("logit_hazard")["role"] == "secondary").all()
    assert t.loc["cox", "auc_diff"] > 0 and t.loc["cox", "p_holm"] < 0.05


def test_rolling_refit_trains_with_a_one_year_embargo_before_each_test_year():
    p = synthetic_panel(n_firms=200, seed=7)
    res, folds = experiment.rolling(p, years=[2017, 2018], models=["logit_hazard"], features=["x1"],
                                    eval_end=pd.Timestamp("2019-12-31"), fast=True)
    assert res[["year", "model"]].values.tolist() == [[2017, "logit_hazard"], [2018, "logit_hazard"]]
    f = folds.set_index("year")
    assert f.loc[2017, "train_last_q"] == pd.Timestamp("2015-12-31")
    assert f.loc[2018, "train_last_q"] == pd.Timestamp("2016-12-31")
    assert f.loc[2018, "test_first_q"] == pd.Timestamp("2018-03-31")
    assert (res["auc"] > 0.6).all()


def test_reliable_signals_keep_categories_the_two_readers_agree_on():
    kappa = {c: 0.8 for c in experiment.FEATURE_CATEGORIES}
    kappa |= {"going_concern": 0.0, "sanction": 0.51, "emergency_capital": 0.6, "own_recovery": 0.9}
    out = experiment.reliable_signals(kappa, threshold=0.6)
    assert "sig_going_concern_12m" not in out and "sig_sanction_12m" not in out
    assert "sig_emergency_capital_12m" in out
    assert "sig_own_recovery_12m" not in out  # never a predictor, whatever the agreement
    assert "sig_any_12m" not in out  # the any-signal count mixes in the categories left out
    assert set(out) < set(experiment.SIGNALS)


def test_reliable_set_is_the_one_named_in_the_preregistration_amendment():
    import json
    from pathlib import Path

    agree = json.loads((Path(__file__).parents[1] / "results" / "llm_agreement.json").read_text())
    out = experiment.reliable_signals({c: v["kappa"] for c, v in agree.items()})
    assert out == [f"sig_{c}_12m" for c in ("debt_restructuring", "covenant", "missed_payment", "asset_sale_liquidity",
                                            "emergency_capital", "ceo_cfo_change", "rating_downgrade")]
