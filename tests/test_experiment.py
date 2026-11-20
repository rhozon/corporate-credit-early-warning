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
