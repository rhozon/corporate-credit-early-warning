import numpy as np
import pandas as pd
import pytest

from distress import evaluate, stats


def naive_delong(y, p1, p2):
    """Textbook DeLong (1988) with explicit pairwise kernels, for checking the fast version."""
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()

    def comps(p):
        x, z = p[pos], p[neg]
        psi = (x[:, None] > z[None, :]).astype(float) + 0.5 * (x[:, None] == z[None, :])
        return psi.mean(), psi.mean(axis=1), psi.mean(axis=0)

    a1, v10_1, v01_1 = comps(p1)
    a2, v10_2, v01_2 = comps(p2)
    s10 = np.cov(np.vstack([v10_1, v10_2]))
    s01 = np.cov(np.vstack([v01_1, v01_2]))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n
    return a1 - a2, (a1 - a2) / np.sqrt(var)


def test_delong_matches_the_textbook_formula():
    rng = np.random.default_rng(1)
    y = (rng.random(400) < 0.2).astype(int)
    p1 = y * 1.0 + rng.normal(size=400)
    p2 = y * 0.6 + rng.normal(size=400)
    diff, z, p = stats.delong(y, p1, p2)
    d_ref, z_ref = naive_delong(y, p1, p2)
    assert diff == pytest.approx(d_ref)
    assert z == pytest.approx(z_ref, rel=1e-6)
    assert 0 < p < 0.5  # one-sided, first scorer better


def test_holm_adjustment():
    assert stats.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_cluster_bootstrap_interval_contains_estimate():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"cd_cvm": np.repeat(np.arange(60), 5), "v": rng.normal(size=300)})
    est, lo, hi = stats.cluster_bootstrap(lambda d: d["v"].mean(), df, by="cd_cvm", B=300, seed=2026)
    assert lo <= est <= hi
    assert hi - lo > 0


def test_ks_is_the_largest_gap_between_score_distributions():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.6, 0.4, 0.8, 0.9])
    # at threshold 0.4: TPR 1, FPR 1/3 -> 2/3; at 0.8: TPR 2/3, FPR 0 -> 2/3
    assert evaluate.metrics(y, p)["ks"] == pytest.approx(2 / 3)


def test_calibration_of_well_calibrated_scores():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.01, 0.3, size=20000)
    y = (rng.random(20000) < p).astype(int)
    m = evaluate.metrics(y, p)
    assert m["calib_slope"] == pytest.approx(1.0, abs=0.1)
    assert m["calib_intercept"] == pytest.approx(0.0, abs=0.15)
    assert 0 < m["brier"] < 0.25 and 0.5 < m["auc"] < 1


def test_survival_metrics_skip_horizons_beyond_follow_up():
    rng = np.random.default_rng(5)
    n = 400
    risk = rng.normal(size=n)
    d = np.minimum(rng.exponential(scale=30 * np.exp(-risk)), 30.0)
    e = (d < 30).astype(int)
    pdc = pd.DataFrame({f"pd{h}": 1 - np.exp(-h / (30 * np.exp(-risk))) for h in (12, 24, 36)})
    out = evaluate.surv_metrics(d, e, d, e, risk, pdc)
    assert np.isnan(out["auc_t36"])
    assert out["auc_t12"] > 0.6 and np.isfinite(out["ibs"])


def test_survival_metrics_reward_correct_ranking():
    rng = np.random.default_rng(4)
    n = 600
    risk = rng.normal(size=n)
    t_event = rng.exponential(scale=40 * np.exp(-risk))
    censor = rng.uniform(5, 60, size=n)
    d = np.minimum(np.minimum(t_event, censor), 36)
    e = (t_event <= np.minimum(censor, 36)).astype(int)
    pdc = pd.DataFrame({f"pd{h}": 1 - np.exp(-h / (40 * np.exp(-risk))) for h in (12, 24, 36)})
    good = evaluate.surv_metrics(d, e, d, e, risk, pdc)
    bad = evaluate.surv_metrics(d, e, d, e, -risk, 1 - pdc)
    assert good["c_harrell"] > 0.65 > bad["c_harrell"]
    assert good["c_uno"] > 0.65
    assert good["auc_t12"] > 0.65
    assert good["ibs"] < bad["ibs"]


def test_a_test_event_at_the_end_of_training_follow_up_does_not_break_the_censoring_weights():
    # Training outcomes are censored at 36 months, so the censoring curve reaches zero there; a test event at
    # exactly 36 months lies past every scored horizon and must count only as a firm still at risk before it.
    rng = np.random.default_rng(5)
    n = 400
    risk = rng.normal(size=n)
    t_event = rng.exponential(scale=40 * np.exp(-risk))
    d = np.minimum(t_event, 36.0)
    e = (t_event <= 36.0).astype(int)
    pdc = pd.DataFrame({f"pd{h}": 1 - np.exp(-h / (40 * np.exp(-risk))) for h in (12, 24, 36)})
    d_te, e_te = d.copy(), e.copy()
    d_te[0], e_te[0] = 36.0, 1
    out = evaluate.surv_metrics(d, e, d_te, e_te, risk, pdc)
    d_ce, e_ce = d_te.copy(), e_te.copy()
    d_ce[0], e_ce[0] = 35.999, 0
    ref = evaluate.surv_metrics(d, e, d_ce, e_ce, risk, pdc)
    assert np.isfinite(out["auc_t12"]) and np.isfinite(out["ibs"])
    assert out["auc_t12"] == pytest.approx(ref["auc_t12"]) and out["ibs"] == pytest.approx(ref["ibs"])
