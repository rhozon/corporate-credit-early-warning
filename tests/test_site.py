import json

import pandas as pd

from distress import site

RESULTS = site.RESULTS


def _load(tmp_path):
    out = tmp_path / "data.json"
    site.export_site_json(out)
    text = out.read_text(encoding="utf-8")
    return out, json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))


def test_site_json_is_strict_small_and_matches_results(tmp_path):
    out, d = _load(tmp_path)
    assert out.stat().st_size < 1_000_000
    h1 = pd.read_csv(RESULTS / "test_h1.csv").set_index("model")
    assert [m["model"] for m in d["models"]] == list(h1.index)
    xgb = next(m for m in d["models"] if m["model"] == "xgb_cox")
    assert abs(xgb["auc"] - h1.loc["xgb_cox", "auc"]) < 1e-6
    h3 = pd.read_csv(RESULTS / "test_h3.csv")
    r = h3[(h3.model == "cox") & (h3.horizon == 36)].iloc[0]
    c = next(x for x in d["calibration"] if x["model"] == "cox" and x["horizon"] == 36)
    assert abs(c["mean_pd"] - r.mean_pd) < 1e-6 and abs(c["observed"] - r.observed) < 1e-6
    assert len(d["drivers"]) == len(pd.read_csv(RESULTS / "drivers.csv"))


def test_only_companies_that_went_through_a_recovery_are_named(tmp_path):
    _, d = _load(tmp_path)
    memos = json.loads((RESULTS / "memos" / "summary.json").read_text(encoding="utf-8"))
    allowed = {m["company"] for m in memos}
    assert {c["company"] for c in d["cases"]} == allowed
    assert all(c["event"] in ("RJ", "RE") for c in d["cases"])
    assert all("Fatos das ferramentas" not in c["memo"] and c["memo"].strip() for c in d["cases"])
