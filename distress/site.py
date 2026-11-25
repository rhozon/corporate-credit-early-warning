"""Data for the interactive part of the case page.

    python -m distress.cli site path/to/data.json

Only aggregate results of the out-of-time test and the four memos on companies that
already went through a recovery. Scores of other companies are never exported.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
H1_COLS = ["n", "events", "auc", "auc_lo", "auc_hi", "pr_auc", "ks", "brier", "c_uno", "auc_t12", "auc_t24", "auc_t36",
           "calib_intercept", "calib_slope"]


def _clean(v):
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return round(float(v), 6) if isinstance(v, float) else int(v)
    return v


def _records(df: pd.DataFrame) -> list[dict]:
    return [{k: _clean(v.item() if hasattr(v, "item") else v) for k, v in r.items()} for r in df.to_dict(orient="records")]


def _memo(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    body, _, facts = text.partition("## Fatos das ferramentas")
    lines = body.strip().splitlines()
    tool = json.loads(re.search(r"```json\s*(\{.*\})\s*```", facts, re.S).group(1))
    return {"company": lines[0].lstrip("# ").strip(), "memo": "\n".join(lines[1:]).strip(),
            "indicators": tool.get("indicadores", {}), "drivers": tool.get("fatores_de_risco", [])}


def build() -> dict:
    h1 = pd.read_csv(RESULTS / "test_h1.csv")
    models = _records(h1[["model"] + [c for c in H1_COLS if c in h1.columns]])
    delong = _records(pd.read_csv(RESULTS / "test_delong.csv")[["model", "versus", "auc_diff", "diff_lo", "diff_hi", "p_holm", "p_cluster_holm"]])
    calib = _records(pd.read_csv(RESULTS / "test_h3.csv"))
    drivers = _records(pd.read_csv(RESULTS / "drivers.csv"))
    summary = json.loads((RESULTS / "memos" / "summary.json").read_text(encoding="utf-8"))
    memos = {m["company"]: m for m in (_memo(p) for p in sorted((RESULTS / "memos").glob("*.md")))}
    cases = []
    for s in summary:
        m = memos[s["company"]]
        cases.append({**{k: s[k] for k in ("company", "analysis_date", "event", "event_date", "pd12", "pd24", "pd36", "blocked")},
                      "unsupported": len(s.get("unsupported", [])), "memo": m["memo"], "indicators": m["indicators"],
                      "drivers": m["drivers"]})
    observed = {int(r["horizon"]): r["observed"] for r in calib if r["model"] == calib[0]["model"]}
    return {"models": models, "delong": delong, "calibration": calib, "drivers": drivers, "cases": cases, "observed": observed}


def export_site_json(path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build(), ensure_ascii=False, allow_nan=False, separators=(",", ":")), encoding="utf-8")
    return path
