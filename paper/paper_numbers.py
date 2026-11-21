"""Every figure quoted in paper.qmd, read from the repository's data and results files.

Nothing in the text is typed by hand: the paper imports ``N`` and formats from it. A figure whose
source file does not exist yet is ``None``, and ``fmt`` prints it as a visible gap, so a draft can
never pass for a finished paper.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GAP = "[pending]"


def _json(rel: str):
    p = ROOT / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _events() -> pd.DataFrame:
    p = ROOT / "data" / "events_reviewed.csv"
    return pd.read_csv(p if p.exists() else ROOT / "data" / "events_draft.csv", parse_dates=["event_date"])


def load() -> dict:
    n: dict = {}
    ev = _events()
    n["events_all"] = len(ev)
    n["event_firms"] = ev["cd_cvm"].nunique()
    n["events_window"] = int((ev["event_date"] >= "2011-01-01").sum())
    n["left_truncated"] = int(ev["left_truncated"].sum()) if "left_truncated" in ev else None
    n["events_reviewed"] = (ROOT / "data" / "events_reviewed.csv").exists()

    p = pd.read_parquet(ROOT / "data" / "panel.parquet", columns=["q", "cd_cvm", "event"])
    n["panel_rows"], n["panel_firms"] = len(p), p["cd_cvm"].nunique()
    n["panel_first"], n["panel_last"] = p["q"].min().date(), p["q"].max().date()
    train, test = p[p["q"] <= "2018-12-31"], p[p["q"] >= "2020-01-01"]
    n["train_rows"], n["train_events"] = len(train), int(train["event"].sum())
    n["test_rows"], n["test_events"] = len(test), int(test["event"].sum())

    n["prisma"] = _json("slr/prisma.json")
    n["gaps"] = _json("slr/gaps_counts.json")
    n["core"] = _json("slr/core_retrieval.json")
    n["signals_kappa"] = _json("results/llm_agreement.json")
    n["subject_loss"] = _json("results/subject_vs_fulltext.json")
    n["llm_models"] = _json("results/llm_models.json")
    sig = ROOT / "data" / "signals_primary.parquet"
    if sig.exists():
        s = pd.read_parquet(sig)
        cats = [c for c in s.columns if c not in ("cd_cvm", "delivered", "none", "doc_id")]
        n["signal_docs"] = len(s)
        n["signal_any"] = int((s[cats].sum(axis=1) > 0).sum())
        n["signal_counts"] = {c: int(s[c].sum()) for c in cats}
    n["validation"] = pd.read_csv(ROOT / "results" / "validation_base.csv") \
        if (ROOT / "results" / "validation_base.csv").exists() else None
    for name in ("test_h1", "test_delong", "test_h3"):
        f = ROOT / "results" / f"{name}.csv"
        n[name] = pd.read_csv(f) if f.exists() else None
    n["test_h2"] = _json("results/test_h2.json")
    return n


def fmt(x, digits: int = 0, pct: bool = False) -> str:
    """Thousands with a comma, fixed decimals; ``None`` shows as a gap."""
    if x is None:
        return GAP
    if pct:
        return f"{100 * x:.{digits}f}%"
    if isinstance(x, (int,)) or (isinstance(x, float) and digits == 0):
        return f"{int(round(x)):,}"
    return f"{x:,.{digits}f}"


def table(df: pd.DataFrame | None, digits: int = 3) -> str:
    """A pipe table for the PDF; a gap line when the results are not there yet."""
    if df is None:
        return f"\n{GAP}\n"
    cells = [[fmt(v, digits) if isinstance(v, float) else str(v) for v in row] for row in df.itertuples(index=False)]
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "|".join(":---" if k == 0 else "---:" for k in range(len(df.columns))) + "|"
    return "\n".join([head, rule] + ["| " + " | ".join(r) + " |" for r in cells]) + "\n"


def event_date(cd_cvm: int, nth: int = 0) -> str:
    """Date of a past case's event, read from the event base (the n-th episode of that firm)."""
    ev = _events()
    d = ev.loc[ev["cd_cvm"] == cd_cvm, "event_date"].sort_values()
    return d.iloc[nth].strftime("%B %Y") if len(d) > nth else GAP


N = load()
