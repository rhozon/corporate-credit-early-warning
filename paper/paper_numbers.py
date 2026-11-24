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
    n["window_firms"] = int(ev.loc[ev["event_date"] >= "2011-01-01", "cd_cvm"].nunique())
    n["left_truncated"] = int(ev.loc[ev["left_truncated"].astype(bool), "cd_cvm"].nunique()) \
        if "left_truncated" in ev else None
    n["events_reviewed"] = (ROOT / "data" / "events_reviewed.csv").exists()
    second = ROOT / "data" / "events_second_reader.csv"
    if n["events_reviewed"] and second.exists():
        draft = pd.read_csv(ROOT / "data" / "events_draft.csv")
        checked = ev[ev["review"] != "not_in_sheet"]
        rebuilt = int((ev["review"] == "rebuilt_after_drop").sum())
        r = {"checked": len(pd.read_csv(second)),
             "stand": int((checked["review"] == "confirmed").sum()),
             "unsure": int(checked["review_note"].fillna("").str.startswith("segundo leitor incerto").sum()),
             "moved": int((checked["review"] == "date_corrected").sum()),
             "dropped": len(draft) - len(ev) + rebuilt, "rebuilt": rebuilt}
        r["merged"] = r["checked"] - r["stand"] - r["moved"] - r["dropped"]
        n["review"] = r
    else:
        n["review"] = None

    p = pd.read_parquet(ROOT / "data" / "panel.parquet", columns=["q", "cd_cvm", "event", "y12"])
    n["panel_rows"], n["panel_firms"] = len(p), p["cd_cvm"].nunique()
    n["panel_first"], n["panel_last"] = p["q"].min().date(), p["q"].max().date()
    train, test = p[p["q"] <= "2018-12-31"], p[p["q"] >= "2020-01-01"]
    n["train_rows"], n["train_events"] = len(train), int(train["event"].sum())
    n["train_rate12"] = float(train["y12"].mean())
    n["test_rows"], n["test_events"] = len(test), int(test["event"].sum())

    n["prisma"] = _json("slr/prisma.json")
    n["gaps"] = _json("slr/gaps_counts.json")
    n["core"] = _json("slr/core_retrieval.json")
    n["signals_kappa"] = _json("results/llm_agreement.json")
    n["subject_loss"] = _json("results/subject_vs_fulltext.json")
    n["llm_models"] = _json("results/llm_models.json")
    n["memos"] = _json("results/memos/summary.json")
    # Document-level labels are private (they read as alerts on named firms); the counts are public.
    counts = _json("results/signal_counts.json") or {}
    n["signal_docs"], n["signal_any"] = counts.get("documents"), counts.get("any_signal")
    n["signal_counts"] = counts.get("by_category")
    n["validation"] = pd.read_csv(ROOT / "results" / "validation_base.csv") \
        if (ROOT / "results" / "validation_base.csv").exists() else None
    for name in ("test_h1", "test_delong", "test_h3"):
        f = ROOT / "results" / f"{name}.csv"
        n[name] = pd.read_csv(f) if f.exists() else None
    n["test_h2"] = _json("results/test_h2.json")
    n["test_h2_reliable"] = _json("results/test_h2_reliable.json")
    n["test_h2_exploratory"] = _json("results/test_h2_exploratory.json")
    for name in ("test_h2_ablation", "test_robust_narrow", "test_robust_rolling", "drivers",
                 "test_auditor_exploratory"):
        f = ROOT / "results" / f"{name}.csv"
        n[name] = pd.read_csv(f) if f.exists() else None
    a = pd.read_parquet(ROOT / "data" / "panel.parquet")
    if "auditor_change" in a:
        n["auditor_rate"] = float(a["auditor_change"].mean())
        n["auditor_rate_y12"] = float(a.loc[a["y12"] == 1, "auditor_change"].mean())
    else:
        n["auditor_rate"] = n["auditor_rate_y12"] = None
    n["claims"] = _claims(n)
    return n


def _names(ms: list[str]) -> str:
    ms = [MODEL_NAMES.get(m, m) for m in ms]
    return ", ".join(ms[:-1]) + (" and " if len(ms) > 1 else "") + ms[-1] if ms else ""


def _claims(n: dict) -> dict:
    """Sentences whose direction depends on the results, so the text cannot outlive a rerun."""
    out = {"abstract_h1": "", "exploratory_gain": None, "beats_altman_c": [], "beats_logit_c": []}
    dl = n.get("test_delong")
    if dl is not None and "p_cluster_holm" in dl:
        ok = dl[dl["p_cluster_holm"] < 0.05]
        a = list(ok.loc[ok["versus"] == "altman", "model"])
        lg = list(ok.loc[ok["versus"] == "logit_hazard", "model"])
        out["beats_altman_c"], out["beats_logit_c"] = a, lg
        both = [m for m in a if m in lg]
        if both:
            out["abstract_h1"] = f"{_names(both)} beat both benchmarks, also when firms are resampled whole."
        elif a:
            out["abstract_h1"] = (f"{_names(a)} beat the Altman score when firms are resampled whole; no model's "
                                  f"edge over the hazard logit survives that test.")
        else:
            out["abstract_h1"] = "Once firms are resampled whole, no model's edge over the benchmarks is significant."
    x = n.get("test_h2_exploratory")
    if x and "p_cluster_one_sided" in x.get("all", {}):
        out["exploratory_gain"] = bool(x["all"]["p_one_sided"] < 0.05 and x["all"]["p_cluster_one_sided"] < 0.05)
    return out


MODEL_NAMES = {"altman": "Altman score", "logit_hazard": "Hazard logit", "cox": "Cox", "xgb_pd12": "XGBoost PD",
               "xgb_cox": "XGBoost survival", "rsf": "Random survival forest", "deepsurv": "DeepSurv",
               "deephit": "DeepHit"}

CATEGORY_NAMES = {"debt_restructuring": "Debt restructuring", "covenant": "Covenant waiver or breach",
                  "missed_payment": "Missed payment", "asset_sale_liquidity": "Asset sale for liquidity",
                  "emergency_capital": "Emergency capital", "going_concern": "Going concern",
                  "ceo_cfo_change": "Executive change", "rating_downgrade": "Rating downgrade",
                  "own_recovery": "Own recovery", "subsidiary_recovery": "Subsidiary recovery",
                  "sanction": "Sanction"}


PREDICTOR_NAMES = {
    "wc_ta": "Working capital / assets", "re_ta": "Retained earnings / assets", "ebit_ta": "EBIT / assets",
    "eq_tl": "Equity / liabilities", "lev": "Leverage", "curr": "Current ratio", "cash_ta": "Cash / assets",
    "int_cov": "Interest coverage", "margin": "Net margin", "sales_g": "Revenue growth",
    "log_size_real": "Size (real assets, log)", "neg_eq": "Negative equity",
    "loss_q4": "Quarter-ends with a twelve-month loss (last four)", "filing_delay_days": "Filing delay",
    "n_restatements_4q": "Restatements (last four quarters)", "info_age_days": "Age of the latest filing",
    "gc_flag": "Going-concern flag", "prior_event": "Previous recovery", "selic": "Selic rate",
    "ipca12": "Inflation (12 months)", "ibc_yoy": "Activity index growth", "unemp": "Unemployment",
    "pj_default": "Corporate loan default rate", "spread": "Corporate lending spread",
    "auditor_change": "Change of audit firm (12 months)"}


WORDS = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]


def spell(n) -> str:
    """Counts up to ten in words, as prose wants them; larger counts in figures."""
    return WORDS[int(n)] if 0 <= int(n) <= 10 else fmt(int(n))


def fmt(x, digits: int = 0, pct: bool = False) -> str:
    """Thousands with a comma, fixed decimals; ``None`` shows as a gap."""
    if x is None:
        return GAP
    if pct:
        return f"{100 * x:.{digits}f}%"
    if isinstance(x, (int,)) or (isinstance(x, float) and digits == 0):
        return f"{int(round(x)):,}"
    return f"{round(x, digits) + 0.0:,.{digits}f}"  # + 0.0 turns -0.00 into 0.00


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
