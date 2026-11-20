"""Command line for the pipeline.

    python -m distress.cli events      # candidate events, LLM adjudication, review sheet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import cvm, events
from .llm.adjudicate import event_classifier
from .llm.client import LLM

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PRIMARY = ("qwen/qwen3-235b-a22b-2507", 1.4e-7, 8e-7)
SECONDARY = ("openai/gpt-oss-120b", 9e-8, 3.6e-7)
IPE_YEARS = range(2003, 2027)


def llm(which=PRIMARY, max_tokens=300) -> LLM:
    model, pin, pout = which
    return LLM(model, cap_usd=20.0, price_in=pin, price_out=pout, max_tokens=max_tokens)


def cmd_events(a):
    cad = cvm.read_cad()
    ipe = cvm.read_ipe(IPE_YEARS)
    cands = events.candidate_events(cad, ipe)
    print("candidates by source:", cands["source"].value_counts().to_dict())
    adj = events.adjudicate(cands, event_classifier(llm()))
    adj.to_csv(DATA / "events_adjudicated.csv", index=False)
    ev = events.build_events(adj)
    ev = events.non_financial(ev, cad)
    ev.to_csv(DATA / "events_draft.csv", index=False)
    print(f"events: {len(ev)} in {ev['cd_cvm'].nunique()} firms; in window (2011+): "
          f"{(ev['event_date'] >= events.WINDOW_START).sum()}; left-truncated: {ev['left_truncated'].sum()}")
    print("LLM spend so far:", llm().ledger())
    if a.sheet:
        events.review_sheet(ev[ev["event_date"] >= pd.Timestamp("2008-01-01")], Path(a.sheet))
        print("review sheet:", a.sheet)


ITR_YEARS, DFP_YEARS = range(2011, 2027), range(2010, 2027)


def load_events() -> pd.DataFrame:
    """Reviewed events when the author's corrections exist, the draft otherwise."""
    path = DATA / "events_reviewed.csv"
    if not path.exists():
        path = DATA / "events_draft.csv"
    return pd.read_csv(path, parse_dates=["event_date", "episode_end"])


def cmd_panel(a):
    from . import features, macro as macro_mod, panel

    codes = {c for cs in features.STOCK.values() for c in cs} | set(features.FLOW.values())
    parts = []
    for kind, years in (("itr", ITR_YEARS), ("dfp", DFP_YEARS)):
        for table in ("BPA", "BPP", "DRE"):
            st = cvm.read_statements(kind, years, table)
            parts.append(st[st["code"].isin(codes)])
    stmts = pd.concat(parts, ignore_index=True)
    print("statement rows:", len(stmts), "firms:", stmts["cd_cvm"].nunique())
    pit = features.point_in_time(stmts)
    m = macro_mod.build_macro()
    ipca = macro_mod.sgs.series(433, "2008-01-01")
    index = (1 + ipca / 100).cumprod()
    deflator = index / index.iloc[-1]
    r = features.ratios(pit, deflator=deflator)

    par = pd.concat([cvm.read_parecer("itr", ITR_YEARS), cvm.read_parecer("dfp", DFP_YEARS)], ignore_index=True)
    gc = features.going_concern_rule(par).sort_values("dt_receb")
    r = pd.merge_asof(r.sort_values("as_of"), gc[["cd_cvm", "dt_receb", "gc_flag"]],
                      left_on="as_of", right_on="dt_receb", by="cd_cvm", direction="backward")
    r["gc_flag"] = r["gc_flag"].fillna(False).astype(int)
    r["z2"] = features.altman_z2(r)
    r = r.drop(columns=["dt_receb"])
    r.to_parquet(DATA / "features.parquet", index=False)

    cad = cvm.read_cad().drop_duplicates("cd_cvm")
    end = pd.Timestamp(a.end)
    p = panel.build_panel(r, load_events(), cad, m, start="2011-03-31", end=str(end.date()))
    p.to_parquet(DATA / "panel.parquet", index=False)
    train, test = panel.split(p, data_end=pd.Timestamp(a.data_end))
    print(f"panel rows {len(p)}, firms {p['cd_cvm'].nunique()}; train rows {len(train)} "
          f"(events {int(train['event'].sum())}), test rows {len(test)} (events {int(test['event'].sum())})")


def main(argv=None):
    p = argparse.ArgumentParser(prog="distress")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("events")
    e.add_argument("--sheet", default=None)
    e.set_defaults(fn=cmd_events)
    pn = sub.add_parser("panel")
    pn.add_argument("--end", default="2026-09-30")
    pn.add_argument("--data-end", default="2026-11-20")
    pn.set_defaults(fn=cmd_panel)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
