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
# Event adjudication ran on Qwen3 via OpenRouter; signals and screening run on Gemma 4 (open weights) via
# Google AI Studio, free tier.
EVENTS_MODEL = ("qwen/qwen3-235b-a22b-2507", 1.4e-7, 8e-7)
PRIMARY = ("google/gemma-4-26b-a4b-it", 0.0, 0.0)
SECONDARY = ("google/gemma-4-31b-it", 0.0, 0.0)
IPE_YEARS = range(2003, 2027)


def llm(which=PRIMARY, max_tokens=300) -> LLM:
    model, pin, pout = which
    return LLM(model, cap_usd=20.0, price_in=pin, price_out=pout, max_tokens=max_tokens)


def cmd_events(a):
    cad = cvm.read_cad()
    ipe = cvm.read_ipe(IPE_YEARS)
    cands = events.candidate_events(cad, ipe)
    print("candidates by source:", cands["source"].value_counts().to_dict())
    adj = events.adjudicate(cands, event_classifier(llm(EVENTS_MODEL)))
    adj.to_csv(DATA / "events_adjudicated.csv", index=False)
    ev = events.build_events(adj)
    ev = events.non_financial(ev, cad)
    ev.to_csv(DATA / "events_draft.csv", index=False)
    print(f"events: {len(ev)} in {ev['cd_cvm'].nunique()} firms; in window (2011+): "
          f"{(ev['event_date'] >= events.WINDOW_START).sum()}; left-truncated: {ev['left_truncated'].sum()}")
    print("LLM spend so far:", llm(EVENTS_MODEL).ledger())
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

    m = macro_mod.build_macro()
    pit_path = DATA / "pit.parquet"
    if pit_path.exists():
        pit = pd.read_parquet(pit_path)
    else:
        codes = {c for cs in features.STOCK.values() for c in cs} | set(features.FLOW.values())
        parts = []
        for kind, years in (("itr", ITR_YEARS), ("dfp", DFP_YEARS)):
            for table in ("BPA", "BPP", "DRE"):
                st = cvm.read_statements(kind, years, table)
                parts.append(st[st["code"].isin(codes)])
        stmts = pd.concat(parts, ignore_index=True)
        print("statement rows:", len(stmts), "firms:", stmts["cd_cvm"].nunique())
        pit = features.point_in_time(stmts)
        pit.to_parquet(pit_path, index=False)
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


def cmd_signals(a):
    import json

    from .llm import signals

    cad = cvm.read_cad().drop_duplicates("cd_cvm")
    ipe = cvm.read_ipe(range(2010, 2027))
    fr = ipe[ipe["category"].str.startswith("Fato Relevante")]
    fr = events.non_financial(fr, cad).dropna(subset=["delivered"]).sort_values(["cd_cvm", "delivered"])
    fr = fr[fr["subject"].str.strip() != ""].reset_index(drop=True)
    print("material facts to classify:", len(fr))
    out = {}
    # The second reader codes a random subsample, enough to estimate agreement.
    sub = fr.sample(n=min(a.secondary_sample, len(fr)), random_state=2026).sort_index()
    for tag, which, docs in (("primary", PRIMARY, fr), ("secondary", SECONDARY, sub)):
        if a.only and tag != a.only:
            continue
        model = llm(which, max_tokens=3000)
        cl = signals.classify_subjects(docs, model, batch=40, workers=a.workers)
        cl.index = docs.index
        cl.to_parquet(DATA / f"signals_{tag}.parquet")
        out[tag] = cl
        print(tag, "done:", len(cl), "documents")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "llm_models.json").write_text(json.dumps(
        {"primary": {"model": PRIMARY[0], "usd_per_token_in": PRIMARY[1], "usd_per_token_out": PRIMARY[2]},
         "secondary": {"model": SECONDARY[0], "usd_per_token_in": SECONDARY[1], "usd_per_token_out": SECONDARY[2]},
         "temperature": 0, "batch": 40, "secondary_sample": a.secondary_sample}, indent=2))
    if len(out) == 2:
        kappa = signals.agreement(out["primary"].loc[out["secondary"].index], out["secondary"])
        (ROOT / "results" / "llm_agreement.json").write_text(json.dumps(kappa, indent=2))
        print("kappa:", {k: round(v, 2) for k, v in kappa.items()})
    if a.sheet and "primary" in out:
        cl = out["primary"].assign(name=fr["name"], subject=fr["subject"], link=fr["link"])
        cl.reset_index(drop=False).rename(columns={"index": "doc_id"}).to_parquet(DATA / "signals_primary_docs.parquet")
        signals.gold_sheet(cl, Path(a.sheet), n=150)
        print("gold sheet:", a.sheet)


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
    sg = sub.add_parser("signals")
    sg.add_argument("--workers", type=int, default=4)
    sg.add_argument("--secondary-sample", type=int, default=3000)
    sg.add_argument("--only", choices=["primary", "secondary"], default=None)
    sg.add_argument("--sheet", default=None)
    sg.set_defaults(fn=cmd_signals)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
