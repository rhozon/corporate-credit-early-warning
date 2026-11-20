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


def main(argv=None):
    p = argparse.ArgumentParser(prog="distress")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("events")
    e.add_argument("--sheet", default=None)
    e.set_defaults(fn=cmd_events)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
