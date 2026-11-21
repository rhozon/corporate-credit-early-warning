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
# Per-firm outputs that read as risk signals (labels, predictions) stay out of the repository.
PRIVATE = DATA / "private"
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
        cl.to_parquet(PRIVATE / f"signals_{tag}.parquet")
        out[tag] = cl
        print(tag, "done:", len(cl), "documents")
    if "primary" not in out and (PRIVATE / "signals_primary.parquet").exists():
        out["primary"] = pd.read_parquet(PRIVATE / "signals_primary.parquet")
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
        cl.reset_index(drop=False).rename(columns={"index": "doc_id"}).to_parquet(DATA / "raw" / "signals_primary_docs.parquet")
        signals.gold_sheet(cl, Path(a.sheet), n=150)
        print("gold sheet:", a.sheet)


def slr_records() -> tuple[int, pd.DataFrame]:
    from .slr import review

    raw = pd.concat([review.normalise_s2(review.search_s2(s), s) for s in "abc"], ignore_index=True)
    recs = review.dedupe(raw[raw["title"].str.strip() != ""])
    return len(raw), recs


def cmd_slr(a):
    import json

    from .slr import review

    identified, recs = slr_records()
    if a.step == "screen":
        model = llm(PRIMARY if a.which == "primary" else SECONDARY, max_tokens=3000)
        dec = review.screen_with(model, recs, batch=10, workers=a.workers)
        out = pd.DataFrame({"openalex_id": recs["openalex_id"], "decision": [d["decision"] for d in dec],
                            "reason": [d["reason"] for d in dec]})
        out.to_csv(review.SLR / f"screen_{a.which}.csv", index=False)
        print(a.which, out["decision"].value_counts().to_dict())
    elif a.step == "extract":
        final = review.SLR / "screening.csv"
        if final.exists():
            inc = pd.read_csv(final)
            ids = set(inc.loc[inc["final"] == "include", "openalex_id"])
        else:
            s = pd.read_csv(review.SLR / "screen_primary.csv")
            ids = set(s.loc[s["decision"] == "include", "openalex_id"])
        sel = recs[recs["openalex_id"].isin(ids)]
        ex = review.extract(llm(PRIMARY, max_tokens=4000), sel, batch=8, workers=a.workers)
        ex.to_parquet(review.SLR / "extraction.parquet", index=False)
        t = review.synthesis(ex)
        print({k: v for k, v in t.items() if k not in ("families", "validation")})
        print(t["families"].to_string())
        print(t["validation"].to_string())
    elif a.step == "combine":
        p = pd.read_csv(review.SLR / "screen_primary.csv").set_index("openalex_id")
        s = pd.read_csv(review.SLR / "screen_secondary.csv").set_index("openalex_id")
        r = recs.set_index("openalex_id")
        r["decision_a"], r["reason_a"] = p["decision"], p["reason"]
        r["decision_b"], r["reason_b"] = s["decision"], s["reason"]
        r["final"] = review.combine(r["decision_a"], r["decision_b"])
        from sklearn.metrics import cohen_kappa_score
        kappa = cohen_kappa_score(r["decision_a"], r["decision_b"])
        r.reset_index().to_csv(review.SLR / "screening.csv", index=False)
        counts = review.prisma_counts(identified, len(recs), r) | {"kappa_screeners": round(float(kappa), 3),
                                                                   "agreement": round(float((r["decision_a"] == r["decision_b"]).mean()), 3)}
        (review.SLR / "prisma.json").write_text(json.dumps(counts, indent=2))
        print(counts)
    elif a.step == "core":
        from concurrent.futures import ThreadPoolExecutor

        from .slr import fulltext as ft

        ranked = ft.rank_core(pd.read_parquet(review.SLR / "extraction.parquet"))
        cache, texts = review.SLR / "fulltext", {}
        first = ranked.drop_duplicates("openalex_id").set_index("openalex_id")
        pool = ranked[(ranked["rank"] <= 60) | ((ranked["stratum"] == "general") & (ranked["rank"] <= 200))]
        works = ft.openalex_works(first.loc[pool["openalex_id"].unique(), "doi"].dropna().tolist(),
                                  review.SLR / "raw" / "openalex_locations.json")

        def urls(i):
            alt = ft.alt_urls(works.get(first.at[i, "doi"]) or {}) if pd.notna(first.at[i, "doi"]) else []
            return list(dict.fromkeys(alt + [first.at[i, "oa_url"]]))[:4]

        def fetch(ids):
            todo = [i for i in ids if i not in texts]
            with ThreadPoolExecutor(max_workers=6) as ex:
                for i, t in zip(todo, ex.map(lambda i: ft.fetch_text(urls(i), cache, i), todo)):
                    texts[i] = t

        # Walk each stratum in rank order until it has twice its quota with text (overlaps between
        # strata eat into the quota) or 60 records were tried; then the general list for the fill.
        for s, need in [(s, 2 * a.per_stratum) for s in ft.STRATA] + [("general", a.core_size + 20)]:
            ids = ranked.loc[ranked["stratum"] == s].sort_values("rank")["openalex_id"].tolist()
            for k in range(0, min(len(ids), 60 if s != "general" else 200), 12):
                fetch(ids[k:k + 12])
                if sum(bool(texts.get(i)) for i in ids) >= need:
                    break
        have = {i for i, t in texts.items() if t}
        core = ft.pick_core(ranked, have, per_stratum=a.per_stratum, n=a.core_size)
        cols = ["openalex_id", "stratum", "doi", "title", "year", "venue", "cited_by", "score", "oa_url"]
        core[cols].to_csv(review.SLR / "core_list.csv", index=False)
        out = ft.extract_fulltext(llm(PRIMARY, max_tokens=1500), core, texts, workers=a.workers)
        out.to_parquet(review.SLR / "core_extraction.parquet", index=False)
        stats = {"tried": len(texts), "full_text": len(have), "core": len(core),
                 "read_ok": int(out["ft_read_ok"].sum()), "by_stratum": core["stratum"].value_counts().to_dict()}
        (review.SLR / "core_retrieval.json").write_text(json.dumps(stats, indent=2))
        print(stats)


def cmd_fulltext_sample(a):
    """A random sample of material facts read in full, to measure what the subject line misses."""
    import json
    from concurrent.futures import ThreadPoolExecutor

    from .llm import signals

    docs = pd.read_parquet(DATA / "raw" / "signals_primary_docs.parquet")
    cnpj = cvm.read_cad().drop_duplicates("cd_cvm").set_index("cd_cvm")["cnpj"]
    sample = docs.sample(n=a.n, random_state=2026).copy()
    sample["cnpj"] = sample["cd_cvm"].map(cnpj).fillna("")
    cache = DATA / "raw" / "ipe_text"
    with ThreadPoolExecutor(max_workers=3) as ex:
        got = list(ex.map(lambda r: signals.fetch_doc(r.link, cache, str(r.doc_id)), sample.itertuples()))
    texts = dict(zip(sample.index, got))
    full = signals.classify_fulltext(sample, texts, llm(PRIMARY, max_tokens=400), workers=a.workers)
    subj = sample[signals.CATEGORIES + ["none"]].reset_index(drop=True)
    ok = full["read_ok"].values
    loss = signals.subject_loss(subj[ok], full[ok])
    res = {"sample": a.n, "read": int(ok.sum()), "model": PRIMARY[0], "seed": 2026,
           "kappa": signals.agreement(subj[ok], full[ok]), "loss": loss}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "subject_vs_fulltext.json").write_text(json.dumps(res, indent=2, default=float))
    full.assign(doc_id=sample["doc_id"].values).to_parquet(PRIVATE / "signals_fulltext_sample.parquet", index=False)
    print({k: res[k] for k in ("sample", "read")}, "any:", loss["any"])


def cmd_refs(a):
    """paper/references.bib from Crossref-verified DOIs only: the cited works in refs_seed.csv,
    resolved by bibliographic search, and the studies of the review's full-text core."""
    from .slr import review

    paper = ROOT / "paper"
    seed = pd.read_csv(paper / "refs_seed.csv")
    done = paper / "refs_resolved.csv"
    old = pd.read_csv(done).set_index("key")["doi"].dropna().to_dict() if done.exists() else {}
    hint = seed["doi_hint"].where(seed["doi_hint"].notna(), None) if "doi_hint" in seed else [None] * len(seed)
    seed["doi"] = [old.get(k) or review.resolve_reference(au, int(y), t, doi=h)
                   for k, au, y, t, h in zip(seed["key"], seed["author"], seed["year"], seed["title"], hint)]
    seed[["key", "author", "year", "title", "role", "doi"]].to_csv(done, index=False)
    core = review.SLR / "core_list.csv"
    core_dois = pd.read_csv(core)["doi"].dropna().str.lower().tolist() if core.exists() else []
    keys = dict(zip(seed["doi"].dropna(), seed.loc[seed["doi"].notna(), "key"]))
    dois = list(dict.fromkeys(seed["doi"].dropna().tolist() + core_dois))
    bib = review.bibtex(dois, keys=keys)
    extra = (paper / "references_extra.bib").read_text(encoding="utf-8")
    (paper / "references.bib").write_text(bib + "\n" + extra, encoding="utf-8")
    print({"seed": len(seed), "resolved": int(seed["doi"].notna().sum()), "core_dois": len(core_dois),
           "bib_entries": bib.count("\n@") + bib.startswith("@")})
    print("unresolved:", seed.loc[seed["doi"].isna(), "key"].tolist())


def cmd_validate(a):
    """Development run inside the training years. The test period is not touched."""
    from . import experiment, panel as panel_mod
    from .models import MODELS

    p = pd.read_parquet(DATA / "panel.parquet")
    train, _ = panel_mod.split(p, data_end=pd.Timestamp(a.data_end))
    fit, val, fit_end, val_end = experiment.validation_split(train)
    print(f"fit rows {len(fit)} events {int(fit['event'].sum())}; validation rows {len(val)} events {int(val['event'].sum())}")
    res, _ = experiment.run(fit, val, experiment.BASE, list(MODELS), fit_end, val_end, fast=a.fast)
    (ROOT / "results").mkdir(exist_ok=True)
    res.to_csv(ROOT / "results" / "validation_base.csv", index=False)
    cols = ["model", "events", "auc", "pr_auc", "ks", "brier", "calib_slope", "c_harrell", "c_uno", "auc_t12", "ibs"]
    print(res[cols].round(3).to_string(index=False))


def cmd_test(a):
    """Opens the out-of-time test: H1 (all models), H2 (pre-registered primary test and ablations), H3.

    Refuses to run on the draft event list: the test is opened once, after the author's review.
    Firm-level predictions are written to data/private only; results/ gets aggregates.
    """
    import json

    from . import experiment, panel as panel_mod
    from .llm import signals
    from .models import MODELS

    reviewed = DATA / "events_reviewed.csv"
    if not reviewed.exists():
        raise SystemExit("data/events_reviewed.csv is missing: the test stays closed until the event review.")
    if (DATA / "panel.parquet").stat().st_mtime < reviewed.stat().st_mtime:
        raise SystemExit("panel.parquet is older than the reviewed events: run `distress.cli panel` first.")
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    p = pd.read_parquet(DATA / "panel.parquet")
    p = p.join(signals.signal_features(pd.read_parquet(PRIVATE / "signals_primary.parquet"), p))
    train, test = panel_mod.split(p, data_end=pd.Timestamp(a.data_end))
    fit_end, eval_end = pd.Timestamp("2019-12-31"), pd.Timestamp(a.data_end)
    print(f"train rows {len(train)} events {int(train['event'].sum())}; "
          f"test rows {len(test)} events {int(test['event'].sum())}")

    res, preds = experiment.run(train, test, experiment.BASE, list(MODELS), fit_end, eval_end,
                                bootstrap=a.bootstrap)
    res.to_csv(out / "test_h1.csv", index=False)
    preds.to_parquet(PRIVATE / "test_preds.parquet", index=False)
    experiment.delong_table(preds).to_csv(out / "test_delong.csv", index=False)
    experiment.calibration_by_horizon(preds).to_csv(out / "test_h3.csv", index=False)

    h2 = experiment.h2_test(train, test, experiment.BASE, experiment.SIGNALS, fit_end, eval_end)
    (out / "test_h2.json").write_text(json.dumps(h2, indent=2, default=float))
    res_s, preds_s = experiment.run(train, test, experiment.BASE + experiment.SIGNALS, list(MODELS), fit_end,
                                    eval_end)
    preds_s.to_parquet(PRIVATE / "test_preds_signals.parquet", index=False)
    experiment.ablation_table(preds, preds_s).to_csv(out / "test_h2_ablation.csv", index=False)
    print(res[["model", "auc", "auc_lo", "auc_hi", "c_uno", "ibs"]].round(3).to_string(index=False))
    print("H2 primary:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in h2.items()})

    # Robustness 1: narrow event definition (judicial recovery and bankruptcy; out-of-court left out).
    from . import macro as macro_mod
    ev = load_events()
    narrow = ev[ev["event_type"].isin(["RJ", "FAL"])]
    cad = cvm.read_cad().drop_duplicates("cd_cvm")
    pn = panel_mod.build_panel(pd.read_parquet(DATA / "features.parquet"), narrow, cad, macro_mod.build_macro(),
                               start="2011-03-31", end=str(pd.Timestamp(p["q"].max()).date()))
    tr_n, te_n = panel_mod.split(pn, data_end=eval_end)
    res_n, _ = experiment.run(tr_n, te_n, experiment.BASE, list(MODELS), fit_end, eval_end)
    res_n.assign(events_definition="narrow").to_csv(out / "test_robust_narrow.csv", index=False)
    # Robustness 2: annual refit with a moving origin over the test years.
    last = (eval_end - pd.DateOffset(months=12)).year
    res_r, folds = experiment.rolling(p, range(2021, last + 1), list(MODELS), experiment.BASE, eval_end=eval_end)
    res_r.to_csv(out / "test_robust_rolling.csv", index=False)
    folds.to_csv(out / "test_robust_rolling_folds.csv", index=False)
    print("narrow:", res_n[["model", "events", "auc"]].round(3).to_dict("records"))


def main(argv=None):
    PRIVATE.mkdir(parents=True, exist_ok=True)
    p = argparse.ArgumentParser(prog="distress")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("events")
    e.add_argument("--sheet", default=None)
    e.set_defaults(fn=cmd_events)
    pn = sub.add_parser("panel")
    pn.add_argument("--end", default="2026-09-30")
    pn.add_argument("--data-end", default="2026-11-20")
    pn.set_defaults(fn=cmd_panel)
    ts = sub.add_parser("test")
    ts.add_argument("--data-end", default="2026-11-20")
    ts.add_argument("--bootstrap", type=int, default=1000)
    ts.set_defaults(fn=cmd_test)
    sg = sub.add_parser("signals")
    sg.add_argument("--workers", type=int, default=4)
    sg.add_argument("--secondary-sample", type=int, default=3000)
    sg.add_argument("--only", choices=["primary", "secondary"], default=None)
    sg.add_argument("--sheet", default=None)
    sg.set_defaults(fn=cmd_signals)
    sl = sub.add_parser("slr")
    sl.add_argument("step", choices=["screen", "combine", "extract", "core"])
    sl.add_argument("--which", choices=["primary", "secondary"], default="primary")
    sl.add_argument("--workers", type=int, default=3)
    sl.add_argument("--per-stratum", type=int, default=10)
    sl.add_argument("--core-size", type=int, default=60)
    sl.set_defaults(fn=cmd_slr)
    rf = sub.add_parser("refs")
    rf.set_defaults(fn=cmd_refs)
    fs = sub.add_parser("fulltext-sample")
    fs.add_argument("--n", type=int, default=200)
    fs.add_argument("--workers", type=int, default=2)
    fs.set_defaults(fn=cmd_fulltext_sample)
    va = sub.add_parser("validate")
    va.add_argument("--data-end", default="2026-11-20")
    va.add_argument("--fast", action="store_true")
    va.set_defaults(fn=cmd_validate)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
