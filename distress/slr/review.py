"""Systematic literature review: search, deduplication, dual screening, PRISMA counts, references.

The protocol is in ``slr/protocol.md`` and was written before the search ran.
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SLR = ROOT / "slr"
OPENALEX = "https://api.openalex.org/works"
UA = {"User-Agent": "systematic-review-script/1.0"}

STRANDS = {
    "a": '("bankruptcy prediction" OR "default prediction" OR "financial distress" OR "corporate default" OR '
         '"corporate failure" OR "business failure") AND ("survival analysis" OR "hazard model" OR '
         '"machine learning" OR "deep learning" OR "neural network")',
    "b": '("large language model" OR "LLM" OR "GPT" OR "textual analysis" OR "text mining" OR '
         '"natural language processing") AND ("credit risk" OR "default prediction" OR "bankruptcy" OR '
         '"financial distress")',
    "c": '(Brazil OR Brazilian OR Brasil) AND (bankruptcy OR "financial distress" OR insolvency OR '
         '"judicial recovery" OR "corporate default" OR "recuperação judicial" OR "falência")',
}
YEARS = "2000-2026"
TYPES = "article|review|preprint"

# Same strings in Semantic Scholar bulk-search syntax (+ is AND, | is OR).
S2 = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
S2_FIELDS = "title,abstract,year,venue,journal,externalIds,citationCount,openAccessPdf,publicationTypes"
S2_STRANDS = {
    "a": '("bankruptcy prediction" | "default prediction" | "financial distress" | "corporate default" | '
         '"corporate failure" | "business failure") + ("survival analysis" | "hazard model" | '
         '"machine learning" | "deep learning" | "neural network")',
    "b": '("large language model" | LLM | GPT | "textual analysis" | "text mining" | '
         '"natural language processing") + ("credit risk" | "default prediction" | bankruptcy | '
         '"financial distress")',
    "c": '(Brazil | Brazilian | Brasil) + (bankruptcy | "financial distress" | insolvency | '
         '"judicial recovery" | "corporate default" | "recuperação judicial" | falência)',
}


# search ----------------------------------------------------------------------
def search(strand: str, cache_dir: Path = SLR / "raw", per_page: int = 200) -> list[dict]:
    import requests

    path = Path(cache_dir) / f"openalex_{strand}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    flt = f"title_and_abstract.search:{STRANDS[strand]},publication_year:{YEARS},type:{TYPES}"
    out, cursor = [], "*"
    while cursor:
        for i in range(5):
            r = requests.get(OPENALEX, params={"filter": flt, "per-page": per_page, "cursor": cursor}, headers=UA, timeout=120)
            if r.status_code in (429, 500, 502, 503) and i < 4:
                time.sleep(10 * (i + 1))
                continue
            r.raise_for_status()
            break
        data = r.json()
        out += data["results"]
        cursor = data["meta"].get("next_cursor")
        if not data["results"]:
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def search_s2(strand: str, cache_dir: Path = SLR / "raw") -> list[dict]:
    """Semantic Scholar bulk search, paged by token, cached per strand."""
    import requests

    path = Path(cache_dir) / f"s2_{strand}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    params = {"query": S2_STRANDS[strand], "fields": S2_FIELDS, "year": YEARS}
    out, token = [], None
    while True:
        if token:
            params["token"] = token
        for i in range(8):
            r = requests.get(S2, params=params, headers=UA, timeout=120)
            if r.status_code in (429, 500, 502, 503, 504) and i < 7:
                time.sleep(5 * (i + 1))
                continue
            r.raise_for_status()
            break
        data = r.json()
        out += data.get("data") or []
        token = data.get("token")
        if not token:
            break
        time.sleep(1.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def normalise_s2(raw: list[dict], strand: str) -> pd.DataFrame:
    rows = []
    for w in raw:
        ext = w.get("externalIds") or {}
        journal = (w.get("journal") or {}).get("name") or w.get("venue") or None
        pdf = w.get("openAccessPdf") or {}
        types = w.get("publicationTypes") or []
        rows.append({"openalex_id": f"s2:{w.get('paperId')}", "doi": _doi(ext.get("DOI")),
                     "title": w.get("title") or "", "year": w.get("year"),
                     "type": types[0] if types else None, "language": None,
                     "abstract": w.get("abstract") or "", "venue": journal or None, "source_id": None,
                     "cited_by": w.get("citationCount") or 0, "is_oa": bool(pdf.get("url")),
                     "oa_url": pdf.get("url"), "strand": strand})
    return pd.DataFrame(rows)


def abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    pos = sorted((p, w) for w, ps in inv.items() for p in ps)
    return " ".join(w for _, w in pos)


def _doi(x) -> str | None:
    if not x:
        return None
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", str(x)).strip().lower() or None


def normalise(raw: list[dict], strand: str) -> pd.DataFrame:
    rows = []
    for w in raw:
        src = ((w.get("primary_location") or {}).get("source") or {})
        oa = w.get("open_access") or {}
        rows.append({"openalex_id": w["id"], "doi": _doi(w.get("doi")), "title": w.get("title") or "",
                     "year": w.get("publication_year"), "type": w.get("type"), "language": w.get("language"),
                     "abstract": abstract(w.get("abstract_inverted_index")), "venue": src.get("display_name"),
                     "source_id": src.get("id"), "cited_by": w.get("cited_by_count", 0),
                     "is_oa": bool(oa.get("is_oa")), "oa_url": oa.get("oa_url"), "strand": strand})
    return pd.DataFrame(rows)


def _title_key(t: str) -> str:
    t = unicodedata.normalize("NFKD", t or "")
    return re.sub(r"[^a-z0-9]", "", "".join(c for c in t if not unicodedata.combining(c)).lower())


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_t"] = df["title"].map(_title_key)
    with_doi = df[df["doi"].notna()].drop_duplicates("doi")
    keep = pd.concat([with_doi, df[df["doi"].isna()]])
    keep = keep.drop_duplicates("_t")
    return keep.drop(columns="_t").reset_index(drop=True)


# screening ---------------------------------------------------------------------
SCREEN_SYSTEM = """You screen records for a systematic review on predicting corporate default with statistical,
machine learning, deep learning or survival models, and on using text or language models for credit risk.

Include a record when ALL hold:
1. It is an empirical study (not an editorial, not a pure literature review, not a book chapter summary).
2. The unit of analysis is non-financial firms (listed or private companies, SMEs), OR it applies text mining /
   language models to credit risk or financial distress of firms or borrowers.
3. The outcome is default, bankruptcy, judicial recovery, insolvency or financial distress.
4. It estimates a predictive model and reports some measure of predictive performance.

Exclude: consumer-only credit scoring without text methods; banks or insurers as the only subjects; sovereign or
country risk; studies on stock returns of distressed firms; theoretical models without data.

Answer only with JSON: {"items": [{"i": 1, "decision": "include" | "exclude", "reason": "few words"}, ...]}."""


def parse_screen(obj, n: int) -> list[dict]:
    out = [{"decision": "exclude", "reason": "no answer"} for _ in range(n)]
    for it in (obj.get("items", []) if isinstance(obj, dict) else []):
        d = it.get("decision")
        if d not in ("include", "exclude"):
            raise ValueError(f"unknown decision {d!r}")
        i = int(it.get("i", 0))
        if 1 <= i <= n:
            out[i - 1] = {"decision": d, "reason": it.get("reason", "")}
    return out


def _screen_batch(llm, recs: pd.DataFrame) -> list[dict]:
    user = "\n\n".join(f"{k + 1}. TITLE: {t}\nABSTRACT: {a[:1500]}" for k, (t, a) in
                       enumerate(zip(recs["title"], recs["abstract"])))
    try:
        return parse_screen(llm.complete(SCREEN_SYSTEM, user, schema={"type": "object"}), len(recs))
    except ValueError:
        if len(recs) == 1:
            return [{"decision": "exclude", "reason": "unparseable"}]
        return [d for k in range(len(recs)) for d in _screen_batch(llm, recs.iloc[[k]])]


def screen_with(llm, recs: pd.DataFrame, batch: int = 10, workers: int = 8) -> list[dict]:
    chunks = [recs.iloc[i:i + batch] for i in range(0, len(recs), batch)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return [d for r in ex.map(lambda c: _screen_batch(llm, c), chunks) for d in r]


FAMILIES = {"statistical", "ml_trees", "svm", "neural_net", "deep_learning", "survival", "ensemble_hybrid",
            "text_llm", "other"}
VALIDATION = {"in_sample", "holdout_random", "cross_validation", "out_of_time", "unclear"}
UNITS = {"listed", "private_sme", "mixed", "loans", "unclear"}

EXTRACT_SYSTEM = """For each numbered study (title and abstract), extract:
- families: model families used, from: statistical (discriminant, logit, probit), ml_trees (random forest, boosting,
  decision trees), svm, neural_net (shallow neural networks), deep_learning (deep, recurrent, convolutional,
  transformer networks), survival (hazard, Cox, survival forests or survival networks), ensemble_hybrid, text_llm
  (text mining, NLP, language models), other.
- validation: in_sample, holdout_random, cross_validation, out_of_time (later period held out), or unclear.
- country: main country of the data, or "multi", or "unclear".
- unit: listed, private_sme, mixed, loans, or unclear.
- uses_text: true if text (reports, filings, news) is an input. uses_llm: true if a large language model is used.
- n_events: number of default or distress cases if stated, else null.
Answer only JSON: {"items": [{"i": 1, "families": [...], "validation": "...", "country": "...", "unit": "...",
"uses_text": false, "uses_llm": false, "n_events": null}, ...]}."""


def as_list(x) -> list[str]:
    """A list of vocabulary terms, also when the model writes them as one string ("survival, ml_trees")."""
    if x is None:
        return []
    if isinstance(x, str):
        return [t for t in re.split(r"[,;|/\s]+", x.strip()) if t]
    return [str(t).strip() for t in x if str(t).strip()]


def parse_extract(obj, n: int) -> list[dict]:
    blank = {"families": [], "validation": "unclear", "country": "unclear", "unit": "unclear",
             "uses_text": False, "uses_llm": False, "n_events": None}
    out = [dict(blank) for _ in range(n)]
    for it in (obj.get("items", []) if isinstance(obj, dict) else []):
        fam = as_list(it.get("families"))
        if set(fam) - FAMILIES or it.get("validation", "unclear") not in VALIDATION:
            raise ValueError(f"out of vocabulary: {it}")
        i = int(it.get("i", 0))
        if 1 <= i <= n:
            ev = it.get("n_events")
            out[i - 1] = {"families": fam, "validation": it.get("validation", "unclear"),
                          "country": str(it.get("country") or "unclear"),
                          "unit": it.get("unit") if it.get("unit") in UNITS else "unclear",
                          "uses_text": bool(it.get("uses_text")), "uses_llm": bool(it.get("uses_llm")),
                          "n_events": int(ev) if isinstance(ev, (int, float)) else None}
    return out


def _extract_batch(llm, recs: pd.DataFrame) -> list[dict]:
    user = "\n\n".join(f"{k + 1}. TITLE: {t}\nABSTRACT: {a[:1800]}" for k, (t, a) in
                       enumerate(zip(recs["title"], recs["abstract"])))
    try:
        return parse_extract(llm.complete(EXTRACT_SYSTEM, user, schema={"type": "object"}), len(recs))
    except ValueError:
        if len(recs) == 1:
            return parse_extract({}, 1)
        return [d for k in range(len(recs)) for d in _extract_batch(llm, recs.iloc[[k]])]


def extract(llm, recs: pd.DataFrame, batch: int = 8, workers: int = 4) -> pd.DataFrame:
    chunks = [recs.iloc[i:i + batch] for i in range(0, len(recs), batch)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = [d for r in ex.map(lambda c: _extract_batch(llm, c), chunks) for d in r]
    return pd.concat([recs.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def synthesis(ex: pd.DataFrame) -> dict:
    period = pd.cut(ex["year"], [1999, 2009, 2019, 2100], labels=["2000-2009", "2010-2019", "2020-2026"])
    long = ex.assign(period=period).explode("families").dropna(subset=["families"])
    fam = pd.crosstab(long["families"], long["period"]).reindex(columns=["2000-2009", "2010-2019", "2020-2026"],
                                                                fill_value=0)
    val = ex["validation"].value_counts()
    brazil = ex["country"].astype(str).str.contains("brazil|brasil", case=False)
    return {"families": fam, "validation": val, "share_out_of_time": float((ex["validation"] == "out_of_time").mean()),
            "n_brazil": int(brazil.sum()), "n_llm": int(ex["uses_llm"].sum()), "n_text": int(ex["uses_text"].sum()),
            "n": int(len(ex))}


def gap_counts(ex: pd.DataFrame) -> dict:
    """The counts behind the gaps: model families crossed with text, language models, Brazil and validation."""
    fam = ex["families"].map(lambda x: set(as_list(x)))
    has = lambda f: fam.map(lambda s: f in s)  # noqa: E731
    surv, dl = has("survival"), has("deep_learning")
    txt = has("text_llm") | ex["uses_text"].astype(bool)
    llm = ex["uses_llm"].astype(bool)
    br = ex["country"].astype(str).str.contains("brazil|brasil", case=False)
    oot = ex["validation"] == "out_of_time"
    c = lambda m: int(m.sum())  # noqa: E731
    return {"n": int(len(ex)), "survival": c(surv), "survival_and_text": c(surv & txt), "survival_and_llm": c(surv & llm),
            "llm": c(llm), "llm_and_out_of_time": c(llm & oot), "text": c(txt), "brazil": c(br),
            "brazil_survival": c(br & surv), "brazil_deep": c(br & dl), "brazil_text": c(br & txt),
            "brazil_llm": c(br & llm), "out_of_time": c(oot), "validation_unclear": c(ex["validation"] == "unclear"),
            "deep_and_survival": c(dl & surv)}


def combine(a: pd.Series, b: pd.Series) -> pd.Series:
    """Inclusive rule (protocol amendment 2): a record either screener includes goes to full text."""
    return pd.Series(["include" if "include" in (x, y) else "exclude" for x, y in zip(a, b)], index=a.index)


def prisma_counts(identified: int, after_dedup: int, screened: pd.DataFrame) -> dict:
    inc = screened["final"] == "include"
    return {"identified": int(identified), "duplicates_removed": int(identified - after_dedup),
            "screened": int(len(screened)), "excluded_title_abstract": int((~inc).sum()),
            "included": int(inc.sum()), "full_text_open_access": int((inc & screened["is_oa"]).sum())}


# references ------------------------------------------------------------------------
def verify_doi(doi: str) -> dict | None:
    import requests

    for i in range(4):
        r = requests.get(f"https://api.crossref.org/works/{doi}", headers=UA, timeout=60)
        if r.status_code == 404:
            return None
        if r.status_code in (429, 500, 502, 503) and i < 3:
            time.sleep(5 * (i + 1))
            continue
        r.raise_for_status()
        return r.json()["message"]
    return None


def crossref_search(query: str, rows: int = 5) -> list[dict]:
    import requests

    for i in range(4):
        r = requests.get("https://api.crossref.org/works", headers=UA, timeout=60,
                         params={"query.bibliographic": query, "rows": rows})
        if r.status_code in (429, 500, 502, 503) and i < 3:
            time.sleep(5 * (i + 1))
            continue
        r.raise_for_status()
        return r.json()["message"]["items"]
    return []


def _words(t: str) -> set[str]:
    t = unicodedata.normalize("NFKD", t or "")
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    return {w for w in re.findall(r"[a-z0-9]+", t) if len(w) > 2}


def _matches(it: dict, author: str, year: int, title: str) -> bool:
    """Surname among the first three authors (group authors come first in some records), year within
    one (online-first dates), and at least 80% of the title's words in the record's title and subtitle."""
    names = {_title_key(a.get("family", "")) for a in (it.get("author") or [])[:3]}
    got_year = (it.get("issued", {}).get("date-parts") or [[None]])[0][0]
    want = _words(title)
    got = _words(" ".join((it.get("title") or [""])[:1] + (it.get("subtitle") or [])[:1]))
    return (_title_key(author) in names and bool(got_year) and abs(int(got_year) - int(year)) <= 1
            and bool(want) and len(want & got) / len(want) >= 0.8)


def resolve_reference(author: str, year: int, title: str, doi: str | None = None) -> str | None:
    """DOI of a cited work, accepted only when Crossref's record matches author, year and title.

    With ``doi`` the suggested DOI is checked; otherwise Crossref's bibliographic search is used."""
    if doi:
        rec = verify_doi(doi.lower())
        return rec["DOI"].lower() if rec and _matches(rec, author, year, title) else None
    hits = [it for it in crossref_search(f"{author} {year} {title}", rows=10) if _matches(it, author, year, title)]
    # The published version over a working paper or preprint of the same work.
    hits.sort(key=lambda it: it.get("type") not in ("journal-article", "proceedings-article"))
    return hits[0]["DOI"].lower() if hits else None


def _bibkey(m: dict) -> str:
    fam = (m.get("author") or [{"family": "anon"}])[0].get("family", "anon")
    year = (m.get("issued", {}).get("date-parts") or [[0]])[0][0]
    word = re.sub(r"[^A-Za-z]", "", (m.get("title") or ["x"])[0].split()[0])
    return _title_key(f"{fam}{year}{word}")


def _esc(s: str) -> str:
    return str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def bibtex(dois, keys: dict[str, str] | None = None) -> str:
    entries, keys = [], keys or {}
    for doi in dois:
        m = verify_doi(doi)
        if not m:
            continue
        kind = {"journal-article": "article", "proceedings-article": "inproceedings"}.get(m.get("type"), "misc")
        authors = " and ".join(f"{a.get('family', '')}, {a.get('given', '')}".strip(", ")
                               for a in m.get("author", []))
        year = (m.get("issued", {}).get("date-parts") or [[None]])[0][0]
        fields = {"title": (m.get("title") or [""])[0], "author": authors, "year": year,
                  "journal" if kind == "article" else "booktitle": (m.get("container-title") or [""])[0],
                  "volume": m.get("volume"), "number": m.get("issue"), "pages": m.get("page"), "doi": m.get("DOI")}
        body = ",\n".join(f"  {k} = {{{_esc(v)}}}" for k, v in fields.items() if v)
        entries.append(f"@{kind}{{{keys.get(doi) or _bibkey(m)},\n{body}\n}}")
    return "\n\n".join(entries) + ("\n" if entries else "")
