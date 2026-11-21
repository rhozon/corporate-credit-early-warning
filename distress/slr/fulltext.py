"""Full-text core of the review: selection, retrieval of open-access texts, structured extraction.

Every included study is coded from its abstract (``review.extract``). A core of about 60 open-access
studies is also read in full. The core is drawn by stratum, so that the strands the gaps are about
(survival, deep learning, text and language models, Brazil, out-of-time validation) are each read in
depth, and then filled with the most cited studies overall. Citations are taken per year since
publication, so recent work is not crowded out by age alone.
"""
from __future__ import annotations

import html as _html
import io
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd

from .review import FAMILIES, UA, UNITS, VALIDATION

STRATA = ("survival", "deep_learning", "text_llm", "brazil", "out_of_time")
MIN_CHARS = 12_000  # below this a page is a landing page or an abstract, not a paper


# selection ---------------------------------------------------------------------------
def _strata(r) -> list[str]:
    fam = set(r["families"])
    out = [s for s in ("survival", "deep_learning") if s in fam]
    if "text_llm" in fam or r["uses_text"] or r["uses_llm"]:
        out.append("text_llm")
    if re.search("brazil|brasil", str(r["country"]), re.I):
        out.append("brazil")
    if r["validation"] == "out_of_time":
        out.append("out_of_time")
    return out + ["general"]


def rank_core(ex: pd.DataFrame, year_now: int = 2026) -> pd.DataFrame:
    """One row per (stratum, open-access record), ranked by citations per year within the stratum."""
    oa = ex[ex["is_oa"].fillna(False).astype(bool) & ex["oa_url"].notna()].copy()
    oa["score"] = oa["cited_by"].fillna(0) / (year_now - oa["year"].fillna(year_now) + 1).clip(lower=1)
    oa["stratum"] = oa.apply(_strata, axis=1)
    long = oa.explode("stratum").sort_values(["stratum", "score"], ascending=[True, False])
    long["rank"] = long.groupby("stratum").cumcount() + 1
    return long.reset_index(drop=True)


def pick_core(ranked: pd.DataFrame, have_text: set[str], per_stratum: int = 10, n: int = 60) -> pd.DataFrame:
    """A quota per stratum from records whose full text was retrieved, then the best scores overall."""
    picked: dict[str, str] = {}
    ok = ranked[ranked["openalex_id"].isin(have_text)]
    for s in STRATA:
        for oid in ok[ok["stratum"] == s].sort_values("rank")["openalex_id"]:
            if sum(v == s for v in picked.values()) >= per_stratum or len(picked) >= n:
                break
            picked.setdefault(oid, s)
    for oid in ok[ok["stratum"] == "general"].sort_values("rank")["openalex_id"]:
        if len(picked) >= n:
            break
        picked.setdefault(oid, "general")
    first = ranked.drop_duplicates("openalex_id").set_index("openalex_id")
    core = first.loc[list(picked)].drop(columns=["stratum", "rank"]).reset_index()
    core.insert(1, "stratum", list(picked.values()))
    return core


# retrieval ---------------------------------------------------------------------------
def pdf_link(page: str, base_url: str) -> str | None:
    m = re.search(r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)', page, re.I) or \
        re.search(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']citation_pdf_url', page, re.I)
    return urljoin(base_url, _html.unescape(m.group(1))) if m else None


def html_text(page: str) -> str:
    t = re.sub(r"(?is)<(script|style|noscript)\b.*?</\1>", " ", page)
    t = re.sub(r"(?s)<!--.*?-->", " ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", _html.unescape(t)).strip()


def trim(text: str, head: int = 30_000, tail: int = 6_000) -> str:
    """Data and method sit early in a paper, conclusions late; the middle is cut first."""
    if len(text) <= head + tail:
        return text
    return text[:head] + "\n[...]\n" + text[-tail:]


def looks_like_paper(text: str) -> bool:
    """Long enough, with a reference list and the usual sections; a repository or publisher page
    full of menus and an abstract fails this."""
    if len(text) < MIN_CHARS:
        return False
    t = text.lower()
    sections = sum(bool(re.search(rf"\b{w}", t)) for w in ("introduction", "result", "conclusion", "method", "data"))
    return bool(re.search(r"\b(references|bibliography|referências)\b", t)) and sections >= 2


def _pdf_text(content: bytes) -> str:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(content))
        return " ".join((p.extract_text() or "") for p in reader.pages[:60])
    except Exception:
        return ""


def _get(url: str) -> tuple[str, bytes]:
    import requests

    r = requests.get(url, headers={**UA, "Accept": "application/pdf,text/html;q=0.9"}, timeout=60,
                     allow_redirects=True)
    r.raise_for_status()
    return r.headers.get("Content-Type", ""), r.content


def _as_text(ctype: str, content: bytes) -> tuple[str, str]:
    """(kind, text) for a response: kind is 'pdf' or 'html'."""
    if "pdf" in ctype.lower() or content[:5] == b"%PDF-":
        return "pdf", _pdf_text(content)
    return "html", content.decode("utf-8", errors="replace")


def alt_urls(work: dict) -> list[str]:
    """Places an OpenAlex work can be read: PDF links (repository copies first), then repository pages.

    Publisher sites often refuse scripted requests; arXiv, SSRN mirrors and university repositories
    usually serve the same open-access text without that.
    """
    locs = [loc for loc in work.get("locations") or [] if loc]
    repo = lambda loc: ((loc.get("source") or {}).get("type") == "repository")  # noqa: E731
    pdfs = [loc["pdf_url"] for loc in sorted(locs, key=lambda loc: not repo(loc)) if loc.get("pdf_url")]
    pages = [loc["landing_page_url"] for loc in locs if repo(loc) and loc.get("landing_page_url")]
    return list(dict.fromkeys(pdfs + pages))


def openalex_works(dois: list[str], cache: Path) -> dict[str, dict]:
    """OpenAlex records by DOI, fetched 50 at a time and cached."""
    import json
    import time

    import requests

    have = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    todo = [d for d in dict.fromkeys(dois) if d and d not in have]
    for k in range(0, len(todo), 50):
        chunk = todo[k:k + 50]
        r = requests.get("https://api.openalex.org/works", headers=UA, timeout=60,
                         params={"filter": "doi:" + "|".join(chunk), "per-page": 50,
                                 "select": "doi,locations,best_oa_location"})
        r.raise_for_status()
        for w in r.json().get("results", []):
            have[(w.get("doi") or "").replace("https://doi.org/", "").lower()] = w
        for d in chunk:
            have.setdefault(d, {})
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(have), encoding="utf-8")
        time.sleep(1)
    return have


def _one(url: str, get) -> str:
    kind, body = _as_text(*get(url))
    if kind == "pdf":
        return body
    text, link = "", pdf_link(body, url)
    if link:
        try:
            k2, b2 = _as_text(*get(link))
            text = b2 if k2 == "pdf" else ""
        except Exception:
            text = ""
    return text if looks_like_paper(text) else html_text(body)


def fetch_text(urls, cache_dir: Path, key: str, get=None) -> str | None:
    """Full text of an open-access paper, or None when only a landing page or abstract is reachable.

    ``urls`` is tried in order until one gives a full text. Results, including misses, are cached as
    text files so a rerun sends no request twice.
    """
    get = get or _get
    path = Path(cache_dir) / f"{re.sub(r'[^A-Za-z0-9]+', '_', key)[-80:]}.txt"
    if path.exists():
        t = path.read_text(encoding="utf-8")
        return t if looks_like_paper(t) else None
    text = ""
    for url in ([urls] if isinstance(urls, str) else list(urls)):
        try:
            text = re.sub(r"\s+", " ", _one(url, get)).strip()
        except Exception:
            text = ""
        if looks_like_paper(text):
            break
    ok = looks_like_paper(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if ok else "", encoding="utf-8")
    return text if ok else None


# extraction --------------------------------------------------------------------------
FULLTEXT_SYSTEM = """You read the full text of a study on predicting corporate default or financial distress.
Report only what the text states; use null or "unclear" when it does not say. Fields:
- country: main country of the data, "multi", or "unclear".
- data: the data source in a few words (for example "Compustat and CRSP", "CVM filings", "bank loan book").
- unit: listed, private_sme, mixed, loans, or unclear.
- period: sample years, for example "2005-2019", or "unclear".
- n_firms, n_events: number of firms and of default or distress cases, as integers, or null.
- methods: model families, from: statistical, ml_trees, svm, neural_net, deep_learning, survival,
  ensemble_hybrid, text_llm, other.
- best_model: the model the authors report as best, in a few words.
- validation: in_sample, holdout_random, cross_validation, out_of_time (a later period held out), or unclear.
- horizon_months: prediction horizon in months, or null.
- metric: the main performance metric (for example "AUC"). best_value: its value for the best model, or null.
- calibration_reported: true if calibration (Brier score, calibration curve, Hosmer-Lemeshow) is reported.
- uses_text: true if text is an input. uses_llm: true if a large language model is used.
- finding: the main result in one sentence. limitations: the main limitation the authors state, in one sentence.
Answer only JSON with exactly these keys."""


def _int(x):
    if isinstance(x, bool) or x is None:
        return None
    if isinstance(x, (int, float)):
        return int(x)
    m = re.fullmatch(r"\s*([\d,.\s]+)\s*", str(x))
    return int(re.sub(r"[^\d]", "", m.group(1))) if m and re.search(r"\d", m.group(1)) else None


def _float(x):
    try:
        return None if x is None or isinstance(x, bool) else float(x)
    except (TypeError, ValueError):
        return None


def parse_fulltext(obj) -> dict:
    obj = obj if isinstance(obj, dict) else {}
    methods = list(obj.get("methods") or [])
    validation = obj.get("validation") or "unclear"
    if set(methods) - FAMILIES or validation not in VALIDATION:
        raise ValueError(f"out of vocabulary: methods={methods} validation={validation}")
    s = lambda k, n=300: str(obj.get(k) or "unclear")[:n]  # noqa: E731
    return {"country": s("country", 60), "data": s("data", 120),
            "unit": obj.get("unit") if obj.get("unit") in UNITS else "unclear", "period": s("period", 30),
            "n_firms": _int(obj.get("n_firms")), "n_events": _int(obj.get("n_events")), "methods": methods,
            "best_model": s("best_model", 80), "validation": validation,
            "horizon_months": _int(obj.get("horizon_months")), "metric": s("metric", 40),
            "best_value": _float(obj.get("best_value")),
            "calibration_reported": bool(obj.get("calibration_reported")),
            "uses_text": bool(obj.get("uses_text")), "uses_llm": bool(obj.get("uses_llm")),
            "finding": s("finding"), "limitations": s("limitations")}


def _blank() -> dict:
    return {**parse_fulltext({}), "read_ok": False}


def _read_one(llm, title: str, text: str) -> dict:
    try:
        return {**parse_fulltext(llm.complete(FULLTEXT_SYSTEM, f"TITLE: {title}\n\nTEXT:\n{trim(text)}",
                                              schema={"type": "object"})), "read_ok": True}
    except ValueError:
        return _blank()


def extract_fulltext(llm, core: pd.DataFrame, texts: dict[str, str], workers: int = 2) -> pd.DataFrame:
    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(lambda r: _read_one(llm, r.title, texts[r.openalex_id]), core.itertuples()))
    return pd.concat([core.reset_index(drop=True), pd.DataFrame(rows).add_prefix("ft_")], axis=1)
