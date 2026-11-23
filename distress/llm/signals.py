"""Distress signals read from material-fact subjects by a language model.

Each subject line is classified, with the company masked, into a closed set of
categories. Counts of each category over the previous 4 and 12 months become
features of the firm-quarter panel. Documents announcing the company's own
recovery are classified (they help audit the event base) but never used as a
feature, because they coincide with the outcome.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .client import LLM
from .mask import mask

CATEGORIES = ["debt_restructuring", "covenant", "missed_payment", "asset_sale_liquidity", "emergency_capital",
              "going_concern", "ceo_cfo_change", "rating_downgrade", "own_recovery", "subsidiary_recovery",
              "sanction"]
FEATURE_CATEGORIES = [c for c in CATEGORIES if c != "own_recovery"]
LABELS_PT = {
    "debt_restructuring": "renegociação de dívida",
    "covenant": "covenant (quebra ou waiver)",
    "missed_payment": "atraso de pagamento",
    "asset_sale_liquidity": "venda de ativo por liquidez",
    "emergency_capital": "aumento de capital emergencial",
    "going_concern": "dúvida de continuidade",
    "ceo_cfo_change": "troca de CEO ou CFO",
    "rating_downgrade": "rebaixamento de rating",
    "own_recovery": "recuperação da própria empresa",
    "subsidiary_recovery": "recuperação de controlada",
    "sanction": "sanção",
}

SYSTEM = """Você lê assuntos de Fatos Relevantes de companhias abertas brasileiras. O nome da empresa foi trocado por [EMPRESA].
Para cada assunto numerado, marque as categorias que ele indica. Pode haver mais de uma. Use "none" se nenhuma se aplica.

Categorias:
- debt_restructuring: renegociação, reperfilamento, alongamento ou reestruturação de dívidas.
- covenant: quebra, descumprimento, waiver ou dispensa de covenant ou de cláusula de vencimento antecipado.
- missed_payment: atraso ou não pagamento de juros, amortização, debênture ou outra obrigação; vencimento antecipado declarado.
- asset_sale_liquidity: venda de ativos, unidades ou participações para fazer caixa ou reduzir dívida.
- emergency_capital: aumento de capital, aporte ou empréstimo de acionista para reforçar liquidez.
- going_concern: dúvida sobre continuidade operacional, prejuízo relevante com alerta, patrimônio líquido negativo.
- ceo_cfo_change: renúncia, destituição ou troca de diretor-presidente ou diretor financeiro.
- rating_downgrade: rebaixamento de rating ou perspectiva negativa.
- own_recovery: pedido, deferimento ou andamento de recuperação judicial ou extrajudicial ou falência da própria [EMPRESA].
- subsidiary_recovery: recuperação judicial ou extrajudicial ou falência de controlada ou coligada.
- sanction: multa, sanção, intervenção regulatória ou investigação relevante.

Responda só com JSON: {"items": [{"i": 1, "labels": ["..."]}, ...]}."""

GENERIC = {"CIA", "COMPANHIA", "GRUPO", "EMPRESA", "BANCO", "BRASIL", "BRASILEIRA", "INDUSTRIA", "IND",
           "SA", "S.A.", "PARTICIPACOES", "PARTICIPAÇÕES", "COMERCIO", "COMÉRCIO"}


def name_variants(name: str) -> list[str]:
    name = (name or "").strip()
    base = re.split(r"\s+-\s+", name)[0]
    stem = re.sub(r"\b(S\.?\s?A\.?|S/A|LTDA)\b\.?", "", base, flags=re.I).strip(" .-")
    out = {name, base, stem}
    first = stem.split()[0] if stem.split() else ""
    if len(first) >= 4 and first.upper() not in GENERIC:
        out.add(first)
    return [v for v in out if v]


def parse_batch(obj, n: int) -> list[set]:
    out = [{"none"} for _ in range(n)]
    items = obj.get("items", []) if isinstance(obj, dict) else []
    allowed = set(CATEGORIES) | {"none"}
    for it in items:
        i = int(it.get("i", 0))
        labels = set(it.get("labels") or [])
        bad = labels - allowed
        if bad:
            raise ValueError(f"labels outside the schema: {sorted(bad)}")
        if 1 <= i <= n:
            out[i - 1] = labels or {"none"}
    return out


def _classify_batch(llm: LLM, subjects: list[str]) -> list[set]:
    user = "\n".join(f"{k + 1}. {s}" for k, s in enumerate(subjects))
    try:
        return parse_batch(llm.complete(SYSTEM, user, schema={"type": "object"}), len(subjects))
    except ValueError:
        if len(subjects) == 1:
            return [{"none"}]
        return [lab for s in subjects for lab in _classify_batch(llm, [s])]


def classify_subjects(docs: pd.DataFrame, llm: LLM, batch: int = 25, workers: int = 1) -> pd.DataFrame:
    masked = [mask(s, name_variants(n), c) for s, n, c in zip(docs["subject"], docs["name"], docs["cnpj"])]
    chunks = [masked[i:i + batch] for i in range(0, len(masked), batch)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(lambda ch: _classify_batch(llm, ch), chunks))
    labels = [lab for r in results for lab in r]
    out = docs[["cd_cvm", "delivered"]].copy().reset_index(drop=True)
    for c in CATEGORIES + ["none"]:
        out[c] = [int(c in lab) for lab in labels]
    return out


# Full text of a sample of documents, to measure what the subject line alone misses. The category
# block is cut from SYSTEM, which stays byte-identical so the cached subject answers remain valid.
DOC_SYSTEM = ("Você lê o texto completo de um Fato Relevante de companhia aberta brasileira. O nome da empresa "
              "foi trocado por [EMPRESA].\nMarque as categorias que o documento indica. Pode haver mais de uma. "
              'Use "none" se nenhuma se aplica.\n\n'
              + SYSTEM[SYSTEM.index("Categorias:"):SYSTEM.index("Responda")]
              + 'Responda só com JSON: {"labels": ["..."]}.')


def _pdf_text(content: bytes) -> str:
    from ..slr.fulltext import _pdf_text as pdf

    return pdf(content)


def fetch_doc(url: str, cache_dir: Path, key: str, get=None) -> str | None:
    """Text of a filing from the CVM document server (a PDF, whatever the headers say), cached."""
    from ..slr.fulltext import _get, html_text

    path = Path(cache_dir) / f"{key}.txt"
    if path.exists():
        t = path.read_text(encoding="utf-8")
        return t or None
    try:
        ctype, content = (get or _get)(url)
        is_pdf = content[:5] == b"%PDF-" or "pdf" in ctype.lower()
        text = _pdf_text(content) if is_pdf else html_text(content.decode("utf-8", errors="replace"))
    except Exception:
        text = ""
    text = re.sub(r"\s+", " ", text).strip().encode("utf-8", "ignore").decode("utf-8")
    text = text if len(text) >= 100 else ""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text or None


def _classify_doc(llm: LLM, text: str) -> set:
    try:
        obj = llm.complete(DOC_SYSTEM, text, schema={"type": "object"})
        labels = obj.get("labels", []) if isinstance(obj, dict) else []
        return parse_batch({"items": [{"i": 1, "labels": labels}]}, 1)[0]
    except ValueError:
        return {"none"}


def classify_fulltext(docs: pd.DataFrame, texts: dict, llm: LLM, workers: int = 2,
                      head: int = 12_000, tail: int = 3_000) -> pd.DataFrame:
    """Same categories as ``classify_subjects``, read from each document's full text (masked, trimmed)."""
    rows = list(docs.itertuples())

    def one(r):
        t = texts.get(r.Index)
        if not t:
            return None
        t = t if len(t) <= head + tail else t[:head] + "\n[...]\n" + t[-tail:]
        return _classify_doc(llm, mask(t, name_variants(r.name), r.cnpj))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        labels = list(ex.map(one, rows))
    out = docs[["cd_cvm", "delivered"]].copy().reset_index(drop=True)
    for c in CATEGORIES + ["none"]:
        out[c] = [int(lab is not None and c in lab) for lab in labels]
    out["read_ok"] = [lab is not None for lab in labels]
    return out


def signal_counts(classified: pd.DataFrame) -> dict:
    """Documents classified, documents with any signal (and with any signal used as a feature), per category
    and documents per year: the counts the paper quotes, written from the labels."""
    lab = classified[CATEGORIES].astype(bool)
    years = pd.to_datetime(classified["delivered"]).dt.year.value_counts().sort_index()
    return {"documents": int(len(classified)), "any_signal": int(lab.any(axis=1).sum()),
            "any_feature_signal": int(lab[FEATURE_CATEGORIES].any(axis=1).sum()),
            "by_category": {c: int(lab[c].sum()) for c in CATEGORIES},
            "by_year": {str(y): int(n) for y, n in years.items()}}


FULLTEXT_NOTE = ("Full-text flags of the company's own recovery come almost all from firms already in an open "
                 "recovery episode, whose filings restate that status; they are not new information.")


def fulltext_checks(full: pd.DataFrame, events: pd.DataFrame) -> dict:
    """How many full-text flags of own recovery and debt restructuring fall inside an episode already open
    for the firm (event date to episode end) when the document was filed."""
    read = full[full["read_ok"].astype(bool)]
    ev = events.assign(event_date=pd.to_datetime(events["event_date"]), episode_end=pd.to_datetime(events["episode_end"]))

    def inside(r) -> bool:
        e = ev[ev["cd_cvm"] == r.cd_cvm]
        when = pd.Timestamp(r.delivered)
        return bool(((e["event_date"] <= when) & (when <= e["episode_end"])).any())

    out = {}
    for c in ("own_recovery", "debt_restructuring"):
        flagged = read[read[c].astype(bool)]
        out[f"{c}_full_text_inside_open_episode"] = int(sum(inside(r) for r in flagged.itertuples()))
        out[f"{c}_full_text"] = int(len(flagged))
    out["note"] = FULLTEXT_NOTE
    return out


def subject_loss(subject: pd.DataFrame, full: pd.DataFrame) -> dict:
    """Per category: documents flagged from the full text, from the subject, by both, and the share of
    full-text flags the subject also finds."""
    def row(f, s):
        n = int(f.sum())
        return {"full_text": n, "subject": int(s.sum()), "both": int((f & s).sum()),
                "recall_of_subject": float((f & s).sum() / n) if n else float("nan")}

    out = {c: row(full[c].values == 1, subject[c].values == 1) for c in CATEGORIES}
    out["any"] = row(full[CATEGORIES].values.sum(axis=1) > 0, subject[CATEGORIES].values.sum(axis=1) > 0)
    return out


def agreement(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    from sklearn.metrics import cohen_kappa_score

    out = {}
    for c in [c for c in CATEGORIES if c in a.columns and c in b.columns]:
        x, y = a[c].values, b[c].values
        out[c] = float(cohen_kappa_score(x, y)) if len(set(x) | set(y)) > 1 else np.nan
    return out


def agreement_table(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Per category: how often each reader flags it, how often both do, raw agreement and kappa.

    Kappa alone hides prevalence: on a rare category one reader's few extra flags drive it to zero
    while raw agreement stays near one, so the counts are reported beside it.
    """
    kappa = agreement(a, b)
    out = {}
    for c, k in kappa.items():
        x, y = a[c].values == 1, b[c].values == 1
        out[c] = {"n": int(len(x)), "primary": int(x.sum()), "secondary": int(y.sum()), "both": int((x & y).sum()),
                  "agreement": float((x == y).mean()), "kappa": k}
    return out


def signal_features(classified: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Counts of each category in the 4 and 12 months before each panel quarter (documents strictly before q)."""
    cols = {f"sig_{c}_{w}m": np.zeros(len(panel), dtype=int) for c in FEATURE_CATEGORIES for w in (4, 12)}
    cols["sig_any_12m"] = np.zeros(len(panel), dtype=int)
    by_firm = {cd: g.sort_values("delivered") for cd, g in classified.groupby("cd_cvm")}
    pos = {idx: k for k, idx in enumerate(panel.index)}
    for cd, rows in panel.groupby("cd_cvm"):
        g = by_firm.get(cd)
        if g is None:
            continue
        t = g["delivered"].values
        anyflag = g[FEATURE_CATEGORIES].sum(axis=1).gt(0).astype(int).values
        csum = {c: np.concatenate([[0], np.cumsum(g[c].values)]) for c in FEATURE_CATEGORIES}
        cany = np.concatenate([[0], np.cumsum(anyflag)])
        for idx, q in zip(rows.index, rows["q"]):
            hi = np.searchsorted(t, np.datetime64(q), side="left")
            for w in (4, 12):
                lo = np.searchsorted(t, np.datetime64(q - pd.DateOffset(months=w)), side="left")
                for c in FEATURE_CATEGORIES:
                    cols[f"sig_{c}_{w}m"][pos[idx]] = csum[c][hi] - csum[c][lo]
                if w == 12:
                    cols["sig_any_12m"][pos[idx]] = cany[hi] - cany[lo]
    return pd.DataFrame(cols, index=panel.index)


def _first_category(df: pd.DataFrame) -> np.ndarray:
    """The first category a document is flagged with, in CATEGORIES order, or "none"."""
    flags = df[CATEGORIES].astype(bool).values
    first = np.array(CATEGORIES, dtype=object)[flags.argmax(axis=1)]
    return np.where(flags.any(axis=1), first, "none")


def gold_sample(classified: pd.DataFrame, n: int = 150, seed: int = 2026) -> list[int]:
    """Positions of the documents drawn for hand labelling: an equal quota from each stratum (the first
    category the primary reader flags, or none), then a uniform draw from the rest up to n."""
    rng = np.random.default_rng(seed)
    df = classified.reset_index(drop=True)
    first = pd.Series(_first_category(df), index=df.index)
    per = max(1, n // (len(CATEGORIES) + 1))
    picked = []
    for _, g in df.groupby(first):
        picked += list(rng.choice(g.index, size=min(len(g), per), replace=False))
    rest = df.index.difference(picked)
    if len(picked) < n and len(rest):
        picked += list(rng.choice(rest, size=min(n - len(picked), len(rest)), replace=False))
    return [int(i) for i in rng.permutation(picked[:n])]


def inclusion_prob(classified: pd.DataFrame, n: int = 150) -> np.ndarray:
    """Chance that each document enters ``gold_sample``: its stratum quota plus its share of the uniform
    fill. Weighting labels by the inverse turns sample counts into estimates for all documents."""
    first = pd.Series(_first_category(classified.reset_index(drop=True)))
    sizes = first.value_counts()
    quota = sizes.clip(upper=max(1, n // (len(CATEGORIES) + 1)))
    total, drawn = len(first), int(quota.sum())
    if drawn > n:
        raise ValueError("the stratum quotas exceed the sample size")
    fill = min(n - drawn, total - drawn) / (total - drawn) if total > drawn else 0.0
    q = (first.map(quota) / first.map(sizes)).values
    return q + (1 - q) * fill


def gold_sheet(classified: pd.DataFrame, path: Path, n: int = 150, seed: int = 2026) -> pd.DataFrame:
    """Sheet for the author to label, drawn by ``gold_sample`` (stratified by the primary reader's category)."""
    df = classified.reset_index(drop=True)
    sample = df.loc[gold_sample(classified, n, seed)]
    sheet = pd.DataFrame({"id": sample.index, "empresa": sample["name"], "data": sample["delivered"].dt.date,
                          "assunto": sample["subject"], "link": sample["link"]})
    for c in CATEGORIES:
        sheet[LABELS_PT[c]] = ""
    sheet["nenhuma"] = ""
    sheet["observação"] = ""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sheet.to_excel(path, index=False)
    return sheet


def _mark(v) -> int:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 0
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) and v in (0, 1):
        return int(v)
    t = str(v).strip().lower()
    if t in ("", "0", "n", "nao", "não"):
        return 0
    if t in ("x", "1", "s", "sim"):
        return 1
    raise ValueError(f"unclear mark {v!r} in the label sheet: use X, or leave the cell blank")


def read_gold(sheet: pd.DataFrame) -> pd.DataFrame:
    """The author's labels: one row per document id, 0/1 per category. Every row needs a category or
    "nenhuma", never both."""
    ids = sheet["id"].astype(int).values
    marks = pd.DataFrame({c: [_mark(v) for v in sheet[LABELS_PT[c]]] for c in CATEGORIES}, index=ids)
    none = np.array([_mark(v) for v in sheet["nenhuma"]])
    flagged = marks.sum(axis=1).values > 0
    blank = ids[~flagged & (none == 0)]
    if len(blank):
        raise ValueError(f"{len(blank)} row(s) with no label (mark a category or nenhuma): ids {list(blank[:10])}")
    both = ids[flagged & (none == 1)]
    if len(both):
        raise ValueError(f"{len(both)} row(s) marked nenhuma and a category: ids {list(both[:10])}")
    return marks


def gold_metrics(gold: pd.DataFrame, readers: dict[str, pd.DataFrame], pi: pd.Series) -> dict:
    """Precision and recall of each reader against the author's labels, per category and for any feature
    category. Each document counts 1 / (its chance of being drawn), so the stratified sample estimates the
    rates over all documents; the raw counts are reported beside them."""
    w = 1 / pi.loc[gold.index].values

    def flag(df, c):
        cols = FEATURE_CATEGORIES if c == "any" else [c]
        return df.loc[gold.index, cols].values.sum(axis=1) > 0

    def score(h, m):
        both = h & m
        return {"flags": int(m.sum()), "both": int(both.sum()),
                "precision": float((w * both).sum() / (w * m).sum()) if m.any() else None,
                "recall": float((w * both).sum() / (w * h).sum()) if h.any() else None}

    out = {}
    for c in CATEGORIES + ["any"]:
        h = flag(gold, c)
        out[c] = {"human": int(h.sum())} | {name: score(h, flag(r, c)) for name, r in readers.items()}
    return out
