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


def agreement(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    from sklearn.metrics import cohen_kappa_score

    out = {}
    for c in [c for c in CATEGORIES if c in a.columns and c in b.columns]:
        x, y = a[c].values, b[c].values
        out[c] = float(cohen_kappa_score(x, y)) if len(set(x) | set(y)) > 1 else np.nan
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


def gold_sheet(classified: pd.DataFrame, path: Path, n: int = 150, seed: int = 2026) -> pd.DataFrame:
    """Stratified sample for the author to label: every category and period represented."""
    rng = np.random.default_rng(seed)
    df = classified.reset_index(drop=True).copy()
    df["primary"] = [next((c for c in CATEGORIES if r[c]), "none") for _, r in df.iterrows()]
    df["period"] = pd.cut(df["delivered"].dt.year, [2000, 2014, 2019, 2100], labels=["-2014", "2015-19", "2020-"])
    per = max(1, n // (len(CATEGORIES) + 1))
    picked = []
    for _, g in df.groupby(["primary"], observed=True):
        k = min(len(g), per)
        picked += list(rng.choice(g.index, size=k, replace=False))
    rest = df.index.difference(picked)
    if len(picked) < n and len(rest):
        picked += list(rng.choice(rest, size=min(n - len(picked), len(rest)), replace=False))
    sample = df.loc[rng.permutation(picked[:n])]
    sheet = pd.DataFrame({"id": sample.index, "empresa": sample["name"], "data": sample["delivered"].dt.date,
                          "assunto": sample["subject"], "link": sample["link"]})
    for c in CATEGORIES:
        sheet[LABELS_PT[c]] = ""
    sheet["nenhuma"] = ""
    sheet["observação"] = ""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sheet.to_excel(path, index=False)
    return sheet
