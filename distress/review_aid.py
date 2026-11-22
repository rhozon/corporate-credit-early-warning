"""Suggestions for the author's review of the event list.

A language model reads the filings near each draft event and suggests an answer: whether this company
itself filed, and on what date. The suggestion goes into its own column of the review sheet; the
author's S/N column is never written here, so every event is still decided by hand.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd

TOLERANCE = pd.Timedelta(days=7)  # a filing announced a few days after the petition keeps its draft date

SUGGEST_SYSTEM = """Você confere eventos de recuperação judicial, recuperação extrajudicial e falência de \
companhias abertas brasileiras. Recebe o nome de uma empresa, a data e o tipo que uma lista automática \
atribuiu a ela, e o texto de documentos que a empresa entregou à CVM perto dessa data.
Responda se ESTA empresa (não uma controlada, coligada, cliente ou outra empresa do grupo citada no texto) \
pediu recuperação judicial ou extrajudicial, ou teve falência decretada, e em que data foi o pedido \
(ajuizamento ou protocolo do pedido; não o deferimento, a aprovação do plano ou a homologação).
Os documentos foram entregues à CVM por esta empresa. Quando o texto diz "a Companhia" ou traz o CNPJ \
dela, fala desta empresa, mesmo com outro nome: empresas mudam de nome, e um nome antigo conta como esta \
empresa.
Campos:
- own_filing: "sim", "não" ou "incerto".
- filer: quem pediu, quando não foi esta empresa; senão null.
- filing_date: data do pedido no formato dd/mm/aaaa, se o texto disser; senão null.
- event_type: "RJ", "RE" ou "FAL", ou null.
- evidence: trecho curto do texto (até 200 caracteres) que sustenta a resposta.
Responda só JSON com exatamente essas chaves."""


def candidate_docs(ev: pd.Series, adjudicated: pd.DataFrame, k: int = 4, window_days: int = 365) -> pd.DataFrame:
    """The event's own document first, then the firm's other recovery-related filings nearest in time."""
    firm = adjudicated[(adjudicated["cd_cvm"] == ev["cd_cvm"]) & adjudicated["link"].notna()].copy()
    firm["date"] = pd.to_datetime(firm["date"])
    firm["gap"] = (firm["date"] - pd.Timestamp(ev["event_date"])).abs()
    own_link = ev.get("link")
    rest = firm[(firm["gap"] <= pd.Timedelta(days=window_days)) & (firm["link"] != own_link)]
    rest = rest.sort_values(["gap", "date"], kind="stable")
    rows = []
    if isinstance(own_link, str) and own_link:
        rows.append({"date": pd.Timestamp(ev["event_date"]), "evidence": ev.get("evidence"), "link": own_link})
    rows += rest[["date", "evidence", "link"]].to_dict("records")
    return pd.DataFrame(rows, columns=["date", "evidence", "link"]).drop_duplicates("link").head(k) \
        .reset_index(drop=True)


def suggest_user(name: str, cnpj: str | None, event_date, event_type: str, parts: list[str]) -> str:
    """The model's input for one event: the company (with its CNPJ, which survives a change of name),
    what the draft list says, and the documents."""
    head = f"EMPRESA: {name}\n" + (f"CNPJ: {cnpj}\n" if cnpj else "")
    return (head + f"DATA NA LISTA: {pd.Timestamp(event_date):%d/%m/%Y}\nTIPO NA LISTA: {event_type}\n\n"
            + "\n\n".join(parts))


def _date(x) -> pd.Timestamp | None:
    if not x:
        return None
    d = pd.to_datetime(str(x), dayfirst=True, errors="coerce")
    return None if pd.isna(d) else d


def parse_suggestion(obj) -> dict:
    obj = obj if isinstance(obj, dict) else {}
    own = str(obj.get("own_filing") or "incerto").strip().lower().replace("nao", "não")
    return {"own": own if own in ("sim", "não") else "incerto",
            "filer": (str(obj["filer"]).strip() or None) if obj.get("filer") else None,
            "date": _date(obj.get("filing_date")),
            "type": obj.get("event_type") if obj.get("event_type") in ("RJ", "RE", "FAL") else None,
            "evidence": str(obj.get("evidence") or "").strip()[:200]}


def suggestion_text(event_date, s: dict) -> str:
    """One line for the sheet: the suggested answer, then the passage that supports it."""
    if s["own"] == "sim" and s["date"] is not None:
        d = s["date"].strftime("%d/%m/%Y")
        head = f"S (pedido em {d})" if abs(s["date"] - pd.Timestamp(event_date)) <= TOLERANCE \
            else f"N, data do pedido {d}"
    elif s["own"] == "não":
        head = f"N, quem pediu: {s['filer'] or 'outra empresa'}"
    else:
        head = "incerto: ler o documento"
    return head + (f" | {s['evidence']}" if s["evidence"] else "")


def _day(v):
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    return pd.to_datetime(v).date()


def write_suggestions(path: Path, texts: dict) -> int:
    """Fill a "sugestão" column in every tab, matching rows by (CVM code, date, type)."""
    from openpyxl import load_workbook

    wb = load_workbook(path)
    n = 0
    for ws in wb.worksheets:
        head = [c.value for c in ws[1]]
        if "sugestão" not in head:
            ws.cell(row=1, column=len(head) + 1, value="sugestão").font = ws.cell(row=1, column=1).font.copy()
            head.append("sugestão")
        col = {h: i + 1 for i, h in enumerate(head)}
        ws.column_dimensions[ws.cell(row=1, column=col["sugestão"]).column_letter].width = 80
        for row in range(2, ws.max_row + 1):
            key = (int(ws.cell(row=row, column=col["código CVM"]).value),
                   _day(ws.cell(row=row, column=col["data"]).value), ws.cell(row=row, column=col["tipo"]).value)
            if key in texts:
                ws.cell(row=row, column=col["sugestão"], value=texts[key])
                n += 1
    wb.save(path)
    return n
