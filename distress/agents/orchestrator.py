"""Credit memo written by agents, checked by a deterministic critic.

The orchestrator gathers facts with the tools (panel, PD curve, drivers, macro,
disclosures). A reader subagent summarises the recent material facts; a writer
subagent turns the facts into a memo for a credit committee; the critic extracts
every number and date from the memo and blocks it if any of them is not in the
facts. A blocked memo is not shown, and the offending values are reported.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import pandas as pd

from ..llm.client import LLM
from . import guardrail
from .tools import Tools

READER = """Você é o subagente leitor de um comitê de crédito. Recebe a lista dos fatos relevantes recentes de uma
companhia aberta, com data e assunto. Resuma em até 4 frases os sinais de estresse financeiro que eles indicam
(renegociação de dívida, covenants, atrasos, venda de ativos, aumento de capital, troca de diretoria, rating).
Não escreva números nem datas. Se não houver sinal, diga isso."""

WRITER = """Você é o subagente redator de um comitê de crédito. Escreva um parecer de crédito em português, com até
6 parágrafos curtos: situação financeira, probabilidade de default, principais fatores, divulgações recentes,
contexto macroeconômico e recomendação de monitoramento.
Regra obrigatória: use SOMENTE números e datas que aparecem nos FATOS, escritos como estão ou arredondados
(probabilidades podem ser escritas em %). Não calcule nada novo. Não invente valores, datas ou nomes.
Os fatores de risco vêm em ordem de peso no modelo: cite-os pelo nome; o valor de um fator, quando houver, está
nos indicadores ou no contexto macro."""


@dataclass
class Memo:
    text: str
    draft: str
    facts: dict
    blocked: bool
    unsupported: list = field(default_factory=list)


def gather(cd, as_of: pd.Timestamp, tools: Tools) -> dict:
    return {"empresa": tools.name, "data_da_analise": as_of.strftime("%d/%m/%Y"),
            "indicadores": tools.panel_row(cd, as_of), "probabilidade_de_default": tools.pd_curve(cd, as_of),
            "fatores_de_risco": tools.top_drivers(cd, as_of), "macro": tools.macro_context(cd, as_of),
            "fatos_relevantes": tools.recent_disclosures(cd, as_of),
            "legenda": {"pd12": "PD 12 meses", "pd24": "PD 24 meses", "pd36": "PD 36 meses"}}


def credit_memo(cd, as_of: pd.Timestamp, llm: LLM, tools: Tools) -> Memo:
    facts = gather(cd, as_of, tools)
    docs = "\n".join(f"- {d['data']}: {d['subject']}" for d in facts["fatos_relevantes"]) or "(nenhum)"
    summary = llm.complete(READER, docs)
    payload = json.dumps(facts, ensure_ascii=False, indent=1)
    draft = llm.complete(WRITER, f"FATOS:\n{payload}\n\nRESUMO DO LEITOR:\n{summary}")

    numbers = {**facts["indicadores"], **facts["probabilidade_de_default"], **facts["macro"]}
    allowed = guardrail.allowed_numbers({k: v for k, v in numbers.items() if not isinstance(v, str)})
    allowed |= {v for d in facts["fatos_relevantes"] for v in guardrail.numbers_in(d["subject"])}
    dates = {facts["indicadores"]["data_base"], facts["data_da_analise"]} | {d["data"] for d in facts["fatos_relevantes"]}
    bad = guardrail.check(draft, allowed, dates)
    return Memo(text="" if bad else draft, draft=draft, facts=facts, blocked=bool(bad), unsupported=bad)
