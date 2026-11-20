"""Decide whether a material fact announces the issuer's own recovery or bankruptcy.

Only the subject line is read. The answer is one of four statuses:

* ``filing``: the company (or a group that includes it) filed for, or obtained,
  judicial or extrajudicial recovery, bankruptcy or liquidation;
* ``ongoing``: the document is about the company's own process already under way
  (plan, creditors' meeting, court decision);
* ``exit``: the company's own process ended;
* ``other``: anything else (a client, a subsidiary alone, a rumour, a denial).

``filing`` and ``ongoing`` count as the company's own signal for dating an episode.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

from .client import LLM

STATUSES = {"filing", "ongoing", "exit", "other"}
TYPES = {"RJ", "RE", "FAL", "LIQ"}

SYSTEM = """Você classifica o assunto de Fatos Relevantes publicados por companhias abertas brasileiras na CVM.
Responda só com um objeto JSON: {"status": ..., "event_type": ...}.

status:
- "filing": a própria companhia emissora, ou um grupo que a inclui, pediu ou obteve recuperação judicial, recuperação extrajudicial, falência ou liquidação.
- "ongoing": trata do processo da própria companhia já em curso (plano, assembleia de credores, decisão judicial, aditamento, cumprimento).
- "exit": encerramento ou extinção do processo da própria companhia.
- "other": qualquer outra coisa: processo de cliente, fornecedor ou controlada sem a emissora, boato, esclarecimento negando o pedido, menção genérica.

event_type: "RJ" (recuperação judicial), "RE" (recuperação extrajudicial), "FAL" (falência), "LIQ" (liquidação) ou null quando status for "other"."""

USER = "Companhia emissora: {name}\nAssunto do Fato Relevante: {subject}"


def event_classifier(llm: LLM) -> Callable[[pd.Series], dict]:
    def classify(row: pd.Series) -> dict:
        ans = llm.complete(SYSTEM, USER.format(name=row["name"], subject=row["evidence"]), schema={"type": "object"})
        status = ans.get("status") if isinstance(ans, dict) else None
        etype = ans.get("event_type") if isinstance(ans, dict) else None
        if status not in STATUSES:
            return {"own_filing": False, "event_type": None, "status": "other"}
        etype = etype if etype in TYPES and status != "other" else None
        return {"own_filing": status in ("filing", "ongoing"), "event_type": etype, "status": status}

    return classify
