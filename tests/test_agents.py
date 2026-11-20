import json

import pandas as pd
import pytest

from distress.agents import guardrail
from distress.agents.orchestrator import credit_memo
from distress.agents.tools import Tools
from distress.llm.client import LLM


class Transport:
    """Plays the reader and the writer: returns canned text depending on the prompt."""

    def __init__(self, memo_text):
        self.memo_text, self.calls = memo_text, []

    def __call__(self, payload):
        self.calls.append(payload)
        sys = payload["messages"][0]["content"]
        content = "Sinais recentes: renegociação de dívida e waiver de covenant." if "leitor" in sys else self.memo_text
        return {"choices": [{"message": {"content": content}}], "usage": {"cost": 0.0}}


@pytest.fixture
def tools():
    panel = pd.DataFrame({"cd_cvm": [7, 7], "q": pd.to_datetime(["2021-12-31", "2022-03-31"]),
                          "lev": [0.71, 0.78], "curr": [1.1, 0.9], "int_cov": [1.5, 0.8], "z2": [4.2, 3.6],
                          "selic": [9.15, 11.65], "pj_default": [1.9, 2.0]})
    docs = pd.DataFrame({"cd_cvm": [7, 7, 7], "delivered": pd.to_datetime(["2021-11-10", "2022-02-15", "2022-06-01"]),
                         "subject": ["Reperfilamento de dívidas", "Waiver de covenants", "Pedido de recuperação judicial"],
                         "debt_restructuring": [1, 0, 0], "covenant": [0, 1, 0], "own_recovery": [0, 0, 1]})
    pd_fn = lambda rows: pd.DataFrame({"pd12": [0.183], "pd24": [0.291], "pd36": [0.352]})
    drivers_fn = lambda rows: [("int_cov", 0.9), ("lev", 0.6)]
    return Tools(panel=panel, docs=docs, pd_fn=pd_fn, drivers_fn=drivers_fn, name="EMPRESA X")


def test_disclosures_never_come_from_after_the_as_of_date(tools):
    d = tools.recent_disclosures(7, pd.Timestamp("2022-04-01"))
    assert [x["subject"] for x in d] == ["Waiver de covenants", "Reperfilamento de dívidas"]


def test_memo_that_copies_tool_numbers_passes(tools, tmp_path):
    text = ("Em 31/03/2022 a alavancagem era 0,78 e a cobertura de juros 0,8. A PD de 12 meses é 18,3% "
            "e a de 36 meses 35,2%. Selic de 11,65%.")
    memo = credit_memo(7, pd.Timestamp("2022-04-01"), LLM("m", cache_dir=tmp_path, transport=Transport(text)), tools)
    assert not memo.blocked, memo.unsupported
    assert "18,3%" in memo.text


def test_memo_with_an_invented_number_is_blocked_and_names_it(tools, tmp_path):
    text = "A PD de 12 meses é 18,3%, e a dívida líquida soma R$ 4,2 bilhões."
    memo = credit_memo(7, pd.Timestamp("2022-04-01"), LLM("m", cache_dir=tmp_path, transport=Transport(text)), tools)
    assert memo.blocked
    assert "4,2" in memo.unsupported
    assert memo.text == ""


def test_invented_date_is_blocked(tools, tmp_path):
    text = "Em 15/09/2021 a companhia renegociou dívidas; PD de 12 meses 18,3%."
    memo = credit_memo(7, pd.Timestamp("2022-04-01"), LLM("m", cache_dir=tmp_path, transport=Transport(text)), tools)
    assert memo.blocked and "15/09/2021" in memo.unsupported


def test_number_normalisation_accepts_equivalent_formats():
    allowed = guardrail.allowed_numbers({"pd12": 0.183, "lev": 0.78, "selic": 11.65})
    assert guardrail.check("PD 18,3% e 18.3%, alavancagem 0.78 e 78%, Selic 11,65%", allowed, set()) == []
