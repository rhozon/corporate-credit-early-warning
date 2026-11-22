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


def test_past_case_memo_is_dated_twelve_months_before_the_filing():
    from distress.agents import cases

    assert cases.analysis_date(pd.Timestamp("2023-01-19")) == pd.Timestamp("2022-01-19")


def test_drivers_are_the_features_that_push_the_hazard_up_most():
    import numpy as np

    from distress.agents import cases
    from distress.models.classic import HazardLogit

    rng = np.random.default_rng(1)
    n = 3000
    train = pd.DataFrame({"lev": rng.normal(size=n), "curr": rng.normal(size=n), "noise": rng.normal(size=n),
                          "q": pd.Timestamp("2015-03-31")})
    train["event"] = (rng.random(n) < 1 / (1 + np.exp(-(-3 + 2 * train["lev"] - 1.5 * train["curr"])))).astype(int)
    m = HazardLogit(features=["lev", "curr", "noise"], fast=True).fit(train, pd.Timestamp("2019-12-31"))
    row = pd.DataFrame({"lev": [2.0], "curr": [-1.0], "noise": [0.0]})
    got = cases.logit_drivers(m, row, k=3)
    assert [f for f, _ in got[:2]] == ["lev", "curr"]  # high leverage and low liquidity both raise the hazard
    assert all(v > 0 for _, v in got)
    assert cases.altman_z2(pd.DataFrame({"wc_ta": [0.1], "re_ta": [0.2], "ebit_ta": [0.05], "eq_tl": [0.5]})).item() \
        == pytest.approx(3.25 + 0.656 + 0.652 + 0.336 + 0.525)


def test_a_blocked_memo_is_written_without_its_draft(tmp_path):
    from distress.agents import cases
    from distress.agents.orchestrator import Memo

    facts = {"empresa": "EMPRESA X", "data_da_analise": "19/01/2022", "probabilidade_de_default": {"pd12": 0.183}}
    path = tmp_path / "x.md"
    cases.write_memo(path, Memo(text="", draft="dívida de R$ 4,2 bilhões", facts=facts, blocked=True,
                                unsupported=["4,2"]), event="RJ em 19/01/2023")
    s = path.read_text(encoding="utf-8")
    assert "bloqueado" in s.lower() and "4,2" in s and "bilhões" not in s
    ok = tmp_path / "y.md"
    cases.write_memo(ok, Memo(text="Parecer: PD de 18,3%.", draft="Parecer: PD de 18,3%.", facts=facts,
                              blocked=False), event="RJ em 19/01/2023")
    assert "Parecer: PD de 18,3%." in ok.read_text(encoding="utf-8")


def _tools_with(macro: dict, subjects: list[str]):
    panel = pd.DataFrame({"cd_cvm": [7], "q": pd.to_datetime(["2022-03-31"]), "lev": [0.78], **{k: [v] for k, v in macro.items()}})
    docs = pd.DataFrame({"cd_cvm": 7, "delivered": pd.to_datetime(["2022-02-15"] * len(subjects)), "subject": subjects,
                         "own_recovery": 0})
    return Tools(panel=panel, docs=docs, pd_fn=lambda r: pd.DataFrame({"pd12": [0.183], "pd24": [0.291], "pd36": [0.352]}),
                 drivers_fn=lambda r: [("lev", 0.6)], name="EMPRESA X")


def test_a_negative_value_copied_with_its_sign_passes_and_a_flipped_sign_is_blocked(tmp_path):
    t = _tools_with({"ibc_yoy": -1.34, "margin": -0.01}, ["Fato relevante"])
    ok = "O IBC-Br caiu -1,34% e a margem líquida foi de -0.01. PD de 12 meses de 18,3%."
    memo = credit_memo(7, pd.Timestamp("2022-04-01"), LLM("m", cache_dir=tmp_path / "a", transport=Transport(ok)), t)
    assert not memo.blocked, memo.unsupported
    flipped = "O IBC-Br subiu 1,34%. PD de 12 meses de 18,3%."
    memo = credit_memo(7, pd.Timestamp("2022-04-01"), LLM("m", cache_dir=tmp_path / "b", transport=Transport(flipped)), t)
    assert memo.blocked and memo.unsupported == ["1,34"]


def test_numbers_inside_a_disclosure_subject_count_as_sourced(tmp_path):
    t = _tools_with({}, ["Prévia Operacional do 3º Trimestre do Ano-Safra 2024'25"])
    text = "A companhia divulgou a prévia operacional do ano-safra 2024'25 em 15/02/2022. PD de 12 meses de 18,3%."
    memo = credit_memo(7, pd.Timestamp("2022-04-01"), LLM("m", cache_dir=tmp_path, transport=Transport(text)), t)
    assert not memo.blocked, memo.unsupported


def test_facts_name_indicators_in_words_and_drivers_carry_no_number(tools):
    row = tools.panel_row(7, pd.Timestamp("2022-04-01"))
    assert row["alavancagem (passivo/ativo)"] == 0.78 and "lev" not in row
    assert tools.macro_context(7, pd.Timestamp("2022-04-01"))["Selic (% a.a.)"] == 11.65
    # A driver's weight in the model is not a value of the indicator; the writer gets the names only.
    assert tools.top_drivers(7, pd.Timestamp("2022-04-01")) == ["cobertura de juros", "alavancagem (passivo/ativo)"]


def test_memo_text_passes_through_the_cleaner_before_it_is_written(tmp_path):
    from distress.agents import cases
    from distress.agents.orchestrator import Memo

    facts = {"empresa": "EMPRESA X", "data_da_analise": "19/01/2022", "fatos_relevantes": [{"subject": "Parceria com Bradesco"}]}
    path = tmp_path / "x.md"
    cases.write_memo(path, Memo(text="Parceria com Bradesco.", draft="", facts=facts, blocked=False), event="RJ",
                     clean=lambda s: s.replace("Bradesco", "[outra companhia]"))
    assert "Bradesco" not in path.read_text(encoding="utf-8")
