import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from distress import cvm, features


def rows(cd, period, receb, values, version=1, consolidated=True, ytd_start=None):
    """Long statement rows. Flow accounts (code 3.*) carry a YTD start date."""
    out = []
    for code, v in values.items():
        out.append({"cd_cvm": cd, "dt_refer": period, "version": version, "consolidated": consolidated,
                    "code": code, "value": float(v), "dt_receb": pd.Timestamp(receb),
                    "dt_ini": (ytd_start or period[:4] + "-01-01") if code.startswith("3") else None})
    return out


STOCKS = {"1": 1000, "1.01": 400, "1.01.01": 50, "2.01": 250, "2.02": 350, "2.03": 400,
          "2.03.04": 100, "2.03.05": 50}
FLOWS = {"3.01": 800, "3.05": 120, "3.06.02": -40, "3.11": 60}


def annual_firm(cd=10, consolidated=True):
    return pd.DataFrame(rows(cd, "2019-12-31", "2020-03-20", {**STOCKS, **FLOWS}, consolidated=consolidated))


def test_ratios_on_a_hand_built_statement():
    r = features.ratios(features.point_in_time(annual_firm())).iloc[0]
    assert r["wc_ta"] == pytest.approx(0.15)
    assert r["re_ta"] == pytest.approx(0.15)
    assert r["ebit_ta"] == pytest.approx(0.12)
    assert r["eq_tl"] == pytest.approx(400 / 600)
    assert r["lev"] == pytest.approx(0.6)
    assert r["curr"] == pytest.approx(1.6)
    assert r["cash_ta"] == pytest.approx(0.05)
    assert r["int_cov"] == pytest.approx(3.0)
    assert r["margin"] == pytest.approx(0.075)
    assert r["neg_eq"] == 0


def test_altman_emerging_market_score():
    r = features.ratios(features.point_in_time(annual_firm()))
    z = features.altman_z2(r).iloc[0]
    assert z == pytest.approx(3.25 + 6.56 * 0.15 + 3.26 * 0.15 + 6.72 * 0.12 + 1.05 * (400 / 600))


def test_individual_only_firm_gets_values():
    pit = features.point_in_time(annual_firm(cd=11, consolidated=False))
    assert pit["ta"].iloc[0] == 1000


def test_consolidated_preferred_over_individual():
    both = pd.concat([annual_firm(consolidated=True),
                      pd.DataFrame(rows(10, "2019-12-31", "2020-03-20", {"1": 999}, consolidated=False))])
    assert features.point_in_time(both)["ta"].iloc[0] == 1000


def test_restatement_is_invisible_before_its_own_filing():
    v1 = annual_firm()
    v2 = pd.DataFrame(rows(10, "2019-12-31", "2020-05-10", {**STOCKS, "1": 800, **FLOWS}, version=2))
    pit = features.point_in_time(pd.concat([v1, v2])).set_index("as_of")
    assert pit.loc[pd.Timestamp("2020-03-20"), "ta"] == 1000  # what was known in April 2020
    assert pit.loc[pd.Timestamp("2020-05-10"), "ta"] == 800
    assert pit.loc[pd.Timestamp("2020-05-10"), "n_restatements_4q"] == 1


def test_trailing_twelve_months_from_year_to_date():
    st = pd.concat([
        pd.DataFrame(rows(10, "2019-03-31", "2019-05-10", {**STOCKS, "3.01": 200, "3.05": 30, "3.06.02": -10, "3.11": 15})),
        annual_firm(),
        pd.DataFrame(rows(10, "2020-03-31", "2020-05-12", {**STOCKS, "3.01": 210, "3.05": 25, "3.06.02": -12, "3.11": 5})),
    ])
    pit = features.point_in_time(st).set_index("as_of")
    q = pit.loc[pd.Timestamp("2020-05-12")]
    assert q["rev_ttm"] == 810  # 210 + 800 - 200
    assert q["dt_refer"] == pd.Timestamp("2020-03-31")
    assert q["filing_delay_days"] == 42
    assert np.isnan(q["rev_ttm_prev"])  # 2018 annual not available, so no growth


def test_going_concern_rule_separates_doubt_from_boilerplate():
    p = pd.DataFrame({"cd_cvm": [1, 2], "dt_refer": ["2024-03-31"] * 2, "dt_receb": pd.to_datetime(["2024-05-10"] * 2),
                      "kind": ["Relatório da Revisão Especial"] * 2,
                      "text": ["Chamamos a atenção para a nota 1, que indica a existência de incerteza relevante "
                               "relacionada com a continuidade operacional da Companhia.",
                               "A administração avaliou a continuidade operacional e não identificou incertezas."]})
    gc = features.going_concern_rule(p).set_index("cd_cvm")["gc_flag"]
    assert gc[1] and not gc[2]


def test_read_parecer_links_text_to_filing_date(tmp_path):
    idx = ("CNPJ_CIA;DT_REFER;VERSAO;DENOM_CIA;CD_CVM;CATEG_DOC;ID_DOC;DT_RECEB;LINK_DOC\n"
           "00.000.000/0001-00;2024-03-31;1;ACME;001234;ITR;1;2024-05-10;x\n")
    par = ("CNPJ_CIA;DT_REFER;VERSAO;DENOM_CIA;TP_RELAT_ESP;TP_PARECER_DECL;NUM_ITEM_PARECER_DECL;TXT_PARECER_DECL\n"
           "00.000.000/0001-00;2024-03-31;1;ACME;;Relatório da Revisão Especial;1;incerteza relevante\n")
    with zipfile.ZipFile(tmp_path / "itr_cia_aberta_2024.zip", "w") as z:
        z.writestr("itr_cia_aberta_2024.csv", idx.encode("latin1"))
        z.writestr("itr_cia_aberta_parecer_2024.csv", par.encode("latin1"))
    p = cvm.read_parecer("itr", [2024], tmp_path)
    assert p[["cd_cvm", "dt_receb", "text"]].iloc[0].tolist() == [1234, pd.Timestamp("2024-05-10"), "incerteza relevante"]
