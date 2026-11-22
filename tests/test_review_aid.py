import pandas as pd
import pytest

from distress import events, review_aid


def adjudicated():
    return pd.DataFrame({
        "cd_cvm": [7, 7, 7, 7, 8],
        "date": pd.to_datetime(["2012-09-03", "2012-11-23", "2010-01-05", "2012-12-20", "2012-11-23"]),
        "source": ["material_fact", "material_fact", "material_fact", "recovery_category", "material_fact"],
        "evidence": ["Aprovação do plano da controlada", "Pedido de recuperação judicial", "antigo",
                     "Relatório mensal", "outra empresa"],
        "link": ["L1", "L2", "L3", "L4", "L5"],
    })


def test_candidate_docs_start_with_the_event_link_and_add_the_firms_nearby_filings():
    ev = pd.Series({"cd_cvm": 7, "event_date": pd.Timestamp("2012-09-03"), "link": "L1"})
    docs = review_aid.candidate_docs(ev, adjudicated(), k=3)
    assert docs["link"].tolist() == ["L1", "L2", "L4"]  # own link, then nearest; other firm and 2010 left out


def test_candidate_docs_for_a_registry_event_without_link():
    ev = pd.Series({"cd_cvm": 7, "event_date": pd.Timestamp("2012-11-20"), "link": None})
    assert review_aid.candidate_docs(ev, adjudicated(), k=2)["link"].tolist() == ["L2", "L4"]


@pytest.mark.parametrize("obj, start", [
    ({"own_filing": "sim", "filing_date": "05/09/2012", "evidence": "a"}, "S"),
    ({"own_filing": "sim", "filing_date": "23/11/2012", "evidence": "a"}, "N, data do pedido 23/11/2012"),
    ({"own_filing": "não", "filer": "Renova Energia", "evidence": "a"}, "N, quem pediu: Renova Energia"),
    ({"own_filing": "incerto", "evidence": "a"}, "incerto"),
    ({"own_filing": "sim", "filing_date": None, "evidence": "a"}, "incerto"),
])
def test_suggestion_text(obj, start):
    s = review_aid.suggestion_text(pd.Timestamp("2012-09-03"), review_aid.parse_suggestion(obj))
    assert s.startswith(start)


def test_suggestions_go_into_a_new_column_of_both_tabs_and_leave_the_answers_alone(tmp_path):
    from openpyxl import load_workbook

    ev = pd.DataFrame({"cd_cvm": [1, 2], "name": ["ACME SA - EM RECUPERAÇÃO JUDICIAL", "GAMMA SA"],
                       "event_date": pd.to_datetime(["2016-06-20", "2019-12-17"]), "event_type": ["RJ", "RJ"],
                       "evidence": "x", "link": "http://x", "episode_end": pd.NaT, "n_prior_events": 0,
                       "left_truncated": False})
    path = tmp_path / "eventos.xlsx"
    events.review_sheet(ev, path)
    wb = load_workbook(path)
    wb["rápidas"]["G2"] = "S"
    wb.save(path)
    review_aid.write_suggestions(path, {(1, pd.Timestamp("2016-06-20").date(), "RJ"): "S (pedido em 20/06/2016)",
                                        (2, pd.Timestamp("2019-12-17").date(), "RJ"): "N, quem pediu: Renova"})
    tabs = pd.read_excel(path, sheet_name=None)
    assert tabs["rápidas"]["sugestão"].tolist() == ["S (pedido em 20/06/2016)"]
    assert tabs["conferir com cuidado"]["sugestão"].tolist() == ["N, quem pediu: Renova"]
    assert tabs["rápidas"]["confere (S/N)"].tolist() == ["S"]
    assert len(load_workbook(path)["rápidas"].data_validations.dataValidation) == 1


def test_prompt_gives_the_cnpj_so_a_former_name_counts_as_the_same_company():
    user = review_aid.suggest_user("EQUATORIAL PARÁ DISTRIBUIDORA", "04.895.728/0001-80",
                                   pd.Timestamp("2012-02-28"), "RJ", ["--- Documento A", "--- Documento B"])
    assert user.startswith("EMPRESA: EQUATORIAL PARÁ DISTRIBUIDORA\nCNPJ: 04.895.728/0001-80\n")
    assert "DATA NA LISTA: 28/02/2012\nTIPO NA LISTA: RJ\n\n--- Documento A\n\n--- Documento B" in user
    assert "CNPJ" not in review_aid.suggest_user("LAEP", None, pd.Timestamp("2009-08-26"), "RJ", ["x"])
    assert "nome antigo" in review_aid.SUGGEST_SYSTEM


@pytest.mark.parametrize("text, answer, fixed", [
    ("S (pedido em 05/09/2012) | ajuizou", True, None),
    ("N, data do pedido 23/11/2012 | Ajuizamento", False, "2012-11-23"),
    ("N, quem pediu: Renova Energia S.A. | sua coligada", False, None),
    ("incerto: ler o documento | x", None, None),
    ("incerto: nenhum documento legível perto da data", None, None),
])
def test_decide_reads_the_suggested_answer(text, answer, fixed):
    assert review_aid.decide(text) == (answer, None if fixed is None else pd.Timestamp(fixed))


def test_second_reader_review_moves_drops_and_keeps_events():
    draft = pd.DataFrame({
        "cd_cvm": [7, 8, 9, 9, 10, 11],
        "event_date": pd.to_datetime(["2012-09-03", "2014-02-03", "2013-04-15", "2021-08-23", "2019-12-17",
                                      "2005-03-01"]),
        "event_type": ["RJ", "RJ", "RJ", "RJ", "RJ", "RJ"],
    })
    sug = pd.DataFrame({
        "cd_cvm": [7, 8, 9, 9, 10],
        "event_date": draft["event_date"][:5].dt.strftime("%Y-%m-%d"),
        "event_type": "RJ",
        "suggestion": ["N, data do pedido 23/11/2012 | a", "incerto: nenhum documento legível perto da data",
                       "S (pedido em 15/04/2013) | b", "N, data do pedido 09/04/2013 | c",
                       "N, quem pediu: Renova Energia S.A. | d"],
    })
    out = events.apply_review(draft, review_aid.rule_sheet(draft, sug))
    got = {(r.cd_cvm, r.event_date.strftime("%Y-%m-%d"), r.review) for r in out.itertuples()}
    assert got == {(7, "2012-11-23", "date_corrected"),  # date moved to the petition
                   (8, "2014-02-03", "confirmed"),        # second reader unsure: the draft stands
                   (9, "2013-04-15", "confirmed"),        # the 2021 row pointed back to this petition
                   (11, "2005-03-01", "not_in_sheet")}    # firm 10 dropped: another company filed
    assert out.loc[out.cd_cvm == 8, "review_note"].item().startswith("segundo leitor incerto")
