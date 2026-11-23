import pandas as pd
import pytest

from distress import events


def cad_row(cd, status="FASE OPERACIONAL", date=None, name="ACME SA"):
    return {"cd_cvm": cd, "cnpj": "00", "name": name, "status": "ATIVO", "status_date": pd.NaT,
            "cancel_date": pd.NaT, "cancel_reason": None, "issuer_status": status,
            "issuer_status_date": pd.Timestamp(date) if date else pd.NaT, "sector": "Comércio",
            "register_date": pd.Timestamp("2000-01-01")}


def ipe_row(cd, category, subject, date, name="ACME SA"):
    return {"cd_cvm": cd, "cnpj": "00", "name": name, "category": category, "type": "", "subject": subject,
            "delivered": pd.Timestamp(date), "link": "http://x"}


RJ_CAT = "Informações de Companhias em Recuperação Judicial ou Extrajudicial"


def own_filing_classifier(row):
    s = row["evidence"].lower()
    own = "pedido de recuperação judicial" in s and "controlada" not in s
    return {"own_filing": own, "event_type": "RJ" if own else None}


def test_candidates_come_from_the_three_sources():
    cad = pd.DataFrame([cad_row(1, "EM RECUPERAÇÃO JUDICIAL OU EQUIVALENTE", "2024-03-10")])
    ipe = pd.DataFrame([
        ipe_row(1, "Fato Relevante", "Pedido de Recuperação Judicial", "2024-02-20"),
        ipe_row(1, RJ_CAT, "", "2024-04-05"),
        ipe_row(1, "Fato Relevante", "Resultado do trimestre", "2024-01-10"),
    ])
    c = events.candidate_events(cad, ipe)
    assert set(c["source"]) == {"registry", "material_fact", "recovery_category"}
    assert len(c) == 3  # the unrelated material fact is not a candidate


def test_earliest_own_filing_wins():
    cad = pd.DataFrame([cad_row(1, "EM RECUPERAÇÃO JUDICIAL OU EQUIVALENTE", "2024-03-10")])
    ipe = pd.DataFrame([
        ipe_row(1, "Fato Relevante", "Pedido de Recuperação Judicial", "2024-02-20"),
        ipe_row(1, RJ_CAT, "", "2024-04-05"),
    ])
    ev = events.build_events(events.adjudicate(events.candidate_events(cad, ipe), own_filing_classifier))
    assert len(ev) == 1
    assert ev["event_date"].iloc[0] == pd.Timestamp("2024-02-20")
    assert ev["event_type"].iloc[0] == "RJ"


def test_subsidiary_mention_is_not_an_event():
    cad = pd.DataFrame([cad_row(2)])
    ipe = pd.DataFrame([ipe_row(2, "Fato Relevante", "Pedido de recuperação judicial de controlada", "2024-02-20")])
    ev = events.build_events(events.adjudicate(events.candidate_events(cad, ipe), own_filing_classifier))
    assert ev.empty


def test_firm_in_recovery_at_window_start_is_left_truncated():
    cad = pd.DataFrame([cad_row(3)])
    ipe = pd.DataFrame([ipe_row(3, RJ_CAT, "", d) for d in ("2009-06-01", "2010-03-01", "2010-11-01", "2011-05-01")])
    ev = events.build_events(events.adjudicate(events.candidate_events(cad, ipe), own_filing_classifier))
    assert ev["left_truncated"].tolist() == [True]


def test_finished_recovery_before_window_is_history_not_truncation():
    cad = pd.DataFrame([cad_row(4)])
    ipe = pd.DataFrame([ipe_row(4, RJ_CAT, "", d) for d in ("2005-01-01", "2005-08-01")]
                       + [ipe_row(4, "Fato Relevante", "Pedido de Recuperação Judicial", "2016-06-20")])
    ev = events.build_events(events.adjudicate(events.candidate_events(cad, ipe), own_filing_classifier))
    assert ev["left_truncated"].tolist() == [False, False]
    assert ev["n_prior_events"].tolist() == [0, 1]
    assert ev["event_date"].tolist() == [pd.Timestamp("2005-01-01"), pd.Timestamp("2016-06-20")]


def test_two_episodes_separated_by_more_than_a_year_without_filings():
    cad = pd.DataFrame([cad_row(5)])
    first = [ipe_row(5, "Fato Relevante", "Pedido de Recuperação Judicial", "2016-06-20")]
    first += [ipe_row(5, RJ_CAT, "", f"2016-{m:02d}-15") for m in range(7, 13)]
    second = [ipe_row(5, "Fato Relevante", "Pedido de Recuperação Judicial", "2023-03-01")]
    ev = events.build_events(events.adjudicate(events.candidate_events(cad, pd.DataFrame(first + second)),
                                               own_filing_classifier))
    assert ev["n_prior_events"].tolist() == [0, 1]
    assert ev["episode_end"].iloc[0] == pd.Timestamp("2016-12-15")


def status_classifier(row):
    s = row["evidence"].lower()
    if "encerramento" in s:
        return {"own_filing": False, "event_type": "RJ", "status": "exit"}
    if "pedido de recuperação extrajudicial" in s:
        return {"own_filing": True, "event_type": "RE", "status": "filing"}
    if "pedido de recuperação judicial" in s:
        return {"own_filing": True, "event_type": "RJ", "status": "filing"}
    return {"own_filing": False, "event_type": None, "status": "other"}


def run(ipe_rows, cad_rows=None):
    cad = pd.DataFrame(cad_rows or [cad_row(9)])
    return events.build_events(events.adjudicate(events.candidate_events(cad, pd.DataFrame(ipe_rows)), status_classifier))


def test_exit_then_new_filing_within_a_year_is_a_new_episode():
    rows = [ipe_row(9, "Fato Relevante", "Pedido de Recuperação Judicial", "2016-06-20")]
    rows += [ipe_row(9, RJ_CAT, "", f"20{y}-{m:02d}-10") for y in (17, 18, 19, 20, 21, 22) for m in (3, 9)]
    rows += [ipe_row(9, "Fato Relevante", "Encerramento da recuperação judicial", "2022-12-14"),
             ipe_row(9, "Fato Relevante", "Pedido de Recuperação Judicial", "2023-03-01")]
    ev = run(rows)
    assert ev["event_date"].tolist() == [pd.Timestamp("2016-06-20"), pd.Timestamp("2023-03-01")]


def test_sparse_monitoring_filings_do_not_open_new_episodes():
    rows = [ipe_row(9, "Fato Relevante", "Pedido de Recuperação Judicial", "2011-04-06")]
    rows += [ipe_row(9, RJ_CAT, "", d) for d in ("2013-08-21", "2016-12-06", "2018-08-23")]
    ev = run(rows)
    assert len(ev) == 1 and ev["episode_end"].iloc[0] == pd.Timestamp("2018-08-23")


def test_extrajudicial_turning_judicial_is_a_new_event():
    rows = [ipe_row(9, "Fato Relevante", "Pedido de Recuperação Extrajudicial", "2024-04-29"),
            ipe_row(9, RJ_CAT, "", "2024-09-10"),
            ipe_row(9, "Fato Relevante", "Pedido de Recuperação Judicial", "2025-02-01")]
    ev = run(rows)
    assert ev["event_type"].tolist() == ["RE", "RJ"]


def test_current_registry_status_keeps_the_episode_open():
    # In recovery since 2010 by the registry, monthly reports stop in 2010: still in recovery, so left-truncated.
    cad = pd.DataFrame([cad_row(6, "EM RECUPERAÇÃO JUDICIAL OU EQUIVALENTE", "2010-01-01")])
    ipe = pd.DataFrame([ipe_row(6, RJ_CAT, "", d) for d in ("2010-03-01", "2010-06-01")])
    ev = events.build_events(events.adjudicate(events.candidate_events(cad, ipe), own_filing_classifier))
    assert ev["left_truncated"].tolist() == [True]
    assert ev["episode_end"].iloc[0] >= events.OPEN_END


def test_bankruptcy_is_absorbing():
    def classifier(row):
        if "falência" in row["evidence"].lower():
            return {"own_filing": True, "event_type": "FAL", "status": "filing"}
        return status_classifier(row)

    rows = [ipe_row(9, "Fato Relevante", "Decretação de falência", "2018-05-02"),
            ipe_row(9, "Fato Relevante", "Pedido de Recuperação Judicial", "2021-03-01")]
    ev = events.build_events(events.adjudicate(events.candidate_events(pd.DataFrame([cad_row(9)]),
                                                                       pd.DataFrame(rows)), classifier))
    assert ev["event_type"].tolist() == ["FAL"] and ev["episode_end"].iloc[0] >= events.OPEN_END


def test_liquidation_is_not_a_default_event():
    ev = run([], [cad_row(9, "LIQUIDAÇÃO EXTRAJUDICIAL", "2024-12-03")])
    assert ev.empty


def test_non_financial_filter_survives_duplicated_registry_rows():
    cad = pd.DataFrame([cad_row(1), cad_row(1), {**cad_row(2), "sector": "Bancos"}])
    ev = pd.DataFrame({"cd_cvm": [1, 2], "event_date": pd.to_datetime(["2020-01-01", "2020-01-01"])})
    assert events.non_financial(ev, cad)["cd_cvm"].tolist() == [1]


def test_review_sheet_has_the_columns_rodrigo_fills(tmp_path):
    ev = pd.DataFrame([{"cd_cvm": 1, "name": "ACME SA", "event_date": pd.Timestamp("2024-02-20"),
                        "event_type": "RJ", "evidence": "Pedido de Recuperação Judicial", "link": "http://x",
                        "episode_end": pd.Timestamp("2024-06-01"), "n_prior_events": 0, "left_truncated": False}])
    path = tmp_path / "eventos.xlsx"
    events.review_sheet(ev, path)
    back = pd.read_excel(path)
    assert list(back.columns) == ["código CVM", "empresa", "data", "tipo", "evidência", "link",
                                  "confere (S/N)", "data correta", "observação"]
    assert back["empresa"].iloc[0] == "ACME SA"


def review_inputs():
    draft = pd.DataFrame({"cd_cvm": [1, 2, 3, 4], "name": ["A", "B", "C", "D"],
                          "event_date": pd.to_datetime(["2005-03-01", "2016-06-20", "2019-02-10", "2023-01-19"]),
                          "event_type": ["RJ", "RJ", "RE", "RJ"], "evidence": "x", "link": "y",
                          "episode_end": pd.NaT, "n_prior_events": 0, "left_truncated": False})
    sheet = pd.DataFrame({"código CVM": [2, 3, 4], "empresa": ["B", "C", "D"],
                          "data": ["2016-06-20", "2019-02-10", "2023-01-19"], "tipo": ["RJ", "RE", "RJ"],
                          "confere (S/N)": ["S", "n", "Não"], "data correta": ["", "", "12/01/2023"],
                          "observação": ["", "cliente, não a própria", "data do pedido"]})
    return draft, sheet


def test_review_confirms_corrects_and_drops_events():
    from distress.events import apply_review

    draft, sheet = review_inputs()
    out = apply_review(draft, sheet).set_index("cd_cvm")
    assert list(out.index) == [1, 2, 4]                       # C dropped; A (before the sheet) kept as is
    assert out.loc[4, "event_date"] == pd.Timestamp("2023-01-12") and out.loc[4, "review"] == "date_corrected"
    assert out.loc[2, "review"] == "confirmed" and out.loc[1, "review"] == "not_in_sheet"
    assert out.loc[4, "review_note"] == "data do pedido"


def test_review_drop_keeps_a_later_own_filing_of_the_same_episode():
    from distress.events import apply_review

    # The episode opened with a creditor's bankruptcy request, which the review rejects; the company's own
    # judicial-recovery filing three years later, inside the same episode, must survive as an event.
    draft = pd.DataFrame({"cd_cvm": [7], "name": ["G"], "event_date": pd.to_datetime(["2019-03-01"]),
                          "event_type": ["FAL"], "evidence": "x", "link": "y",
                          "episode_end": pd.to_datetime(["2022-09-01"]), "n_prior_events": 0, "left_truncated": False})
    sheet = pd.DataFrame({"código CVM": [7], "empresa": ["G"], "data": ["2019-03-01"], "tipo": ["FAL"],
                          "confere (S/N)": ["N"], "data correta": [""], "observação": ["pedido de credor"]})
    signals = pd.DataFrame({"cd_cvm": 7, "name": "G", "source": ["material_fact", "material_fact", "registry"],
                            "date": pd.to_datetime(["2019-03-01", "2022-05-10", "2022-05-12"]),
                            "evidence": ["Pedido de falência por credor", "Pedido de recuperação judicial",
                                         "EM RECUPERAÇÃO JUDICIAL"], "link": "z",
                            "event_type": ["FAL", "RJ", "RJ"], "own_filing": True, "status": "filing"})
    out = apply_review(draft, sheet, signals)
    assert out["event_date"].tolist() == [pd.Timestamp("2022-05-10")]
    assert out["event_type"].tolist() == ["RJ"] and out["review"].tolist() == ["rebuilt_after_drop"]


def test_review_recomputes_episode_fields():
    from distress.events import apply_review

    # Firm 1: the 2010 event moves into 2011, so it is no longer left-truncated. Firm 2: its first event is
    # dropped, so the second is now its first.
    draft = pd.DataFrame({"cd_cvm": [1, 2, 2], "name": "A",
                          "event_date": pd.to_datetime(["2010-12-01", "2015-01-10", "2019-04-01"]),
                          "event_type": "RJ", "evidence": "x", "link": "y",
                          "episode_end": pd.to_datetime(["2011-03-01", "2015-06-01", "2019-09-01"]),
                          "n_prior_events": [0, 0, 1], "left_truncated": [True, False, False]})
    sheet = pd.DataFrame({"código CVM": [1, 2, 2], "empresa": "A", "data": ["2010-12-01", "2015-01-10", "2019-04-01"],
                          "tipo": "RJ", "confere (S/N)": ["N", "N", "S"], "data correta": ["01/02/2011", "", ""],
                          "observação": ""})
    out = apply_review(draft, sheet).set_index(["cd_cvm", "event_date"])
    assert not out.loc[(1, pd.Timestamp("2011-02-01")), "left_truncated"]
    assert out.loc[(2, pd.Timestamp("2019-04-01")), "n_prior_events"] == 0


def test_review_with_blank_answers_is_refused():
    from distress.events import apply_review

    draft, sheet = review_inputs()
    sheet.loc[0, "confere (S/N)"] = ""
    with pytest.raises(ValueError, match="1 row"):
        apply_review(draft, sheet)


def _three_events():
    names = ["ACME SA - EM RECUPERAÇÃO JUDICIAL", "MASSA FALIDA DE BETA SA", "GAMMA ENERGIA SA"]
    return pd.DataFrame({"cd_cvm": [1, 2, 3], "name": names,
                         "event_date": pd.to_datetime(["2016-06-20", "2013-01-09", "2019-12-17"]),
                         "event_type": ["RJ", "FAL", "RJ"], "evidence": "x", "link": "http://x",
                         "episode_end": pd.NaT, "n_prior_events": 0, "left_truncated": False})


def test_review_sheet_puts_events_the_name_does_not_confirm_on_the_first_tab(tmp_path):
    path = tmp_path / "eventos.xlsx"
    events.review_sheet(_three_events(), path)
    tabs = pd.read_excel(path, sheet_name=None)
    assert list(tabs) == ["conferir com cuidado", "rápidas"]
    assert tabs["conferir com cuidado"]["código CVM"].tolist() == [3]
    assert tabs["rápidas"]["código CVM"].tolist() == [1, 2]


def test_review_answers_are_read_from_every_tab(tmp_path):
    from openpyxl import load_workbook

    path = tmp_path / "eventos.xlsx"
    ev = _three_events()
    events.review_sheet(ev, path)
    wb = load_workbook(path)
    for ws in wb.worksheets:
        col = [c.value for c in ws[1]].index("confere (S/N)") + 1
        for row in range(2, ws.max_row + 1):
            ws.cell(row=row, column=col, value="S")
    wb.save(path)
    out = events.apply_review(ev, events.read_review(path))
    assert sorted(out["cd_cvm"]) == [1, 2, 3] and set(out["review"]) == {"confirmed"}
