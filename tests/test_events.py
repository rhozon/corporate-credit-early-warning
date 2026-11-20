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
