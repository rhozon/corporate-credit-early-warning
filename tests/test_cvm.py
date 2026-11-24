from pathlib import Path

import pandas as pd

from distress import cvm

RAW = Path(__file__).parent / "fixtures" / "raw"


def test_read_cad_normalises_codes_and_keeps_status():
    cad = cvm.read_cad(RAW)
    assert {"cd_cvm", "cnpj", "name", "status", "issuer_status", "issuer_status_date", "sector"} <= set(cad.columns)
    assert cad["cd_cvm"].dtype.kind == "i"
    assert cad["issuer_status_date"].dtype.kind == "M"


def test_read_ipe_parses_dates_and_categories():
    ipe = cvm.read_ipe([2025], RAW)
    assert list(ipe.columns) == ["cd_cvm", "cnpj", "name", "category", "type", "subject", "delivered", "link"]
    assert ipe["delivered"].dtype.kind == "M"
    assert ipe["category"].str.contains("Recupera").any()
    assert ipe["category"].str.startswith("Fato Relevante").any()


def test_read_statements_keeps_current_period_and_attaches_filing_date():
    st = cvm.read_statements("itr", [2024], "BPA", RAW)
    assert list(st.columns) == ["cd_cvm", "dt_refer", "version", "consolidated", "code", "value", "dt_receb"]
    q1 = st[(st.dt_refer == "2024-03-31") & (st.code == "1") & st.consolidated]
    # only the current period ("ULTIMO") survives, scaled from thousands to reais
    assert len(q1) == 1
    assert q1["value"].iloc[0] == 1_067_292_000 * 1000
    assert q1["dt_receb"].iloc[0] == pd.Timestamp("2024-05-13")
    assert st["cd_cvm"].unique().tolist() == [9512]


def test_read_statements_marks_individual_rows():
    st = cvm.read_statements("itr", [2024], "BPA", RAW)
    assert set(st["consolidated"]) == {True, False}


def test_download_uses_cache_without_network(tmp_path, monkeypatch):
    target = tmp_path / "ipe_cia_aberta_2025.zip"
    target.write_bytes(b"cached")

    def boom(*a, **k):
        raise AssertionError("network called")

    monkeypatch.setattr(cvm, "_get", boom)
    assert cvm.download("ipe", 2025, tmp_path) == target


def test_read_fca_joins_each_auditor_row_to_its_form_receipt_date(tmp_path):
    import zipfile

    from distress import cvm

    head = "CNPJ_CIA;DT_REFER;VERSAO;DENOM_CIA;CD_CVM;CATEG_DOC;ID_DOC;DT_RECEB;LINK_DOC\n"
    main = head + "11.222.333/0001-44;2015-01-01;1;ACME SA;001234;FCA;555;2015-05-01;x\n"
    aud = ("CNPJ_Companhia;Data_Referencia;Versao;ID_Documento;Nome_Empresarial;Auditor;CPF_CNPJ_Auditor;"
           "Codigo_CVM_Auditor;Origem_Auditor;Data_Inicio_Atuacao_Auditor;Data_Fim_Atuacao_Auditor;"
           "Responsavel_Tecnico;CPF_Responsavel_Tecnico;Data_Inicio_Atuacao_Responsavel_Tecnico;"
           "Data_Fim_Atuacao_Responsavel_Tecnico\n"
           "11.222.333/0001-44;2015-01-01;1;555;ACME SA;KPMG;57.755.217/0001-29;004189;Nacional;2011-01-01;;"
           "Fulano;000;2011-01-01;\n")
    with zipfile.ZipFile(tmp_path / "fca_cia_aberta_2015.zip", "w") as z:
        z.writestr("fca_cia_aberta_2015.csv", main.encode("latin1"))
        z.writestr("fca_cia_aberta_auditor_2015.csv", aud.encode("latin1"))
    docs, auditors = cvm.read_fca([2015], raw_dir=tmp_path)
    assert docs.to_dict("records") == [{"cd_cvm": 1234, "id_doc": 555, "dt_receb": pd.Timestamp("2015-05-01")}]
    a = auditors.iloc[0]
    assert a["id_doc"] == 555 and a["auditor_cnpj"] == "57.755.217/0001-29"
    assert a["start"] == pd.Timestamp("2011-01-01") and pd.isna(a["end"])
