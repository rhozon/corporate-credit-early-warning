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
