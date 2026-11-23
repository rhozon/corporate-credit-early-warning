"""No company without a recovery or bankruptcy event is named next to a probability, score or alert in
anything this repository publishes. Past cases (firms with an event) may be named."""
from pathlib import Path

import pandas as pd

from distress import public

ROOT = Path(__file__).resolve().parents[1]


def test_a_solvent_name_next_to_a_probability_is_caught():
    names = ["VALE S.A.", "Vale"]
    assert public.leaks("No trimestre, a Vale S.A. tinha PD de 12 meses de 3,2%.", names)
    assert public.leaks("Ranking: 1. Vale", names)


def test_a_name_far_from_any_risk_word_is_fine():
    text = "A Vale publicou um fato relevante sobre dividendos." + " ..." * 80 + " A PD média foi 1%."
    assert public.leaks(text, ["Vale"]) == []


def test_firms_with_an_event_are_not_in_the_solvent_list():
    cad = pd.DataFrame({"cd_cvm": [1, 2, 3], "name": ["AMERICANAS S.A.", "VALE S.A.", "CIA ABC S.A."]})
    names = public.solvent_names(cad, event_firms={1})
    assert "Vale" in names or "VALE" in names
    assert not any("AMERICANAS" in n.upper() for n in names)
    assert "CIA" not in names  # generic words are not names


def test_one_word_names_count_only_as_proper_nouns_and_dictionary_words_are_listed():
    cad = pd.DataFrame({"cd_cvm": [1, 2], "name": ["CAMBUCI SA", "VALE S.A."], "trade_name": ["PENALTY", "VALE"]})
    names = public.solvent_names(cad, event_firms=set())
    assert public.leaks("an L2 penalty chosen by the probability of default", names) == []
    assert public.leaks("o vale do rio teve 3% de queda", names) == []
    assert public.leaks("Vale: PD 2%", names)
    assert "PENALTY" in public.COMMON_WORDS


def test_trade_names_are_guarded_too():
    cad = pd.DataFrame({"cd_cvm": [9], "name": ["PETRÓLEO BRASILEIRO S.A. PETROBRAS"], "trade_name": ["PETROBRAS"]})
    assert public.leaks("Petrobras: PD 2%", public.solvent_names(cad, event_firms=set()))


def test_tables_with_a_company_name_and_a_risk_column_are_caught():
    assert public.table_leaks(pd.DataFrame({"empresa": ["X"], "pd12": [0.1]}))
    assert public.table_leaks(pd.DataFrame({"name": ["X"], "risk_score": [3]}))
    assert not public.table_leaks(pd.DataFrame({"name": ["X"], "event_date": ["2020-01-01"]}))


def test_registry_codes_count_as_names_because_the_registry_is_public():
    assert public.table_leaks(pd.DataFrame({"cd_cvm": [1], "pd12": [0.1]}))
    assert public.table_leaks(pd.DataFrame({"cd_cvm": [1], "z2": [1.2]}))
    assert public.table_leaks(pd.DataFrame({"cnpj": ["1"], "sig_any_12m": [3]}))
    assert public.table_leaks(pd.DataFrame({"cd_cvm": [1], "going_concern": [1], "covenant": [0]}))
    assert not public.table_leaks(pd.DataFrame({"cd_cvm": [1], "lev": [0.5], "curr": [1.2]}))


def test_a_past_case_name_does_not_trip_the_guard_on_words_it_shares_with_solvent_firms():
    names = ["BAHIA", "CASAS", "Bahia Sul"]
    text = "GRUPO CASAS BAHIA S.A.: PD de 12 meses de 4,3%."
    assert public.leaks(text, names)  # the shared words alone look like solvent firms
    assert public.leaks(text, names, keep=["GRUPO CASAS BAHIA S.A.", "GRUPO CASAS BAHIA"]) == []


def test_mask_replaces_solvent_names_and_keeps_the_past_case():
    text = "Parceria entre Via e Bradesco renovada. GRUPO CASAS BAHIA S.A. tem PD de 12 meses de 4,3%."
    out = public.mask(text, ["Bradesco", "BAHIA"], keep=["GRUPO CASAS BAHIA S.A."])
    assert out == "Parceria entre Via e [outra companhia] renovada. GRUPO CASAS BAHIA S.A. tem PD de 12 meses de 4,3%."


def test_past_case_names_cover_only_firms_with_an_event():
    cad = pd.DataFrame({"cd_cvm": [1, 2], "name": ["GRUPO CASAS BAHIA S.A.", "VALE S.A."], "trade_name": [None, "VALE"]})
    keep = public.past_case_names(cad, event_firms={1})
    assert "GRUPO CASAS BAHIA" in keep and not any("VALE" in k.upper() for k in keep)


def _public_files():
    pats = ["README.md", "PREREGISTRATION.md", "paper/*.qmd", "paper/*.md", "results/**/*.md", "results/**/*.json", "results/**/*.csv"]
    return [p for pat in pats for p in ROOT.glob(pat) if p.is_file()]


def test_nothing_published_names_a_solvent_firm_next_to_a_risk_figure():
    cad, firms = pd.read_csv(ROOT / "data" / "company_names.csv"), public.event_firms(ROOT / "data")
    names, keep = public.solvent_names(cad, firms), public.past_case_names(cad, firms)
    found = {str(p.relative_to(ROOT)): hits for p in _public_files()
             if (hits := public.leaks(p.read_text(encoding="utf-8", errors="replace"), names, keep=keep))}
    assert found == {}


def test_no_published_table_pairs_names_with_risk_columns():
    bad = []
    for p in list(ROOT.glob("results/**/*.csv")) + list(ROOT.glob("results/**/*.parquet")) + \
            list(ROOT.glob("data/*.parquet")) + list(ROOT.glob("data/*.csv")):
        df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, nrows=5)
        if public.table_leaks(df):
            bad.append(str(p.relative_to(ROOT)))
    assert bad == []


def _guard_inputs():
    cad, firms = pd.read_csv(ROOT / "data" / "company_names.csv"), public.event_firms(ROOT / "data")
    return public.solvent_names(cad, firms), public.past_case_names(cad, firms)


def test_the_shipped_cache_names_no_solvent_firm_next_to_a_risk_figure():
    # The packed cache holds the memo drafts as the model wrote them; they ship too.
    names, keep = _guard_inputs()
    pack = ROOT / "cache" / "llm_answers.jsonl.gz"
    assert public.scan_file(pack, names, keep) == []


def test_the_rendered_paper_names_no_solvent_firm_next_to_a_risk_figure():
    import pytest

    pdf = ROOT / "paper" / "paper.pdf"
    if not pdf.exists():
        pytest.skip("paper not rendered")
    names, keep = _guard_inputs()
    assert public.scan_file(pdf, names, keep) == []


def test_scan_file_reads_html_pdf_and_packed_cache(tmp_path):
    import gzip
    import json

    html = tmp_path / "p.html"
    html.write_text("<p>A <b>Bradesco</b> tem PD de 12 meses de 4,3%.</p>", encoding="utf-8")
    pack = tmp_path / "a.jsonl.gz"
    with gzip.open(pack, "wt", encoding="utf-8") as f:
        f.write(json.dumps({"key": "k", "content": "Bradesco: PD de 2%"}) + "\n")
        f.write(json.dumps({"key": "j", "content": "sem nome"}) + "\n")
    assert [n for n, _ in public.scan_file(html, ["Bradesco"])] == ["Bradesco"]
    assert [n for n, _ in public.scan_file(pack, ["Bradesco"])] == ["Bradesco"]
