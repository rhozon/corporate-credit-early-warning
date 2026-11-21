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


def _public_files():
    pats = ["README.md", "paper/*.qmd", "paper/*.md", "results/**/*.md", "results/**/*.json", "results/**/*.csv"]
    return [p for pat in pats for p in ROOT.glob(pat) if p.is_file()]


def test_nothing_published_names_a_solvent_firm_next_to_a_risk_figure():
    names = public.solvent_names(pd.read_csv(ROOT / "data" / "company_names.csv"), public.event_firms(ROOT / "data"))
    found = {str(p.relative_to(ROOT)): hits for p in _public_files()
             if (hits := public.leaks(p.read_text(encoding="utf-8", errors="replace"), names))}
    assert found == {}


def test_no_published_table_pairs_names_with_risk_columns():
    bad = []
    for p in list(ROOT.glob("results/**/*.csv")) + list(ROOT.glob("results/**/*.parquet")) + \
            list(ROOT.glob("data/*.parquet")) + list(ROOT.glob("data/*.csv")):
        df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, nrows=5)
        if public.table_leaks(df):
            bad.append(str(p.relative_to(ROOT)))
    assert bad == []
