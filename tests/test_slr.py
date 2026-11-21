import json

import pandas as pd
import pytest

from distress.slr import review


def work(id_, title, doi=None, year=2020, inv=None, oa=True, src="Journal of Finance", cites=10):
    return {"id": f"https://openalex.org/{id_}", "doi": f"https://doi.org/{doi}" if doi else None,
            "title": title, "publication_year": year, "type": "article", "language": "en",
            "abstract_inverted_index": inv or {"Firms": [0], "default": [1], "prediction": [2]},
            "cited_by_count": cites, "open_access": {"is_oa": oa, "oa_url": "http://x.pdf" if oa else None},
            "primary_location": {"source": {"id": "https://openalex.org/S1", "display_name": src}}}


def test_abstract_is_rebuilt_from_the_inverted_index():
    inv = {"Survival": [0], "models": [1, 4], "predict": [2], "default;": [3]}
    assert review.abstract(inv) == "Survival models predict default; models"


def test_records_are_normalised_and_deduplicated_by_doi_and_title():
    raw = [work("W1", "Machine learning and bankruptcy", doi="10.1/abc"),
           work("W2", "Machine Learning and Bankruptcy.", doi=None),
           work("W3", "Another paper", doi="10.1/ABC")]
    recs = review.normalise(raw, strand="a")
    deduped = review.dedupe(recs)
    assert len(recs) == 3 and len(deduped) == 1
    assert deduped.iloc[0]["doi"] == "10.1/abc"
    assert deduped.iloc[0]["venue"] == "Journal of Finance"


def test_semantic_scholar_records_share_the_same_columns():
    raw = [{"paperId": "abc", "externalIds": {"DOI": "10.1/XY"}, "title": "Deep survival for default",
            "abstract": "We predict default.", "year": 2021, "venue": "Expert Syst. Appl.",
            "journal": {"name": "Expert Systems with Applications"}, "citationCount": 42,
            "openAccessPdf": {"url": "http://oa.pdf"}, "publicationTypes": ["JournalArticle"]},
           {"paperId": "def", "externalIds": {}, "title": "No abstract", "abstract": None, "year": 2019,
            "venue": "", "journal": None, "citationCount": 0, "openAccessPdf": None, "publicationTypes": None}]
    df = review.normalise_s2(raw, strand="a")
    oa = review.normalise([work("W1", "t", doi="10.1/a")], strand="a")
    assert list(df.columns) == list(oa.columns)
    assert df.iloc[0][["doi", "venue", "is_oa", "cited_by"]].tolist() == ["10.1/xy", "Expert Systems with Applications", True, 42]
    assert df.iloc[1]["abstract"] == "" and not df.iloc[1]["is_oa"]


def test_every_strand_has_a_semantic_scholar_query():
    assert set(review.S2_STRANDS) == set(review.STRANDS)
    assert all("+" in q and "|" in q for q in review.S2_STRANDS.values())


def test_screening_parser_accepts_only_known_decisions():
    ok = review.parse_screen({"items": [{"i": 1, "decision": "include", "reason": "x"},
                                        {"i": 2, "decision": "exclude", "reason": "consumer"}]}, n=2)
    assert [d["decision"] for d in ok] == ["include", "exclude"]
    with pytest.raises(ValueError):
        review.parse_screen({"items": [{"i": 1, "decision": "maybe"}]}, n=1)


def test_inclusive_rule_keeps_a_record_either_screener_includes():
    a = pd.DataFrame({"decision": ["include", "exclude", "exclude", "include"]})
    b = pd.DataFrame({"decision": ["include", "include", "exclude", "exclude"]})
    assert review.combine(a["decision"], b["decision"]).tolist() == ["include", "include", "exclude", "include"]


def test_extraction_parser_validates_vocabularies():
    good = {"items": [{"i": 1, "families": ["survival", "ml_trees"], "validation": "out_of_time", "country": "Brazil",
                       "unit": "listed", "uses_text": False, "uses_llm": False, "n_events": 120}]}
    out = review.parse_extract(good, n=2)
    assert out[0]["families"] == ["survival", "ml_trees"] and out[0]["n_events"] == 120
    assert out[1]["validation"] == "unclear" and out[1]["families"] == []
    with pytest.raises(ValueError):
        review.parse_extract({"items": [{"i": 1, "families": ["astrology"], "validation": "unclear"}]}, n=1)


def test_synthesis_tables_count_families_and_validation():
    ex = pd.DataFrame({"year": [2005, 2015, 2021, 2023],
                       "families": [["statistical"], ["ml_trees", "statistical"], ["deep_learning"], ["survival", "text_llm"]],
                       "validation": ["in_sample", "holdout_random", "out_of_time", "out_of_time"],
                       "uses_text": [False, False, False, True], "uses_llm": [False, False, False, True],
                       "country": ["USA", "China", "Brazil", "Brazil"]})
    t = review.synthesis(ex)
    assert t["families"].loc["statistical", "2000-2009"] == 1 and t["families"].loc["statistical", "2010-2019"] == 1
    assert t["validation"]["out_of_time"] == 2
    assert t["share_out_of_time"] == 0.5 and t["n_brazil"] == 2 and t["n_llm"] == 1


def test_prisma_counts_add_up():
    screened = pd.DataFrame({"final": ["include", "exclude", "exclude", "include"],
                             "is_oa": [True, True, False, False]})
    c = review.prisma_counts(identified=10, after_dedup=4, screened=screened)
    assert c["identified"] == 10 and c["duplicates_removed"] == 6
    assert c["screened"] == 4 and c["excluded_title_abstract"] == 2
    assert c["included"] == 2 and c["full_text_open_access"] == 1
    assert c["included"] == c["screened"] - c["excluded_title_abstract"]


def crossref_item(doi, title, family, year):
    return {"DOI": doi, "title": [title], "author": [{"family": family, "given": "A"}],
            "issued": {"date-parts": [[year]]}}


def test_reference_resolves_only_on_author_year_and_title_match(monkeypatch):
    hits = [crossref_item("10.1/wrong-author", "Forecasting bankruptcy more accurately: a simple hazard model",
                          "Smith", 2001),
            crossref_item("10.1/right", "Forecasting Bankruptcy More Accurately: A Simple Hazard Model",
                          "Shumway", 2001)]
    monkeypatch.setattr(review, "crossref_search", lambda q, rows=5: hits)
    assert review.resolve_reference("Shumway", 2001, "Forecasting bankruptcy more accurately") == "10.1/right"
    assert review.resolve_reference("Shumway", 1995, "Forecasting bankruptcy more accurately") is None
    assert review.resolve_reference("Shumway", 2001, "Deep learning for credit scoring in China") is None


def test_journal_version_is_preferred_over_a_preprint(monkeypatch):
    pre = crossref_item("10.3386/w1", "Multi-period corporate default prediction with stochastic covariates",
                        "Duffie", 2006) | {"type": "report"}
    pub = crossref_item("10.1016/j.jfineco.1", "Multi-period corporate default prediction with stochastic covariates",
                        "Duffie", 2007) | {"type": "journal-article"}
    monkeypatch.setattr(review, "crossref_search", lambda q, rows=10: [pre, pub])
    assert review.resolve_reference("Duffie", 2007, "Multi-period corporate default prediction") == "10.1016/j.jfineco.1"


def test_subtitle_and_group_author_do_not_block_a_match(monkeypatch):
    xgb = crossref_item("10.1145/x", "XGBoost", "Chen", 2016) | {"subtitle": ["A Scalable Tree Boosting System"]}
    cal = {"DOI": "10.1186/c", "title": ["Calibration: the Achilles heel of predictive analytics"],
           "author": [{"name": "Topic Group of the STRATOS initiative"}, {"family": "Van Calster", "given": "B"}],
           "issued": {"date-parts": [[2019]]}}
    monkeypatch.setattr(review, "crossref_search", lambda q, rows=10: [xgb, cal])
    assert review.resolve_reference("Chen", 2016, "XGBoost: a scalable tree boosting system") == "10.1145/x"
    assert review.resolve_reference("Van Calster", 2019, "Calibration: the Achilles heel") == "10.1186/c"


def test_a_suggested_doi_is_accepted_only_if_its_record_matches(monkeypatch):
    rec = crossref_item("10.1136/bmj.n71", "The PRISMA 2020 statement: an updated guideline for reporting "
                        "systematic reviews", "Page", 2021)
    monkeypatch.setattr(review, "verify_doi", lambda doi: rec if doi == "10.1136/bmj.n71" else None)
    monkeypatch.setattr(review, "crossref_search", lambda q, rows=10: [])
    assert review.resolve_reference("Page", 2021, "The PRISMA 2020 statement", doi="10.1136/BMJ.N71") == \
        "10.1136/bmj.n71"
    assert review.resolve_reference("Page", 2021, "Deep survival forests", doi="10.1136/bmj.n71") is None
    assert review.resolve_reference("Page", 2021, "The PRISMA 2020 statement", doi="10.9/none") is None


def test_author_match_ignores_accents_and_case(monkeypatch):
    hits = [crossref_item("10.1/p", "Random survival forests", "Ishwaran", 2008)]
    monkeypatch.setattr(review, "crossref_search", lambda q, rows=5: hits)
    assert review.resolve_reference("ISHWARAN", 2008, "Random survival forests") == "10.1/p"
    hits[0]["author"][0]["family"] = "Pölsterl"
    hits[0]["title"] = ["scikit-survival"]
    assert review.resolve_reference("Polsterl", 2008, "scikit-survival") == "10.1/p"


def test_bibtex_uses_given_citation_keys(monkeypatch):
    found = {"10.1/real": {"title": ["Real paper"], "author": [{"family": "Silva", "given": "Ana"}],
                           "issued": {"date-parts": [[2019]]}, "container-title": ["J. Test"], "DOI": "10.1/real",
                           "type": "journal-article"}}
    monkeypatch.setattr(review, "verify_doi", lambda doi: found.get(doi))
    assert review.bibtex(["10.1/real"], keys={"10.1/real": "silva2019"}).startswith("@article{silva2019,")


def test_bibtex_contains_only_verified_records(monkeypatch):
    found = {"10.1/real": {"title": ["Real paper"], "author": [{"family": "Silva", "given": "Ana"}],
                           "issued": {"date-parts": [[2019]]}, "container-title": ["J. Test"], "DOI": "10.1/real",
                           "type": "journal-article", "volume": "3", "page": "1-10"}}
    monkeypatch.setattr(review, "verify_doi", lambda doi: found.get(doi))
    bib = review.bibtex(["10.1/real", "10.1/fake"])
    assert "Real paper" in bib and "10.1/real" in bib
    assert "10.1/fake" not in bib
    assert bib.count("@article") == 1


def test_extraction_accepts_families_written_as_text():
    out = review.parse_extract({"items": [{"i": 1, "families": "survival, ml_trees", "validation": "unclear"},
                                          {"i": 2, "families": "survival", "validation": "out_of_time"}]}, n=2)
    assert out[0]["families"] == ["survival", "ml_trees"] and out[1]["families"] == ["survival"]
