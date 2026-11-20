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


def test_prisma_counts_add_up():
    screened = pd.DataFrame({"final": ["include", "exclude", "exclude", "include"],
                             "is_oa": [True, True, False, False]})
    c = review.prisma_counts(identified=10, after_dedup=4, screened=screened)
    assert c["identified"] == 10 and c["duplicates_removed"] == 6
    assert c["screened"] == 4 and c["excluded_title_abstract"] == 2
    assert c["included"] == 2 and c["full_text_open_access"] == 1
    assert c["included"] == c["screened"] - c["excluded_title_abstract"]


def test_bibtex_contains_only_verified_records(monkeypatch):
    found = {"10.1/real": {"title": ["Real paper"], "author": [{"family": "Silva", "given": "Ana"}],
                           "issued": {"date-parts": [[2019]]}, "container-title": ["J. Test"], "DOI": "10.1/real",
                           "type": "journal-article", "volume": "3", "page": "1-10"}}
    monkeypatch.setattr(review, "verify_doi", lambda doi: found.get(doi))
    bib = review.bibtex(["10.1/real", "10.1/fake"])
    assert "Real paper" in bib and "10.1/real" in bib
    assert "10.1/fake" not in bib
    assert bib.count("@article") == 1
