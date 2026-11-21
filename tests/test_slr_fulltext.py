import pandas as pd
import pytest

from distress.slr import fulltext as ft


def rec(id_, cites, year=2020, fam=("ml_trees",), country="USA", val="holdout_random", llm=False, oa=True):
    return {"openalex_id": id_, "doi": f"10.1/{id_}", "title": f"T {id_}", "year": year, "cited_by": cites,
            "is_oa": oa, "oa_url": f"http://x/{id_}.pdf" if oa else None, "families": list(fam),
            "country": country, "validation": val, "uses_text": llm, "uses_llm": llm}


def test_ranking_uses_citations_per_year_and_skips_closed_access():
    ex = pd.DataFrame([rec("old", 100, year=2006), rec("new", 30, year=2024), rec("closed", 999, oa=False)])
    r = ft.rank_core(ex, year_now=2026)
    gen = r[r["stratum"] == "general"].sort_values("rank")
    assert gen["openalex_id"].tolist() == ["new", "old"]  # 30/3 = 10 per year beats 100/21
    assert "closed" not in set(r["openalex_id"])


def test_a_record_belongs_to_every_stratum_it_matches():
    ex = pd.DataFrame([rec("s", 5, fam=("survival", "deep_learning"), country="Brazil", val="out_of_time", llm=True)])
    assert set(ft.rank_core(ex)["stratum"]) == {"survival", "deep_learning", "text_llm", "brazil", "out_of_time",
                                                "general"}


def test_core_takes_a_quota_per_stratum_from_records_with_text_then_fills_by_score():
    rows = [rec(f"s{i}", 50 - i, fam=("survival",)) for i in range(4)] + \
           [rec(f"g{i}", 500 - i) for i in range(5)]
    ranked = ft.rank_core(pd.DataFrame(rows))
    have = {"s1", "s2", "s3", "g0", "g1", "g2", "g3", "g4"}  # s0 has no retrievable text
    core = ft.pick_core(ranked, have, per_stratum=2, n=5)
    assert core["openalex_id"].tolist()[:2] == ["s1", "s2"]  # quota met from the next ones in line
    assert len(core) == 5 and core["openalex_id"].is_unique
    assert set(core["openalex_id"]) <= have
    assert core.set_index("openalex_id").loc["g0", "stratum"] == "general"


def test_pdf_link_is_read_from_the_citation_meta_tag():
    html = '<html><head><meta name="citation_pdf_url" content="/doi/pdf/10.1/x"></head></html>'
    assert ft.pdf_link(html, "https://pub.org/doi/10.1/x") == "https://pub.org/doi/pdf/10.1/x"
    assert ft.pdf_link("<html></html>", "https://pub.org") is None


def test_html_text_drops_scripts_and_tags():
    html = "<html><script>var a=1;</script><style>p{}</style><p>Survival &amp; default</p><div>models</div></html>"
    assert ft.html_text(html) == "Survival & default models"


def test_trim_keeps_head_and_tail():
    text = "A" * 100 + "B" * 100 + "C" * 50
    out = ft.trim(text, head=100, tail=50)
    assert out.startswith("A" * 100) and out.endswith("C" * 50) and "B" not in out


class Fake:
    def __init__(self, pages):
        self.pages, self.calls = pages, []

    def __call__(self, url):
        self.calls.append(url)
        return self.pages[url]


def test_landing_page_with_full_body_is_kept_and_cached(tmp_path):
    body = ("<html><body><h2>Introduction</h2>" + "<p>credit default hazard model results</p>" * 600 +
            "<h2>References</h2></body></html>")
    get = Fake({"http://x/a": ("text/html", body.encode())})
    t = ft.fetch_text("http://x/a", tmp_path, "a", get=get)
    assert t and len(t) >= ft.MIN_CHARS
    assert ft.fetch_text("http://x/a", tmp_path, "a", get=get) == t and len(get.calls) == 1


def test_abstract_only_landing_page_counts_as_no_full_text(tmp_path):
    get = Fake({"http://x/b": ("text/html", b"<html><p>Abstract only.</p></html>")})
    assert ft.fetch_text("http://x/b", tmp_path, "b", get=get) is None


def test_long_repository_page_without_paper_structure_is_rejected(tmp_path):
    nav = "<html>" + "<a>Research Outputs</a><a>Profiles</a><a>Datasets</a><a>Contact us</a>" * 400 + "</html>"
    get = Fake({"http://x/nav": ("text/html", nav.encode())})
    assert ft.fetch_text("http://x/nav", tmp_path, "nav", get=get) is None
    assert not ft.looks_like_paper("Abstract only. " * 2000)
    assert ft.looks_like_paper("Introduction " + "x " * 9000 + " Results ... References")


def test_landing_page_link_to_pdf_is_followed(tmp_path, monkeypatch):
    landing = b'<html><head><meta name="citation_pdf_url" content="http://x/c.pdf"></head><p>short</p></html>'
    get = Fake({"http://x/c": ("text/html", landing), "http://x/c.pdf": ("application/pdf", b"%PDF-fake")})
    monkeypatch.setattr(ft, "_pdf_text", lambda b: "full text of the paper " * 1000 + "Introduction Results References")
    t = ft.fetch_text("http://x/c", tmp_path, "c", get=get)
    assert get.calls == ["http://x/c", "http://x/c.pdf"] and t.startswith("full text")


def test_alternative_urls_put_repository_copies_first():
    work = {"locations": [
        {"pdf_url": "https://doi.org/10.1/x", "landing_page_url": "https://doi.org/10.1/x",
         "source": {"type": "journal"}},
        {"pdf_url": None, "landing_page_url": "https://repo.uni.edu/123", "source": {"type": "repository"}},
        {"pdf_url": "https://arxiv.org/pdf/1.pdf", "landing_page_url": "https://arxiv.org/abs/1",
         "source": {"type": "repository"}},
        {"pdf_url": None, "landing_page_url": None, "source": None}]}
    assert ft.alt_urls(work) == ["https://arxiv.org/pdf/1.pdf", "https://doi.org/10.1/x",
                                 "https://repo.uni.edu/123", "https://arxiv.org/abs/1"]


def test_next_url_is_tried_when_one_is_blocked(tmp_path):
    body = ("<p>" + "survival model of default " * 800 + "Introduction Results References</p>").encode()

    calls = []

    def get(url):
        calls.append(url)
        if url == "http://blocked/a":
            raise RuntimeError("403")
        return "text/html", body

    t = ft.fetch_text(["http://blocked/a", "http://repo/a"], tmp_path, "k", get=get)
    assert calls == ["http://blocked/a", "http://repo/a"]
    assert t and t.startswith("survival")


def test_text_with_broken_characters_is_cached_without_error(tmp_path, monkeypatch):
    monkeypatch.setattr(ft, "_pdf_text", lambda b: "Introduction \ud835 bad glyph " + "x " * 9000 + "Results References")
    get = Fake({"http://x/s.pdf": ("application/pdf", b"%PDF-fake")})
    t = ft.fetch_text("http://x/s.pdf", tmp_path, "s", get=get)
    assert t and "\ud835" not in t
    assert ft.fetch_text("http://x/s.pdf", tmp_path, "s", get=get) == t


def test_fulltext_parser_validates_vocabularies_and_types():
    good = {"country": "Brazil", "data": "CVM filings", "unit": "listed", "period": "2010-2020",
            "n_firms": "300", "n_events": 45, "methods": ["survival", "ml_trees"], "best_model": "XGBoost",
            "validation": "out_of_time", "horizon_months": 12, "metric": "AUC", "best_value": 0.87,
            "calibration_reported": True, "uses_text": False, "uses_llm": False,
            "finding": "Boosting beats logit out of time.", "limitations": "Few events."}
    out = ft.parse_fulltext(good)
    assert out["n_firms"] == 300 and out["methods"] == ["survival", "ml_trees"] and out["best_value"] == 0.87
    assert ft.parse_fulltext({"methods": [], "validation": "unclear", "unit": "banana"})["unit"] == "unclear"
    with pytest.raises(ValueError):
        ft.parse_fulltext({"methods": ["astrology"], "validation": "unclear"})
    with pytest.raises(ValueError):
        ft.parse_fulltext({"methods": [], "validation": "sometimes"})


def test_fulltext_parser_accepts_methods_written_as_text():
    assert ft.parse_fulltext({"methods": "survival", "validation": "unclear"})["methods"] == ["survival"]
    assert ft.parse_fulltext({"methods": "statistical, ml_trees; other", "validation": "unclear"})["methods"] == \
        ["statistical", "ml_trees", "other"]
