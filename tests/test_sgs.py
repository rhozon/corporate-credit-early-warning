import json

import pandas as pd

from distress import sgs


def test_parse_sgs_json_to_monthly_float_series():
    payload = json.dumps([
        {"data": "01/01/2024", "valor": "11.65"},
        {"data": "01/02/2024", "valor": "11,15"},
        {"data": "01/03/2024", "valor": "10.65"},
    ])
    s = sgs.parse(payload)
    assert s.index.tolist() == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01"), pd.Timestamp("2024-03-01")]
    assert s.dtype == float
    assert s.iloc[1] == 11.15


def test_daily_series_is_averaged_to_month_start():
    payload = json.dumps([
        {"data": "02/01/2024", "valor": "1.0"},
        {"data": "03/01/2024", "valor": "3.0"},
        {"data": "01/02/2024", "valor": "5.0"},
    ])
    s = sgs.parse(payload)
    assert s.loc["2024-01-01"] == 2.0
    assert s.loc["2024-02-01"] == 5.0


def test_long_requests_are_split_into_ten_year_windows():
    windows = sgs.windows("2008-01-01", "2026-11-20")
    assert windows[0] == ("01/01/2008", "31/12/2017")
    assert windows[1] == ("01/01/2018", "20/11/2026")
    assert all(int(b[-4:]) - int(a[-4:]) <= 9 for a, b in windows)


def test_fetch_concatenates_windows(monkeypatch):
    calls = []

    def fake_get(code, a, b):
        calls.append((a, b))
        return json.dumps([{"data": a, "valor": "1.0"}])

    monkeypatch.setattr(sgs, "_get_window", fake_get)
    out = json.loads(sgs._fetch(432, "2008-01-01", end="2026-11-20"))
    assert len(calls) == 2 and len(out) == 2


def test_series_reads_cache_first(tmp_path, monkeypatch):
    (tmp_path / "sgs_432.json").write_text(json.dumps([{"data": "01/01/2024", "valor": "11.65"}]))
    monkeypatch.setattr(sgs, "_fetch", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    s = sgs.series(432, "2024-01-01", cache_dir=tmp_path)
    assert s.iloc[0] == 11.65
