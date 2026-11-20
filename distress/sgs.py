"""Time series from the Banco Central do Brasil SGS API, as monthly floats."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados?formato=json&dataInicial={start}"
CACHE = Path(__file__).resolve().parents[1] / "data" / "macro"


def _fetch(code: int, start: str) -> str:
    import requests

    d = pd.Timestamp(start).strftime("%d/%m/%Y")
    r = requests.get(URL.format(code=code, start=d), timeout=120)
    r.raise_for_status()
    return r.text


def parse(payload: str) -> pd.Series:
    """SGS JSON to a month-start series. Daily series are averaged within the month."""
    rows = json.loads(payload)
    dates = pd.to_datetime([r["data"] for r in rows], format="%d/%m/%Y")
    values = [float(str(r["valor"]).replace(",", ".")) for r in rows]
    s = pd.Series(values, index=dates, dtype=float)
    return s.groupby(s.index.to_period("M").to_timestamp()).mean()


def series(code: int, start: str = "2008-01-01", cache_dir: Path = CACHE) -> pd.Series:
    path = Path(cache_dir) / f"sgs_{code}.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_fetch(code, start), encoding="utf-8")
    s = parse(path.read_text(encoding="utf-8"))
    return s.loc[pd.Timestamp(start):].rename(code)
