"""Time series from the Banco Central do Brasil SGS API, as monthly floats."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados?formato=json&dataInicial={start}&dataFinal={end}"
CACHE = Path(__file__).resolve().parents[1] / "data" / "macro"


def windows(start: str, end: str) -> list[tuple[str, str]]:
    """Calendar windows of at most ten years; the API refuses longer requests for daily series."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    out = []
    while s <= e:
        stop = min(pd.Timestamp(s.year + 9, 12, 31), e)
        out.append((s.strftime("%d/%m/%Y"), stop.strftime("%d/%m/%Y")))
        s = stop + pd.Timedelta(days=1)
    return out


def _get_window(code: int, a: str, b: str, tries: int = 5) -> str:
    import time

    import requests

    for i in range(tries):
        r = requests.get(URL.format(code=code, start=a, end=b), timeout=120)
        if r.status_code >= 500 and i < tries - 1:
            time.sleep(5 * (i + 1))
            continue
        r.raise_for_status()
        return r.text
    raise RuntimeError("unreachable")


def _fetch(code: int, start: str, end: str | None = None) -> str:
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    rows = []
    for a, b in windows(start, end):
        rows += json.loads(_get_window(code, a, b))
    return json.dumps(rows)


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
