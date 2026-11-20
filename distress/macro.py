"""Macroeconomic covariates from the Banco Central SGS, monthly."""
from __future__ import annotations

from typing import Callable

import pandas as pd

from . import sgs

# SGS codes. Meanings were checked against the SGS catalogue before use.
CODES = {"selic": 432, "ipca": 433, "ibc": 24363, "unemp": 24369, "pj_default": 21083, "spread": 20783}

# Months between the reference month and its publication.
LAGS = {"selic": 0, "ipca12": 1, "ibc_yoy": 2, "unemp": 2, "pj_default": 2, "spread": 2}


def build_macro(series_fn: Callable[[int, str], pd.Series] = sgs.series, start: str = "2008-01-01") -> pd.DataFrame:
    s = {k: series_fn(code, start) for k, code in CODES.items()}
    m = pd.DataFrame({
        "selic": s["selic"],
        "ipca12": ((1 + s["ipca"] / 100).rolling(12).apply(lambda x: x.prod(), raw=True) - 1) * 100,
        "ibc_yoy": (s["ibc"] / s["ibc"].shift(12) - 1) * 100,
        "unemp": s["unemp"],
        "pj_default": s["pj_default"],
        "spread": s["spread"],
    })
    return m.asfreq("MS")
