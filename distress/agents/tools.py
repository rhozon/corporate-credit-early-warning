"""Deterministic tools the agents call. They compute; the language models only write.

Every tool takes the analysis date (``as_of``) and only returns what was public on
that date: the latest panel quarter on or before it, documents delivered before it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

RATIO_FIELDS = {"lev": "alavancagem (passivo/ativo)", "curr": "liquidez corrente", "int_cov": "cobertura de juros",
                "cash_ta": "caixa/ativo", "margin": "margem líquida", "z2": "escore Z'' de Altman",
                "loss_q4": "trimestres com prejuízo em 12 meses (de 4)", "gc_flag": "ressalva de continuidade"}
MACRO_FIELDS = {"selic": "Selic (% a.a.)", "ipca12": "IPCA 12 meses (%)", "ibc_yoy": "IBC-Br, variação anual (%)",
                "pj_default": "inadimplência PJ (%)", "spread": "spread médio (p.p.)"}


def _fmt_date(t) -> str:
    return pd.Timestamp(t).strftime("%d/%m/%Y")


@dataclass
class Tools:
    panel: pd.DataFrame
    docs: pd.DataFrame
    pd_fn: Callable[[pd.DataFrame], pd.DataFrame]
    drivers_fn: Callable[[pd.DataFrame], list]
    name: str

    def _row(self, cd, as_of) -> pd.DataFrame:
        p = self.panel[(self.panel["cd_cvm"] == cd) & (self.panel["q"] <= as_of)]
        if p.empty:
            raise LookupError(f"no panel quarter for {cd} on or before {as_of:%Y-%m-%d}")
        return p.sort_values("q").tail(1)

    def panel_row(self, cd, as_of) -> dict:
        r = self._row(cd, as_of).iloc[0]
        out = {"data_base": _fmt_date(r["q"])}
        out.update({k: round(float(r[k]), 2) for k in RATIO_FIELDS if k in r and pd.notna(r[k])})
        return out

    def pd_curve(self, cd, as_of) -> dict:
        p = self.pd_fn(self._row(cd, as_of)).iloc[0]
        return {k: round(float(p[k]), 3) for k in ("pd12", "pd24", "pd36")}

    def top_drivers(self, cd, as_of) -> list:
        return [(f, round(float(v), 2)) for f, v in self.drivers_fn(self._row(cd, as_of))]

    def macro_context(self, cd, as_of) -> dict:
        r = self._row(cd, as_of).iloc[0]
        return {k: round(float(r[k]), 2) for k in MACRO_FIELDS if k in r and pd.notna(r[k])}

    def recent_disclosures(self, cd, as_of, n: int = 10) -> list[dict]:
        d = self.docs[(self.docs["cd_cvm"] == cd) & (self.docs["delivered"] < as_of)]
        if "own_recovery" in d:
            d = d[d["own_recovery"] == 0]
        d = d.sort_values("delivered", ascending=False).head(n)
        return [{"data": _fmt_date(t), "subject": s} for t, s in zip(d["delivered"], d["subject"])]
