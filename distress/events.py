"""Distress events: judicial or extrajudicial recovery, bankruptcy, liquidation.

CVM has no single table of filings with dates, so the event base is assembled
from three public traces and then reviewed by hand:

* the registry's current issuer status and the date it started;
* the filing category CVM keeps for companies in recovery (monthly reports,
  creditor lists), present for as long as the recovery lasts;
* material facts whose subject mentions recovery or bankruptcy. Many of these
  are about a client, a subsidiary or a rumour, so each one is classified as
  "this company filed" or not (``adjudicate``).

Signals of the same firm less than ``EPISODE_GAP`` apart belong to one episode.
The event date is the earliest signal of the episode.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Callable

import pandas as pd

WINDOW_START = pd.Timestamp("2011-01-01")
EPISODE_GAP = pd.Timedelta(days=365)
_SUBJECT = re.compile(r"recuperacao judicial|recuperacao extrajudicial|falencia")


def _plain(s) -> str:
    s = unicodedata.normalize("NFKD", str(s or ""))
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def status_type(status) -> str | None:
    s = _plain(status)
    if "recuperacao extrajudicial" in s:
        return "RE"
    if "recuperacao judicial" in s:
        return "RJ"
    if "falida" in s or "falencia" in s:
        return "FAL"
    if "liquidacao" in s:
        return "LIQ"
    return None


def candidate_events(cad: pd.DataFrame, ipe: pd.DataFrame) -> pd.DataFrame:
    cols = ["cd_cvm", "name", "source", "date", "evidence", "link", "event_type"]
    reg = cad.assign(event_type=cad["issuer_status"].map(status_type))
    reg = reg[reg["event_type"].notna() & reg["issuer_status_date"].notna()]
    reg = pd.DataFrame({"cd_cvm": reg["cd_cvm"], "name": reg["name"], "source": "registry",
                        "date": reg["issuer_status_date"], "evidence": reg["issuer_status"],
                        "link": None, "event_type": reg["event_type"]})

    cat = ipe[ipe["category"].map(_plain).str.contains("recuperacao")]
    cat = pd.DataFrame({"cd_cvm": cat["cd_cvm"], "name": cat["name"], "source": "recovery_category",
                        "date": cat["delivered"], "evidence": (cat["type"] + " " + cat["subject"]).str.strip(),
                        "link": cat["link"], "event_type": None})

    fr = ipe[ipe["category"].str.startswith("Fato Relevante")
             & ipe["subject"].map(_plain).str.contains(_SUBJECT)]
    fr = pd.DataFrame({"cd_cvm": fr["cd_cvm"], "name": fr["name"], "source": "material_fact",
                       "date": fr["delivered"], "evidence": fr["subject"], "link": fr["link"],
                       "event_type": None})
    out = pd.concat([reg, cat, fr], ignore_index=True)[cols]
    return out.sort_values(["cd_cvm", "date"], kind="stable").reset_index(drop=True)


def adjudicate(candidates: pd.DataFrame, classifier: Callable[[pd.Series], dict]) -> pd.DataFrame:
    """Mark which signals say that this company itself filed.

    Registry statuses and recovery filings are the company's own by construction.
    Material facts go through ``classifier``, which returns
    ``{"own_filing": bool, "event_type": "RJ" | "RE" | "FAL" | "LIQ" | None}``.
    """
    out = candidates.copy()
    out["own_filing"] = True
    mf = out["source"] == "material_fact"
    for i in out.index[mf]:
        verdict = classifier(out.loc[i])
        out.at[i, "own_filing"] = bool(verdict.get("own_filing"))
        if verdict.get("event_type"):
            out.at[i, "event_type"] = verdict["event_type"]
    return out


def build_events(adjudicated: pd.DataFrame) -> pd.DataFrame:
    cols = ["cd_cvm", "name", "event_date", "event_type", "evidence", "link",
            "episode_end", "n_prior_events", "left_truncated"]
    own = adjudicated[adjudicated["own_filing"]].sort_values(["cd_cvm", "date"], kind="stable")
    if own.empty:
        return pd.DataFrame(columns=cols)
    gap = own.groupby("cd_cvm")["date"].diff() > EPISODE_GAP
    own = own.assign(episode=gap.groupby(own["cd_cvm"]).cumsum())
    rows = []
    for (cd, _), g in own.groupby(["cd_cvm", "episode"], sort=True):
        first = g.iloc[0]
        typed = g["event_type"].dropna()
        start, end = g["date"].min(), g["date"].max()
        rows.append({"cd_cvm": cd, "name": first["name"], "event_date": start,
                     "event_type": typed.iloc[0] if len(typed) else "RJ",
                     "evidence": first["evidence"], "link": first["link"], "episode_end": end,
                     "left_truncated": bool(start < WINDOW_START <= end)})
    ev = pd.DataFrame(rows)
    ev["n_prior_events"] = ev.groupby("cd_cvm").cumcount()
    return ev[cols].reset_index(drop=True)


def review_sheet(ev: pd.DataFrame, path: Path) -> None:
    """Spreadsheet for the author to confirm each event and its date."""
    sheet = pd.DataFrame({
        "código CVM": ev["cd_cvm"],
        "empresa": ev["name"],
        "data": pd.to_datetime(ev["event_date"]).dt.date,
        "tipo": ev["event_type"],
        "evidência": ev["evidence"],
        "link": ev["link"],
        "confere (S/N)": "",
        "data correta": "",
        "observação": "",
    })
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sheet.to_excel(path, index=False)
