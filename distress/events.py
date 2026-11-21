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


IPE_COLS = ["cd_cvm", "cnpj", "name", "category", "type", "subject", "delivered", "link"]
ESCALATION = {("RE", "RJ"), ("RE", "FAL")}


def candidate_events(cad: pd.DataFrame, ipe: pd.DataFrame) -> pd.DataFrame:
    cols = ["cd_cvm", "name", "source", "date", "evidence", "link", "event_type"]
    if ipe.empty:
        ipe = pd.DataFrame(columns=IPE_COLS).astype({"category": str, "subject": str, "type": str})
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
    out["status"] = out["source"].map({"registry": "filing", "recovery_category": "ongoing"})
    mf = out["source"] == "material_fact"
    for i in out.index[mf]:
        verdict = classifier(out.loc[i])
        own = bool(verdict.get("own_filing"))
        out.at[i, "own_filing"] = own
        out.at[i, "status"] = verdict.get("status") or ("filing" if own else "other")
        if verdict.get("event_type"):
            out.at[i, "event_type"] = verdict["event_type"]
    return out


def build_events(adjudicated: pd.DataFrame) -> pd.DataFrame:
    """Group a firm's own signals into episodes.

    An episode opens only with a filing (a material fact announcing it, or the
    registry status), except for a firm's very first signal, which may be a
    monitoring filing of a recovery that began before the data. Monitoring filings
    never open an episode, however sparse they are. A new filing opens a new
    episode when the previous one ended (explicit exit or more than a year without
    any signal) or when it escalates (extrajudicial to judicial or bankruptcy).
    Liquidation is excluded: for non-financial issuers it is usually a voluntary
    dissolution, not a default.
    """
    cols = ["cd_cvm", "name", "event_date", "event_type", "evidence", "link",
            "episode_end", "n_prior_events", "left_truncated"]
    sig = adjudicated[(adjudicated["own_filing"] | (adjudicated["status"] == "exit"))
                      & (adjudicated["event_type"] != "LIQ")]
    order = {"filing": 0, "ongoing": 1, "exit": 2}
    sig = sig.assign(_o=sig["status"].map(order)).sort_values(["cd_cvm", "date", "_o"], kind="stable")
    rows = []
    for cd, g in sig.groupby("cd_cvm", sort=True):
        cur = None
        for s in g.itertuples(index=False):
            typ = s.event_type if isinstance(s.event_type, str) else None
            if s.status == "exit":
                if cur is not None:
                    cur["exited"] = True
                    cur["episode_end"] = max(cur["episode_end"], s.date)
                continue
            if cur is None:
                cur = _open(cd, s, typ)
                continue
            new = s.status == "filing" and (
                cur["exited"] or s.date - cur["episode_end"] > EPISODE_GAP
                or (cur["event_type"], typ) in ESCALATION)
            if new:
                rows.append(cur)
                cur = _open(cd, s, typ)
            else:
                cur["episode_end"] = max(cur["episode_end"], s.date)
                cur["event_type"] = cur["event_type"] or typ
        if cur is not None:
            rows.append(cur)
    if not rows:
        return pd.DataFrame(columns=cols)
    ev = pd.DataFrame(rows)
    ev["event_type"] = ev["event_type"].fillna("RJ")
    ev["left_truncated"] = [bool(a < WINDOW_START <= b) for a, b in zip(ev["event_date"], ev["episode_end"])]
    ev["n_prior_events"] = ev.groupby("cd_cvm").cumcount()
    return ev[cols].reset_index(drop=True)


def _open(cd, s, typ) -> dict:
    return {"cd_cvm": cd, "name": s.name, "event_date": s.date, "event_type": typ, "evidence": s.evidence,
            "link": s.link, "episode_end": s.date, "exited": False}


def non_financial(ev: pd.DataFrame, cad: pd.DataFrame) -> pd.DataFrame:
    """Drop banks, insurers and other financial issuers. The registry may repeat a company."""
    from .panel import is_financial

    sector = cad.drop_duplicates("cd_cvm").set_index("cd_cvm")["sector"]
    return ev[~ev["cd_cvm"].map(sector).map(is_financial).fillna(False).astype(bool)]


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


def _yes(v) -> bool | None:
    s = _plain(str(v if v is not None else "")).strip().lower() if not (isinstance(v, float) and pd.isna(v)) else ""
    if s in ("s", "sim", "y", "yes"):
        return True
    if s in ("n", "nao", "no"):
        return False
    return None


def apply_review(draft: pd.DataFrame, sheet: pd.DataFrame) -> pd.DataFrame:
    """The author's review of the event list applied to the draft.

    "S" confirms an event; "N" with a corrected date moves it; "N" without a date removes it (not an
    event of the company). Events older than the sheet are kept unchanged. A blank answer stops the
    process: a partly reviewed list does not open the test.
    """
    ans = sheet["confere (S/N)"].map(_yes)
    if ans.isna().any():
        raise ValueError(f"{int(ans.isna().sum())} row(s) of the review sheet have no S/N answer")
    out = draft.copy()
    out["review"], out["review_note"] = "not_in_sheet", ""
    key = list(zip(out["cd_cvm"].astype(int), pd.to_datetime(out["event_date"]).dt.date, out["event_type"]))
    pos = {k: i for i, k in enumerate(key)}
    drop = []
    for (_, r), ok in zip(sheet.iterrows(), ans):
        k = (int(r["código CVM"]), pd.to_datetime(r["data"]).date(), r["tipo"])
        if k not in pos:
            raise ValueError(f"review row not found in the draft: {k}")
        i = out.index[pos[k]]
        note = r.get("observação")
        out.at[i, "review_note"] = "" if note is None or (isinstance(note, float) and pd.isna(note)) else str(note)
        fixed = r.get("data correta")
        has_fix = fixed is not None and not (isinstance(fixed, float) and pd.isna(fixed)) and str(fixed).strip()
        if ok:
            out.at[i, "review"] = "confirmed"
        elif has_fix:
            out.at[i, "event_date"] = pd.to_datetime(fixed, dayfirst=True)
            out.at[i, "review"] = "date_corrected"
        else:
            drop.append(i)
    return out.drop(index=drop).sort_values(["cd_cvm", "event_date"]).reset_index(drop=True)
