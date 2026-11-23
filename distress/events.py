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
# End date of an episode that has not ended: the registry still shows the status, or a bankruptcy.
OPEN_END = pd.Timestamp("2099-12-31")
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

    The registry status is the current one, so the episode that holds it has not
    ended, however long ago its last filing was, unless an exit was announced after
    the status began. A bankruptcy never ends. Both run to ``OPEN_END``, and only an
    escalation opens a new episode after them.
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
                    cur["exited"], cur["open"] = True, False
                    cur["episode_end"] = max(cur["episode_end"], s.date)
                continue
            if cur is None:
                cur = _open(cd, s, typ)
            else:
                ongoing = cur["open"] or cur["event_type"] == "FAL"
                new = s.status == "filing" and (
                    (cur["event_type"], typ) in ESCALATION
                    or (not ongoing and (cur["exited"] or s.date - cur["episode_end"] > EPISODE_GAP)))
                if new:
                    rows.append(cur)
                    cur = _open(cd, s, typ)
                else:
                    cur["episode_end"] = max(cur["episode_end"], s.date)
                    cur["event_type"] = cur["event_type"] or typ
            cur["open"] = cur["open"] or getattr(s, "source", None) == "registry"
        if cur is not None:
            rows.append(cur)
    if not rows:
        return pd.DataFrame(columns=cols)
    ev = pd.DataFrame(rows)
    ev["event_type"] = ev["event_type"].fillna("RJ")
    ev.loc[ev["open"].astype(bool) | (ev["event_type"] == "FAL"), "episode_end"] = OPEN_END
    return finalize(ev)[cols].reset_index(drop=True)


def finalize(ev: pd.DataFrame) -> pd.DataFrame:
    """Fields that follow from the dates: an episode ends no earlier than it starts, left truncation
    (in recovery when the window opens) and the number of earlier events of the firm."""
    ev = ev.assign(event_date=pd.to_datetime(ev["event_date"]))
    ev = ev.sort_values(["cd_cvm", "event_date"], kind="stable").reset_index(drop=True)
    end = pd.to_datetime(ev["episode_end"]) if "episode_end" in ev else pd.Series(pd.NaT, index=ev.index)
    end = end.fillna(ev["event_date"])
    ev["episode_end"] = end.where(end >= ev["event_date"], ev["event_date"])
    ev["left_truncated"] = [bool(a < WINDOW_START <= b) for a, b in zip(ev["event_date"], ev["episode_end"])]
    ev["n_prior_events"] = ev.groupby("cd_cvm").cumcount()
    return ev


def _open(cd, s, typ) -> dict:
    return {"cd_cvm": cd, "name": s.name, "event_date": s.date, "event_type": typ, "evidence": s.evidence,
            "link": s.link, "episode_end": s.date, "exited": False, "open": False}


def non_financial(ev: pd.DataFrame, cad: pd.DataFrame) -> pd.DataFrame:
    """Drop banks, insurers and other financial issuers. The registry may repeat a company."""
    from .panel import is_financial

    sector = cad.drop_duplicates("cd_cvm").set_index("cd_cvm")["sector"]
    return ev[~ev["cd_cvm"].map(sector).map(is_financial).fillna(False).astype(bool)]


_NAME_SAYS = re.compile(r"recupera|falid|liquida")
REVIEW_TABS = ("conferir com cuidado", "rápidas")


def review_sheet(ev: pd.DataFrame, path: Path) -> None:
    """Spreadsheet for the author to confirm each event and its date.

    Two tabs: events whose company name does not already say recovery, bankruptcy or liquidation
    (where most errors are) come first; the rest, where usually only the date needs a look, second.
    The S/N cell has a drop-down list and the link is clickable.
    """
    from openpyxl.styles import Font
    from openpyxl.worksheet.datavalidation import DataValidation

    sheet = pd.DataFrame({
        "código CVM": ev["cd_cvm"].values,
        "empresa": ev["name"].values,
        "data": pd.to_datetime(ev["event_date"]).dt.date.values,
        "tipo": ev["event_type"].values,
        "evidência": ev["evidence"].values,
        "link": ev["link"].values,
        "confere (S/N)": "",
        "data correta": "",
        "observação": "",
    })
    says = sheet["empresa"].astype(str).map(lambda n: bool(_NAME_SAYS.search(_plain(n))))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    widths = {"A": 11, "B": 48, "C": 11, "D": 6, "E": 48, "F": 12, "G": 13, "H": 13, "I": 40}
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for tab, part in zip(REVIEW_TABS, (sheet[~says], sheet[says])):
            part.to_excel(xw, sheet_name=tab, index=False)
            ws = xw.sheets[tab]
            ws.freeze_panes = "B2"
            for col, w in widths.items():
                ws.column_dimensions[col].width = w
            for cell in ws[1]:
                cell.font = Font(bold=True)
            for row in range(2, ws.max_row + 1):
                link = ws.cell(row=row, column=6)
                if link.value:
                    link.hyperlink, link.value, link.style = str(link.value), "abrir", "Hyperlink"
                ws.cell(row=row, column=3).number_format = "DD/MM/YYYY"
            dv = DataValidation(type="list", formula1='"S,N"', allow_blank=True)
            ws.add_data_validation(dv)
            if ws.max_row > 1:
                dv.add(f"G2:G{ws.max_row}")


def read_review(path: Path) -> pd.DataFrame:
    """Every tab of the review sheet, as one table."""
    tabs = pd.read_excel(path, sheet_name=None)
    return pd.concat(tabs.values(), ignore_index=True)


def _yes(v) -> bool | None:
    s = _plain(str(v if v is not None else "")).strip().lower() if not (isinstance(v, float) and pd.isna(v)) else ""
    if s in ("s", "sim", "y", "yes"):
        return True
    if s in ("n", "nao", "no"):
        return False
    return None


def apply_review(draft: pd.DataFrame, sheet: pd.DataFrame, signals: pd.DataFrame | None = None) -> pd.DataFrame:
    """A review of the event list applied to the draft.

    "S" confirms an event; "N" with a corrected date moves it; "N" without a date removes it (not an
    event of the company). Events older than the sheet are kept unchanged. A blank answer stops the
    process: a partly reviewed list does not open the test.

    Removing an event rejects the document that opened its episode, not the episode: with the
    adjudicated ``signals``, the company's first own filing later in the same episode opens it instead.
    Episode end, left truncation and the count of earlier events are recomputed after every change.
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
    rebuilt = [] if signals is None else [r for r in (_later_filing(out.loc[i], signals) for i in drop) if r is not None]
    kept = pd.concat([out.drop(index=drop), pd.DataFrame(rebuilt)], ignore_index=True) if rebuilt else out.drop(index=drop)
    return finalize(kept)[list(out.columns)]


def _later_filing(ev: pd.Series, signals: pd.DataFrame) -> dict | None:
    """The company's first own filing after a rejected opener, inside the same episode."""
    start = pd.Timestamp(ev["event_date"])
    end = pd.to_datetime(ev["episode_end"])
    end = start if pd.isna(end) else end
    own = signals["own_filing"].astype(str).str.lower().eq("true")
    when = pd.to_datetime(signals["date"])
    s = signals[(signals["cd_cvm"] == ev["cd_cvm"]) & own & (signals["status"] == "filing")
                & (when > start) & (when <= end) & (signals["event_type"] != "LIQ")]
    if s.empty:
        return None
    s = s.assign(_d=pd.to_datetime(s["date"])).sort_values("_d", kind="stable").iloc[0]
    typ = s["event_type"] if isinstance(s["event_type"], str) else ev["event_type"]
    return {**ev.to_dict(), "event_date": s["_d"], "event_type": typ, "evidence": s["evidence"], "link": s["link"],
            "review": "rebuilt_after_drop",
            "review_note": f"documento de {start:%d/%m/%Y} rejeitado; episódio aberto pelo pedido seguinte da empresa"}
