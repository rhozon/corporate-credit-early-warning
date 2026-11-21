"""Guard for what this repository publishes.

The models score every listed company, but only firms that already had a recovery or bankruptcy
event (past cases) may be named next to a probability, score, ranking or alert. A solvent company
named beside such a figure could be read as a credit opinion about it. ``leaks`` finds that pattern
in text; ``table_leaks`` finds tables that pair a company identifier with a risk or distress column.
The CVM code and the CNPJ count as identifiers, because the public registry maps them to names.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .llm.signals import CATEGORIES, name_variants

RISK = re.compile(r"\bPDs?\b|probabilit|probabilidade|\bscor|pontua|\balert|\branking|\branked\b|\d\s?%", re.I)
IDS = {"name", "nome", "empresa", "company", "denom_social", "firm", "cd_cvm", "cnpj", "trade_name"}
RISK_COLS = re.compile(r"^(pd|prob|score|risk|alert|rank|sig_)|(^|_)pd(_|\d|$)|^z2$|altman", re.I)
DISTRESS_LABELS = set(CATEGORIES)

# Registered trade names that are also ordinary words. A one-word name in this list is not taken as a
# company mention; every entry is a dictionary word, added when it produced a false alarm.
COMMON_WORDS = {"BANK", "BASE", "BRAZILIAN", "CREDIT", "FRAME", "IMPORT", "PENALTY"}


def event_firms(data_dir: Path) -> set[int]:
    path = Path(data_dir) / "events_reviewed.csv"
    if not path.exists():
        path = Path(data_dir) / "events_draft.csv"
    return set(pd.read_csv(path)["cd_cvm"].astype(int))


def solvent_names(cad: pd.DataFrame, event_firms: set[int]) -> list[str]:
    """Name variants of every registered company without an event (at least four letters)."""
    out = set()
    trade = cad["trade_name"] if "trade_name" in cad else pd.Series([None] * len(cad), index=cad.index)
    for cd, name, tn in zip(cad["cd_cvm"], cad["name"], trade):
        if int(cd) in event_firms:
            continue
        for n in [name, tn]:
            if isinstance(n, str) and n.strip():
                out.update(v.strip() for v in name_variants(n) if len(v.strip()) >= 4)
    return sorted(v for v in out if v.upper() not in COMMON_WORDS)


def _pattern(name: str) -> re.Pattern:
    """Several words: any case. One word: only as a proper noun (Vale or VALE, not vale)."""
    if " " in name.strip():
        return re.compile(r"(?<!\w)" + re.escape(name) + r"(?!\w)", re.I)
    w = name.strip()
    forms = {w.upper(), w[:1].upper() + w[1:].lower()}
    return re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, forms)) + r")(?!\w)")


def leaks(text: str, names: list[str], window: int = 200) -> list[tuple[str, str]]:
    """(name, snippet) for each solvent name with a risk word or a percentage within ``window`` characters."""
    low, found = text.lower(), {}
    for n in names:
        if n.lower() not in low:
            continue
        for m in _pattern(n).finditer(text):
            near = text[max(0, m.start() - window):m.end() + window]
            if RISK.search(near):
                found.setdefault(n, near.replace("\n", " ")[:2 * window])
                break
    return sorted(found.items())


def table_leaks(df: pd.DataFrame) -> bool:
    cols = [str(c) for c in df.columns]
    has_id = any(c.lower() in IDS for c in cols)
    has_risk = any(RISK_COLS.search(c) for c in cols) or any(c in DISTRESS_LABELS for c in cols)
    return has_id and has_risk
