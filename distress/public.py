"""Guard for what this repository publishes.

The models score every listed company, but only firms that already had a recovery or bankruptcy
event (past cases) may be named next to a probability, score, ranking or alert. A solvent company
named beside such a figure could be read as a credit opinion about it. ``leaks`` finds that pattern
in text; ``table_leaks`` finds tables that pair company names with risk columns.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .llm.signals import name_variants

RISK = re.compile(r"\bPDs?\b|probabilit|probabilidade|\bscor|pontua|\balert|\branking|\branked\b|\d\s?%", re.I)
NAME_COLS = {"name", "nome", "empresa", "company", "denom_social", "firm"}
RISK_COLS = re.compile(r"^(pd|prob|score|risk|alert|rank)|(^|_)pd(_|\d|$)", re.I)


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
    return sorted(out)


def leaks(text: str, names: list[str], window: int = 200) -> list[tuple[str, str]]:
    """(name, snippet) for each solvent name with a risk word or a percentage within ``window`` characters."""
    low, found = text.lower(), {}
    for n in names:
        key = n.lower()
        if key not in low:
            continue
        for m in re.finditer(r"(?<!\w)" + re.escape(key) + r"(?!\w)", low):
            near = text[max(0, m.start() - window):m.end() + window]
            if RISK.search(near):
                found.setdefault(n, near.replace("\n", " ")[:2 * window])
                break
    return sorted(found.items())


def table_leaks(df: pd.DataFrame) -> bool:
    cols = [str(c) for c in df.columns]
    return any(c.lower() in NAME_COLS for c in cols) and any(RISK_COLS.search(c) for c in cols)
