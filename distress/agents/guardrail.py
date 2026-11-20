"""Numeric guardrail: every number and date in a memo must come from the tools.

Numbers are read in Brazilian (1.234,5) or international (1,234.5 / 0.78) notation.
A probability or ratio of 0.183 may appear as 0,183, 18,3% or 18.3%; a rounded form
is accepted when it matches the source at the number of decimals written.
"""
from __future__ import annotations

import re

DATE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
NUM = re.compile(r"(?<![\w/])(\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?)(\s?%)?(?![\w/])")
DEFAULT_ALLOWED = {1.0, 3.0, 4.0, 12.0, 24.0, 36.0, 80.0, 95.0}


def _value(tok: str) -> tuple[float, int]:
    if "," in tok and "." in tok:                      # 1.234,56
        whole, dec = tok.replace(".", "").split(",")
        return float(f"{whole}.{dec}"), len(dec)
    if "," in tok:                                     # 18,3
        whole, dec = tok.split(",")
        return float(f"{whole}.{dec}"), len(dec)
    if re.fullmatch(r"[1-9]\d{0,2}(\.\d{3})+", tok):  # 1.234 as thousands
        return float(tok.replace(".", "")), 0
    if "." in tok:
        return float(tok), len(tok.split(".")[1])
    return float(tok), 0


def allowed_numbers(facts: dict) -> set[float]:
    out = set(DEFAULT_ALLOWED)
    for v in facts.values():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        out.add(float(v))
        if abs(v) <= 1.5:
            out.add(float(v) * 100)
    return out


def check(text: str, allowed: set[float], allowed_dates: set[str]) -> list[str]:
    bad = []
    for m in DATE.finditer(text):
        d = f"{int(m.group(1)):02d}/{int(m.group(2)):02d}/{m.group(3)}"
        if d not in allowed_dates:
            bad.append(m.group(0))
    years = {d[-4:] for d in allowed_dates}
    scrubbed = DATE.sub(" ", text)
    for m in NUM.finditer(scrubbed):
        tok = m.group(1)
        v, dec = _value(tok)
        if dec == 0 and 1990 <= v <= 2035 and not m.group(2):
            if tok not in years:
                bad.append(tok)
            continue
        if not any(abs(round(a, dec) - v) < 1e-9 for a in allowed):
            bad.append(tok)
    return bad
