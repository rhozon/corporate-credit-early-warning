"""Hide company identity from prompts.

Language models may remember how a well-known company's story ended. Masking the
name and the CNPJ keeps the classification of a filing about what the filing
says, not about what the model knows happened next.
"""
from __future__ import annotations

import re

TOKEN = "[EMPRESA]"


def mask(text: str, names: list[str], cnpj: str | None = None) -> str:
    out = text
    if cnpj:
        digits = re.sub(r"\D", "", cnpj)
        if digits:
            pattern = "".join(re.escape(d) + r"[.\-/\s]?" for d in digits[:-1]) + re.escape(digits[-1])
            out = re.sub(pattern, TOKEN, out)
    # Whole words only: a short name such as "Oi" must not cut "apoio", nor "Vale" cut "equivalente".
    for name in sorted({n for n in names if n and n.strip()}, key=len, reverse=True):
        out = re.sub(r"(?<!\w)" + re.escape(name.strip()) + r"(?!\w)", TOKEN, out, flags=re.IGNORECASE)
    return out
