"""Credit memos on past cases, each written as of twelve months before the company's filing.

Only firms that went on to file are named; the memo uses what was public on the analysis date and
the hazard logit fitted on the training years, so none of the cases was seen in fitting.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .orchestrator import Memo

Z2_WEIGHTS = {"wc_ta": 6.56, "re_ta": 3.26, "ebit_ta": 6.72, "eq_tl": 1.05}


def analysis_date(event_date) -> pd.Timestamp:
    return pd.Timestamp(event_date) - pd.DateOffset(months=12)


def altman_z2(rows: pd.DataFrame) -> pd.Series:
    """Altman's Z'' for emerging markets (with the 3.25 constant) from the panel ratios."""
    return 3.25 + sum(w * rows[c].astype(float) for c, w in Z2_WEIGHTS.items())


def logit_drivers(model, rows: pd.DataFrame, k: int = 3) -> list[tuple[str, float]]:
    """The features that raise this row's hazard most: coefficient times standardised value, positive only."""
    contrib = model.clf.coef_[0] * model.prep.transform(rows[model.features])[0]
    order = np.argsort(-contrib)
    return [(model.features[i], float(contrib[i])) for i in order[:k] if contrib[i] > 0]


def write_memo(path: Path, memo: Memo, event: str, clean=lambda s: s) -> None:
    """Markdown for one past case: the memo when the critic lets it through, otherwise the values it blocked.
    ``clean`` runs over the whole page before it is written (the CLI masks other listed companies)."""
    f = memo.facts
    head = [f"# {f['empresa']}", "", f"Data da análise: {f['data_da_analise']}. Evento posterior: {event}.", ""]
    if memo.blocked:
        body = ["**Parecer bloqueado pela trava numérica.** Valores sem fonte nas ferramentas: "
                + ", ".join(memo.unsupported) + "."]
    else:
        body = [memo.text.strip()]
    facts = ["", "## Fatos das ferramentas", "",
             "```json", json.dumps({k: v for k, v in f.items() if k != "legenda"}, ensure_ascii=False, indent=1),
             "```"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(clean("\n".join(head + body + facts) + "\n"), encoding="utf-8")
