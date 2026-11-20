"""Firm-quarter panel for survival and 12-month default models.

Each row is a firm at a calendar quarter end ``q``. Its features are the latest
point-in-time row known on ``q``; its macro covariates are the values published
by ``q``. The row's interval is (q, q + 3 months]: ``event`` says whether the
firm filed for recovery or bankruptcy in it, ``y12`` whether it did within the
next twelve months. Quarters inside a recovery episode are not at risk and are
dropped; after the episode ends the firm is at risk again, with ``prior_event``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .macro import LAGS

FINANCIAL_SECTORS = ("banco", "intermedia", "segurad", "securitiz", "arrendamento", "previd", "credito imob")
MAX_INFO_AGE_DAYS = 550
TRAIN_END = pd.Timestamp("2018-12-31")
TEST_START = pd.Timestamp("2020-01-01")


def _months(a: pd.Timestamp, b: pd.Timestamp) -> float:
    return (b.year - a.year) * 12 + (b.month - a.month) + (b.day - a.day) / 30.0


def is_financial(sector) -> bool:
    import unicodedata

    s = unicodedata.normalize("NFKD", str(sector or ""))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return any(k in s for k in FINANCIAL_SECTORS)


def build_panel(feats: pd.DataFrame, events: pd.DataFrame, cad: pd.DataFrame, macro: pd.DataFrame,
                start: str = "2011-03-31", end: str | None = None, exclude=(), lags: dict | None = None) -> pd.DataFrame:
    lags = LAGS if lags is None else lags
    end = end or feats["as_of"].max()
    grid = pd.date_range(start, end, freq="QE")
    ev = events.copy()
    truncated = set(ev.loc[ev["left_truncated"].astype(bool), "cd_cvm"])
    financial = set(cad.loc[cad["sector"].map(is_financial), "cd_cvm"])
    drop = truncated | financial | set(exclude)
    cancel = cad.set_index("cd_cvm")["cancel_date"].to_dict()

    rows = []
    for cd, f in feats[~feats["cd_cvm"].isin(drop)].groupby("cd_cvm"):
        f = f.sort_values("as_of")
        fe = ev[ev["cd_cvm"] == cd].sort_values("event_date")
        dates, ends = fe["event_date"].tolist(), fe["episode_end"].tolist()
        c = cancel.get(cd, pd.NaT)
        qs = [q for q in grid if q >= f["as_of"].iloc[0] and (pd.isna(c) or q < c)]
        qs = [q for q in qs if not any(d < q <= e for d, e in zip(dates, ends))]
        if not qs:
            continue
        picked = pd.merge_asof(pd.DataFrame({"q": qs}), f.rename(columns={"as_of": "feature_as_of"}),
                               left_on="q", right_on="feature_as_of", direction="backward")
        first_q = qs[0]
        for i, r in picked.iterrows():
            q = r["q"]
            nxt = [d for d in dates if d > q]
            t_next = nxt[0] if nxt else pd.NaT
            row = r.to_dict()
            row.update({
                "cd_cvm": cd,
                "info_age_days": (q - r["feature_as_of"]).days,
                "start": (q.year - first_q.year) * 12 + (q.month - first_q.month),
                "event": int(pd.notna(t_next) and t_next <= q + pd.offsets.QuarterEnd(1)),
                "y12": int(pd.notna(t_next) and t_next <= q + pd.DateOffset(months=12)),
                "t_to_event": _months(q, t_next) if pd.notna(t_next) else np.nan,
                "prior_event": int(any(e < q for e in ends)),
                "censored_reason": "",
            })
            row["stop"] = row["start"] + 3
            rows.append(row)
        if rows and rows[-1]["cd_cvm"] == cd and not rows[-1]["event"]:
            rows[-1]["censored_reason"] = "cancelled" if pd.notna(c) else "end"
    p = pd.DataFrame(rows)
    if p.empty:
        return p
    p = p[p["info_age_days"] <= MAX_INFO_AGE_DAYS]
    month = p["q"].dt.to_period("M").dt.to_timestamp()
    for col, lag in lags.items():
        p[col] = macro[col].reindex(month - pd.DateOffset(months=lag)).values
    return p.reset_index(drop=True)


def split(p: pd.DataFrame, data_end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on quarters up to 2018, test from 2020; 2019 is an embargo year.

    A 2018 row's twelve-month window ends in 2019, so no training label overlaps the
    test period. ``y12_complete`` marks test rows whose window is fully observed.
    """
    train = p[p["q"] <= TRAIN_END].copy()
    test = p[p["q"] >= TEST_START].copy()
    test["y12_complete"] = test["q"] + pd.DateOffset(months=12) <= data_end
    return train, test
