"""Point-in-time accounting and disclosure features.

Every feature is computed "as of" a date on which new information reached CVM
(a filing or a restatement). On that date the model may use, for each period,
only the latest version filed up to that date. Flow accounts arrive year-to-date;
trailing twelve months are rebuilt as YTD(t) + annual(t-1) - YTD(t - 1 year).
"""
from __future__ import annotations

import re
import unicodedata
from bisect import bisect_right
from collections import defaultdict

import numpy as np
import pandas as pd

# Standard CVM codes for commercial, industrial and other companies.
STOCK = {"ta": ["1"], "ca": ["1.01"], "cash": ["1.01.01"], "cl": ["2.01"], "ncl": ["2.02"],
         "eq": ["2.03"], "re": ["2.03.04", "2.03.05"]}
FLOW = {"rev": "3.01", "ebit": "3.05", "finexp": "3.06.02", "ni": "3.11"}


def _scope(st: pd.DataFrame) -> pd.DataFrame:
    """Consolidated statements when the filing has them, individual otherwise."""
    has_con = st.groupby(["cd_cvm", "dt_refer", "version"])["consolidated"].transform("any")
    return st[st["consolidated"] == has_con]


def _ytd_only(st: pd.DataFrame) -> pd.DataFrame:
    flows = st["code"].str.startswith("3")
    if "dt_ini" not in st or not flows.any():
        return st
    f = st[flows].copy()
    first = f.groupby(["cd_cvm", "dt_refer", "version", "code"])["dt_ini"].transform("min")
    return pd.concat([st[~flows], f[f["dt_ini"] == first]])


def _fiscal_year_end(st: pd.DataFrame) -> dict:
    """Month in which each firm's fiscal year ends, from the start date of its year-to-date flows."""
    if "dt_ini" not in st:
        return {}
    f = st.loc[st["code"].str.startswith("3") & st["dt_ini"].notna(), ["cd_cvm", "dt_ini"]].reset_index(drop=True)
    f = f.assign(m=pd.to_datetime(f["dt_ini"], errors="coerce").dt.month).dropna(subset=["m"])
    if f.empty:
        return {}
    mode = f.groupby("cd_cvm")["m"].agg(lambda m: m.value_counts().idxmax())
    return {cd: (int(m) - 2) % 12 + 1 for cd, m in mode.items()}


def point_in_time(stmts: pd.DataFrame) -> pd.DataFrame:
    st = stmts.copy()
    st["dt_refer"] = pd.to_datetime(st["dt_refer"])
    st = _ytd_only(_scope(st))
    st = st.sort_values("dt_receb", kind="stable")

    vals: dict = defaultdict(list)
    for cd, per, code, receb, v in st[["cd_cvm", "dt_refer", "code", "dt_receb", "value"]].itertuples(index=False):
        vals[(cd, per, code)].append((receb, v))
    filings = st.drop_duplicates(["cd_cvm", "dt_refer", "version"])[["cd_cvm", "dt_refer", "version", "dt_receb"]]
    fy_end = _fiscal_year_end(st)

    rows = []
    for cd, fl in filings.groupby("cd_cvm"):
        fe = fy_end.get(cd, 12)
        first_filed = fl.groupby("dt_refer")["dt_receb"].min()
        restatements = fl.loc[fl["version"] > 1, "dt_receb"].sort_values().tolist()
        for t in sorted(fl["dt_receb"].unique()):
            t = pd.Timestamp(t)

            def get(per, code):
                hist = vals.get((cd, per, code))
                if not hist:
                    return np.nan
                i = bisect_right([h[0] for h in hist], t)
                return hist[i - 1][1] if i else np.nan

            def stock(per, codes):
                xs = [get(per, c) for c in codes]
                xs = [x for x in xs if not np.isnan(x)]
                return float(sum(xs)) if xs else np.nan

            def ttm(per, code):
                # Year to date plus the last annual figure minus the same period a year earlier. The fiscal
                # year may end in a month other than December (sugar and ethanol firms close in March).
                ytd = get(per, code)
                if per.month == fe:
                    return ytd
                year = per.year if per.month > fe else per.year - 1
                prev_annual = get(pd.Timestamp(year, fe, 1) + pd.offsets.MonthEnd(0), code)
                prev_ytd = get(per - pd.DateOffset(years=1) + pd.offsets.MonthEnd(0), code)
                return ytd + prev_annual - prev_ytd

            known = first_filed[first_filed <= t]
            p = known.index.max()
            row = {"cd_cvm": cd, "as_of": t, "dt_refer": p}
            row.update({k: stock(p, codes) for k, codes in STOCK.items()})
            row["tl"] = row["cl"] + row["ncl"]
            for k, code in FLOW.items():
                row[f"{k}_ttm"] = ttm(p, code)
            row["finexp_ttm"] = abs(row["finexp_ttm"])
            row["rev_ttm_prev"] = ttm(p - pd.DateOffset(years=1) + pd.offsets.MonthEnd(0), FLOW["rev"])
            quarters = [p - pd.DateOffset(months=3 * k) + pd.offsets.MonthEnd(0) for k in range(4)]
            ni = [ttm(q, FLOW["ni"]) for q in quarters]
            row["loss_q4"] = int(sum(1 for x in ni if not np.isnan(x) and x < 0))
            row["filing_delay_days"] = int((first_filed[p] - p).days)
            row["n_restatements_4q"] = sum(1 for r in restatements if t - pd.Timedelta(days=365) < r <= t)
            rows.append(row)
    return pd.DataFrame(rows)


def _div(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
    out[~np.isfinite(out)] = np.nan
    return out


def ratios(pit: pd.DataFrame, deflator: pd.Series | None = None) -> pd.DataFrame:
    r = pit.copy()
    r["wc_ta"] = _div(r["ca"] - r["cl"], r["ta"])
    r["re_ta"] = _div(r["re"], r["ta"])
    r["ebit_ta"] = _div(r["ebit_ttm"], r["ta"])
    r["eq_tl"] = _div(r["eq"], r["tl"])
    r["lev"] = _div(r["tl"], r["ta"])
    r["curr"] = _div(r["ca"], r["cl"])
    r["cash_ta"] = _div(r["cash"], r["ta"])
    r["int_cov"] = _div(r["ebit_ttm"], r["finexp_ttm"])
    r["margin"] = _div(r["ni_ttm"], r["rev_ttm"])
    r["sales_g"] = _div(r["rev_ttm"], r["rev_ttm_prev"]) - 1
    size = r["ta"].astype(float)
    if deflator is not None:
        size = size / deflator.reindex(r["dt_refer"].dt.to_period("M").dt.to_timestamp()).values
    r["log_size_real"] = np.log(size.where(size > 0))
    r["neg_eq"] = (r["eq"] < 0).astype(int)
    return r


def altman_z2(r: pd.DataFrame) -> pd.Series:
    """Altman (2005) emerging-market score: 3.25 + 6.56 X1 + 3.26 X2 + 6.72 X3 + 1.05 X4."""
    return 3.25 + 6.56 * r["wc_ta"] + 3.26 * r["re_ta"] + 6.72 * r["ebit_ta"] + 1.05 * r["eq_tl"]


def _plain(s) -> str:
    s = unicodedata.normalize("NFKD", str(s or ""))
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


# Since the 2016 reports every audit opinion carries the auditor-responsibility paragraph ("se existe
# incerteza relevante em relação a eventos ou condições que possam levantar dúvida significativa..."),
# so the rule only accepts statements: the section heading, "indicam a existência de incerteza",
# "pode levantar dúvida", "suscita dúvida substancial". The conditional "se existe" and the
# subjunctive "possam levantar" of that paragraph never match.
_U = r"incertezas? (?:relevantes?|significativas?|materia(?:l|is))"
_D = r"duvidas? (?:significativas?|substanci(?:al|ais)|relevantes?)"
_GC = re.compile(
    rf"\b{_U} relacionadas? (?:com|a) (?:a )?continuidade operacional"
    rf"|\b{_U} (?:sobre|quanto a|em relacao a) (?:a )?(?:sua )?(?:capacidade de )?continuidade operacional"
    rf"|\b(?:indicam|indica|indicaria|indicariam|indicador|indicativo|evidenciam|evidencia|demonstram|representam"
    rf"|configuram)\b[^.]{{0,40}}\b{_U}[^.]{{0,250}}continuidade operacional"
    rf"|(?<!se )(?<!que )\b(?:existe|existem|ha) (?:uma )?{_U}[^.]{{0,250}}continuidade operacional"
    rf"|\b(?:pode|podem|podera|poderao) levantar (?:uma )?duvidas?"
    rf"(?: (?:significativas?|substanci(?:al|ais)|relevantes?))?[^.]{{0,200}}continuidade operacional"
    rf"|\b(?:suscita|suscitam|levanta|levantam|gera|geram|indica|indicam) (?:uma )?{_D}[^.]{{0,80}}continuidade operacional")
# Doubt about an investee is not doubt about the filer.
_GC_INVESTEE = re.compile(r"\s*(?:da|de sua|das) (?:controlada|controladas|investida|coligada|subsidiaria)")


def _gc_text(t) -> bool:
    s = re.sub(r"\s+", " ", _plain(t))
    return any(not _GC_INVESTEE.match(s, m.end()) for m in _GC.finditer(s))


def going_concern_rule(parecer: pd.DataFrame) -> pd.DataFrame:
    """Flag filings whose auditor text states a material uncertainty about the filer's going concern."""
    p = parecer.assign(gc_flag=parecer["text"].map(_gc_text))
    return p.groupby(["cd_cvm", "dt_refer", "dt_receb"], as_index=False)["gc_flag"].any()
