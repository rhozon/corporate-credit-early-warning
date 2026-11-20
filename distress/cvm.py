"""Public data of listed companies from CVM (dados.cvm.gov.br).

Three sources are used:

* the company registry (``cad``), with the current issuer status (operational,
  judicial recovery, bankruptcy...) and the date it started;
* the index of periodic and eventual filings (``ipe``): material facts, notices,
  and the category CVM keeps for companies in judicial or extrajudicial recovery;
* quarterly (``itr``) and annual (``dfp``) financial statements, whose index gives
  the date each filing reached CVM (``DT_RECEB``). That date, not the end of the
  quarter, is when the numbers became public.

Files are downloaded once into ``raw_dir`` and read from there afterwards.
"""
from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import pandas as pd

BASE = "https://dados.cvm.gov.br/dados/CIA_ABERTA"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

_URLS = {
    "cad": "CAD/DADOS/cad_cia_aberta.csv",
    "ipe": "DOC/IPE/DADOS/ipe_cia_aberta_{year}.zip",
    "itr": "DOC/ITR/DADOS/itr_cia_aberta_{year}.zip",
    "dfp": "DOC/DFP/DADOS/dfp_cia_aberta_{year}.zip",
}


def _get(url: str, timeout: int = 300, tries: int = 4) -> bytes:
    import requests

    for i in range(tries):
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            return r.content
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))
    raise RuntimeError("unreachable")


def download(kind: str, year: int | None = None, raw_dir: Path = RAW) -> Path:
    """Path of the cached file, downloading it the first time."""
    rel = _URLS[kind].format(year=year)
    path = Path(raw_dir) / Path(rel).name
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(_get(f"{BASE}/{rel}"))
    tmp.replace(path)
    return path


def _csv(data, **kw) -> pd.DataFrame:
    return pd.read_csv(data, sep=";", encoding="latin1", dtype=str, **kw)


def read_cad(raw_dir: Path = RAW) -> pd.DataFrame:
    d = _csv(download("cad", None, raw_dir))
    out = pd.DataFrame({
        "cd_cvm": d["CD_CVM"].astype(int),
        "cnpj": d["CNPJ_CIA"],
        "name": d["DENOM_SOCIAL"],
        "status": d["SIT"],
        "status_date": pd.to_datetime(d["DT_INI_SIT"], errors="coerce"),
        "cancel_date": pd.to_datetime(d["DT_CANCEL"], errors="coerce"),
        "cancel_reason": d["MOTIVO_CANCEL"],
        "issuer_status": d["SIT_EMISSOR"],
        "issuer_status_date": pd.to_datetime(d["DT_INI_SIT_EMISSOR"], errors="coerce"),
        "sector": d["SETOR_ATIV"],
        "register_date": pd.to_datetime(d["DT_REG"], errors="coerce"),
    })
    return out


def read_ipe(years, raw_dir: Path = RAW) -> pd.DataFrame:
    frames = []
    for y in years:
        z = zipfile.ZipFile(download("ipe", y, raw_dir))
        frames.append(_csv(z.open(z.namelist()[0])))
    d = pd.concat(frames, ignore_index=True)
    return pd.DataFrame({
        "cd_cvm": d["Codigo_CVM"].astype(int),
        "cnpj": d["CNPJ_Companhia"],
        "name": d["Nome_Companhia"],
        "category": d["Categoria"].fillna(""),
        "type": d["Tipo"].fillna(""),
        "subject": d["Assunto"].fillna(""),
        "delivered": pd.to_datetime(d["Data_Entrega"], errors="coerce"),
        "link": d["Link_Download"],
    })


def _index(z: zipfile.ZipFile, kind: str, year: int) -> pd.DataFrame:
    idx = _csv(z.open(f"{kind}_cia_aberta_{year}.csv"))
    idx = idx.assign(cd_cvm=idx["CD_CVM"].astype(int), version=idx["VERSAO"].astype(int),
                     dt_receb=pd.to_datetime(idx["DT_RECEB"], errors="coerce"))
    return idx[["cd_cvm", "DT_REFER", "version", "dt_receb"]].rename(columns={"DT_REFER": "dt_refer"})


def read_statements(kind: str, years, table: str, raw_dir: Path = RAW) -> pd.DataFrame:
    """Account values of one statement table, current period only, in reais.

    ``table`` is the CVM file stem: BPA (assets), BPP (liabilities and equity),
    DRE (income), DFC_MI (cash flow, indirect method). Both the consolidated and
    the individual (parent-only) versions are returned, flagged by ``consolidated``.
    """
    frames = []
    for y in years:
        z = zipfile.ZipFile(download(kind, y, raw_dir))
        names = set(z.namelist())
        idx = _index(z, kind, y)
        for scope, flag in (("con", True), ("ind", False)):
            name = f"{kind}_cia_aberta_{table}_{scope}_{y}.csv"
            if name not in names:
                continue
            d = _csv(z.open(name))
            d = d[d["ORDEM_EXERC"].str.startswith("ÚLTIMO")]
            scale = d["ESCALA_MOEDA"].map({"MIL": 1000.0, "UNIDADE": 1.0}).fillna(1.0)
            out = pd.DataFrame({
                "cd_cvm": d["CD_CVM"].astype(int),
                "dt_refer": d["DT_REFER"],
                "version": d["VERSAO"].astype(int),
                "consolidated": flag,
                "code": d["CD_CONTA"],
                "value": pd.to_numeric(d["VL_CONTA"], errors="coerce") * scale,
            })
            if "DT_INI_EXERC" in d:
                out["dt_ini"] = d["DT_INI_EXERC"].values
            frames.append(out.merge(idx, on=["cd_cvm", "dt_refer", "version"], how="left"))
    st = pd.concat(frames, ignore_index=True)
    cols = ["cd_cvm", "dt_refer", "version", "consolidated", "code", "value", "dt_receb"]
    return st[cols + [c for c in st.columns if c not in cols]]


def read_parecer(kind: str, years, raw_dir: Path = RAW) -> pd.DataFrame:
    """Auditor reports and management statements, with the filing date of each."""
    frames = []
    for y in years:
        z = zipfile.ZipFile(download(kind, y, raw_dir))
        name = f"{kind}_cia_aberta_parecer_{y}.csv"
        if name not in z.namelist():
            continue
        idx = _csv(z.open(f"{kind}_cia_aberta_{y}.csv"))
        idx = idx.assign(cd_cvm=idx["CD_CVM"].astype(int), version=idx["VERSAO"].astype(int),
                         dt_receb=pd.to_datetime(idx["DT_RECEB"], errors="coerce"))
        d = _csv(z.open(name)).assign(version=lambda x: x["VERSAO"].astype(int))
        d = d.merge(idx[["CNPJ_CIA", "DT_REFER", "version", "cd_cvm", "dt_receb"]],
                    on=["CNPJ_CIA", "DT_REFER", "version"], how="inner")
        frames.append(pd.DataFrame({"cd_cvm": d["cd_cvm"], "dt_refer": d["DT_REFER"], "version": d["version"],
                                    "dt_receb": d["dt_receb"], "kind": d["TP_PARECER_DECL"],
                                    "text": d["TXT_PARECER_DECL"].fillna("")}))
    cols = ["cd_cvm", "dt_refer", "version", "dt_receb", "kind", "text"]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
