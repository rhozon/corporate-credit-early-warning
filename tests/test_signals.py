import json

import numpy as np
import pandas as pd
import pytest

from distress.llm import signals
from distress.llm.client import LLM


class Transport:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def __call__(self, payload):
        self.calls.append(payload)
        content = self.reply(payload) if callable(self.reply) else self.reply
        return {"choices": [{"message": {"content": json.dumps(content)}}], "usage": {"cost": 0.0001}}


def docs(n=3):
    return pd.DataFrame({"cd_cvm": [1] * n, "name": ["ACME S.A."] * n, "cnpj": ["11.222.333/0001-44"] * n,
                         "delivered": pd.date_range("2020-01-10", periods=n, freq="30D"),
                         "subject": [f"ACME S.A. assunto {i}" for i in range(n)], "link": ["x"] * n})


def test_parse_rejects_labels_outside_the_schema():
    with pytest.raises(ValueError):
        signals.parse_batch({"items": [{"i": 1, "labels": ["made_up"]}]}, n=1)
    ok = signals.parse_batch({"items": [{"i": 1, "labels": ["covenant", "missed_payment"]}]}, n=1)
    assert ok == [{"covenant", "missed_payment"}]


def test_missing_items_default_to_none_label():
    assert signals.parse_batch({"items": []}, n=2) == [{"none"}, {"none"}]


def test_classify_masks_the_company_and_returns_one_row_per_document(tmp_path):
    t = Transport({"items": [{"i": 1, "labels": ["debt_restructuring"]}, {"i": 2, "labels": ["none"]},
                             {"i": 3, "labels": ["covenant"]}]})
    out = signals.classify_subjects(docs(), LLM("m", cache_dir=tmp_path, transport=t), batch=25)
    sent = t.calls[0]["messages"][1]["content"]
    assert "ACME" not in sent and "[EMPRESA]" in sent
    assert out["debt_restructuring"].tolist() == [1, 0, 0]
    assert out["covenant"].tolist() == [0, 0, 1]
    assert len(out) == 3


def test_kappa_matches_hand_computation():
    a = pd.DataFrame({"covenant": [1, 1, 0, 0, 1, 0, 0, 0]})
    b = pd.DataFrame({"covenant": [1, 0, 0, 0, 1, 0, 1, 0]})
    # po = 6/8; pe = (3/8)(3/8) + (5/8)(5/8) = 34/64; kappa = (0.75 - 34/64) / (1 - 34/64)
    assert signals.agreement(a, b)["covenant"] == pytest.approx((0.75 - 34 / 64) / (1 - 34 / 64))


def test_features_only_use_documents_before_the_quarter():
    cl = pd.DataFrame({"cd_cvm": [1, 1, 1], "delivered": pd.to_datetime(["2020-01-15", "2020-03-31", "2020-05-01"]),
                       **{c: [1, 1, 1] if c == "covenant" else [0, 0, 0] for c in signals.CATEGORIES}})
    panel = pd.DataFrame({"cd_cvm": [1, 1], "q": pd.to_datetime(["2020-03-31", "2020-06-30"])})
    f = signals.signal_features(cl, panel)
    assert f["sig_covenant_4m"].tolist() == [1, 2]   # the 31 March filing counts only from the next quarter
    assert f["sig_covenant_12m"].tolist() == [1, 3]
    assert "sig_own_recovery_12m" not in f.columns


def test_gold_sheet_has_a_column_per_category(tmp_path):
    cl = signals.classify_subjects(docs(), LLM("m", cache_dir=tmp_path,
                                   transport=Transport({"items": [{"i": k, "labels": ["none"]} for k in (1, 2, 3)]})))
    path = tmp_path / "rotulos.xlsx"
    signals.gold_sheet(cl.assign(name="ACME S.A.", subject=docs()["subject"], link="x"), path, n=3, seed=1)
    sheet = pd.read_excel(path)
    assert {"id", "empresa", "data", "assunto", "link"} <= set(sheet.columns)
    assert all(signals.LABELS_PT[c] in sheet.columns for c in signals.CATEGORIES)
    assert len(sheet) == 3
