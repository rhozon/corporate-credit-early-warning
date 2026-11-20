import json

import pandas as pd

from distress.llm.adjudicate import event_classifier
from distress.llm.client import LLM


class Transport:
    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def __call__(self, payload):
        self.calls.append(payload)
        return {"choices": [{"message": {"content": json.dumps(self.answer)}}],
                "usage": {"cost": 0.0001}}


def row(subject, name="OI S.A."):
    return pd.Series({"cd_cvm": 1, "name": name, "source": "material_fact", "date": pd.Timestamp("2016-06-20"),
                      "evidence": subject, "link": "x", "event_type": None})


def test_own_filing_maps_to_event(tmp_path):
    t = Transport({"status": "filing", "event_type": "RJ"})
    clf = event_classifier(LLM("m", cache_dir=tmp_path, transport=t))
    assert clf(row("Pedido de Recuperação Judicial")) == {"own_filing": True, "event_type": "RJ", "status": "filing"}
    sent = t.calls[0]["messages"][1]["content"]
    assert "Pedido de Recuperação Judicial" in sent and "OI S.A." in sent


def test_ongoing_process_counts_as_own_and_other_does_not(tmp_path):
    ongoing = event_classifier(LLM("m", cache_dir=tmp_path / "a", transport=Transport({"status": "ongoing", "event_type": "RJ"})))
    other = event_classifier(LLM("m", cache_dir=tmp_path / "b", transport=Transport({"status": "other", "event_type": None})))
    assert ongoing(row("Aprovação do plano"))["own_filing"] is True
    assert other(row("Recuperação judicial de cliente"))["own_filing"] is False


def test_exit_is_flagged_but_not_a_filing(tmp_path):
    clf = event_classifier(LLM("m", cache_dir=tmp_path, transport=Transport({"status": "exit", "event_type": "RJ"})))
    out = clf(row("Encerramento da recuperação judicial"))
    assert out["own_filing"] is False and out["status"] == "exit"


def test_unknown_labels_are_rejected_to_other(tmp_path):
    clf = event_classifier(LLM("m", cache_dir=tmp_path, transport=Transport({"status": "maybe", "event_type": "XX"})))
    out = clf(row("Algo"))
    assert out == {"own_filing": False, "event_type": None, "status": "other"}
