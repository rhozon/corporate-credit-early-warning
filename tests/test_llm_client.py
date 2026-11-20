import json

import pytest

from distress.llm import client as llm_client
from distress.llm.client import LLM, BudgetExceeded
from distress.llm.mask import mask


class Transport:
    """Stands in for the OpenRouter HTTP call and records what was sent."""

    def __init__(self, content='{"ok": true}', cost=0.001):
        self.calls, self.content, self.cost = [], content, cost

    def __call__(self, payload):
        self.calls.append(payload)
        return {"choices": [{"message": {"content": self.content}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": self.cost}}


def make(tmp_path, transport, cap=20.0, price_in=1e-6):
    return LLM("vendor/model-x", cap_usd=cap, cache_dir=tmp_path, transport=transport,
               price_in=price_in, price_out=2e-6, max_tokens=200)


def test_second_identical_call_is_served_from_cache(tmp_path):
    t = Transport()
    llm = make(tmp_path, t)
    a = llm.complete("sys", "user", schema={"type": "object"})
    b = llm.complete("sys", "user", schema={"type": "object"})
    assert a == b == {"ok": True}
    assert len(t.calls) == 1


def test_temperature_zero_and_model_are_sent(tmp_path):
    t = Transport()
    make(tmp_path, t).complete("sys", "user")
    assert t.calls[0]["temperature"] == 0
    assert t.calls[0]["model"] == "vendor/model-x"


def test_ledger_accumulates_reported_cost(tmp_path):
    t = Transport(cost=0.25)
    llm = make(tmp_path, t)
    llm.complete("s", "u1")
    llm.complete("s", "u2")
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert ledger["spent_usd"] == pytest.approx(0.5)
    assert ledger["calls"] == 2


def test_cap_blocks_the_call_that_would_cross_it(tmp_path):
    t = Transport(cost=0.25)
    llm = make(tmp_path, t, cap=0.30, price_in=1e-4)
    llm.complete("s", "u1")
    with pytest.raises(BudgetExceeded):
        llm.complete("s", "u2" * 2000)  # about 1,300 tokens at 1e-4 per token: over what is left
    assert len(t.calls) == 1
    assert json.loads((tmp_path / "ledger.json").read_text())["spent_usd"] == pytest.approx(0.25)


def test_failed_cache_write_leaves_no_entry_and_rerun_resumes(tmp_path, monkeypatch):
    t = Transport()
    llm = make(tmp_path, t)
    real_replace = llm_client._replace

    def crash(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(llm_client, "_replace", crash)
    with pytest.raises(OSError):
        llm.complete("s", "u")
    assert not list(tmp_path.rglob("*.json.part"))  # no half-written entry kept
    assert [p.name for p in tmp_path.rglob("*.json") if p.name != "ledger.json"] == []
    monkeypatch.setattr(llm_client, "_replace", real_replace)
    assert llm.complete("s", "u") == '{"ok": true}'
    assert len(t.calls) == 2


def test_ledger_is_exact_under_concurrent_calls(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import time

    class Slow(Transport):
        def __call__(self, payload):
            time.sleep(0.01)
            return super().__call__(payload)

    llm = make(tmp_path, Slow(cost=0.01))
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(lambda i: llm.complete("s", f"u{i}"), range(40)))
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert ledger["calls"] == 40
    assert ledger["spent_usd"] == pytest.approx(0.40)


def test_google_payload_folds_system_into_user_and_sets_temperature():
    payload = {"model": "google/gemma-4-31b-it", "temperature": 0, "max_tokens": 300,
               "messages": [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}]}
    url, body = llm_client._google_request(payload)
    assert url.endswith("/models/gemma-4-31b-it:generateContent")
    assert "key" not in url  # the key travels in a header, so errors and logs never print it
    assert body["contents"][0]["parts"][0]["text"] == "SYS\n\nUSER"
    assert body["generationConfig"] == {"temperature": 0, "maxOutputTokens": 300,
                                        "thinkingConfig": {"thinkingLevel": "minimal"}}


def test_google_response_maps_to_chat_shape_with_zero_cost():
    resp = {"candidates": [{"content": {"parts": [{"text": '{"a":'}, {"text": " 1}"}]}}],
            "usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 7}}
    out = llm_client._google_parse(resp)
    assert out["choices"][0]["message"]["content"] == '{"a": 1}'
    assert out["usage"] == {"prompt_tokens": 50, "completion_tokens": 7, "cost": 0.0}


def test_google_models_pick_the_google_transport(tmp_path):
    assert LLM("google/gemma-4-31b-it", cache_dir=tmp_path).transport is llm_client._google
    assert LLM("qwen/qwen3", cache_dir=tmp_path).transport is llm_client._http


def test_transient_write_failure_is_retried(tmp_path, monkeypatch):
    from pathlib import Path

    real = Path.write_text
    state = {"n": 0}

    def flaky(self, *a, **k):
        if self.suffix == ".part" and state["n"] == 0:
            state["n"] += 1
            raise PermissionError("locked by sync client")
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "write_text", flaky)
    assert make(tmp_path, Transport()).complete("s", "u") == '{"ok": true}'
    assert state["n"] == 1


def test_cache_pack_round_trip(tmp_path):
    src, dst, pack = tmp_path / "a", tmp_path / "b", tmp_path / "answers.jsonl.gz"
    llm = make(src, Transport())
    llm.complete("s", "u1")
    llm.complete("s", "u2")
    assert llm_client.pack_cache(src, pack) == 2
    assert llm_client.unpack_cache(pack, dst) == 2
    t = Transport()
    assert make(dst, t).complete("s", "u1") == '{"ok": true}' and t.calls == []


def test_invalid_json_under_schema_raises(tmp_path):
    llm = make(tmp_path, Transport(content="not json"))
    with pytest.raises(ValueError):
        llm.complete("s", "u", schema={"type": "object"})


def test_mask_hides_names_and_cnpj_in_any_format():
    text = "A AMERICANAS S.A. (CNPJ 00.776.574/0001-56) informa; Americanas e 00776574000156."
    out = mask(text, ["AMERICANAS S.A.", "Americanas"], "00.776.574/0001-56")
    assert "mericanas" not in out
    assert "00776574" not in out.replace(".", "").replace("/", "").replace("-", "")
    assert out.count("[EMPRESA]") >= 3
