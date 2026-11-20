"""Chat-completion client for OpenRouter with a response cache and a hard spending cap.

Every request is deterministic (temperature 0) and cached on disk by a hash of
model, prompts and schema, so a rerun of the pipeline costs nothing and gives the
same answers. Spending is recorded in ``ledger.json`` from the cost the provider
reports; a call is refused before it is sent if a conservative estimate of its
cost would take the total past the cap.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Callable

URL = "https://openrouter.ai/api/v1/chat/completions"
CACHE = Path(__file__).resolve().parents[2] / "cache" / "llm"


class BudgetExceeded(RuntimeError):
    pass


def _replace(src: Path, dst: Path) -> None:
    os.replace(src, dst)


def _atomic_write(path: Path, text: str) -> None:
    """Write next to the target and rename, so a crash never leaves a half file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".json.part")
    part.write_text(text, encoding="utf-8")
    try:
        _replace(part, path)
    except Exception:
        part.unlink(missing_ok=True)
        raise


def _http(payload: dict, timeout: int = 120, tries: int = 4) -> dict:
    import requests

    key = os.environ["OPENROUTER_API_KEY"]
    for i in range(tries):
        r = requests.post(URL, json=payload, timeout=timeout,
                          headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        if r.status_code in (429, 500, 502, 503, 504) and i < tries - 1:
            time.sleep(4 * 2 ** i)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("unreachable")


def _parse_json(text: str):
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[t.find("\n") + 1:] if "\n" in t else t
    try:
        return json.loads(t)
    except json.JSONDecodeError as e:
        raise ValueError(f"model did not return valid JSON: {text[:200]!r}") from e


class LLM:
    def __init__(self, model: str, cap_usd: float = 20.0, cache_dir: Path = CACHE,
                 transport: Callable[[dict], dict] | None = None, price_in: float = 1e-6,
                 price_out: float = 4e-6, max_tokens: int = 800, safety: float = 1.5):
        self.model, self.cap, self.max_tokens, self.safety = model, cap_usd, max_tokens, safety
        self.cache_dir = Path(cache_dir)
        self.transport = transport or _http
        self.price_in, self.price_out = price_in, price_out

    # ledger -------------------------------------------------------------
    @property
    def _ledger_path(self) -> Path:
        return self.cache_dir / "ledger.json"

    def ledger(self) -> dict:
        if self._ledger_path.exists():
            return json.loads(self._ledger_path.read_text())
        return {"spent_usd": 0.0, "calls": 0}

    def _charge(self, usd: float) -> None:
        led = self.ledger()
        led["spent_usd"] += usd
        led["calls"] += 1
        _atomic_write(self._ledger_path, json.dumps(led))

    # calls --------------------------------------------------------------
    def _key(self, system: str, user: str, schema) -> str:
        blob = json.dumps([self.model, system, user, schema], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _estimate(self, system: str, user: str) -> float:
        tokens_in = (len(system) + len(user)) / 3.0
        return self.safety * (tokens_in * self.price_in + self.max_tokens * self.price_out)

    def complete(self, system: str, user: str, schema: dict | None = None):
        key = self._key(system, user, schema)
        path = self.cache_dir / key[:2] / f"{key}.json"
        if path.exists():
            content = json.loads(path.read_text(encoding="utf-8"))["content"]
            return _parse_json(content) if schema is not None else content

        if self.ledger()["spent_usd"] + self._estimate(system, user) > self.cap:
            raise BudgetExceeded(f"cap of US$ {self.cap:.2f} would be exceeded")
        payload = {"model": self.model, "temperature": 0, "max_tokens": self.max_tokens,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                   "usage": {"include": True}}
        if schema is not None:
            payload["response_format"] = {"type": "json_object"}
        resp = self.transport(payload)
        usage = resp.get("usage", {})
        cost = usage.get("cost")
        if cost is None:
            cost = usage.get("prompt_tokens", 0) * self.price_in + usage.get("completion_tokens", 0) * self.price_out
        self._charge(float(cost))
        content = resp["choices"][0]["message"]["content"] or ""
        parsed = _parse_json(content) if schema is not None else content

        _atomic_write(path, json.dumps({"model": self.model, "content": content, "usage": usage}, ensure_ascii=False))
        return parsed
