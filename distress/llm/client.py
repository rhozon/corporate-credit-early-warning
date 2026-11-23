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
import threading
import time
from pathlib import Path
from typing import Callable

URL = "https://openrouter.ai/api/v1/chat/completions"
# The live cache sits outside any synced folder (sync clients lock files mid-write).
# The repository ships it packed in one file: see ``pack_cache`` and ``unpack_cache``.
CACHE = Path(os.environ.get("DISTRESS_LLM_CACHE", Path.home() / ".cache" / "distress" / "llm"))
PACKED = Path(__file__).resolve().parents[2] / "cache" / "llm_answers.jsonl.gz"
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(str(path.resolve()), threading.Lock())


class BudgetExceeded(RuntimeError):
    pass


def _replace(src: Path, dst: Path, tries: int = 20) -> None:
    """os.replace, retried briefly: on Windows a just-written file can be held by a scanner."""
    for i in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(0.02 * (i + 1))


def _atomic_write(path: Path, text: str, tries: int = 10) -> None:
    """Write next to the target and rename, so a crash never leaves a half file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".json.part")
    for i in range(tries):
        try:
            part.write_text(text, encoding="utf-8")
            break
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(0.05 * (i + 1))
    try:
        _replace(part, path)
    except Exception:
        part.unlink(missing_ok=True)
        raise


def pack_cache(cache_dir: Path, out: Path, transform: Callable[[str], str] | None = None) -> int:
    """All cached answers in one gzipped JSON-lines file, keyed by their hash. ``transform`` rewrites each
    answer's text before it ships (the repository masks solvent companies' names in the memo drafts)."""
    import gzip

    n = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as f:
        for p in sorted(Path(cache_dir).rglob("*.json")):
            if p.name == "ledger.json":
                continue
            rec = json.loads(p.read_text(encoding="utf-8"))
            if transform is not None and isinstance(rec.get("content"), str):
                rec["content"] = transform(rec["content"])
            f.write(json.dumps({"key": p.stem, **rec}, ensure_ascii=False) + "\n")
            n += 1
    return n


def unpack_cache(packed: Path, cache_dir: Path) -> int:
    import gzip

    n = 0
    with gzip.open(packed, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            key = rec.pop("key")
            path = Path(cache_dir) / key[:2] / f"{key}.json"
            if not path.exists():
                _atomic_write(path, json.dumps(rec, ensure_ascii=False))
                n += 1
    return n


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


GOOGLE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _google_request(payload: dict) -> tuple[str, dict]:
    """Chat payload to a Gemini API request. Gemma models take no separate system instruction."""
    model = payload["model"].split("/", 1)[1]
    text = "\n\n".join(m["content"] for m in payload["messages"])
    body = {"contents": [{"role": "user", "parts": [{"text": text}]}],
            "generationConfig": {"temperature": payload.get("temperature", 0),
                                 "maxOutputTokens": payload.get("max_tokens", 800),
                                 # Gemma 4 reasons before answering; keep that to a minimum for classification.
                                 "thinkingConfig": {"thinkingLevel": "minimal"}}}
    return GOOGLE.format(model=model), body


def _google_parse(resp: dict) -> dict:
    parts = ((resp.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    u = resp.get("usageMetadata", {})
    # No cost field: the client prices the tokens with the model's configured rates (zero on the free tier).
    return {"choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": u.get("promptTokenCount", 0),
                      "completion_tokens": u.get("candidatesTokenCount", 0)}}


def _google(payload: dict, timeout: int = 180, tries: int = 30) -> dict:
    """Google AI Studio (free tier). Rate limits answer 429 and overload 503; back off and retry.

    The key goes in the ``x-goog-api-key`` header so that no URL, traceback or log ever carries it.
    """
    import requests

    url, body = _google_request(payload)
    headers = {"x-goog-api-key": os.environ["GOOGLE_API_KEY"], "Content-Type": "application/json"}
    for i in range(tries):
        try:
            r = requests.post(url, json=body, headers=headers, timeout=timeout)
        except requests.exceptions.RequestException:
            if i == tries - 1:
                raise
            time.sleep(min(120, 10 * (i + 1)))
            continue
        if r.status_code in (429, 500, 503) and i < tries - 1:
            time.sleep(min(120, 10 * (i + 1)))
            continue
        if r.status_code >= 400:
            raise RuntimeError(f"Google API error {r.status_code} for {payload['model']}")
        return _google_parse(r.json())
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
        self.transport = transport or (_google if model.startswith("google/") else _http)
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
        with _lock_for(self._ledger_path):
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

        with _lock_for(self._ledger_path):
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
        # Cached before parsing: an answer that is not valid JSON is paid once and fails the same way on a rerun,
        # so the caller's fallback replays from the cache too.
        _atomic_write(path, json.dumps({"model": self.model, "content": content, "usage": usage}, ensure_ascii=False))
        return _parse_json(content) if schema is not None else content
