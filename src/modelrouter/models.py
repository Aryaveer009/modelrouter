"""Model clients: a real OpenAI-compatible client and a deterministic offline mock."""

from __future__ import annotations

import asyncio
import hashlib
import math
import random
import re
import time
from dataclasses import dataclass
from typing import Protocol

import httpx

from .config import ModelSpec
from .costing import estimate_tokens
from .types import Completion, Message, Usage


class ModelError(RuntimeError):
    """A model call failed after retries."""


class ModelClient(Protocol):
    spec: ModelSpec

    async def complete(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        want_logprobs: bool = False,
    ) -> Completion: ...


def messages_to_text(messages: list[Message]) -> str:
    return "\n".join(m.get("content", "") for m in messages)


class OpenAICompatClient:
    """Minimal async client for any `/chat/completions` endpoint (OpenAI, Groq, vLLM, ...)."""

    def __init__(self, spec: ModelSpec, *, timeout_s: float = 60.0, max_retries: int = 5) -> None:
        self.spec = spec
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(
            base_url=spec.base_url,
            headers={"Authorization": f"Bearer {spec.api_key}"},
            timeout=timeout_s,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def complete(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        want_logprobs: bool = False,
    ) -> Completion:
        payload: dict[str, object] = {
            **self.spec.extra_body,
            "model": self.spec.name,
            "messages": messages,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if want_logprobs:
            payload["logprobs"] = True

        last_error = "unknown error"
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            try:
                resp = await self._client.post("/chat/completions", json=payload)
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                await self._backoff(attempt, None)
                continue
            elapsed = time.perf_counter() - started

            if resp.status_code == 200:
                return self._parse(resp.json(), messages, elapsed)

            last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
            if resp.status_code == 400 and self._adapt_payload(payload, resp.text):
                continue  # retry immediately with an adjusted payload
            if resp.status_code in {408, 409, 429} or resp.status_code >= 500:
                hint = resp.headers.get("retry-after") or _retry_hint(resp.text)
                if hint is not None and _as_seconds(hint) > MAX_WAIT_S:
                    break  # e.g. a daily quota that refills in hours: fail fast, resume later
                await self._backoff(attempt, hint)
                continue
            break  # other 4xx errors are not retryable
        raise ModelError(f"{self.spec.name}: {last_error}")

    @staticmethod
    def _adapt_payload(payload: dict[str, object], body: str) -> bool:
        """Handle providers that renamed/removed parameters. Returns True if payload changed."""
        lowered = body.lower()
        if "max_completion_tokens" in lowered and "max_tokens" in payload:
            payload["max_completion_tokens"] = payload.pop("max_tokens")
            return True
        if "temperature" in lowered and "temperature" in payload:
            del payload["temperature"]
            return True
        if "logprobs" in lowered and "logprobs" in payload:
            del payload["logprobs"]
            return True
        return False

    @staticmethod
    async def _backoff(attempt: int, retry_after: str | None) -> None:
        delay = min(30.0, 1.5 * (2**attempt)) + random.random()
        if retry_after:
            delay = max(delay, _as_seconds(retry_after) + random.random())
        await asyncio.sleep(delay)

    def _parse(self, data: dict, messages: list[Message], elapsed: float) -> Completion:
        try:
            choice = data["choices"][0]
            text = choice["message"].get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelError(f"{self.spec.name}: malformed response: {str(data)[:300]}") from exc
        usage_raw = data.get("usage") or {}
        if "prompt_tokens" in usage_raw and "completion_tokens" in usage_raw:
            prompt = int(usage_raw["prompt_tokens"])
            completion = int(usage_raw["completion_tokens"])
            # Some providers (e.g. Gemini's OpenAI endpoint) bill hidden "thinking" tokens as
            # output but leave them out of completion_tokens; total_tokens still includes them.
            total = usage_raw.get("total_tokens")
            if isinstance(total, int):
                completion = max(completion, total - prompt)
            usage = Usage(prompt, completion)
        else:
            usage = Usage(
                estimate_tokens(messages_to_text(messages)), estimate_tokens(text), estimated=True
            )
        return Completion(
            text=text,
            usage=usage,
            latency_s=elapsed,
            model=data.get("model") or self.spec.name,
            finish_reason=choice.get("finish_reason"),
            avg_logprob=_avg_logprob(choice.get("logprobs")),
        )


MAX_WAIT_S = 300.0  # never sleep longer than this on a rate-limit hint
_RETRY_IN = re.compile(r"retry in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", re.IGNORECASE)


def _retry_hint(body: str) -> str | None:
    """Parse hints like 'Please retry in 6.27s' / '5h17m2.4s' from a provider error body."""
    m = _RETRY_IN.search(body)
    if not m or not any(m.groups()):
        return None
    h, mi, sec = m.groups()
    return str(int(h or 0) * 3600 + int(mi or 0) * 60 + float(sec or 0))


def _as_seconds(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return 0.0


def _avg_logprob(logprobs: dict | None) -> float | None:
    if not logprobs:
        return None
    content = logprobs.get("content")
    if not content:
        return None
    values = [t["logprob"] for t in content if isinstance(t.get("logprob"), (int, float))]
    return sum(values) / len(values) if values else None


# --------------------------------------------------------------------------- mock


@dataclass(frozen=True)
class OracleEntry:
    """Ground truth the mock model may 'know' (used only for offline harness tests)."""

    gold: str
    difficulty: float  # 0 = trivial, 1 = very hard


class MockClient:
    """Deterministic fake model for tests and demos. Never used for reported results.

    Without an oracle it echoes the query. With an oracle it answers correctly with a
    probability that depends on its tier and the query's difficulty, and sometimes hedges
    when it is about to be wrong, so the fallback path is exercised.
    """

    def __init__(
        self,
        spec: ModelSpec,
        tier: str,
        *,
        oracle: dict[str, OracleEntry] | None = None,
        seed: int = 0,
    ) -> None:
        self.spec = spec
        self.tier = tier
        self.oracle = oracle or {}
        self.seed = seed

    async def complete(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        want_logprobs: bool = False,
    ) -> Completion:
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        digest = hashlib.sha256(f"{self.seed}|{self.tier}|{user}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        entry = self.oracle.get(user)
        if entry is None:
            text = f"[mock-{self.tier}] You asked: {user[:80]}"
        else:
            if self.tier == "cheap":
                p_correct = 0.97 - 0.85 * entry.difficulty
            else:
                p_correct = 0.99 - 0.15 * entry.difficulty
            if rng.random() < p_correct:
                answer = entry.gold
                text = f"Reasoning briefly.\nFinal answer: {answer}"
            elif rng.random() < 0.4:
                text = "I'm not sure about this one."
            else:
                text = "Reasoning briefly.\nFinal answer: " + _wrong_answer(entry.gold, rng)
        prompt_tokens = estimate_tokens(messages_to_text(messages))
        completion_tokens = estimate_tokens(text)
        base, per_tok = (0.25, 0.004) if self.tier == "cheap" else (0.9, 0.012)
        return Completion(
            text=text,
            usage=Usage(prompt_tokens, completion_tokens),
            latency_s=base + per_tok * completion_tokens,  # synthetic, no sleeping
            model=self.spec.name,
            finish_reason="stop",
            avg_logprob=-0.2 - rng.random() if want_logprobs else None,
        )


def _wrong_answer(gold: str, rng: random.Random) -> str:
    if re.fullmatch(r"[A-D]", gold):
        return rng.choice([c for c in "ABCD" if c != gold])
    try:
        value = float(gold)
        shifted = value + rng.choice([-2, -1, 1, 2, 10])
        return (
            str(int(shifted))
            if math.isfinite(shifted) and shifted == int(shifted)
            else str(shifted)
        )
    except ValueError:
        return "unknown entity"
