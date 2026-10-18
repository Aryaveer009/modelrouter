"""Run a model over benchmark records with caching, concurrency and a spend guard."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..config import ModelSpec
from ..costing import compute_cost, estimate_tokens
from ..data.records import Record
from ..models import ModelClient, ModelError
from ..types import Completion, Message, Usage
from .tasks import EXPECTED_OUTPUT_TOKENS, MAX_TOKENS, build_messages


class BudgetExceeded(RuntimeError):
    """Actual spend passed the user's hard budget; the run was stopped."""


def messages_key(messages: list[Message]) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()[:20]


def record_key(record: Record) -> str:
    return messages_key(build_messages(record))


class ResponseCache:
    """Append-only JSONL cache so an interrupted run resumes without re-spending money."""

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        self._items: dict[str, Completion] = {}
        self._lock = threading.Lock()
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    self._items[d["key"]] = Completion(
                        text=d["text"],
                        usage=Usage(
                            d["prompt_tokens"], d["completion_tokens"], d.get("est", False)
                        ),
                        latency_s=d["latency_s"],
                        model=d["model"],
                        finish_reason=d.get("finish_reason"),
                        avg_logprob=d.get("avg_logprob"),
                    )

    def get(self, key: str) -> Completion | None:
        return self._items.get(key)

    def put(self, key: str, c: Completion) -> None:
        with self._lock:
            self._items[key] = c
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(
                        json.dumps(
                            {
                                "key": key,
                                "text": c.text,
                                "prompt_tokens": c.usage.prompt_tokens,
                                "completion_tokens": c.usage.completion_tokens,
                                "est": c.usage.estimated,
                                "latency_s": c.latency_s,
                                "model": c.model,
                                "finish_reason": c.finish_reason,
                                "avg_logprob": c.avg_logprob,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )


@dataclass
class RunStats:
    calls: int = 0
    cached: int = 0
    errors: int = 0
    spend_usd: float = 0.0


async def run_model(
    records: list[Record],
    client: ModelClient,
    *,
    cache: ResponseCache | None = None,
    concurrency: int = 4,
    temperature: float = 0.0,
    budget_usd: float | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, Completion], RunStats]:
    """Return {record.id: Completion}. Failed calls get a Completion with `.error` set."""
    cache = cache or ResponseCache(None)
    results: dict[str, Completion] = {}
    stats = RunStats()
    sem = asyncio.Semaphore(concurrency)
    stop = asyncio.Event()
    done = 0

    async def one(rec: Record) -> None:
        nonlocal done
        key = record_key(rec)
        hit = cache.get(key)
        if hit is not None:
            results[rec.id] = hit
            stats.cached += 1
        else:
            async with sem:
                if stop.is_set():
                    return
                try:
                    comp = await client.complete(
                        build_messages(rec),
                        temperature=temperature,
                        max_tokens=MAX_TOKENS[rec.task],
                    )
                except ModelError as exc:
                    comp = Completion("", Usage(0, 0), 0.0, client.spec.name, None, error=str(exc))
                    stats.errors += 1
                else:
                    cache.put(key, comp)
                stats.calls += 1
                stats.spend_usd += compute_cost(comp.usage, client.spec)
                results[rec.id] = comp
                if budget_usd is not None and stats.spend_usd > budget_usd:
                    stop.set()
        done += 1
        if progress and done % 25 == 0:
            progress(
                f"  {client.spec.name}: {done}/{len(records)} "
                f"(spent ${stats.spend_usd:.4f}, errors {stats.errors})"
            )

    await asyncio.gather(*(one(r) for r in records))
    if stop.is_set():
        raise BudgetExceeded(
            f"stopped: spend ${stats.spend_usd:.4f} exceeded budget ${budget_usd:.4f}"
        )
    return results, stats


class ReplayClient:
    """Serves previously recorded completions so the *real* Router code can be evaluated at
    many thresholds without paying for new model calls."""

    def __init__(self, spec: ModelSpec, table: dict[str, Completion]) -> None:
        self.spec = spec
        self._table = table

    async def complete(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        want_logprobs: bool = False,
    ) -> Completion:
        comp = self._table.get(messages_key(messages))
        if comp is None:
            raise ModelError(f"no recorded response for {self.spec.name}")
        if comp.error:
            raise ModelError(comp.error)
        return comp


# --------------------------------------------------------------------- cost estimation


@dataclass
class CostEstimate:
    model: str
    n_calls: int
    prompt_tokens: int
    expected_completion_tokens: int
    max_completion_tokens: int
    expected_usd: float
    worst_case_usd: float


def estimate_cost(records: list[Record], spec: ModelSpec) -> CostEstimate:
    """Pre-flight estimate for running `spec` over `records` (one call per record)."""
    prompt = sum(
        estimate_tokens("\n".join(m["content"] for m in build_messages(r))) for r in records
    )
    expected = sum(EXPECTED_OUTPUT_TOKENS[r.task] for r in records)
    worst = sum(MAX_TOKENS[r.task] for r in records)
    return CostEstimate(
        model=spec.name,
        n_calls=len(records),
        prompt_tokens=prompt,
        expected_completion_tokens=expected,
        max_completion_tokens=worst,
        expected_usd=compute_cost(Usage(prompt, expected), spec),
        worst_case_usd=compute_cost(Usage(prompt, worst), spec),
    )


def confirm_spend(estimates: list[CostEstimate], *, assume_yes: bool, out=None) -> bool:
    """Print the estimate and ask the user to confirm. Returns True to proceed."""
    out = out or sys.stderr
    print("\nEstimated API spend for this run (token counts are approximate):", file=out)
    for e in estimates:
        print(
            f"  {e.model:<32} {e.n_calls:>4} calls  ~{e.prompt_tokens:,} in / "
            f"~{e.expected_completion_tokens:,} out tokens  "
            f"expected ${e.expected_usd:.4f}  worst-case ${e.worst_case_usd:.4f}",
            file=out,
        )
    exp = sum(e.expected_usd for e in estimates)
    worst = sum(e.worst_case_usd for e in estimates)
    print(f"  {'TOTAL':<32} expected ${exp:.4f}   worst-case ${worst:.4f}\n", file=out)
    if assume_yes:
        print("--yes given: proceeding without prompt.", file=out)
        return True
    if not sys.stdin.isatty():
        print("Not an interactive terminal; re-run with --yes to confirm the spend.", file=out)
        return False
    return input("Proceed and spend real money? [y/N] ").strip().lower() in {"y", "yes"}
