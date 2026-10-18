"""Routing policy: classify, call the chosen model, and escalate when the answer looks bad."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from .classifier import DifficultyClassifier, p_hard
from .costing import compute_cost
from .models import ModelClient, ModelError
from .types import Completion, Message, Tier, Usage
from .uncertainty import CheckResult, check_answer

Checker = Callable[[str, Completion], CheckResult]


@dataclass
class Attempt:
    tier: Tier
    completion: Completion | None
    cost_usd: float
    error: str | None = None


@dataclass
class RouteResult:
    query: str
    p_hard: float
    confidence_threshold: float
    initial_tier: Tier
    final_tier: Tier
    final: Completion
    attempts: list[Attempt] = field(default_factory=list)
    escalated: bool = False
    escalation_reason: str | None = None
    forced: bool = False
    classify_s: float = 0.0
    cost_usd: float = 0.0
    baseline_strong_cost_usd: float = 0.0  # what the strong model alone would have cost
    latency_s: float = 0.0  # classification + every model call

    @property
    def usage(self) -> Usage:
        """Total tokens spent across all attempts."""
        calls = [a.completion.usage for a in self.attempts if a.completion]
        return Usage(sum(u.prompt_tokens for u in calls), sum(u.completion_tokens for u in calls))

    @property
    def saved_usd(self) -> float:
        return self.baseline_strong_cost_usd - self.cost_usd


def last_user_text(messages: list[Message]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return messages[-1].get("content", "") if messages else ""


class Router:
    """Sends easy queries to `cheap`, hard ones to `strong`, escalating on bad cheap answers.

    A query goes to the cheap model only when the classifier's confidence that it is easy,
    ``1 - P(hard)``, is at least ``confidence_threshold``.
    """

    def __init__(
        self,
        classifier: DifficultyClassifier,
        cheap: ModelClient,
        strong: ModelClient,
        *,
        confidence_threshold: float = 0.5,
        fallback_enabled: bool = True,
        min_avg_logprob: float | None = None,
        checker: Checker | None = None,
    ) -> None:
        self.classifier = classifier
        self.clients: dict[Tier, ModelClient] = {"cheap": cheap, "strong": strong}
        self.confidence_threshold = confidence_threshold
        self.fallback_enabled = fallback_enabled
        self.min_avg_logprob = min_avg_logprob
        self._checker = checker

    def decide(self, p: float) -> Tier:
        """Policy: cheap iff P(easy) >= threshold."""
        return "cheap" if (1.0 - p) >= self.confidence_threshold else "strong"

    def _check(self, query: str, completion: Completion) -> CheckResult:
        if self._checker is not None:
            return self._checker(query, completion)
        return check_answer(query, completion, min_avg_logprob=self.min_avg_logprob)

    async def _call(
        self,
        tier: Tier,
        messages: list[Message],
        temperature: float | None,
        max_tokens: int | None,
    ) -> Attempt:
        client = self.clients[tier]
        try:
            completion = await client.complete(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                want_logprobs=self.min_avg_logprob is not None and tier == "cheap",
            )
        except ModelError as exc:
            return Attempt(tier, None, 0.0, error=str(exc))
        return Attempt(tier, completion, compute_cost(completion.usage, client.spec))

    async def handle(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        force: Tier | None = None,
    ) -> RouteResult:
        """Route one request. Raises ModelError if no model could produce an answer."""
        query = last_user_text(messages)

        t0 = time.perf_counter()
        p = p_hard(self.classifier, query)
        classify_s = time.perf_counter() - t0

        initial = force or self.decide(p)
        attempts: list[Attempt] = []
        escalated = False
        reason: str | None = None

        first = await self._call(initial, messages, temperature, max_tokens)
        attempts.append(first)
        final_attempt = first

        if initial == "cheap":
            if first.completion is None:
                reason = "error"
            else:
                verdict = self._check(query, first.completion)
                if not verdict.ok:
                    reason = verdict.reason
            # A forced-cheap request is honoured as-is; only auto-routed ones escalate.
            if reason and self.fallback_enabled and force is None:
                escalated = True
                final_attempt = await self._call("strong", messages, temperature, max_tokens)
                attempts.append(final_attempt)

        if final_attempt.completion is None:
            raise ModelError(final_attempt.error or "model call failed")

        final = final_attempt.completion
        cost = sum(a.cost_usd for a in attempts)
        strong_spec = self.clients["strong"].spec
        if final_attempt.tier == "strong":
            baseline = final_attempt.cost_usd
        else:  # strong model never ran: price the same tokens at strong rates
            baseline = compute_cost(final.usage, strong_spec)

        return RouteResult(
            query=query,
            p_hard=p,
            confidence_threshold=self.confidence_threshold,
            initial_tier=initial,
            final_tier=final_attempt.tier,
            final=final,
            attempts=attempts,
            escalated=escalated,
            escalation_reason=reason if escalated else None,
            forced=force is not None,
            classify_s=classify_s,
            cost_usd=cost,
            baseline_strong_cost_usd=baseline,
            latency_s=classify_s + sum(a.completion.latency_s for a in attempts if a.completion),
        )
