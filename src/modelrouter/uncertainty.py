"""Checks that decide whether a cheap model's answer should be trusted or escalated."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .types import Completion

# Phrases that signal the model itself does not know. Kept specific to avoid flagging
# ordinary answers that merely contain "cannot" or "unclear".
HEDGE_PATTERNS = re.compile(
    r"\b(i(?:'m| am) not (?:sure|certain)|i (?:do not|don't) know|i(?:'m| am) unable to|"
    r"i can(?:not|'t) (?:determine|answer|say)|cannot be determined|not enough information|"
    r"insufficient information|i have no (?:idea|information)|as an ai(?: language model)?)\b",
    re.IGNORECASE,
)
_MC_OPTIONS = re.compile(r"(?m)^\s*\(?[A-D][\.\):]\s+\S")
_MATH_HINT = re.compile(
    r"\b(how many|how much|calculate|compute|total|sum|difference|product|average|percent|"
    r"each|per|solve|equation)\b",
    re.IGNORECASE,
)
_LETTER = re.compile(r"(?<![A-Za-z])[A-D](?![A-Za-z])")


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    reason: str = "ok"


def looks_multiple_choice(query: str) -> bool:
    return len(_MC_OPTIONS.findall(query)) >= 2


def check_answer(
    query: str,
    completion: Completion,
    *,
    min_avg_logprob: float | None = None,
) -> CheckResult:
    """Return whether the cheap model's answer passes basic sanity and confidence checks."""
    text = completion.text.strip()
    if completion.error:
        return CheckResult(False, "error")
    if not text:
        return CheckResult(False, "empty")
    if completion.finish_reason == "length":
        return CheckResult(False, "truncated")
    if HEDGE_PATTERNS.search(text):
        return CheckResult(False, "hedging")
    if looks_multiple_choice(query) and not _LETTER.search(text):
        return CheckResult(False, "no_choice_letter")
    if (
        _MATH_HINT.search(query)
        and any(c.isdigit() for c in query)
        and not any(c.isdigit() for c in text)
    ):
        return CheckResult(False, "no_number")
    if (
        min_avg_logprob is not None
        and completion.avg_logprob is not None
        and completion.avg_logprob < min_avg_logprob
    ):
        return CheckResult(False, "low_confidence")
    return CheckResult(True)
