"""Small shared data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Tier = Literal["cheap", "strong"]
Message = dict[str, str]


@dataclass(frozen=True)
class Usage:
    """Token usage of one model call. `estimated` is True when the provider sent none."""

    prompt_tokens: int
    completion_tokens: int
    estimated: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class Completion:
    """Result of one model call."""

    text: str
    usage: Usage
    latency_s: float
    model: str
    finish_reason: str | None = "stop"
    avg_logprob: float | None = None
    error: str | None = None
    extra: dict[str, object] = field(default_factory=dict)
