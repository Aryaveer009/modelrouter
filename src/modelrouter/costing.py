"""Cost accounting. Prices are USD per 1M tokens and come from configuration."""

from __future__ import annotations

import math
from collections.abc import Iterable

from .config import ModelSpec
from .types import Usage


def estimate_tokens(text: str) -> int:
    """Rough token count (~4 chars/token) used only when a provider reports no usage."""
    return max(1, math.ceil(len(text) / 4))


def compute_cost(usage: Usage, spec: ModelSpec) -> float:
    """Dollar cost of one call under `spec`'s prices."""
    return (
        usage.prompt_tokens * spec.price_in_per_m + usage.completion_tokens * spec.price_out_per_m
    ) / 1_000_000


def counterfactual_cost(usage: Usage, spec: ModelSpec) -> float:
    """What the same token counts would have cost on `spec` (used for 'savings' estimates)."""
    return compute_cost(usage, spec)


def total_cost(usages: Iterable[tuple[Usage, ModelSpec]]) -> float:
    return sum(compute_cost(u, s) for u, s in usages)


def format_usd(value: float) -> str:
    """Human-friendly dollar string that stays readable for sub-cent amounts."""
    if value == 0:
        return "$0"
    if abs(value) < 0.01:
        return f"${value:.5f}"
    return f"${value:.4f}" if abs(value) < 1 else f"${value:,.2f}"
