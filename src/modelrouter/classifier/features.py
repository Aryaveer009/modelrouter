"""Hand-crafted difficulty signals (length, keywords, math/code cues)."""

from __future__ import annotations

import math
import re

FEATURE_NAMES = (
    "log_words",
    "digit_ratio",
    "n_numbers",
    "math_ops",
    "math_keywords",
    "code_signal",
    "reasoning_keywords",
    "multiple_choice",
    "factoid_cue",
    "is_short",
)

_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_MATH_OPS = re.compile(r"(?<=\d)\s*[-+*/=^×÷%]\s*(?=[\d(])")
_MATH_KW = re.compile(
    r"\b(solve|calculate|compute|how many|how much|total|sum|difference|product|average|"
    r"percent|percentage|ratio|remainder|equation|integral|derivative|probability|"
    r"each|per|twice|half|times|fraction)\b",
    re.IGNORECASE,
)
_CODE = re.compile(
    r"(```|\bdef \w+\(|\bclass \w+|\bimport \w+|\breturn\b|[{};]\s*$|"
    r"\bwrite (?:a|an|the)? ?(?:python|java|sql|function|program|script|code)|"
    r"\bimplement\b|\balgorithm\b|\bcomplexity\b)",
    re.IGNORECASE | re.MULTILINE,
)
_REASONING_KW = re.compile(
    r"\b(prove|derive|explain why|step by step|analy[sz]e|compare|evaluate|justify|"
    r"optimi[sz]e|design|trade-?off|which of the following|assume|suppose|given that|"
    r"consequence|implies|if and only if)\b",
    re.IGNORECASE,
)
_MC = re.compile(r"(?m)^\s*\(?[A-D][\.\):]\s+\S")
_FACTOID = re.compile(
    r"\b(what is the capital|who (?:wrote|was|is|invented|discovered|painted|directed|founded)|"
    r"when (?:was|did|is)|where (?:is|was|did)|which (?:country|city|year|river|state)|"
    r"what year|name the|what is the name|in which|how old)\b",
    re.IGNORECASE,
)


def extract_features(text: str) -> dict[str, float]:
    """Compute the feature dict for one query."""
    words = text.split()
    n_words = len(words)
    n_chars = max(1, len(text))
    digits = sum(c.isdigit() for c in text)
    return {
        "log_words": math.log1p(n_words),
        "digit_ratio": digits / n_chars,
        "n_numbers": float(min(len(_NUMBER.findall(text)), 10)),
        "math_ops": float(min(len(_MATH_OPS.findall(text)), 5)),
        "math_keywords": float(min(len(_MATH_KW.findall(text)), 5)),
        "code_signal": float(min(len(_CODE.findall(text)), 3)),
        "reasoning_keywords": float(min(len(_REASONING_KW.findall(text)), 3)),
        "multiple_choice": 1.0 if len(_MC.findall(text)) >= 2 else 0.0,
        "factoid_cue": 1.0 if _FACTOID.search(text) else 0.0,
        "is_short": 1.0 if n_words <= 14 else 0.0,
    }
