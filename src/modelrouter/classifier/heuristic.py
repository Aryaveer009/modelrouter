"""Rule-based baseline difficulty classifier (no training, no model download).

The weights are hand-set priors, not fitted values: longer queries, numbers, arithmetic,
code and explicit reasoning cues push the score towards "hard"; short factoid-style
questions push it towards "easy". It exists as the baseline the learned classifier must beat.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .features import extract_features

WEIGHTS: dict[str, float] = {
    "log_words": 0.45,
    "digit_ratio": 4.0,
    "n_numbers": 0.18,
    "math_ops": 0.35,
    "math_keywords": 0.45,
    "code_signal": 0.8,
    "reasoning_keywords": 0.6,
    "multiple_choice": 0.5,
    "factoid_cue": -1.0,
    "is_short": -0.7,
}
BIAS = -2.0


class HeuristicClassifier:
    name = "heuristic"

    def __init__(self, weights: dict[str, float] | None = None, bias: float = BIAS) -> None:
        self.weights = weights or WEIGHTS
        self.bias = bias

    def score(self, text: str) -> float:
        feats = extract_features(text)
        return self.bias + sum(self.weights.get(k, 0.0) * v for k, v in feats.items())

    def predict_proba_hard(self, texts: Sequence[str]) -> np.ndarray:
        return np.array([1.0 / (1.0 + math.exp(-self.score(t))) for t in texts], dtype=float)
