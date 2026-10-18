"""Difficulty classifier interface."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np


class DifficultyClassifier(Protocol):
    """Estimates P(query is hard), i.e. P(the cheap model will get it wrong)."""

    name: str

    def predict_proba_hard(self, texts: Sequence[str]) -> np.ndarray: ...


def p_hard(classifier: DifficultyClassifier, text: str) -> float:
    return float(classifier.predict_proba_hard([text])[0])
