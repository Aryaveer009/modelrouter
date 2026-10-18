"""Difficulty classifiers."""

from __future__ import annotations

import logging
from pathlib import Path

from .base import DifficultyClassifier, p_hard
from .embedding import EmbeddingClassifier, HashingEmbedder
from .heuristic import HeuristicClassifier
from .random_baseline import RandomClassifier

log = logging.getLogger(__name__)

__all__ = [
    "DifficultyClassifier",
    "EmbeddingClassifier",
    "HashingEmbedder",
    "HeuristicClassifier",
    "RandomClassifier",
    "load_classifier",
    "p_hard",
]


def load_classifier(kind: str, path: str | Path) -> DifficultyClassifier:
    """Build the configured classifier; fall back to the heuristic if the model is unavailable."""
    if kind == "heuristic":
        return HeuristicClassifier()
    try:
        return EmbeddingClassifier.load(path)
    except FileNotFoundError:
        log.warning("classifier file %s not found; falling back to heuristic baseline", path)
    except ImportError as exc:
        log.warning("embedding backend unavailable (%s); falling back to heuristic baseline", exc)
    return HeuristicClassifier()
