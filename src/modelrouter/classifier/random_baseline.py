"""Random 'classifier' used as a sanity-check baseline in the benchmark."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np


class RandomClassifier:
    """Deterministic pseudo-random P(hard) per text. A router that does better than this is
    actually using information about the query."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def predict_proba_hard(self, texts: Sequence[str]) -> np.ndarray:
        out = []
        for t in texts:
            h = hashlib.sha256(f"{self.seed}|{t}".encode()).digest()
            out.append(int.from_bytes(h[:8], "big") / 2**64)
        return np.array(out, dtype=float)
