"""Logistic-regression difficulty classifier on sentence embeddings.

Training uses scikit-learn; inference is a plain dot product, so the fitted model is stored
as human-readable JSON (coefficients + metadata) rather than a pickle.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


class FastEmbedEmbedder:
    """ONNX sentence embeddings via `fastembed` (no PyTorch needed)."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.name = f"fastembed:{model_name}"
        self._model_name = model_name
        self._model: Any = None

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:  # pragma: no cover - depends on install
                raise ImportError(
                    "fastembed is required for the embedding classifier: "
                    "pip install 'modelrouter[embeddings]'"
                ) from exc
            self._model = TextEmbedding(model_name=self._model_name)
        vectors = np.array(list(self._model.embed(list(texts))), dtype=np.float32)
        return _l2_normalize(vectors)


class HashingEmbedder:
    """Dependency-free bag-of-ngrams embedding. Used in tests and as an offline fallback."""

    def __init__(self, n_features: int = 512) -> None:
        from sklearn.feature_extraction.text import HashingVectorizer

        self.name = f"hashing:{n_features}"
        self._vec = HashingVectorizer(
            n_features=n_features, alternate_sign=False, ngram_range=(1, 2), norm=None
        )

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        dense = self._vec.transform(list(texts)).toarray().astype(np.float32)
        return _l2_normalize(np.log1p(dense))


def embedder_from_name(name: str) -> Embedder:
    kind, _, arg = name.partition(":")
    if kind == "fastembed":
        return FastEmbedEmbedder(arg or DEFAULT_EMBEDDING_MODEL)
    if kind == "hashing":
        return HashingEmbedder(int(arg or 512))
    raise ValueError(f"unknown embedder {name!r}")


@dataclass
class TrainReport:
    n_train: int
    n_hard: int
    cv_accuracy: float
    cv_auc: float
    cv_folds: int
    label_source: str
    extra: dict[str, Any] = field(default_factory=dict)


class EmbeddingClassifier:
    name = "embedding"

    def __init__(
        self,
        embedder: Embedder,
        coef: np.ndarray,
        intercept: float,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.embedder = embedder
        self.coef = np.asarray(coef, dtype=np.float64)
        self.intercept = float(intercept)
        self.meta = meta or {}

    def predict_proba_hard(self, texts: Sequence[str]) -> np.ndarray:
        x = self.embedder.embed(texts).astype(np.float64)
        z = x @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-z))

    # -- persistence -----------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "logistic_regression_on_embeddings",
            "embedder": self.embedder.name,
            "coef": self.coef.tolist(),
            "intercept": self.intercept,
            "meta": self.meta,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, embedder: Embedder | None = None) -> EmbeddingClassifier:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        emb = embedder or embedder_from_name(payload["embedder"])
        return cls(emb, np.array(payload["coef"]), payload["intercept"], payload.get("meta"))

    # -- training ---------------------------------------------------------
    @classmethod
    def fit(
        cls,
        texts: Sequence[str],
        labels: Sequence[int],
        embedder: Embedder,
        *,
        C: float = 1.0,
        cv_folds: int = 5,
        label_source: str = "unspecified",
        seed: int = 0,
    ) -> tuple[EmbeddingClassifier, TrainReport]:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.model_selection import StratifiedKFold, cross_val_predict

        x = embedder.embed(texts)
        y = np.asarray(labels, dtype=int)
        if len(set(y.tolist())) < 2:
            raise ValueError("training labels contain a single class; need both easy and hard")

        def make() -> LogisticRegression:
            return LogisticRegression(C=C, max_iter=2000, class_weight="balanced")

        folds = max(2, min(cv_folds, int(np.bincount(y).min())))
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        proba = cross_val_predict(make(), x, y, cv=cv, method="predict_proba")[:, 1]
        report = TrainReport(
            n_train=len(y),
            n_hard=int(y.sum()),
            cv_accuracy=float(accuracy_score(y, proba >= 0.5)),
            cv_auc=float(roc_auc_score(y, proba)),
            cv_folds=folds,
            label_source=label_source,
        )
        model = make().fit(x, y)
        meta = {
            "n_train": report.n_train,
            "n_hard": report.n_hard,
            "cv_accuracy": report.cv_accuracy,
            "cv_auc": report.cv_auc,
            "cv_folds": folds,
            "label_source": label_source,
            "C": C,
        }
        clf = cls(embedder, model.coef_[0], float(model.intercept_[0]), meta)
        return clf, report
