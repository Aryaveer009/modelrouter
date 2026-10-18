"""Wire settings into a ready-to-use Router."""

from __future__ import annotations

from .classifier import DifficultyClassifier, load_classifier
from .config import Settings
from .models import MockClient, ModelClient, OpenAICompatClient
from .router import Router


def build_clients(settings: Settings) -> tuple[ModelClient, ModelClient]:
    if settings.mock:
        return MockClient(settings.cheap, "cheap"), MockClient(settings.strong, "strong")
    settings.validate_live()
    return (
        OpenAICompatClient(settings.cheap, timeout_s=settings.timeout_s),
        OpenAICompatClient(settings.strong, timeout_s=settings.timeout_s),
    )


def build_router(settings: Settings, classifier: DifficultyClassifier | None = None) -> Router:
    cheap, strong = build_clients(settings)
    clf = classifier or load_classifier(settings.classifier, settings.classifier_path)
    return Router(
        clf,
        cheap,
        strong,
        confidence_threshold=settings.confidence_threshold,
        fallback_enabled=settings.fallback_enabled,
        min_avg_logprob=settings.min_avg_logprob,
    )
