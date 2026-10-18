"""Configuration, read entirely from environment variables (and an optional .env file)."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class ModelSpec:
    """One model endpoint plus its prices (USD per 1M tokens)."""

    name: str
    base_url: str
    api_key: str
    price_in_per_m: float
    price_out_per_m: float
    # Extra JSON fields merged into every request body (e.g. {"reasoning_effort": "low"}).
    extra_body: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Settings:
    cheap: ModelSpec
    strong: ModelSpec
    mock: bool = False
    confidence_threshold: float = 0.5
    classifier: str = "embedding"
    classifier_path: Path = Path("models/classifier.json")
    fallback_enabled: bool = True
    min_avg_logprob: float | None = None
    db_path: Path = Path("modelrouter.db")
    log_queries: bool = True
    timeout_s: float = 60.0

    def validate_live(self) -> None:
        """Check that real (non-mock) endpoints are fully configured."""
        missing: list[str] = []
        for prefix, spec in (("CHEAP", self.cheap), ("STRONG", self.strong)):
            if not spec.name:
                missing.append(f"{prefix}_MODEL")
            if not spec.base_url:
                missing.append(f"{prefix}_BASE_URL")
            if not spec.api_key:
                missing.append(f"{prefix}_API_KEY")
        if missing:
            raise ConfigError(
                "Missing environment variables: "
                + ", ".join(missing)
                + ". Copy .env.example to .env, or set MODELROUTER_MOCK=1 for offline mock mode."
            )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, *, use_dotenv: bool = True) -> Settings:
        if env is None:
            if use_dotenv:
                load_dotenv()
            env = os.environ

        def get(key: str, default: str = "") -> str:
            return env.get(key, default).strip()

        def num(key: str, default: float) -> float:
            raw = get(key)
            if not raw:
                return default
            try:
                return float(raw)
            except ValueError as exc:
                raise ConfigError(f"{key} must be a number, got {raw!r}") from exc

        def flag(key: str, default: bool) -> bool:
            raw = get(key).lower()
            if not raw:
                return default
            return raw in {"1", "true", "yes", "on"}

        def extra(prefix: str) -> dict[str, object]:
            raw = get(f"{prefix}_EXTRA_BODY")
            if not raw:
                return {}
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ConfigError(f"{prefix}_EXTRA_BODY must be valid JSON") from exc
            if not isinstance(parsed, dict):
                raise ConfigError(f"{prefix}_EXTRA_BODY must be a JSON object")
            return parsed

        def spec(prefix: str, default_in: float, default_out: float) -> ModelSpec:
            return ModelSpec(
                name=get(f"{prefix}_MODEL"),
                base_url=get(f"{prefix}_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
                api_key=get(f"{prefix}_API_KEY"),
                price_in_per_m=num(f"{prefix}_PRICE_IN_PER_M", default_in),
                price_out_per_m=num(f"{prefix}_PRICE_OUT_PER_M", default_out),
                extra_body=extra(prefix),
            )

        threshold = num("ROUTER_CONFIDENCE_THRESHOLD", 0.5)
        if not 0.0 <= threshold <= 1.0:
            raise ConfigError("ROUTER_CONFIDENCE_THRESHOLD must be in [0, 1]")
        classifier = get("ROUTER_CLASSIFIER", "embedding").lower()
        if classifier not in {"embedding", "heuristic"}:
            raise ConfigError("ROUTER_CLASSIFIER must be 'embedding' or 'heuristic'")
        logprob_raw = get("ROUTER_MIN_AVG_LOGPROB")
        mock = flag("MODELROUTER_MOCK", False)
        cheap = spec("CHEAP", 0.15, 0.60)
        strong = spec("STRONG", 2.50, 10.00)
        if mock:
            # Mock mode needs no credentials; give the fake models readable names.
            cheap = ModelSpec(
                cheap.name or "mock-cheap",
                cheap.base_url,
                "mock",
                cheap.price_in_per_m,
                cheap.price_out_per_m,
            )
            strong = ModelSpec(
                strong.name or "mock-strong",
                strong.base_url,
                "mock",
                strong.price_in_per_m,
                strong.price_out_per_m,
            )
        return cls(
            cheap=cheap,
            strong=strong,
            mock=mock,
            confidence_threshold=threshold,
            classifier=classifier,
            classifier_path=Path(get("ROUTER_CLASSIFIER_PATH", "models/classifier.json")),
            fallback_enabled=flag("ROUTER_FALLBACK", True),
            min_avg_logprob=float(logprob_raw) if logprob_raw else None,
            db_path=Path(get("MODELROUTER_DB", "modelrouter.db")),
            log_queries=flag("MODELROUTER_LOG_QUERIES", True),
            timeout_s=num("MODELROUTER_TIMEOUT_S", 60.0),
        )
