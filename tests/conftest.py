from __future__ import annotations

import pytest

from modelrouter.config import ModelSpec
from modelrouter.models import MockClient

CHEAP = ModelSpec("cheap-m", "http://x", "k", price_in_per_m=1.0, price_out_per_m=2.0)
STRONG = ModelSpec("strong-m", "http://x", "k", price_in_per_m=10.0, price_out_per_m=20.0)


@pytest.fixture
def specs() -> tuple[ModelSpec, ModelSpec]:
    return CHEAP, STRONG


class FixedClassifier:
    """Returns a preset P(hard) for each text (default 0.5)."""

    name = "fixed"

    def __init__(self, table: dict[str, float] | None = None, default: float = 0.5) -> None:
        self.table = table or {}
        self.default = default

    def predict_proba_hard(self, texts):
        import numpy as np

        return np.array([self.table.get(t, self.default) for t in texts])


class ScriptedClient(MockClient):
    """Mock client that returns scripted text/finish_reason and counts calls."""

    def __init__(self, spec, tier, text="Final answer: 4", finish_reason="stop", fail=False):
        super().__init__(spec, tier)
        self.text, self.finish_reason, self.fail = text, finish_reason, fail
        self.calls = 0

    async def complete(self, messages, **kw):
        from modelrouter.models import ModelError

        self.calls += 1
        if self.fail:
            raise ModelError("boom")
        comp = await super().complete(messages, **kw)
        comp.text, comp.finish_reason = self.text, self.finish_reason
        return comp
