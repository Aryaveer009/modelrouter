import asyncio

import pytest

from conftest import CHEAP, STRONG, FixedClassifier, ScriptedClient
from modelrouter.router import Router
from modelrouter.tracking import RequestLog

MSGS = [{"role": "user", "content": "What is 2+2?"}]


def routed(p_hard, text="Final answer: 4"):
    router = Router(
        FixedClassifier(default=p_hard),
        ScriptedClient(CHEAP, "cheap", text),
        ScriptedClient(STRONG, "strong"),
    )
    return asyncio.run(router.handle(MSGS))


def test_record_and_summary(tmp_path):
    log = RequestLog(tmp_path / "t.db")
    a, b, c = routed(0.0), routed(0.0), routed(1.0)
    for r in (a, b, c):
        log.record(r)
    s = log.summary()
    assert s["requests"] == 3
    assert s["cost_usd"] == pytest.approx(a.cost_usd + b.cost_usd + c.cost_usd)
    assert s["pct_answered_by_cheap"] == pytest.approx(200 / 3)
    assert s["saved_usd"] == pytest.approx(a.saved_usd + b.saved_usd + c.saved_usd)
    assert 0 < s["saved_pct"] < 100


def test_recent_returns_newest_first_and_stores_fields(tmp_path):
    log = RequestLog(tmp_path / "t.db")
    log.record(routed(0.0))
    log.record(routed(0.0, text="I'm not sure"))
    rows = log.recent(10)
    assert rows[0]["escalated"] == 1 and rows[0]["escalation_reason"] == "hedging"
    assert rows[1]["final_tier"] == "cheap" and rows[1]["prompt_tokens"] > 0


def test_query_logging_can_be_disabled(tmp_path):
    log = RequestLog(tmp_path / "t.db", log_queries=False)
    log.record(routed(0.0))
    assert log.recent(1)[0]["query"] is None


def test_empty_summary(tmp_path):
    s = RequestLog(tmp_path / "t.db").summary()
    assert s["requests"] == 0 and s["saved_pct"] == 0.0
