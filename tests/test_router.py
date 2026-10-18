import asyncio

import pytest

from conftest import CHEAP, STRONG, FixedClassifier, ScriptedClient
from modelrouter.costing import compute_cost
from modelrouter.models import ModelError
from modelrouter.router import Router
from modelrouter.types import Completion, Usage
from modelrouter.uncertainty import check_answer

Q = "What is 2+2?"
MSGS = [{"role": "user", "content": Q}]


def run(coro):
    return asyncio.run(coro)


def make(
    p_hard=0.1,
    threshold=0.5,
    cheap_text="Final answer: 4",
    cheap_finish="stop",
    cheap_fail=False,
    strong_fail=False,
    **kw,
):
    cheap = ScriptedClient(CHEAP, "cheap", cheap_text, cheap_finish, cheap_fail)
    strong = ScriptedClient(STRONG, "strong", "Final answer: 4", fail=strong_fail)
    router = Router(
        FixedClassifier(default=p_hard), cheap, strong, confidence_threshold=threshold, **kw
    )
    return router, cheap, strong


def test_easy_query_goes_to_cheap_only():
    router, cheap, strong = make(p_hard=0.1)
    res = run(router.handle(MSGS))
    assert res.final_tier == "cheap" and not res.escalated
    assert (cheap.calls, strong.calls) == (1, 0)


def test_hard_query_goes_straight_to_strong():
    router, cheap, strong = make(p_hard=0.9)
    res = run(router.handle(MSGS))
    assert res.final_tier == "strong" and res.initial_tier == "strong"
    assert (cheap.calls, strong.calls) == (0, 1)


@pytest.mark.parametrize(
    "p_hard,threshold,expected",
    [
        (0.5, 0.5, "cheap"),  # P(easy)=0.5 >= 0.5
        (0.51, 0.5, "strong"),
        (0.3, 0.8, "strong"),  # P(easy)=0.7 < 0.8: conservative threshold
        (0.3, 0.6, "cheap"),
        (0.99, 0.0, "cheap"),  # threshold 0 never blocks the cheap model
        (0.0, 1.0, "cheap"),  # P(easy)=1.0 >= 1.0
        (0.01, 1.0, "strong"),
    ],
)
def test_threshold_policy(p_hard, threshold, expected):
    router, _, _ = make(p_hard=p_hard, threshold=threshold)
    assert router.decide(p_hard) == expected


@pytest.mark.parametrize(
    "text,finish,reason",
    [
        ("I'm not sure about that.", "stop", "hedging"),
        ("", "stop", "empty"),
        ("Final answer: 4", "length", "truncated"),
    ],
)
def test_bad_cheap_answer_escalates(text, finish, reason):
    router, cheap, strong = make(cheap_text=text, cheap_finish=finish)
    res = run(router.handle(MSGS))
    assert res.escalated and res.escalation_reason == reason
    assert res.final_tier == "strong" and (cheap.calls, strong.calls) == (1, 1)
    assert res.cost_usd == pytest.approx(sum(a.cost_usd for a in res.attempts))


def test_cheap_error_escalates():
    router, _, strong = make(cheap_fail=True)
    res = run(router.handle(MSGS))
    assert res.escalated and res.escalation_reason == "error" and strong.calls == 1


def test_fallback_can_be_disabled():
    router, _, strong = make(cheap_text="I don't know.", fallback_enabled=False)
    res = run(router.handle(MSGS))
    assert not res.escalated and res.final_tier == "cheap" and strong.calls == 0


def test_forced_cheap_is_not_escalated():
    router, _, strong = make(cheap_text="I don't know.")
    res = run(router.handle(MSGS, force="cheap"))
    assert res.forced and not res.escalated and strong.calls == 0


def test_forced_strong_skips_classifier_choice():
    router, cheap, strong = make(p_hard=0.0)
    res = run(router.handle(MSGS, force="strong"))
    assert res.final_tier == "strong" and cheap.calls == 0


def test_all_models_failing_raises():
    router, _, _ = make(cheap_fail=True, strong_fail=True)
    with pytest.raises(ModelError):
        run(router.handle(MSGS))


def test_cost_accounting_for_cheap_route_and_baseline():
    router, _, _ = make(p_hard=0.0)
    res = run(router.handle(MSGS))
    u = res.final.usage
    assert res.cost_usd == pytest.approx(compute_cost(u, CHEAP))
    assert res.baseline_strong_cost_usd == pytest.approx(compute_cost(u, STRONG))
    assert res.saved_usd > 0


def test_escalation_costs_more_than_going_straight_to_strong():
    router, _, _ = make(cheap_text="I'm not sure.")
    escalated = run(router.handle(MSGS))
    assert escalated.saved_usd < 0  # paid for the failed cheap attempt as well


def test_classifier_uses_last_user_message():
    clf = FixedClassifier({"hard one": 0.99, "first": 0.0})
    router = Router(clf, ScriptedClient(CHEAP, "cheap"), ScriptedClient(STRONG, "strong"))
    res = run(
        router.handle(
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "hard one"},
            ]
        )
    )
    assert res.query == "hard one" and res.final_tier == "strong"


# ---- answer checks


def comp(text, finish="stop", logprob=None, error=None):
    return Completion(text, Usage(1, 1), 0.1, "m", finish, avg_logprob=logprob, error=error)


def test_check_accepts_normal_answer():
    assert check_answer("What is 2+2?", comp("It is 4.")).ok


def test_check_multiple_choice_needs_letter():
    q = "Pick.\nA. red\nB. blue\nC. green\nD. pink"
    assert not check_answer(q, comp("it is probably the colour of the sea")).ok
    assert check_answer(q, comp("Final answer: B")).ok


def test_check_math_needs_a_number():
    assert (
        check_answer("How many apples are left if I have 5 and eat 2?", comp("Three.")).reason
        == "no_number"
    )


def test_check_low_logprob():
    r = check_answer("hi", comp("hello", logprob=-3.0), min_avg_logprob=-1.0)
    assert not r.ok and r.reason == "low_confidence"
    assert check_answer("hi", comp("hello", logprob=-0.2), min_avg_logprob=-1.0).ok
    assert check_answer("hi", comp("hello", logprob=-3.0)).ok  # disabled by default


def test_hedge_regex_has_no_false_positive_on_plain_cannot():
    assert check_answer("q", comp("You cannot divide by zero; the answer is undefined.")).ok
