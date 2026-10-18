import asyncio

import numpy as np
import pytest

from conftest import CHEAP, STRONG
from modelrouter.bench.analysis import (
    Outcome,
    analyse,
    oracle_outcome,
    saved_at_retention,
    split_half_check,
    wilson_interval,
)
from modelrouter.bench.runner import (
    BudgetExceeded,
    ResponseCache,
    confirm_spend,
    estimate_cost,
    run_model,
)
from modelrouter.classifier import HeuristicClassifier, RandomClassifier
from modelrouter.data.records import Record
from modelrouter.models import MockClient, OracleEntry


def make_records(n=60):
    recs = []
    for i in range(n):
        if i % 3 == 0:
            recs.append(
                Record(
                    f"m{i}",
                    "gsm8k",
                    "math",
                    f"A farmer has {i + 5} cows and sells {i} "
                    f"of them at $3 each. How many dollars does he get?",
                    str(i * 3),
                    prior_label=1,
                )
            )
        elif i % 3 == 1:
            recs.append(
                Record(
                    f"q{i}",
                    "triviaqa",
                    "qa",
                    f"Who wrote the famous book number {i}?",
                    f"Author{i}",
                    [f"Author{i}"],
                )
            )
        else:
            recs.append(
                Record(
                    f"c{i}",
                    "mmlu",
                    "mcq",
                    f"Pick the right one ({i}).\nA. a\nB. b\nC. c\nD. d",
                    "ABCD"[i % 4],
                    prior_label=0,
                )
            )
    return recs


def oracle(recs):
    d = {"math": 0.9, "mcq": 0.5, "qa": 0.1}
    return {r.query: OracleEntry(r.gold, d[r.task]) for r in recs}


def test_run_model_caches_and_counts(tmp_path):
    recs = make_records(9)
    client = MockClient(CHEAP, "cheap", oracle=oracle(recs))
    cache = ResponseCache(tmp_path / "c.jsonl")
    comps, stats = asyncio.run(run_model(recs, client, cache=cache))
    assert len(comps) == 9 and stats.calls == 9 and stats.spend_usd > 0
    # a fresh cache object reads the file back: nothing is called again
    comps2, stats2 = asyncio.run(run_model(recs, client, cache=ResponseCache(tmp_path / "c.jsonl")))
    assert stats2.calls == 0 and stats2.cached == 9
    assert comps2[recs[0].id].text == comps[recs[0].id].text


def test_budget_guard_stops_run():
    recs = make_records(30)
    client = MockClient(STRONG, "strong", oracle=oracle(recs))
    with pytest.raises(BudgetExceeded):
        asyncio.run(run_model(recs, client, concurrency=1, budget_usd=1e-9))


def test_estimate_and_confirmation(capsys):
    recs = make_records(30)
    est = estimate_cost(recs, STRONG)
    assert 0 < est.expected_usd < est.worst_case_usd and est.n_calls == 30
    assert confirm_spend([est], assume_yes=True) is True
    assert "TOTAL" in capsys.readouterr().err
    # non-interactive stdin without --yes must refuse to spend
    assert confirm_spend([est], assume_yes=False) is False


def test_wilson_interval_contains_point_estimate():
    lo, hi = wilson_interval(60, 100)
    assert lo < 0.6 < hi and 0 <= lo and hi <= 1
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_oracle_never_worse_than_either_model():
    cheap = Outcome(
        np.array([1, 0, 1, 0], bool),
        np.array([1.0, 1, 1, 1]),
        np.ones(4),
        np.ones(4, bool),
        np.zeros(4, bool),
    )
    strong = Outcome(
        np.array([1, 1, 0, 0], bool),
        np.array([5.0, 5, 5, 5]),
        np.ones(4),
        np.zeros(4, bool),
        np.zeros(4, bool),
    )
    o = oracle_outcome(cheap, strong)
    assert o.correct.tolist() == [True, True, True, False]
    assert o.cost.sum() == 1 + 5 + 1 + 5


def test_full_analysis_on_mock_data():
    recs = make_records(60)
    cheap = MockClient(CHEAP, "cheap", oracle=oracle(recs))
    strong = MockClient(STRONG, "strong", oracle=oracle(recs))
    cc, _ = asyncio.run(run_model(recs, cheap))
    sc, _ = asyncio.run(run_model(recs, strong))
    a = analyse(
        recs,
        cc,
        sc,
        CHEAP,
        STRONG,
        {"heuristic": HeuristicClassifier(), "random": RandomClassifier()},
        default_threshold=0.5,
        default_classifier="heuristic",
        thresholds=[0.0, 0.5, 1.0],
    )
    cheap_m, strong_m = a.baselines["always_cheap"], a.baselines["always_strong"]
    assert strong_m.cost_saved_pct == 0 and cheap_m.total_cost_usd < strong_m.total_cost_usd
    assert cheap_m.cost_saved_pct > 0 and strong_m.accuracy >= cheap_m.accuracy
    # threshold 1.0 sends (almost) everything to strong -> matches always-strong accuracy
    top = a.sweeps["heuristic"][-1]
    assert top.accuracy == pytest.approx(strong_m.accuracy, abs=0.05)
    # the oracle dominates every real router point on accuracy-per-cost
    assert a.baselines["oracle"].accuracy >= strong_m.accuracy - 1e-9
    assert len(a.per_query) == 60 and "cheap_correct" in a.per_query[0]
    assert a.classifier_eval["heuristic"]["auc_predicting_cheap_failure"] is not None
    assert "0.95" in a.saved_at["heuristic"]
    assert a.to_dict()["default_threshold"] == 0.5


def test_saved_at_retention_picks_cheapest_qualifying_point():
    from modelrouter.bench.analysis import StrategyMetrics

    def m(acc, saved):
        return StrategyMetrics(
            "x", "router", "c", 0.5, 100, acc, 0, 1, 1.0, 1.0, 1.0, 50, 0, saved, 0.0
        )

    dummy = Outcome(*(np.zeros(1) for _ in range(5)))
    pts = [(m(0.80, 70.0), dummy), (m(0.88, 40.0), dummy), (m(0.90, 10.0), dummy)]
    assert saved_at_retention(pts, 0.90, 1.0).cost_saved_pct == 10.0
    assert saved_at_retention(pts, 0.90, 0.95).cost_saved_pct == 40.0
    assert saved_at_retention(pts, 0.90, 0.5).cost_saved_pct == 70.0
    assert saved_at_retention(pts, 0.99, 1.0) is None


def test_split_half_check_returns_sane_numbers():
    rng = np.random.default_rng(1)
    n = 100
    strong = Outcome(
        rng.random(n) < 0.9, np.full(n, 5.0), np.ones(n), np.zeros(n, bool), np.zeros(n, bool)
    )
    cheapo = Outcome(
        strong.correct.copy(), np.full(n, 1.0), np.ones(n), np.ones(n, bool), np.zeros(n, bool)
    )
    from modelrouter.bench.analysis import StrategyMetrics

    dummy_m = StrategyMetrics(
        "x", "router", "c", 0.5, n, 0.9, 0, 1, 1.0, 1.0, 1.0, 50, 0, 80.0, 0.0
    )
    out = split_half_check([(dummy_m, cheapo), (dummy_m, strong)], strong, 0.95, trials=50)
    assert out["mean_cost_saved_pct"] == pytest.approx(80.0)
    assert out["mean_accuracy_retained"] == pytest.approx(1.0)


def test_report_files_written(tmp_path):
    pytest.importorskip("matplotlib")
    from modelrouter.bench.report import write_all

    recs = make_records(30)
    cc, _ = asyncio.run(run_model(recs, MockClient(CHEAP, "cheap", oracle=oracle(recs))))
    sc, _ = asyncio.run(run_model(recs, MockClient(STRONG, "strong", oracle=oracle(recs))))
    a = analyse(
        recs,
        cc,
        sc,
        CHEAP,
        STRONG,
        {"heuristic": HeuristicClassifier()},
        default_classifier="heuristic",
        thresholds=[0.0, 0.5, 1.0],
    )
    paths = write_all(
        a,
        tmp_path,
        {"cheap_model": "c", "strong_model": "s", "n_queries": 30},
        default_classifier="heuristic",
        mock=True,
    )
    assert paths["pareto"].stat().st_size > 5000
    md = paths["md"].read_text()
    assert "MOCK RUN" in md and "Always strong" in md
