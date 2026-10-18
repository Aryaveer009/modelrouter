"""Evaluate always-cheap, always-strong and router strategies from recorded model responses.

All model calls happen once (see runner.py). The router is then *replayed* over the recorded
responses at many thresholds using the real `Router` class, so no extra money is spent and
every strategy sees identical model outputs.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ..classifier import DifficultyClassifier
from ..config import ModelSpec
from ..costing import compute_cost
from ..data.records import Record
from ..router import Router
from ..types import Completion
from .runner import ReplayClient, record_key
from .tasks import MAX_TOKENS, build_messages, score

DEFAULT_THRESHOLDS = [round(t, 2) for t in np.arange(0.0, 1.0001, 0.05)]
RETENTIONS = (1.0, 0.99, 0.98, 0.95, 0.90)


@dataclass
class Outcome:
    """Per-query outcome of one strategy."""

    correct: np.ndarray
    cost: np.ndarray
    latency: np.ndarray
    final_cheap: np.ndarray
    escalated: np.ndarray


@dataclass
class StrategyMetrics:
    name: str
    kind: str  # always_cheap | always_strong | router | oracle
    classifier: str | None
    threshold: float | None
    n: int
    accuracy: float
    ci_low: float
    ci_high: float
    total_cost_usd: float
    cost_per_1k_usd: float
    avg_latency_s: float
    pct_answered_by_cheap: float
    escalation_rate_pct: float
    cost_saved_pct: float  # vs always-strong
    accuracy_delta_pp: float  # vs always-strong
    per_source_accuracy: dict[str, float] = field(default_factory=dict)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


class PrecomputedClassifier:
    """Looks up P(hard) computed once per query, so threshold sweeps don't re-embed."""

    def __init__(self, name: str, table: dict[str, float]) -> None:
        self.name = name
        self._table = table

    def predict_proba_hard(self, texts: Sequence[str]) -> np.ndarray:
        return np.array([self._table[t] for t in texts], dtype=float)


def precompute(
    classifier: DifficultyClassifier, records: list[Record]
) -> tuple[dict[str, float], dict[str, float]]:
    """P(hard) and *measured* single-query classification time for every record."""
    classifier.predict_proba_hard([records[0].query])  # warm-up (model load, caches)
    probs: dict[str, float] = {}
    times: dict[str, float] = {}
    for r in records:
        t0 = time.perf_counter()
        probs[r.query] = float(classifier.predict_proba_hard([r.query])[0])
        times[r.query] = time.perf_counter() - t0
    return probs, times


def _metrics(
    name: str,
    kind: str,
    classifier: str | None,
    threshold: float | None,
    records: list[Record],
    o: Outcome,
    strong: Outcome | None,
) -> StrategyMetrics:
    n = len(records)
    k = int(o.correct.sum())
    lo, hi = wilson_interval(k, n)
    total = float(o.cost.sum())
    strong_total = float(strong.cost.sum()) if strong else total
    strong_acc = float(strong.correct.mean()) if strong else float(o.correct.mean())
    per_source = {}
    for src in sorted({r.source for r in records}):
        idx = np.array([r.source == src for r in records])
        per_source[src] = float(o.correct[idx].mean())
    return StrategyMetrics(
        name=name,
        kind=kind,
        classifier=classifier,
        threshold=threshold,
        n=n,
        accuracy=k / n,
        ci_low=lo,
        ci_high=hi,
        total_cost_usd=total,
        cost_per_1k_usd=total / n * 1000,
        avg_latency_s=float(o.latency.mean()),
        pct_answered_by_cheap=float(o.final_cheap.mean() * 100),
        escalation_rate_pct=float(o.escalated.mean() * 100),
        cost_saved_pct=(strong_total - total) / strong_total * 100 if strong_total > 0 else 0.0,
        accuracy_delta_pp=(k / n - strong_acc) * 100,
        per_source_accuracy=per_source,
    )


def direct_outcome(
    records: list[Record], comps: dict[str, Completion], spec: ModelSpec, *, is_cheap: bool
) -> Outcome:
    n = len(records)
    correct = np.zeros(n, dtype=bool)
    cost = np.zeros(n)
    latency = np.zeros(n)
    for i, r in enumerate(records):
        c = comps[r.id]
        correct[i] = (not c.error) and score(r, c.text)
        cost[i] = compute_cost(c.usage, spec)
        latency[i] = c.latency_s
    return Outcome(correct, cost, latency, np.full(n, is_cheap), np.zeros(n, dtype=bool))


def oracle_outcome(cheap: Outcome, strong: Outcome) -> Outcome:
    """Perfect-knowledge router: cheap when cheap is right, otherwise straight to strong."""
    use_cheap = cheap.correct
    return Outcome(
        correct=np.where(use_cheap, cheap.correct, strong.correct),
        cost=np.where(use_cheap, cheap.cost, strong.cost),
        latency=np.where(use_cheap, cheap.latency, strong.latency),
        final_cheap=use_cheap.copy(),
        escalated=np.zeros(len(use_cheap), dtype=bool),
    )


async def router_outcome(
    records: list[Record],
    cheap_comps: dict[str, Completion],
    strong_comps: dict[str, Completion],
    cheap_spec: ModelSpec,
    strong_spec: ModelSpec,
    classifier: PrecomputedClassifier,
    classify_times: dict[str, float],
    threshold: float,
    *,
    fallback: bool = True,
    min_avg_logprob: float | None = None,
) -> tuple[Outcome, list[dict[str, Any]]]:
    cheap_tab = {record_key(r): cheap_comps[r.id] for r in records}
    strong_tab = {record_key(r): strong_comps[r.id] for r in records}
    router = Router(
        classifier,
        ReplayClient(cheap_spec, cheap_tab),
        ReplayClient(strong_spec, strong_tab),
        confidence_threshold=threshold,
        fallback_enabled=fallback,
        min_avg_logprob=min_avg_logprob,
    )
    n = len(records)
    correct = np.zeros(n, dtype=bool)
    cost = np.zeros(n)
    latency = np.zeros(n)
    final_cheap = np.zeros(n, dtype=bool)
    escalated = np.zeros(n, dtype=bool)
    rows: list[dict[str, Any]] = []
    for i, r in enumerate(records):
        res = await router.handle(build_messages(r), temperature=0.0, max_tokens=MAX_TOKENS[r.task])
        correct[i] = score(r, res.final.text)
        cost[i] = res.cost_usd
        # Replace the (instant) lookup time with the measured real classification time.
        latency[i] = res.latency_s - res.classify_s + classify_times[r.query]
        final_cheap[i] = res.final_tier == "cheap"
        escalated[i] = res.escalated
        rows.append(
            {
                "id": r.id,
                "source": r.source,
                "p_hard": round(res.p_hard, 4),
                "initial": res.initial_tier,
                "final": res.final_tier,
                "escalated": res.escalated,
                "reason": res.escalation_reason,
                "correct": bool(correct[i]),
                "cost_usd": cost[i],
            }
        )
    return Outcome(correct, cost, latency, final_cheap, escalated), rows


def saved_at_retention(
    points: list[tuple[StrategyMetrics, Outcome]], strong_acc: float, retention: float
) -> StrategyMetrics | None:
    """Best cost saving among operating points keeping >= retention * strong accuracy."""
    ok = [m for m, _ in points if m.accuracy >= retention * strong_acc - 1e-12]
    return max(ok, key=lambda m: m.cost_saved_pct) if ok else None


def split_half_check(
    sweep: list[tuple[StrategyMetrics, Outcome]],
    strong: Outcome,
    retention: float,
    *,
    trials: int = 300,
    seed: int = 0,
) -> dict[str, float]:
    """Honest estimate of what threshold tuning delivers on *unseen* queries.

    Repeatedly split the queries in half, pick the best threshold on one half (max saving
    subject to the accuracy constraint), and measure saving/accuracy on the other half.
    """
    rng = np.random.default_rng(seed)
    n = len(strong.correct)
    saved, kept = [], []
    for _ in range(trials):
        perm = rng.permutation(n)
        a, b = perm[: n // 2], perm[n // 2 :]
        s_acc_a = strong.correct[a].mean()
        s_cost_a = strong.cost[a].sum()
        best_idx, best_save = None, -1e9
        for j, (_, o) in enumerate(sweep):
            if o.correct[a].mean() >= retention * s_acc_a - 1e-12:
                save = 1 - o.cost[a].sum() / s_cost_a
                if save > best_save:
                    best_idx, best_save = j, save
        if best_idx is None:
            saved.append(0.0)
            kept.append(1.0)
            continue
        o = sweep[best_idx][1]
        saved.append(100 * (1 - o.cost[b].sum() / strong.cost[b].sum()))
        s_acc_b = strong.correct[b].mean()
        kept.append(o.correct[b].mean() / s_acc_b if s_acc_b > 0 else 1.0)
    return {
        "retention_target": retention,
        "mean_cost_saved_pct": float(np.mean(saved)),
        "p05_cost_saved_pct": float(np.percentile(saved, 5)),
        "p95_cost_saved_pct": float(np.percentile(saved, 95)),
        "mean_accuracy_retained": float(np.mean(kept)),
        "p05_accuracy_retained": float(np.percentile(kept, 5)),
        "trials": trials,
    }


@dataclass
class Analysis:
    baselines: dict[str, StrategyMetrics]
    sweeps: dict[str, list[StrategyMetrics]]
    default_threshold: float
    saved_at: dict[str, dict[str, dict[str, Any] | None]]
    robustness: dict[str, dict[str, float]]
    classifier_eval: dict[str, dict[str, float | None]]
    per_query: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "baselines": {k: asdict(v) for k, v in self.baselines.items()},
            "sweeps": {k: [asdict(m) for m in v] for k, v in self.sweeps.items()},
            "default_threshold": self.default_threshold,
            "saved_at_accuracy_retention": self.saved_at,
            "split_half_robustness": self.robustness,
            "classifier_eval": self.classifier_eval,
        }


def analyse(
    records: list[Record],
    cheap_comps: dict[str, Completion],
    strong_comps: dict[str, Completion],
    cheap_spec: ModelSpec,
    strong_spec: ModelSpec,
    classifiers: dict[str, DifficultyClassifier],
    *,
    default_threshold: float = 0.5,
    default_classifier: str = "embedding",
    thresholds: Sequence[float] = tuple(DEFAULT_THRESHOLDS),
    fallback: bool = True,
    min_avg_logprob: float | None = None,
) -> Analysis:
    cheap = direct_outcome(records, cheap_comps, cheap_spec, is_cheap=True)
    strong = direct_outcome(records, strong_comps, strong_spec, is_cheap=False)
    oracle = oracle_outcome(cheap, strong)
    baselines = {
        "always_cheap": _metrics(
            "Always cheap", "always_cheap", None, None, records, cheap, strong
        ),
        "always_strong": _metrics(
            "Always strong", "always_strong", None, None, records, strong, strong
        ),
        "oracle": _metrics(
            "Oracle router (upper bound)", "oracle", None, None, records, oracle, strong
        ),
    }
    strong_acc = baselines["always_strong"].accuracy
    cheap_failed = ~cheap.correct

    sweeps: dict[str, list[StrategyMetrics]] = {}
    sweep_outcomes: dict[str, list[tuple[StrategyMetrics, Outcome]]] = {}
    per_query: list[dict[str, Any]] = []
    clf_eval: dict[str, dict[str, float | None]] = {}

    from sklearn.metrics import roc_auc_score

    for cname, clf in classifiers.items():
        probs, times = precompute(clf, records)
        pre = PrecomputedClassifier(cname, probs)
        p_arr = np.array([probs[r.query] for r in records])
        auc = (
            float(roc_auc_score(cheap_failed, p_arr))
            if 0 < cheap_failed.sum() < len(records)
            else None
        )
        clf_eval[cname] = {
            "auc_predicting_cheap_failure": auc,
            "avg_classify_ms": float(np.mean(list(times.values())) * 1000),
        }
        pts: list[tuple[StrategyMetrics, Outcome]] = []
        for th in thresholds:
            o, rows = asyncio.run(
                router_outcome(
                    records,
                    cheap_comps,
                    strong_comps,
                    cheap_spec,
                    strong_spec,
                    pre,
                    times,
                    th,
                    fallback=fallback,
                    min_avg_logprob=min_avg_logprob,
                )
            )
            m = _metrics(f"Router ({cname}, t={th:.2f})", "router", cname, th, records, o, strong)
            pts.append((m, o))
            if cname == default_classifier and abs(th - default_threshold) < 1e-9:
                per_query = rows
        sweep_outcomes[cname] = pts
        sweeps[cname] = [m for m, _ in pts]

    saved_at: dict[str, dict[str, dict[str, Any] | None]] = {}
    robustness: dict[str, dict[str, float]] = {}
    for cname, pts in sweep_outcomes.items():
        saved_at[cname] = {}
        for ret in RETENTIONS:
            best = saved_at_retention(pts, strong_acc, ret)
            saved_at[cname][f"{ret:.2f}"] = asdict(best) if best else None
        robustness[cname] = split_half_check(pts, strong, 0.95)

    # attach per-query ground truth + both model outcomes to the default run's rows
    by_id = {r.id: i for i, r in enumerate(records)}
    for row in per_query:
        i = by_id[row["id"]]
        row["cheap_correct"] = bool(cheap.correct[i])
        row["strong_correct"] = bool(strong.correct[i])
    return Analysis(baselines, sweeps, default_threshold, saved_at, robustness, clf_eval, per_query)
