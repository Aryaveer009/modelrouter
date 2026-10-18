"""Command line interface: serve, build-data, label, train, bench, demo."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

from . import __author__, __version__
from .bench.analysis import analyse
from .bench.report import write_all
from .bench.runner import (
    BudgetExceeded,
    ResponseCache,
    confirm_spend,
    estimate_cost,
    record_key,
    run_model,
)
from .bench.tasks import score
from .classifier import EmbeddingClassifier, HeuristicClassifier, RandomClassifier
from .classifier.embedding import FastEmbedEmbedder
from .config import ConfigError, Settings
from .data.records import Record, load_records, save_records
from .models import MockClient, ModelClient, OpenAICompatClient, OracleEntry

DATA_DIR = Path("data")


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()


def _stratified(records: list[Record], n: int, seed: int = 0) -> list[Record]:
    """Take n records keeping the source mix, deterministically."""
    if n >= len(records):
        return records
    by_src: dict[str, list[Record]] = {}
    for r in records:
        by_src.setdefault(r.source, []).append(r)
    rng = random.Random(seed)
    picked: list[Record] = []
    sources = sorted(by_src)
    for i, s in enumerate(sources):
        share = n // len(sources) + (1 if i < n % len(sources) else 0)
        items = by_src[s][:]
        rng.shuffle(items)
        picked += items[:share]
    return sorted(picked, key=lambda r: r.id)


def _mock_oracle(records: list[Record]) -> dict[str, OracleEntry]:
    base = {"math": 0.7, "mcq": 0.5, "qa": 0.25}
    out = {}
    for r in records:
        d = base[r.task] + (0.2 if r.prior_label and r.task == "mcq" else 0.0)
        out[r.query] = OracleEntry(gold=r.gold, difficulty=min(d, 1.0))
    return out


def _clients(settings: Settings, records: list[Record]) -> tuple[ModelClient, ModelClient]:
    if settings.mock:
        oracle = _mock_oracle(records)
        return (
            MockClient(settings.cheap, "cheap", oracle=oracle),
            MockClient(settings.strong, "strong", oracle=oracle),
        )
    settings.validate_live()
    return (
        OpenAICompatClient(settings.cheap, timeout_s=settings.timeout_s),
        OpenAICompatClient(settings.strong, timeout_s=settings.timeout_s),
    )


# ---------------------------------------------------------------- commands


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("modelrouter.server:app_factory", factory=True, host=args.host, port=args.port)
    return 0


def cmd_build_data(args: argparse.Namespace) -> int:
    from .data.build import build_datasets

    train, bench = build_datasets(
        bench_per_source=args.bench_per_source, train_per_source=args.train_per_source, log=_log
    )
    save_records(train, DATA_DIR / "train.jsonl")
    save_records(bench, DATA_DIR / "bench.jsonl")
    _log(f"wrote {len(train)} training and {len(bench)} benchmark records to {DATA_DIR}/")
    for name, recs in (("train", train), ("bench", bench)):
        _log(f"  {name}: {dict(Counter(r.source for r in recs))}")
    return 0


def cmd_label(args: argparse.Namespace) -> int:
    """Empirical labelling: a training query is 'hard' iff the cheap model answers it wrongly."""
    settings = Settings.from_env()
    path = DATA_DIR / "train.jsonl"
    records = load_records(path)
    cheap, _ = _clients(settings, records)
    cache = ResponseCache(Path(args.cache_dir) / f"{_slug(settings.cheap.name)}.jsonl")
    todo = [r for r in records if cache.get(record_key(r)) is None]
    if todo and not settings.mock:
        if not confirm_spend([estimate_cost(todo, settings.cheap)], assume_yes=args.yes):
            _log("aborted; nothing spent.")
            return 1
    try:
        comps, stats = asyncio.run(
            run_model(
                records,
                cheap,
                cache=cache,
                concurrency=args.concurrency,
                budget_usd=args.budget,
                progress=_log,
            )
        )
    except BudgetExceeded as exc:
        _log(str(exc))
        return 1
    if settings.mock:
        _log("mock mode: labels NOT written (they would not be real).")
        return 0
    failed = [r.id for r in records if comps[r.id].error]
    if failed:
        _log(f"{len(failed)} calls failed; re-run to retry them. Labels not written.")
        return 1
    for r in records:
        r.cheap_correct = int(score(r, comps[r.id].text))
        r.label_source = "empirical"
    save_records(records, path)
    hard = sum(r.label for r in records)
    _log(
        f"labelled {len(records)} records: {hard} hard ({hard / len(records):.0%}), "
        f"spent ${stats.spend_usd:.4f}"
    )
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    records = load_records(DATA_DIR / "train.jsonl")
    use_empirical = args.labels == "empirical" or (
        args.labels == "auto" and all(r.cheap_correct is not None for r in records)
    )
    if use_empirical and any(r.cheap_correct is None for r in records):
        _log("empirical labels requested but train.jsonl has not been labelled (run `label`).")
        return 1
    labels = [r.label if use_empirical else r.prior_label for r in records]
    source = "empirical (cheap model failed)" if use_empirical else "prior (source/subject tier)"
    _log(f"training on {len(records)} queries, labels: {source}")
    clf, rep = EmbeddingClassifier.fit(
        [r.query for r in records], labels, FastEmbedEmbedder(), C=args.C, label_source=source
    )
    clf.save(args.out)
    _log(
        f"saved {args.out}: {rep.n_hard}/{rep.n_train} hard, "
        f"{rep.cv_folds}-fold CV accuracy {rep.cv_accuracy:.3f}, AUC {rep.cv_auc:.3f}"
    )
    Path(args.out).with_name("training_report.json").write_text(
        json.dumps(rep.__dict__, indent=2), encoding="utf-8"
    )
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    if args.mock:
        settings = replace(settings, mock=True)
        settings = replace(
            settings,
            cheap=replace(settings.cheap, name=settings.cheap.name or "mock-cheap", api_key="mock"),
            strong=replace(
                settings.strong, name=settings.strong.name or "mock-strong", api_key="mock"
            ),
        )
    records = _stratified(load_records(args.data), args.n)
    cheap, strong = _clients(settings, records)
    out_dir = Path(args.out or ("results/mock" if settings.mock else "results"))
    cache_dir = Path(args.cache_dir) / ("mock" if settings.mock else "")
    caches = {
        "cheap": ResponseCache(cache_dir / f"{_slug(settings.cheap.name)}.jsonl"),
        "strong": ResponseCache(cache_dir / f"{_slug(settings.strong.name)}.jsonl"),
    }
    specs = {"cheap": settings.cheap, "strong": settings.strong}
    if args.only_cached:
        # Evaluate only queries both models have already answered (e.g. after hitting a
        # provider's daily quota). The README must say which subset was used.
        records = [r for r in records if all(c.get(record_key(r)) for c in caches.values())]
        if not records:
            _log("--only-cached: no query has answers from both models.")
            return 1
        _log(f"--only-cached: evaluating {len(records)} queries answered by both models.")
    todo = {t: [r for r in records if caches[t].get(record_key(r)) is None] for t in caches}

    if args.from_cache and any(todo.values()):
        _log(f"--from-cache: missing responses ({ {t: len(v) for t, v in todo.items()} }).")
        return 1
    if not settings.mock and any(todo.values()):
        estimates = [estimate_cost(v, specs[t]) for t, v in todo.items() if v]
        if args.dry_run:
            _print_estimate(estimates)
            return 0
        if not confirm_spend(estimates, assume_yes=args.yes):
            _log("aborted; nothing spent.")
            return 1
    elif args.dry_run:
        _log("nothing to spend (mock mode or fully cached).")
        return 0

    budget = args.budget
    comps, stats = {}, {}
    for tier, client in (("cheap", cheap), ("strong", strong)):
        _log(
            f"running {specs[tier].name} on {len(records)} queries ({len(todo[tier])} uncached) ..."
        )
        try:
            comps[tier], stats[tier] = asyncio.run(
                run_model(
                    records,
                    client,
                    cache=caches[tier],
                    concurrency=args.concurrency,
                    budget_usd=budget,
                    progress=_log,
                )
            )
        except BudgetExceeded as exc:
            _log(str(exc))
            return 1
        if budget is not None:
            budget -= stats[tier].spend_usd
        if stats[tier].errors:
            _log(
                f"{stats[tier].errors} {specs[tier].name} calls failed (rate limit/outage). "
                "Not analysing partial data; re-run the same command to resume from the cache."
            )
            return 1

    classifiers: dict = {"heuristic": HeuristicClassifier(), "random": RandomClassifier()}
    if settings.classifier_path.exists():
        classifiers = {
            "embedding": EmbeddingClassifier.load(settings.classifier_path),
            **classifiers,
        }
    elif not settings.mock:
        _log(f"classifier {settings.classifier_path} missing; run `modelrouter train` first.")
        return 1
    else:
        classifiers = {"embedding": _mock_embedding(records), **classifiers}
    default_clf = "embedding" if "embedding" in classifiers else "heuristic"

    _log("analysing ...")
    analysis = analyse(
        records,
        comps["cheap"],
        comps["strong"],
        settings.cheap,
        settings.strong,
        classifiers,
        default_threshold=settings.confidence_threshold,
        default_classifier=default_clf,
        fallback=settings.fallback_enabled,
        min_avg_logprob=settings.min_avg_logprob,
    )
    errors = {t: sum(1 for c in comps[t].values() if c.error) for t in comps}
    meta = {
        "cheap_model": settings.cheap.name,
        "strong_model": settings.strong.name,
        "prices_per_m_tokens": {
            "cheap": [settings.cheap.price_in_per_m, settings.cheap.price_out_per_m],
            "strong": [settings.strong.price_in_per_m, settings.strong.price_out_per_m],
        },
        "n_queries": len(records),
        "by_source": dict(Counter(r.source for r in records)),
        "call_errors": errors,
        "fallback_enabled": settings.fallback_enabled,
        "default_threshold": settings.confidence_threshold,
        "classifier_meta": getattr(classifiers.get("embedding"), "meta", None),
        "temperature": 0.0,
        "version": __version__,
        "author": __author__,
    }
    paths = write_all(analysis, out_dir, meta, default_classifier=default_clf, mock=settings.mock)
    _log(f"wrote {', '.join(str(p) for p in paths.values())}")
    print(paths["md"].read_text(encoding="utf-8"))
    return 0


def _mock_embedding(records: list[Record]) -> EmbeddingClassifier:
    from .classifier import HashingEmbedder

    train_path = DATA_DIR / "train.jsonl"
    train = load_records(train_path) if train_path.exists() else records
    clf, _ = EmbeddingClassifier.fit(
        [r.query for r in train],
        [r.prior_label for r in train],
        HashingEmbedder(),
        label_source="prior",
    )
    return clf


def _print_estimate(estimates) -> None:
    for e in estimates:
        print(
            f"{e.model}: {e.n_calls} calls, expected ${e.expected_usd:.4f}, "
            f"worst-case ${e.worst_case_usd:.4f}"
        )
    print(
        f"TOTAL expected ${sum(e.expected_usd for e in estimates):.4f}, "
        f"worst-case ${sum(e.worst_case_usd for e in estimates):.4f}"
    )


def cmd_demo(args: argparse.Namespace) -> int:
    """Push synthetic traffic through the router in mock mode so the dashboard has data."""
    from .factory import build_router
    from .router import Router
    from .tracking import RequestLog

    settings = replace(Settings.from_env(), mock=True)
    records = load_records(args.data)
    oracle = _mock_oracle(records)
    clf_router = build_router(settings)
    router = Router(
        clf_router.classifier,
        MockClient(settings.cheap, "cheap", oracle=oracle),
        MockClient(settings.strong, "strong", oracle=oracle),
        confidence_threshold=settings.confidence_threshold,
    )
    log = RequestLog(settings.db_path)
    from .bench.tasks import build_messages

    rng = random.Random(0)
    for r in rng.sample(records, min(args.n, len(records))):
        res = asyncio.run(router.handle(build_messages(r)))
        log.record(res, mock=True)
    _log(f"logged {args.n} MOCK requests to {settings.db_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="modelrouter", description=f"ModelRouter {__version__} - by {__author__}"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the OpenAI-compatible proxy")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("build-data", help="download public datasets and write data/*.jsonl")
    s.add_argument("--bench-per-source", type=int, default=100)
    s.add_argument("--train-per-source", type=int, default=250)
    s.set_defaults(fn=cmd_build_data)

    s = sub.add_parser("label", help="empirically label training queries with the cheap model")
    s.add_argument("--yes", action="store_true", help="skip the cost confirmation prompt")
    s.add_argument("--budget", type=float, default=None, help="hard USD cap; abort if exceeded")
    s.add_argument("--concurrency", type=int, default=4)
    s.add_argument("--cache-dir", default="results/cache")
    s.set_defaults(fn=cmd_label)

    s = sub.add_parser("train", help="fit the embedding + logistic regression classifier")
    s.add_argument("--labels", choices=["auto", "prior", "empirical"], default="auto")
    s.add_argument("--C", type=float, default=1.0)
    s.add_argument("--out", default="models/classifier.json")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("bench", help="benchmark always-cheap vs always-strong vs router")
    s.add_argument("--data", default=str(DATA_DIR / "bench.jsonl"))
    s.add_argument("--n", type=int, default=300, help="number of queries (default 300)")
    s.add_argument("--yes", action="store_true", help="skip the cost confirmation prompt")
    s.add_argument("--dry-run", action="store_true", help="only print the cost estimate")
    s.add_argument(
        "--from-cache", action="store_true", help="never call models; use cached responses"
    )
    s.add_argument(
        "--only-cached",
        action="store_true",
        help="evaluate only queries both models already answered (no model calls)",
    )
    s.add_argument("--mock", action="store_true", help="offline fake models (results marked MOCK)")
    s.add_argument("--budget", type=float, default=None, help="hard USD cap; abort if exceeded")
    s.add_argument("--concurrency", type=int, default=4)
    s.add_argument("--cache-dir", default="results/cache")
    s.add_argument("--out", default=None)
    s.set_defaults(fn=cmd_bench)

    s = sub.add_parser("demo", help="log synthetic MOCK traffic for the dashboard")
    s.add_argument("--data", default=str(DATA_DIR / "bench.jsonl"))
    s.add_argument("--n", type=int, default=60)
    s.set_defaults(fn=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args))
    except ConfigError as exc:
        _log(f"configuration error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
