"""Download public datasets and build the labelled training set and the benchmark set.

Sources (fetched through the Hugging Face datasets-server HTTP API):
  * GSM8K          - grade-school math word problems            (MIT licence)
  * MMLU subset    - multiple choice across easy/hard subjects  (MIT licence)
  * TriviaQA       - short factual questions                    (Apache-2.0)

Training and benchmark queries are disjoint (different splits or row ranges).
"""

from __future__ import annotations

import json
import random
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from .records import Record

API = "https://datasets-server.huggingface.co/rows"
CACHE_DIR = Path(".cache/hf_rows")

# Subjects whose questions need multi-step reasoning/knowledge; the rest are recall-style.
MMLU_HARD_SUBJECTS = [
    "abstract_algebra",
    "college_mathematics",
    "college_physics",
    "formal_logic",
    "high_school_mathematics",
    "college_chemistry",
    "electrical_engineering",
    "machine_learning",
]
MMLU_EASY_SUBJECTS = [
    "high_school_geography",
    "high_school_government_and_politics",
    "sociology",
    "us_foreign_policy",
    "marketing",
    "high_school_psychology",
    "world_religions",
    "management",
]


def fetch_rows(
    dataset: str,
    config: str,
    split: str,
    offset: int,
    length: int,
    *,
    client: httpx.Client | None = None,
) -> list[dict[str, Any]]:
    """Fetch `length` rows starting at `offset`, paging in chunks of 100."""
    own = client is None
    client = client or httpx.Client(timeout=60)
    rows: list[dict[str, Any]] = []
    try:
        while len(rows) < length:
            want = min(100, length - len(rows))
            params = {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset + len(rows),
                "length": want,
            }
            data = _get_json(client, params)
            batch = [r["row"] for r in data.get("rows", [])]
            if not batch:
                break
            rows.extend(batch)
        return rows[:length]
    finally:
        if own:
            client.close()


def _get_json(client: httpx.Client, params: dict[str, Any]) -> dict[str, Any]:
    """GET with an on-disk cache (so reruns resume) and patient backoff for HTTP 429."""
    key = "_".join(str(params[k]).replace("/", "-") for k in sorted(params))
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    for attempt in range(9):
        resp = client.get(API, params=params)
        if resp.status_code == 200:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(resp.text, encoding="utf-8")
            time.sleep(0.5)
            return resp.json()
        if resp.status_code in {429, 500, 502, 503, 504}:
            time.sleep(min(90, 5 * 2**attempt))
            continue
        resp.raise_for_status()
    raise RuntimeError(f"datasets-server kept failing for {params}")


def _gsm8k(split: str, offset: int, n: int, hf_split: str, tag: str) -> list[Record]:
    rows = fetch_rows("openai/gsm8k", "main", hf_split, offset, n)
    out = []
    for i, row in enumerate(rows):
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        out.append(
            Record(
                id=f"gsm8k-{hf_split}-{offset + i}",
                source="gsm8k",
                task="math",
                query=row["question"].strip(),
                gold=gold,
                split=split,
                prior_label=1,
            )
        )
    del tag
    return out


def _mmlu(
    split: str, offset: int, per_subject: int, subjects: list[str], hard: set[str]
) -> list[Record]:
    out: list[Record] = []
    with httpx.Client(timeout=60) as client:
        for subj in subjects:
            rows = fetch_rows("cais/mmlu", subj, "test", offset, per_subject, client=client)
            for i, row in enumerate(rows):
                choices = row["choices"]
                body = (
                    row["question"].strip()
                    + "\n"
                    + "\n".join(f"{'ABCD'[j]}. {c}" for j, c in enumerate(choices))
                )
                out.append(
                    Record(
                        id=f"mmlu-{subj}-{offset + i}",
                        source="mmlu",
                        task="mcq",
                        query=body,
                        gold="ABCD"[int(row["answer"])],
                        subject=subj,
                        split=split,
                        prior_label=int(subj in hard),
                    )
                )
    return out


def _trivia(split: str, offset: int, n: int) -> list[Record]:
    rows = fetch_rows("mandarjoshi/trivia_qa", "rc.nocontext", "validation", offset, n)
    out = []
    for i, row in enumerate(rows):
        ans = row["answer"]
        aliases = list(dict.fromkeys([ans["value"], *ans.get("aliases", [])]))
        out.append(
            Record(
                id=f"triviaqa-validation-{offset + i}",
                source="triviaqa",
                task="qa",
                query=row["question"].strip(),
                gold=ans["value"],
                aliases=aliases[:25],
                split=split,
                prior_label=0,
            )
        )
    return out


def _subsample(records: list[Record], n: int, seed: int) -> list[Record]:
    if len(records) <= n:
        return records
    rng = random.Random(seed)
    return sorted(rng.sample(records, n), key=lambda r: r.id)


def build_datasets(
    *,
    bench_per_source: int = 100,
    train_per_source: int = 250,
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> tuple[list[Record], list[Record]]:
    """Return (train, bench). Defaults give 300 benchmark queries and 750 training queries."""
    subjects = MMLU_HARD_SUBJECTS + MMLU_EASY_SUBJECTS
    hard = set(MMLU_HARD_SUBJECTS)
    bench_ps = -(-bench_per_source // len(subjects))  # ceil
    train_ps = -(-train_per_source // len(subjects))

    log("GSM8K ...")
    bench = _gsm8k("bench", 0, bench_per_source, "test", "bench")
    train = _gsm8k("train", 0, train_per_source, "train", "train")
    log("MMLU ...")
    bench += _subsample(_mmlu("bench", 0, bench_ps, subjects, hard), bench_per_source, seed)
    # Training rows come from later offsets of the same subjects, so they never overlap.
    train += _subsample(_mmlu("train", bench_ps, train_ps, subjects, hard), train_per_source, seed)
    log("TriviaQA ...")
    bench += _trivia("bench", 0, bench_per_source)
    train += _trivia("train", 5000, train_per_source)

    # Drop any training query whose text also appears in the benchmark (or twice in training).
    seen = {_norm_q(r.query) for r in bench}
    clean_train: list[Record] = []
    for r in train:
        key = _norm_q(r.query)
        if key not in seen:
            seen.add(key)
            clean_train.append(r)
    assert not {r.id for r in bench} & {r.id for r in clean_train}
    return clean_train, bench


def _norm_q(q: str) -> str:
    return re.sub(r"\W+", " ", q.lower()).strip()
