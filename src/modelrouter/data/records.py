"""The query record schema shared by training, labelling and benchmarking."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Record:
    id: str
    source: str  # gsm8k | mmlu | triviaqa
    task: str  # math | mcq | qa
    query: str
    gold: str  # canonical gold answer (number, letter, or primary alias)
    aliases: list[str] = field(default_factory=list)  # acceptable answers for qa
    subject: str = ""
    split: str = "train"  # train | bench
    prior_label: int = 0  # 1 = hard by source/subject tier
    cheap_correct: int | None = None  # filled by empirical labelling
    label_source: str = "prior"

    @property
    def label(self) -> int:
        """Training label: 1 = hard. Empirical (cheap model failed) wins over the prior."""
        if self.cheap_correct is not None:
            return 1 - int(self.cheap_correct)
        return self.prior_label


def load_records(path: str | Path) -> list[Record]:
    out: list[Record] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(Record(**json.loads(line)))
    return out


def save_records(records: list[Record], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
