"""Prompting and answer scoring for each task type."""

from __future__ import annotations

import re
import unicodedata

from ..data.records import Record
from ..types import Message

SYSTEM_PROMPTS = {
    "math": (
        "Solve the problem. Reason step by step but stay concise. "
        "End with a final line exactly of the form 'Final answer: <number>'."
    ),
    "mcq": (
        "Answer the multiple-choice question. Give brief reasoning, then end with a final line "
        "exactly of the form 'Final answer: <letter>' where <letter> is A, B, C or D."
    ),
    "qa": (
        "Answer the question concisely. End with a final line exactly of the form "
        "'Final answer: <short answer>'."
    ),
}
# Output-token caps per task, and the average output length we assume for cost *estimates*.
MAX_TOKENS = {"math": 512, "mcq": 384, "qa": 128}
EXPECTED_OUTPUT_TOKENS = {"math": 220, "mcq": 120, "qa": 25}


def build_messages(record: Record) -> list[Message]:
    return [
        {"role": "system", "content": SYSTEM_PROMPTS[record.task]},
        {"role": "user", "content": record.query},
    ]


_FINAL = re.compile(r"final answer\s*[:\-]\s*(.+)", re.IGNORECASE)
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_final(text: str) -> str | None:
    matches = _FINAL.findall(text)
    if not matches:
        return None
    return matches[-1].strip().strip("*`_ ").rstrip(".").strip("*`_ ")


def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _parse_number(s: str) -> float | None:
    m = _NUMBER.search(s.replace("$", ""))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def score(record: Record, text: str) -> bool:
    """True if the model's response contains the correct final answer."""
    final = extract_final(text)
    if record.task == "math":
        if final is not None:
            number = _parse_number(final)
        else:
            nums = _NUMBER.findall(text)
            number = _parse_number(nums[-1]) if nums else None
        gold = _parse_number(record.gold)
        return number is not None and gold is not None and abs(number - gold) < 1e-4
    if record.task == "mcq":
        target = final if final is not None else _fallback_choice(text)
        if target is None:
            return False
        scan = target.upper() if len(target) <= 3 else target
        letter = re.search(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", scan)
        return letter is not None and letter.group(1) == record.gold
    # qa: any gold alias appears as a whole-word phrase in the predicted answer
    pred = _normalize(final if final is not None else text.strip().split("\n")[0])
    if not pred:
        return False
    padded = f" {pred} "
    for alias in record.aliases or [record.gold]:
        a = _normalize(alias)
        if a and (pred == a or f" {a} " in padded):
            return True
    return False


def _fallback_choice(text: str) -> str | None:
    m = re.search(r"answer is\s*\(?([A-D])\b", text, re.IGNORECASE)
    return m.group(1).upper() if m else None
