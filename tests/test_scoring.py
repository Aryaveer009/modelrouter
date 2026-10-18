import pytest

from modelrouter.bench.tasks import extract_final, score
from modelrouter.data.records import Record


def rec(task, gold, aliases=None):
    return Record("id", "src", task, "q", gold, aliases or [])


@pytest.mark.parametrize(
    "text,ok",
    [
        ("work...\nFinal answer: 18", True),
        ("Final answer: $1,018.00", False),
        ("Final answer: 18.0.", True),
        ("so 17\nFinal answer: 19", False),
        ("I think it is 18", True),  # falls back to the last number
    ],
)
def test_math(text, ok):
    assert score(rec("math", "18"), text) is ok


def test_math_thousands_separator():
    assert score(rec("math", "1018"), "Final answer: $1,018")


@pytest.mark.parametrize(
    "text,ok",
    [
        ("Final answer: C", True),
        ("Final answer: (c)", True),
        ("Final answer: B", False),
        ("Final answer: C. Because ...", True),
        ("The answer is C", True),
        ("no idea", False),
    ],
)
def test_mcq(text, ok):
    assert score(rec("mcq", "C"), text) is ok


@pytest.mark.parametrize(
    "text,ok",
    [
        ("Final answer: Paris", True),
        ("Final answer: The City of Light", True),
        ("Final answer: Lyon", False),
        ("Final answer: Parisian suburbs", False),
    ],
)
def test_qa_aliases(text, ok):
    assert score(rec("qa", "Paris", ["Paris", "City of Light"]), text) is ok


def test_extract_final_takes_last_and_strips_markdown():
    assert extract_final("Final answer: 1\nFinal answer: **2**.") == "2"
    assert extract_final("nothing") is None
