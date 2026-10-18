import numpy as np
import pytest

from modelrouter.classifier import (
    EmbeddingClassifier,
    HashingEmbedder,
    HeuristicClassifier,
    RandomClassifier,
    load_classifier,
)
from modelrouter.classifier.features import extract_features

EASY = [
    "What is the capital of France?",
    "Who wrote Hamlet?",
    "When was the Eiffel Tower built?",
    "Which country has the city of Lima?",
    "Who painted the Mona Lisa?",
    "What year did WW2 end?",
]
HARD = [
    "Janet has 16 eggs per day, eats 3 and bakes with 4. She sells the rest for $2 each. "
    "How much does she make every day in dollars?",
    "Solve the equation 3x^2 + 5x - 2 = 0 and calculate the sum of the roots times 4.",
    "Write a Python function that implements Dijkstra's algorithm and analyze its complexity.",
    "Prove that the sum of two even integers is even, step by step, and justify each step.",
]


def test_features_detect_signals():
    f = extract_features("Calculate 12 + 30 / 6 and explain why. ```code```")
    assert f["math_ops"] >= 1 and f["code_signal"] >= 1 and f["reasoning_keywords"] >= 1
    assert extract_features("Who wrote Hamlet?")["factoid_cue"] == 1.0


def test_features_multiple_choice_detected():
    q = "Pick one.\nA. one\nB. two\nC. three\nD. four"
    assert extract_features(q)["multiple_choice"] == 1.0


def test_heuristic_ranks_hard_above_easy():
    clf = HeuristicClassifier()
    easy = clf.predict_proba_hard(EASY)
    hard = clf.predict_proba_hard(HARD)
    assert easy.max() < hard.min()
    assert ((0 <= easy) & (easy <= 1)).all()


def test_heuristic_is_deterministic():
    clf = HeuristicClassifier()
    assert clf.predict_proba_hard(HARD)[0] == clf.predict_proba_hard(HARD)[0]


def _train_data():
    texts = EASY * 8 + HARD * 12
    # vary the strings a little so CV folds see distinct rows
    texts = [f"{t} (v{i})" for i, t in enumerate(texts)]
    labels = [0] * (len(EASY) * 8) + [1] * (len(HARD) * 12)
    return texts, labels


def test_embedding_classifier_learns_and_roundtrips(tmp_path):
    texts, labels = _train_data()
    clf, report = EmbeddingClassifier.fit(texts, labels, HashingEmbedder(), label_source="test")
    assert report.cv_auc > 0.9 and report.n_hard == sum(labels)
    p = clf.predict_proba_hard(
        ["Who wrote Macbeth?", "Calculate the sum of 12 + 30 and solve for x"]
    )
    assert p[0] < p[1]

    path = tmp_path / "clf.json"
    clf.save(path)
    loaded = EmbeddingClassifier.load(path)
    np.testing.assert_allclose(
        loaded.predict_proba_hard(texts[:5]), clf.predict_proba_hard(texts[:5])
    )
    assert loaded.meta["label_source"] == "test"


def test_fit_rejects_single_class():
    with pytest.raises(ValueError):
        EmbeddingClassifier.fit(["a", "b", "c"], [0, 0, 0], HashingEmbedder())


def test_random_classifier_is_deterministic_and_in_range():
    r = RandomClassifier(seed=1)
    a = r.predict_proba_hard(["x", "y", "z"])
    np.testing.assert_array_equal(a, RandomClassifier(seed=1).predict_proba_hard(["x", "y", "z"]))
    assert ((0 <= a) & (a < 1)).all()


def test_load_classifier_falls_back_to_heuristic(tmp_path):
    clf = load_classifier("embedding", tmp_path / "missing.json")
    assert clf.name == "heuristic"
    assert load_classifier("heuristic", "ignored").name == "heuristic"
