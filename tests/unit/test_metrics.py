"""Metric correctness against hand-computed values."""

import numpy as np
import pytest

from digit_classifier.training.metrics import confusion_matrix, evaluate_predictions


def test_perfect_predictions() -> None:
    y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    m = evaluate_predictions(y, y)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert all(pc["f1"] == 1.0 for pc in m["per_class"])


def test_hand_computed_case() -> None:
    # true:  0 0 0 1 1 -> pred: 0 0 1 1 1
    y_true = np.array([0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1])
    m = evaluate_predictions(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(4 / 5)
    c0 = m["per_class"][0]  # precision 2/2=1.0, recall 2/3
    assert c0["precision"] == pytest.approx(1.0)
    assert c0["recall"] == pytest.approx(2 / 3)
    assert c0["f1"] == pytest.approx(2 * 1.0 * (2 / 3) / (1.0 + 2 / 3))
    c1 = m["per_class"][1]  # precision 2/3, recall 2/2=1.0
    assert c1["precision"] == pytest.approx(2 / 3)
    assert c1["recall"] == pytest.approx(1.0)


def test_absent_class_yields_zero_not_nan() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    m = evaluate_predictions(y_true, y_pred)
    c9 = m["per_class"][9]
    assert c9["precision"] == 0.0 and c9["recall"] == 0.0 and c9["f1"] == 0.0
    assert c9["support"] == 0


def test_invalid_labels_rejected() -> None:
    with pytest.raises(ValueError, match="outside"):
        confusion_matrix(np.array([0, 10]), np.array([0, 1]))
    with pytest.raises(ValueError, match="zero samples"):
        confusion_matrix(np.array([], dtype=int), np.array([], dtype=int))
    with pytest.raises(ValueError, match="same shape"):
        confusion_matrix(np.array([0, 1]), np.array([0]))
