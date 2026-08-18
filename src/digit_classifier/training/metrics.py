"""Evaluation metrics computed from a confusion matrix.

Implemented in numpy instead of pulling in scikit-learn for this - the
formulas are standard and small enough to write directly, and every one
gets checked against hand-computed values in tests.
"""

from __future__ import annotations

import numpy as np

NUM_CLASSES = 10


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if y_true.size == 0:
        raise ValueError("cannot evaluate zero samples")
    for name, arr in (("y_true", y_true), ("y_pred", y_pred)):
        if arr.min() < 0 or arr.max() >= NUM_CLASSES:
            raise ValueError(f"{name} contains labels outside 0..{NUM_CLASSES - 1}")
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(cm, (y_true.astype(np.int64), y_pred.astype(np.int64)), 1)
    return cm


def metrics_from_confusion(cm: np.ndarray) -> dict:
    total = int(cm.sum())
    correct = int(np.trace(cm))
    tp = np.diag(cm).astype(np.float64)
    pred_totals = cm.sum(axis=0).astype(np.float64)  # column: predicted as c
    true_totals = cm.sum(axis=1).astype(np.float64)  # row: actually c
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(pred_totals > 0, tp / pred_totals, 0.0)
        recall = np.where(true_totals > 0, tp / true_totals, 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
    return {
        "accuracy": correct / total,
        "macro_f1": float(f1.mean()),
        "per_class": [
            {
                "digit": c,
                "precision": float(precision[c]),
                "recall": float(recall[c]),
                "f1": float(f1[c]),
                "support": int(true_totals[c]),
            }
            for c in range(NUM_CLASSES)
        ],
        "confusion_matrix": cm.tolist(),
        "n_samples": total,
    }


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return metrics_from_confusion(confusion_matrix(y_true, y_pred))
