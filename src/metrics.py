"""Shared evaluation metrics, so that every model is scored identically.

Macro-F1 is averaged over the classes PRESENT in y_true of the evaluated
split. All models are compared on the same rows, so this label set is the same
for every model on a given (task, split). That avoids the sklearn default
(union of true and predicted labels), which would penalise a model with an
extra zero whenever it predicts a class that has no support in that split.
"""
from __future__ import annotations

from typing import Iterable

from sklearn.metrics import accuracy_score, f1_score


def eval_labels(y_true: Iterable) -> list[str]:
    """Sorted label set used for averaging: classes present in y_true."""
    return sorted({str(y) for y in y_true})


def macro_f1(y_true: Iterable, y_pred: Iterable) -> float:
    """Macro-F1 over the classes present in y_true."""
    y_true = [str(y) for y in y_true]
    y_pred = [str(y) for y in y_pred]
    return float(f1_score(y_true, y_pred, labels=eval_labels(y_true), average="macro", zero_division=0))


def compute_metrics(y_true: Iterable, y_pred: Iterable) -> dict:
    """f1_macro, f1_weighted, accuracy and n_eval with the shared label convention."""
    y_true = [str(y) for y in y_true]
    y_pred = [str(y) for y in y_pred]
    labels = eval_labels(y_true)
    return {
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "n_eval": len(y_true),
    }
