"""Evaluation helpers for guarded query routing."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from router.labels import ID_LABELS, OOD


@dataclass(frozen=True)
class RouterScores:
    """Core metrics for a guarded query router."""

    id_accuracy: float
    ood_accuracy: float
    gqr_score: float


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Return plain accuracy, or 0.0 for an empty input."""

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0:
        return 0.0
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)


def gqr_score(id_accuracy: float, ood_accuracy: float) -> float:
    """Harmonic mean used by GQR-Bench."""

    denominator = id_accuracy + ood_accuracy
    if denominator == 0:
        return 0.0
    return 2 * id_accuracy * ood_accuracy / denominator


def evaluate_router(y_true: Sequence[int], y_pred: Sequence[int]) -> RouterScores:
    """Compute ID routing accuracy, OOD rejection accuracy, and GQR score."""

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    id_pairs = [
        (true, pred)
        for true, pred in zip(y_true, y_pred)
        if true in ID_LABELS
    ]
    ood_pairs = [
        (true, pred)
        for true, pred in zip(y_true, y_pred)
        if true == OOD
    ]

    id_acc = accuracy(
        [true for true, _ in id_pairs],
        [pred for _, pred in id_pairs],
    )
    ood_acc = accuracy(
        [true for true, _ in ood_pairs],
        [pred for _, pred in ood_pairs],
    )
    return RouterScores(
        id_accuracy=id_acc,
        ood_accuracy=ood_acc,
        gqr_score=gqr_score(id_acc, ood_acc),
    )


def confusion_counts(
    y_true: Iterable[int],
    y_pred: Iterable[int],
) -> dict[tuple[int, int], int]:
    """Return compact confusion counts keyed by (true_label, predicted_label)."""

    return dict(Counter(zip(y_true, y_pred)))
