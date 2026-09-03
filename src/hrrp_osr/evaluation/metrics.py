from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import numpy as np

from hrrp_osr.data.errors import DataValidationError


def _one_dimensional(values: np.ndarray | Iterable[Any], name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise DataValidationError(f"{name} must be a non-empty one-dimensional array")
    return array


def _trapezoid_area(values: np.ndarray, coordinates: np.ndarray) -> float:
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        trapezoid = np.trapz
    return float(trapezoid(values, coordinates))


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true = _one_dimensional(y_true, "y_true")
    pred = _one_dimensional(y_pred, "y_pred")
    if true.shape != pred.shape:
        raise DataValidationError("y_true and y_pred have different shapes")
    return float(np.mean(true == pred))


def macro_f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Iterable[int],
) -> float:
    true = _one_dimensional(y_true, "y_true")
    pred = _one_dimensional(y_pred, "y_pred")
    if true.shape != pred.shape:
        raise DataValidationError("y_true and y_pred have different shapes")
    scores: list[float] = []
    for label in labels:
        true_positive = int(np.count_nonzero((true == label) & (pred == label)))
        false_positive = int(np.count_nonzero((true != label) & (pred == label)))
        false_negative = int(np.count_nonzero((true == label) & (pred != label)))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    if not scores:
        raise DataValidationError("macro-F1 label set is empty")
    return float(np.mean(scores))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        average_rank = (start + 1 + stop) / 2.0
        ranks[order[start:stop]] = average_rank
        start = stop
    return ranks


def binary_auroc(known_scores: np.ndarray, unknown_scores: np.ndarray) -> float:
    known = _one_dimensional(known_scores, "known_scores").astype(np.float64)
    unknown = _one_dimensional(unknown_scores, "unknown_scores").astype(np.float64)
    if not np.isfinite(known).all() or not np.isfinite(unknown).all():
        raise DataValidationError("AUROC scores contain NaN or Inf")
    combined = np.concatenate([known, unknown])
    ranks = _average_ranks(combined)
    positive_rank_sum = float(np.sum(ranks[known.size :]))
    mann_whitney_u = positive_rank_sum - unknown.size * (unknown.size + 1) / 2.0
    return mann_whitney_u / (known.size * unknown.size)


def threshold_for_known_acceptance(
    known_validation_scores: np.ndarray,
    acceptance_rate: float = 0.95,
) -> float:
    scores = _one_dimensional(
        known_validation_scores, "known_validation_scores"
    ).astype(np.float64)
    if not np.isfinite(scores).all():
        raise DataValidationError("known validation scores contain NaN or Inf")
    if not 0.0 < acceptance_rate <= 1.0:
        raise DataValidationError("known acceptance rate must be in (0, 1]")
    accepted_count = int(math.ceil(acceptance_rate * scores.size))
    return float(np.sort(scores)[accepted_count - 1])


def fpr_at_unknown_tpr(
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    target_tpr: float = 0.95,
) -> float:
    known = _one_dimensional(known_scores, "known_scores").astype(np.float64)
    unknown = _one_dimensional(unknown_scores, "unknown_scores").astype(np.float64)
    if not 0.0 < target_tpr <= 1.0:
        raise DataValidationError("unknown target TPR must be in (0, 1]")
    detected_count = int(math.ceil(target_tpr * unknown.size))
    threshold_index = unknown.size - detected_count
    threshold = float(np.sort(unknown)[threshold_index])
    return float(np.mean(known >= threshold))


def oscr_score(
    known_true: np.ndarray,
    known_pred: np.ndarray,
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
) -> float:
    true = _one_dimensional(known_true, "known_true")
    pred = _one_dimensional(known_pred, "known_pred")
    known = _one_dimensional(known_scores, "known_scores").astype(np.float64)
    unknown = _one_dimensional(unknown_scores, "unknown_scores").astype(np.float64)
    if true.shape != pred.shape or true.shape != known.shape:
        raise DataValidationError("known OSCR arrays have different shapes")
    correct = pred == true
    thresholds = np.concatenate(
        [
            np.array([-np.inf]),
            np.unique(np.concatenate([known, unknown])),
            np.array([np.inf]),
        ]
    )
    false_positive_rates = np.array(
        [np.mean(unknown <= threshold) for threshold in thresholds], dtype=np.float64
    )
    correct_classification_rates = np.array(
        [
            np.mean(correct & (known <= threshold))
            for threshold in thresholds
        ],
        dtype=np.float64,
    )
    return _trapezoid_area(correct_classification_rates, false_positive_rates)


def evaluate_open_set(
    *,
    known_true: np.ndarray,
    known_pred: np.ndarray,
    known_unknown_scores: np.ndarray,
    unknown_pred: np.ndarray,
    unknown_unknown_scores: np.ndarray,
    known_validation_scores: np.ndarray,
    known_class_count: int,
    known_acceptance_rate: float = 0.95,
) -> dict[str, float]:
    true_known = _one_dimensional(known_true, "known_true").astype(int)
    pred_known = _one_dimensional(known_pred, "known_pred").astype(int)
    score_known = _one_dimensional(
        known_unknown_scores, "known_unknown_scores"
    ).astype(float)
    pred_unknown = _one_dimensional(unknown_pred, "unknown_pred").astype(int)
    score_unknown = _one_dimensional(
        unknown_unknown_scores, "unknown_unknown_scores"
    ).astype(float)
    if true_known.shape != pred_known.shape or true_known.shape != score_known.shape:
        raise DataValidationError("known evaluation arrays have different shapes")
    if pred_unknown.shape != score_unknown.shape:
        raise DataValidationError("unknown evaluation arrays have different shapes")
    if known_class_count < 2:
        raise DataValidationError("known_class_count must be at least two")

    threshold = threshold_for_known_acceptance(
        known_validation_scores, acceptance_rate=known_acceptance_rate
    )
    operating_point = open_set_operating_point(
        known_true=true_known,
        known_pred=pred_known,
        known_unknown_scores=score_known,
        unknown_pred=pred_unknown,
        unknown_unknown_scores=score_unknown,
        known_class_count=known_class_count,
        threshold=threshold,
    )
    return {
        "known_accuracy": accuracy_score(true_known, pred_known),
        "known_macro_f1": macro_f1_score(
            true_known, pred_known, labels=range(known_class_count)
        ),
        "auroc": binary_auroc(score_known, score_unknown),
        "oscr": oscr_score(
            true_known,
            pred_known,
            score_known,
            score_unknown,
        ),
        "fpr95": fpr_at_unknown_tpr(score_known, score_unknown, target_tpr=0.95),
        "threshold_known_acceptance_target": known_acceptance_rate,
        **operating_point,
    }


def open_set_operating_point(
    *,
    known_true: np.ndarray,
    known_pred: np.ndarray,
    known_unknown_scores: np.ndarray,
    unknown_pred: np.ndarray,
    unknown_unknown_scores: np.ndarray,
    known_class_count: int,
    threshold: float,
) -> dict[str, float]:
    """Evaluate one fixed open-set operating point.

    This is separate from :func:`evaluate_open_set` because some published
    methods define an intrinsic hard boundary.  CBD's original rule is one such
    case: reject when ``p_test - p_boundary > 0``.
    """

    true_known = _one_dimensional(known_true, "known_true").astype(int)
    pred_known = _one_dimensional(known_pred, "known_pred").astype(int)
    score_known = _one_dimensional(
        known_unknown_scores, "known_unknown_scores"
    ).astype(float)
    pred_unknown = _one_dimensional(unknown_pred, "unknown_pred").astype(int)
    score_unknown = _one_dimensional(
        unknown_unknown_scores, "unknown_unknown_scores"
    ).astype(float)
    if true_known.shape != pred_known.shape or true_known.shape != score_known.shape:
        raise DataValidationError("known operating-point arrays have different shapes")
    if pred_unknown.shape != score_unknown.shape:
        raise DataValidationError("unknown operating-point arrays have different shapes")
    if known_class_count < 2:
        raise DataValidationError("known_class_count must be at least two")
    if not np.isfinite(score_known).all() or not np.isfinite(score_unknown).all():
        raise DataValidationError("operating-point scores contain NaN or Inf")
    if not np.isfinite(threshold):
        raise DataValidationError("operating-point threshold must be finite")

    known_rejected = score_known > threshold
    unknown_rejected = score_unknown > threshold
    known_correct_acceptance_rate = float(
        np.mean((pred_known == true_known) & ~known_rejected)
    )
    unknown_rejection_rate = float(np.mean(unknown_rejected))
    operating_sum = known_correct_acceptance_rate + unknown_rejection_rate
    open_set_harmonic_score = (
        0.0
        if operating_sum == 0.0
        else 2.0
        * known_correct_acceptance_rate
        * unknown_rejection_rate
        / operating_sum
    )
    unknown_label = known_class_count
    open_true = np.concatenate(
        [true_known, np.full(score_unknown.size, unknown_label, dtype=int)]
    )
    open_pred = np.concatenate(
        [
            np.where(known_rejected, unknown_label, pred_known),
            np.where(unknown_rejected, unknown_label, pred_unknown),
        ]
    )
    return {
        "threshold": threshold,
        "known_acceptance_rate": float(np.mean(~known_rejected)),
        "known_correct_acceptance_rate": known_correct_acceptance_rate,
        "unknown_rejection_rate": unknown_rejection_rate,
        "open_set_harmonic_score": open_set_harmonic_score,
        "k_plus_1_macro_f1": macro_f1_score(
            open_true, open_pred, labels=range(known_class_count + 1)
        ),
    }


def summarize_metric_repeats(
    repeats: Iterable[Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    materialized = list(repeats)
    if not materialized:
        raise DataValidationError("metric repeat list is empty")
    keys = sorted(materialized[0])
    if any(sorted(item) != keys for item in materialized):
        raise DataValidationError("metric repeat keys are inconsistent")
    return {
        key: {
            "mean": float(np.mean([item[key] for item in materialized])),
            "std": float(np.std([item[key] for item in materialized], ddof=0)),
        }
        for key in keys
    }
