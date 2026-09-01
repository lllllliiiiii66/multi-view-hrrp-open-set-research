from __future__ import annotations

import numpy as np
import pytest

from hrrp_osr.evaluation.metrics import (
    binary_auroc,
    evaluate_open_set,
    fpr_at_unknown_tpr,
    macro_f1_score,
    threshold_for_known_acceptance,
)


def test_hand_calculated_accuracy_macro_f1_auroc_and_fpr95() -> None:
    known_true = np.array([0, 1, 0, 1])
    known_pred = np.array([0, 1, 1, 1])
    known_scores = np.array([0.1, 0.2, 0.8, 0.3])
    unknown_scores = np.array([0.9, 0.7, 0.6, 0.95])
    assert macro_f1_score(known_true, known_pred, labels=[0, 1]) == pytest.approx(
        (2 / 3 + 4 / 5) / 2
    )
    assert binary_auroc(known_scores, unknown_scores) == pytest.approx(14 / 16)
    assert fpr_at_unknown_tpr(known_scores, unknown_scores) == pytest.approx(1 / 4)


def test_known_acceptance_threshold_uses_known_validation_only() -> None:
    validation_scores = np.array([0.4, 0.1, 0.3, 0.2])
    threshold = threshold_for_known_acceptance(validation_scores, acceptance_rate=0.75)
    assert threshold == pytest.approx(0.3)
    assert np.mean(validation_scores <= threshold) == pytest.approx(0.75)


def test_open_set_fixture_metrics_and_score_direction() -> None:
    result = evaluate_open_set(
        known_true=np.array([0, 1, 0, 1]),
        known_pred=np.array([0, 1, 1, 1]),
        known_unknown_scores=np.array([0.1, 0.2, 0.8, 0.3]),
        unknown_pred=np.array([0, 0, 1, 1]),
        unknown_unknown_scores=np.array([0.9, 0.7, 0.6, 0.95]),
        known_validation_scores=np.array([0.4, 0.1, 0.3, 0.2]),
        known_class_count=2,
        known_acceptance_rate=0.75,
    )
    assert result["known_accuracy"] == pytest.approx(0.75)
    assert result["auroc"] == pytest.approx(14 / 16)
    assert result["fpr95"] == pytest.approx(0.25)
    assert result["threshold"] == pytest.approx(0.3)
    assert result["known_acceptance_rate"] == pytest.approx(0.75)
    assert result["unknown_rejection_rate"] == pytest.approx(1.0)
    assert result["k_plus_1_macro_f1"] == pytest.approx((2 / 3 + 1.0 + 8 / 9) / 3)
    assert 0.0 <= result["oscr"] <= 1.0
