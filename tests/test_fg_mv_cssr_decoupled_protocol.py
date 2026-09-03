from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from hrrp_osr.data.errors import DataValidationError
from hrrp_osr.evaluation.ms_mean_factorial import REPORT_METRIC_KEYS
from hrrp_osr.training.fg_mv_cssr_decoupled_protocol import (
    CONFIRMATION_PAIRS,
    CSSR_SEED,
    D0_R2_CLASS_CONDITIONAL_MLS,
    D1_DECOUPLED_REL_CSSR,
    D2_DECOUPLED_ABSREL_CSSR,
    METHODS,
    PILOT_PAIRS,
    TRAINABLE_METHODS,
    audit_shared_r2_predictions,
    build_guided_reference_scores,
    build_identity_and_absorption_rows,
    build_phase_plan,
    build_single_view_schedule,
    class_conditional_mls_for_roles,
    class_conditional_mls_score,
    ddg_false_accept_counts,
    evaluate_confirmation_gate,
    evaluate_pilot_gate,
    guided_scores_from_r2_predictions,
    recompute_metrics_from_prediction_rows,
)


def _single_view_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {
            "experiment_role": "known_calibration",
            "model_label": 0,
            "sample_id": "ignored-calibration",
            "angle_deg": 1,
        }
    ]
    for label in range(5):
        for index in range(144):
            rows.append(
                {
                    "experiment_role": "train_known",
                    "model_label": label,
                    "sample_id": f"class-{label}-sample-{143-index:03d}",
                    "angle_deg": 2 * (index % 180) + 1,
                }
            )
    return rows


def test_single_view_schedule_is_deterministic_unique_and_class_balanced() -> None:
    rows = _single_view_rows()
    first, first_audit = build_single_view_schedule(
        rows, pair_id="N1", angle_fold=0, epoch=1, cssr_seed=CSSR_SEED
    )
    repeated, repeated_audit = build_single_view_schedule(
        rows, pair_id="N1", angle_fold=0, epoch=1, cssr_seed=CSSR_SEED
    )
    next_epoch, _ = build_single_view_schedule(
        rows, pair_id="N1", angle_fold=0, epoch=2, cssr_seed=CSSR_SEED
    )

    assert np.array_equal(first, repeated)
    assert first_audit == repeated_audit
    assert not np.array_equal(first, next_epoch)
    assert len(first) == len(set(first.tolist())) == 720
    assert all(rows[int(index)]["experiment_role"] == "train_known" for index in first)
    assert first_audit["sample_usage_exactly_once"] is True
    assert first_audit["full_batches_class_balanced"] is True
    assert first_audit["pair_multiplicity_used"] is False
    labels = [int(rows[int(index)]["model_label"]) for index in first]
    for start in range(0, 640, 128):
        counts = np.bincount(labels[start : start + 128], minlength=5)
        assert counts.max() - counts.min() <= 1


def test_single_view_schedule_rejects_pair_multiplicity_and_even_angles() -> None:
    duplicated = _single_view_rows()
    duplicated.append(dict(duplicated[1]))
    with pytest.raises(DataValidationError, match="population|multiplicity"):
        build_single_view_schedule(
            duplicated, pair_id="N1", angle_fold=0, epoch=1, cssr_seed=CSSR_SEED
        )

    even = _single_view_rows()
    even[1]["angle_deg"] = 2
    with pytest.raises(DataValidationError, match="even-angle"):
        build_single_view_schedule(
            even, pair_id="N1", angle_fold=0, epoch=1, cssr_seed=CSSR_SEED
        )


def _reference_fixture() -> tuple[list[dict[str, object]], np.ndarray]:
    rows: list[dict[str, object]] = []
    values: list[np.ndarray] = []
    for label in range(5):
        for index, diagonal in enumerate((1.0, 2.0)):
            rows.append(
                {
                    "experiment_role": "known_calibration",
                    "model_label": label,
                    "sample_id": f"cal-{label}-{index}",
                }
            )
            row = np.full(5, 10.0 + label)
            row[label] = diagonal
            values.append(row)
    for index in range(2):
        rows.append(
            {
                "experiment_role": "surrogate_unknown",
                "model_label": -1,
                "sample_id": f"surrogate-{index}",
            }
        )
        values.append(np.full(5, 3.0 + index))
    return rows, np.asarray(values)


def test_guided_reference_is_shared_unique_base_and_leave_one_base_out() -> None:
    rows, values = _reference_fixture()
    arrays, references, reference_ids, metadata = build_guided_reference_scores(
        rows, values
    )
    # cal-0-1 has r=2.  Removing itself leaves [1], so its true-class tail
    # probability is (1 + 0) / (1 + 1) = 0.5.
    assert arrays["known_calibration_p"][1, 0] == pytest.approx(0.5)
    assert [len(reference) for reference in references] == [2] * 5
    assert reference_ids[0] == ("cal-0-0", "cal-0-1")
    assert metadata["shared_reference_across_slots"] is True
    assert metadata["surrogate_unknown_in_reference"] is False
    assert all("surrogate" not in sample_id for ids in reference_ids for sample_id in ids)


def test_guided_reference_rejects_repeated_evaluation_base() -> None:
    rows, values = _reference_fixture()
    rows[-1]["sample_id"] = rows[-2]["sample_id"]
    with pytest.raises(DataValidationError, match="repeats a sample ID"):
        build_guided_reference_scores(rows, values)


def test_guided_score_uses_r2_prediction_and_is_view_swap_invariant() -> None:
    logits = np.asarray([[0.0, 3.0, 1.0, 0.0, -1.0], [0.0, 1.0, 0.0, 4.0, 2.0]])
    anomaly = np.arange(2 * 2 * 5, dtype=np.float64).reshape(2, 2, 5)
    original = guided_scores_from_r2_predictions(logits, anomaly)
    swapped = guided_scores_from_r2_predictions(logits, anomaly[:, ::-1])
    assert np.array_equal(original["known_prediction"], np.asarray([1, 3]))
    assert original["unknown_score"][0] == pytest.approx((1.0 + 6.0) / 2.0)
    assert np.array_equal(original["known_prediction"], swapped["known_prediction"])
    assert np.array_equal(original["unknown_score"], swapped["unknown_score"])


def _mls_fixture() -> tuple[np.ndarray, np.ndarray, list[str]]:
    logits: list[np.ndarray] = []
    labels: list[int] = []
    pair_ids: list[str] = []
    for label in range(5):
        for index, maximum in enumerate((5.0, 3.0)):
            row = np.full(5, -5.0)
            row[label] = maximum
            logits.append(row)
            labels.append(label)
            pair_ids.append(f"pair-{label}-{index}")
    return np.asarray(logits), np.asarray(labels), pair_ids


def test_d0_class_conditional_mls_and_role_wrapper_use_known_only_loo() -> None:
    logits, labels, pair_ids = _mls_fixture()
    scores = class_conditional_mls_score(
        logits[:1],
        calibration_logits=logits,
        calibration_true_labels=labels,
        query_pair_ids=pair_ids[:1],
        calibration_pair_ids=pair_ids,
        leave_one_pair_out=True,
    )
    assert scores[0] == pytest.approx(0.5)

    by_role = class_conditional_mls_for_roles(
        full_calibration_logits=logits,
        full_calibration_labels=labels,
        full_calibration_pair_ids=pair_ids,
        role_logits={
            "known_calibration": logits,
            "surrogate_unknown": logits[:2],
        },
        role_pair_ids={
            "known_calibration": pair_ids,
            "surrogate_unknown": ["surrogate-0", "surrogate-1"],
        },
    )
    assert by_role["known_calibration"].shape == (10,)
    assert by_role["surrogate_unknown"].shape == (2,)


def test_d0_d1_d2_must_preserve_exact_r2_logits_predictions_and_known_metrics() -> None:
    logits, labels, _ = _mls_fixture()
    audit = audit_shared_r2_predictions(
        {method: logits.copy() for method in METHODS}, labels
    )
    assert audit["logits_exactly_equal"] is True
    assert audit["known_predictions_exactly_equal"] is True
    assert audit["known_accuracy"] == 1.0

    changed = {method: logits.copy() for method in METHODS}
    changed[D2_DECOUPLED_ABSREL_CSSR][0, 0] += 1.0e-8
    with pytest.raises(DataValidationError, match="changed frozen R2 fused logits"):
        audit_shared_r2_predictions(changed, labels)


def _prediction_rows(method: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label in range(5):
        for index in range(2):
            rows.append(
                {
                    "method": method,
                    "evaluation_role": "known_calibration",
                    "true_label": label,
                    "predicted_known_label": label,
                    "unknown_score": 0.01 * (2 * label + index),
                    "class_name": f"known-{label}",
                }
            )
    for identity, predictions, scores in (
        ("DDG-112", (1, 1, 2), (0.02, 0.20, 0.30)),
        ("迷你好望角型散货船", (3, 4, 4), (0.03, 0.25, 0.35)),
    ):
        for prediction, score in zip(predictions, scores, strict=True):
            rows.append(
                {
                    "method": method,
                    "evaluation_role": "surrogate_unknown",
                    "true_label": -1,
                    "predicted_known_label": prediction,
                    "unknown_score": score,
                    "class_name": identity,
                }
            )
    return rows


def test_nine_metrics_identity_and_ddg_absorption_interfaces() -> None:
    rows = _prediction_rows(D0_R2_CLASS_CONDITIONAL_MLS)
    metrics = recompute_metrics_from_prediction_rows(rows)
    assert set(REPORT_METRIC_KEYS) <= set(metrics)
    identity, absorption, audit = build_identity_and_absorption_rows(
        rows,
        method=D0_R2_CLASS_CONDITIONAL_MLS,
        pair_id="N1",
        train_class_order=("A", "DDG-1000", "B", "C", "D"),
    )
    assert len(identity) == 2
    assert len(absorption) == 10
    assert all(set(REPORT_METRIC_KEYS) <= set(row) for row in identity)
    counts = ddg_false_accept_counts(
        absorption, method=D0_R2_CLASS_CONDITIONAL_MLS, allow_missing=True
    )
    assert counts["DDG-112_absorbed_as_DDG-1000"] == 1
    assert counts["DDG-1000_absorbed_as_DDG-112"] is None
    assert audit["all_known_destinations_reported"] is True


_IDENTITIES = {
    "N1": ("DDG-112", "迷你好望角型散货船"),
    "N4": ("DDG-1000", "集装箱船达飞罗尔多夫级"),
    "N2": ("油气轮MARVEL CRANE", "迷你好望角型散货船"),
    "N0": ("CVN77", "DDG-112"),
    "N3": ("DDG-1000", "油气轮MARVEL CRANE"),
    "N5": ("爱达魔都号", "集装箱船达飞罗尔多夫级"),
    "N6": ("CVN77", "爱达魔都号"),
}


def _gate_rows(
    pair_ids: tuple[str, ...],
    methods: tuple[str, ...],
    deltas: dict[str, float],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metrics: list[dict[str, object]] = []
    identities: list[dict[str, object]] = []
    for pair_id in pair_ids:
        for method in methods:
            delta = deltas.get(method, 0.0)
            metrics.append(
                {
                    "pair_id": pair_id,
                    "method": method,
                    "auroc": 0.60 + delta,
                    "oscr": 0.50 + delta,
                    "known_correct_acceptance_rate": 0.90,
                    "fpr95": 0.20 - delta,
                }
            )
            for identity in _IDENTITIES[pair_id]:
                identities.append(
                    {
                        "pair_id": pair_id,
                        "method": method,
                        "surrogate_identity": identity,
                        "auroc": 0.60 + delta,
                    }
                )
    return metrics, identities


def _pilot_absorption(
    d1_count: int = 4,
    d2_count: int = 3,
) -> list[dict[str, object]]:
    counts = {
        D0_R2_CLASS_CONDITIONAL_MLS: 5,
        D1_DECOUPLED_REL_CSSR: d1_count,
        D2_DECOUPLED_ABSREL_CSSR: d2_count,
    }
    rows: list[dict[str, object]] = []
    for method, count in counts.items():
        rows.extend(
            [
                {
                    "pair_id": "N1",
                    "method": method,
                    "surrogate_identity": "DDG-112",
                    "absorbed_as_known_identity": "DDG-1000",
                    "false_accept_count": count,
                },
                {
                    "pair_id": "N4",
                    "method": method,
                    "surrogate_identity": "DDG-1000",
                    "absorbed_as_known_identity": "DDG-112",
                    "false_accept_count": count,
                },
            ]
        )
    return rows


def test_pilot_gate_prefers_d1_unless_d2_adds_two_points() -> None:
    metrics, identities = _gate_rows(
        PILOT_PAIRS,
        METHODS,
        {D1_DECOUPLED_REL_CSSR: 0.03, D2_DECOUPLED_ABSREL_CSSR: 0.04},
    )
    d1 = evaluate_pilot_gate(metrics, identities, _pilot_absorption())
    assert d1["status"] == "evaluated"
    assert d1["selected_method"] == D1_DECOUPLED_REL_CSSR
    assert d1["signal"] == "decoupled_relative_signal"
    assert d1["confirmation_allowed"] is True

    metrics, identities = _gate_rows(
        PILOT_PAIRS,
        METHODS,
        {D1_DECOUPLED_REL_CSSR: 0.03, D2_DECOUPLED_ABSREL_CSSR: 0.051},
    )
    d2 = evaluate_pilot_gate(metrics, identities, _pilot_absorption())
    assert d2["selected_method"] == D2_DECOUPLED_ABSREL_CSSR
    assert d2["signal"] == "decoupled_absolute_alignment_signal"


def test_pilot_gate_selects_d2_when_d1_fails_and_enforces_ddg_directions() -> None:
    metrics, identities = _gate_rows(
        PILOT_PAIRS,
        METHODS,
        {D1_DECOUPLED_REL_CSSR: 0.0, D2_DECOUPLED_ABSREL_CSSR: 0.03},
    )
    result = evaluate_pilot_gate(metrics, identities, _pilot_absorption())
    assert result["selected_method"] == D2_DECOUPLED_ABSREL_CSSR

    metrics, identities = _gate_rows(
        PILOT_PAIRS,
        METHODS,
        {D1_DECOUPLED_REL_CSSR: 0.03, D2_DECOUPLED_ABSREL_CSSR: 0.0},
    )
    worsened = evaluate_pilot_gate(
        metrics, identities, _pilot_absorption(d1_count=6)
    )
    assert worsened["selected_method"] is None
    assert worsened["signal"] == "decoupled_cssr_failed"
    checks = worsened["candidate_gates"][D1_DECOUPLED_REL_CSSR]["checks"]
    assert checks["ddg_112_to_1000_not_worse"] is False
    assert checks["ddg_1000_to_112_not_worse"] is False


def test_incomplete_pilot_is_not_evaluated_and_never_authorizes_confirmation() -> None:
    metrics, identities = _gate_rows(
        PILOT_PAIRS,
        METHODS,
        {D1_DECOUPLED_REL_CSSR: 0.03, D2_DECOUPLED_ABSREL_CSSR: 0.04},
    )
    result = evaluate_pilot_gate(metrics[:-1], identities, _pilot_absorption())
    assert result["status"] == "not_evaluated"
    assert result["selected_method"] is None
    assert result["confirmation_allowed"] is False
    assert result["final_unknown_test_authorized"] is False
    assert result["missing"]


def test_phase_plan_is_bounded_and_confirmation_is_conditional() -> None:
    smoke = build_phase_plan("smoke")
    pilot = build_phase_plan("pilot")
    confirmation = build_phase_plan(
        "confirmation", selected_method=D1_DECOUPLED_REL_CSSR
    )
    assert [(row["pair_id"], row["method"]) for row in smoke] == [
        ("N1", method) for method in TRAINABLE_METHODS
    ]
    assert len(pilot) == 6
    assert tuple(dict.fromkeys(str(row["pair_id"]) for row in pilot)) == PILOT_PAIRS
    assert len(confirmation) == 4
    assert tuple(str(row["pair_id"]) for row in confirmation) == CONFIRMATION_PAIRS
    assert {row["method"] for row in confirmation} == {D1_DECOUPLED_REL_CSSR}
    assert all(row["final_unknown_test_authorized"] is False for row in confirmation)
    with pytest.raises(DataValidationError, match="requires an audited selected"):
        build_phase_plan("confirmation")


def test_confirmation_gate_passes_rejects_and_refuses_incomplete_evidence() -> None:
    methods = (D0_R2_CLASS_CONDITIONAL_MLS, D1_DECOUPLED_REL_CSSR)
    metrics, identities = _gate_rows(
        CONFIRMATION_PAIRS, methods, {D1_DECOUPLED_REL_CSSR: 0.03}
    )
    passed = evaluate_confirmation_gate(
        metrics, identities, D1_DECOUPLED_REL_CSSR
    )
    assert passed["decision"] == "decoupled_cssr_worth_full_validation"
    assert passed["final_unknown_test_authorized"] is False

    failed_metrics, failed_identities = _gate_rows(
        CONFIRMATION_PAIRS, methods, {D1_DECOUPLED_REL_CSSR: 0.01}
    )
    failed = evaluate_confirmation_gate(
        failed_metrics, failed_identities, D1_DECOUPLED_REL_CSSR
    )
    assert failed["decision"] == "decoupled_cssr_rejected"
    assert failed["final_unknown_test_authorized"] is False

    incomplete = evaluate_confirmation_gate(
        metrics[:-1], identities, D1_DECOUPLED_REL_CSSR
    )
    assert incomplete["status"] == "not_evaluated"
    assert incomplete["confirmation_allowed"] is False
    assert incomplete["final_unknown_test_authorized"] is False
