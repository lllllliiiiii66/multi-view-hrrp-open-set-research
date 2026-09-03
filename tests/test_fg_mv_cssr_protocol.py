from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pytest
import yaml

from hrrp_osr.data.errors import DataConfigError, DataValidationError
from hrrp_osr.training.fg_mv_cssr_pilot import (
    build_phase_plan,
    compute_b0_b4_scores,
    compute_class_conditional_mls_scores,
    compute_conformal_p_values,
    evaluate_confirmation_gate,
    evaluate_pilot_gate,
    load_fg_mv_cssr_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/experiments/cssr/fg_mv_cssr_frozen_r2_v1.yaml"
)

B0 = "B0_GLOBAL_MLS"
B1 = "B1_CLASS_CONDITIONAL_MLS"
B2 = "B2_INDEPENDENT_VIEW_CSSR"
B3 = "B3_COMMON_CLASS_CSSR"
B4 = "B4_FUSION_GUIDED_CSSR"

PILOT_PAIRS = ("N1", "N4", "N2")
CONFIRMATION_PAIRS = ("N0", "N3", "N5", "N6")


def _write_changed_config(tmp_path: Path, config: Mapping[str, Any]) -> Path:
    path = tmp_path / "changed.yaml"
    path.write_text(
        yaml.safe_dump(dict(config), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def test_config_freezes_r2_cssr_scope_and_known_only_boundaries() -> None:
    config = load_fg_mv_cssr_config(CONFIG_PATH)

    assert config["experiment_id"] == "fg_mv_cssr_frozen_r2_v1"
    assert config["prior_r2"]["method"] == "R2_MS_MEAN_CE"
    assert config["prior_r2"]["angle_fold"] == 0
    assert config["prior_r2"]["initialization_seed"] == 20260830
    assert config["prior_r2"]["checkpoint_epoch"] == 100
    assert config["prior_r2"]["strict_load_required"] is True
    assert config["prior_r2"]["old_logits_exact_match_required"] is True
    assert config["classes"]["pilot_pairs"] == list(PILOT_PAIRS)
    assert config["classes"]["confirmation_pairs"] == list(CONFIRMATION_PAIRS)

    scope = config["evidence_scope"]
    assert scope["source_known_odd_angle_only"] is True
    for key in (
        "final_unknown_classes_used",
        "even_angle_test_used",
        "surrogate_unknown_used_for_cssr_training",
        "surrogate_unknown_used_for_reference_distribution",
        "surrogate_unknown_used_for_threshold",
        "known_calibration_used_for_cssr_training",
        "r2_retrained_or_finetuned",
        "arpl_used",
        "pseudo_unknown_used",
        "angle_metadata_used_by_model",
    ):
        assert scope[key] is False

    assert config["data"]["final_test_pairs_generated"] is False
    assert config["cssr_training"]["epochs"] == 30
    assert config["cssr_training"]["formal_checkpoint_epoch"] == 30
    assert config["cssr_training"]["early_stopping"] is False
    assert config["pcssr_core_1d"]["gamma"] == pytest.approx(0.1)
    assert config["calibration"]["threshold_known_acceptance_rate"] == pytest.approx(
        0.95
    )
    assert config["scores"] == [B0, B1, B2, B3, B4]
    assert config["decisions"]["final_unknown_test_authorized"] is False


@pytest.mark.parametrize(
    "mutate",
    (
        lambda config: config["evidence_scope"].update(
            final_unknown_classes_used=True
        ),
        lambda config: config["evidence_scope"].update(even_angle_test_used=True),
        lambda config: config["evidence_scope"].update(
            surrogate_unknown_used_for_reference_distribution=True
        ),
        lambda config: config["evidence_scope"].update(
            known_calibration_used_for_cssr_training=True
        ),
        lambda config: config["evidence_scope"].update(
            r2_retrained_or_finetuned=True
        ),
        lambda config: config["data"].update(final_test_pairs_generated=True),
        lambda config: config["prior_r2"].update(checkpoint_epoch=70),
        lambda config: config["pcssr_core_1d"].update(gamma=0.2),
        lambda config: config["classes"].update(pilot_pairs=["N1", "N2", "N4"]),
        lambda config: config["classes"].update(
            confirmation_pairs=["N0", "N3", "N5", "N1"]
        ),
        lambda config: config["calibration"].update(
            threshold_known_acceptance_rate=0.90
        ),
        lambda config: config["decisions"].update(
            final_unknown_test_authorized=True
        ),
    ),
)
def test_config_rejects_frozen_protocol_mutations(
    tmp_path: Path, mutate: Callable[[dict[str, Any]], None]
) -> None:
    config = copy.deepcopy(yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")))
    mutate(config)
    with pytest.raises(DataConfigError):
        load_fg_mv_cssr_config(_write_changed_config(tmp_path, config))


def test_phase_plans_are_exact_deterministic_and_disjoint() -> None:
    config = load_fg_mv_cssr_config(CONFIG_PATH)

    smoke = build_phase_plan(config, "smoke")
    pilot = build_phase_plan(config, "pilot")
    confirmation = build_phase_plan(config, "confirmation")

    assert smoke == [
        {
            "phase": "smoke",
            "mode": "smoke",
            "pair_id": "N1",
            "angle_fold": 0,
            "r2_seed": 20260830,
            "cssr_seed": 20260903,
        }
    ]
    assert [unit["pair_id"] for unit in pilot] == list(PILOT_PAIRS)
    assert [unit["pair_id"] for unit in confirmation] == list(CONFIRMATION_PAIRS)
    assert all(unit["phase"] == "pilot" and unit["mode"] == "full" for unit in pilot)
    assert all(
        unit["phase"] == "confirmation" and unit["mode"] == "full"
        for unit in confirmation
    )
    assert not set(PILOT_PAIRS) & set(CONFIRMATION_PAIRS)
    assert pilot == build_phase_plan(copy.deepcopy(config), "pilot")
    assert confirmation == build_phase_plan(copy.deepcopy(config), "confirmation")
    for unit in [*pilot, *confirmation]:
        assert unit["angle_fold"] == 0
        assert unit["r2_seed"] == 20260830
        assert unit["cssr_seed"] == 20260903

    with pytest.raises(DataValidationError, match="phase"):
        build_phase_plan(config, "final_test")


def test_conformal_p_values_use_greater_equal_tail_and_plus_one_smoothing() -> None:
    query = np.array([0.0, 2.0, 4.0], dtype=np.float64)
    reference = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    observed = compute_conformal_p_values(query, reference)

    np.testing.assert_array_equal(observed, np.array([1.0, 0.75, 0.25]))
    assert np.all(np.diff(observed) < 0.0)
    anomaly = -np.log(observed + 1.0e-8)
    assert np.all(np.diff(anomaly) > 0.0)


def test_conformal_leave_one_out_removes_matching_base_id_once() -> None:
    reference = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    reference_ids = ("base-a", "base-b", "base-c")

    ordinary = compute_conformal_p_values(np.array([2.0]), reference)
    leave_one_out = compute_conformal_p_values(
        np.array([2.0]),
        reference,
        query_sample_ids=("base-b",),
        reference_sample_ids=reference_ids,
        leave_one_out=True,
    )

    np.testing.assert_array_equal(ordinary, np.array([0.75]))
    np.testing.assert_allclose(leave_one_out, np.array([2.0 / 3.0]))
    with pytest.raises(DataValidationError, match="sample.*id|ID"):
        compute_conformal_p_values(
            np.array([2.0]), reference, leave_one_out=True
        )


def test_class_conditional_mls_uses_correct_known_references_and_lower_tail() -> None:
    # The last reference is deliberately a wrong known prediction.  Its very
    # small value must not enter the true-class reference distribution.
    reference_nonconformity = np.array([-5.0, -3.0, -4.0, -2.0, -100.0])
    reference_true = np.array([0, 0, 1, 1, 0])
    reference_predicted = np.array([0, 0, 1, 1, 1])

    observed = compute_class_conditional_mls_scores(
        np.array([-4.0, -3.0, 0.0]),
        np.array([0, 1, 0]),
        reference_nonconformity=reference_nonconformity,
        reference_true_labels=reference_true,
        reference_predicted_labels=reference_predicted,
    )

    np.testing.assert_allclose(observed, np.array([2.0 / 3.0, 2.0 / 3.0, 1.0]))
    assert observed[2] > observed[0]


def test_class_conditional_mls_leave_one_pair_out_and_empty_reference_fail() -> None:
    reference_nonconformity = np.array([-5.0, -3.0, -4.0, -2.0])
    reference_labels = np.array([0, 0, 1, 1])
    reference_ids = ("pair-a", "pair-b", "pair-c", "pair-d")

    leave_one_out = compute_class_conditional_mls_scores(
        np.array([-5.0]),
        np.array([0]),
        reference_nonconformity=reference_nonconformity,
        reference_true_labels=reference_labels,
        reference_predicted_labels=reference_labels,
        query_pair_ids=("pair-a",),
        reference_pair_ids=reference_ids,
        leave_one_out=True,
    )
    np.testing.assert_array_equal(leave_one_out, np.array([0.5]))

    with pytest.raises(DataValidationError, match="empty.*reference|reference.*empty"):
        compute_class_conditional_mls_scores(
            np.array([-1.0]),
            np.array([2]),
            reference_nonconformity=reference_nonconformity,
            reference_true_labels=reference_labels,
            reference_predicted_labels=reference_labels,
        )


def test_b0_b4_scores_match_frozen_formulas_without_changing_r2_prediction() -> None:
    logits = np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0]])
    view_class_anomaly = np.array(
        [
            [[1.0, 4.0, 2.0], [3.0, 2.0, 6.0]],
            [[4.0, 1.0, 5.0], [2.0, 7.0, 3.0]],
        ]
    )
    b1_scores = np.array([0.7, 0.8])

    scores = compute_b0_b4_scores(logits, view_class_anomaly, b1_scores)

    np.testing.assert_array_equal(scores["known_prediction"], np.array([0, 1]))
    np.testing.assert_array_equal(scores[B0], np.array([-2.0, -3.0]))
    np.testing.assert_array_equal(scores[B1], b1_scores)
    np.testing.assert_array_equal(scores[B2], np.array([1.5, 1.5]))
    np.testing.assert_array_equal(scores[B3], np.array([2.0, 3.0]))
    np.testing.assert_array_equal(scores[B4], np.array([2.0, 4.0]))
    np.testing.assert_array_equal(scores["k_common"], np.array([0, 0]))
    assert scores["known_prediction"][1] != scores["k_common"][1]
    assert np.all(scores[B2] <= scores[B3])
    assert np.all(scores[B3] <= scores[B4])


def _gate_rows(
    pair_ids: Sequence[str],
    *,
    b3_auroc_delta: Sequence[float],
    b4_auroc_delta: Sequence[float],
    b3_oscr_delta: float = 0.0,
    b4_oscr_delta: float = 0.0,
    b3_kccr_delta: float = 0.0,
    b4_kccr_delta: float = 0.0,
    b3_fpr95_delta: float = 0.0,
    b4_fpr95_delta: float = 0.0,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, pair_id in enumerate(pair_ids):
        baseline = {
            "pair_id": pair_id,
            "method": B1,
            "auroc": 0.50,
            "oscr": 0.50,
            "known_correct_acceptance_rate": 0.80,
            "fpr95": 0.40,
        }
        rows.append(baseline)
        for method, auroc_delta, oscr_delta, kccr_delta, fpr95_delta in (
            (
                B3,
                b3_auroc_delta[index],
                b3_oscr_delta,
                b3_kccr_delta,
                b3_fpr95_delta,
            ),
            (
                B4,
                b4_auroc_delta[index],
                b4_oscr_delta,
                b4_kccr_delta,
                b4_fpr95_delta,
            ),
        ):
            rows.append(
                {
                    "pair_id": pair_id,
                    "method": method,
                    "auroc": baseline["auroc"] + auroc_delta,
                    "oscr": baseline["oscr"] + oscr_delta,
                    "known_correct_acceptance_rate": baseline[
                        "known_correct_acceptance_rate"
                    ]
                    + kccr_delta,
                    "fpr95": baseline["fpr95"] + fpr95_delta,
                }
            )
    return rows


def test_pilot_gate_is_inclusive_and_b4_wins_an_auroc_tie() -> None:
    config = load_fg_mv_cssr_config(CONFIG_PATH)
    rows = _gate_rows(
        PILOT_PAIRS,
        b3_auroc_delta=(0.02, 0.02, 0.02),
        b4_auroc_delta=(0.01, 0.02, 0.03),
        b3_kccr_delta=-0.01,
        b4_kccr_delta=-0.01,
        b3_fpr95_delta=0.02,
        b4_fpr95_delta=0.02,
    )

    result = evaluate_pilot_gate(rows, config)
    reversed_result = evaluate_pilot_gate(list(reversed(rows)), config)

    assert result == reversed_result
    assert result["signal"] == "fusion_guided_signal"
    assert result["selected_rule"] == B4
    assert result["confirmation_allowed"] is True
    assert result["candidates"][B3]["passed"] is True
    assert result["candidates"][B4]["passed"] is True
    assert result["candidates"][B4]["positive_pair_count"] == 3
    assert result["candidates"][B4]["mean_auroc_delta"] == pytest.approx(0.02)


def test_pilot_gate_selects_b3_only_when_its_passing_auroc_is_higher() -> None:
    config = load_fg_mv_cssr_config(CONFIG_PATH)
    rows = _gate_rows(
        PILOT_PAIRS,
        b3_auroc_delta=(0.03, 0.03, 0.03),
        b4_auroc_delta=(0.02, 0.02, 0.02),
    )

    result = evaluate_pilot_gate(rows, config)

    assert result["signal"] == "common_class_signal"
    assert result["selected_rule"] == B3
    assert result["confirmation_allowed"] is True


@pytest.mark.parametrize(
    ("b4_deltas", "b4_oscr", "b4_kccr", "b4_fpr95"),
    (
        ((0.01, 0.01, 0.01), 0.0, 0.0, 0.0),
        ((0.06, 0.00, 0.00), 0.0, 0.0, 0.0),
        ((0.03, 0.03, 0.03), -0.001, 0.0, 0.0),
        ((0.03, 0.03, 0.03), 0.0, -0.011, 0.0),
        ((0.03, 0.03, 0.03), 0.0, 0.0, 0.021),
    ),
)
def test_pilot_gate_conservatively_stops_when_a_required_condition_fails(
    b4_deltas: Sequence[float],
    b4_oscr: float,
    b4_kccr: float,
    b4_fpr95: float,
) -> None:
    config = load_fg_mv_cssr_config(CONFIG_PATH)
    rows = _gate_rows(
        PILOT_PAIRS,
        b3_auroc_delta=(0.0, 0.0, 0.0),
        b4_auroc_delta=b4_deltas,
        b4_oscr_delta=b4_oscr,
        b4_kccr_delta=b4_kccr,
        b4_fpr95_delta=b4_fpr95,
    )

    result = evaluate_pilot_gate(rows, config)

    assert result["signal"] == "no_cssr_signal"
    assert result["selected_rule"] is None
    assert result["confirmation_allowed"] is False


def test_confirmation_gate_passes_all_inclusive_boundaries() -> None:
    config = load_fg_mv_cssr_config(CONFIG_PATH)
    rows = _gate_rows(
        CONFIRMATION_PAIRS,
        b3_auroc_delta=(0.03, 0.03, 0.02, 0.00),
        b4_auroc_delta=(0.03, 0.03, 0.02, 0.00),
        b3_kccr_delta=-0.01,
        b4_kccr_delta=-0.01,
        b3_fpr95_delta=0.02,
        b4_fpr95_delta=0.02,
    )

    result = evaluate_confirmation_gate(rows, B4, config)

    assert result["passed"] is True
    assert result["selected_rule"] == B4
    assert result["positive_pair_count"] == 3
    assert result["mean_auroc_delta"] == pytest.approx(0.02)
    assert result["decision"] == "worth_later_full_validation_only"


def test_confirmation_gate_fails_with_only_two_positive_pairs() -> None:
    config = load_fg_mv_cssr_config(CONFIG_PATH)
    rows = _gate_rows(
        CONFIRMATION_PAIRS,
        b3_auroc_delta=(0.05, 0.05, 0.00, 0.00),
        b4_auroc_delta=(0.05, 0.05, 0.00, 0.00),
    )

    result = evaluate_confirmation_gate(rows, B4, config)

    assert result["mean_auroc_delta"] == pytest.approx(0.025)
    assert result["positive_pair_count"] == 2
    assert result["passed"] is False
    assert result["decision"] == "stop_cssr_route"


@pytest.mark.parametrize("selected_rule", (B0, B1, B2, "UNKNOWN"))
def test_confirmation_gate_rejects_nonselected_score_rules(selected_rule: str) -> None:
    config = load_fg_mv_cssr_config(CONFIG_PATH)
    rows = _gate_rows(
        CONFIRMATION_PAIRS,
        b3_auroc_delta=(0.03, 0.03, 0.03, 0.03),
        b4_auroc_delta=(0.03, 0.03, 0.03, 0.03),
    )
    with pytest.raises(DataValidationError, match="B3|B4|selected"):
        evaluate_confirmation_gate(rows, selected_rule, config)


def test_gates_reject_missing_duplicate_or_wrong_phase_pairs() -> None:
    config = load_fg_mv_cssr_config(CONFIG_PATH)
    pilot_rows = _gate_rows(
        PILOT_PAIRS,
        b3_auroc_delta=(0.03, 0.03, 0.03),
        b4_auroc_delta=(0.03, 0.03, 0.03),
    )
    with pytest.raises(DataValidationError, match="pair|row"):
        evaluate_pilot_gate(pilot_rows[:-1], config)
    with pytest.raises(DataValidationError, match="duplicate|pair|row"):
        evaluate_pilot_gate([*pilot_rows, pilot_rows[-1]], config)

    wrong_confirmation = _gate_rows(
        ("N0", "N3", "N5", "N1"),
        b3_auroc_delta=(0.03, 0.03, 0.03, 0.03),
        b4_auroc_delta=(0.03, 0.03, 0.03, 0.03),
    )
    with pytest.raises(DataValidationError, match="pair"):
        evaluate_confirmation_gate(wrong_confirmation, B4, config)
