from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pytest
import yaml

from hrrp_osr.data.errors import DataConfigError, DataValidationError
from hrrp_osr.training.fg_mv_cssr_e2e_redesign import (
    CONFIRMATION_PAIRS,
    CSSR_METHODS,
    METHODS,
    PILOT_PAIRS,
    TRAINABLE_METHODS,
    build_epoch_pair_schedule,
    build_guided_reference_scores,
    build_phase_plan,
    compute_method_scores,
    evaluate_confirmation_gate,
    evaluate_pilot_gate,
    load_fg_mv_cssr_e2e_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/experiments/cssr/fg_mv_cssr_e2e_redesign_v2.yaml"
)

Q0 = "Q0_FROZEN_R2_CC_MLS"
Q1 = "Q1_CE_FINETUNE_CONTROL"
Q2 = "Q2_E2E_REL_CSSR_1X1"
Q3 = "Q3_E2E_ABSREL_CSSR_1X1"
Q4 = "Q4_E2E_ABSREL_CSSR_LOCAL3"


def _write_changed_config(tmp_path: Path, config: Mapping[str, Any]) -> Path:
    path = tmp_path / "changed.yaml"
    path.write_text(
        yaml.safe_dump(dict(config), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def test_config_freezes_the_complete_e2e_redesign_contract() -> None:
    config = load_fg_mv_cssr_e2e_config(CONFIG_PATH)

    assert config["experiment_id"] == "fg_mv_cssr_e2e_redesign_v2"
    assert config["stage"] == "P3_fg_mv_cssr_e2e_redesign_fast_iteration"
    assert tuple(config["methods"]["ordered"]) == METHODS == (Q0, Q1, Q2, Q3, Q4)
    assert tuple(config["methods"]["trainable"]) == TRAINABLE_METHODS == (
        Q1,
        Q2,
        Q3,
        Q4,
    )
    assert tuple(config["methods"]["cssr_candidates"]) == CSSR_METHODS == (
        Q2,
        Q3,
        Q4,
    )
    assert tuple(config["classes"]["pilot_pairs"]) == PILOT_PAIRS
    assert tuple(config["classes"]["confirmation_pairs"]) == CONFIRMATION_PAIRS
    assert config["prior_r2"]["method"] == "R2_MS_MEAN_CE"
    assert config["prior_r2"]["initialization_seed"] == 20260830
    assert config["prior_r2"]["checkpoint_epoch"] == 100
    assert config["prior_r2"]["strict_load_required"] is True

    scope = config["evidence_scope"]
    assert scope["source_known_odd_angle_only"] is True
    for field in (
        "final_unknown_classes_used",
        "even_angle_test_used",
        "surrogate_unknown_used_for_training",
        "surrogate_unknown_used_for_reference_distribution",
        "surrogate_unknown_used_for_threshold",
        "known_calibration_used_for_training",
        "arpl_used",
        "pseudo_unknown_used",
        "angle_metadata_used_by_model",
    ):
        assert scope[field] is False

    data = config["data"]
    assert data["angle_fold"] == 0
    assert data["development_angle_parity"] == "odd"
    assert data["train_unique_base_samples_per_class"] == 144
    assert data["known_calibration_unique_base_samples_per_class"] == 36
    assert data["final_test_pairs_generated"] is False
    assert data["smoke"] == {
        "pair_id": "N1",
        "methods": [Q1, Q2, Q3, Q4],
        "epochs": 1,
        "full_train_unique_base_schedule": True,
        "evaluation_pairs_per_class": 2,
        "diagnostic_only": True,
    }
    schedule = data["dynamic_pair_schedule"]
    assert schedule["algorithm"] == "deterministic_cross_frame_derangement_v1"
    assert schedule["random_generator"] == "numpy_PCG64"
    assert schedule["matching"] == "deterministic_augmenting_path_bipartite"
    assert schedule["maximum_matching_attempts"] == 4096
    assert schedule["unordered_pair_unique_within_class_epoch"] is True
    assert schedule["view1_usage_per_base_per_epoch"] == 1
    assert schedule["view2_usage_per_base_per_epoch"] == 1
    assert schedule["dataloader_shuffle"] is False

    assert config["trainable_scope"]["frozen_modules"] == [
        "encoder.stem",
        "encoder.stages.0",
        "encoder.stages.1",
    ]
    assert config["trainable_scope"]["frozen_modules_forced_eval"] is True
    assert config["trainable_scope"]["reset_rng_after_model_and_optimizer_construction"] is True
    assert config["autoencoders"]["q2_q3_kernel_size"] == 1
    assert config["autoencoders"]["q4_kernel_size"] == 3
    assert config["autoencoders"]["q4_padding"] == 1
    assert config["loss"]["separation"]["margin"] == pytest.approx(0.2)
    assert config["loss"]["weights"][Q2] == {
        "classification": 1.0,
        "relative": 0.5,
        "absolute": 0.0,
        "separation": 0.0,
    }
    assert config["loss"]["weights"][Q3] == config["loss"]["weights"][Q4]
    assert config["loss"]["gradient_audit"]["maximum_ratio"] == pytest.approx(100.0)
    assert config["loss"]["gradient_audit"]["consecutive_epoch_mean_violations_to_fail"] == 3

    training = config["training"]
    assert training["finetune_seed"] == 20260904
    assert training["epochs"] == 20
    assert training["batch_size_pairs"] == 64
    assert training["warmup_epochs"] == 2
    assert training["scheduler"] == "linear_warmup_then_cosine_to_zero"
    assert training["early_stopping"] is False
    assert training["formal_checkpoint_epoch"] == 20
    assert training["performance_checkpoint_selection"] is False
    assert training["diagnostic_epochs"] == [0, 5, 10, 15, 20]

    assert config["calibration"]["cssr_reference_shared_across_view_slots"] is True
    assert config["calibration"]["cssr_leave_one_base_sample_out"] is True
    assert config["calibration"]["threshold_source"] == "own_known_calibration_only"
    assert config["calibration"]["threshold_known_acceptance_rate"] == pytest.approx(0.95)
    assert config["scores"]["main_by_method"] == {
        Q0: "class_conditional_mls",
        Q1: "class_conditional_mls",
        Q2: "fusion_guided_reconstruction",
        Q3: "fusion_guided_reconstruction",
        Q4: "fusion_guided_reconstruction",
    }
    assert config["diagnostics"]["brier"] == "per_sample_sum_over_class_then_sample_mean"
    assert config["diagnostics"]["ece_bins"] == 15
    assert config["diagnostics"]["kl_direction"] == "epoch0_to_current"
    assert config["pilot_gate"]["minimum_identity_auroc"] == pytest.approx(0.40)
    assert config["confirmation_gate"]["minimum_positive_pair_count_vs_q1"] == 3
    assert config["runtime"]["expected_gpu_model"] == "NVIDIA GeForce RTX 4090"
    assert config["outputs"]["namespace"] == "artifacts/cssr/fg_mv_cssr_e2e_redesign_v2"
    assert config["decisions"] == {
        "final_unknown_test_authorized": False,
        "second_angle_fold_authorized": False,
        "extra_seed_authorized": False,
        "arpl_authorized": False,
        "automatic_followon_method_authorized": False,
    }


@pytest.mark.parametrize(
    "mutate",
    (
        lambda config: config["evidence_scope"].update(final_unknown_classes_used=True),
        lambda config: config["evidence_scope"].update(even_angle_test_used=True),
        lambda config: config["evidence_scope"].update(surrogate_unknown_used_for_training=True),
        lambda config: config["data"].update(angle_fold=1),
        lambda config: config["data"]["dynamic_pair_schedule"].update(
            algorithm="fixed_500_pairs"
        ),
        lambda config: config["data"]["dynamic_pair_schedule"].update(
            unordered_pair_unique_within_class_epoch=False
        ),
        lambda config: config["data"]["dynamic_pair_schedule"].update(
            maximum_matching_attempts=4095
        ),
        lambda config: config["data"]["smoke"].update(epochs=2),
        lambda config: config["classes"].update(pilot_pairs=["N1", "N2", "N4"]),
        lambda config: config["classes"].update(
            confirmation_pairs=["N0", "N3", "N5", "N1"]
        ),
        lambda config: config["prior_r2"].update(checkpoint_epoch=70),
        lambda config: config["trainable_scope"].update(frozen_modules_forced_eval=False),
        lambda config: config["autoencoders"].update(q4_kernel_size=5),
        lambda config: config["loss"]["weights"][Q3].update(absolute=0.5),
        lambda config: config["loss"]["gradient_audit"].update(maximum_ratio=1000.0),
        lambda config: config["training"].update(epochs=30),
        lambda config: config["training"].update(early_stopping=True),
        lambda config: config["calibration"].update(
            cssr_reference_shared_across_view_slots=False
        ),
        lambda config: config["calibration"].update(threshold_known_acceptance_rate=0.90),
        lambda config: config["scores"]["main_by_method"].update(
            Q2_E2E_REL_CSSR_1X1="class_conditional_mls"
        ),
        lambda config: config["pilot_gate"].update(minimum_mean_auroc_delta_vs_q1=0.01),
        lambda config: config["decisions"].update(final_unknown_test_authorized=True),
    ),
)
def test_config_rejects_any_preregistered_protocol_mutation(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    config = copy.deepcopy(yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")))
    mutate(config)
    with pytest.raises(DataConfigError):
        load_fg_mv_cssr_e2e_config(_write_changed_config(tmp_path, config))


def test_phase_plans_are_exact_and_never_contain_q0_training() -> None:
    config = load_fg_mv_cssr_e2e_config(CONFIG_PATH)

    smoke = build_phase_plan(config, "smoke")
    pilot = build_phase_plan(config, "pilot")
    confirmation = build_phase_plan(config, "confirmation", selected_method=Q3)

    assert [(unit["pair_id"], unit["method"]) for unit in smoke] == [
        ("N1", method) for method in TRAINABLE_METHODS
    ]
    assert [(unit["pair_id"], unit["method"]) for unit in pilot] == [
        (pair_id, method)
        for pair_id in PILOT_PAIRS
        for method in TRAINABLE_METHODS
    ]
    assert [(unit["pair_id"], unit["method"]) for unit in confirmation] == [
        (pair_id, method)
        for pair_id in CONFIRMATION_PAIRS
        for method in (Q1, Q3)
    ]
    assert len(smoke) == 4
    assert len(pilot) == 12
    assert len(confirmation) == 8
    assert all(unit["angle_fold"] == 0 for unit in [*smoke, *pilot, *confirmation])
    assert all(unit["model_seed"] == 20260830 for unit in [*smoke, *pilot, *confirmation])
    assert all(unit["finetune_seed"] == 20260904 for unit in [*smoke, *pilot, *confirmation])
    assert all(unit["method"] != Q0 for unit in [*smoke, *pilot, *confirmation])
    assert smoke == build_phase_plan(copy.deepcopy(config), "smoke")
    assert pilot == build_phase_plan(copy.deepcopy(config), "pilot")

    with pytest.raises(DataValidationError, match="selected|candidate|confirmation"):
        build_phase_plan(config, "confirmation")
    with pytest.raises(DataValidationError, match="selected|candidate|CSSR"):
        build_phase_plan(config, "confirmation", selected_method=Q1)
    with pytest.raises(DataValidationError, match="phase|final"):
        build_phase_plan(config, "final_test")


def _formal_train_unique_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_label in range(5):
        for frame_id in range(24):
            odd_angles = [
                angle
                for angle in range(frame_id * 15, frame_id * 15 + 15)
                if angle % 2 == 1
            ][:6]
            for within_frame, angle in enumerate(odd_angles):
                sample_id = f"class-{model_label}-frame-{frame_id}-base-{within_frame}"
                rows.append(
                    {
                        "experiment_role": "train_known",
                        "sample_id": sample_id,
                        "processed_row_index": len(rows),
                        "class_name": f"known-{model_label}",
                        "model_label": model_label,
                        "angle_deg": angle,
                        "frame_id": frame_id,
                    }
                )
    assert len(rows) == 5 * 144
    return rows


def test_epoch_pair_schedule_is_deterministic_cross_frame_and_exactly_balanced() -> None:
    unique_rows = _formal_train_unique_rows()
    kwargs = {
        "pair_id": "N1",
        "angle_fold": 0,
        "epoch": 1,
        "finetune_seed": 20260904,
    }

    rows, audit = build_epoch_pair_schedule(unique_rows, **kwargs)
    repeated_rows, repeated_audit = build_epoch_pair_schedule(unique_rows, **kwargs)
    next_epoch_rows, next_epoch_audit = build_epoch_pair_schedule(
        unique_rows,
        pair_id="N1",
        angle_fold=0,
        epoch=2,
        finetune_seed=20260904,
    )

    assert rows == repeated_rows
    assert audit == repeated_audit
    assert len(rows) == 720
    assert Counter(int(row["model_label"]) for row in rows) == Counter(
        {index: 144 for index in range(5)}
    )
    assert all(int(row["view1_frame_id"]) != int(row["view2_frame_id"]) for row in rows)
    assert all(row["view1_sample_id"] != row["view2_sample_id"] for row in rows)
    assert Counter(str(row["view1_sample_id"]) for row in rows) == Counter(
        {str(row["sample_id"]): 1 for row in unique_rows}
    )
    assert Counter(str(row["view2_sample_id"]) for row in rows) == Counter(
        {str(row["sample_id"]): 1 for row in unique_rows}
    )
    unordered = [
        (int(row["model_label"]), frozenset((row["view1_sample_id"], row["view2_sample_id"])))
        for row in rows
    ]
    assert len(unordered) == len(set(unordered))
    assert audit["all_constraints_passed"] is True
    assert audit["cross_frame"] is True
    assert audit["unordered_pair_unique"] is True
    assert all(0 <= int(row["matching_attempt"]) < 4096 for row in rows)
    assert audit["pair_count"] == 720
    assert audit["class_epoch_seeds"] == {
        "0": 16209363467369299594,
        "1": 16154408230741051923,
        "2": 12906226879913970204,
        "3": 5508868972370026702,
        "4": 17576268649687492106,
    }
    assert {
        label: {int(row["matching_attempt"]) for row in rows if row["model_label"] == label}
        for label in range(5)
    } == {0: {1}, 1: {0}, 2: {0}, 3: {0}, 4: {1}}
    assert (
        audit["epoch_manifest_sha256"]
        == "083a9df3b9fd268675876c4b364e5099a1507d08dcb8711bea4fae0bbc65e380"
    )
    assert rows != next_epoch_rows
    assert audit["epoch_manifest_sha256"] != next_epoch_audit["epoch_manifest_sha256"]


def test_epoch_pair_schedule_fails_instead_of_falling_back_when_infeasible() -> None:
    impossible = [
        {
            "experiment_role": "train_known",
            "sample_id": f"same-frame-{index}",
            "processed_row_index": index,
            "class_name": "known-0",
            "model_label": 0,
            "angle_deg": 1,
            "frame_id": 0,
        }
        for index in range(4)
    ]

    with pytest.raises(DataValidationError, match="match|frame|infeasible|derangement"):
        build_epoch_pair_schedule(
            impossible,
            pair_id="N1",
            angle_fold=0,
            epoch=1,
            finetune_seed=20260904,
        )


def _reference_fixture() -> tuple[list[dict[str, Any]], np.ndarray]:
    rows: list[dict[str, Any]] = []
    r_rows: list[list[float]] = []
    for class_index in range(5):
        for sample_index in range(36):
            rows.append(
                {
                    "experiment_role": "known_calibration",
                    "sample_id": f"cal-{class_index}-{sample_index}",
                    "model_label": class_index,
                    "class_name": f"known-{class_index}",
                }
            )
            r_rows.append([float(sample_index + 1)] * 5)
    for sample_index in range(2):
        rows.append(
            {
                "experiment_role": "surrogate_unknown",
                "sample_id": f"surrogate-{sample_index}",
                "model_label": 5,
                "class_name": f"surrogate-{sample_index}",
            }
        )
        r_rows.append([10_000.0] * 5)
    return rows, np.asarray(r_rows, dtype=np.float64)


def test_guided_reference_is_shared_unique_base_known_only_and_leave_one_out() -> None:
    unique_rows, r_values = _reference_fixture()

    arrays, references, reference_ids, metadata = build_guided_reference_scores(
        unique_rows,
        r_values,
        epsilon=1.0e-8,
    )

    assert metadata["shared_reference_across_slots"] is True
    assert metadata["surrogate_unknown_in_reference"] is False
    assert metadata["calibration_leave_one_base_sample_out"] is True
    assert metadata["reference_counts"] == [36] * 5
    for class_index in range(5):
        np.testing.assert_array_equal(references[class_index], np.arange(1.0, 37.0))
        assert len(reference_ids[class_index]) == len(set(reference_ids[class_index])) == 36
        assert all(value.startswith(f"cal-{class_index}-") for value in reference_ids[class_index])
        assert not any("surrogate" in value for value in reference_ids[class_index])

    # cal-0-35 has r=36 for all classes.  Its true-class reference removes
    # itself: (1 + 0) / (35 + 1).  Other classes keep all 36 bases.
    assert arrays["known_calibration_p"][35, 0] == pytest.approx(1.0 / 36.0)
    assert arrays["known_calibration_p"][35, 1] == pytest.approx(2.0 / 37.0)
    np.testing.assert_allclose(
        arrays["surrogate_unknown_p"],
        np.full((2, 5), 1.0 / 37.0),
    )
    assert set(metadata["score_by_sample"]) == {
        str(row["sample_id"]) for row in unique_rows
    }


def test_method_score_contract_uses_only_the_preregistered_main_score() -> None:
    fused_logits = np.asarray([[3.0, 2.0, 1.0], [0.0, 4.0, 2.0]])
    guided_anomaly = np.asarray(
        [
            [[1.0, 8.0, 6.0], [3.0, 9.0, 7.0]],
            [[4.0, 2.0, 5.0], [6.0, 6.0, 3.0]],
        ]
    )
    class_conditional_mls = np.asarray([0.2, 0.8])

    for method in (Q0, Q1):
        scores = compute_method_scores(
            method,
            fused_logits,
            guided_anomaly,
            class_conditional_mls,
        )
        np.testing.assert_array_equal(scores["known_prediction"], np.asarray([0, 1]))
        np.testing.assert_array_equal(scores["main_unknown_score"], class_conditional_mls)
        assert scores["main_score_name"] == "class_conditional_mls"
        assert scores["diagnostic_class_conditional_mls"] is None

    expected_guided = np.asarray([2.0, 4.0])
    for method in (Q2, Q3, Q4):
        scores = compute_method_scores(
            method,
            fused_logits,
            guided_anomaly,
            class_conditional_mls,
        )
        np.testing.assert_array_equal(scores["known_prediction"], np.asarray([0, 1]))
        np.testing.assert_array_equal(scores["main_unknown_score"], expected_guided)
        np.testing.assert_array_equal(
            scores["diagnostic_class_conditional_mls"], class_conditional_mls
        )
        assert scores["main_score_name"] == "fusion_guided_reconstruction"

        swapped = compute_method_scores(
            method,
            fused_logits,
            guided_anomaly[:, [1, 0]],
            class_conditional_mls,
        )
        np.testing.assert_array_equal(
            swapped["main_unknown_score"], scores["main_unknown_score"]
        )

    with pytest.raises(DataValidationError, match="method"):
        compute_method_scores(
            "B4_FUSION_GUIDED_CSSR",
            fused_logits,
            guided_anomaly,
            class_conditional_mls,
        )


def _gate_inputs(
    pair_ids: Sequence[str],
    candidate_deltas: Mapping[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    methods = (Q0, Q1, *candidate_deltas)
    metric_rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    for pair_id in pair_ids:
        for method in methods:
            delta = float(candidate_deltas.get(method, 0.0))
            metric_rows.append(
                {
                    "pair_id": pair_id,
                    "method": method,
                    "auroc": 0.50 + delta,
                    "oscr": 0.50,
                    "known_correct_acceptance_rate": 0.80,
                    "fpr95": 0.40,
                }
            )
            for identity_index in range(2):
                identity_rows.append(
                    {
                        "pair_id": pair_id,
                        "method": method,
                        "surrogate_identity": f"{pair_id}-identity-{identity_index}",
                        "auroc": 0.50 + delta,
                    }
                )
    return metric_rows, identity_rows


def _row(
    rows: list[dict[str, Any]],
    *,
    pair_id: str,
    method: str,
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row["pair_id"] == pair_id and row["method"] == method
    ]
    assert len(matches) == 1
    return matches[0]


def _identity_row(
    rows: list[dict[str, Any]],
    *,
    pair_id: str,
    method: str,
    identity_index: int = 0,
) -> dict[str, Any]:
    identity = f"{pair_id}-identity-{identity_index}"
    matches = [
        row
        for row in rows
        if row["pair_id"] == pair_id
        and row["method"] == method
        and row["surrogate_identity"] == identity
    ]
    assert len(matches) == 1
    return matches[0]


def _set_inclusive_identity_boundary(
    identity_rows: list[dict[str, Any]],
    pair_ids: Sequence[str],
    candidate: str,
) -> None:
    for pair_id in pair_ids:
        for identity_index in range(2):
            _identity_row(
                identity_rows,
                pair_id=pair_id,
                method=Q1,
                identity_index=identity_index,
            )["auroc"] = 0.50
            _identity_row(
                identity_rows,
                pair_id=pair_id,
                method=candidate,
                identity_index=identity_index,
            )["auroc"] = 0.40


def _break_gate_condition(
    metric_rows: list[dict[str, Any]],
    identity_rows: list[dict[str, Any]],
    *,
    pair_ids: Sequence[str],
    candidate: str,
    condition: str,
) -> None:
    if condition == "mean_auroc_vs_q1":
        for pair_id in pair_ids:
            _row(metric_rows, pair_id=pair_id, method=candidate)["auroc"] = 0.51
    elif condition == "positive_pair_count":
        positive_count = 1 if len(pair_ids) == 3 else 2
        positive_delta = 0.02 * len(pair_ids) / positive_count
        for index, pair_id in enumerate(pair_ids):
            _row(metric_rows, pair_id=pair_id, method=candidate)["auroc"] = (
                0.50 + positive_delta if index < positive_count else 0.50
            )
    elif condition == "mean_oscr":
        for pair_id in pair_ids:
            _row(metric_rows, pair_id=pair_id, method=candidate)["oscr"] = 0.499
    elif condition == "mean_kccr":
        for pair_id in pair_ids:
            _row(metric_rows, pair_id=pair_id, method=candidate)[
                "known_correct_acceptance_rate"
            ] = 0.789
    elif condition == "mean_fpr95":
        for pair_id in pair_ids:
            _row(metric_rows, pair_id=pair_id, method=candidate)["fpr95"] = 0.421
    elif condition == "mean_auroc_vs_q0":
        for pair_id in pair_ids:
            _row(metric_rows, pair_id=pair_id, method=Q0)["auroc"] = 0.515
    elif condition == "identity_absolute":
        _identity_row(
            identity_rows, pair_id=pair_ids[0], method=Q1
        )["auroc"] = 0.49
        _identity_row(
            identity_rows, pair_id=pair_ids[0], method=candidate
        )["auroc"] = 0.399
    elif condition == "identity_delta_vs_q1":
        _identity_row(identity_rows, pair_id=pair_ids[0], method=Q1)["auroc"] = 0.60
        _identity_row(
            identity_rows, pair_id=pair_ids[0], method=candidate
        )["auroc"] = 0.499
    else:
        raise AssertionError(f"unknown gate condition {condition}")


GATE_CONDITIONS = (
    "mean_auroc_vs_q1",
    "positive_pair_count",
    "mean_oscr",
    "mean_kccr",
    "mean_fpr95",
    "mean_auroc_vs_q0",
    "identity_absolute",
    "identity_delta_vs_q1",
)


def test_pilot_gate_accepts_all_eight_inclusive_boundaries() -> None:
    config = load_fg_mv_cssr_e2e_config(CONFIG_PATH)
    metric_rows, identity_rows = _gate_inputs(PILOT_PAIRS, {Q2: 0.02, Q3: 0.0, Q4: 0.0})
    for pair_id in PILOT_PAIRS:
        _row(metric_rows, pair_id=pair_id, method=Q0)["auroc"] = 0.51
        _row(metric_rows, pair_id=pair_id, method=Q2)[
            "known_correct_acceptance_rate"
        ] = 0.79
        _row(metric_rows, pair_id=pair_id, method=Q2)["fpr95"] = 0.42
    _set_inclusive_identity_boundary(identity_rows, PILOT_PAIRS, Q2)

    result = evaluate_pilot_gate(metric_rows, identity_rows, config)

    assert result["candidate_gates"][Q2]["passed"] is True
    assert all(result["candidate_gates"][Q2]["checks"].values())
    assert result["selected_method"] == Q2
    assert result["signal"] == "e2e_alignment_signal"
    assert result["confirmation_allowed"] is True
    assert result["final_unknown_test_authorized"] is False


@pytest.mark.parametrize("condition", GATE_CONDITIONS)
def test_each_pilot_gate_condition_is_independently_required(condition: str) -> None:
    config = load_fg_mv_cssr_e2e_config(CONFIG_PATH)
    metric_rows, identity_rows = _gate_inputs(PILOT_PAIRS, {Q2: 0.02, Q3: 0.0, Q4: 0.0})
    _break_gate_condition(
        metric_rows,
        identity_rows,
        pair_ids=PILOT_PAIRS,
        candidate=Q2,
        condition=condition,
    )

    result = evaluate_pilot_gate(metric_rows, identity_rows, config)

    assert result["candidate_gates"][Q2]["passed"] is False
    assert result["selected_method"] is None
    assert result["signal"] == "cssr_redesign_failed"
    assert result["confirmation_allowed"] is False
    assert result["final_unknown_test_authorized"] is False


@pytest.mark.parametrize(
    ("candidate_deltas", "expected_method", "expected_signal"),
    (
        ({Q2: 0.02, Q3: 0.03, Q4: 0.039}, Q2, "e2e_alignment_signal"),
        ({Q2: 0.02, Q3: 0.04, Q4: 0.04}, Q3, "absolute_alignment_signal"),
        ({Q2: 0.02, Q3: 0.04, Q4: 0.05}, Q4, "local_structure_signal"),
        ({Q2: 0.00, Q3: 0.02, Q4: 0.039}, Q3, "absolute_alignment_signal"),
        ({Q2: 0.00, Q3: 0.02, Q4: 0.04}, Q4, "local_structure_signal"),
        ({Q2: 0.00, Q3: 0.00, Q4: 0.02}, Q4, "local_structure_signal"),
        ({Q2: 0.00, Q3: 0.00, Q4: 0.00}, None, "cssr_redesign_failed"),
    ),
)
def test_pilot_selection_is_deterministic_and_complexity_ordered(
    candidate_deltas: Mapping[str, float],
    expected_method: str | None,
    expected_signal: str,
) -> None:
    config = load_fg_mv_cssr_e2e_config(CONFIG_PATH)
    metric_rows, identity_rows = _gate_inputs(PILOT_PAIRS, candidate_deltas)

    result = evaluate_pilot_gate(metric_rows, identity_rows, config)
    reversed_result = evaluate_pilot_gate(
        list(reversed(metric_rows)), list(reversed(identity_rows)), config
    )

    assert result == reversed_result
    assert result["selected_method"] == expected_method
    assert result["signal"] == expected_signal
    assert result["confirmation_allowed"] is (expected_method is not None)
    assert result["final_unknown_test_authorized"] is False


def test_confirmation_gate_accepts_all_eight_inclusive_boundaries() -> None:
    config = load_fg_mv_cssr_e2e_config(CONFIG_PATH)
    metric_rows, identity_rows = _gate_inputs(CONFIRMATION_PAIRS, {Q2: 0.02})
    for pair_id in CONFIRMATION_PAIRS:
        _row(metric_rows, pair_id=pair_id, method=Q0)["auroc"] = 0.51
        _row(metric_rows, pair_id=pair_id, method=Q2)[
            "known_correct_acceptance_rate"
        ] = 0.79
        _row(metric_rows, pair_id=pair_id, method=Q2)["fpr95"] = 0.42
    _set_inclusive_identity_boundary(identity_rows, CONFIRMATION_PAIRS, Q2)

    result = evaluate_confirmation_gate(
        metric_rows, identity_rows, Q2, config
    )

    assert result["passed"] is True
    assert all(result["checks"].values())
    assert result["decision"] == "cssr_redesign_worth_full_validation"
    assert result["final_unknown_test_authorized"] is False


@pytest.mark.parametrize("condition", GATE_CONDITIONS)
def test_each_confirmation_gate_condition_is_independently_required(
    condition: str,
) -> None:
    config = load_fg_mv_cssr_e2e_config(CONFIG_PATH)
    metric_rows, identity_rows = _gate_inputs(CONFIRMATION_PAIRS, {Q2: 0.02})
    _break_gate_condition(
        metric_rows,
        identity_rows,
        pair_ids=CONFIRMATION_PAIRS,
        candidate=Q2,
        condition=condition,
    )

    result = evaluate_confirmation_gate(
        metric_rows, identity_rows, Q2, config
    )

    assert result["passed"] is False
    assert result["decision"] == "cssr_redesign_rejected"
    assert result["final_unknown_test_authorized"] is False


@pytest.mark.parametrize("selected_method", (Q0, Q1, "UNKNOWN"))
def test_confirmation_gate_rejects_non_cssr_selected_methods(
    selected_method: str,
) -> None:
    config = load_fg_mv_cssr_e2e_config(CONFIG_PATH)
    metric_rows, identity_rows = _gate_inputs(CONFIRMATION_PAIRS, {Q2: 0.03})
    with pytest.raises(DataValidationError, match="selected|candidate|CSSR"):
        evaluate_confirmation_gate(
            metric_rows,
            identity_rows,
            selected_method,
            config,
        )


def test_all_protocol_decisions_permanently_keep_final_test_unauthorized() -> None:
    config = load_fg_mv_cssr_e2e_config(CONFIG_PATH)
    failed_metrics, failed_identities = _gate_inputs(
        PILOT_PAIRS, {Q2: 0.0, Q3: 0.0, Q4: 0.0}
    )
    passed_metrics, passed_identities = _gate_inputs(
        CONFIRMATION_PAIRS, {Q2: 0.03}
    )

    pilot = evaluate_pilot_gate(failed_metrics, failed_identities, config)
    confirmation = evaluate_confirmation_gate(
        passed_metrics, passed_identities, Q2, config
    )

    assert config["decisions"]["final_unknown_test_authorized"] is False
    assert pilot["final_unknown_test_authorized"] is False
    assert confirmation["final_unknown_test_authorized"] is False
    assert all(
        unit["pair_id"] in set(PILOT_PAIRS) | set(CONFIRMATION_PAIRS)
        for unit in [
            *build_phase_plan(config, "smoke"),
            *build_phase_plan(config, "pilot"),
            *build_phase_plan(config, "confirmation", selected_method=Q2),
        ]
    )
