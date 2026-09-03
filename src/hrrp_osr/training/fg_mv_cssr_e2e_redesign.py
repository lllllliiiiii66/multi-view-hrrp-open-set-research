from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import yaml
from torch import nn

from hrrp_osr.data.errors import DataConfigError, DataValidationError
from hrrp_osr.data.manifest import file_sha256
from hrrp_osr.evaluation.metrics import accuracy_score, evaluate_open_set, macro_f1_score
from hrrp_osr.evaluation.ms_mean_factorial import REPORT_METRIC_KEYS
from hrrp_osr.models.cssr_e2e_1d import (
    CSSR_VARIANTS,
    FGMVCSSRE2EModel,
    LOSS_WEIGHTS,
    Q1_CE_FINETUNE_CONTROL,
    Q2_E2E_REL_CSSR_1X1,
    Q3_E2E_ABSREL_CSSR_1X1,
    Q4_E2E_ABSREL_CSSR_LOCAL3,
    TRAINABLE_VARIANTS,
)
from hrrp_osr.training.arpl_pilot import _resolve_device, _set_determinism
from hrrp_osr.training.fg_mv_cssr_pilot import (
    EXPECTED_R2_ROOT_HASH_MANIFEST_SHA256,
    EXPECTED_R2_UNIT_HASHES,
    _artifact_hashes,
    _array_sha256,
    _atomic_write_bytes,
    _configure_numerical_runtime,
    _load_bundle,
    _load_prior_config,
    _prepare_frozen_split,
    _read_csv,
    _read_json,
    _role_manifest_rows,
    _sequence_sha256,
    _smoke_pair_indices,
    _write_csv,
    _write_json,
    build_unique_base_sample_manifest,
    compute_class_conditional_mls_scores,
    cssr_conformal_p_values,
    load_and_audit_frozen_r2,
)


EXPERIMENT_ID = "fg_mv_cssr_e2e_redesign_v2"
CONFIG_RELATIVE_PATH = "configs/experiments/cssr/fg_mv_cssr_e2e_redesign_v2.yaml"
Q0_FROZEN_R2_CC_MLS = "Q0_FROZEN_R2_CC_MLS"
METHODS = (Q0_FROZEN_R2_CC_MLS, *TRAINABLE_VARIANTS)
TRAINABLE_METHODS = tuple(TRAINABLE_VARIANTS)
CSSR_METHODS = tuple(TRAINABLE_VARIANTS[1:])
PILOT_PAIRS = ("N1", "N4", "N2")
CONFIRMATION_PAIRS = ("N0", "N3", "N5", "N6")
ANGLE_FOLD = 0
MODEL_SEED = 20260830
FINETUNE_SEED = 20260904
PAIR_SCHEDULE_VERSION = "fg_mv_cssr_e2e_pair_schedule_v1"
BATCH_ORDER_VERSION = "fg_mv_cssr_e2e_batch_order_v1"
GATE_TOLERANCE = 1.0e-12
LABEL_BY_METHOD = {
    Q2_E2E_REL_CSSR_1X1: "e2e_alignment_signal",
    Q3_E2E_ABSREL_CSSR_1X1: "absolute_alignment_signal",
    Q4_E2E_ABSREL_CSSR_LOCAL3: "local_structure_signal",
}
TASK_SOURCE_FILES = (
    CONFIG_RELATIVE_PATH,
    "src/hrrp_osr/amdr/data.py",
    "src/hrrp_osr/amdr/model.py",
    "src/hrrp_osr/amdr/reduction.py",
    "src/hrrp_osr/amdr/smoke.py",
    "src/hrrp_osr/data/config.py",
    "src/hrrp_osr/data/errors.py",
    "src/hrrp_osr/data/manifest.py",
    "src/hrrp_osr/data/processed.py",
    "src/hrrp_osr/data/protocol.py",
    "src/hrrp_osr/evaluation/metrics.py",
    "src/hrrp_osr/evaluation/ms_mean_factorial.py",
    "src/hrrp_osr/models/arpl.py",
    "src/hrrp_osr/models/cnn1d.py",
    "src/hrrp_osr/models/cssr_1d.py",
    "src/hrrp_osr/models/cssr_e2e_1d.py",
    "src/hrrp_osr/models/hrrp_ms_resnet.py",
    "src/hrrp_osr/models/mv_rpformer.py",
    "src/hrrp_osr/models/ms_mean_factorial.py",
    "src/hrrp_osr/training/arpl_mv_evidence.py",
    "src/hrrp_osr/training/arpl_pilot.py",
    "src/hrrp_osr/training/fg_mv_cssr_pilot.py",
    "src/hrrp_osr/training/mv_rpformer.py",
    "src/hrrp_osr/training/ms_mean_head_factorial.py",
    "src/hrrp_osr/training/fg_mv_cssr_e2e_redesign.py",
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DataConfigError(f"{name} must be a mapping")
    return value


def _require(errors: list[str], observed: Any, expected: Any, name: str) -> None:
    if observed != expected:
        errors.append(f"{name} changed: expected {expected!r}, observed {observed!r}")


def _derived_seed(payload: str) -> int:
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _render_csv(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise DataValidationError("cannot render an empty CSV")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def load_fg_mv_cssr_e2e_config(path: str | Path) -> dict[str, Any]:
    """Load the single frozen E2E redesign configuration without fallback."""

    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = dict(_mapping(yaml.safe_load(handle), "E2E CSSR config"))
    errors: list[str] = []
    _require(errors, config.get("schema_version"), 1, "schema_version")
    _require(
        errors,
        config.get("stage"),
        "P3_fg_mv_cssr_e2e_redesign_fast_iteration",
        "stage",
    )
    _require(errors, config.get("experiment_id"), EXPERIMENT_ID, "experiment_id")
    _require(
        errors,
        config.get("result_scope"),
        "diagnostic_smoke_then_conditional_surrogate_confirmation",
        "result_scope",
    )

    evidence = _mapping(config.get("evidence_scope"), "evidence_scope")
    _require(errors, evidence.get("source_known_odd_angle_only"), True, "odd-only")
    for name in (
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
        _require(errors, evidence.get(name), False, f"evidence_scope.{name}")

    prior = _mapping(config.get("prior_r2"), "prior_r2")
    expected_prior = {
        "result_commit": "edb05062d07be1984067f91759d6029cd9c0bf9a",
        "formal_code_commit": "62e318de82b4221b599e06b1166483673e9c1cd3",
        "experiment_id": "ms_mean_head_factorial_surrogate_v1",
        "method": "R2_MS_MEAN_CE",
        "phase": "confirmation",
        "angle_fold": ANGLE_FOLD,
        "initialization_seed": MODEL_SEED,
        "checkpoint_epoch": 100,
        "checkpoint_selection": "fixed_final_epoch",
        "source_config": "configs/experiments/arpl/ms_mean_head_factorial_surrogate_v1.yaml",
        "source_config_sha256": "c11daa6e2e5a7d7b72bc36840e60fc871f332c4fc85652636c729aa2eba14c71",
        "unit_relative_template": "{pair_id}/fold_0/seed_20260830/R2_MS_MEAN_CE",
        "root_artifact_hash_manifest_sha256": EXPECTED_R2_ROOT_HASH_MANIFEST_SHA256,
        "strict_load_required": True,
        "old_logits_exact_match_required": True,
    }
    for name, expected in expected_prior.items():
        _require(errors, prior.get(name), expected, f"prior_r2.{name}")
    observed_unit_hashes = {
        str(pair_id): dict(_mapping(value, f"prior_r2.{pair_id}"))
        for pair_id, value in _mapping(
            prior.get("unit_artifact_hashes"), "prior_r2.unit_artifact_hashes"
        ).items()
    }
    _require(errors, observed_unit_hashes, EXPECTED_R2_UNIT_HASHES, "prior R2 hashes")

    expected_bundle = {
        "dataset_id": "hrrp_10class_theta83_hh_v1",
        "preprocessing_id": "hrrp_padding_complex_gaussian_v1",
        "profiles_sha256": "2dd92282c125f0f677cf1f2dfce828781c8ba4385cf9ae552c4a2c56033c3f5b",
        "manifest_sha256": "748b9f30629c3b3cbe66c6a1dac30863fdab2d81a214e46d8bc3ef7c6022a08a",
        "bundle_sha256": "79176091b5b745e5df5957f1bb4ade7304c1403a4a32e64622e012db67d5a0c5",
    }
    _require(errors, dict(_mapping(config.get("bundle"), "bundle")), expected_bundle, "bundle")

    classes = _mapping(config.get("classes"), "classes")
    _require(
        errors,
        list(classes.get("source_known_order", [])),
        [
            "CVN77",
            "DDG-1000",
            "DDG-112",
            "油气轮MARVEL CRANE",
            "爱达魔都号",
            "迷你好望角型散货船",
            "集装箱船达飞罗尔多夫级",
        ],
        "source known order",
    )
    expected_pairs = [
        {"pair_id": "N0", "surrogate_unknown_indices": [0, 2], "train_known_indices": [1, 3, 4, 5, 6]},
        {"pair_id": "N1", "surrogate_unknown_indices": [2, 5], "train_known_indices": [0, 1, 3, 4, 6]},
        {"pair_id": "N2", "surrogate_unknown_indices": [3, 5], "train_known_indices": [0, 1, 2, 4, 6]},
        {"pair_id": "N3", "surrogate_unknown_indices": [1, 3], "train_known_indices": [0, 2, 4, 5, 6]},
        {"pair_id": "N4", "surrogate_unknown_indices": [1, 6], "train_known_indices": [0, 2, 3, 4, 5]},
        {"pair_id": "N5", "surrogate_unknown_indices": [4, 6], "train_known_indices": [0, 1, 2, 3, 5]},
        {"pair_id": "N6", "surrogate_unknown_indices": [0, 4], "train_known_indices": [1, 2, 3, 5, 6]},
    ]
    _require(errors, list(classes.get("identity_pairs", [])), expected_pairs, "identity pairs")
    _require(errors, list(classes.get("pilot_pairs", [])), list(PILOT_PAIRS), "pilot pairs")
    _require(
        errors,
        list(classes.get("confirmation_pairs", [])),
        list(CONFIRMATION_PAIRS),
        "confirmation pairs",
    )

    data = _mapping(config.get("data"), "data")
    for name, expected in {
        "angle_fold": ANGLE_FOLD,
        "development_angle_parity": "odd",
        "view_count": 2,
        "train_unique_base_samples_per_class": 144,
        "known_calibration_unique_base_samples_per_class": 36,
        "evaluation_pairs_per_class": 500,
        "final_test_pairs_generated": False,
    }.items():
        _require(errors, data.get(name), expected, f"data.{name}")
    expected_schedule = {
        "algorithm": "deterministic_cross_frame_derangement_v1",
        "base_order": ["model_label", "sample_id"],
        "epoch_seed_material": "fg_mv_cssr_e2e_pair_schedule_v1|finetune_seed|pair_id|fold|epoch|model_label",
        "epoch_seed_hash": "sha256_first_8_bytes_big_endian_unsigned",
        "random_generator": "numpy_PCG64",
        "matching": "deterministic_augmenting_path_bipartite",
        "maximum_matching_attempts": 4096,
        "allowed_edge": "different_15_degree_frame_only",
        "view1_usage_per_base_per_epoch": 1,
        "view2_usage_per_base_per_epoch": 1,
        "unordered_pair_unique_within_class_epoch": True,
        "dataloader_shuffle": False,
        "batch_order_seed_material": "fg_mv_cssr_e2e_batch_order_v1|finetune_seed|pair_id|fold|epoch",
        "save_epoch_manifests": True,
        "save_usage_audit": True,
        "fail_if_infeasible": True,
    }
    _require(
        errors,
        dict(_mapping(data.get("dynamic_pair_schedule"), "dynamic pair schedule")),
        expected_schedule,
        "dynamic pair schedule",
    )
    _require(
        errors,
        dict(_mapping(data.get("smoke"), "smoke")),
        {
            "pair_id": "N1",
            "methods": list(TRAINABLE_METHODS),
            "epochs": 1,
            "full_train_unique_base_schedule": True,
            "evaluation_pairs_per_class": 2,
            "diagnostic_only": True,
        },
        "smoke",
    )

    _require(
        errors,
        dict(_mapping(config.get("normalization"), "normalization")),
        {"method": "reuse_exact_r2_global_scalar_zscore", "epsilon": 1.0e-8},
        "normalization",
    )
    _require(
        errors,
        dict(_mapping(config.get("r2_model"), "r2_model")),
        {
            "architecture": "ms_mean_head_factorial_v1",
            "encoder": "hrrp_ms_resnet_1d_v1",
            "input_length": 601,
            "feature_map_shape": [128, 76],
            "pooled_feature_dim": 128,
            "fusion": "arithmetic_mean",
            "head": "linear_ce",
            "prediction": "own_fused_logits_argmax",
            "dropout": 0.1,
        },
        "r2 model",
    )
    methods = _mapping(config.get("methods"), "methods")
    _require(errors, list(methods.get("ordered", [])), list(METHODS), "methods.ordered")
    _require(
        errors, list(methods.get("trainable", [])), list(TRAINABLE_METHODS), "methods.trainable"
    )
    _require(errors, list(methods.get("cssr_candidates", [])), list(CSSR_METHODS), "CSSR candidates")
    _require(errors, methods.get("q0_retraining"), False, "Q0 retraining")

    scope = _mapping(config.get("trainable_scope"), "trainable_scope")
    _require(errors, list(scope.get("frozen_modules", [])), ["encoder.stem", "encoder.stages.0", "encoder.stages.1"], "frozen modules")
    _require(errors, list(scope.get("trainable_modules", [])), ["encoder.stages.2", "encoder.projection", "global_head"], "trainable modules")
    for name in (
        "frozen_modules_forced_eval",
        "cssr_autoencoders_trainable_for_q2_q4",
        "frozen_parameters_and_buffers_hash_required",
        "strict_r2_load_required",
        "epoch0_common_state_exact_match_required",
        "reset_rng_after_model_and_optimizer_construction",
    ):
        _require(errors, scope.get(name), True, f"trainable_scope.{name}")

    autoencoders = _mapping(config.get("autoencoders"), "autoencoders")
    expected_autoencoders = {
        "class_count": 5,
        "input_channels": 128,
        "latent_channels": 64,
        "independent_per_class": True,
        "bias": False,
        "activation": "Tanh",
        "q2_q3_kernel_size": 1,
        "q2_q3_padding": 0,
        "q2_q3_same_initialization_required": True,
        "q4_kernel_size": 3,
        "q4_padding": 1,
        "skip_connection": False,
        "normalization_layer": "none",
        "attention": False,
        "extra_hidden_layers": 0,
    }
    _require(errors, dict(autoencoders), expected_autoencoders, "autoencoders")

    loss = _mapping(config.get("loss"), "loss")
    expected_weights = {method: dict(LOSS_WEIGHTS[method]) for method in TRAINABLE_METHODS}
    _require(errors, dict(_mapping(loss.get("weights"), "loss.weights")), expected_weights, "loss weights")
    _require(errors, loss.get("batch_reduction"), "arithmetic_mean", "loss batch reduction")
    gradient = _mapping(loss.get("gradient_audit"), "loss.gradient_audit")
    for name, expected in {
        "ratio_scope": "weighted_auxiliary_to_classification_on_last_residual_stage",
        "denominator_floor": 1.0e-12,
        "maximum_ratio": 100.0,
        "consecutive_epoch_mean_violations_to_fail": 3,
        "ae_gradients_record_only": True,
        "fail_on_nonfinite": True,
    }.items():
        observed = float(gradient.get(name)) if isinstance(expected, float) else gradient.get(name)
        _require(errors, observed, expected, f"gradient_audit.{name}")
    absolute = _mapping(loss.get("absolute_reconstruction"), "absolute reconstruction")
    _require(errors, absolute.get("numerator"), "per_sample_view_mean_absolute_reconstruction_error_over_channels_and_positions", "absolute numerator")
    _require(errors, absolute.get("denominator"), "per_sample_view_mean_absolute_activation_over_channels_and_positions_plus_epsilon", "absolute denominator")
    _require(errors, float(absolute.get("epsilon", -1)), 1.0e-8, "absolute epsilon")
    separation = _mapping(loss.get("separation"), "separation")
    _require(errors, separation.get("nearest_wrong_class"), "minimum_normalized_absolute_reconstruction_error", "separation nearest wrong class")
    _require(errors, float(separation.get("margin", -1)), 0.2, "separation margin")
    relative = _mapping(loss.get("relative"), "relative loss")
    _require(errors, float(relative.get("gamma", -1)), 0.1, "relative gamma")
    _require(errors, float(relative.get("clip_min", 0)), -100.0, "relative clip min")
    _require(errors, float(relative.get("clip_max", 0)), 100.0, "relative clip max")
    _require(errors, relative.get("reconstruction_error"), "channel_sum_L1", "relative reconstruction error")
    _require(
        errors,
        relative.get("probability_order"),
        "class_softmax_per_position_then_position_mean",
        "relative probability order",
    )
    _require(errors, relative.get("view_reduction"), "arithmetic_mean", "relative view reduction")

    training = _mapping(config.get("training"), "training")
    expected_training = {
        "optimizer": "AdamW",
        "epochs": 20,
        "batch_size_pairs": 64,
        "warmup_epochs": 2,
        "scheduler": "linear_warmup_then_cosine_to_zero",
        "scheduler_step_timing": "before_each_epoch_optimizer_updates",
        "scheduler_factor_formula": "epoch<=2:epoch/2;epoch>2:0.5*(1+cos(pi*(epoch-3)/18));post_epoch_20:0",
        "all_formal_epochs_have_nonzero_learning_rate": True,
        "zero_reached_after_formal_epoch": True,
        "gradient_clip_norm": 5.0,
        "finetune_seed": FINETUNE_SEED,
        "lr_last_stage": 3.0e-5,
        "lr_projection_and_ce_head": 1.0e-4,
        "lr_autoencoders": 1.0e-3,
        "weight_decay_last_stage": 5.0e-4,
        "weight_decay_projection_and_head": 5.0e-4,
        "weight_decay_autoencoders": 1.0e-4,
        "early_stopping": False,
        "formal_checkpoint_epoch": 20,
        "performance_checkpoint_selection": False,
        "diagnostic_epochs": [0, 5, 10, 15, 20],
    }
    for name, expected in expected_training.items():
        observed = float(training.get(name)) if isinstance(expected, float) else training.get(name)
        _require(errors, observed, expected, f"training.{name}")

    diagnostics = _mapping(config.get("diagnostics"), "diagnostics")
    _require(errors, diagnostics.get("known_calibration_only"), True, "known-only diagnostics")
    _require(errors, diagnostics.get("every_epoch"), True, "every-epoch diagnostics")
    _require(errors, list(diagnostics.get("metric_names", [])), ["accuracy", "macro_f1", "nll", "brier", "ece", "mean_max_logit", "mean_top1_top2_logit_margin", "mean_single_view_feature_norm", "mean_fused_feature_norm", "ce_head_weight_norm", "mean_fused_feature_l2_drift_from_epoch0", "mean_kl_epoch0_to_current"], "diagnostic metric names")
    _require(errors, diagnostics.get("brier"), "per_sample_sum_over_class_then_sample_mean", "Brier definition")
    _require(errors, int(diagnostics.get("ece_bins", -1)), 15, "ECE bins")
    _require(errors, diagnostics.get("ece_bin_type"), "equal_width", "ECE bin type")
    _require(errors, diagnostics.get("feature_drift"), "mean_pairwise_fused_feature_l2_from_epoch0", "feature drift")
    _require(errors, diagnostics.get("kl_direction"), "epoch0_to_current", "KL direction")
    _require(errors, diagnostics.get("diagnostic_only"), True, "diagnostics only")

    calibration = _mapping(config.get("calibration"), "calibration")
    expected_calibration = {
        "cssr_reference_population": "unique_known_calibration_base_samples_by_true_class",
        "cssr_reference_shared_across_view_slots": True,
        "cssr_reference_tail": "greater_than_or_equal_r",
        "cssr_p_value_smoothing": "plus_one_numerator_and_denominator",
        "cssr_score_transform": "negative_log_p_plus_epsilon",
        "cssr_leave_one_base_sample_out": True,
        "mls_reference": "correctly_predicted_known_calibration_pairs_by_true_class",
        "mls_nonconformity": "negative_maximum_fused_logit",
        "mls_anomaly_quantile_tail": "less_than_or_equal_nonconformity",
        "mls_leave_one_pair_out": True,
        "empty_class_reference_policy": "fail",
        "score_epsilon": 1.0e-8,
        "threshold_source": "own_known_calibration_only",
        "threshold_known_acceptance_rate": 0.95,
        "threshold_rule": "exact_sorted_rank_ceiling",
    }
    _require(errors, dict(calibration), expected_calibration, "calibration")
    scores = _mapping(config.get("scores"), "scores")
    _require(errors, scores.get("direction"), "larger_is_more_unknown", "score direction")
    _require(errors, scores.get("known_prediction"), "own_fused_ce_argmax", "known prediction")
    _require(
        errors,
        dict(_mapping(scores.get("main_by_method"), "main scores")),
        {
            Q0_FROZEN_R2_CC_MLS: "class_conditional_mls",
            Q1_CE_FINETUNE_CONTROL: "class_conditional_mls",
            Q2_E2E_REL_CSSR_1X1: "fusion_guided_reconstruction",
            Q3_E2E_ABSREL_CSSR_1X1: "fusion_guided_reconstruction",
            Q4_E2E_ABSREL_CSSR_LOCAL3: "fusion_guided_reconstruction",
        },
        "main scores",
    )
    _require(
        errors,
        dict(_mapping(scores.get("diagnostic_only_by_method"), "diagnostic scores")),
        {
            Q2_E2E_REL_CSSR_1X1: ["class_conditional_mls"],
            Q3_E2E_ABSREL_CSSR_1X1: ["class_conditional_mls"],
            Q4_E2E_ABSREL_CSSR_LOCAL3: ["class_conditional_mls"],
        },
        "diagnostic-only scores",
    )
    prohibited = set(scores.get("prohibited", []))
    _require(
        errors,
        prohibited,
        {
            "independent_view_cssr",
            "common_class_cssr",
            "max_view",
            "score_weighting",
            "mls_cssr_linear_combination",
            "learned_score_fusion",
            "top2_class_mixture",
            "learned_rejector",
        },
        "prohibited scores",
    )
    evaluation = _mapping(config.get("evaluation"), "evaluation")
    _require(errors, list(evaluation.get("report_metrics", [])), list(REPORT_METRIC_KEYS), "report metrics")
    _require(errors, evaluation.get("per_pair_identity_context"), True, "per-pair identity context")
    _require(errors, evaluation.get("absorption_by_known_class"), True, "absorption audit")
    _require(errors, evaluation.get("ddg_bidirectional_absorption"), True, "DDG audit")
    _require(errors, evaluation.get("marvel_cape_reverse_diagnostic"), True, "MARVEL/Cape audit")

    for gate_name, positive_count in (("pilot_gate", 2), ("confirmation_gate", 3)):
        gate = _mapping(config.get(gate_name), gate_name)
        for name, expected in {
            "baseline": Q1_CE_FINETUNE_CONTROL,
            "frozen_reference": Q0_FROZEN_R2_CC_MLS,
            "minimum_mean_auroc_delta_vs_q1": 0.02,
            "minimum_positive_pair_count_vs_q1": positive_count,
            "minimum_mean_oscr_delta_vs_q1": 0.0,
            "maximum_mean_kccr_drop_vs_q1": 0.01,
            "maximum_mean_fpr95_increase_vs_q1": 0.02,
            "minimum_mean_auroc_delta_vs_q0": 0.01,
            "minimum_identity_auroc": 0.40,
            "minimum_identity_auroc_delta_vs_q1": -0.10,
        }.items():
            observed = float(gate.get(name)) if isinstance(expected, float) else gate.get(name)
            _require(errors, observed, expected, f"{gate_name}.{name}")
    pilot_gate = _mapping(config.get("pilot_gate"), "pilot_gate")
    _require(errors, list(pilot_gate.get("candidates", [])), list(CSSR_METHODS), "pilot candidates")
    _require(errors, list(pilot_gate.get("complexity_order", [])), list(CSSR_METHODS), "complexity order")
    _require(errors, float(pilot_gate.get("replacement_minimum_mean_auroc_gain", -1)), 0.02, "replacement gain")
    _require(errors, pilot_gate.get("no_candidate_label"), "cssr_redesign_failed", "no candidate label")
    _require(errors, pilot_gate.get("no_candidate_runs_confirmation"), False, "no candidate confirmation")
    confirmation_gate = _mapping(config.get("confirmation_gate"), "confirmation_gate")
    _require(errors, confirmation_gate.get("success_label"), "cssr_redesign_worth_full_validation", "confirmation success")
    _require(errors, confirmation_gate.get("failure_label"), "cssr_redesign_rejected", "confirmation failure")

    runtime = _mapping(config.get("runtime"), "runtime")
    for name, expected in {
        "formal_device": "cuda",
        "expected_gpu_model": "NVIDIA GeForce RTX 4090",
        "maximum_parallel_tasks": 4,
        "deterministic_algorithms": True,
        "amp": False,
        "tf32": False,
        "torch_compile": False,
        "num_workers": 0,
    }.items():
        _require(errors, runtime.get(name), expected, f"runtime.{name}")
    outputs = _mapping(config.get("outputs"), "outputs")
    _require(errors, outputs.get("namespace"), "artifacts/cssr/fg_mv_cssr_e2e_redesign_v2", "output namespace")
    _require(errors, outputs.get("fail_if_output_nonempty"), True, "output overwrite policy")
    for name in (
        "save_resolved_config",
        "save_r2_reference_and_hash",
        "save_epoch_pair_schedules",
        "save_schedule_usage_audit",
        "save_checkpoint",
        "save_training_and_gradient_log",
        "save_confidence_diagnostics",
        "save_reference_distributions",
        "save_predictions_and_scores",
        "save_metrics_and_error_analysis",
        "save_environment_and_hashes",
    ):
        _require(errors, outputs.get(name), True, f"outputs.{name}")
    decisions = _mapping(config.get("decisions"), "decisions")
    for name in (
        "final_unknown_test_authorized",
        "second_angle_fold_authorized",
        "extra_seed_authorized",
        "arpl_authorized",
        "automatic_followon_method_authorized",
    ):
        _require(errors, decisions.get(name), False, f"decisions.{name}")
    if errors:
        raise DataConfigError("Invalid E2E CSSR config:\n- " + "\n- ".join(errors))
    config["_config_path"] = str(config_path)
    config["_config_sha256"] = file_sha256(config_path)
    return config


def build_epoch_pair_schedule(
    unique_rows: Sequence[Mapping[str, Any]],
    *,
    pair_id: str,
    angle_fold: int,
    epoch: int,
    finetune_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the frozen epoch-wise cross-frame perfect matching schedule."""

    if epoch < 1:
        raise DataValidationError("epoch must be one-based")
    train_rows = [
        dict(row)
        for row in unique_rows
        if str(row.get("experiment_role")) == "train_known"
    ]
    if not train_rows:
        raise DataValidationError("dynamic schedule has no train-known base samples")
    sample_ids = [str(row["sample_id"]) for row in train_rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise DataValidationError("dynamic schedule repeats a train-known sample ID")

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        label = int(row["model_label"])
        if int(row["angle_deg"]) % 2 != 1:
            raise DataValidationError("even-angle base entered dynamic training")
        if int(row["frame_id"]) != int(row["angle_deg"]) // 15:
            raise DataValidationError("dynamic schedule frame metadata changed")
        grouped[label].append(row)
    labels = tuple(sorted(grouped))
    if labels != tuple(range(len(labels))):
        raise DataValidationError("dynamic schedule labels are not contiguous")

    scheduled: list[dict[str, Any]] = []
    class_seeds: dict[str, int] = {}
    class_matching_attempts: dict[str, int] = {}
    for label in labels:
        ordered = sorted(
            grouped[label], key=lambda row: (int(row["model_label"]), str(row["sample_id"]))
        )
        if len(ordered) < 2 or len({int(row["frame_id"]) for row in ordered}) < 2:
            raise DataValidationError("cross-frame derangement requires at least two frames")
        seed_payload = (
            f"{PAIR_SCHEDULE_VERSION}|{int(finetune_seed)}|{pair_id}|"
            f"{int(angle_fold)}|{int(epoch)}|{label}"
        )
        epoch_seed = _derived_seed(seed_payload)
        class_seeds[str(label)] = epoch_seed
        rng = np.random.Generator(np.random.PCG64(epoch_seed))
        matched_left_to_right: dict[int, int] | None = None
        left_order: list[int] = []
        matching_attempt = -1
        for attempt in range(4096):
            current_left_order = [
                int(value) for value in rng.permutation(len(ordered))
            ]
            candidates: dict[int, list[int]] = {}
            for left in current_left_order:
                valid = np.asarray(
                    [
                        right
                        for right, row in enumerate(ordered)
                        if int(row["frame_id"])
                        != int(ordered[left]["frame_id"])
                    ],
                    dtype=np.int64,
                )
                if valid.size == 0:
                    raise DataValidationError(
                        "one train base has no cross-frame partner"
                    )
                candidates[left] = [
                    int(value) for value in valid[rng.permutation(valid.size)]
                ]
            matched_right_to_left: dict[int, int] = {}

            def augment(left: int, visited: set[int]) -> bool:
                for right in candidates[left]:
                    if right in visited:
                        continue
                    visited.add(right)
                    previous = matched_right_to_left.get(right)
                    if previous is None or augment(previous, visited):
                        matched_right_to_left[right] = left
                        return True
                return False

            complete = all(
                augment(left, set()) for left in current_left_order
            )
            if not complete:
                continue
            candidate_mapping = {
                left: right for right, left in matched_right_to_left.items()
            }
            if set(candidate_mapping) != set(range(len(ordered))):
                continue
            has_two_cycle = any(
                candidate_mapping.get(right) == left
                for left, right in candidate_mapping.items()
            )
            if has_two_cycle:
                continue
            matched_left_to_right = candidate_mapping
            left_order = current_left_order
            matching_attempt = attempt
            break
        if matched_left_to_right is None:
            raise DataValidationError(
                f"cross-frame unordered-unique matching failed for label {label}"
            )
        class_matching_attempts[str(label)] = matching_attempt
        for left in left_order:
            right = matched_left_to_right[left]
            view1 = ordered[left]
            view2 = ordered[right]
            identity = "\0".join(
                (
                    EXPERIMENT_ID,
                    pair_id,
                    str(angle_fold),
                    str(epoch),
                    str(label),
                    str(view1["sample_id"]),
                    str(view2["sample_id"]),
                )
            )
            scheduled.append(
                {
                    "epoch": int(epoch),
                    "epoch_seed": epoch_seed,
                    "matching_attempt": matching_attempt,
                    "identity_pair_id": str(pair_id),
                    "pair_id": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
                    "model_label": label,
                    "class_name": str(view1["class_name"]),
                    "view1_sample_id": str(view1["sample_id"]),
                    "view2_sample_id": str(view2["sample_id"]),
                    "view1_row_index": int(view1["processed_row_index"]),
                    "view2_row_index": int(view2["processed_row_index"]),
                    "view1_frame_id": int(view1["frame_id"]),
                    "view2_frame_id": int(view2["frame_id"]),
                    "view1_angle_deg": int(view1["angle_deg"]),
                    "view2_angle_deg": int(view2["angle_deg"]),
                }
            )

    batch_seed = _derived_seed(
        f"{BATCH_ORDER_VERSION}|{int(finetune_seed)}|{pair_id}|"
        f"{int(angle_fold)}|{int(epoch)}"
    )
    batch_rng = np.random.Generator(np.random.PCG64(batch_seed))
    scheduled = [scheduled[int(index)] for index in batch_rng.permutation(len(scheduled))]
    for index, row in enumerate(scheduled):
        row["epoch_pair_index"] = index

    view1_usage = Counter(str(row["view1_sample_id"]) for row in scheduled)
    view2_usage = Counter(str(row["view2_sample_id"]) for row in scheduled)
    expected_ids = set(sample_ids)
    cross_frame = all(
        int(row["view1_frame_id"]) != int(row["view2_frame_id"])
        for row in scheduled
    )
    class_counts = Counter(int(row["model_label"]) for row in scheduled)
    unordered_keys = {
        (
            int(row["model_label"]),
            *sorted((str(row["view1_sample_id"]), str(row["view2_sample_id"]))),
        )
        for row in scheduled
    }
    unordered_unique = len(unordered_keys) == len(scheduled)
    all_constraints = (
        set(view1_usage) == expected_ids
        and set(view2_usage) == expected_ids
        and set(view1_usage.values()) == {1}
        and set(view2_usage.values()) == {1}
        and cross_frame
        and unordered_unique
        and len({str(row["pair_id"]) for row in scheduled}) == len(scheduled)
    )
    if not all_constraints:
        raise DataValidationError("dynamic epoch schedule failed its frozen constraints")
    manifest_sha = hashlib.sha256(_render_csv(scheduled)).hexdigest()
    audit = {
        "status": "passed",
        "pair_id": str(pair_id),
        "angle_fold": int(angle_fold),
        "epoch": int(epoch),
        "pair_count": len(scheduled),
        "class_counts": {str(key): int(value) for key, value in sorted(class_counts.items())},
        "class_epoch_seeds": class_seeds,
        "class_matching_attempts": class_matching_attempts,
        "batch_order_seed": batch_seed,
        "view1_usage": dict(sorted(view1_usage.items())),
        "view2_usage": dict(sorted(view2_usage.items())),
        "view1_exactly_once": set(view1_usage.values()) == {1},
        "view2_exactly_once": set(view2_usage.values()) == {1},
        "cross_frame": cross_frame,
        "unordered_pair_unique": unordered_unique,
        "all_constraints_passed": True,
        "epoch_manifest_sha256": manifest_sha,
        "sample_id_population_sha256": _sequence_sha256(sorted(expected_ids)),
        "pair_schedule_version": PAIR_SCHEDULE_VERSION,
        "batch_order_version": BATCH_ORDER_VERSION,
        "maximum_matching_attempts": 4096,
    }
    return scheduled, audit


def build_guided_reference_scores(
    unique_rows: Sequence[Mapping[str, Any]],
    r_values: np.ndarray,
    *,
    epsilon: float,
) -> tuple[
    dict[str, np.ndarray],
    list[np.ndarray],
    list[tuple[str, ...]],
    dict[str, Any],
]:
    """Fit one shared unique-base reconstruction reference per known class."""

    values = np.asarray(r_values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(unique_rows):
        raise DataValidationError("unique-base rows and r values do not align")
    if values.shape[1] < 2 or not np.isfinite(values).all() or epsilon <= 0.0:
        raise DataValidationError("guided reconstruction values are invalid")
    class_count = int(values.shape[1])
    role_indices = {
        role: np.asarray(
            [
                index
                for index, row in enumerate(unique_rows)
                if str(row["experiment_role"]) == role
            ],
            dtype=np.int64,
        )
        for role in ("known_calibration", "surrogate_unknown")
    }
    calibration = role_indices["known_calibration"]
    surrogate = role_indices["surrogate_unknown"]
    if calibration.size == 0 or surrogate.size == 0:
        raise DataValidationError("guided reference roles are empty")
    calibration_labels = np.asarray(
        [int(unique_rows[index]["model_label"]) for index in calibration], dtype=np.int64
    )
    calibration_ids = tuple(str(unique_rows[index]["sample_id"]) for index in calibration)
    references: list[np.ndarray] = []
    reference_ids: list[tuple[str, ...]] = []
    for class_index in range(class_count):
        selected = calibration[calibration_labels == class_index]
        ids = tuple(str(unique_rows[index]["sample_id"]) for index in selected)
        if not ids or len(ids) != len(set(ids)):
            raise DataValidationError("guided reference must contain unique bases per class")
        references.append(np.asarray(values[selected, class_index], dtype=np.float64))
        reference_ids.append(ids)
    calibration_p = cssr_conformal_p_values(
        values[calibration],
        references,
        sample_ids=calibration_ids,
        reference_sample_ids=reference_ids,
        true_labels=calibration_labels,
        leave_one_base_sample_out=True,
    )
    surrogate_p = cssr_conformal_p_values(values[surrogate], references)
    calibration_a = -np.log(calibration_p + float(epsilon))
    surrogate_a = -np.log(surrogate_p + float(epsilon))
    if not np.isfinite(calibration_a).all() or not np.isfinite(surrogate_a).all():
        raise DataValidationError("guided anomaly contains NaN or Inf")
    score_by_sample: dict[str, np.ndarray] = {}
    r_by_sample: dict[str, np.ndarray] = {}
    p_by_sample: dict[str, np.ndarray] = {}
    for role, anomaly, p_values in (
        ("known_calibration", calibration_a, calibration_p),
        ("surrogate_unknown", surrogate_a, surrogate_p),
    ):
        for local_index, unique_index in enumerate(role_indices[role]):
            sample_id = str(unique_rows[int(unique_index)]["sample_id"])
            score_by_sample[sample_id] = anomaly[local_index]
            r_by_sample[sample_id] = values[int(unique_index)]
            p_by_sample[sample_id] = p_values[local_index]
    arrays = {
        "r": values,
        "known_calibration_p": calibration_p,
        "known_calibration_a": calibration_a,
        "surrogate_unknown_p": surrogate_p,
        "surrogate_unknown_a": surrogate_a,
    }
    metadata = {
        "status": "passed",
        "reference_counts": [len(reference) for reference in references],
        "reference_sample_id_hashes": [_sequence_sha256(ids) for ids in reference_ids],
        "shared_reference_across_slots": True,
        "calibration_leave_one_base_sample_out": True,
        "surrogate_unknown_in_reference": False,
        "pair_multiplicity_used": False,
        "score_by_sample": score_by_sample,
        "r_by_sample": r_by_sample,
        "p_by_sample": p_by_sample,
    }
    return arrays, references, reference_ids, metadata


def compute_method_scores(
    method: str,
    fused_logits: np.ndarray,
    guided_anomaly: np.ndarray | None,
    cc_mls_scores: np.ndarray,
) -> dict[str, Any]:
    """Select only the preregistered main score for one method."""

    if method not in METHODS:
        raise DataValidationError(f"unknown E2E CSSR method: {method}")
    logits = np.asarray(fused_logits, dtype=np.float64)
    mls = np.asarray(cc_mls_scores, dtype=np.float64)
    if logits.ndim != 2 or mls.shape != (logits.shape[0],):
        raise DataValidationError("method score inputs do not align")
    prediction = logits.argmax(axis=1).astype(np.int64)
    if method in (Q0_FROZEN_R2_CC_MLS, Q1_CE_FINETUNE_CONTROL):
        main = mls
        main_name = "class_conditional_mls"
        diagnostic_mls: np.ndarray | None = None
    else:
        if guided_anomaly is None:
            raise DataValidationError("CSSR method lacks guided anomaly values")
        anomaly = np.asarray(guided_anomaly, dtype=np.float64)
        if anomaly.shape != (logits.shape[0], 2, logits.shape[1]):
            raise DataValidationError("guided anomaly must be [samples,2,classes]")
        main = anomaly[
            np.arange(logits.shape[0])[:, None],
            np.arange(2)[None, :],
            prediction[:, None],
        ].mean(axis=1)
        main_name = "fusion_guided_reconstruction"
        diagnostic_mls = mls
    if not np.isfinite(main).all():
        raise DataValidationError("main unknown score contains NaN or Inf")
    return {
        "known_prediction": prediction,
        "main_unknown_score": main,
        "main_score_name": main_name,
        "diagnostic_class_conditional_mls": diagnostic_mls,
    }


def _rows_by_pair_method(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_pairs: Sequence[str],
    expected_methods: Sequence[str],
) -> dict[str, dict[str, Mapping[str, Any]]]:
    result: dict[str, dict[str, Mapping[str, Any]]] = {
        pair_id: {} for pair_id in expected_pairs
    }
    for row in rows:
        pair_id = str(row.get("pair_id"))
        method = str(row.get("method"))
        if pair_id not in result or method not in expected_methods:
            raise DataValidationError("gate row has an unexpected pair or method")
        if method in result[pair_id]:
            raise DataValidationError("gate rows contain a duplicate pair/method")
        result[pair_id][method] = row
    if any(set(result[pair_id]) != set(expected_methods) for pair_id in expected_pairs):
        raise DataValidationError("gate rows omit a required pair/method")
    return result


def _identity_map(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_pairs: Sequence[str],
    expected_methods: Sequence[str],
) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    identities_by_pair: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        pair_id = str(row.get("pair_id"))
        method = str(row.get("method"))
        identity = str(row.get("surrogate_identity", row.get("identity", "")))
        if pair_id not in expected_pairs or method not in expected_methods or not identity:
            raise DataValidationError("identity gate row is outside the frozen plan")
        key = (pair_id, method, identity)
        if key in result:
            raise DataValidationError("identity gate rows contain a duplicate")
        result[key] = row
        identities_by_pair[pair_id].add(identity)
    for pair_id in expected_pairs:
        identities = identities_by_pair[pair_id]
        if len(identities) != 2:
            raise DataValidationError("each gate pair must contain two surrogate identities")
        for method in expected_methods:
            if any((pair_id, method, identity) not in result for identity in identities):
                raise DataValidationError("identity gate rows omit a required method")
    return result


def _candidate_gate(
    metric_map: Mapping[str, Mapping[str, Mapping[str, Any]]],
    identity_map: Mapping[tuple[str, str, str], Mapping[str, Any]],
    *,
    pair_ids: Sequence[str],
    candidate: str,
    baseline: str,
    frozen_reference: str,
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    pair_deltas: list[dict[str, Any]] = []
    for pair_id in pair_ids:
        candidate_row = metric_map[pair_id][candidate]
        baseline_row = metric_map[pair_id][baseline]
        frozen_row = metric_map[pair_id][frozen_reference]
        pair_deltas.append(
            {
                "pair_id": pair_id,
                "delta_auroc_vs_q1": float(candidate_row["auroc"])
                - float(baseline_row["auroc"]),
                "delta_oscr_vs_q1": float(candidate_row["oscr"])
                - float(baseline_row["oscr"]),
                "delta_kccr_vs_q1": float(
                    candidate_row["known_correct_acceptance_rate"]
                )
                - float(baseline_row["known_correct_acceptance_rate"]),
                "delta_fpr95_vs_q1": float(candidate_row["fpr95"])
                - float(baseline_row["fpr95"]),
                "delta_auroc_vs_q0": float(candidate_row["auroc"])
                - float(frozen_row["auroc"]),
            }
        )
    if not all(
        math.isfinite(float(value))
        for row in pair_deltas
        for key, value in row.items()
        if key != "pair_id"
    ):
        raise DataValidationError("gate pair delta contains NaN or Inf")
    mean_deltas = {
        key: float(np.mean([float(row[key]) for row in pair_deltas]))
        for key in (
            "delta_auroc_vs_q1",
            "delta_oscr_vs_q1",
            "delta_kccr_vs_q1",
            "delta_fpr95_vs_q1",
            "delta_auroc_vs_q0",
        )
    }
    positive_count = sum(float(row["delta_auroc_vs_q1"]) > 0.0 for row in pair_deltas)
    identity_rows: list[dict[str, Any]] = []
    for (pair_id, method, identity), row in sorted(identity_map.items()):
        if method != candidate or pair_id not in pair_ids:
            continue
        baseline_row = identity_map[(pair_id, baseline, identity)]
        identity_rows.append(
            {
                "pair_id": pair_id,
                "surrogate_identity": identity,
                "candidate_auroc": float(row["auroc"]),
                "q1_auroc": float(baseline_row["auroc"]),
                "delta_auroc_vs_q1": float(row["auroc"])
                - float(baseline_row["auroc"]),
            }
        )
    if len(identity_rows) != 2 * len(pair_ids):
        raise DataValidationError("candidate identity gate population is incomplete")
    minimum_identity_auroc = min(row["candidate_auroc"] for row in identity_rows)
    minimum_identity_delta = min(row["delta_auroc_vs_q1"] for row in identity_rows)
    checks = {
        "mean_auroc_delta_vs_q1": mean_deltas["delta_auroc_vs_q1"]
        + GATE_TOLERANCE
        >= float(gate["minimum_mean_auroc_delta_vs_q1"]),
        "positive_pair_count_vs_q1": positive_count
        >= int(gate["minimum_positive_pair_count_vs_q1"]),
        "mean_oscr_delta_vs_q1": mean_deltas["delta_oscr_vs_q1"]
        + GATE_TOLERANCE
        >= float(gate["minimum_mean_oscr_delta_vs_q1"]),
        "mean_kccr_delta_vs_q1": mean_deltas["delta_kccr_vs_q1"]
        + GATE_TOLERANCE
        >= -float(gate["maximum_mean_kccr_drop_vs_q1"]),
        "mean_fpr95_delta_vs_q1": mean_deltas["delta_fpr95_vs_q1"]
        <= float(gate["maximum_mean_fpr95_increase_vs_q1"])
        + GATE_TOLERANCE,
        "mean_auroc_delta_vs_q0": mean_deltas["delta_auroc_vs_q0"]
        + GATE_TOLERANCE
        >= float(gate["minimum_mean_auroc_delta_vs_q0"]),
        "minimum_identity_auroc": minimum_identity_auroc + GATE_TOLERANCE
        >= float(gate["minimum_identity_auroc"]),
        "minimum_identity_auroc_delta_vs_q1": minimum_identity_delta
        + GATE_TOLERANCE
        >= float(gate["minimum_identity_auroc_delta_vs_q1"]),
    }
    return {
        "candidate": candidate,
        "baseline": baseline,
        "frozen_reference": frozen_reference,
        "pair_deltas": pair_deltas,
        "mean_deltas": mean_deltas,
        "positive_auroc_pair_count": positive_count,
        "identity_evidence": identity_rows,
        "minimum_identity_auroc": minimum_identity_auroc,
        "minimum_identity_auroc_delta_vs_q1": minimum_identity_delta,
        "checks": checks,
        "passed": all(checks.values()),
        "mean_candidate_auroc": float(
            np.mean([float(metric_map[pair][candidate]["auroc"]) for pair in pair_ids])
        ),
    }


def evaluate_pilot_gate(
    rows: Sequence[Mapping[str, Any]],
    identity_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gate = _mapping(config["pilot_gate"], "pilot_gate")
    metric_map = _rows_by_pair_method(
        rows, expected_pairs=PILOT_PAIRS, expected_methods=METHODS
    )
    identities = _identity_map(
        identity_rows, expected_pairs=PILOT_PAIRS, expected_methods=METHODS
    )
    candidates = {
        method: _candidate_gate(
            metric_map,
            identities,
            pair_ids=PILOT_PAIRS,
            candidate=method,
            baseline=str(gate["baseline"]),
            frozen_reference=str(gate["frozen_reference"]),
            gate=gate,
        )
        for method in CSSR_METHODS
    }
    passed = {method for method, result in candidates.items() if result["passed"]}
    replacement_gain = float(gate["replacement_minimum_mean_auroc_gain"])
    selected: str | None = None
    if Q2_E2E_REL_CSSR_1X1 in passed:
        selected = Q2_E2E_REL_CSSR_1X1
        q2_mean = candidates[selected]["mean_candidate_auroc"]
        replacements = [
            method
            for method in (Q3_E2E_ABSREL_CSSR_1X1, Q4_E2E_ABSREL_CSSR_LOCAL3)
            if method in passed
            and candidates[method]["mean_candidate_auroc"] + GATE_TOLERANCE
            >= q2_mean + replacement_gain
        ]
        if replacements:
            selected = sorted(
                replacements,
                key=lambda method: (
                    -float(candidates[method]["mean_candidate_auroc"]),
                    CSSR_METHODS.index(method),
                ),
            )[0]
    elif Q3_E2E_ABSREL_CSSR_1X1 in passed:
        selected = Q3_E2E_ABSREL_CSSR_1X1
        if (
            Q4_E2E_ABSREL_CSSR_LOCAL3 in passed
            and candidates[Q4_E2E_ABSREL_CSSR_LOCAL3]["mean_candidate_auroc"]
            + GATE_TOLERANCE
            >= candidates[selected]["mean_candidate_auroc"] + replacement_gain
        ):
            selected = Q4_E2E_ABSREL_CSSR_LOCAL3
    elif Q4_E2E_ABSREL_CSSR_LOCAL3 in passed:
        selected = Q4_E2E_ABSREL_CSSR_LOCAL3
    signal = (
        str(gate["no_candidate_label"])
        if selected is None
        else LABEL_BY_METHOD[selected]
    )
    return {
        "signal": signal,
        "selected_method": selected,
        "confirmation_allowed": selected is not None,
        "candidate_gates": candidates,
        "complexity_order": list(CSSR_METHODS),
        "final_unknown_test_authorized": False,
    }


def evaluate_confirmation_gate(
    rows: Sequence[Mapping[str, Any]],
    identity_rows: Sequence[Mapping[str, Any]],
    selected_method: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if selected_method not in CSSR_METHODS:
        raise DataValidationError("confirmation requires one selected CSSR candidate")
    gate = _mapping(config["confirmation_gate"], "confirmation_gate")
    expected_methods = (
        Q0_FROZEN_R2_CC_MLS,
        Q1_CE_FINETUNE_CONTROL,
        selected_method,
    )
    metric_map = _rows_by_pair_method(
        rows, expected_pairs=CONFIRMATION_PAIRS, expected_methods=expected_methods
    )
    identities = _identity_map(
        identity_rows,
        expected_pairs=CONFIRMATION_PAIRS,
        expected_methods=expected_methods,
    )
    evidence = _candidate_gate(
        metric_map,
        identities,
        pair_ids=CONFIRMATION_PAIRS,
        candidate=selected_method,
        baseline=str(gate["baseline"]),
        frozen_reference=str(gate["frozen_reference"]),
        gate=gate,
    )
    return {
        **evidence,
        "selected_method": selected_method,
        "decision": str(gate["success_label"] if evidence["passed"] else gate["failure_label"]),
        "final_unknown_test_authorized": False,
    }


def build_phase_plan(
    config: Mapping[str, Any],
    phase: str,
    selected_method: str | None = None,
) -> list[dict[str, Any]]:
    if phase == "smoke":
        pair_ids = (str(config["data"]["smoke"]["pair_id"]),)
        methods = tuple(config["data"]["smoke"]["methods"])
        mode = "smoke"
    elif phase == "pilot":
        pair_ids = tuple(config["classes"]["pilot_pairs"])
        methods = TRAINABLE_METHODS
        mode = "full"
    elif phase == "confirmation":
        if selected_method not in CSSR_METHODS:
            raise DataValidationError("confirmation plan requires the selected CSSR method")
        pair_ids = tuple(config["classes"]["confirmation_pairs"])
        methods = (Q1_CE_FINETUNE_CONTROL, str(selected_method))
        mode = "full"
    else:
        raise DataValidationError("phase must be smoke, pilot, or confirmation")
    return [
        {
            "phase": phase,
            "mode": mode,
            "pair_id": pair_id,
            "angle_fold": ANGLE_FOLD,
            "model_seed": MODEL_SEED,
            "finetune_seed": FINETUNE_SEED,
            "method": method,
        }
        for pair_id in pair_ids
        for method in methods
    ]


def _state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _selected_state(
    model: FGMVCSSRE2EModel, prefixes: Sequence[str]
) -> dict[str, torch.Tensor]:
    return {
        name: value
        for name, value in model.state_dict().items()
        if any(name.startswith(prefix) for prefix in prefixes)
    }


def _common_r2_state_sha256(model: FGMVCSSRE2EModel) -> str:
    return _state_sha256(
        _selected_state(
            model,
            (
                "r2_model.encoder.",
                "r2_model.global_head.",
            ),
        )
    )


def _frozen_prefix_sha256(model: FGMVCSSRE2EModel) -> str:
    return _state_sha256(
        _selected_state(
            model,
            (
                "r2_model.encoder.stem.",
                "r2_model.encoder.stages.0.",
                "r2_model.encoder.stages.1.",
            ),
        )
    )


def task_source_hashes(project_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in TASK_SOURCE_FILES:
        path = project_root / relative
        if not path.is_file():
            raise DataValidationError(f"missing E2E task source: {relative}")
        result[relative] = file_sha256(path)
    return result


def _configure_runtime(config: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    numerical = _configure_numerical_runtime(config)
    if device.type != "cuda":
        raise DataValidationError("E2E CSSR smoke and formal units require CUDA")
    observed = torch.cuda.get_device_name(device)
    if observed != str(config["runtime"]["expected_gpu_model"]):
        raise DataValidationError(
            f"E2E CSSR requires {config['runtime']['expected_gpu_model']}; observed {observed}"
        )
    return {
        "device": str(device),
        "device_name": observed,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "amp": False,
        "torch_compile": False,
        "num_workers": int(config["runtime"]["num_workers"]),
        **numerical,
    }


def _git_environment(project_root: Path, device: torch.device) -> dict[str, Any]:
    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "git_status_porcelain": git("status", "--porcelain"),
    }


def _normalized_unique_base_inputs(
    bundle: Any,
    prepared: Any,
    unique_rows: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    indices = np.asarray(
        [int(row["processed_row_index"]) for row in unique_rows], dtype=np.int64
    )
    values = np.asarray(bundle.profiles[indices], dtype=np.float64)
    values = (values - float(prepared.normalization.mean)) / float(
        prepared.normalization.std
    )
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (len(unique_rows), 601) or not np.isfinite(result).all():
        raise DataValidationError("normalized unique-base inputs are invalid")
    return result


def _materialize_schedule_inputs(
    schedule: Sequence[Mapping[str, Any]],
    *,
    unique_rows: Sequence[Mapping[str, Any]],
    unique_inputs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    by_id = {
        str(row["sample_id"]): index for index, row in enumerate(unique_rows)
    }
    view1 = np.asarray(
        [by_id[str(row["view1_sample_id"])] for row in schedule], dtype=np.int64
    )
    view2 = np.asarray(
        [by_id[str(row["view2_sample_id"])] for row in schedule], dtype=np.int64
    )
    inputs = np.stack([unique_inputs[view1], unique_inputs[view2]], axis=1)
    labels = np.asarray([int(row["model_label"]) for row in schedule], dtype=np.int64)
    if inputs.shape != (len(schedule), 2, 601) or not np.isfinite(inputs).all():
        raise DataValidationError("dynamic schedule inputs are invalid")
    return np.asarray(inputs, dtype=np.float32), labels


def _build_optimizer(
    model: FGMVCSSRE2EModel, config: Mapping[str, Any]
) -> torch.optim.Optimizer:
    groups = model.trainable_parameter_groups()
    training = config["training"]
    specifications = (
        (
            "last_stage",
            "lr_last_stage",
            "weight_decay_last_stage",
        ),
        (
            "projection_and_ce_head",
            "lr_projection_and_ce_head",
            "weight_decay_projection_and_head",
        ),
        (
            "autoencoders",
            "lr_autoencoders",
            "weight_decay_autoencoders",
        ),
    )
    optimizer_groups = []
    for name, lr_name, decay_name in specifications:
        parameters = groups[name]
        if not parameters:
            continue
        optimizer_groups.append(
            {
                "params": parameters,
                "lr": float(training[lr_name]),
                "weight_decay": float(training[decay_name]),
                "group_name": name,
                "base_lr": float(training[lr_name]),
            }
        )
    return torch.optim.AdamW(optimizer_groups)


def _learning_rate_factor(epoch: int, *, warmup_epochs: int, total_epochs: int) -> float:
    if not 1 <= epoch <= total_epochs:
        raise DataValidationError("epoch is outside the frozen LR schedule")
    if epoch <= warmup_epochs:
        return float(epoch) / float(warmup_epochs)
    # Apply all 20 optimizer updates at a positive LR.  Zero is the conceptual
    # boundary after the final update, not the factor used for epoch 20 itself.
    progress = float(epoch - warmup_epochs - 1) / float(total_epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _set_optimizer_lrs(
    optimizer: torch.optim.Optimizer, factor: float
) -> dict[str, float]:
    result: dict[str, float] = {}
    for group in optimizer.param_groups:
        name = str(group["group_name"])
        value = float(group["base_lr"]) * float(factor)
        group["lr"] = value
        result[name] = value
    return result


def _group_gradient_norms(
    loss: torch.Tensor,
    parameter_groups: Mapping[str, Sequence[nn.Parameter]],
    *,
    retain_graph: bool,
) -> dict[str, float]:
    names = tuple(parameter_groups)
    flattened = tuple(
        parameter for name in names for parameter in parameter_groups[name]
    )
    if not flattened or not loss.requires_grad:
        return {name: 0.0 for name in names}
    gradients = torch.autograd.grad(
        loss,
        flattened,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    result: dict[str, float] = {}
    offset = 0
    for name in names:
        squared = loss.new_zeros(())
        count = len(parameter_groups[name])
        for gradient in gradients[offset : offset + count]:
            if gradient is not None:
                squared = squared + gradient.detach().square().sum()
        offset += count
        value = float(torch.sqrt(squared).item())
        if not math.isfinite(value):
            raise DataValidationError("gradient norm is NaN or Inf")
        result[name] = value
    return result


def infer_e2e_model(
    model: FGMVCSSRE2EModel,
    inputs: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    values = np.asarray(inputs, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (2, 601):
        raise DataValidationError("E2E inference inputs must be [n,2,601]")
    collected: dict[str, list[np.ndarray]] = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            output = model(torch.from_numpy(values[start : start + batch_size]).to(device))
            collected["fused_logits"].append(
                output.fused_logits.detach().cpu().numpy().astype(np.float64)
            )
            collected["fused_features"].append(
                output.fused_features.detach().cpu().numpy().astype(np.float32)
            )
            collected["per_view_features"].append(
                output.per_view_features.detach().cpu().numpy().astype(np.float32)
            )
            if output.normalized_reconstruction_errors is not None:
                collected["r"].append(
                    output.normalized_reconstruction_errors.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
    result = {name: np.concatenate(items, axis=0) for name, items in collected.items()}
    required = {"fused_logits", "fused_features", "per_view_features"}
    if not required <= set(result) or not all(np.isfinite(value).all() for value in result.values()):
        raise DataValidationError("E2E inference produced invalid arrays")
    if model.variant in CSSR_VARIANTS and "r" not in result:
        raise DataValidationError("CSSR inference omitted reconstruction errors")
    return result


def _softmax_numpy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def calibration_diagnostics(
    arrays: Mapping[str, np.ndarray],
    labels: np.ndarray,
    *,
    model: FGMVCSSRE2EModel,
    epoch0_arrays: Mapping[str, np.ndarray],
    ece_bins: int,
) -> dict[str, float]:
    logits = np.asarray(arrays["fused_logits"], dtype=np.float64)
    true = np.asarray(labels, dtype=np.int64)
    probabilities = _softmax_numpy(logits)
    prediction = logits.argmax(axis=1)
    selected = probabilities[np.arange(len(true)), true]
    nll = float(-np.log(np.clip(selected, 1.0e-300, 1.0)).mean())
    one_hot = np.eye(logits.shape[1], dtype=np.float64)[true]
    brier = float(np.square(probabilities - one_hot).sum(axis=1).mean())
    confidence = probabilities.max(axis=1)
    correct = prediction == true
    edges = np.linspace(0.0, 1.0, ece_bins + 1)
    ece = 0.0
    for index in range(ece_bins):
        if index == ece_bins - 1:
            selected_bin = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            selected_bin = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if np.any(selected_bin):
            ece += float(np.mean(selected_bin)) * abs(
                float(np.mean(correct[selected_bin])) - float(np.mean(confidence[selected_bin]))
            )
    top_two = np.partition(logits, kth=logits.shape[1] - 2, axis=1)[:, -2:]
    top_two.sort(axis=1)
    current_features = np.asarray(arrays["fused_features"], dtype=np.float64)
    initial_features = np.asarray(epoch0_arrays["fused_features"], dtype=np.float64)
    initial_probabilities = _softmax_numpy(
        np.asarray(epoch0_arrays["fused_logits"], dtype=np.float64)
    )
    kl = np.sum(
        initial_probabilities
        * (
            np.log(np.clip(initial_probabilities, 1.0e-300, 1.0))
            - np.log(np.clip(probabilities, 1.0e-300, 1.0))
        ),
        axis=1,
    )
    result = {
        "accuracy": accuracy_score(true, prediction),
        "macro_f1": macro_f1_score(true, prediction, range(logits.shape[1])),
        "nll": nll,
        "brier": brier,
        "ece": float(ece),
        "mean_max_logit": float(logits.max(axis=1).mean()),
        "mean_top1_top2_logit_margin": float((top_two[:, 1] - top_two[:, 0]).mean()),
        "mean_single_view_feature_norm": float(
            np.linalg.norm(np.asarray(arrays["per_view_features"], dtype=np.float64), axis=2).mean()
        ),
        "mean_fused_feature_norm": float(np.linalg.norm(current_features, axis=1).mean()),
        "ce_head_weight_norm": float(model.global_head.weight.detach().norm().item()),
        "mean_fused_feature_l2_drift_from_epoch0": float(
            np.linalg.norm(current_features - initial_features, axis=1).mean()
        ),
        "mean_kl_epoch0_to_current": float(kl.mean()),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise DataValidationError("known calibration diagnostic is NaN or Inf")
    return result


def train_e2e_method(
    model: FGMVCSSRE2EModel,
    *,
    unique_rows: Sequence[Mapping[str, Any]],
    unique_inputs: np.ndarray,
    prepared: Any,
    pair_id: str,
    config: Mapping[str, Any],
    device: torch.device,
    smoke: bool,
    frozen_r2_arrays: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    """Run the one frozen 20-epoch (or diagnostic one-epoch) training path."""

    training = config["training"]
    epochs = int(config["data"]["smoke"]["epochs"] if smoke else training["epochs"])
    batch_size = int(training["batch_size_pairs"])
    model = model.to(device)
    optimizer = _build_optimizer(model, config)
    _set_determinism(FINETUNE_SEED, bool(config["runtime"]["deterministic_algorithms"]))
    epoch0_common_hash = _common_r2_state_sha256(model)
    frozen_before = _frozen_prefix_sha256(model)
    ae_initial_hash = (
        None
        if model.cssr_core is None
        else _state_sha256(model.cssr_core.state_dict())
    )
    epoch0_arrays = infer_e2e_model(
        model,
        prepared.inputs["known_calibration"],
        device=device,
        batch_size=batch_size,
    )
    frozen_logits = np.asarray(
        frozen_r2_arrays["known_calibration"]["global_logits"], dtype=np.float64
    )
    frozen_features = np.asarray(
        frozen_r2_arrays["known_calibration"]["fused_features"], dtype=np.float32
    )
    if not np.array_equal(epoch0_arrays["fused_logits"], frozen_logits):
        raise DataValidationError("epoch-0 wrapper logits differ from frozen R2")
    if not np.array_equal(epoch0_arrays["fused_features"], frozen_features):
        raise DataValidationError("epoch-0 wrapper features differ from frozen R2")
    confidence_log: list[dict[str, Any]] = [
        {
            "epoch": 0,
            **calibration_diagnostics(
                epoch0_arrays,
                prepared.labels["known_calibration"],
                model=model,
                epoch0_arrays=epoch0_arrays,
                ece_bins=int(config["diagnostics"]["ece_bins"]),
            ),
            "diagnostic_only": True,
            "surrogate_unknown_used": False,
        }
    ]
    stage_parameters = tuple(model.trainable_parameter_groups()["last_stage"])
    ae_parameters = tuple(model.trainable_parameter_groups()["autoencoders"])
    gradient_config = config["loss"]["gradient_audit"]
    violation_streak = {name: 0 for name in ("relative", "absolute", "separation")}
    training_log: list[dict[str, Any]] = []
    schedules: list[list[dict[str, Any]]] = []
    schedule_audits: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        schedule, schedule_audit = build_epoch_pair_schedule(
            unique_rows,
            pair_id=pair_id,
            angle_fold=ANGLE_FOLD,
            epoch=epoch,
            finetune_seed=FINETUNE_SEED,
        )
        epoch_inputs, epoch_labels = _materialize_schedule_inputs(
            schedule, unique_rows=unique_rows, unique_inputs=unique_inputs
        )
        schedules.append(schedule)
        schedule_audits.append(schedule_audit)
        factor = _learning_rate_factor(
            epoch,
            warmup_epochs=int(training["warmup_epochs"]),
            total_epochs=int(training["epochs"]),
        )
        learning_rates = _set_optimizer_lrs(optimizer, factor)
        model.train()
        if (
            model.encoder.stem.training
            or model.encoder.stages[0].training
            or model.encoder.stages[1].training
        ):
            raise DataValidationError("a frozen early R2 module entered train mode")
        totals = defaultdict(float)
        batch_count = 0
        example_count = 0
        gradient_sums = defaultdict(float)
        ratio_sums = defaultdict(float)
        for start in range(0, len(epoch_labels), batch_size):
            inputs = torch.from_numpy(epoch_inputs[start : start + batch_size]).to(device)
            labels = torch.from_numpy(epoch_labels[start : start + batch_size]).to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(inputs)
            losses = model.loss(output, labels)
            terms = {
                "classification": losses.classification_loss,
                "relative": losses.relative_loss,
                "absolute": losses.absolute_loss,
                "separation": losses.separation_loss,
            }
            if not torch.isfinite(losses.total_loss):
                raise DataValidationError("E2E training loss is NaN or Inf")
            raw_stage_norms: dict[str, float] = {}
            raw_ae_norms: dict[str, float] = {}
            for name, term in terms.items():
                norms = _group_gradient_norms(
                    term,
                    {"last_stage": stage_parameters, "autoencoders": ae_parameters},
                    retain_graph=True,
                )
                raw_stage_norms[name] = norms["last_stage"]
                raw_ae_norms[name] = norms["autoencoders"]
                weight = float(losses.component_weights[name])
                gradient_sums[f"{name}_last_stage_raw"] += raw_stage_norms[name]
                gradient_sums[f"{name}_last_stage_weighted"] += weight * raw_stage_norms[name]
                gradient_sums[f"{name}_ae_raw"] += raw_ae_norms[name]
                gradient_sums[f"{name}_ae_weighted"] += weight * raw_ae_norms[name]
            ce_norm = raw_stage_norms["classification"]
            denominator = max(ce_norm, float(gradient_config["denominator_floor"]))
            for name in ("relative", "absolute", "separation"):
                if float(losses.component_weights[name]) > 0.0:
                    ratio_sums[name] += (
                        float(losses.component_weights[name]) * raw_stage_norms[name]
                    ) / denominator
            losses.total_loss.backward()
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise DataValidationError("E2E training gradient is NaN or Inf")
            clip_value = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                max_norm=float(training["gradient_clip_norm"]),
            )
            if not torch.isfinite(clip_value):
                raise DataValidationError("E2E clipped gradient norm is NaN or Inf")
            optimizer.step()
            if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
                raise DataValidationError("E2E parameter is NaN or Inf")
            count = int(labels.numel())
            example_count += count
            batch_count += 1
            totals["total_loss"] += float(losses.total_loss.detach()) * count
            for name, term in terms.items():
                totals[f"{name}_loss"] += float(term.detach()) * count
            totals["train_correct"] += int(
                (output.fused_logits.argmax(dim=1) == labels).sum().item()
            )
            if losses.true_class_r is not None:
                totals["true_class_r"] += float(losses.true_class_r.detach().sum())
                totals["nearest_wrong_class_r"] += float(
                    losses.nearest_wrong_class_r.detach().sum()
                )
                totals["reconstruction_margin"] += float(
                    losses.reconstruction_margin.detach().sum()
                )
                totals["r_element_count"] += int(losses.true_class_r.numel())
        if example_count != len(epoch_labels) or batch_count <= 0:
            raise DataValidationError("dynamic epoch did not consume every pair exactly once")
        epoch_ratios: dict[str, float | None] = {}
        for name in ("relative", "absolute", "separation"):
            if float(model.component_weights[name]) > 0.0:
                ratio = ratio_sums[name] / batch_count
                epoch_ratios[name] = ratio
                violation_streak[name] = (
                    violation_streak[name] + 1
                    if ratio > float(gradient_config["maximum_ratio"])
                    else 0
                )
            else:
                epoch_ratios[name] = None
                violation_streak[name] = 0
        if max(violation_streak.values()) >= int(
            gradient_config["consecutive_epoch_mean_violations_to_fail"]
        ):
            raise DataValidationError(
                "weighted auxiliary gradient exceeded the frozen 100x stability limit"
            )
        if _frozen_prefix_sha256(model) != frozen_before:
            raise DataValidationError("frozen R2 parameters or buffers changed")
        current_arrays = infer_e2e_model(
            model,
            prepared.inputs["known_calibration"],
            device=device,
            batch_size=batch_size,
        )
        diagnostics = calibration_diagnostics(
            current_arrays,
            prepared.labels["known_calibration"],
            model=model,
            epoch0_arrays=epoch0_arrays,
            ece_bins=int(config["diagnostics"]["ece_bins"]),
        )
        confidence_log.append(
            {
                "epoch": epoch,
                **diagnostics,
                "diagnostic_only": True,
                "surrogate_unknown_used": False,
            }
        )
        r_count = int(totals.get("r_element_count", 0))
        training_log.append(
            {
                "epoch": epoch,
                "method": model.variant,
                "learning_rate_factor": factor,
                "learning_rates": learning_rates,
                "train_total_loss": totals["total_loss"] / example_count,
                "train_classification_loss": totals["classification_loss"] / example_count,
                "train_relative_loss": totals["relative_loss"] / example_count,
                "train_absolute_loss": totals["absolute_loss"] / example_count,
                "train_separation_loss": totals["separation_loss"] / example_count,
                "train_accuracy": totals["train_correct"] / example_count,
                "train_true_class_r": None if r_count == 0 else totals["true_class_r"] / r_count,
                "train_nearest_wrong_class_r": None if r_count == 0 else totals["nearest_wrong_class_r"] / r_count,
                "train_reconstruction_margin": None if r_count == 0 else totals["reconstruction_margin"] / r_count,
                "gradient_norm_epoch_batch_means": {
                    key: value / batch_count for key, value in sorted(gradient_sums.items())
                },
                "weighted_auxiliary_to_ce_last_stage_ratio_epoch_batch_means": epoch_ratios,
                "gradient_ratio_violation_streak": dict(violation_streak),
                "pair_schedule_sha256": schedule_audit["epoch_manifest_sha256"],
                "pair_count": len(schedule),
                "elapsed_seconds": time.perf_counter() - started,
                "checkpoint_selected_for_performance": False,
                "known_calibration_used_for_training": False,
                "surrogate_unknown_used_for_training": False,
                "final_unknown_used": False,
            }
        )
    frozen_after = _frozen_prefix_sha256(model)
    schedule_hash = _sequence_sha256(
        audit["epoch_manifest_sha256"] for audit in schedule_audits
    )
    return {
        "model": model.eval(),
        "training_log": training_log,
        "confidence_log": confidence_log,
        "schedules": schedules,
        "schedule_audits": schedule_audits,
        "epoch0_arrays": epoch0_arrays,
        "audit": {
            "status": "passed",
            "method": model.variant,
            "epochs": epochs,
            "formal_checkpoint_epoch": None if smoke else int(training["formal_checkpoint_epoch"]),
            "checkpoint_selection": "fixed_final_epoch",
            "early_stopping": False,
            "epoch0_common_r2_state_sha256": epoch0_common_hash,
            "ae_initial_state_sha256": ae_initial_hash,
            "frozen_prefix_state_sha256_before": frozen_before,
            "frozen_prefix_state_sha256_after": frozen_after,
            "frozen_prefix_unchanged": frozen_before == frozen_after,
            "schedule_sha256": schedule_hash,
            "epoch_schedule_sha256": [
                audit["epoch_manifest_sha256"] for audit in schedule_audits
            ],
            "train_pairs_per_epoch": len(schedules[0]),
            "train_unique_base_sample_count": sum(
                str(row["experiment_role"]) == "train_known" for row in unique_rows
            ),
            "all_parameters_finite": all(
                bool(torch.isfinite(parameter).all()) for parameter in model.parameters()
            ),
            "known_calibration_used_for_training": False,
            "surrogate_unknown_used_for_training": False,
            "final_unknown_used": False,
            "even_angle_test_used": False,
        },
    }


def _evaluation_role_indices(prepared: Any, *, smoke: bool, config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if smoke:
        per_class = int(config["data"]["smoke"]["evaluation_pairs_per_class"])
        return {
            role: _smoke_pair_indices(prepared, role, per_class)
            for role in ("known_calibration", "surrogate_unknown")
        }
    return {
        role: np.arange(len(prepared.labels[role]), dtype=np.int64)
        for role in ("known_calibration", "surrogate_unknown")
    }


def _class_conditional_mls_for_roles(
    *,
    full_calibration_logits: np.ndarray,
    full_calibration_labels: np.ndarray,
    full_calibration_pair_ids: Sequence[str],
    role_logits: Mapping[str, np.ndarray],
    role_pair_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, np.ndarray]:
    reference_logits = np.asarray(full_calibration_logits, dtype=np.float64)
    reference_labels = np.asarray(full_calibration_labels, dtype=np.int64)
    reference_prediction = reference_logits.argmax(axis=1)
    reference_nonconformity = -reference_logits.max(axis=1)
    result: dict[str, np.ndarray] = {}
    for role in ("known_calibration", "surrogate_unknown"):
        logits = np.asarray(role_logits[role], dtype=np.float64)
        prediction = logits.argmax(axis=1)
        leave_one_out = role == "known_calibration"
        result[role] = compute_class_conditional_mls_scores(
            -logits.max(axis=1),
            prediction,
            reference_nonconformity=reference_nonconformity,
            reference_true_labels=reference_labels,
            reference_predicted_labels=reference_prediction,
            query_pair_ids=(
                tuple(str(row["pair_id"]) for row in role_pair_rows[role])
                if leave_one_out
                else None
            ),
            reference_pair_ids=(
                tuple(str(value) for value in full_calibration_pair_ids)
                if leave_one_out
                else None
            ),
            leave_one_out=leave_one_out,
        )
    return result


def _pair_guided_anomaly(
    rows: Sequence[Mapping[str, Any]],
    score_by_sample: Mapping[str, np.ndarray],
) -> np.ndarray:
    values = np.asarray(
        [
            [
                score_by_sample[str(row["view1_sample_id"])],
                score_by_sample[str(row["view2_sample_id"])],
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    if values.shape != (len(rows), 2, 5) or not np.isfinite(values).all():
        raise DataValidationError("pair guided anomaly tensor is invalid")
    return values


def _evaluate_score_arrays(
    *,
    prepared: Any,
    role_indices: Mapping[str, np.ndarray],
    score_arrays: Mapping[str, Mapping[str, Any]],
    acceptance_rate: float,
) -> dict[str, float]:
    known = score_arrays["known_calibration"]
    unknown = score_arrays["surrogate_unknown"]
    values = evaluate_open_set(
        known_true=np.asarray(prepared.labels["known_calibration"])[
            role_indices["known_calibration"]
        ],
        known_pred=np.asarray(known["known_prediction"], dtype=np.int64),
        known_unknown_scores=np.asarray(known["main_unknown_score"], dtype=np.float64),
        unknown_pred=np.asarray(unknown["known_prediction"], dtype=np.int64),
        unknown_unknown_scores=np.asarray(unknown["main_unknown_score"], dtype=np.float64),
        known_validation_scores=np.asarray(known["main_unknown_score"], dtype=np.float64),
        known_class_count=5,
        known_acceptance_rate=float(acceptance_rate),
    )
    if any(key not in values for key in REPORT_METRIC_KEYS):
        raise DataValidationError("open-set evaluator omitted a frozen metric")
    if not all(math.isfinite(float(value)) for value in values.values()):
        raise DataValidationError("open-set metric is NaN or Inf")
    return {key: float(value) for key, value in values.items()}


def _build_method_prediction_rows(
    *,
    method: str,
    prepared: Any,
    role_indices: Mapping[str, np.ndarray],
    role_pair_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    role_logits: Mapping[str, np.ndarray],
    role_scores: Mapping[str, Mapping[str, Any]],
    metrics: Mapping[str, float],
    reference_metadata: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    threshold = float(metrics["threshold"])
    score_by_sample = None if reference_metadata is None else reference_metadata["score_by_sample"]
    r_by_sample = None if reference_metadata is None else reference_metadata["r_by_sample"]
    p_by_sample = None if reference_metadata is None else reference_metadata["p_by_sample"]
    for role in ("known_calibration", "surrogate_unknown"):
        indices = role_indices[role]
        logits = np.asarray(role_logits[role], dtype=np.float64)
        scores = role_scores[role]
        prediction = np.asarray(scores["known_prediction"], dtype=np.int64)
        unknown_score = np.asarray(scores["main_unknown_score"], dtype=np.float64)
        diagnostic = scores["diagnostic_class_conditional_mls"]
        for local_index, source_index in enumerate(indices):
            source = role_pair_rows[role][local_index]
            predicted = int(prediction[local_index])
            rejected = bool(unknown_score[local_index] > threshold)
            view1_id = str(source["view1_sample_id"])
            view2_id = str(source["view2_sample_id"])
            row = {
                "method": method,
                "pair_id": str(source["pair_id"]),
                "evaluation_role": role,
                "class_name": str(source["class_name"]),
                "surrogate_identity": (
                    str(source["class_name"]) if role == "surrogate_unknown" else ""
                ),
                "true_label": int(prepared.labels[role][int(source_index)]),
                "predicted_known_label": predicted,
                "predicted_known_class_name": str(prepared.train_class_order[predicted]),
                "fused_logits": json.dumps(logits[local_index].tolist(), separators=(",", ":")),
                "main_score_name": str(scores["main_score_name"]),
                "unknown_score": float(unknown_score[local_index]),
                "threshold": threshold,
                "rejected": rejected,
                "open_set_prediction": 5 if rejected else predicted,
                "diagnostic_class_conditional_mls": (
                    "" if diagnostic is None else float(np.asarray(diagnostic)[local_index])
                ),
                "view1_sample_id": view1_id,
                "view2_sample_id": view2_id,
                "view1_angle_deg": int(source["view1_angle_deg"]),
                "view2_angle_deg": int(source["view2_angle_deg"]),
                "view1_frame_id": int(source["view1_frame_id"]),
                "view2_frame_id": int(source["view2_frame_id"]),
                "view1_r": "" if r_by_sample is None else json.dumps(r_by_sample[view1_id].tolist(), separators=(",", ":")),
                "view2_r": "" if r_by_sample is None else json.dumps(r_by_sample[view2_id].tolist(), separators=(",", ":")),
                "view1_p_value": "" if p_by_sample is None else json.dumps(p_by_sample[view1_id].tolist(), separators=(",", ":")),
                "view2_p_value": "" if p_by_sample is None else json.dumps(p_by_sample[view2_id].tolist(), separators=(",", ":")),
                "view1_a": "" if score_by_sample is None else json.dumps(score_by_sample[view1_id].tolist(), separators=(",", ":")),
                "view2_a": "" if score_by_sample is None else json.dumps(score_by_sample[view2_id].tolist(), separators=(",", ":")),
                "final_unknown_used": False,
                "even_angle_test_used": False,
            }
            rows.append(row)
    return rows


def recompute_method_metrics_from_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    known_class_count: int = 5,
    known_acceptance_rate: float = 0.95,
) -> dict[str, float]:
    known = [row for row in rows if str(row["evaluation_role"]) == "known_calibration"]
    unknown = [row for row in rows if str(row["evaluation_role"]) == "surrogate_unknown"]
    if not known or not unknown:
        raise DataValidationError("prediction rows lack known or surrogate samples")
    values = evaluate_open_set(
        known_true=np.asarray([int(row["true_label"]) for row in known], dtype=np.int64),
        known_pred=np.asarray([int(row["predicted_known_label"]) for row in known], dtype=np.int64),
        known_unknown_scores=np.asarray([float(row["unknown_score"]) for row in known]),
        unknown_pred=np.asarray([int(row["predicted_known_label"]) for row in unknown], dtype=np.int64),
        unknown_unknown_scores=np.asarray([float(row["unknown_score"]) for row in unknown]),
        known_validation_scores=np.asarray([float(row["unknown_score"]) for row in known]),
        known_class_count=known_class_count,
        known_acceptance_rate=known_acceptance_rate,
    )
    return {key: float(value) for key, value in values.items()}


def _metrics_exact(
    expected: Mapping[str, Any], observed: Mapping[str, Any], *, context: str
) -> None:
    if set(expected) != set(observed):
        raise DataValidationError(f"{context} metric keys changed")
    for key in expected:
        if not math.isclose(
            float(expected[key]), float(observed[key]), rel_tol=0.0, abs_tol=1.0e-15
        ):
            raise DataValidationError(f"{context}.{key} does not reproduce")


def build_identity_and_absorption_rows(
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    pair_id: str,
    train_class_order: Sequence[str],
    acceptance_rate: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    known = [row for row in prediction_rows if str(row["evaluation_role"]) == "known_calibration"]
    unknown = [row for row in prediction_rows if str(row["evaluation_role"]) == "surrogate_unknown"]
    identities = tuple(sorted({str(row["class_name"]) for row in unknown}))
    if len(identities) != 2:
        raise DataValidationError("one surrogate pair must contain exactly two identities")
    known_true = np.asarray([int(row["true_label"]) for row in known], dtype=np.int64)
    known_pred = np.asarray([int(row["predicted_known_label"]) for row in known], dtype=np.int64)
    known_scores = np.asarray([float(row["unknown_score"]) for row in known])
    identity_rows: list[dict[str, Any]] = []
    absorption_rows: list[dict[str, Any]] = []
    for identity in identities:
        selected = [row for row in unknown if str(row["class_name"]) == identity]
        metrics = evaluate_open_set(
            known_true=known_true,
            known_pred=known_pred,
            known_unknown_scores=known_scores,
            unknown_pred=np.asarray([int(row["predicted_known_label"]) for row in selected]),
            unknown_unknown_scores=np.asarray([float(row["unknown_score"]) for row in selected]),
            known_validation_scores=known_scores,
            known_class_count=len(train_class_order),
            known_acceptance_rate=float(acceptance_rate),
        )
        identity_rows.append(
            {
                "pair_id": pair_id,
                "method": method,
                "surrogate_identity": identity,
                **{key: float(metrics[key]) for key in REPORT_METRIC_KEYS},
                "threshold": float(metrics["threshold"]),
            }
        )
        accepted = [row for row in selected if not _csv_bool(row["rejected"])]
        counts = Counter(str(row["predicted_known_class_name"]) for row in accepted)
        for known_class in train_class_order:
            count = int(counts.get(str(known_class), 0))
            absorption_rows.append(
                {
                    "pair_id": pair_id,
                    "method": method,
                    "surrogate_identity": identity,
                    "absorbed_as_known_identity": str(known_class),
                    "false_accept_count": count,
                    "total_surrogate_count": len(selected),
                    "total_false_accept_count": len(accepted),
                    "rate_over_all_surrogate": count / len(selected),
                    "composition_within_false_accepts": 0.0 if not accepted else count / len(accepted),
                }
            )
    ddg = {
        "DDG-1000_absorbed_as_DDG-112": sum(
            row["surrogate_identity"] == "DDG-1000"
            and row["absorbed_as_known_identity"] == "DDG-112"
            and int(row["false_accept_count"]) > 0
            and int(row["false_accept_count"])
            for row in absorption_rows
        ),
        "DDG-112_absorbed_as_DDG-1000": sum(
            row["surrogate_identity"] == "DDG-112"
            and row["absorbed_as_known_identity"] == "DDG-1000"
            and int(row["false_accept_count"]) > 0
            and int(row["false_accept_count"])
            for row in absorption_rows
        ),
    }
    return identity_rows, absorption_rows, {
        "pair_id": pair_id,
        "method": method,
        "surrogate_identities": list(identities),
        "ddg_bidirectional_absorption": ddg,
        "marvel_cape_reverse_diagnostic": {
            row["surrogate_identity"]: {
                item["absorbed_as_known_identity"]: int(item["false_accept_count"])
                for item in absorption_rows
                if item["surrogate_identity"] == row["surrogate_identity"]
            }
            for row in identity_rows
            if row["surrogate_identity"] in {"油气轮MARVEL CRANE", "迷你好望角型散货船"}
        },
        "angle_metadata_used_by_model": False,
        "final_unknown_used": False,
    }


def evaluate_unit_model(
    *,
    model: FGMVCSSRE2EModel,
    prepared: Any,
    unique_rows: Sequence[Mapping[str, Any]],
    unique_inputs: np.ndarray,
    frozen_r2_arrays: Mapping[str, Mapping[str, np.ndarray]],
    pair_id: str,
    config: Mapping[str, Any],
    device: torch.device,
    smoke: bool,
) -> dict[str, Any]:
    batch_size = int(config["training"]["batch_size_pairs"])
    role_indices = _evaluation_role_indices(prepared, smoke=smoke, config=config)
    replay_inputs_by_role = {
        "known_calibration": np.asarray(
            prepared.inputs["known_calibration"], dtype=np.float32
        ),
        "surrogate_unknown": np.asarray(
            prepared.inputs["surrogate_unknown"][role_indices["surrogate_unknown"]]
            if smoke
            else prepared.inputs["surrogate_unknown"],
            dtype=np.float32,
        ),
    }
    own_full = {
        role: infer_e2e_model(
            model, replay_inputs_by_role[role], device=device, batch_size=batch_size
        )
        for role in ("known_calibration", "surrogate_unknown")
    }
    role_pair_rows = {
        role: [
            _role_manifest_rows(prepared, role)[int(index)]
            for index in role_indices[role]
        ]
        for role in ("known_calibration", "surrogate_unknown")
    }
    own_role_logits = {
        "known_calibration": np.asarray(
            own_full["known_calibration"]["fused_logits"]
        )[role_indices["known_calibration"]],
        "surrogate_unknown": np.asarray(
            own_full["surrogate_unknown"]["fused_logits"]
        ),
    }
    q0_role_logits = {
        role: np.asarray(frozen_r2_arrays[role]["global_logits"])[role_indices[role]]
        for role in role_indices
    }
    own_mls = _class_conditional_mls_for_roles(
        full_calibration_logits=own_full["known_calibration"]["fused_logits"],
        full_calibration_labels=prepared.labels["known_calibration"],
        full_calibration_pair_ids=prepared.pair_ids["known_calibration"],
        role_logits=own_role_logits,
        role_pair_rows=role_pair_rows,
    )
    q0_mls = _class_conditional_mls_for_roles(
        full_calibration_logits=frozen_r2_arrays["known_calibration"]["global_logits"],
        full_calibration_labels=prepared.labels["known_calibration"],
        full_calibration_pair_ids=prepared.pair_ids["known_calibration"],
        role_logits=q0_role_logits,
        role_pair_rows=role_pair_rows,
    )

    reference_arrays: dict[str, np.ndarray] = {
        "own_full_calibration_logits": np.asarray(own_full["known_calibration"]["fused_logits"]),
        "q0_full_calibration_logits": np.asarray(frozen_r2_arrays["known_calibration"]["global_logits"]),
        "full_calibration_labels": np.asarray(prepared.labels["known_calibration"], dtype=np.int64),
        "full_calibration_pair_ids": np.asarray(prepared.pair_ids["known_calibration"], dtype=np.str_),
    }
    guided_by_role: dict[str, np.ndarray] | None = None
    unique_output: dict[str, np.ndarray] | None = None
    reference_values: list[np.ndarray] = []
    reference_ids: list[tuple[str, ...]] = []
    reference_metadata: dict[str, Any] | None = None
    reference_base_rows: list[dict[str, Any]] = []
    reference_base_inputs: np.ndarray | None = None
    if model.variant in CSSR_METHODS:
        selected_surrogate_ids = {
            str(row[f"view{view}_sample_id"])
            for row in role_pair_rows["surrogate_unknown"]
            for view in (1, 2)
        }
        reference_indices = [
            index
            for index, row in enumerate(unique_rows)
            if str(row["experiment_role"]) == "known_calibration"
            or (
                str(row["experiment_role"]) == "surrogate_unknown"
                and (not smoke or str(row["sample_id"]) in selected_surrogate_ids)
            )
        ]
        reference_base_rows = [dict(unique_rows[index]) for index in reference_indices]
        reference_base_inputs = np.asarray(
            unique_inputs[np.asarray(reference_indices, dtype=np.int64)], dtype=np.float32
        )
        duplicate_inputs = np.stack(
            [reference_base_inputs, reference_base_inputs], axis=1
        )
        unique_output = infer_e2e_model(
            model, duplicate_inputs, device=device, batch_size=batch_size
        )
        unique_r_all = np.asarray(unique_output["r"], dtype=np.float64)
        if not np.array_equal(unique_r_all[:, 0], unique_r_all[:, 1]):
            raise DataValidationError("identical unique-base slot queries produced different r")
        guided_arrays, reference_values, reference_ids, reference_metadata = (
            build_guided_reference_scores(
                reference_base_rows,
                unique_r_all[:, 0],
                epsilon=float(config["calibration"]["score_epsilon"]),
            )
        )
        reference_arrays.update(guided_arrays)
        guided_by_role = {
            role: _pair_guided_anomaly(
                role_pair_rows[role], reference_metadata["score_by_sample"]
            )
            for role in role_pair_rows
        }
    own_scores = {
        role: compute_method_scores(
            model.variant,
            own_role_logits[role],
            None if guided_by_role is None else guided_by_role[role],
            own_mls[role],
        )
        for role in role_pair_rows
    }
    q0_scores = {
        role: compute_method_scores(
            Q0_FROZEN_R2_CC_MLS,
            q0_role_logits[role],
            None,
            q0_mls[role],
        )
        for role in role_pair_rows
    }
    acceptance_rate = float(config["calibration"]["threshold_known_acceptance_rate"])
    own_metrics = _evaluate_score_arrays(
        prepared=prepared,
        role_indices=role_indices,
        score_arrays=own_scores,
        acceptance_rate=acceptance_rate,
    )
    q0_metrics = _evaluate_score_arrays(
        prepared=prepared,
        role_indices=role_indices,
        score_arrays=q0_scores,
        acceptance_rate=acceptance_rate,
    )
    own_rows = _build_method_prediction_rows(
        method=model.variant,
        prepared=prepared,
        role_indices=role_indices,
        role_pair_rows=role_pair_rows,
        role_logits=own_role_logits,
        role_scores=own_scores,
        metrics=own_metrics,
        reference_metadata=reference_metadata,
    )
    q0_rows = _build_method_prediction_rows(
        method=Q0_FROZEN_R2_CC_MLS,
        prepared=prepared,
        role_indices=role_indices,
        role_pair_rows=role_pair_rows,
        role_logits=q0_role_logits,
        role_scores=q0_scores,
        metrics=q0_metrics,
        reference_metadata=None,
    )
    _metrics_exact(
        own_metrics,
        recompute_method_metrics_from_prediction_rows(
            own_rows, known_acceptance_rate=acceptance_rate
        ),
        context=model.variant,
    )
    _metrics_exact(
        q0_metrics,
        recompute_method_metrics_from_prediction_rows(
            q0_rows, known_acceptance_rate=acceptance_rate
        ),
        context=Q0_FROZEN_R2_CC_MLS,
    )
    own_identity, own_absorption, own_error = build_identity_and_absorption_rows(
        own_rows,
        method=model.variant,
        pair_id=pair_id,
        train_class_order=prepared.train_class_order,
        acceptance_rate=acceptance_rate,
    )
    q0_identity, q0_absorption, q0_error = build_identity_and_absorption_rows(
        q0_rows,
        method=Q0_FROZEN_R2_CC_MLS,
        pair_id=pair_id,
        train_class_order=prepared.train_class_order,
        acceptance_rate=acceptance_rate,
    )
    return {
        "role_indices": role_indices,
        "reference_arrays": reference_arrays,
        "reference_values": reference_values,
        "reference_ids": reference_ids,
        "reference_metadata": reference_metadata,
        "own_prediction_rows": own_rows,
        "q0_prediction_rows": q0_rows,
        "own_metrics": own_metrics,
        "q0_metrics": q0_metrics,
        "identity_metric_rows": [*own_identity, *q0_identity],
        "absorption_rows": [*own_absorption, *q0_absorption],
        "error_analysis": {"method": own_error, "q0": q0_error},
        "own_full_outputs": own_full,
        "unique_full_output": unique_output,
        "replay_inputs_by_role": replay_inputs_by_role,
        "reference_base_rows": reference_base_rows,
        "reference_base_inputs": reference_base_inputs,
    }


def _csv_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text_value = str(value).strip().lower()
    if text_value == "true":
        return True
    if text_value == "false":
        return False
    raise DataValidationError(f"invalid CSV boolean: {value!r}")


def _unit_destination(root: Path, pair_id: str, method: str) -> Path:
    return root / pair_id / "fold_0" / f"seed_{FINETUNE_SEED}" / method


def _normalization_record(prepared: Any) -> dict[str, Any]:
    return {
        "method": "reuse_exact_r2_global_scalar_zscore",
        "mean": float(prepared.normalization.mean),
        "std": float(prepared.normalization.std),
        "epsilon": float(prepared.normalization.epsilon),
        "unique_base_sample_count": int(
            prepared.normalization.unique_base_sample_count
        ),
    }


def _reference_metadata_for_json(
    metadata: Mapping[str, Any] | None,
    reference_ids: Sequence[Sequence[str]],
) -> dict[str, Any]:
    if metadata is None:
        return {
            "status": "not_applicable_q1_mls_only",
            "shared_reference_across_slots": True,
            "surrogate_unknown_in_reference": False,
            "reference_sample_ids": [],
        }
    result = {
        key: value
        for key, value in metadata.items()
        if key not in {"score_by_sample", "r_by_sample", "p_by_sample"}
    }
    result["reference_sample_ids"] = [list(values) for values in reference_ids]
    return result


def _save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    _atomic_write_bytes(path, buffer.getvalue())


def save_unit_result(
    destination: Path,
    *,
    phase: str,
    pair_id: str,
    method: str,
    smoke: bool,
    config: Mapping[str, Any],
    prepared: Any,
    unique_rows: Sequence[Mapping[str, Any]],
    training_result: Mapping[str, Any],
    evaluation_result: Mapping[str, Any],
    r2_audit: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
    environment: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    confirmation_authorization: Mapping[str, Any] | None,
    wall_time_seconds: float,
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=False)
    model: FGMVCSSRE2EModel = training_result["model"]
    _atomic_write_bytes(destination / "source_pair_manifest.csv", prepared.pair_manifest_bytes)
    _write_csv(destination / "unique_base_sample_manifest.csv", unique_rows)
    evaluation_manifest: list[dict[str, Any]] = []
    for role in ("known_calibration", "surrogate_unknown"):
        full_rows = _role_manifest_rows(prepared, role)
        for local_index, source_index in enumerate(evaluation_result["role_indices"][role]):
            evaluation_manifest.append(
                {
                    **full_rows[int(source_index)],
                    "evaluation_subset_role": role,
                    "evaluation_subset_index": local_index,
                }
            )
    _write_csv(destination / "evaluation_pair_manifest.csv", evaluation_manifest)

    for epoch_rows, epoch_audit in zip(
        training_result["schedules"], training_result["schedule_audits"], strict=True
    ):
        epoch = int(epoch_audit["epoch"])
        _write_csv(destination / "pair_schedules" / f"epoch_{epoch:03d}.csv", epoch_rows)
        _write_json(
            destination / "pair_schedule_audits" / f"epoch_{epoch:03d}.json",
            epoch_audit,
        )

    _atomic_write_bytes(
        destination / "training_log.jsonl",
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in training_result["training_log"]
        ).encode("utf-8"),
    )
    _write_json(destination / "training_audit.json", training_result["audit"])
    confidence_rows = [
        {"pair_id": pair_id, "method": method, **row}
        for row in training_result["confidence_log"]
    ]
    _write_csv(destination / "confidence_diagnostics.csv", confidence_rows)
    _write_json(destination / "r2_reference_audit.json", dict(r2_audit))
    _write_json(destination / "normalization.json", _normalization_record(prepared))

    reference_arrays = dict(evaluation_result["reference_arrays"])
    for index, values in enumerate(evaluation_result["reference_values"]):
        reference_arrays[f"class_{index}_reference_r"] = np.asarray(values)
    _save_npz(destination / "reference_scores.npz", reference_arrays)
    if evaluation_result["reference_base_rows"]:
        _write_csv(
            destination / "reference_base_sample_manifest.csv",
            evaluation_result["reference_base_rows"],
        )
    _write_json(
        destination / "reference_distribution.json",
        _reference_metadata_for_json(
            evaluation_result["reference_metadata"],
            evaluation_result["reference_ids"],
        ),
    )
    _write_csv(destination / "predictions_and_scores.csv", evaluation_result["own_prediction_rows"])
    _write_csv(destination / "q0_predictions_and_scores.csv", evaluation_result["q0_prediction_rows"])
    _write_json(destination / "metrics.json", evaluation_result["own_metrics"])
    _write_json(destination / "q0_metrics.json", evaluation_result["q0_metrics"])
    _write_csv(destination / "identity_metrics.csv", evaluation_result["identity_metric_rows"])
    _write_csv(destination / "absorption_by_known_class.csv", evaluation_result["absorption_rows"])
    _write_json(destination / "error_analysis.json", evaluation_result["error_analysis"])

    checkpoint = {
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "mode": "smoke" if smoke else "full",
        "pair_id": pair_id,
        "method": method,
        "architecture": model.architecture_id,
        "model_state_dict": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        "r2_model_state_dict": {
            name: value.detach().cpu().clone()
            for name, value in model.r2_model.state_dict().items()
        },
        "checkpoint_epoch": int(training_result["audit"]["epochs"]),
        "formal_checkpoint": not smoke,
        "checkpoint_selection": "fixed_final_epoch",
        "finetune_seed": FINETUNE_SEED,
        "train_class_order": tuple(prepared.train_class_order),
        "source_pair_manifest_sha256": prepared.pair_manifest_sha256,
        "unique_base_manifest_sha256": hashlib.sha256(_render_csv(unique_rows)).hexdigest(),
        "schedule_sha256": training_result["audit"]["schedule_sha256"],
        "config_sha256": config["_config_sha256"],
        "r2_checkpoint_sha256": r2_audit["checkpoint_sha256"],
        "final_unknown_used": False,
        "even_angle_test_used": False,
    }
    checkpoint_buffer = io.BytesIO()
    torch.save(checkpoint, checkpoint_buffer)
    _atomic_write_bytes(destination / "checkpoint.pt", checkpoint_buffer.getvalue())

    replay_arrays: dict[str, np.ndarray] = {}
    for role in ("known_calibration", "surrogate_unknown"):
        replay_arrays[f"{role}_inputs"] = np.asarray(
            evaluation_result["replay_inputs_by_role"][role], dtype=np.float32
        )
        replay_arrays.update(
            {
                f"{role}_expected_{key}": value
                for key, value in evaluation_result["own_full_outputs"][role].items()
            }
        )
    if evaluation_result["unique_full_output"] is not None:
        reference_base_inputs = np.asarray(
            evaluation_result["reference_base_inputs"], dtype=np.float32
        )
        replay_arrays["reference_base_inputs"] = np.stack(
            [reference_base_inputs, reference_base_inputs], axis=1
        )
        replay_arrays.update(
            {
                f"reference_base_expected_{key}": value
                for key, value in evaluation_result["unique_full_output"].items()
            }
        )
    _save_npz(destination / "checkpoint_replay.npz", replay_arrays)

    resolved = dict(config)
    resolved["_resolved"] = {
        "phase": phase,
        "mode": "smoke" if smoke else "full",
        "pair_id": pair_id,
        "method": method,
        "angle_fold": ANGLE_FOLD,
        "model_seed": MODEL_SEED,
        "finetune_seed": FINETUNE_SEED,
        "source_pair_manifest_sha256": prepared.pair_manifest_sha256,
        "unique_base_manifest_sha256": checkpoint["unique_base_manifest_sha256"],
        "r2_checkpoint_sha256": r2_audit["checkpoint_sha256"],
        "confirmation_authorization": confirmation_authorization,
        "test_features_materialized": False,
    }
    _atomic_write_bytes(
        destination / "resolved_config.yaml",
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False).encode("utf-8"),
    )
    unit_contract = {
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "mode": "smoke" if smoke else "full",
        "pair_id": pair_id,
        "method": method,
        "angle_fold": ANGLE_FOLD,
        "model_seed": MODEL_SEED,
        "finetune_seed": FINETUNE_SEED,
        "config_sha256": config["_config_sha256"],
        "source_hashes": dict(source_hashes),
        "runtime_contract": dict(runtime_contract),
        "confirmation_authorization": confirmation_authorization,
        "known_prediction_source": "own_fused_ce_argmax",
        "main_score": config["scores"]["main_by_method"][method],
        "threshold_source": "own_known_calibration_only",
        "dynamic_train_pairs_per_epoch": int(training_result["audit"]["train_pairs_per_epoch"]),
        "surrogate_unknown_used_for_training": False,
        "surrogate_unknown_used_for_reference_distribution": False,
        "surrogate_unknown_used_for_threshold": False,
        "known_calibration_used_for_training": False,
        "final_unknown_used": False,
        "even_angle_test_used": False,
        "test_pairs_generated": False,
        "test_features_materialized": False,
        "arpl_used": False,
        "pseudo_unknown_used": False,
    }
    _write_json(destination / "unit_contract.json", unit_contract)
    _write_json(destination / "environment.json", dict(environment))
    summary = {
        "status": "complete",
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "mode": "smoke" if smoke else "full",
        "pair_id": pair_id,
        "method": method,
        "metrics": evaluation_result["own_metrics"],
        "q0_metrics": evaluation_result["q0_metrics"],
        "identity_metrics": evaluation_result["identity_metric_rows"],
        "schedule_sha256": training_result["audit"]["schedule_sha256"],
        "epoch0_common_r2_state_sha256": training_result["audit"]["epoch0_common_r2_state_sha256"],
        "ae_initial_state_sha256": training_result["audit"]["ae_initial_state_sha256"],
        "wall_time_seconds": float(wall_time_seconds),
        "diagnostic_only": smoke,
        "final_unknown_used": False,
        "even_angle_test_used": False,
    }
    _write_json(destination / "unit_summary.json", summary)
    _write_json(destination / "artifact_hashes.json", _artifact_hashes(destination))
    _write_json(
        destination / "_SUCCESS.json",
        {
            "status": "complete",
            "unit_summary_sha256": file_sha256(destination / "unit_summary.json"),
            "artifact_hashes_sha256": file_sha256(destination / "artifact_hashes.json"),
        },
    )
    return summary


def _read_authorized_pilot(
    pilot_root: str | Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    root = Path(pilot_root).resolve()
    success_path = root / "_PHASE_SUCCESS.json"
    if not success_path.is_file():
        raise DataValidationError("pilot phase lacks its post-audit success seal")
    success = _read_json(success_path)
    recorded_hashes = _read_json(root / "artifact_hashes.json")
    if (
        success.get("status") != "complete"
        or success.get("phase_summary_sha256")
        != file_sha256(root / "phase_summary.json")
        or success.get("artifact_hashes_sha256")
        != file_sha256(root / "artifact_hashes.json")
    ):
        raise DataValidationError("pilot post-audit success seal is invalid")
    for relative in (
        "pilot_gate.json",
        "phase_summary.json",
        "phase_integrity_audit.json",
        "task_plan.json",
    ):
        if recorded_hashes.get(relative) != file_sha256(root / relative):
            raise DataValidationError(f"pilot authorization artifact changed: {relative}")
    summary = _read_json(root / "phase_summary.json")
    integrity = _read_json(root / "phase_integrity_audit.json")
    gate_path = root / "pilot_gate.json"
    gate = _read_json(gate_path)
    selected = gate.get("selected_method")
    if (
        summary.get("phase") != "pilot"
        or summary.get("config_sha256") != config["_config_sha256"]
        or summary.get("gate") != gate
        or summary.get("decision") != gate.get("signal")
        or summary.get("final_unknown_used") is not False
        or integrity.get("status") != "passed"
        or integrity.get("final_unknown_used") is not False
        or gate.get("confirmation_allowed") is not True
        or selected not in CSSR_METHODS
    ):
        raise DataValidationError("audited pilot gate does not authorize confirmation")
    return {
        "pilot_root": str(root),
        "pilot_gate_sha256": file_sha256(gate_path),
        "selected_method": str(selected),
        "signal": str(gate["signal"]),
    }


def run_unit(
    config_path: str | Path,
    bundle_root: str | Path,
    r2_results_root: str | Path,
    phase_root: str | Path,
    *,
    phase: str,
    pair_id: str,
    method: str,
    device_request: str = "auto",
    pilot_root: str | Path | None = None,
) -> dict[str, Any]:
    config = load_fg_mv_cssr_e2e_config(config_path)
    confirmation_authorization = None
    selected_method = None
    if phase == "confirmation":
        if pilot_root is None:
            raise DataValidationError("confirmation requires an audited pilot root")
        confirmation_authorization = _read_authorized_pilot(pilot_root, config)
        selected_method = str(confirmation_authorization["selected_method"])
    elif pilot_root is not None:
        raise DataValidationError("pilot root only applies to confirmation")
    plan = build_phase_plan(config, phase, selected_method=selected_method)
    if (pair_id, method) not in {
        (str(unit["pair_id"]), str(unit["method"])) for unit in plan
    }:
        raise DataValidationError("unit is outside the frozen phase plan")
    root = Path(phase_root).resolve()
    destination = _unit_destination(root, pair_id, method)
    if destination.exists():
        raise DataValidationError(f"E2E CSSR output already exists: {destination}")
    staging = destination.parent / f".{method}.staging"
    if staging.exists():
        raise DataValidationError(f"stale E2E CSSR staging output exists: {staging}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    project_root = Path(config["_config_path"]).parents[3]
    source_hashes = task_source_hashes(project_root)
    device = _resolve_device(device_request)
    runtime_contract = _configure_runtime(config, device)
    started = time.perf_counter()
    prior_config = _load_prior_config(project_root, config)
    bundle = _load_bundle(bundle_root, config)
    prepared = _prepare_frozen_split(bundle, prior_config, config, pair_id)
    r2_model, frozen_r2_arrays, r2_audit = load_and_audit_frozen_r2(
        project_root=project_root,
        r2_results_root=r2_results_root,
        pair_id=pair_id,
        config=config,
        prepared=prepared,
        prior_config=prior_config,
        device=device,
    )
    unique_rows = build_unique_base_sample_manifest(prepared, bundle)
    unique_inputs = _normalized_unique_base_inputs(
        bundle, prepared, unique_rows
    )
    wrapper = FGMVCSSRE2EModel(
        r2_model,
        method,
        autoencoder_seed=FINETUNE_SEED,
    )
    training_result = train_e2e_method(
        wrapper,
        unique_rows=unique_rows,
        unique_inputs=unique_inputs,
        prepared=prepared,
        pair_id=pair_id,
        config=config,
        device=device,
        smoke=phase == "smoke",
        frozen_r2_arrays=frozen_r2_arrays,
    )
    evaluation_result = evaluate_unit_model(
        model=training_result["model"],
        prepared=prepared,
        unique_rows=unique_rows,
        unique_inputs=unique_inputs,
        frozen_r2_arrays=frozen_r2_arrays,
        pair_id=pair_id,
        config=config,
        device=device,
        smoke=phase == "smoke",
    )
    if task_source_hashes(project_root) != source_hashes:
        raise DataValidationError("task source changed while E2E unit was running")
    environment = _git_environment(project_root, device)
    environment["runtime_contract"] = runtime_contract
    environment["task_source_hashes"] = source_hashes
    summary = save_unit_result(
        staging,
        phase=phase,
        pair_id=pair_id,
        method=method,
        smoke=phase == "smoke",
        config=config,
        prepared=prepared,
        unique_rows=unique_rows,
        training_result=training_result,
        evaluation_result=evaluation_result,
        r2_audit=r2_audit,
        runtime_contract=runtime_contract,
        environment=environment,
        source_hashes=source_hashes,
        confirmation_authorization=confirmation_authorization,
        wall_time_seconds=time.perf_counter() - started,
    )
    staging.replace(destination)
    return {**summary, "destination": str(destination)}


def _audit_saved_prediction_scores(
    rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    reference_arrays: Mapping[str, np.ndarray],
    acceptance_rate: float,
) -> dict[str, float]:
    if not rows or {str(row["method"]) for row in rows} != {method}:
        raise DataValidationError("prediction artifact method population changed")
    for row in rows:
        score = float(row["unknown_score"])
        threshold = float(row["threshold"])
        rejected = _csv_bool(row["rejected"])
        prediction = int(row["predicted_known_label"])
        if rejected != (score > threshold):
            raise DataValidationError("prediction rejection flag changed")
        if int(row["open_set_prediction"]) != (5 if rejected else prediction):
            raise DataValidationError("open-set prediction changed")
        if _csv_bool(row["final_unknown_used"]) or _csv_bool(row["even_angle_test_used"]):
            raise DataValidationError("forbidden final-test evidence entered predictions")

    role_rows = {
        role: [row for row in rows if str(row["evaluation_role"]) == role]
        for role in ("known_calibration", "surrogate_unknown")
    }
    role_logits = {
        role: np.asarray(
            [json.loads(str(row["fused_logits"])) for row in selected],
            dtype=np.float64,
        )
        for role, selected in role_rows.items()
    }
    prefix = "q0" if method == Q0_FROZEN_R2_CC_MLS else "own"
    mls = _class_conditional_mls_for_roles(
        full_calibration_logits=np.asarray(
            reference_arrays[f"{prefix}_full_calibration_logits"], dtype=np.float64
        ),
        full_calibration_labels=np.asarray(
            reference_arrays["full_calibration_labels"], dtype=np.int64
        ),
        full_calibration_pair_ids=tuple(
            reference_arrays["full_calibration_pair_ids"].tolist()
        ),
        role_logits=role_logits,
        role_pair_rows=role_rows,
    )
    for role, selected in role_rows.items():
        for index, row in enumerate(selected):
            if method in CSSR_METHODS:
                diagnostic = float(row["diagnostic_class_conditional_mls"])
                if not math.isclose(diagnostic, float(mls[role][index]), rel_tol=0.0, abs_tol=1.0e-15):
                    raise DataValidationError("diagnostic MLS does not reproduce")
                prediction = int(row["predicted_known_label"])
                view1 = np.asarray(json.loads(str(row["view1_a"])), dtype=np.float64)
                view2 = np.asarray(json.loads(str(row["view2_a"])), dtype=np.float64)
                guided = float((view1[prediction] + view2[prediction]) / 2.0)
                swapped = float((view2[prediction] + view1[prediction]) / 2.0)
                if not math.isclose(guided, float(row["unknown_score"]), rel_tol=0.0, abs_tol=1.0e-15):
                    raise DataValidationError("guided reconstruction score does not reproduce")
                if guided != swapped:
                    raise DataValidationError("guided reconstruction is not view-swap invariant")
            elif not math.isclose(
                float(row["unknown_score"]),
                float(mls[role][index]),
                rel_tol=0.0,
                abs_tol=1.0e-15,
            ):
                raise DataValidationError("main class-conditional MLS does not reproduce")
    return recompute_method_metrics_from_prediction_rows(
        rows, known_acceptance_rate=acceptance_rate
    )


def audit_unit_result(
    unit_root: str | Path,
    *,
    config: Mapping[str, Any],
    phase: str,
    pair_id: str,
    method: str,
) -> dict[str, Any]:
    root = Path(unit_root).resolve()
    success_path = root / "_SUCCESS.json"
    if not success_path.is_file():
        raise DataValidationError(f"E2E unit is not complete: {root}")
    success = _read_json(success_path)
    if (
        success.get("status") != "complete"
        or success.get("unit_summary_sha256") != file_sha256(root / "unit_summary.json")
        or success.get("artifact_hashes_sha256") != file_sha256(root / "artifact_hashes.json")
    ):
        raise DataValidationError("E2E unit success marker is invalid")
    recorded_hashes = _read_json(root / "artifact_hashes.json")
    if recorded_hashes != _artifact_hashes(root):
        raise DataValidationError("E2E unit artifact hash audit failed")

    contract = _read_json(root / "unit_contract.json")
    expected_mode = "smoke" if phase == "smoke" else "full"
    expected_epochs = 1 if phase == "smoke" else int(config["training"]["epochs"])
    expected_contract = {
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "mode": expected_mode,
        "pair_id": pair_id,
        "method": method,
        "angle_fold": ANGLE_FOLD,
        "model_seed": MODEL_SEED,
        "finetune_seed": FINETUNE_SEED,
        "config_sha256": config["_config_sha256"],
        "known_prediction_source": "own_fused_ce_argmax",
        "main_score": config["scores"]["main_by_method"][method],
        "threshold_source": "own_known_calibration_only",
        "dynamic_train_pairs_per_epoch": 720,
        "surrogate_unknown_used_for_training": False,
        "surrogate_unknown_used_for_reference_distribution": False,
        "surrogate_unknown_used_for_threshold": False,
        "known_calibration_used_for_training": False,
        "final_unknown_used": False,
        "even_angle_test_used": False,
        "test_pairs_generated": False,
        "test_features_materialized": False,
        "arpl_used": False,
        "pseudo_unknown_used": False,
    }
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            raise DataValidationError(f"unit contract changed: {key}")
    project_root = Path(config["_config_path"]).parents[3]
    if contract.get("source_hashes") != task_source_hashes(project_root):
        raise DataValidationError("unit source hash binding changed")
    recorded_runtime = _mapping(contract.get("runtime_contract"), "unit runtime contract")
    audit_device = _resolve_device(str(recorded_runtime["device"]))
    audit_runtime = _configure_runtime(config, audit_device)
    if dict(recorded_runtime) != audit_runtime:
        raise DataValidationError("audit runtime differs from the recorded unit runtime")

    unique_rows = _read_csv(root / "unique_base_sample_manifest.csv")
    if len(unique_rows) != 972:
        raise DataValidationError("unique-base manifest population changed")
    schedule_hashes: list[str] = []
    for epoch in range(1, expected_epochs + 1):
        schedule_path = root / "pair_schedules" / f"epoch_{epoch:03d}.csv"
        audit_path = root / "pair_schedule_audits" / f"epoch_{epoch:03d}.json"
        expected_schedule, expected_audit = build_epoch_pair_schedule(
            unique_rows,
            pair_id=pair_id,
            angle_fold=ANGLE_FOLD,
            epoch=epoch,
            finetune_seed=FINETUNE_SEED,
        )
        if schedule_path.read_bytes() != _render_csv(expected_schedule):
            raise DataValidationError(f"epoch {epoch} pair schedule does not reproduce")
        if _read_json(audit_path) != expected_audit:
            raise DataValidationError(f"epoch {epoch} schedule audit does not reproduce")
        schedule_hashes.append(str(expected_audit["epoch_manifest_sha256"]))
    training_audit = _read_json(root / "training_audit.json")
    combined_schedule_hash = _sequence_sha256(schedule_hashes)
    if (
        training_audit.get("status") != "passed"
        or int(training_audit.get("epochs", -1)) != expected_epochs
        or int(training_audit.get("train_pairs_per_epoch", -1)) != 720
        or training_audit.get("schedule_sha256") != combined_schedule_hash
        or training_audit.get("frozen_prefix_unchanged") is not True
        or training_audit.get("all_parameters_finite") is not True
        or training_audit.get("known_calibration_used_for_training") is not False
        or training_audit.get("surrogate_unknown_used_for_training") is not False
        or training_audit.get("final_unknown_used") is not False
    ):
        raise DataValidationError("E2E training audit changed")
    training_rows = [
        json.loads(line)
        for line in (root / "training_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_lr_groups = {
        "last_stage": float(config["training"]["lr_last_stage"]),
        "projection_and_ce_head": float(
            config["training"]["lr_projection_and_ce_head"]
        ),
        **(
            {}
            if method == Q1_CE_FINETUNE_CONTROL
            else {"autoencoders": float(config["training"]["lr_autoencoders"])}
        ),
    }
    if len(training_rows) != expected_epochs:
        raise DataValidationError("training log epoch count changed")
    for epoch, row in enumerate(training_rows, start=1):
        factor = _learning_rate_factor(
            epoch,
            warmup_epochs=int(config["training"]["warmup_epochs"]),
            total_epochs=int(config["training"]["epochs"]),
        )
        observed_lrs = {key: float(value) for key, value in row["learning_rates"].items()}
        expected_lrs = {
            key: base * factor for key, base in expected_lr_groups.items()
        }
        if (
            int(row["epoch"]) != epoch
            or str(row["method"]) != method
            or float(row["learning_rate_factor"]) != factor
            or observed_lrs != expected_lrs
            or any(value <= 0.0 for value in observed_lrs.values())
            or str(row["pair_schedule_sha256"]) != schedule_hashes[epoch - 1]
            or int(row["pair_count"]) != 720
            or row["known_calibration_used_for_training"] is not False
            or row["surrogate_unknown_used_for_training"] is not False
            or row["final_unknown_used"] is not False
        ):
            raise DataValidationError(f"training log contract changed at epoch {epoch}")
        numeric_fields = (
            "train_total_loss",
            "train_classification_loss",
            "train_relative_loss",
            "train_absolute_loss",
            "train_separation_loss",
            "train_accuracy",
        )
        if any(not math.isfinite(float(row[key])) for key in numeric_fields):
            raise DataValidationError("training log contains NaN or Inf")
    confidence_rows = _read_csv(root / "confidence_diagnostics.csv")
    if (
        len(confidence_rows) != expected_epochs + 1
        or [int(row["epoch"]) for row in confidence_rows]
        != list(range(expected_epochs + 1))
        or {str(row["pair_id"]) for row in confidence_rows} != {pair_id}
        or {str(row["method"]) for row in confidence_rows} != {method}
        or any(
            not math.isfinite(float(row[key]))
            for row in confidence_rows
            for key in (
                "accuracy",
                "macro_f1",
                "nll",
                "brier",
                "ece",
                "mean_max_logit",
                "mean_top1_top2_logit_margin",
                "mean_single_view_feature_norm",
                "mean_fused_feature_norm",
                "ce_head_weight_norm",
                "mean_fused_feature_l2_drift_from_epoch0",
                "mean_kl_epoch0_to_current",
            )
        )
        or any(_csv_bool(row["surrogate_unknown_used"]) for row in confidence_rows)
    ):
        raise DataValidationError("confidence diagnostic artifact changed")

    checkpoint = torch.load(root / "checkpoint.pt", map_location="cpu", weights_only=False)
    expected_checkpoint = {
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "mode": expected_mode,
        "pair_id": pair_id,
        "method": method,
        "architecture": EXPERIMENT_ID,
        "checkpoint_epoch": expected_epochs,
        "formal_checkpoint": phase != "smoke",
        "checkpoint_selection": "fixed_final_epoch",
        "finetune_seed": FINETUNE_SEED,
        "config_sha256": config["_config_sha256"],
        "schedule_sha256": combined_schedule_hash,
        "final_unknown_used": False,
        "even_angle_test_used": False,
    }
    for key, expected in expected_checkpoint.items():
        if checkpoint.get(key) != expected:
            raise DataValidationError(f"checkpoint metadata changed: {key}")
    if file_sha256(root / "source_pair_manifest.csv") != str(
        checkpoint["source_pair_manifest_sha256"]
    ):
        raise DataValidationError("source pair manifest binding changed")
    if hashlib.sha256((root / "unique_base_sample_manifest.csv").read_bytes()).hexdigest() != str(
        checkpoint["unique_base_manifest_sha256"]
    ):
        raise DataValidationError("unique-base manifest binding changed")
    evaluation_manifest = _read_csv(root / "evaluation_pair_manifest.csv")
    if not evaluation_manifest or any(
        int(row["view1_angle_deg"]) % 2 == 0
        or int(row["view2_angle_deg"]) % 2 == 0
        or str(row["experiment_role"])
        not in {"known_calibration", "surrogate_unknown"}
        for row in evaluation_manifest
    ):
        raise DataValidationError("evaluation manifest contains forbidden evidence")
    restored = FGMVCSSRE2EModel.from_r2_state_dict(
        checkpoint["r2_model_state_dict"],
        method,
        known_class_count=5,
        autoencoder_seed=FINETUNE_SEED,
    )
    incompatible = restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise DataValidationError("E2E checkpoint did not strict-load")
    if not all(bool(torch.isfinite(parameter).all()) for parameter in restored.parameters()):
        raise DataValidationError("E2E checkpoint contains NaN or Inf")
    restored.to(audit_device).eval()
    with np.load(root / "checkpoint_replay.npz", allow_pickle=False) as saved:
        replay = {name: saved[name] for name in saved.files}
    replay_roles = ["known_calibration", "surrogate_unknown"]
    if method in CSSR_METHODS:
        replay_roles.append("reference_base")
    for role in replay_roles:
        replay_output = infer_e2e_model(
            restored,
            np.asarray(replay[f"{role}_inputs"], dtype=np.float32),
            device=audit_device,
            batch_size=int(config["training"]["batch_size_pairs"]),
        )
        for name, observed in replay_output.items():
            expected = np.asarray(replay[f"{role}_expected_{name}"])
            if not np.array_equal(observed, expected):
                raise DataValidationError(
                    f"checkpoint replay changed: {role}.{name}"
                )

    with np.load(root / "reference_scores.npz", allow_pickle=False) as saved:
        reference_arrays = {name: saved[name] for name in saved.files}
    if method in CSSR_METHODS:
        reference_base_rows = _read_csv(root / "reference_base_sample_manifest.csv")
        rebuilt_arrays, rebuilt_values, rebuilt_ids, _ = build_guided_reference_scores(
            reference_base_rows,
            np.asarray(reference_arrays["r"], dtype=np.float64),
            epsilon=float(config["calibration"]["score_epsilon"]),
        )
        for name in ("known_calibration_p", "known_calibration_a", "surrogate_unknown_p", "surrogate_unknown_a"):
            if not np.array_equal(rebuilt_arrays[name], reference_arrays[name]):
                raise DataValidationError(f"shared CSSR reference does not reproduce: {name}")
        for index, values in enumerate(rebuilt_values):
            if not np.array_equal(values, reference_arrays[f"class_{index}_reference_r"]):
                raise DataValidationError("class reference distribution changed")
        metadata = _read_json(root / "reference_distribution.json")
        if metadata.get("reference_sample_ids") != [list(values) for values in rebuilt_ids]:
            raise DataValidationError("reference sample IDs changed")

    acceptance_rate = float(config["calibration"]["threshold_known_acceptance_rate"])
    own_rows = _read_csv(root / "predictions_and_scores.csv")
    q0_rows = _read_csv(root / "q0_predictions_and_scores.csv")
    own_metrics = _read_json(root / "metrics.json")
    q0_metrics = _read_json(root / "q0_metrics.json")
    _metrics_exact(
        own_metrics,
        _audit_saved_prediction_scores(
            own_rows,
            method=method,
            reference_arrays=reference_arrays,
            acceptance_rate=acceptance_rate,
        ),
        context=method,
    )
    _metrics_exact(
        q0_metrics,
        _audit_saved_prediction_scores(
            q0_rows,
            method=Q0_FROZEN_R2_CC_MLS,
            reference_arrays=reference_arrays,
            acceptance_rate=acceptance_rate,
        ),
        context=Q0_FROZEN_R2_CC_MLS,
    )
    identity_rows = _read_csv(root / "identity_metrics.csv")
    own_identity, own_absorption, own_error = build_identity_and_absorption_rows(
        own_rows,
        method=method,
        pair_id=pair_id,
        train_class_order=tuple(checkpoint["train_class_order"]),
        acceptance_rate=acceptance_rate,
    )
    q0_identity, q0_absorption, q0_error = build_identity_and_absorption_rows(
        q0_rows,
        method=Q0_FROZEN_R2_CC_MLS,
        pair_id=pair_id,
        train_class_order=tuple(checkpoint["train_class_order"]),
        acceptance_rate=acceptance_rate,
    )
    expected_identity = [*own_identity, *q0_identity]
    expected_absorption = [*own_absorption, *q0_absorption]
    if (root / "identity_metrics.csv").read_bytes() != _render_csv(expected_identity):
        raise DataValidationError("identity metrics do not reproduce")
    if (root / "absorption_by_known_class.csv").read_bytes() != _render_csv(expected_absorption):
        raise DataValidationError("absorption analysis does not reproduce")
    if _read_json(root / "error_analysis.json") != {"method": own_error, "q0": q0_error}:
        raise DataValidationError("identity error analysis does not reproduce")
    summary = _read_json(root / "unit_summary.json")
    if (
        summary.get("status") != "complete"
        or summary.get("pair_id") != pair_id
        or summary.get("method") != method
        or summary.get("metrics") != own_metrics
        or summary.get("q0_metrics") != q0_metrics
        or summary.get("schedule_sha256") != combined_schedule_hash
    ):
        raise DataValidationError("unit summary changed")
    return {
        "status": "passed",
        "phase": phase,
        "pair_id": pair_id,
        "method": method,
        "destination": str(root),
        "artifact_count": len(recorded_hashes),
        "metrics": own_metrics,
        "q0_metrics": q0_metrics,
        "identity_metrics": identity_rows,
        "schedule_sha256": combined_schedule_hash,
        "epoch0_common_r2_state_sha256": training_audit["epoch0_common_r2_state_sha256"],
        "ae_initial_state_sha256": training_audit["ae_initial_state_sha256"],
        "checkpoint_strict_load": True,
        "checkpoint_replay": "bitwise_exact",
        "checkpoint_replay_maximum_absolute_error": 0.0,
        "audit_runtime": audit_runtime,
        "metric_recomputation": "exact",
        "shared_reference_recomputation": "exact" if method in CSSR_METHODS else "not_applicable",
        "final_unknown_used": False,
        "even_angle_test_used": False,
    }


def _phase_authorization_and_plan(
    *,
    config: Mapping[str, Any],
    phase: str,
    pilot_root: str | Path | None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if phase == "confirmation":
        if pilot_root is None:
            raise DataValidationError("confirmation phase requires an audited pilot root")
        authorization = _read_authorized_pilot(pilot_root, config)
        plan = build_phase_plan(
            config,
            phase,
            selected_method=str(authorization["selected_method"]),
        )
        return authorization, plan
    if pilot_root is not None:
        raise DataValidationError("pilot root only applies to confirmation")
    return None, build_phase_plan(config, phase)


def _phase_rows_from_audits(
    audits: Sequence[Mapping[str, Any]],
    *,
    plan: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[str, dict[str, dict[str, float]]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    audit_map = {
        (str(audit["pair_id"]), str(audit["method"])): audit for audit in audits
    }
    if set(audit_map) != {
        (str(unit["pair_id"]), str(unit["method"])) for unit in plan
    }:
        raise DataValidationError("phase unit audit population changed")
    pair_order = tuple(dict.fromkeys(str(unit["pair_id"]) for unit in plan))
    methods_by_pair = {
        pair_id: tuple(
            str(unit["method"])
            for unit in plan
            if str(unit["pair_id"]) == pair_id
        )
        for pair_id in pair_order
    }
    metrics_by_pair: dict[str, dict[str, dict[str, float]]] = {}
    metric_rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    integrity_pairs: dict[str, Any] = {}
    for pair_id in pair_order:
        method_audits = [audit_map[(pair_id, method)] for method in methods_by_pair[pair_id]]
        q0_metrics = dict(method_audits[0]["q0_metrics"])
        q0_identity = [
            dict(row)
            for row in method_audits[0]["identity_metrics"]
            if str(row["method"]) == Q0_FROZEN_R2_CC_MLS
        ]
        for audit in method_audits[1:]:
            if dict(audit["q0_metrics"]) != q0_metrics:
                raise DataValidationError(f"Q0 metrics differ across {pair_id} units")
            current_q0_identity = [
                dict(row)
                for row in audit["identity_metrics"]
                if str(row["method"]) == Q0_FROZEN_R2_CC_MLS
            ]
            if current_q0_identity != q0_identity:
                raise DataValidationError(f"Q0 identity metrics differ across {pair_id} units")
        pair_metrics: dict[str, dict[str, float]] = {
            Q0_FROZEN_R2_CC_MLS: q0_metrics
        }
        for audit in method_audits:
            pair_metrics[str(audit["method"])] = dict(audit["metrics"])
        metrics_by_pair[pair_id] = pair_metrics
        ordered_methods = (Q0_FROZEN_R2_CC_MLS, *methods_by_pair[pair_id])
        for method in ordered_methods:
            metric_rows.append(
                {
                    "pair_id": pair_id,
                    "method": method,
                    **{key: float(pair_metrics[method][key]) for key in REPORT_METRIC_KEYS},
                    "threshold": float(pair_metrics[method]["threshold"]),
                }
            )
        identity_rows.extend(q0_identity)
        for audit in method_audits:
            identity_rows.extend(
                dict(row)
                for row in audit["identity_metrics"]
                if str(row["method"]) == str(audit["method"])
            )
        schedule_hashes = {str(audit["schedule_sha256"]) for audit in method_audits}
        common_hashes = {
            str(audit["epoch0_common_r2_state_sha256"]) for audit in method_audits
        }
        if len(schedule_hashes) != 1:
            raise DataValidationError(f"Q1-Q4 schedule differs within {pair_id}")
        if len(common_hashes) != 1:
            raise DataValidationError(f"Q1-Q4 epoch-0 R2 state differs within {pair_id}")
        ae_hashes = {
            str(audit["method"]): audit["ae_initial_state_sha256"]
            for audit in method_audits
        }
        if (
            Q2_E2E_REL_CSSR_1X1 in ae_hashes
            and Q3_E2E_ABSREL_CSSR_1X1 in ae_hashes
            and ae_hashes[Q2_E2E_REL_CSSR_1X1]
            != ae_hashes[Q3_E2E_ABSREL_CSSR_1X1]
        ):
            raise DataValidationError(f"Q2/Q3 AE initialization differs within {pair_id}")
        integrity_pairs[pair_id] = {
            "schedule_sha256": next(iter(schedule_hashes)),
            "epoch0_common_r2_state_sha256": next(iter(common_hashes)),
            "ae_initial_state_sha256_by_method": ae_hashes,
            "q0_reused_without_training": True,
        }
    integrity = {
        "status": "passed",
        "pairs": integrity_pairs,
        "shared_schedule_across_methods": True,
        "epoch0_common_state_across_methods": True,
        "q2_q3_same_ae_initialization": all(
            values["ae_initial_state_sha256_by_method"].get(Q2_E2E_REL_CSSR_1X1)
            == values["ae_initial_state_sha256_by_method"].get(Q3_E2E_ABSREL_CSSR_1X1)
            for values in integrity_pairs.values()
            if Q2_E2E_REL_CSSR_1X1
            in values["ae_initial_state_sha256_by_method"]
        ),
        "final_unknown_used": False,
        "even_angle_test_used": False,
    }
    return metrics_by_pair, metric_rows, identity_rows, integrity


def _phase_absorption_rows(
    phase_root: Path,
    plan: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen_q0: set[str] = set()
    for unit in plan:
        pair_id = str(unit["pair_id"])
        method = str(unit["method"])
        rows = _read_csv(_unit_destination(phase_root, pair_id, method) / "absorption_by_known_class.csv")
        result.extend(row for row in rows if str(row["method"]) == method)
        if pair_id not in seen_q0:
            result.extend(
                row for row in rows if str(row["method"]) == Q0_FROZEN_R2_CC_MLS
            )
            seen_q0.add(pair_id)
    return result


def _phase_confidence_rows(
    phase_root: Path,
    plan: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        row
        for unit in plan
        for row in _read_csv(
            _unit_destination(
                phase_root, str(unit["pair_id"]), str(unit["method"])
            )
            / "confidence_diagnostics.csv"
        )
    ]


def aggregate_phase_root(
    phase_root: str | Path,
    *,
    config: Mapping[str, Any],
    phase: str,
    pilot_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(phase_root).resolve()
    if (root / "_PHASE_SUCCESS.json").exists():
        raise DataValidationError(f"phase output is already aggregated: {root}")
    authorization, plan = _phase_authorization_and_plan(
        config=config, phase=phase, pilot_root=pilot_root
    )
    audits = [
        audit_unit_result(
            _unit_destination(root, str(unit["pair_id"]), str(unit["method"])),
            config=config,
            phase=phase,
            pair_id=str(unit["pair_id"]),
            method=str(unit["method"]),
        )
        for unit in plan
    ]
    metrics_by_pair, metric_rows, identity_rows, integrity = _phase_rows_from_audits(
        audits, plan=plan
    )
    absorption_rows = _phase_absorption_rows(root, plan)
    confidence_rows = _phase_confidence_rows(root, plan)
    _write_json(root / "task_plan.json", {"phase": phase, "units": plan})
    _write_json(
        root / "launcher_status.json",
        {
            "status": "all_units_complete_and_audited",
            "phase": phase,
            "unit_count": len(audits),
            "units": [audit["destination"] for audit in audits],
        },
    )
    _write_csv(root / "metrics_by_pair.csv", metric_rows)
    _write_json(root / "metrics_by_pair.json", metrics_by_pair)
    _write_csv(root / "identity_metrics.csv", identity_rows)
    _write_csv(root / "absorption_by_known_class.csv", absorption_rows)
    _write_csv(root / "confidence_diagnostics.csv", confidence_rows)
    _write_json(root / "phase_integrity_audit.json", integrity)
    if phase == "pilot":
        gate = evaluate_pilot_gate(metric_rows, identity_rows, config)
        _write_json(root / "pilot_gate.json", gate)
        decision = str(gate["signal"])
    elif phase == "confirmation":
        selected = str(authorization["selected_method"])
        gate = evaluate_confirmation_gate(metric_rows, identity_rows, selected, config)
        _write_json(root / "confirmation_gate.json", gate)
        decision = str(gate["decision"])
    else:
        gate = None
        decision = "diagnostic_smoke_only"
    summary = {
        "status": "complete",
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "pair_ids": list(dict.fromkeys(str(unit["pair_id"]) for unit in plan)),
        "unit_count": len(audits),
        "metrics": metrics_by_pair,
        "gate": gate,
        "decision": decision,
        "confirmation_authorization": authorization,
        "config_sha256": config["_config_sha256"],
        "diagnostic_only": phase == "smoke",
        "final_unknown_used": False,
        "even_angle_test_used": False,
        "automatic_followon_authorized": False,
    }
    _write_json(root / "phase_summary.json", summary)
    _write_json(root / "artifact_hashes.json", _artifact_hashes(root))
    _write_json(
        root / "_PHASE_SUCCESS.json",
        {
            "status": "complete",
            "phase_summary_sha256": file_sha256(root / "phase_summary.json"),
            "artifact_hashes_sha256": file_sha256(root / "artifact_hashes.json"),
        },
    )
    return summary


def audit_phase_root(
    phase_root: str | Path,
    *,
    config: Mapping[str, Any],
    phase: str,
    pilot_root: str | Path | None,
) -> dict[str, Any]:
    root = Path(phase_root).resolve()
    success_path = root / "_PHASE_SUCCESS.json"
    if not success_path.is_file():
        raise DataValidationError(f"phase is not complete: {root}")
    success = _read_json(success_path)
    if (
        success.get("status") != "complete"
        or success.get("phase_summary_sha256") != file_sha256(root / "phase_summary.json")
        or success.get("artifact_hashes_sha256") != file_sha256(root / "artifact_hashes.json")
    ):
        raise DataValidationError("phase success marker is invalid")
    recorded_hashes = _read_json(root / "artifact_hashes.json")
    if recorded_hashes != _artifact_hashes(root):
        raise DataValidationError("phase artifact hash audit failed")
    stored_summary = _read_json(root / "phase_summary.json")
    if phase == "confirmation" and pilot_root is None:
        stored_authorization = stored_summary.get("confirmation_authorization")
        if not isinstance(stored_authorization, Mapping):
            raise DataValidationError("confirmation summary lacks pilot authorization")
        pilot_root = str(stored_authorization["pilot_root"])
    authorization, plan = _phase_authorization_and_plan(
        config=config, phase=phase, pilot_root=pilot_root
    )
    audits = [
        audit_unit_result(
            _unit_destination(root, str(unit["pair_id"]), str(unit["method"])),
            config=config,
            phase=phase,
            pair_id=str(unit["pair_id"]),
            method=str(unit["method"]),
        )
        for unit in plan
    ]
    metrics_by_pair, metric_rows, identity_rows, integrity = _phase_rows_from_audits(
        audits, plan=plan
    )
    if _read_json(root / "metrics_by_pair.json") != metrics_by_pair:
        raise DataValidationError("phase metric JSON does not reproduce")
    if (root / "metrics_by_pair.csv").read_bytes() != _render_csv(metric_rows):
        raise DataValidationError("phase metric table does not reproduce")
    if (root / "identity_metrics.csv").read_bytes() != _render_csv(identity_rows):
        raise DataValidationError("phase identity table does not reproduce")
    if _read_json(root / "phase_integrity_audit.json") != integrity:
        raise DataValidationError("phase integrity audit does not reproduce")
    absorption_rows = _phase_absorption_rows(root, plan)
    if (root / "absorption_by_known_class.csv").read_bytes() != _render_csv(absorption_rows):
        raise DataValidationError("phase absorption table does not reproduce")
    confidence_rows = _phase_confidence_rows(root, plan)
    if (root / "confidence_diagnostics.csv").read_bytes() != _render_csv(confidence_rows):
        raise DataValidationError("phase confidence diagnostic table does not reproduce")
    if phase == "pilot":
        gate = evaluate_pilot_gate(metric_rows, identity_rows, config)
        if _read_json(root / "pilot_gate.json") != gate:
            raise DataValidationError("pilot gate does not reproduce")
        decision = str(gate["signal"])
    elif phase == "confirmation":
        gate = evaluate_confirmation_gate(
            metric_rows,
            identity_rows,
            str(authorization["selected_method"]),
            config,
        )
        if _read_json(root / "confirmation_gate.json") != gate:
            raise DataValidationError("confirmation gate does not reproduce")
        decision = str(gate["decision"])
    else:
        gate = None
        decision = "diagnostic_smoke_only"
    if (
        stored_summary.get("status") != "complete"
        or stored_summary.get("experiment_id") != EXPERIMENT_ID
        or stored_summary.get("phase") != phase
        or stored_summary.get("metrics") != metrics_by_pair
        or stored_summary.get("gate") != gate
        or stored_summary.get("decision") != decision
        or stored_summary.get("final_unknown_used") is not False
        or stored_summary.get("even_angle_test_used") is not False
    ):
        raise DataValidationError("phase summary does not reproduce")
    return {
        "status": "passed",
        "phase": phase,
        "root": str(root),
        "unit_count": len(audits),
        "artifact_count": len(recorded_hashes),
        "metric_recomputation": "exact",
        "gate_recomputation": "exact" if gate is not None else "not_applicable",
        "checkpoint_replay": "passed",
        "schedule_recomputation": "exact",
        "decision": decision,
        "gate": gate,
        "final_unknown_used": False,
        "even_angle_test_used": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preregistered FG-MV-CSSR end-to-end redesign runner"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    load = commands.add_parser("load-config")
    load.add_argument("--config", default=CONFIG_RELATIVE_PATH)

    plan = commands.add_parser("plan")
    plan.add_argument("--config", default=CONFIG_RELATIVE_PATH)
    plan.add_argument("--phase", choices=("smoke", "pilot", "confirmation"), required=True)
    plan.add_argument("--selected-method", choices=CSSR_METHODS)
    plan.add_argument("--pilot-root")

    run = commands.add_parser("run-unit")
    run.add_argument("--config", default=CONFIG_RELATIVE_PATH)
    run.add_argument("--bundle-root", required=True)
    run.add_argument("--r2-results-root", required=True)
    run.add_argument("--phase-root", required=True)
    run.add_argument("--phase", choices=("smoke", "pilot", "confirmation"), required=True)
    run.add_argument("--pair-id", required=True)
    run.add_argument("--method", choices=TRAINABLE_METHODS, required=True)
    run.add_argument("--device", default="auto")
    run.add_argument("--pilot-root")

    audit_unit_parser = commands.add_parser("audit-unit")
    audit_unit_parser.add_argument("--config", default=CONFIG_RELATIVE_PATH)
    audit_unit_parser.add_argument("--unit-root", required=True)
    audit_unit_parser.add_argument("--phase", choices=("smoke", "pilot", "confirmation"), required=True)
    audit_unit_parser.add_argument("--pair-id", required=True)
    audit_unit_parser.add_argument("--method", choices=TRAINABLE_METHODS, required=True)

    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--config", default=CONFIG_RELATIVE_PATH)
    aggregate.add_argument("--phase-root", required=True)
    aggregate.add_argument("--phase", choices=("smoke", "pilot", "confirmation"), required=True)
    aggregate.add_argument("--pilot-root")

    audit_phase_parser = commands.add_parser("audit-phase")
    audit_phase_parser.add_argument("--config", default=CONFIG_RELATIVE_PATH)
    audit_phase_parser.add_argument("--phase-root", required=True)
    audit_phase_parser.add_argument("--phase", choices=("smoke", "pilot", "confirmation"), required=True)
    audit_phase_parser.add_argument("--pilot-root")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = load_fg_mv_cssr_e2e_config(arguments.config)
    if arguments.command == "load-config":
        result = {
            "status": "passed",
            "experiment_id": config["experiment_id"],
            "config_path": config["_config_path"],
            "config_sha256": config["_config_sha256"],
            "final_unknown_test_authorized": False,
        }
    elif arguments.command == "plan":
        selected_method = arguments.selected_method
        authorization = None
        if arguments.pilot_root is not None:
            if arguments.phase != "confirmation" or selected_method is not None:
                raise DataValidationError(
                    "pilot-root plan authorization only applies to confirmation without an override"
                )
            authorization = _read_authorized_pilot(arguments.pilot_root, config)
            selected_method = str(authorization["selected_method"])
        result = {
            "status": "planned",
            "phase": arguments.phase,
            "authorization": authorization,
            "units": build_phase_plan(
                config, arguments.phase, selected_method=selected_method
            ),
            "final_unknown_test_authorized": False,
        }
    elif arguments.command == "run-unit":
        result = run_unit(
            arguments.config,
            arguments.bundle_root,
            arguments.r2_results_root,
            arguments.phase_root,
            phase=arguments.phase,
            pair_id=arguments.pair_id,
            method=arguments.method,
            device_request=arguments.device,
            pilot_root=arguments.pilot_root,
        )
    elif arguments.command == "audit-unit":
        result = audit_unit_result(
            arguments.unit_root,
            config=config,
            phase=arguments.phase,
            pair_id=arguments.pair_id,
            method=arguments.method,
        )
    elif arguments.command == "aggregate":
        result = aggregate_phase_root(
            arguments.phase_root,
            config=config,
            phase=arguments.phase,
            pilot_root=arguments.pilot_root,
        )
    elif arguments.command == "audit-phase":
        result = audit_phase_root(
            arguments.phase_root,
            config=config,
            phase=arguments.phase,
            pilot_root=arguments.pilot_root,
        )
    else:
        raise AssertionError("unreachable command")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
