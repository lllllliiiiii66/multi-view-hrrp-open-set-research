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
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hrrp_osr.data.errors import DataConfigError, DataValidationError
from hrrp_osr.data.manifest import file_sha256
from hrrp_osr.data.processed import ProcessedBundle, load_processed_bundle
from hrrp_osr.evaluation.metrics import evaluate_open_set
from hrrp_osr.evaluation.ms_mean_factorial import REPORT_METRIC_KEYS
from hrrp_osr.models.arpl import ARPLReciprocalHead
from hrrp_osr.models.cssr_1d import PCSSRCore1D
from hrrp_osr.models.ms_mean_factorial import MSMeanHeadFactorialModel
from hrrp_osr.training.arpl_pilot import (
    PreparedSurrogateSplit,
    _resolve_device,
    _set_determinism,
)
from hrrp_osr.training.ms_mean_head_factorial import (
    _build_model,
    _prepare_split,
    infer_model,
    load_ms_mean_head_factorial_config,
)


EXPERIMENT_ID = "fg_mv_cssr_frozen_r2_v1"
CONFIG_RELATIVE_PATH = "configs/experiments/cssr/fg_mv_cssr_frozen_r2_v1.yaml"
PRIOR_CONFIG_RELATIVE_PATH = (
    "configs/experiments/arpl/ms_mean_head_factorial_surrogate_v1.yaml"
)
R2_METHOD = "R2_MS_MEAN_CE"
R2_SEED = 20260830
CSSR_SEED = 20260903
ANGLE_FOLD = 0
PILOT_PAIRS = ("N1", "N4", "N2")
CONFIRMATION_PAIRS = ("N0", "N3", "N5", "N6")
SCORE_RULES = (
    "B0_GLOBAL_MLS",
    "B1_CLASS_CONDITIONAL_MLS",
    "B2_INDEPENDENT_VIEW_CSSR",
    "B3_COMMON_CLASS_CSSR",
    "B4_FUSION_GUIDED_CSSR",
)
TASK_SOURCE_FILES = (
    CONFIG_RELATIVE_PATH,
    "src/hrrp_osr/data/manifest.py",
    "src/hrrp_osr/data/processed.py",
    "src/hrrp_osr/evaluation/metrics.py",
    "src/hrrp_osr/evaluation/ms_mean_factorial.py",
    "src/hrrp_osr/models/cssr_1d.py",
    "src/hrrp_osr/models/hrrp_ms_resnet.py",
    "src/hrrp_osr/models/ms_mean_factorial.py",
    "src/hrrp_osr/training/arpl_pilot.py",
    "src/hrrp_osr/training/ms_mean_head_factorial.py",
    "src/hrrp_osr/training/fg_mv_cssr_pilot.py",
)
GATE_TOLERANCE = 1.0e-12
EXPECTED_R2_ROOT_HASH_MANIFEST_SHA256 = (
    "edcf281df07443724d0ade1a0b2d8b20305f85b83099fb74e1c6417ee5d5477c"
)
EXPECTED_R2_UNIT_HASHES = {
    "N0": {
        "checkpoint.pt": "142a85b3a090213684126cf695b08fec259724a0bd8399dc1adb40b114aab192",
        "pair_manifest.csv": "37dac18016223e08451c6551e279a6136ed494cb9c86edb5f0a938d71a2b115d",
        "features_logits_scores.npz": "a6bc7f4b1c095976964716e70c72666c0133f0bb3b5a63aacbc5612d9a888c93",
    },
    "N1": {
        "checkpoint.pt": "a4f6fa3235fbb5cf74b712588a0318f614a05287adec4ee881820424cddbcbaa",
        "pair_manifest.csv": "0b8a97dcfd744896bbae912c1363379201ced18a55107f80b2d2f3256fb5c5bc",
        "features_logits_scores.npz": "b43da73179b8ddb0e0ae1f97b3724e9fcffe9ce32f10aaa6466cc8f408a74275",
    },
    "N2": {
        "checkpoint.pt": "14e2ac7b686c901112f969fe0bd7f53c29646e7c015bae794d30c39051f9c0b9",
        "pair_manifest.csv": "1a7dc0031cf5b32a41131289fb4117a144463c025e93bc7a487e56a3c8c8bd2d",
        "features_logits_scores.npz": "58e8086e8ba27e2c4537d98d5ec1e6faaaa1bdf3d47cfec6ad9278227279114a",
    },
    "N3": {
        "checkpoint.pt": "6427a09f3e4a5e67ff652fea6e44c8364b62381acc8338099dccc818ac284bc9",
        "pair_manifest.csv": "53fead93617851f8646dc7c76ff3773b6c55a720d3be17feda462535994e7d27",
        "features_logits_scores.npz": "05ef84488ee515e09afbdf4504fdd1dd8597347faa63442a49abc32265caa6e8",
    },
    "N4": {
        "checkpoint.pt": "169387ad7a87463110ac7a2cd45afd7dac49428538c93c84975162e425d94ff5",
        "pair_manifest.csv": "8b0202d1e08ae83eec4bf07fc1dbb6a3f39fef2378ac15e57635709d8872b41a",
        "features_logits_scores.npz": "942e6c14d2237120ca9937a23df7f095ce718ea072933ee52d0ab2d3c3c79e95",
    },
    "N5": {
        "checkpoint.pt": "74cde2c6b30f1fa96219fe20777dfc632575c8c3c0281706ca016ef2497642df",
        "pair_manifest.csv": "a706c63e47f8522510c2926e70a8072ca8ca183c5ef74957b8451d28d2c47c80",
        "features_logits_scores.npz": "ab748cce5fbb8f1299fe720311e2c1da3805bda070b3b47b5db46f886805eee5",
    },
    "N6": {
        "checkpoint.pt": "178dbaa9e461d28825124b688752ed5c1005a8f0265963ef57e5c27a0a65e86e",
        "pair_manifest.csv": "46b454fc313573121fcf6ad214b91f9e21a2cb996a38d3beaf9c83d8321ce140",
        "features_logits_scores.npz": "b11a125b08a236f182857030fc07770d22a1890f5d20e68efa0f3fade4a4b20b",
    },
}


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DataConfigError(f"{name} must be a mapping")
    return value


def _require(
    errors: list[str], observed: Any, expected: Any, name: str
) -> None:
    if observed != expected:
        errors.append(f"{name} changed: expected {expected!r}, observed {observed!r}")


def load_fg_mv_cssr_config(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate the preregistered fast CSSR experiment."""

    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = dict(_mapping(yaml.safe_load(handle), "CSSR config"))
    errors: list[str] = []
    _require(errors, config.get("schema_version"), 1, "schema_version")
    _require(
        errors,
        config.get("stage"),
        "P3_frozen_r2_cssr_fast_iteration",
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
        "surrogate_unknown_used_for_cssr_training",
        "surrogate_unknown_used_for_reference_distribution",
        "surrogate_unknown_used_for_threshold",
        "known_calibration_used_for_cssr_training",
        "r2_retrained_or_finetuned",
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
        "method": R2_METHOD,
        "phase": "confirmation",
        "angle_fold": ANGLE_FOLD,
        "initialization_seed": R2_SEED,
        "checkpoint_epoch": 100,
        "checkpoint_selection": "fixed_final_epoch",
        "source_config": PRIOR_CONFIG_RELATIVE_PATH,
        "source_config_sha256": "c11daa6e2e5a7d7b72bc36840e60fc871f332c4fc85652636c729aa2eba14c71",
        "unit_relative_template": "{pair_id}/fold_0/seed_20260830/R2_MS_MEAN_CE",
        "root_artifact_hash_manifest_sha256": EXPECTED_R2_ROOT_HASH_MANIFEST_SHA256,
        "strict_load_required": True,
        "frozen": True,
        "old_logits_exact_match_required": True,
    }
    for name, expected in expected_prior.items():
        _require(errors, prior.get(name), expected, f"prior_r2.{name}")
    _require(
        errors,
        {
            str(pair_id): dict(_mapping(values, f"prior_r2 unit {pair_id}"))
            for pair_id, values in _mapping(
                prior.get("unit_artifact_hashes"), "prior_r2.unit_artifact_hashes"
            ).items()
        },
        EXPECTED_R2_UNIT_HASHES,
        "prior_r2.unit_artifact_hashes",
    )

    reference = _mapping(config.get("official_reference"), "official_reference")
    _require(errors, reference.get("repository"), "https://github.com/xyzedd/CSSR", "official repository")
    _require(
        errors,
        reference.get("commit"),
        "d5a99e91f310ec274c7bfe5796fb270719a07ab3",
        "official commit",
    )
    expected_official_hashes = {
        "methods/cssr.py": "0d23558c6a3cc4bf068036502a8ab43ee6278aecd91d96741f7375a142d9c5a3",
        "methods/cssr_ft.py": "31244f194d91f6cab0bdf34eb14a0ed3b58f25b6c49a44042bb96baa9977fb16",
        "configs/basic.json": "672375c6838004ae604509ba57098c7fefd17b6ac0f38e7c955fc8c09ba3192a",
        "configs/pcssr.json": "353b0768cc6ee60ac76c110a22da8bdb5c15179260d4abeb2f43fee422d24c6b",
        "configs/rcssr.json": "af40084644b4794559403f91e9d43a3008420df78484d87ec825e6d48b3d6f68",
    }
    _require(errors, dict(_mapping(reference.get("files"), "official files")), expected_official_hashes, "official hashes")
    _require(errors, reference.get("implementation_scope"), "PCSSR_CORE_1D", "implementation scope")
    _require(errors, reference.get("complete_pcssr_claimed"), False, "complete pCSSR claim")

    bundle = _mapping(config.get("bundle"), "bundle")
    expected_bundle = {
        "dataset_id": "hrrp_10class_theta83_hh_v1",
        "preprocessing_id": "hrrp_padding_complex_gaussian_v1",
        "profiles_sha256": "2dd92282c125f0f677cf1f2dfce828781c8ba4385cf9ae552c4a2c56033c3f5b",
        "manifest_sha256": "748b9f30629c3b3cbe66c6a1dac30863fdab2d81a214e46d8bc3ef7c6022a08a",
        "bundle_sha256": "79176091b5b745e5df5957f1bb4ade7304c1403a4a32e64622e012db67d5a0c5",
    }
    for name, expected in expected_bundle.items():
        _require(errors, bundle.get(name), expected, f"bundle.{name}")

    classes = _mapping(config.get("classes"), "classes")
    pair_rows = list(classes.get("identity_pairs", []))
    expected_pair_rows = [
        {"pair_id": "N0", "surrogate_unknown_indices": [0, 2], "train_known_indices": [1, 3, 4, 5, 6]},
        {"pair_id": "N1", "surrogate_unknown_indices": [2, 5], "train_known_indices": [0, 1, 3, 4, 6]},
        {"pair_id": "N2", "surrogate_unknown_indices": [3, 5], "train_known_indices": [0, 1, 2, 4, 6]},
        {"pair_id": "N3", "surrogate_unknown_indices": [1, 3], "train_known_indices": [0, 2, 4, 5, 6]},
        {"pair_id": "N4", "surrogate_unknown_indices": [1, 6], "train_known_indices": [0, 2, 3, 4, 5]},
        {"pair_id": "N5", "surrogate_unknown_indices": [4, 6], "train_known_indices": [0, 1, 2, 3, 5]},
        {"pair_id": "N6", "surrogate_unknown_indices": [0, 4], "train_known_indices": [1, 2, 3, 5, 6]},
    ]
    _require(errors, pair_rows, expected_pair_rows, "identity_pairs")
    _require(errors, list(classes.get("pilot_pairs", [])), list(PILOT_PAIRS), "pilot pairs")
    _require(errors, list(classes.get("confirmation_pairs", [])), list(CONFIRMATION_PAIRS), "confirmation pairs")

    data = _mapping(config.get("data"), "data")
    expected_data = {
        "angle_fold": ANGLE_FOLD,
        "development_angle_parity": "odd",
        "view_count": 2,
        "distinct_frames": True,
        "slot_order": "randomized_seeded",
        "pairs_per_class": 500,
        "cssr_train_population": "unique_train_known_base_samples_only",
        "cssr_reference_population": "unique_known_calibration_base_samples_by_true_class",
        "pair_multiplicity_weight": False,
        "final_test_pairs_generated": False,
    }
    for name, expected in expected_data.items():
        _require(errors, data.get(name), expected, f"data.{name}")
    _require(
        errors,
        dict(_mapping(data.get("smoke"), "data.smoke")),
        {
            "pair_id": "N1",
            "unique_train_base_samples_per_class": 2,
            "evaluation_pairs_per_class": 2,
            "cssr_epochs": 1,
        },
        "smoke contract",
    )

    normalization = _mapping(config.get("normalization"), "normalization")
    _require(errors, normalization.get("method"), "reuse_exact_r2_global_scalar_zscore", "normalization method")
    _require(errors, float(normalization.get("epsilon", -1)), 1.0e-8, "normalization epsilon")
    r2_model = _mapping(config.get("r2_model"), "r2_model")
    for name, expected in {
        "architecture": "ms_mean_head_factorial_v1",
        "encoder": "hrrp_ms_resnet_1d_v1",
        "input_length": 601,
        "feature_map_shape": [128, 76],
        "pooled_feature_dim": 128,
        "fusion": "arithmetic_mean",
        "head": "linear_ce",
        "prediction": "fused_logits_argmax",
        "feature_map_interface": "forward_feature_map",
        "feature_map_location": "after_final_residual_stage_before_global_pooling",
    }.items():
        _require(errors, r2_model.get(name), expected, f"r2_model.{name}")

    core = _mapping(config.get("pcssr_core_1d"), "pcssr_core_1d")
    expected_core = {
        "classes": "one_independent_autoencoder_per_train_known_class",
        "input_channels": 128,
        "latent_channels": 64,
        "encoder": "Conv1d_1x1_then_Tanh",
        "decoder": "Conv1d_1x1",
        "bias": False,
        "skip_connection": False,
        "gamma": 0.1,
        "reconstruction_error": "channel_sum_L1",
        "reconstruction_logit": "negative_gamma_error",
        "clip_min": -100.0,
        "clip_max": 100.0,
        "probability_order": "class_softmax_per_position_then_position_mean",
        "loss": "negative_log_probability_of_true_class",
        "scale_normalization": "per_position_mean_absolute_activation_squared",
        "rho_direction": "larger_is_more_inconsistent",
        "rho_epsilon": 1.0e-8,
        "official_full_score_ensemble_used": False,
    }
    for name, expected in expected_core.items():
        observed = float(core.get(name)) if isinstance(expected, float) else core.get(name)
        _require(errors, observed, expected, f"pcssr_core_1d.{name}")

    training = _mapping(config.get("cssr_training"), "cssr_training")
    expected_training = {
        "optimizer": "AdamW",
        "learning_rate": 1.0e-3,
        "weight_decay": 1.0e-4,
        "batch_size": 128,
        "epochs": 30,
        "scheduler": "none",
        "initialization_seed": CSSR_SEED,
        "dataloader_seed": CSSR_SEED,
        "early_stopping": False,
        "augmentation": "none",
        "formal_checkpoint_epoch": 30,
        "performance_checkpoint_selection": False,
        "known_calibration_diagnostic_only": True,
    }
    for name, expected in expected_training.items():
        observed = float(training.get(name)) if isinstance(expected, float) else training.get(name)
        _require(errors, observed, expected, f"cssr_training.{name}")

    calibration = _mapping(config.get("calibration"), "calibration")
    expected_calibration = {
        "cssr_reference_tail": "greater_than_or_equal_rho",
        "cssr_p_value_smoothing": "plus_one_numerator_and_denominator",
        "cssr_score_transform": "negative_log_p_plus_epsilon",
        "cssr_leave_one_base_sample_out": True,
        "mls_reference": "correctly_predicted_known_calibration_pairs_by_true_class",
        "mls_nonconformity": "negative_maximum_fused_logit",
        "mls_anomaly_quantile_tail": "less_than_or_equal_nonconformity",
        "mls_leave_one_pair_out": True,
        "empty_class_reference_policy": "fail",
        "score_epsilon": 1.0e-8,
        "threshold_source": "known_calibration_only",
        "threshold_known_acceptance_rate": 0.95,
        "threshold_rule": "exact_sorted_rank_ceiling",
    }
    for name, expected in expected_calibration.items():
        observed = float(calibration.get(name)) if isinstance(expected, float) else calibration.get(name)
        _require(errors, observed, expected, f"calibration.{name}")

    _require(errors, list(config.get("scores", [])), list(SCORE_RULES), "score rules")
    _require(errors, config.get("known_prediction_source"), "frozen_r2_fused_ce_for_all_rules", "known prediction source")
    _require(errors, config.get("unknown_score_direction"), "larger_is_more_unknown", "score direction")
    prohibited = set(config.get("prohibited", []))
    _require(
        errors,
        prohibited,
        {"max_view", "score_weighting", "mls_cssr_linear_combination", "learned_score_fusion", "top2_class_mixture", "mlp_rejector"},
        "prohibited scores",
    )

    evaluation = _mapping(config.get("evaluation"), "evaluation")
    _require(errors, list(evaluation.get("report_metrics", [])), list(REPORT_METRIC_KEYS), "report metrics")
    _require(errors, evaluation.get("per_surrogate_identity"), True, "per-identity metrics")
    _require(errors, evaluation.get("absorption_by_known_class"), True, "absorption audit")
    _require(errors, evaluation.get("angle_reconstruction_diagnostic_only"), True, "angle audit")

    pilot_gate = _mapping(config.get("pilot_gate"), "pilot_gate")
    confirmation_gate = _mapping(config.get("confirmation_gate"), "confirmation_gate")
    for gate, name, count in ((pilot_gate, "pilot_gate", 2), (confirmation_gate, "confirmation_gate", 3)):
        _require(errors, gate.get("baseline"), SCORE_RULES[1], f"{name}.baseline")
        _require(errors, float(gate.get("minimum_mean_auroc_delta", -1)), 0.02, f"{name}.minimum_mean_auroc_delta")
        _require(errors, int(gate.get("minimum_positive_pair_count", -1)), count, f"{name}.minimum_positive_pair_count")
        _require(errors, float(gate.get("minimum_mean_oscr_delta", -1)), 0.0, f"{name}.minimum_mean_oscr_delta")
        _require(errors, float(gate.get("maximum_mean_kccr_drop", -1)), 0.01, f"{name}.maximum_mean_kccr_drop")
        _require(errors, float(gate.get("maximum_mean_fpr95_increase", -1)), 0.02, f"{name}.maximum_mean_fpr95_increase")
    _require(errors, list(pilot_gate.get("candidates", [])), [SCORE_RULES[3], SCORE_RULES[4]], "pilot candidates")
    _require(errors, pilot_gate.get("no_cssr_signal_runs_confirmation"), False, "conditional confirmation")

    runtime = _mapping(config.get("runtime"), "runtime")
    expected_runtime = {
        "formal_device": "cuda",
        "expected_gpu_model": "NVIDIA GeForce RTX 4090",
        "maximum_parallel_tasks": 4,
        "deterministic_algorithms": True,
        "amp": False,
        "tf32": False,
        "torch_compile": False,
        "num_workers": 0,
    }
    for name, expected in expected_runtime.items():
        _require(errors, runtime.get(name), expected, f"runtime.{name}")
    decisions = _mapping(config.get("decisions"), "decisions")
    for name in (
        "final_unknown_test_authorized",
        "cssr_arpl_combination_authorized",
        "end_to_end_cssr_authorized",
    ):
        _require(errors, decisions.get(name), False, f"decisions.{name}")
    outputs = _mapping(config.get("outputs"), "outputs")
    _require(errors, outputs.get("namespace"), "artifacts/cssr/fg_mv_cssr_frozen_r2_v1", "output namespace")
    _require(errors, outputs.get("fail_if_output_nonempty"), True, "output overwrite policy")
    if errors:
        raise DataConfigError("Invalid frozen-R2 CSSR config:\n- " + "\n- ".join(errors))
    config["_config_path"] = str(config_path)
    config["_config_sha256"] = file_sha256(config_path)
    return config


def build_phase_plan(config: Mapping[str, Any], phase: str) -> list[dict[str, Any]]:
    pair_ids: Sequence[str]
    if phase == "smoke":
        pair_ids = (str(config["data"]["smoke"]["pair_id"]),)
    elif phase == "pilot":
        pair_ids = tuple(config["classes"]["pilot_pairs"])
    elif phase == "confirmation":
        pair_ids = tuple(config["classes"]["confirmation_pairs"])
    else:
        raise DataValidationError("phase must be smoke, pilot, or confirmation")
    return [
        {
            "phase": phase,
            "mode": "smoke" if phase == "smoke" else "full",
            "pair_id": pair_id,
            "angle_fold": ANGLE_FOLD,
            "r2_seed": R2_SEED,
            "cssr_seed": CSSR_SEED,
        }
        for pair_id in pair_ids
    ]


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _sequence_sha256(values: Iterable[Any]) -> str:
    payload = json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cssr_conformal_p_values(
    rho: np.ndarray,
    reference_values: Sequence[np.ndarray],
    *,
    sample_ids: Sequence[str] | None = None,
    reference_sample_ids: Sequence[Sequence[str]] | None = None,
    true_labels: np.ndarray | None = None,
    leave_one_base_sample_out: bool = False,
) -> np.ndarray:
    """Convert all-class reconstruction inconsistency to conformal p-values."""

    values = np.asarray(rho, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(reference_values):
        raise DataValidationError("rho/reference class dimensions differ")
    if not np.isfinite(values).all():
        raise DataValidationError("rho contains NaN or Inf")
    if leave_one_base_sample_out:
        if sample_ids is None or reference_sample_ids is None or true_labels is None:
            raise DataValidationError("leave-one-out requires sample IDs and true labels")
        if len(sample_ids) != values.shape[0] or len(reference_sample_ids) != values.shape[1]:
            raise DataValidationError("leave-one-out metadata shape mismatch")
        labels = np.asarray(true_labels, dtype=np.int64)
        if labels.shape != (values.shape[0],):
            raise DataValidationError("true_labels shape mismatch")
    else:
        labels = np.full(values.shape[0], -1, dtype=np.int64)
    result = np.empty_like(values)
    for row_index in range(values.shape[0]):
        for class_index, source in enumerate(reference_values):
            reference = np.asarray(source, dtype=np.float64)
            if reference.ndim != 1 or not np.isfinite(reference).all():
                raise DataValidationError("CSSR reference values must be finite vectors")
            if leave_one_base_sample_out and int(labels[row_index]) == class_index:
                ids = tuple(str(value) for value in reference_sample_ids[class_index])
                if len(ids) != reference.size:
                    raise DataValidationError("CSSR reference IDs do not align with values")
                keep = np.asarray([value != str(sample_ids[row_index]) for value in ids])
                reference = reference[keep]
            if reference.size == 0:
                raise DataValidationError("CSSR class reference is empty after leave-one-out")
            result[row_index, class_index] = (
                1.0 + float(np.count_nonzero(reference >= values[row_index, class_index]))
            ) / (float(reference.size) + 1.0)
    return result


def compute_conformal_p_values(
    query: np.ndarray,
    reference: np.ndarray,
    *,
    query_sample_ids: Sequence[str] | None = None,
    reference_sample_ids: Sequence[str] | None = None,
    leave_one_out: bool = False,
) -> np.ndarray:
    """One-class convenience API used by protocol tests and reference fitting."""

    query_values = np.asarray(query, dtype=np.float64)
    reference_values = np.asarray(reference, dtype=np.float64)
    if query_values.ndim != 1 or reference_values.ndim != 1:
        raise DataValidationError("conformal query and reference must be vectors")
    if not np.isfinite(query_values).all() or not np.isfinite(reference_values).all():
        raise DataValidationError("conformal values contain NaN or Inf")
    if leave_one_out and (query_sample_ids is None or reference_sample_ids is None):
        raise DataValidationError("leave-one-out requires sample ID metadata")
    if query_sample_ids is not None and len(query_sample_ids) != query_values.size:
        raise DataValidationError("query sample IDs do not align")
    if reference_sample_ids is not None and len(reference_sample_ids) != reference_values.size:
        raise DataValidationError("reference sample IDs do not align")
    result = np.empty(query_values.size, dtype=np.float64)
    for index, value in enumerate(query_values):
        current = reference_values
        if leave_one_out:
            current = reference_values[
                np.asarray(
                    [
                        str(sample_id) != str(query_sample_ids[index])
                        for sample_id in reference_sample_ids
                    ],
                    dtype=bool,
                )
            ]
        if current.size == 0:
            raise DataValidationError("conformal reference is empty after leave-one-out")
        result[index] = (1.0 + float(np.count_nonzero(current >= value))) / (
            float(current.size) + 1.0
        )
    return result


def class_conditional_mls_scores(
    calibration_logits: np.ndarray,
    calibration_true: np.ndarray,
    target_logits: np.ndarray,
    *,
    target_calibration_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Prediction-class empirical MLS anomaly score with optional pair LOO."""

    cal_logits = np.asarray(calibration_logits, dtype=np.float64)
    target = np.asarray(target_logits, dtype=np.float64)
    cal_true = np.asarray(calibration_true, dtype=np.int64)
    if cal_logits.ndim != 2 or target.ndim != 2 or cal_logits.shape[1] != target.shape[1]:
        raise DataValidationError("MLS logits must be [samples, same classes]")
    if cal_true.shape != (cal_logits.shape[0],):
        raise DataValidationError("MLS calibration labels do not align")
    cal_pred = cal_logits.argmax(axis=1)
    target_pred = target.argmax(axis=1)
    cal_nonconformity = -cal_logits.max(axis=1)
    target_nonconformity = -target.max(axis=1)
    references: list[tuple[np.ndarray, np.ndarray]] = []
    for class_index in range(cal_logits.shape[1]):
        indices = np.flatnonzero((cal_true == class_index) & (cal_pred == class_index))
        if indices.size == 0:
            raise DataValidationError(f"MLS class {class_index} reference is empty")
        references.append((cal_nonconformity[indices], indices))
    if target_calibration_indices is not None:
        target_indices = np.asarray(target_calibration_indices, dtype=np.int64)
        if target_indices.shape != (target.shape[0],):
            raise DataValidationError("target calibration indices do not align")
    else:
        target_indices = np.full(target.shape[0], -1, dtype=np.int64)
    scores = np.empty(target.shape[0], dtype=np.float64)
    for row_index, predicted in enumerate(target_pred):
        reference, indices = references[int(predicted)]
        if target_indices[row_index] >= 0:
            reference = reference[indices != target_indices[row_index]]
        if reference.size == 0:
            raise DataValidationError("MLS class reference is empty after leave-one-pair-out")
        scores[row_index] = (
            1.0 + float(np.count_nonzero(reference <= target_nonconformity[row_index]))
        ) / (float(reference.size) + 1.0)
    return scores


def compute_class_conditional_mls_scores(
    query_nonconformity: np.ndarray,
    query_predicted_labels: np.ndarray,
    *,
    reference_nonconformity: np.ndarray,
    reference_true_labels: np.ndarray,
    reference_predicted_labels: np.ndarray,
    query_pair_ids: Sequence[str] | None = None,
    reference_pair_ids: Sequence[str] | None = None,
    leave_one_out: bool = False,
) -> np.ndarray:
    """Direct class-conditional MLS implementation on preregistered r=-maxlogit."""

    query = np.asarray(query_nonconformity, dtype=np.float64)
    query_pred = np.asarray(query_predicted_labels, dtype=np.int64)
    reference = np.asarray(reference_nonconformity, dtype=np.float64)
    reference_true = np.asarray(reference_true_labels, dtype=np.int64)
    reference_pred = np.asarray(reference_predicted_labels, dtype=np.int64)
    if query.ndim != 1 or query_pred.shape != query.shape:
        raise DataValidationError("MLS query arrays do not align")
    if reference.ndim != 1 or reference_true.shape != reference.shape or reference_pred.shape != reference.shape:
        raise DataValidationError("MLS reference arrays do not align")
    if leave_one_out and (query_pair_ids is None or reference_pair_ids is None):
        raise DataValidationError("MLS leave-one-out requires pair IDs")
    if query_pair_ids is not None and len(query_pair_ids) != query.size:
        raise DataValidationError("MLS query pair IDs do not align")
    if reference_pair_ids is not None and len(reference_pair_ids) != reference.size:
        raise DataValidationError("MLS reference pair IDs do not align")
    result = np.empty(query.size, dtype=np.float64)
    for index, (value, predicted) in enumerate(zip(query, query_pred, strict=True)):
        keep = (reference_true == int(predicted)) & (reference_pred == int(predicted))
        if leave_one_out:
            keep &= np.asarray(
                [
                    str(pair_id) != str(query_pair_ids[index])
                    for pair_id in reference_pair_ids
                ],
                dtype=bool,
            )
        current = reference[keep]
        if current.size == 0:
            raise DataValidationError(f"MLS class {int(predicted)} reference is empty")
        result[index] = (1.0 + float(np.count_nonzero(current <= value))) / (
            float(current.size) + 1.0
        )
    return result


def compute_b0_b4_scores(
    fused_logits: np.ndarray,
    view_nonconformity: np.ndarray,
    b1_scores: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute B0 and B2-B4 exactly; B1 is calibrated separately."""

    logits = np.asarray(fused_logits, dtype=np.float64)
    values = np.asarray(view_nonconformity, dtype=np.float64)
    if logits.ndim != 2 or values.ndim != 3:
        raise DataValidationError("logits/a must be [n,k] and [n,v,k]")
    if values.shape[0] != logits.shape[0] or values.shape[2] != logits.shape[1]:
        raise DataValidationError("logits/a dimensions differ")
    if values.shape[1] != 2:
        raise DataValidationError("frozen experiment requires exactly two views")
    if not np.isfinite(logits).all() or not np.isfinite(values).all():
        raise DataValidationError("score inputs contain NaN or Inf")
    y_hat = logits.argmax(axis=1).astype(np.int64)
    common_by_class = values.mean(axis=1)
    k_common = common_by_class.argmin(axis=1).astype(np.int64)
    row_indices = np.arange(logits.shape[0])
    scores: dict[str, np.ndarray] = {
        "B0_GLOBAL_MLS": -logits.max(axis=1),
        "B2_INDEPENDENT_VIEW_CSSR": values.min(axis=2).mean(axis=1),
        "B3_COMMON_CLASS_CSSR": common_by_class.min(axis=1),
        "B4_FUSION_GUIDED_CSSR": common_by_class[row_indices, y_hat],
        "known_prediction": y_hat,
        "k_common": k_common,
    }
    if b1_scores is not None:
        b1 = np.asarray(b1_scores, dtype=np.float64)
        if b1.shape != (logits.shape[0],) or not np.isfinite(b1).all():
            raise DataValidationError("B1 scores do not align")
        scores["B1_CLASS_CONDITIONAL_MLS"] = b1
    return scores


def _candidate_gate(
    metrics_by_pair: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    candidate: str,
    baseline: str,
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    pair_rows: list[dict[str, Any]] = []
    for pair_id in metrics_by_pair:
        pair = metrics_by_pair[pair_id]
        if candidate not in pair or baseline not in pair:
            raise DataValidationError(f"missing {candidate}/{baseline} metrics for {pair_id}")
        candidate_metrics = pair[candidate]
        baseline_metrics = pair[baseline]
        deltas = {
            "auroc": float(candidate_metrics["auroc"]) - float(baseline_metrics["auroc"]),
            "oscr": float(candidate_metrics["oscr"]) - float(baseline_metrics["oscr"]),
            "known_correct_acceptance_rate": float(candidate_metrics["known_correct_acceptance_rate"]) - float(baseline_metrics["known_correct_acceptance_rate"]),
            "fpr95": float(candidate_metrics["fpr95"]) - float(baseline_metrics["fpr95"]),
        }
        if not all(math.isfinite(value) for value in deltas.values()):
            raise DataValidationError("gate delta is not finite")
        pair_rows.append({"pair_id": pair_id, **{f"delta_{key}": value for key, value in deltas.items()}})
    mean = {
        key: float(np.mean([row[f"delta_{key}"] for row in pair_rows]))
        for key in ("auroc", "oscr", "known_correct_acceptance_rate", "fpr95")
    }
    positive_count = sum(row["delta_auroc"] > 0.0 for row in pair_rows)
    checks = {
        "mean_auroc_delta": mean["auroc"] + GATE_TOLERANCE >= float(gate["minimum_mean_auroc_delta"]),
        "positive_pair_count": positive_count >= int(gate["minimum_positive_pair_count"]),
        "mean_oscr_delta": mean["oscr"] + GATE_TOLERANCE >= float(gate["minimum_mean_oscr_delta"]),
        "mean_kccr_delta": mean["known_correct_acceptance_rate"] + GATE_TOLERANCE >= -float(gate["maximum_mean_kccr_drop"]),
        "mean_fpr95_delta": mean["fpr95"] <= float(gate["maximum_mean_fpr95_increase"]) + GATE_TOLERANCE,
    }
    return {
        "candidate": candidate,
        "baseline": baseline,
        "pair_deltas": pair_rows,
        "mean_deltas": mean,
        "positive_auroc_pair_count": positive_count,
        "checks": checks,
        "passed": all(checks.values()),
    }


def apply_pilot_gate(
    metrics_by_pair: Mapping[str, Mapping[str, Mapping[str, float]]],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    if tuple(metrics_by_pair) != PILOT_PAIRS:
        raise DataValidationError(f"pilot gate requires ordered pairs {PILOT_PAIRS}")
    baseline = str(gate["baseline"])
    b3 = _candidate_gate(metrics_by_pair, candidate=SCORE_RULES[3], baseline=baseline, gate=gate)
    b4 = _candidate_gate(metrics_by_pair, candidate=SCORE_RULES[4], baseline=baseline, gate=gate)
    b3_mean = float(np.mean([metrics_by_pair[pair][SCORE_RULES[3]]["auroc"] for pair in PILOT_PAIRS]))
    b4_mean = float(np.mean([metrics_by_pair[pair][SCORE_RULES[4]]["auroc"] for pair in PILOT_PAIRS]))
    if b4["passed"] and b4_mean + GATE_TOLERANCE >= b3_mean:
        signal, selected = "fusion_guided_signal", SCORE_RULES[4]
    elif b3["passed"] and b3_mean > b4_mean + GATE_TOLERANCE:
        signal, selected = "common_class_signal", SCORE_RULES[3]
    else:
        signal, selected = "no_cssr_signal", None
    return {
        "signal": signal,
        "selected_rule": selected,
        "confirmation_authorized": selected is not None,
        "mean_auroc": {SCORE_RULES[3]: b3_mean, SCORE_RULES[4]: b4_mean},
        "candidate_gates": {SCORE_RULES[3]: b3, SCORE_RULES[4]: b4},
    }


def apply_confirmation_gate(
    metrics_by_pair: Mapping[str, Mapping[str, Mapping[str, float]]],
    gate: Mapping[str, Any],
    selected_rule: str,
) -> dict[str, Any]:
    if tuple(metrics_by_pair) != CONFIRMATION_PAIRS:
        raise DataValidationError(f"confirmation requires ordered pairs {CONFIRMATION_PAIRS}")
    if selected_rule not in SCORE_RULES[3:]:
        raise DataValidationError("confirmation rule must be frozen B3 or B4")
    result = _candidate_gate(
        metrics_by_pair,
        candidate=selected_rule,
        baseline=str(gate["baseline"]),
        gate=gate,
    )
    return {
        **result,
        "decision": "worth_later_full_validation" if result["passed"] else "stop_cssr_route",
        "final_unknown_test_authorized": False,
        "cssr_arpl_combination_authorized": False,
    }


def _gate_rows_to_mapping(
    rows: Sequence[Mapping[str, Any]], expected_pairs: Sequence[str]
) -> dict[str, dict[str, dict[str, float]]]:
    expected_methods = {SCORE_RULES[1], SCORE_RULES[3], SCORE_RULES[4]}
    mapping: dict[str, dict[str, dict[str, float]]] = {
        pair_id: {} for pair_id in expected_pairs
    }
    for row in rows:
        pair_id = str(row.get("pair_id"))
        method = str(row.get("method"))
        if pair_id not in mapping or method not in expected_methods:
            raise DataValidationError("gate row has an unexpected pair or method")
        if method in mapping[pair_id]:
            raise DataValidationError("gate contains a duplicate pair/method row")
        mapping[pair_id][method] = {
            key: float(row[key])
            for key in (
                "auroc",
                "oscr",
                "known_correct_acceptance_rate",
                "fpr95",
            )
        }
    if any(set(mapping[pair_id]) != expected_methods for pair_id in expected_pairs):
        raise DataValidationError("gate rows are missing a required pair/method row")
    return mapping


def _flatten_candidate_gate(result: Mapping[str, Any]) -> dict[str, Any]:
    mean = result["mean_deltas"]
    return {
        "passed": bool(result["passed"]),
        "positive_pair_count": int(result["positive_auroc_pair_count"]),
        "mean_auroc_delta": float(mean["auroc"]),
        "mean_oscr_delta": float(mean["oscr"]),
        "mean_kccr_delta": float(mean["known_correct_acceptance_rate"]),
        "mean_fpr95_delta": float(mean["fpr95"]),
        "checks": dict(result["checks"]),
        "pair_deltas": list(result["pair_deltas"]),
    }


def evaluate_pilot_gate(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    mapping = _gate_rows_to_mapping(rows, PILOT_PAIRS)
    raw = apply_pilot_gate(mapping, config["pilot_gate"])
    return {
        "signal": raw["signal"],
        "selected_rule": raw["selected_rule"],
        "confirmation_allowed": bool(raw["confirmation_authorized"]),
        "mean_auroc": dict(raw["mean_auroc"]),
        "candidates": {
            rule: _flatten_candidate_gate(result)
            for rule, result in raw["candidate_gates"].items()
        },
    }


def evaluate_confirmation_gate(
    rows: Sequence[Mapping[str, Any]],
    selected_rule: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    mapping = _gate_rows_to_mapping(rows, CONFIRMATION_PAIRS)
    raw = apply_confirmation_gate(
        mapping, config["confirmation_gate"], selected_rule
    )
    flattened = _flatten_candidate_gate(raw)
    return {
        **flattened,
        "selected_rule": selected_rule,
        "decision": (
            "worth_later_full_validation_only"
            if flattened["passed"]
            else "stop_cssr_route"
        ),
        "final_unknown_test_authorized": False,
        "cssr_arpl_combination_authorized": False,
    }


def _atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_write_bytes(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )


def _render_csv(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise DataValidationError("cannot render an empty CSV")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_write_bytes(path, _render_csv(rows))


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _artifact_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name
        not in {"artifact_hashes.json", "_SUCCESS.json", "_PHASE_SUCCESS.json"}
    }


def task_source_hashes(project_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in TASK_SOURCE_FILES:
        path = project_root / relative
        if not path.is_file():
            raise DataValidationError(f"missing task source: {relative}")
        result[relative] = file_sha256(path)
    return result


def _spec_for_pair(config: Mapping[str, Any], pair_id: str) -> dict[str, Any]:
    matches = [
        dict(row)
        for row in config["classes"]["identity_pairs"]
        if str(row["pair_id"]) == pair_id
    ]
    if len(matches) != 1:
        raise DataValidationError(f"pair {pair_id} is outside the frozen plan")
    return {**matches[0], "angle_fold": ANGLE_FOLD, "unit_id": f"{pair_id}_F0"}


def _verify_artifact_hash_manifest(unit_root: Path) -> dict[str, str]:
    manifest_path = unit_root / "artifact_hashes.json"
    if not manifest_path.is_file():
        raise DataValidationError(f"missing prior artifact hashes: {manifest_path}")
    recorded = _read_json(manifest_path)
    if not isinstance(recorded, Mapping) or not recorded:
        raise DataValidationError("prior artifact hash manifest is invalid")
    mismatches = []
    for relative, expected in recorded.items():
        path = unit_root / str(relative)
        if not path.is_file() or file_sha256(path) != str(expected):
            mismatches.append(str(relative))
    if mismatches:
        raise DataValidationError(f"prior R2 artifact hash mismatch: {mismatches}")
    return {str(key): str(value) for key, value in recorded.items()}


def _load_prior_config(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    path = project_root / str(config["prior_r2"]["source_config"])
    if file_sha256(path) != str(config["prior_r2"]["source_config_sha256"]):
        raise DataValidationError("prior R2 config file hash changed")
    return load_ms_mean_head_factorial_config(path)


def _load_bundle(bundle_root: str | Path, config: Mapping[str, Any]) -> ProcessedBundle:
    bundle = config["bundle"]
    return load_processed_bundle(
        bundle_root,
        expected_profiles_sha256=str(bundle["profiles_sha256"]),
        expected_manifest_sha256=str(bundle["manifest_sha256"]),
        expected_bundle_sha256=str(bundle["bundle_sha256"]),
    )


def _prepare_frozen_split(
    bundle: ProcessedBundle,
    prior_config: Mapping[str, Any],
    cssr_config: Mapping[str, Any],
    pair_id: str,
) -> PreparedSurrogateSplit:
    spec = _spec_for_pair(cssr_config, pair_id)
    prepared = _prepare_split(bundle, prior_config, spec, mode="full")
    if prepared.angle_fold != ANGLE_FOLD:
        raise DataValidationError("R2 split angle fold changed")
    if len(prepared.train_class_order) != 5 or len(prepared.surrogate_class_order) != 2:
        raise DataValidationError("R2 split is not the frozen 5/2 partition")
    if any(
        int(row["view1_angle_deg"]) % 2 == 0
        or int(row["view2_angle_deg"]) % 2 == 0
        for row in prepared.pair_manifest_rows
    ):
        raise DataValidationError("even-angle sample entered CSSR split")
    return prepared


def load_and_audit_frozen_r2(
    *,
    project_root: Path,
    r2_results_root: str | Path,
    pair_id: str,
    config: Mapping[str, Any],
    prepared: PreparedSurrogateSplit,
    prior_config: Mapping[str, Any],
    device: torch.device,
) -> tuple[MSMeanHeadFactorialModel, dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Strict-load the old R2 and prove its manifest and outputs are unchanged."""

    relative = str(config["prior_r2"]["unit_relative_template"]).format(pair_id=pair_id)
    results_root = Path(r2_results_root).resolve()
    root_hash_manifest_path = results_root / "artifact_hashes.json"
    if (
        not root_hash_manifest_path.is_file()
        or file_sha256(root_hash_manifest_path)
        != str(config["prior_r2"]["root_artifact_hash_manifest_sha256"])
    ):
        raise DataValidationError("frozen R2 root artifact hash manifest changed")
    root_hash_manifest = _read_json(root_hash_manifest_path)
    expected_unit_hashes = dict(
        config["prior_r2"]["unit_artifact_hashes"][pair_id]
    )
    for filename, expected_hash in expected_unit_hashes.items():
        root_key = f"{relative}/{filename}"
        if root_hash_manifest.get(root_key) != expected_hash:
            raise DataValidationError(f"frozen R2 root binding changed for {root_key}")
    unit_root = results_root / relative
    if not unit_root.is_dir():
        raise DataValidationError(f"missing frozen R2 unit: {unit_root}")
    prior_hashes = _verify_artifact_hash_manifest(unit_root)
    for filename, expected_hash in expected_unit_hashes.items():
        path = unit_root / filename
        if not path.is_file() or file_sha256(path) != expected_hash:
            raise DataValidationError(
                f"frozen R2 expected artifact changed: {pair_id}/{filename}"
            )
    saved_manifest = unit_root / "pair_manifest.csv"
    saved_bytes = saved_manifest.read_bytes()
    saved_hash = file_sha256(saved_manifest)
    if (
        saved_bytes != prepared.pair_manifest_bytes
        or saved_hash != prepared.pair_manifest_sha256
    ):
        raise DataValidationError("rebuilt pair manifest differs from frozen R2 manifest")

    checkpoint_path = unit_root / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_checkpoint = {
        "experiment_id": str(config["prior_r2"]["experiment_id"]),
        "phase": "confirmation",
        "pair_id": pair_id,
        "angle_fold": ANGLE_FOLD,
        "method": R2_METHOD,
        "checkpoint_epoch": 100,
        "formal_checkpoint": True,
        "checkpoint_selection": "fixed_final_epoch",
        "initialization_seed": R2_SEED,
        "config_sha256": str(config["prior_r2"]["source_config_sha256"]),
        "pair_manifest_sha256": prepared.pair_manifest_sha256,
        "pseudo_unknown_generated": False,
    }
    for name, expected in expected_checkpoint.items():
        if checkpoint.get(name) != expected:
            raise DataValidationError(
                f"frozen R2 checkpoint {name} changed: {checkpoint.get(name)!r}"
            )
    if tuple(checkpoint.get("train_class_order", ())) != prepared.train_class_order:
        raise DataValidationError("frozen R2 train class order changed")
    if tuple(checkpoint.get("surrogate_class_order", ())) != prepared.surrogate_class_order:
        raise DataValidationError("frozen R2 surrogate class order changed")
    observed_normalization = {
        key: float(value) if key != "unique_base_sample_count" else int(value)
        for key, value in dict(checkpoint.get("normalization", {})).items()
    }
    expected_normalization = asdict(prepared.normalization)
    if observed_normalization != expected_normalization:
        raise DataValidationError("frozen R2 normalization changed")

    model = _build_model(R2_METHOD, len(prepared.train_class_order), prior_config)
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise DataValidationError("frozen R2 checkpoint was not strict-load compatible")
    model.requires_grad_(False).eval().to(device)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise DataValidationError("frozen R2 has trainable parameters")
    if any(isinstance(module, ARPLReciprocalHead) for module in model.modules()):
        raise DataValidationError("ARPL module was instantiated in frozen CE R2")
    if any(model.forbidden_component_status.values()):
        raise DataValidationError("frozen R2 contains a prohibited component")

    arrays = {
        role: infer_model(
            model,
            prepared.inputs[role],
            prepared.labels[role],
            device=device,
            batch_size=int(prior_config["training"]["batch_size"]),
        )
        for role in ("train", "known_calibration", "surrogate_unknown")
    }
    sealed_path = unit_root / "features_logits_scores.npz"
    exact_checks: dict[str, bool] = {}
    maximum_errors: dict[str, float] = {}
    with np.load(sealed_path, allow_pickle=False) as sealed:
        for role, role_arrays in arrays.items():
            for name, current in role_arrays.items():
                key = f"{role}_{name}"
                if key not in sealed:
                    raise DataValidationError(f"sealed R2 output lacks {key}")
                expected = sealed[key]
                if expected.shape != current.shape or expected.dtype != current.dtype:
                    raise DataValidationError(f"sealed R2 output shape/dtype changed for {key}")
                exact = bool(np.array_equal(expected, current))
                exact_checks[key] = exact
                maximum_errors[key] = (
                    0.0
                    if exact
                    else float(np.max(np.abs(expected.astype(np.float64) - current.astype(np.float64))))
                )
    if not all(exact_checks.values()):
        failed = [key for key, value in exact_checks.items() if not value]
        raise DataValidationError(f"frozen R2 output exact regression failed: {failed}")

    audit = {
        "status": "passed",
        "pair_id": pair_id,
        "unit_root": str(unit_root),
        "prior_formal_code_commit": config["prior_r2"]["formal_code_commit"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "pair_manifest_sha256": saved_hash,
        "artifact_hash_manifest_sha256": file_sha256(unit_root / "artifact_hashes.json"),
        "root_artifact_hash_manifest_sha256": file_sha256(
            root_hash_manifest_path
        ),
        "expected_unit_artifact_hashes": expected_unit_hashes,
        "artifact_count": len(prior_hashes),
        "strict_load": True,
        "state_dict_key_count": len(checkpoint["model_state_dict"]),
        "all_parameters_frozen": True,
        "arpl_module_instantiated": False,
        "old_outputs_exact": True,
        "old_output_exact_checks": exact_checks,
        "old_output_maximum_absolute_errors": maximum_errors,
        "train_class_order": list(prepared.train_class_order),
        "surrogate_class_order": list(prepared.surrogate_class_order),
        "normalization": expected_normalization,
        "final_unknown_used": False,
        "even_angle_test_used": False,
    }
    return model, arrays, audit


def build_unique_base_sample_manifest(
    prepared: PreparedSurrogateSplit,
    bundle: ProcessedBundle,
) -> list[dict[str, Any]]:
    """Expand pair views and deduplicate by role/sample_id without pair weights."""

    bundle_by_id = {str(row["sample_id"]): row for row in bundle.rows}
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    role_order = {"train_known": 0, "known_calibration": 1, "surrogate_unknown": 2}
    for pair_row in prepared.pair_manifest_rows:
        role = str(pair_row["experiment_role"])
        if role not in role_order:
            raise DataValidationError(f"unexpected pair role {role}")
        for view in (1, 2):
            sample_id = str(pair_row[f"view{view}_sample_id"])
            row_index = int(pair_row[f"view{view}_row_index"])
            class_name = str(pair_row["class_name"])
            angle = int(pair_row[f"view{view}_angle_deg"])
            frame = int(pair_row[f"view{view}_frame_id"])
            source = bundle_by_id.get(sample_id)
            if source is None:
                raise DataValidationError(f"pair references missing base sample {sample_id}")
            if (
                int(source["processed_row_index"]) != row_index
                or str(source["class_name"]) != class_name
                or int(source["angle_deg"]) != angle
            ):
                raise DataValidationError("pair/base sample metadata mismatch")
            if angle % 2 == 0:
                raise DataValidationError("even-angle sample entered unique base manifest")
            key = (role, sample_id)
            value = {
                "experiment_role": role,
                "sample_id": sample_id,
                "processed_row_index": row_index,
                "class_name": class_name,
                "model_label": int(pair_row["model_label"]),
                "angle_deg": angle,
                "frame_id": frame,
                "source_class_role": str(source["class_role"]),
            }
            if key in unique and unique[key] != value:
                raise DataValidationError("one base sample has inconsistent pair metadata")
            unique[key] = value
    rows = sorted(
        unique.values(),
        key=lambda row: (
            role_order[str(row["experiment_role"])],
            int(row["model_label"]),
            str(row["class_name"]),
            str(row["sample_id"]),
        ),
    )
    train = [row for row in rows if row["experiment_role"] == "train_known"]
    calibration = [row for row in rows if row["experiment_role"] == "known_calibration"]
    surrogate = [row for row in rows if row["experiment_role"] == "surrogate_unknown"]
    if len(train) != 720 or Counter(row["model_label"] for row in train) != Counter({index: 144 for index in range(5)}):
        raise DataValidationError("unique train-known base population is not 5 x 144")
    if len(calibration) != 180 or Counter(row["model_label"] for row in calibration) != Counter({index: 36 for index in range(5)}):
        raise DataValidationError("unique calibration base population is not 5 x 36")
    if len(surrogate) != 72:
        raise DataValidationError("unique surrogate base population is not 72")
    train_ids = {row["sample_id"] for row in train}
    eval_ids = {row["sample_id"] for row in calibration + surrogate}
    if train_ids & eval_ids:
        raise DataValidationError("base sample leaked between CSSR train and evaluation")
    return rows


def extract_frozen_feature_maps(
    *,
    model: MSMeanHeadFactorialModel,
    bundle: ProcessedBundle,
    prepared: PreparedSurrogateSplit,
    rows: Sequence[Mapping[str, Any]],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    indices = np.asarray([int(row["processed_row_index"]) for row in rows], dtype=np.int64)
    inputs = np.asarray(bundle.profiles[indices], dtype=np.float64)
    inputs = np.asarray(
        (inputs - float(prepared.normalization.mean)) / float(prepared.normalization.std),
        dtype=np.float32,
    )
    if not np.isfinite(inputs).all():
        raise DataValidationError("normalized unique-base inputs are not finite")
    model.eval()
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            tensor = torch.from_numpy(inputs[start : start + batch_size]).to(device)
            values = model.encoder.forward_feature_map(tensor)
            batches.append(values.detach().cpu().numpy().astype(np.float32))
    features = np.concatenate(batches, axis=0)
    if features.shape != (len(rows), 128, 76):
        raise DataValidationError(f"unexpected R2 feature-map shape {features.shape}")
    if not np.isfinite(features).all():
        raise DataValidationError("R2 feature maps contain NaN or Inf")
    return features, {
        "status": "passed",
        "shape": list(features.shape),
        "dtype": str(features.dtype),
        "feature_map_sha256": _array_sha256(features),
        "sample_id_order_sha256": _sequence_sha256(row["sample_id"] for row in rows),
        "input_shape": [128, 76],
        "all_finite": True,
        "r2_eval_mode": not model.training,
        "r2_parameters_frozen": not any(parameter.requires_grad for parameter in model.parameters()),
    }


def _cssr_diagnostics(
    model: PCSSRCore1D,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals = defaultdict(float)
    count = 0
    with torch.no_grad():
        for start in range(0, labels.numel(), batch_size):
            x = features[start : start + batch_size].to(device)
            y = labels[start : start + batch_size].to(device)
            loss, output = model.loss(x, y)
            errors = output.reconstruction_errors.mean(dim=-1)
            correct = errors.gather(1, y.unsqueeze(1)).squeeze(1)
            wrong = errors.masked_fill(
                torch.nn.functional.one_hot(y, model.num_classes).bool(),
                float("inf"),
            ).min(dim=1).values
            batch_count = int(y.numel())
            totals["loss"] += float(loss) * batch_count
            totals["accuracy"] += float((output.probabilities.argmax(dim=1) == y).sum())
            totals["correct_class_reconstruction_error"] += float(correct.sum())
            totals["minimum_wrong_class_reconstruction_error"] += float(wrong.sum())
            totals["reconstruction_margin_wrong_minus_correct"] += float((wrong - correct).sum())
            count += batch_count
    if count == 0:
        raise DataValidationError("CSSR diagnostic population is empty")
    result = {key: value / count for key, value in totals.items()}
    result["ae_weight_norm"] = math.sqrt(
        sum(float(parameter.detach().square().sum()) for parameter in model.parameters())
    )
    if not all(math.isfinite(value) for value in result.values()):
        raise DataValidationError("CSSR diagnostics contain NaN or Inf")
    return result


def train_pcssr_core(
    *,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    calibration_features: np.ndarray,
    calibration_labels: np.ndarray,
    config: Mapping[str, Any],
    device: torch.device,
    smoke: bool,
) -> tuple[PCSSRCore1D, list[dict[str, Any]], dict[str, Any]]:
    training = config["cssr_training"]
    core = config["pcssr_core_1d"]
    train_labels = np.asarray(train_labels, dtype=np.int64)
    calibration_labels = np.asarray(calibration_labels, dtype=np.int64)
    if Counter(train_labels.tolist()) != Counter({index: int(np.count_nonzero(train_labels == index)) for index in range(5)}):
        raise DataValidationError("CSSR train labels are invalid")
    class_counts = [int(np.count_nonzero(train_labels == index)) for index in range(5)]
    if not class_counts or len(set(class_counts)) != 1 or min(class_counts) <= 0:
        raise DataValidationError("CSSR train classes must be nonempty and balanced")
    _set_determinism(CSSR_SEED, bool(config["runtime"]["deterministic_algorithms"]))
    model = PCSSRCore1D(
        num_classes=5,
        input_channels=int(core["input_channels"]),
        latent_channels=int(core["latent_channels"]),
        gamma=float(core["gamma"]),
        clip_length=abs(float(core["clip_min"])),
        epsilon=float(core["rho_epsilon"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    x_train = torch.from_numpy(np.asarray(train_features, dtype=np.float32))
    y_train = torch.from_numpy(train_labels)
    x_cal = torch.from_numpy(np.asarray(calibration_features, dtype=np.float32))
    y_cal = torch.from_numpy(calibration_labels)
    batch_size = int(training["batch_size"])
    epochs = int(config["data"]["smoke"]["cssr_epochs"] if smoke else training["epochs"])
    generator = torch.Generator().manual_seed(int(training["dataloader_seed"]))
    log: list[dict[str, Any]] = []
    epoch_order_hashes: list[str] = []
    for epoch in range(1, epochs + 1):
        order = torch.randperm(y_train.numel(), generator=generator)
        epoch_order_hashes.append(_array_sha256(order.numpy().astype(np.int64)))
        dataset = TensorDataset(x_train[order], y_train[order])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for batch_features, batch_labels in DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(config["runtime"]["num_workers"]),
        ):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model.loss(batch_features.to(device), batch_labels.to(device))
            if not torch.isfinite(loss):
                raise DataValidationError("CSSR training loss is not finite")
            loss.backward()
            optimizer.step()
        train_diagnostics = _cssr_diagnostics(
            model, x_train, y_train, batch_size=batch_size, device=device
        )
        cal_diagnostics = _cssr_diagnostics(
            model, x_cal, y_cal, batch_size=batch_size, device=device
        )
        log.append(
            {
                "epoch": epoch,
                "train_cssr_loss": train_diagnostics["loss"],
                "train_cssr_accuracy": train_diagnostics["accuracy"],
                "known_calibration_cssr_loss_diagnostic": cal_diagnostics["loss"],
                "known_calibration_cssr_accuracy_diagnostic": cal_diagnostics["accuracy"],
                "train_correct_class_reconstruction_error": train_diagnostics["correct_class_reconstruction_error"],
                "train_minimum_wrong_class_reconstruction_error": train_diagnostics["minimum_wrong_class_reconstruction_error"],
                "train_reconstruction_margin_wrong_minus_correct": train_diagnostics["reconstruction_margin_wrong_minus_correct"],
                "known_calibration_correct_class_reconstruction_error_diagnostic": cal_diagnostics["correct_class_reconstruction_error"],
                "known_calibration_minimum_wrong_class_reconstruction_error_diagnostic": cal_diagnostics["minimum_wrong_class_reconstruction_error"],
                "known_calibration_reconstruction_margin_wrong_minus_correct_diagnostic": cal_diagnostics["reconstruction_margin_wrong_minus_correct"],
                "ae_weight_norm": train_diagnostics["ae_weight_norm"],
                "sample_order_sha256": epoch_order_hashes[-1],
                "checkpoint_selected_for_performance": False,
                "surrogate_unknown_used": False,
                "final_unknown_used": False,
            }
        )
    model.eval()
    weight_norms = {
        str(index): math.sqrt(
            sum(float(parameter.detach().square().sum()) for parameter in autoencoder.parameters())
        )
        for index, autoencoder in enumerate(model.class_autoencoders)
    }
    audit = {
        "status": "passed",
        "epochs": epochs,
        "formal_checkpoint_epoch": None if smoke else int(training["formal_checkpoint_epoch"]),
        "checkpoint_selection": "fixed_final_epoch",
        "early_stopping": False,
        "train_unique_sample_count": int(train_labels.size),
        "train_class_counts": class_counts,
        "known_calibration_used_for_training": False,
        "surrogate_unknown_used_for_training": False,
        "pair_multiplicity_weight": False,
        "epoch_sample_order_hashes": epoch_order_hashes,
        "training_order_sha256": _sequence_sha256(epoch_order_hashes),
        "class_ae_weight_norms": weight_norms,
        "all_parameters_finite": all(
            bool(torch.isfinite(parameter).all()) for parameter in model.parameters()
        ),
    }
    return model, log, audit


def infer_cssr_rho(
    model: PCSSRCore1D,
    features: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    values = torch.from_numpy(np.asarray(features, dtype=np.float32))
    collected: list[np.ndarray] = []
    error_collected: list[np.ndarray] = []
    floor_hit_count = 0
    position_count = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, values.shape[0], batch_size):
            batch = values[start : start + batch_size].to(device)
            output = model(batch)
            rho = model.reconstruction_inconsistency(batch, output=output)
            collected.append(rho.detach().cpu().numpy().astype(np.float64))
            error_collected.append(
                output.reconstruction_errors.mean(dim=-1)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            activation = batch.abs().mean(dim=1)
            floor_hit_count += int((activation < model.epsilon).sum())
            position_count += int(activation.numel())
    result = np.concatenate(collected, axis=0)
    reconstruction_error = np.concatenate(error_collected, axis=0)
    if result.shape != (features.shape[0], model.num_classes):
        raise DataValidationError("CSSR rho shape is invalid")
    if not np.isfinite(result).all():
        raise DataValidationError("CSSR rho contains NaN or Inf")
    if reconstruction_error.shape != result.shape or not np.isfinite(
        reconstruction_error
    ).all():
        raise DataValidationError("CSSR reconstruction errors are invalid")
    return result, reconstruction_error, {
        "rho_shape": list(result.shape),
        "rho_sha256": _array_sha256(result),
        "rho_min": float(result.min()),
        "rho_max": float(result.max()),
        "mean_l1_reconstruction_error_sha256": _array_sha256(
            reconstruction_error
        ),
        "activation_floor_epsilon": model.epsilon,
        "activation_floor_hit_count": floor_hit_count,
        "activation_position_count": position_count,
    }


def build_cssr_reference_scores(
    *,
    unique_rows: Sequence[Mapping[str, Any]],
    rho: np.ndarray,
    epsilon: float,
) -> tuple[
    dict[str, np.ndarray],
    list[np.ndarray],
    list[tuple[str, ...]],
    dict[str, Any],
]:
    if rho.shape != (len(unique_rows), 5):
        raise DataValidationError("unique base rows and rho do not align")
    calibration_indices = np.asarray(
        [
            index
            for index, row in enumerate(unique_rows)
            if row["experiment_role"] == "known_calibration"
        ],
        dtype=np.int64,
    )
    surrogate_indices = np.asarray(
        [
            index
            for index, row in enumerate(unique_rows)
            if row["experiment_role"] == "surrogate_unknown"
        ],
        dtype=np.int64,
    )
    calibration_labels = np.asarray(
        [int(unique_rows[index]["model_label"]) for index in calibration_indices],
        dtype=np.int64,
    )
    calibration_ids = tuple(
        str(unique_rows[index]["sample_id"]) for index in calibration_indices
    )
    references: list[np.ndarray] = []
    reference_ids: list[tuple[str, ...]] = []
    for class_index in range(5):
        selected = calibration_indices[calibration_labels == class_index]
        references.append(np.asarray(rho[selected, class_index], dtype=np.float64))
        reference_ids.append(
            tuple(str(unique_rows[index]["sample_id"]) for index in selected)
        )
    if [len(values) for values in references] != [36] * 5:
        raise DataValidationError("CSSR reference population is not 36 unique bases/class")
    calibration_p = cssr_conformal_p_values(
        rho[calibration_indices],
        references,
        sample_ids=calibration_ids,
        reference_sample_ids=reference_ids,
        true_labels=calibration_labels,
        leave_one_base_sample_out=True,
    )
    surrogate_p = cssr_conformal_p_values(rho[surrogate_indices], references)
    p_by_role = {
        "known_calibration": calibration_p,
        "surrogate_unknown": surrogate_p,
    }
    a_by_role = {
        role: -np.log(values + float(epsilon)) for role, values in p_by_role.items()
    }
    if not all(np.isfinite(values).all() for values in a_by_role.values()):
        raise DataValidationError("CSSR conformal anomaly contains NaN or Inf")
    sample_index_by_role = {
        "known_calibration": calibration_indices,
        "surrogate_unknown": surrogate_indices,
    }
    score_by_sample: dict[str, np.ndarray] = {}
    rho_by_sample: dict[str, np.ndarray] = {}
    p_by_sample: dict[str, np.ndarray] = {}
    for role, indices in sample_index_by_role.items():
        for local_index, unique_index in enumerate(indices):
            sample_id = str(unique_rows[int(unique_index)]["sample_id"])
            score_by_sample[sample_id] = a_by_role[role][local_index]
            rho_by_sample[sample_id] = rho[int(unique_index)]
            p_by_sample[sample_id] = p_by_role[role][local_index]
    arrays = {
        "rho": rho,
        "known_calibration_p": calibration_p,
        "known_calibration_a": a_by_role["known_calibration"],
        "surrogate_unknown_p": surrogate_p,
        "surrogate_unknown_a": a_by_role["surrogate_unknown"],
    }
    metadata = {
        "status": "passed",
        "reference_counts": [len(values) for values in references],
        "reference_sample_id_hashes": [
            _sequence_sha256(values) for values in reference_ids
        ],
        "calibration_leave_one_base_sample_out": True,
        "surrogate_unknown_in_reference": False,
        "tail": "reference_rho_greater_than_or_equal_query_rho",
        "smoothing": "plus_one_numerator_and_denominator",
        "epsilon": float(epsilon),
        "score_by_sample": score_by_sample,
        "rho_by_sample": rho_by_sample,
        "p_by_sample": p_by_sample,
    }
    return arrays, references, reference_ids, metadata


def _role_manifest_rows(
    prepared: PreparedSurrogateSplit, role: str
) -> list[dict[str, Any]]:
    experiment_role = {
        "train": "train_known",
        "known_calibration": "known_calibration",
        "surrogate_unknown": "surrogate_unknown",
    }[role]
    rows = [
        dict(row)
        for row in prepared.pair_manifest_rows
        if str(row["experiment_role"]) == experiment_role
    ]
    if len(rows) != len(prepared.pair_ids[role]):
        raise DataValidationError("role manifest rows do not align with pair arrays")
    if tuple(str(row["pair_id"]) for row in rows) != tuple(prepared.pair_ids[role]):
        raise DataValidationError("role pair ID order changed")
    return rows


def _smoke_pair_indices(
    prepared: PreparedSurrogateSplit, role: str, per_class: int
) -> np.ndarray:
    rows = _role_manifest_rows(prepared, role)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["class_name"])].append(index)
    expected_classes = (
        prepared.train_class_order
        if role == "known_calibration"
        else prepared.surrogate_class_order
    )
    if tuple(grouped) != tuple(expected_classes):
        raise DataValidationError("smoke pair class order changed")
    selected = [
        index
        for class_name in expected_classes
        for index in grouped[class_name][:per_class]
    ]
    if len(selected) != len(expected_classes) * per_class:
        raise DataValidationError("insufficient pairs for smoke subset")
    return np.asarray(selected, dtype=np.int64)


def _pair_a_values(
    rows: Sequence[Mapping[str, Any]],
    score_by_sample: Mapping[str, np.ndarray],
) -> np.ndarray:
    result = np.asarray(
        [
            [
                score_by_sample[str(row["view1_sample_id"])],
                score_by_sample[str(row["view2_sample_id"])],
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    if result.shape != (len(rows), 2, 5):
        raise DataValidationError("pair CSSR anomaly tensor shape is invalid")
    return result


def _evaluate_all_scores(
    *,
    known_logits: np.ndarray,
    known_labels: np.ndarray,
    known_pair_ids: Sequence[str],
    unknown_logits: np.ndarray,
    full_calibration_logits: np.ndarray,
    full_calibration_labels: np.ndarray,
    full_calibration_pair_ids: Sequence[str],
    known_a: np.ndarray,
    unknown_a: np.ndarray,
    acceptance_rate: float,
    known_calibration_indices: np.ndarray,
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray], dict[str, np.ndarray]]:
    full_calibration_pred = full_calibration_logits.argmax(axis=1)
    full_calibration_nonconformity = -full_calibration_logits.max(axis=1)
    known_pred = known_logits.argmax(axis=1)
    unknown_pred = unknown_logits.argmax(axis=1)
    known_b1 = compute_class_conditional_mls_scores(
        -known_logits.max(axis=1),
        known_pred,
        reference_nonconformity=full_calibration_nonconformity,
        reference_true_labels=full_calibration_labels,
        reference_predicted_labels=full_calibration_pred,
        query_pair_ids=known_pair_ids,
        reference_pair_ids=full_calibration_pair_ids,
        leave_one_out=True,
    )
    unknown_b1 = compute_class_conditional_mls_scores(
        -unknown_logits.max(axis=1),
        unknown_pred,
        reference_nonconformity=full_calibration_nonconformity,
        reference_true_labels=full_calibration_labels,
        reference_predicted_labels=full_calibration_pred,
        leave_one_out=False,
    )
    known_scores = compute_b0_b4_scores(known_logits, known_a, known_b1)
    unknown_scores = compute_b0_b4_scores(unknown_logits, unknown_a, unknown_b1)
    metrics: dict[str, dict[str, float]] = {}
    for rule in SCORE_RULES:
        values = evaluate_open_set(
            known_true=known_labels,
            known_pred=known_pred,
            known_unknown_scores=known_scores[rule],
            unknown_pred=unknown_pred,
            unknown_unknown_scores=unknown_scores[rule],
            known_validation_scores=known_scores[rule],
            known_class_count=known_logits.shape[1],
            known_acceptance_rate=acceptance_rate,
        )
        if any(key not in values for key in REPORT_METRIC_KEYS):
            raise DataValidationError("open-set evaluator omitted a frozen metric")
        metrics[rule] = {key: float(value) for key, value in values.items()}
    expected_accuracy = metrics[SCORE_RULES[0]]["known_accuracy"]
    expected_macro_f1 = metrics[SCORE_RULES[0]]["known_macro_f1"]
    if any(
        not math.isclose(values["known_accuracy"], expected_accuracy, rel_tol=0.0, abs_tol=0.0)
        or not math.isclose(values["known_macro_f1"], expected_macro_f1, rel_tol=0.0, abs_tol=0.0)
        for values in metrics.values()
    ):
        raise DataValidationError("B0-B4 changed the frozen known prediction")
    # The argument is retained and audited to show smoke/full queries are rows
    # from the same full calibration manifest rather than a rebuilt split.
    if known_calibration_indices.shape != (known_labels.shape[0],):
        raise DataValidationError("known calibration subset indices do not align")
    return metrics, known_scores, unknown_scores


def _build_prediction_rows(
    *,
    prepared: PreparedSurrogateSplit,
    role_indices: Mapping[str, np.ndarray],
    role_logits: Mapping[str, np.ndarray],
    role_scores: Mapping[str, Mapping[str, np.ndarray]],
    metrics: Mapping[str, Mapping[str, float]],
    reference_metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    score_by_sample = reference_metadata["score_by_sample"]
    rho_by_sample = reference_metadata["rho_by_sample"]
    p_by_sample = reference_metadata["p_by_sample"]
    reconstruction_error_by_sample = reference_metadata[
        "reconstruction_error_by_sample"
    ]
    rows: list[dict[str, Any]] = []
    for role in ("known_calibration", "surrogate_unknown"):
        full_rows = _role_manifest_rows(prepared, role)
        indices = role_indices[role]
        logits = role_logits[role]
        scores = role_scores[role]
        predictions = scores["known_prediction"]
        k_common = scores["k_common"]
        for local_index, source_index in enumerate(indices):
            source = full_rows[int(source_index)]
            view1_id = str(source["view1_sample_id"])
            view2_id = str(source["view2_sample_id"])
            predicted = int(predictions[local_index])
            true_label = int(prepared.labels[role][int(source_index)])
            row: dict[str, Any] = {
                "pair_id": str(source["pair_id"]),
                "evaluation_role": role,
                "class_name": str(source["class_name"]),
                "true_label": true_label,
                "predicted_known_label": predicted,
                "predicted_known_class_name": prepared.train_class_order[predicted],
                "k_common": int(k_common[local_index]),
                "k_common_class_name": prepared.train_class_order[int(k_common[local_index])],
                "r2_kcommon_agree": predicted == int(k_common[local_index]),
                "r2_fused_logits": json.dumps(logits[local_index].tolist(), separators=(",", ":")),
                "view1_sample_id": view1_id,
                "view2_sample_id": view2_id,
                "view1_angle_deg": int(source["view1_angle_deg"]),
                "view2_angle_deg": int(source["view2_angle_deg"]),
                "view1_frame_id": int(source["view1_frame_id"]),
                "view2_frame_id": int(source["view2_frame_id"]),
                "view1_rho": json.dumps(rho_by_sample[view1_id].tolist(), separators=(",", ":")),
                "view2_rho": json.dumps(rho_by_sample[view2_id].tolist(), separators=(",", ":")),
                "view1_p_value": json.dumps(p_by_sample[view1_id].tolist(), separators=(",", ":")),
                "view2_p_value": json.dumps(p_by_sample[view2_id].tolist(), separators=(",", ":")),
                "view1_a": json.dumps(score_by_sample[view1_id].tolist(), separators=(",", ":")),
                "view2_a": json.dumps(score_by_sample[view2_id].tolist(), separators=(",", ":")),
                "view1_mean_l1_reconstruction_error": json.dumps(
                    reconstruction_error_by_sample[view1_id].tolist(),
                    separators=(",", ":"),
                ),
                "view2_mean_l1_reconstruction_error": json.dumps(
                    reconstruction_error_by_sample[view2_id].tolist(),
                    separators=(",", ":"),
                ),
            }
            for rule in SCORE_RULES:
                field = rule.lower()
                score = float(scores[rule][local_index])
                threshold = float(metrics[rule]["threshold"])
                row[f"{field}_unknown_score"] = score
                row[f"{field}_threshold"] = threshold
                row[f"{field}_rejected"] = score > threshold
            rows.append(row)
    return rows


def recompute_metrics_from_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    known_class_count: int = 5,
    known_acceptance_rate: float = 0.95,
) -> dict[str, dict[str, float]]:
    known = [row for row in rows if str(row["evaluation_role"]) == "known_calibration"]
    unknown = [row for row in rows if str(row["evaluation_role"]) == "surrogate_unknown"]
    if not known or not unknown:
        raise DataValidationError("prediction rows lack known or surrogate samples")
    known_true = np.asarray([int(row["true_label"]) for row in known], dtype=np.int64)
    known_pred = np.asarray([int(row["predicted_known_label"]) for row in known], dtype=np.int64)
    unknown_pred = np.asarray([int(row["predicted_known_label"]) for row in unknown], dtype=np.int64)
    result: dict[str, dict[str, float]] = {}
    for rule in SCORE_RULES:
        field = f"{rule.lower()}_unknown_score"
        known_scores = np.asarray([float(row[field]) for row in known], dtype=np.float64)
        unknown_scores = np.asarray([float(row[field]) for row in unknown], dtype=np.float64)
        result[rule] = {
            key: float(value)
            for key, value in evaluate_open_set(
                known_true=known_true,
                known_pred=known_pred,
                known_unknown_scores=known_scores,
                unknown_pred=unknown_pred,
                unknown_unknown_scores=unknown_scores,
                known_validation_scores=known_scores,
                known_class_count=known_class_count,
                known_acceptance_rate=known_acceptance_rate,
            ).items()
        }
    return result


def _assert_metrics_equal(
    expected: Mapping[str, Mapping[str, float]],
    observed: Mapping[str, Mapping[str, float]],
) -> None:
    for rule in SCORE_RULES:
        if set(expected[rule]) != set(observed[rule]):
            raise DataValidationError(f"metric keys changed for {rule}")
        for key in expected[rule]:
            if not math.isclose(
                float(expected[rule][key]),
                float(observed[rule][key]),
                rel_tol=0.0,
                abs_tol=1.0e-15,
            ):
                raise DataValidationError(f"prediction rows do not reproduce {rule}.{key}")


def build_error_analysis(
    rows: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Mapping[str, float]],
    *,
    train_class_order: Sequence[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    known = [row for row in rows if row["evaluation_role"] == "known_calibration"]
    unknown = [row for row in rows if row["evaluation_role"] == "surrogate_unknown"]
    known_true = np.asarray([int(row["true_label"]) for row in known], dtype=np.int64)
    known_pred = np.asarray([int(row["predicted_known_label"]) for row in known], dtype=np.int64)
    unknown_identities = tuple(dict.fromkeys(str(row["class_name"]) for row in unknown))
    per_identity: dict[str, dict[str, dict[str, float]]] = {}
    for identity in unknown_identities:
        selected = [row for row in unknown if str(row["class_name"]) == identity]
        identity_pred = np.asarray(
            [int(row["predicted_known_label"]) for row in selected], dtype=np.int64
        )
        per_identity[identity] = {}
        for rule in SCORE_RULES:
            field = f"{rule.lower()}_unknown_score"
            known_scores = np.asarray([float(row[field]) for row in known])
            identity_scores = np.asarray([float(row[field]) for row in selected])
            values = evaluate_open_set(
                known_true=known_true,
                known_pred=known_pred,
                known_unknown_scores=known_scores,
                unknown_pred=identity_pred,
                unknown_unknown_scores=identity_scores,
                known_validation_scores=known_scores,
                known_class_count=len(train_class_order),
                known_acceptance_rate=0.95,
            )
            per_identity[identity][rule] = {
                key: float(values[key])
                for key in (
                    "auroc",
                    "unknown_rejection_rate",
                    "fpr95",
                    "oscr",
                    "threshold",
                )
            }

    absorption_rows: list[dict[str, Any]] = []
    for identity in unknown_identities:
        identity_rows = [row for row in unknown if str(row["class_name"]) == identity]
        for rule in SCORE_RULES:
            reject_field = f"{rule.lower()}_rejected"
            accepted = [row for row in identity_rows if not bool(row[reject_field])]
            counts = Counter(str(row["predicted_known_class_name"]) for row in accepted)
            for known_class in train_class_order:
                count = int(counts.get(str(known_class), 0))
                absorption_rows.append(
                    {
                        "surrogate_identity": identity,
                        "method": rule,
                        "absorbed_as_known_identity": str(known_class),
                        "false_accept_count": count,
                        "total_surrogate_count": len(identity_rows),
                        "total_false_accept_count": len(accepted),
                        "rate_over_all_surrogate": count / len(identity_rows),
                        "composition_within_false_accepts": (
                            0.0 if not accepted else count / len(accepted)
                        ),
                    }
                )

    angle_rows: list[dict[str, Any]] = []
    for row in rows:
        predicted = int(row["predicted_known_label"])
        common = int(row["k_common"])
        for view in (1, 2):
            rho = np.asarray(json.loads(str(row[f"view{view}_rho"])), dtype=np.float64)
            reconstruction_error = np.asarray(
                json.loads(str(row[f"view{view}_mean_l1_reconstruction_error"])),
                dtype=np.float64,
            )
            angle_rows.append(
                {
                    "pair_id": str(row["pair_id"]),
                    "evaluation_role": str(row["evaluation_role"]),
                    "class_name": str(row["class_name"]),
                    "sample_id": str(row[f"view{view}_sample_id"]),
                    "view_slot": view,
                    "angle_deg": int(row[f"view{view}_angle_deg"]),
                    "frame_id": int(row[f"view{view}_frame_id"]),
                    "rho_min_over_class": float(rho.min()),
                    "rho_r2_predicted_class": float(rho[predicted]),
                    "rho_common_class": float(rho[common]),
                    "mean_l1_error_min_over_class": float(
                        reconstruction_error.min()
                    ),
                    "mean_l1_error_r2_predicted_class": float(
                        reconstruction_error[predicted]
                    ),
                    "mean_l1_error_common_class": float(
                        reconstruction_error[common]
                    ),
                    "r2_predicted_class_name": str(row["predicted_known_class_name"]),
                    "common_class_name": str(row["k_common_class_name"]),
                }
            )

    known_correct = [
        row
        for row in known
        if int(row["true_label"]) == int(row["predicted_known_label"])
    ]
    r2_kcommon_mismatch = [row for row in rows if not bool(row["r2_kcommon_agree"])]
    mutual = {
        "DDG-1000_absorbed_as_DDG-112": {},
        "DDG-112_absorbed_as_DDG-1000": {},
    }
    for rule in SCORE_RULES:
        reject_field = f"{rule.lower()}_rejected"
        mutual["DDG-1000_absorbed_as_DDG-112"][rule] = sum(
            str(row["class_name"]) == "DDG-1000"
            and str(row["predicted_known_class_name"]) == "DDG-112"
            and not bool(row[reject_field])
            for row in unknown
        )
        mutual["DDG-112_absorbed_as_DDG-1000"][rule] = sum(
            str(row["class_name"]) == "DDG-112"
            and str(row["predicted_known_class_name"]) == "DDG-1000"
            and not bool(row[reject_field])
            for row in unknown
        )
    summary = {
        "per_surrogate_identity": per_identity,
        "known_r2_correct_count": len(known_correct),
        "known_r2_correct_but_cssr_rejected": {
            rule: sum(bool(row[f"{rule.lower()}_rejected"]) for row in known_correct)
            for rule in SCORE_RULES[2:]
        },
        "r2_b0_accepted_unknown_but_cssr_rejected": {
            rule: sum(
                not bool(row[f"{SCORE_RULES[0].lower()}_rejected"])
                and bool(row[f"{rule.lower()}_rejected"])
                for row in unknown
            )
            for rule in SCORE_RULES[2:]
        },
        "r2_kcommon_mismatch_count": len(r2_kcommon_mismatch),
        "known_mismatch_r2_correct_count": sum(
            row["evaluation_role"] == "known_calibration"
            and int(row["predicted_known_label"]) == int(row["true_label"])
            for row in r2_kcommon_mismatch
        ),
        "known_mismatch_kcommon_correct_count": sum(
            row["evaluation_role"] == "known_calibration"
            and int(row["k_common"]) == int(row["true_label"])
            for row in r2_kcommon_mismatch
        ),
        "ddg_mutual_absorption": mutual,
        "angle_metadata_used_by_model": False,
        "angle_analysis_post_hoc_only": True,
        "overall_metrics": metrics,
    }
    return summary, absorption_rows, angle_rows


def _configure_numerical_runtime(config: Mapping[str, Any]) -> dict[str, Any]:
    runtime = config["runtime"]
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise DataValidationError("CUBLAS_WORKSPACE_CONFIG changed")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(bool(runtime["deterministic_algorithms"]))
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "tf32": False,
        "cudnn_benchmark": False,
    }


def _configure_runtime(config: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    numerical_runtime = _configure_numerical_runtime(config)
    runtime = config["runtime"]
    if device.type != "cuda":
        raise DataValidationError("smoke and formal CSSR units require CUDA")
    observed_gpu = torch.cuda.get_device_name(device)
    if observed_gpu != str(runtime["expected_gpu_model"]):
        raise DataValidationError(
            f"CSSR unit requires {runtime['expected_gpu_model']}; observed {observed_gpu}"
        )
    return {
        "device": str(device),
        "device_name": observed_gpu,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "amp": False,
        "torch_compile": False,
        "num_workers": int(runtime["num_workers"]),
        **numerical_runtime,
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


def _unit_destination(root: Path, pair_id: str) -> Path:
    return root / pair_id / "fold_0" / f"seed_{CSSR_SEED}" / "PCSSR_CORE_1D"


def _read_authorized_pilot(
    pilot_root: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(pilot_root).resolve()
    audit_phase_root(root, config=config, phase="pilot", pilot_root=None)
    gate_path = root / "pilot_gate.json"
    gate = _read_json(gate_path)
    if (
        gate.get("confirmation_allowed") is not True
        or gate.get("selected_rule") not in SCORE_RULES[3:]
        or gate.get("signal") == "no_cssr_signal"
    ):
        raise DataValidationError("pilot gate does not authorize confirmation")
    return {
        "pilot_root": str(root),
        "pilot_gate_sha256": file_sha256(gate_path),
        "selected_rule": str(gate["selected_rule"]),
        "signal": str(gate["signal"]),
    }


def save_unit_result(
    destination: Path,
    *,
    phase: str,
    pair_id: str,
    smoke: bool,
    config: Mapping[str, Any],
    prepared: PreparedSurrogateSplit,
    unique_rows: Sequence[Mapping[str, Any]],
    feature_maps: np.ndarray,
    feature_audit: Mapping[str, Any],
    cssr_model: PCSSRCore1D,
    training_log: Sequence[Mapping[str, Any]],
    training_audit: Mapping[str, Any],
    r2_audit: Mapping[str, Any],
    reference_arrays: Mapping[str, np.ndarray],
    reference_values: Sequence[np.ndarray],
    reference_ids: Sequence[Sequence[str]],
    reference_metadata: Mapping[str, Any],
    prediction_rows: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Mapping[str, float]],
    error_analysis: Mapping[str, Any],
    absorption_rows: Sequence[Mapping[str, Any]],
    angle_rows: Sequence[Mapping[str, Any]],
    role_indices: Mapping[str, np.ndarray],
    runtime_contract: Mapping[str, Any],
    environment: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    confirmation_authorization: Mapping[str, Any] | None,
    wall_time_seconds: float,
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=False)
    _atomic_write_bytes(destination / "source_pair_manifest.csv", prepared.pair_manifest_bytes)
    _write_csv(destination / "unique_base_sample_manifest.csv", unique_rows)
    subset_manifest_rows: list[dict[str, Any]] = []
    for role in ("known_calibration", "surrogate_unknown"):
        full = _role_manifest_rows(prepared, role)
        subset_manifest_rows.extend(
            {
                **full[int(index)],
                "evaluation_subset_role": role,
                "evaluation_subset_index": local_index,
            }
            for local_index, index in enumerate(role_indices[role])
        )
    _write_csv(destination / "evaluation_pair_manifest.csv", subset_manifest_rows)

    feature_buffer = io.BytesIO()
    np.savez_compressed(
        feature_buffer,
        feature_maps=feature_maps,
        sample_ids=np.asarray([row["sample_id"] for row in unique_rows], dtype=np.str_),
    )
    _atomic_write_bytes(destination / "feature_maps.npz", feature_buffer.getvalue())
    reference_buffer = io.BytesIO()
    np.savez_compressed(
        reference_buffer,
        **reference_arrays,
        **{
            f"class_{index}_reference_rho": values
            for index, values in enumerate(reference_values)
        },
    )
    _atomic_write_bytes(
        destination / "base_reconstruction_scores.npz", reference_buffer.getvalue()
    )
    reference_serializable = {
        key: value
        for key, value in reference_metadata.items()
        if key
        not in {
            "score_by_sample",
            "rho_by_sample",
            "p_by_sample",
            "reconstruction_error_by_sample",
        }
    }
    reference_serializable["reference_sample_ids"] = [list(values) for values in reference_ids]
    _write_json(destination / "reference_distribution.json", reference_serializable)

    checkpoint = {
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "pair_id": pair_id,
        "architecture": cssr_model.architecture_id,
        "model_state_dict": {
            name: value.detach().cpu().clone()
            for name, value in cssr_model.state_dict().items()
        },
        "checkpoint_epoch": int(training_audit["epochs"]),
        "formal_checkpoint": not smoke,
        "checkpoint_selection": "fixed_final_epoch",
        "cssr_initialization_seed": CSSR_SEED,
        "train_class_order": prepared.train_class_order,
        "source_pair_manifest_sha256": prepared.pair_manifest_sha256,
        "unique_base_manifest_sha256": hashlib.sha256(_render_csv(unique_rows)).hexdigest(),
        "feature_map_sha256": feature_audit["feature_map_sha256"],
        "config_sha256": config["_config_sha256"],
        "r2_checkpoint_sha256": r2_audit["checkpoint_sha256"],
        "r2_frozen": True,
        "surrogate_unknown_used_for_training": False,
        "final_unknown_used": False,
        "even_angle_test_used": False,
    }
    checkpoint_buffer = io.BytesIO()
    torch.save(checkpoint, checkpoint_buffer)
    _atomic_write_bytes(destination / "cssr_checkpoint.pt", checkpoint_buffer.getvalue())
    _atomic_write_bytes(
        destination / "training_log.jsonl",
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in training_log
        ).encode("utf-8"),
    )
    _write_json(destination / "training_audit.json", dict(training_audit))
    _write_json(destination / "r2_reference_audit.json", dict(r2_audit))
    _write_json(destination / "feature_map_audit.json", dict(feature_audit))
    _write_csv(destination / "predictions_and_scores.csv", prediction_rows)
    _write_json(destination / "metrics.json", metrics)
    _write_json(destination / "error_analysis.json", error_analysis)
    _write_csv(destination / "absorption_by_known_class.csv", absorption_rows)
    _write_csv(destination / "angle_reconstruction_diagnostic.csv", angle_rows)
    _write_json(destination / "environment.json", environment)

    resolved = dict(config)
    resolved["_resolved"] = {
        "phase": phase,
        "mode": "smoke" if smoke else "full",
        "pair_id": pair_id,
        "angle_fold": ANGLE_FOLD,
        "r2_seed": R2_SEED,
        "cssr_seed": CSSR_SEED,
        "r2_checkpoint_path": r2_audit["checkpoint_path"],
        "r2_checkpoint_sha256": r2_audit["checkpoint_sha256"],
        "source_pair_manifest_sha256": prepared.pair_manifest_sha256,
        "evaluation_pair_manifest_sha256": hashlib.sha256(_render_csv(subset_manifest_rows)).hexdigest(),
        "train_class_order": list(prepared.train_class_order),
        "surrogate_class_order": list(prepared.surrogate_class_order),
        "cssr_epoch": int(training_audit["epochs"]),
        "confirmation_authorization": confirmation_authorization,
        "final_unknown_used": False,
        "even_angle_test_used": False,
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
        "angle_fold": ANGLE_FOLD,
        "r2_seed": R2_SEED,
        "cssr_seed": CSSR_SEED,
        "score_rules": list(SCORE_RULES),
        "known_prediction_source": "frozen_r2_fused_ce_for_all_rules",
        "threshold_source": "known_calibration_only",
        "config_sha256": config["_config_sha256"],
        "source_hashes": source_hashes,
        "runtime_contract": runtime_contract,
        "confirmation_authorization": confirmation_authorization,
        "r2_retrained_or_finetuned": False,
        "arpl_used": False,
        "pseudo_unknown_used": False,
        "surrogate_unknown_used_for_cssr_training": False,
        "surrogate_unknown_used_for_reference_distribution": False,
        "surrogate_unknown_used_for_threshold": False,
        "known_calibration_used_for_cssr_training": False,
        "final_unknown_used": False,
        "even_angle_test_used": False,
        "test_pairs_generated": False,
        "test_features_materialized": False,
    }
    _write_json(destination / "unit_contract.json", unit_contract)
    summary = {
        "status": "complete",
        "phase": phase,
        "mode": "smoke" if smoke else "full",
        "pair_id": pair_id,
        "metrics": metrics,
        "r2_checkpoint_sha256": r2_audit["checkpoint_sha256"],
        "source_pair_manifest_sha256": prepared.pair_manifest_sha256,
        "unique_train_base_sample_count": int(training_audit["train_unique_sample_count"]),
        "cssr_checkpoint_epoch": int(training_audit["epochs"]),
        "wall_time_seconds": wall_time_seconds,
        "final_unknown_used": False,
        "even_angle_test_used": False,
    }
    _write_json(destination / "unit_summary.json", summary)
    hashes = _artifact_hashes(destination)
    _write_json(destination / "artifact_hashes.json", hashes)
    _write_json(
        destination / "_SUCCESS.json",
        {
            "status": "complete",
            "unit_summary_sha256": file_sha256(destination / "unit_summary.json"),
            "artifact_hashes_sha256": file_sha256(destination / "artifact_hashes.json"),
        },
    )
    return summary


def run_unit(
    config_path: str | Path,
    bundle_root: str | Path,
    r2_results_root: str | Path,
    phase_root: str | Path,
    *,
    phase: str,
    pair_id: str,
    device_request: str = "auto",
    pilot_root: str | Path | None = None,
) -> dict[str, Any]:
    config = load_fg_mv_cssr_config(config_path)
    plan = build_phase_plan(config, phase)
    if pair_id not in {str(unit["pair_id"]) for unit in plan}:
        raise DataValidationError(f"pair {pair_id} is outside the frozen {phase} plan")
    confirmation_authorization = None
    if phase == "confirmation":
        if pilot_root is None:
            raise DataValidationError("confirmation requires an audited pilot root")
        confirmation_authorization = _read_authorized_pilot(pilot_root, config)
    elif pilot_root is not None:
        raise DataValidationError("pilot root is only valid for confirmation")

    root = Path(phase_root).resolve()
    destination = _unit_destination(root, pair_id)
    if destination.exists():
        raise DataValidationError(f"CSSR output already exists: {destination}")
    staging = destination.parent / ".PCSSR_CORE_1D.staging"
    if staging.exists():
        raise DataValidationError(f"stale CSSR staging output exists: {staging}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    project_root = Path(config["_config_path"]).parents[3]
    source_hashes = task_source_hashes(project_root)
    device = _resolve_device(device_request)
    runtime_contract = _configure_runtime(config, device)
    started = time.perf_counter()
    prior_config = _load_prior_config(project_root, config)
    bundle = _load_bundle(bundle_root, config)
    prepared = _prepare_frozen_split(
        bundle, prior_config, config, pair_id
    )
    model, r2_arrays, r2_audit = load_and_audit_frozen_r2(
        project_root=project_root,
        r2_results_root=r2_results_root,
        pair_id=pair_id,
        config=config,
        prepared=prepared,
        prior_config=prior_config,
        device=device,
    )
    unique_rows = build_unique_base_sample_manifest(prepared, bundle)
    unique_manifest_sha = hashlib.sha256(_render_csv(unique_rows)).hexdigest()
    feature_maps, feature_audit = extract_frozen_feature_maps(
        model=model,
        bundle=bundle,
        prepared=prepared,
        rows=unique_rows,
        device=device,
        batch_size=int(config["cssr_training"]["batch_size"]),
    )
    feature_audit = {
        **feature_audit,
        "unique_base_manifest_sha256": unique_manifest_sha,
        "role_counts": dict(Counter(str(row["experiment_role"]) for row in unique_rows)),
        "pair_multiplicity_used": False,
    }

    train_indices_all = [
        index
        for index, row in enumerate(unique_rows)
        if row["experiment_role"] == "train_known"
    ]
    if phase == "smoke":
        per_class = int(config["data"]["smoke"]["unique_train_base_samples_per_class"])
        train_indices = []
        for class_index in range(5):
            selected = [
                index
                for index in train_indices_all
                if int(unique_rows[index]["model_label"]) == class_index
            ][:per_class]
            if len(selected) != per_class:
                raise DataValidationError("insufficient unique train bases for smoke")
            train_indices.extend(selected)
    else:
        train_indices = train_indices_all
    calibration_unique_indices = [
        index
        for index, row in enumerate(unique_rows)
        if row["experiment_role"] == "known_calibration"
    ]
    train_labels = np.asarray(
        [int(unique_rows[index]["model_label"]) for index in train_indices],
        dtype=np.int64,
    )
    calibration_labels = np.asarray(
        [int(unique_rows[index]["model_label"]) for index in calibration_unique_indices],
        dtype=np.int64,
    )
    cssr_model, training_log, training_audit = train_pcssr_core(
        train_features=feature_maps[np.asarray(train_indices, dtype=np.int64)],
        train_labels=train_labels,
        calibration_features=feature_maps[
            np.asarray(calibration_unique_indices, dtype=np.int64)
        ],
        calibration_labels=calibration_labels,
        config=config,
        device=device,
        smoke=phase == "smoke",
    )
    training_audit = {
        **training_audit,
        "train_sample_id_order_sha256": _sequence_sha256(
            unique_rows[index]["sample_id"] for index in train_indices
        ),
        "full_unique_train_population_count": len(train_indices_all),
        "smoke_subset": phase == "smoke",
        "r2_parameters_still_frozen": not any(
            parameter.requires_grad for parameter in model.parameters()
        ),
    }
    rho, reconstruction_error, rho_audit = infer_cssr_rho(
        cssr_model,
        feature_maps,
        device=device,
        batch_size=int(config["cssr_training"]["batch_size"]),
    )
    feature_audit = {**feature_audit, "rho_audit": rho_audit}
    reference_arrays, reference_values, reference_ids, reference_metadata = (
        build_cssr_reference_scores(
            unique_rows=unique_rows,
            rho=rho,
            epsilon=float(config["calibration"]["score_epsilon"]),
        )
    )
    reference_arrays = {
        **reference_arrays,
        "mean_l1_reconstruction_error": reconstruction_error,
        "mls_calibration_logits": r2_arrays["known_calibration"]["global_logits"],
        "mls_calibration_labels": prepared.labels["known_calibration"],
        "mls_calibration_pair_ids": np.asarray(
            prepared.pair_ids["known_calibration"], dtype=np.str_
        ),
    }
    reference_metadata["reconstruction_error_by_sample"] = {
        str(row["sample_id"]): reconstruction_error[index]
        for index, row in enumerate(unique_rows)
    }

    if phase == "smoke":
        evaluation_pairs_per_class = int(
            config["data"]["smoke"]["evaluation_pairs_per_class"]
        )
        role_indices = {
            role: _smoke_pair_indices(prepared, role, evaluation_pairs_per_class)
            for role in ("known_calibration", "surrogate_unknown")
        }
    else:
        role_indices = {
            role: np.arange(len(prepared.labels[role]), dtype=np.int64)
            for role in ("known_calibration", "surrogate_unknown")
        }
    role_logits = {
        role: r2_arrays[role]["global_logits"][indices]
        for role, indices in role_indices.items()
    }
    role_pair_rows = {
        role: [
            _role_manifest_rows(prepared, role)[int(index)] for index in indices
        ]
        for role, indices in role_indices.items()
    }
    role_a = {
        role: _pair_a_values(rows, reference_metadata["score_by_sample"])
        for role, rows in role_pair_rows.items()
    }
    metrics, known_scores, unknown_scores = _evaluate_all_scores(
        known_logits=role_logits["known_calibration"],
        known_labels=prepared.labels["known_calibration"][role_indices["known_calibration"]],
        known_pair_ids=[row["pair_id"] for row in role_pair_rows["known_calibration"]],
        unknown_logits=role_logits["surrogate_unknown"],
        full_calibration_logits=r2_arrays["known_calibration"]["global_logits"],
        full_calibration_labels=prepared.labels["known_calibration"],
        full_calibration_pair_ids=prepared.pair_ids["known_calibration"],
        known_a=role_a["known_calibration"],
        unknown_a=role_a["surrogate_unknown"],
        acceptance_rate=float(config["calibration"]["threshold_known_acceptance_rate"]),
        known_calibration_indices=role_indices["known_calibration"],
    )
    if phase != "smoke":
        prior_metrics = _read_json(Path(r2_audit["unit_root"]) / "metrics.json")
        for key, expected in prior_metrics.items():
            if not math.isclose(
                float(metrics[SCORE_RULES[0]][key]),
                float(expected),
                rel_tol=0.0,
                abs_tol=1.0e-15,
            ):
                raise DataValidationError(f"B0 no longer reproduces prior R2 metric {key}")
        r2_audit["b0_metrics_exact_regression"] = True
    else:
        r2_audit["b0_metrics_exact_regression"] = "not_applicable_smoke_subset"

    prediction_rows = _build_prediction_rows(
        prepared=prepared,
        role_indices=role_indices,
        role_logits=role_logits,
        role_scores={
            "known_calibration": known_scores,
            "surrogate_unknown": unknown_scores,
        },
        metrics=metrics,
        reference_metadata=reference_metadata,
    )
    recomputed = recompute_metrics_from_prediction_rows(
        prediction_rows,
        known_class_count=5,
        known_acceptance_rate=float(config["calibration"]["threshold_known_acceptance_rate"]),
    )
    _assert_metrics_equal(metrics, recomputed)
    error_analysis, absorption_rows, angle_rows = build_error_analysis(
        prediction_rows, metrics, train_class_order=prepared.train_class_order
    )
    if task_source_hashes(project_root) != source_hashes:
        raise DataValidationError("task source changed while CSSR unit was running")
    environment = _git_environment(project_root, device)
    environment["runtime_contract"] = runtime_contract
    environment["task_source_hashes"] = source_hashes
    environment["official_reference_execution"] = (
        "device_neutral_semantic_oracle_differential_not_direct_official_import"
    )
    summary = save_unit_result(
        staging,
        phase=phase,
        pair_id=pair_id,
        smoke=phase == "smoke",
        config=config,
        prepared=prepared,
        unique_rows=unique_rows,
        feature_maps=feature_maps,
        feature_audit=feature_audit,
        cssr_model=cssr_model,
        training_log=training_log,
        training_audit=training_audit,
        r2_audit=r2_audit,
        reference_arrays=reference_arrays,
        reference_values=reference_values,
        reference_ids=reference_ids,
        reference_metadata=reference_metadata,
        prediction_rows=prediction_rows,
        metrics=metrics,
        error_analysis=error_analysis,
        absorption_rows=absorption_rows,
        angle_rows=angle_rows,
        role_indices=role_indices,
        runtime_contract=runtime_contract,
        environment=environment,
        source_hashes=source_hashes,
        confirmation_authorization=confirmation_authorization,
        wall_time_seconds=time.perf_counter() - started,
    )
    staging.replace(destination)
    return summary


def _csv_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise DataValidationError(f"invalid CSV boolean: {value!r}")


def audit_unit_result(
    destination: str | Path,
    *,
    config: Mapping[str, Any],
    phase: str,
    pair_id: str,
) -> dict[str, Any]:
    # ``aggregate`` and ``audit`` run in fresh processes.  Reapply the same
    # CUDA numerical contract used by ``run-unit`` before replaying Conv1d;
    # otherwise a container default such as cuDNN TF32 can change rho.
    _configure_numerical_runtime(config)
    root = Path(destination).resolve()
    required = {
        "source_pair_manifest.csv", "unique_base_sample_manifest.csv",
        "evaluation_pair_manifest.csv", "feature_maps.npz",
        "base_reconstruction_scores.npz", "reference_distribution.json",
        "cssr_checkpoint.pt", "training_log.jsonl", "training_audit.json",
        "r2_reference_audit.json", "feature_map_audit.json",
        "predictions_and_scores.csv", "metrics.json", "error_analysis.json",
        "absorption_by_known_class.csv", "angle_reconstruction_diagnostic.csv",
        "environment.json", "resolved_config.yaml", "unit_contract.json",
        "unit_summary.json", "artifact_hashes.json", "_SUCCESS.json",
    }
    observed_files = {path.name for path in root.iterdir() if path.is_file()}
    if not required <= observed_files:
        raise DataValidationError(
            f"CSSR unit is missing files: {sorted(required - observed_files)}"
        )
    success = _read_json(root / "_SUCCESS.json")
    if (
        success.get("status") != "complete"
        or success.get("unit_summary_sha256")
        != file_sha256(root / "unit_summary.json")
        or success.get("artifact_hashes_sha256")
        != file_sha256(root / "artifact_hashes.json")
    ):
        raise DataValidationError("CSSR unit success marker is invalid")
    recorded_hashes = _read_json(root / "artifact_hashes.json")
    if recorded_hashes != _artifact_hashes(root):
        raise DataValidationError("CSSR unit artifact hash audit failed")

    contract = _read_json(root / "unit_contract.json")
    expected_mode = "smoke" if phase == "smoke" else "full"
    expected_contract = {
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "mode": expected_mode,
        "pair_id": pair_id,
        "angle_fold": ANGLE_FOLD,
        "r2_seed": R2_SEED,
        "cssr_seed": CSSR_SEED,
        "config_sha256": config["_config_sha256"],
        "r2_retrained_or_finetuned": False,
        "arpl_used": False,
        "pseudo_unknown_used": False,
        "surrogate_unknown_used_for_cssr_training": False,
        "surrogate_unknown_used_for_reference_distribution": False,
        "surrogate_unknown_used_for_threshold": False,
        "known_calibration_used_for_cssr_training": False,
        "final_unknown_used": False,
        "even_angle_test_used": False,
        "test_pairs_generated": False,
        "test_features_materialized": False,
    }
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            raise DataValidationError(f"CSSR unit contract changed at {key}")
    if list(contract.get("score_rules", [])) != list(SCORE_RULES):
        raise DataValidationError("CSSR unit score rules changed")
    r2_audit = _read_json(root / "r2_reference_audit.json")
    if (
        r2_audit.get("status") != "passed"
        or r2_audit.get("strict_load") is not True
        or r2_audit.get("all_parameters_frozen") is not True
        or r2_audit.get("arpl_module_instantiated") is not False
        or r2_audit.get("old_outputs_exact") is not True
    ):
        raise DataValidationError("frozen R2 audit is incomplete")
    if (
        file_sha256(root / "source_pair_manifest.csv")
        != r2_audit["pair_manifest_sha256"]
    ):
        raise DataValidationError("saved source pair manifest differs from R2")

    unique_rows = _read_csv(root / "unique_base_sample_manifest.csv")
    if len(unique_rows) != 972:
        raise DataValidationError("unique base manifest does not contain 972 samples")
    if len({row["sample_id"] for row in unique_rows}) != len(unique_rows):
        raise DataValidationError("unique base manifest repeats a sample ID")
    role_counts = Counter(row["experiment_role"] for row in unique_rows)
    expected_roles = Counter(
        {"train_known": 720, "known_calibration": 180, "surrogate_unknown": 72}
    )
    if role_counts != expected_roles:
        raise DataValidationError("unique base role counts changed")
    if any(int(row["angle_deg"]) % 2 == 0 for row in unique_rows):
        raise DataValidationError("even-angle base entered CSSR artifacts")

    feature_audit = _read_json(root / "feature_map_audit.json")
    with np.load(root / "feature_maps.npz", allow_pickle=False) as data:
        feature_maps = data["feature_maps"]
        sample_ids = data["sample_ids"]
    if feature_maps.shape != (972, 128, 76) or feature_maps.dtype != np.float32:
        raise DataValidationError("saved feature-map tensor shape/dtype changed")
    if _array_sha256(feature_maps) != feature_audit["feature_map_sha256"]:
        raise DataValidationError("saved feature-map hash changed")
    if tuple(sample_ids.tolist()) != tuple(row["sample_id"] for row in unique_rows):
        raise DataValidationError("feature-map sample order changed")

    training_audit = _read_json(root / "training_audit.json")
    expected_train_count = 10 if phase == "smoke" else 720
    expected_epochs = 1 if phase == "smoke" else 30
    if (
        training_audit.get("status") != "passed"
        or int(training_audit.get("train_unique_sample_count", -1))
        != expected_train_count
        or int(training_audit.get("epochs", -1)) != expected_epochs
        or training_audit.get("known_calibration_used_for_training") is not False
        or training_audit.get("surrogate_unknown_used_for_training") is not False
        or training_audit.get("pair_multiplicity_weight") is not False
        or training_audit.get("r2_parameters_still_frozen") is not True
    ):
        raise DataValidationError("CSSR training audit failed")

    checkpoint = torch.load(
        root / "cssr_checkpoint.pt", map_location="cpu", weights_only=False
    )
    if (
        checkpoint.get("experiment_id") != EXPERIMENT_ID
        or checkpoint.get("phase") != phase
        or checkpoint.get("pair_id") != pair_id
        or checkpoint.get("architecture") != "PCSSR_CORE_1D"
        or int(checkpoint.get("checkpoint_epoch", -1)) != expected_epochs
        or bool(checkpoint.get("formal_checkpoint")) != (phase != "smoke")
        or checkpoint.get("r2_frozen") is not True
        or checkpoint.get("final_unknown_used") is not False
        or checkpoint.get("even_angle_test_used") is not False
    ):
        raise DataValidationError("CSSR checkpoint metadata changed")
    core = config["pcssr_core_1d"]
    restored = PCSSRCore1D(
        num_classes=5,
        input_channels=int(core["input_channels"]),
        latent_channels=int(core["latent_channels"]),
        gamma=float(core["gamma"]),
        clip_length=abs(float(core["clip_min"])),
        epsilon=float(core["rho_epsilon"]),
    )
    incompatible = restored.load_state_dict(
        checkpoint["model_state_dict"], strict=True
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise DataValidationError("CSSR checkpoint is not strict-load compatible")
    if not all(torch.isfinite(parameter).all() for parameter in restored.parameters()):
        raise DataValidationError("CSSR checkpoint contains non-finite parameters")

    with np.load(root / "base_reconstruction_scores.npz", allow_pickle=False) as data:
        saved_base_arrays = {name: data[name] for name in data.files}
    saved_rho = np.asarray(saved_base_arrays["rho"], dtype=np.float64)
    saved_error = np.asarray(
        saved_base_arrays["mean_l1_reconstruction_error"], dtype=np.float64
    )
    audit_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    restored.to(audit_device).eval()
    recomputed_rho, recomputed_error, _ = infer_cssr_rho(
        restored,
        feature_maps,
        device=audit_device,
        batch_size=int(config["cssr_training"]["batch_size"]),
    )
    if not np.allclose(recomputed_rho, saved_rho, rtol=1.0e-5, atol=1.0e-6):
        raise DataValidationError("checkpoint/feature maps do not reproduce saved rho")
    if not np.allclose(
        recomputed_error, saved_error, rtol=1.0e-5, atol=1.0e-6
    ):
        raise DataValidationError(
            "checkpoint/feature maps do not reproduce reconstruction error"
        )
    (
        recomputed_reference_arrays,
        recomputed_reference_values,
        recomputed_reference_ids,
        recomputed_reference_metadata,
    ) = build_cssr_reference_scores(
        unique_rows=unique_rows,
        rho=saved_rho,
        epsilon=float(config["calibration"]["score_epsilon"]),
    )
    for name in (
        "known_calibration_p",
        "known_calibration_a",
        "surrogate_unknown_p",
        "surrogate_unknown_a",
    ):
        if not np.array_equal(recomputed_reference_arrays[name], saved_base_arrays[name]):
            raise DataValidationError(f"saved CSSR reference score does not reproduce: {name}")
    for index, values in enumerate(recomputed_reference_values):
        if not np.array_equal(
            values, saved_base_arrays[f"class_{index}_reference_rho"]
        ):
            raise DataValidationError("saved CSSR reference distribution changed")
    reference_metadata = _read_json(root / "reference_distribution.json")
    if reference_metadata.get("reference_sample_ids") != [
        list(values) for values in recomputed_reference_ids
    ]:
        raise DataValidationError("saved CSSR reference sample IDs changed")

    raw_rows = _read_csv(root / "predictions_and_scores.csv")
    expected_prediction_count = 14 if phase == "smoke" else 3500
    if len(raw_rows) != expected_prediction_count:
        raise DataValidationError("CSSR prediction row count changed")
    score_by_sample = recomputed_reference_metadata["score_by_sample"]
    rho_by_sample = recomputed_reference_metadata["rho_by_sample"]
    p_by_sample = recomputed_reference_metadata["p_by_sample"]
    error_by_sample = {
        str(row["sample_id"]): saved_error[index]
        for index, row in enumerate(unique_rows)
    }
    for row in raw_rows:
        for view in (1, 2):
            sample_id = str(row[f"view{view}_sample_id"])
            comparisons = (
                ("rho", rho_by_sample[sample_id]),
                ("p_value", p_by_sample[sample_id]),
                ("a", score_by_sample[sample_id]),
                ("mean_l1_reconstruction_error", error_by_sample[sample_id]),
            )
            for suffix, expected in comparisons:
                observed = np.asarray(
                    json.loads(row[f"view{view}_{suffix}"]), dtype=np.float64
                )
                if not np.array_equal(observed, expected):
                    raise DataValidationError(
                        f"prediction row view{view} {suffix} differs from base artifact"
                    )
    full_calibration_logits = np.asarray(
        saved_base_arrays["mls_calibration_logits"], dtype=np.float64
    )
    full_calibration_labels = np.asarray(
        saved_base_arrays["mls_calibration_labels"], dtype=np.int64
    )
    full_calibration_pair_ids = tuple(
        saved_base_arrays["mls_calibration_pair_ids"].tolist()
    )
    for role, leave_one_out in (
        ("known_calibration", True),
        ("surrogate_unknown", False),
    ):
        role_rows = [row for row in raw_rows if row["evaluation_role"] == role]
        query_logits = np.asarray(
            [json.loads(row["r2_fused_logits"]) for row in role_rows],
            dtype=np.float64,
        )
        recomputed_b1 = compute_class_conditional_mls_scores(
            -query_logits.max(axis=1),
            query_logits.argmax(axis=1),
            reference_nonconformity=-full_calibration_logits.max(axis=1),
            reference_true_labels=full_calibration_labels,
            reference_predicted_labels=full_calibration_logits.argmax(axis=1),
            query_pair_ids=(
                tuple(row["pair_id"] for row in role_rows)
                if leave_one_out
                else None
            ),
            reference_pair_ids=(
                full_calibration_pair_ids if leave_one_out else None
            ),
            leave_one_out=leave_one_out,
        )
        observed_b1 = np.asarray(
            [
                float(row[f"{SCORE_RULES[1].lower()}_unknown_score"])
                for row in role_rows
            ],
            dtype=np.float64,
        )
        if not np.array_equal(recomputed_b1, observed_b1):
            raise DataValidationError("B1 scores do not reproduce from saved references")
    metrics = _read_json(root / "metrics.json")
    recomputed = recompute_metrics_from_prediction_rows(
        raw_rows,
        known_class_count=5,
        known_acceptance_rate=float(
            config["calibration"]["threshold_known_acceptance_rate"]
        ),
    )
    _assert_metrics_equal(metrics, recomputed)
    for row in raw_rows:
        logits = np.asarray(json.loads(row["r2_fused_logits"]), dtype=np.float64)
        view1 = np.asarray(json.loads(row["view1_a"]), dtype=np.float64)
        view2 = np.asarray(json.loads(row["view2_a"]), dtype=np.float64)
        a_values = np.stack([view1, view2], axis=0)[None, ...]
        b1 = np.asarray(
            [float(row[f"{SCORE_RULES[1].lower()}_unknown_score"])],
            dtype=np.float64,
        )
        expected_scores = compute_b0_b4_scores(logits[None, :], a_values, b1)
        swapped_scores = compute_b0_b4_scores(
            logits[None, :], a_values[:, [1, 0]], b1
        )
        if int(expected_scores["known_prediction"][0]) != int(
            row["predicted_known_label"]
        ):
            raise DataValidationError("prediction row y_hat changed")
        if int(expected_scores["k_common"][0]) != int(row["k_common"]):
            raise DataValidationError("prediction row k_common changed")
        for rule in SCORE_RULES:
            observed = float(row[f"{rule.lower()}_unknown_score"])
            if not math.isclose(
                float(expected_scores[rule][0]),
                observed,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            ):
                raise DataValidationError(f"prediction row {rule} formula changed")
            if rule in SCORE_RULES[2:] and not math.isclose(
                float(swapped_scores[rule][0]),
                observed,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            ):
                raise DataValidationError(
                    f"prediction row {rule} is not view-swap invariant"
                )
        for rule in SCORE_RULES:
            score = float(row[f"{rule.lower()}_unknown_score"])
            threshold = float(row[f"{rule.lower()}_threshold"])
            if _csv_bool(row[f"{rule.lower()}_rejected"]) != (score > threshold):
                raise DataValidationError("prediction rejected flag is inconsistent")
    summary = _read_json(root / "unit_summary.json")
    if summary.get("status") != "complete" or summary.get("metrics") != metrics:
        raise DataValidationError("CSSR unit summary changed")
    return {
        "status": "passed",
        "phase": phase,
        "pair_id": pair_id,
        "destination": str(root),
        "artifact_count": len(recorded_hashes),
        "prediction_row_count": len(raw_rows),
        "metric_recomputation": "exact",
        "feature_map_hash_verified": True,
        "cssr_checkpoint_strict_load": True,
        "rho_checkpoint_recomputation": "within_float_tolerance",
        "reference_p_a_recomputation": "exact",
        "b1_reference_recomputation": "exact",
        "r2_frozen_output_regression": "exact",
        "view_swap_invariance": True,
        "final_unknown_used": False,
        "even_angle_test_used": False,
        "metrics": metrics,
    }


def _metric_rows_from_audits(
    audits: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for audit in audits:
        for rule in SCORE_RULES:
            rows.append(
                {
                    "pair_id": str(audit["pair_id"]),
                    "method": rule,
                    **{
                        key: float(audit["metrics"][rule][key])
                        for key in REPORT_METRIC_KEYS
                    },
                    "threshold": float(audit["metrics"][rule]["threshold"]),
                }
            )
    return rows


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
    if phase == "confirmation":
        if pilot_root is None:
            raise DataValidationError("confirmation aggregation requires pilot root")
        authorization = _read_authorized_pilot(pilot_root, config)
    elif pilot_root is not None:
        raise DataValidationError("pilot root only applies to confirmation")
    else:
        authorization = None
    plan = build_phase_plan(config, phase)
    audits = [
        audit_unit_result(
            _unit_destination(root, str(unit["pair_id"])),
            config=config,
            phase=phase,
            pair_id=str(unit["pair_id"]),
        )
        for unit in plan
    ]
    metric_rows = _metric_rows_from_audits(audits)
    _write_csv(root / "metrics_by_pair.csv", metric_rows)
    metrics_by_pair = {
        str(audit["pair_id"]): audit["metrics"] for audit in audits
    }
    _write_json(root / "metrics_by_pair.json", metrics_by_pair)
    gate_rows = [
        row
        for row in metric_rows
        if row["method"] in {SCORE_RULES[1], SCORE_RULES[3], SCORE_RULES[4]}
    ]
    if phase == "pilot":
        gate = evaluate_pilot_gate(gate_rows, config)
        _write_json(root / "pilot_gate.json", gate)
        decision = gate["signal"]
    elif phase == "confirmation":
        selected_rule = str(authorization["selected_rule"])
        gate = evaluate_confirmation_gate(gate_rows, selected_rule, config)
        _write_json(root / "confirmation_gate.json", gate)
        decision = gate["decision"]
    else:
        gate = None
        decision = "diagnostic_smoke_only"
    summary = {
        "status": "complete",
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "pair_ids": [str(unit["pair_id"]) for unit in plan],
        "unit_count": len(audits),
        "metrics": metrics_by_pair,
        "gate": gate,
        "decision": decision,
        "confirmation_authorization": authorization,
        "config_sha256": config["_config_sha256"],
        "final_unknown_used": False,
        "even_angle_test_used": False,
        "cssr_arpl_combination_used": False,
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
        or success.get("phase_summary_sha256")
        != file_sha256(root / "phase_summary.json")
        or success.get("artifact_hashes_sha256")
        != file_sha256(root / "artifact_hashes.json")
    ):
        raise DataValidationError("phase success marker is invalid")
    recorded_hashes = _read_json(root / "artifact_hashes.json")
    if recorded_hashes != _artifact_hashes(root):
        raise DataValidationError("phase artifact hash audit failed")
    plan = build_phase_plan(config, phase)
    audits = [
        audit_unit_result(
            _unit_destination(root, str(unit["pair_id"])),
            config=config,
            phase=phase,
            pair_id=str(unit["pair_id"]),
        )
        for unit in plan
    ]
    metric_rows = _metric_rows_from_audits(audits)
    gate_rows = [
        row
        for row in metric_rows
        if row["method"] in {SCORE_RULES[1], SCORE_RULES[3], SCORE_RULES[4]}
    ]
    summary = _read_json(root / "phase_summary.json")
    stored_metrics = _read_json(root / "metrics_by_pair.json")
    expected_metrics = {
        str(audit["pair_id"]): audit["metrics"] for audit in audits
    }
    if stored_metrics != expected_metrics or summary.get("metrics") != expected_metrics:
        raise DataValidationError("phase metrics do not match audited units")
    if phase == "pilot":
        expected_gate = evaluate_pilot_gate(gate_rows, config)
        if _read_json(root / "pilot_gate.json") != expected_gate:
            raise DataValidationError("pilot gate does not reproduce")
        decision = expected_gate["signal"]
    elif phase == "confirmation":
        if pilot_root is None:
            stored_authorization = summary.get("confirmation_authorization")
            if not isinstance(stored_authorization, Mapping):
                raise DataValidationError("confirmation lacks pilot authorization")
            pilot_root = str(stored_authorization["pilot_root"])
        authorization = _read_authorized_pilot(pilot_root, config)
        expected_gate = evaluate_confirmation_gate(
            gate_rows, str(authorization["selected_rule"]), config
        )
        if _read_json(root / "confirmation_gate.json") != expected_gate:
            raise DataValidationError("confirmation gate does not reproduce")
        decision = expected_gate["decision"]
    else:
        expected_gate = None
        decision = "diagnostic_smoke_only"
    if summary.get("decision") != decision:
        raise DataValidationError("phase decision does not reproduce")
    return {
        "status": "passed",
        "phase": phase,
        "root": str(root),
        "unit_count": len(audits),
        "artifact_count": len(recorded_hashes),
        "metric_recomputation": "exact",
        "gate_recomputation": (
            "exact" if expected_gate is not None else "not_applicable"
        ),
        "decision": decision,
        "gate": expected_gate,
        "final_unknown_used": False,
        "even_angle_test_used": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen R2 + per-view PCSSR_CORE_1D fast P3 experiment"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--config", default=CONFIG_RELATIVE_PATH)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--config", default=CONFIG_RELATIVE_PATH)
    plan.add_argument("--phase", choices=("smoke", "pilot", "confirmation"), required=True)

    run = subparsers.add_parser("run-unit")
    run.add_argument("--config", default=CONFIG_RELATIVE_PATH)
    run.add_argument("--bundle-root", required=True)
    run.add_argument("--r2-results-root", required=True)
    run.add_argument("--phase-root", required=True)
    run.add_argument("--phase", choices=("smoke", "pilot", "confirmation"), required=True)
    run.add_argument("--pair-id", required=True)
    run.add_argument("--device", default="auto")
    run.add_argument("--pilot-root")

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--config", default=CONFIG_RELATIVE_PATH)
    aggregate.add_argument("--phase-root", required=True)
    aggregate.add_argument("--phase", choices=("smoke", "pilot", "confirmation"), required=True)
    aggregate.add_argument("--pilot-root")

    audit = subparsers.add_parser("audit")
    audit.add_argument("--config", default=CONFIG_RELATIVE_PATH)
    audit.add_argument("--phase-root", required=True)
    audit.add_argument("--phase", choices=("smoke", "pilot", "confirmation"), required=True)
    audit.add_argument("--pilot-root")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = load_fg_mv_cssr_config(arguments.config)
    if arguments.command == "validate":
        result = {
            "status": "passed",
            "experiment_id": config["experiment_id"],
            "config_sha256": config["_config_sha256"],
            "final_unknown_test_authorized": False,
            "cssr_arpl_combination_authorized": False,
        }
    elif arguments.command == "plan":
        result = {
            "status": "planned",
            "phase": arguments.phase,
            "units": build_phase_plan(config, arguments.phase),
        }
    elif arguments.command == "run-unit":
        result = run_unit(
            arguments.config,
            arguments.bundle_root,
            arguments.r2_results_root,
            arguments.phase_root,
            phase=arguments.phase,
            pair_id=arguments.pair_id,
            device_request=arguments.device,
            pilot_root=arguments.pilot_root,
        )
    elif arguments.command == "aggregate":
        result = aggregate_phase_root(
            arguments.phase_root,
            config=config,
            phase=arguments.phase,
            pilot_root=arguments.pilot_root,
        )
    elif arguments.command == "audit":
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
