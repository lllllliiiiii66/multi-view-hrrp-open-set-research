from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import yaml
from torch import nn

from hrrp_osr.data.errors import DataConfigError, DataValidationError
from hrrp_osr.data.manifest import file_sha256
from hrrp_osr.evaluation.metrics import evaluate_open_set
from hrrp_osr.evaluation.ms_mean_factorial import REPORT_METRIC_KEYS
from hrrp_osr.models.cssr_decoupled_1d import (
    D1_DECOUPLED_REL_CSSR,
    D2_DECOUPLED_ABSREL_CSSR,
    DECOUPLED_METHODS,
    FGMVCSSRDecoupled1D,
)
from hrrp_osr.training.arpl_pilot import _resolve_device, _set_determinism
from hrrp_osr.training.cssr_gradient_pathology_audit import (
    audit_gradient_audit_phase,
)
from hrrp_osr.training.fg_mv_cssr_e2e_redesign import (
    _build_method_prediction_rows,
    _class_conditional_mls_for_roles,
    _evaluation_role_indices,
    _evaluate_score_arrays,
    _metrics_exact,
    _normalization_record,
    _save_npz,
    build_identity_and_absorption_rows,
    recompute_method_metrics_from_prediction_rows,
)
from hrrp_osr.training.fg_mv_cssr_pilot import (
    _artifact_hashes,
    _array_sha256,
    _atomic_write_bytes,
    _load_bundle,
    _load_prior_config,
    _prepare_frozen_split,
    _read_csv,
    _read_json,
    _render_csv,
    _role_manifest_rows,
    _sequence_sha256,
    _write_csv,
    _write_json,
    build_unique_base_sample_manifest,
    extract_frozen_feature_maps,
    load_and_audit_frozen_r2,
)
from hrrp_osr.training.fg_mv_cssr_decoupled_protocol import (
    CONFIRMATION_PAIRS,
    D0_R2_CLASS_CONDITIONAL_MLS,
    PILOT_PAIRS,
    build_guided_reference_scores,
    build_phase_plan,
    build_single_view_schedule,
    evaluate_confirmation_gate,
    evaluate_pilot_gate,
)


EXPERIMENT_ID = "fg_mv_cssr_decoupled_audit_v3"
CONFIG_RELATIVE_PATH = "configs/experiments/cssr/fg_mv_cssr_decoupled_audit_v3.yaml"
ANGLE_FOLD = 0
R2_SEED = 20260830
CSSR_SEED = 20260905
GLOBAL_MLS = "R2_GLOBAL_MLS"
GATE_TOLERANCE = 1.0e-12
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
    "src/hrrp_osr/models/hrrp_ms_resnet.py",
    "src/hrrp_osr/models/mv_rpformer.py",
    "src/hrrp_osr/models/ms_mean_factorial.py",
    "src/hrrp_osr/models/cssr_1d.py",
    "src/hrrp_osr/models/cssr_e2e_1d.py",
    "src/hrrp_osr/models/cssr_decoupled_1d.py",
    "src/hrrp_osr/training/arpl_mv_evidence.py",
    "src/hrrp_osr/training/arpl_pilot.py",
    "src/hrrp_osr/training/ms_mean_head_factorial.py",
    "src/hrrp_osr/training/mv_rpformer.py",
    "src/hrrp_osr/training/fg_mv_cssr_pilot.py",
    "src/hrrp_osr/training/fg_mv_cssr_e2e_redesign.py",
    "src/hrrp_osr/training/fg_mv_cssr_decoupled_protocol.py",
    "src/hrrp_osr/training/cssr_gradient_pathology_audit.py",
    "src/hrrp_osr/training/fg_mv_cssr_decoupled.py",
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DataConfigError(f"{name} must be a mapping")
    return value


def _require(errors: list[str], observed: Any, expected: Any, name: str) -> None:
    if observed != expected:
        errors.append(f"{name} changed: expected {expected!r}, observed {observed!r}")


def load_fg_mv_cssr_decoupled_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the single preregistered decoupled CSSR configuration."""

    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = dict(_mapping(yaml.safe_load(handle), "decoupled CSSR config"))
    errors: list[str] = []
    _require(errors, config.get("schema_version"), 1, "schema_version")
    _require(
        errors,
        config.get("stage"),
        "P3_fg_mv_cssr_gradient_audit_and_decoupled_validation",
        "stage",
    )
    _require(errors, config.get("experiment_id"), EXPERIMENT_ID, "experiment_id")
    _require(
        errors,
        config.get("result_scope"),
        "diagnostic_gradient_audit_then_smoke_pilot_and_conditional_confirmation",
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
        "r2_retrained_or_finetuned",
        "arpl_used",
        "pseudo_unknown_used",
        "angle_metadata_used_by_model",
    ):
        _require(errors, evidence.get(name), False, f"evidence_scope.{name}")

    prior = _mapping(config.get("prior_r2"), "prior_r2")
    for name, expected in {
        "experiment_id": "ms_mean_head_factorial_surrogate_v1",
        "method": "R2_MS_MEAN_CE",
        "phase": "confirmation",
        "angle_fold": ANGLE_FOLD,
        "initialization_seed": R2_SEED,
        "checkpoint_epoch": 100,
        "checkpoint_selection": "fixed_final_epoch",
        "source_config": "configs/experiments/arpl/ms_mean_head_factorial_surrogate_v1.yaml",
        "source_config_sha256": "c11daa6e2e5a7d7b72bc36840e60fc871f332c4fc85652636c729aa2eba14c71",
        "unit_relative_template": "{pair_id}/fold_0/seed_20260830/R2_MS_MEAN_CE",
        "root_artifact_hash_manifest_sha256": "edcf281df07443724d0ade1a0b2d8b20305f85b83099fb74e1c6417ee5d5477c",
        "strict_load_required": True,
        "frozen": True,
        "old_logits_exact_match_required": True,
    }.items():
        _require(errors, prior.get(name), expected, f"prior_r2.{name}")
    expected_pairs = {
        "N0": ("142a85b3a090213684126cf695b08fec259724a0bd8399dc1adb40b114aab192", "37dac18016223e08451c6551e279a6136ed494cb9c86edb5f0a938d71a2b115d", "a6bc7f4b1c095976964716e70c72666c0133f0bb3b5a63aacbc5612d9a888c93"),
        "N1": ("a4f6fa3235fbb5cf74b712588a0318f614a05287adec4ee881820424cddbcbaa", "0b8a97dcfd744896bbae912c1363379201ced18a55107f80b2d2f3256fb5c5bc", "b43da73179b8ddb0e0ae1f97b3724e9fcffe9ce32f10aaa6466cc8f408a74275"),
        "N2": ("14e2ac7b686c901112f969fe0bd7f53c29646e7c015bae794d30c39051f9c0b9", "1a7dc0031cf5b32a41131289fb4117a144463c025e93bc7a487e56a3c8c8bd2d", "58e8086e8ba27e2c4537d98d5ec1e6faaaa1bdf3d47cfec6ad9278227279114a"),
        "N3": ("6427a09f3e4a5e67ff652fea6e44c8364b62381acc8338099dccc818ac284bc9", "53fead93617851f8646dc7c76ff3773b6c55a720d3be17feda462535994e7d27", "05ef84488ee515e09afbdf4504fdd1dd8597347faa63442a49abc32265caa6e8"),
        "N4": ("169387ad7a87463110ac7a2cd45afd7dac49428538c93c84975162e425d94ff5", "8b0202d1e08ae83eec4bf07fc1dbb6a3f39fef2378ac15e57635709d8872b41a", "942e6c14d2237120ca9937a23df7f095ce718ea072933ee52d0ab2d3c3c79e95"),
        "N5": ("74cde2c6b30f1fa96219fe20777dfc632575c8c3c0281706ca016ef2497642df", "a706c63e47f8522510c2926e70a8072ca8ca183c5ef74957b8451d28d2c47c80", "ab748cce5fbb8f1299fe720311e2c1da3805bda070b3b47b5db46f886805eee5"),
        "N6": ("178dbaa9e461d28825124b688752ed5c1005a8f0265963ef57e5c27a0a65e86e", "46b454fc313573121fcf6ad214b91f9e21a2cb996a38d3beaf9c83d8321ce140", "b11a125b08a236f182857030fc07770d22a1890f5d20e68efa0f3fade4a4b20b"),
    }
    observed_hashes = _mapping(prior.get("unit_artifact_hashes"), "R2 hashes")
    for pair_id, values in expected_pairs.items():
        observed = _mapping(observed_hashes.get(pair_id), f"R2 hashes {pair_id}")
        for filename, expected in zip(
            ("checkpoint.pt", "pair_manifest.csv", "features_logits_scores.npz"),
            values,
            strict=True,
        ):
            _require(errors, observed.get(filename), expected, f"R2 {pair_id}/{filename}")

    _require(
        errors,
        dict(_mapping(config.get("bundle"), "bundle")),
        {
            "dataset_id": "hrrp_10class_theta83_hh_v1",
            "preprocessing_id": "hrrp_padding_complex_gaussian_v1",
            "profiles_sha256": "2dd92282c125f0f677cf1f2dfce828781c8ba4385cf9ae552c4a2c56033c3f5b",
            "manifest_sha256": "748b9f30629c3b3cbe66c6a1dac30863fdab2d81a214e46d8bc3ef7c6022a08a",
            "bundle_sha256": "79176091b5b745e5df5957f1bb4ade7304c1403a4a32e64622e012db67d5a0c5",
        },
        "bundle",
    )
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
    _require(
        errors,
        list(classes.get("identity_pairs", [])),
        [
            {"pair_id": "N0", "surrogate_unknown_indices": [0, 2], "train_known_indices": [1, 3, 4, 5, 6]},
            {"pair_id": "N1", "surrogate_unknown_indices": [2, 5], "train_known_indices": [0, 1, 3, 4, 6]},
            {"pair_id": "N2", "surrogate_unknown_indices": [3, 5], "train_known_indices": [0, 1, 2, 4, 6]},
            {"pair_id": "N3", "surrogate_unknown_indices": [1, 3], "train_known_indices": [0, 2, 4, 5, 6]},
            {"pair_id": "N4", "surrogate_unknown_indices": [1, 6], "train_known_indices": [0, 2, 3, 4, 5]},
            {"pair_id": "N5", "surrogate_unknown_indices": [4, 6], "train_known_indices": [0, 1, 2, 3, 5]},
            {"pair_id": "N6", "surrogate_unknown_indices": [0, 4], "train_known_indices": [1, 2, 3, 5, 6]},
        ],
        "identity pairs",
    )
    _require(errors, list(classes.get("pilot_pairs", [])), list(PILOT_PAIRS), "pilot pairs")
    _require(
        errors,
        list(classes.get("confirmation_pairs", [])),
        list(CONFIRMATION_PAIRS),
        "confirmation pairs",
    )
    data = _mapping(config.get("data"), "data")
    for name, expected in {
        "angle_fold": 0,
        "development_angle_parity": "odd",
        "view_count": 2,
        "train_unique_base_samples_per_class": 144,
        "known_calibration_unique_base_samples_per_class": 36,
        "evaluation_pairs_per_class": 500,
        "final_test_pairs_generated": False,
    }.items():
        _require(errors, data.get(name), expected, f"data.{name}")
    _require(
        errors,
        dict(_mapping(data.get("smoke"), "smoke")),
        {
            "pair_id": "N1",
            "methods": list(DECOUPLED_METHODS),
            "epochs": 6,
            "full_train_unique_base_schedule": True,
            "evaluation_pairs_per_class": 2,
            "diagnostic_only": True,
        },
        "smoke",
    )
    _require(
        errors,
        dict(_mapping(data.get("single_view_schedule"), "single-view schedule")),
        {
            "base_order": ["model_label", "sample_id"],
            "class_seed_material": "fg_mv_cssr_decoupled_single_view_class_v1|cssr_seed|pair_id|fold|epoch|model_label",
            "class_order_seed_material": "fg_mv_cssr_decoupled_single_view_class_order_v1|cssr_seed|pair_id|fold|epoch",
            "seed_hash": "sha256_first_8_bytes_big_endian_unsigned",
            "random_generator": "numpy_PCG64",
            "interleave": "deterministic_round_robin",
            "full_batch_max_class_count_difference": 1,
            "each_unique_base_once_per_epoch": True,
            "dataloader_shuffle": False,
        },
        "single-view schedule",
    )
    _require(
        errors,
        dict(_mapping(config.get("normalization"), "normalization")),
        {"method": "reuse_exact_r2_global_scalar_zscore", "epsilon": 1.0e-8},
        "normalization",
    )
    _require(
        errors,
        dict(_mapping(config.get("r2_model"), "r2 model")),
        {
            "architecture": "ms_mean_head_factorial_v1",
            "encoder": "hrrp_ms_resnet_1d_v1",
            "feature_map_shape": [128, 76],
            "fusion": "arithmetic_mean",
            "head": "linear_ce",
            "prediction": "frozen_fused_logits_argmax",
            "frozen_eval": True,
        },
        "R2 model",
    )
    methods = _mapping(config.get("methods"), "methods")
    _require(errors, methods.get("baseline"), D0_R2_CLASS_CONDITIONAL_MLS, "baseline")
    _require(errors, methods.get("background"), GLOBAL_MLS, "background")
    _require(errors, list(methods.get("candidates", [])), list(DECOUPLED_METHODS), "candidates")
    _require(errors, methods.get("d0_retraining"), False, "D0 retraining")
    model = _mapping(config.get("decoupled_model"), "decoupled_model")
    for name, expected in {"class_count": 5, "input_channels": 128, "latent_channels": 32}.items():
        _require(errors, model.get(name), expected, f"decoupled_model.{name}")
    adapter = _mapping(model.get("adapter"), "adapter")
    for name, expected in {
        "conv1_out_channels": 64,
        "conv1_kernel_size": 3,
        "conv1_padding": 1,
        "conv1_bias": False,
        "group_norm_groups": 8,
        "group_norm_eps": 1.0e-5,
        "group_norm_affine": True,
        "activation": "GELU",
        "conv2_kernel_size": 1,
        "conv2_bias": False,
        "residual_scale": 0.1,
        "shared_across_views": True,
    }.items():
        _require(errors, adapter.get(name), expected, f"adapter.{name}")
    _require(
        errors,
        dict(_mapping(model.get("autoencoder"), "autoencoder")),
        {
            "encoder_kernel_size": 3,
            "encoder_padding": 1,
            "decoder_kernel_size": 3,
            "decoder_padding": 1,
            "bias": False,
            "activation": "Tanh",
            "independent_per_class": True,
            "skip_connection": False,
            "normalization": "none",
        },
        "autoencoder",
    )
    loss = _mapping(config.get("loss"), "loss")
    weights = _mapping(loss.get("weights"), "loss.weights")
    _require(
        errors,
        {name: dict(_mapping(value, name)) for name, value in weights.items()},
        {
            D1_DECOUPLED_REL_CSSR: {"relative": 1.0, "absolute": 0.0, "separation": 0.0},
            D2_DECOUPLED_ABSREL_CSSR: {"relative": 1.0, "absolute": 0.25, "separation": 0.5},
        },
        "loss weights",
    )
    _require(
        errors,
        dict(_mapping(loss.get("relative"), "relative loss")),
        {
            "gamma": 0.1,
            "clip_min": -100.0,
            "clip_max": 100.0,
            "reconstruction_error": "channel_sum_L1",
            "probability_order": "class_softmax_per_position_then_position_mean",
        },
        "relative loss",
    )
    _require(errors, dict(_mapping(loss.get("absolute"), "absolute loss")), {"epsilon": 1.0e-8}, "absolute loss")
    _require(errors, dict(_mapping(loss.get("separation"), "separation loss")), {"margin": 0.2}, "separation loss")
    training = _mapping(config.get("training"), "training")
    for name, expected in {
        "optimizer": "AdamW",
        "epochs": 20,
        "batch_size": 128,
        "warmup_epochs": 2,
        "scheduler": "linear_warmup_then_cosine_positive_through_final_epoch",
        "lr_adapter": 3.0e-4,
        "lr_autoencoders": 1.0e-3,
        "weight_decay_adapter": 1.0e-4,
        "weight_decay_autoencoders": 1.0e-4,
        "gradient_clip_norm": 5.0,
        "cssr_seed": CSSR_SEED,
        "adapter_frozen_epochs": 5,
        "adapter_unfreeze_epoch": 6,
        "early_stopping": False,
        "formal_checkpoint_epoch": 20,
        "checkpoint_selection": "fixed_final_epoch",
    }.items():
        _require(errors, training.get(name), expected, f"training.{name}")
    _require(
        errors,
        training.get("scheduler_factor_formula"),
        "epoch<=2:epoch/2;epoch>2:0.5*(1+cos(pi*(epoch-3)/18));post_epoch_20:0",
        "scheduler formula",
    )
    _require(
        errors,
        dict(_mapping(config.get("diagnostics"), "diagnostics")),
        {
            "every_epoch": True,
            "u_population": "sorted_full_train_known_720",
            "channel_variance_collapse_threshold": 1.0e-12,
            "effective_rank_epsilon": 1.0e-12,
            "singular_value_energy_top_k": 10,
        },
        "diagnostics",
    )
    calibration = _mapping(config.get("calibration"), "calibration")
    for name, expected in {
        "reference_count_per_class": 36,
        "leave_one_base_out": True,
        "score_epsilon": 1.0e-8,
        "guided_class": "frozen_r2_fused_argmax",
        "pair_reduction": "arithmetic_mean",
        "threshold_known_acceptance_rate": 0.95,
    }.items():
        _require(errors, calibration.get(name), expected, f"calibration.{name}")
    for name, expected in {
        "reference_population": "unique_known_calibration_base_by_true_class",
        "tail": "greater_than_or_equal_normalized_absolute_reconstruction_error",
        "smoothing": "plus_one_numerator_and_denominator",
        "score_transform": "negative_log_p_plus_epsilon",
        "threshold_source": "known_calibration_pairs_only",
    }.items():
        _require(errors, calibration.get(name), expected, f"calibration.{name}")
    _require(
        errors,
        list(_mapping(config.get("evaluation"), "evaluation").get("report_metrics", [])),
        list(REPORT_METRIC_KEYS),
        "report metrics",
    )
    _require(
        errors,
        dict(_mapping(config.get("pilot_gate"), "pilot gate")),
        {
            "baseline": D0_R2_CLASS_CONDITIONAL_MLS,
            "candidates": list(DECOUPLED_METHODS),
            "minimum_mean_auroc_delta": 0.02,
            "minimum_positive_pair_count": 2,
            "minimum_mean_oscr_delta": 0.0,
            "maximum_mean_kccr_drop": 0.01,
            "maximum_mean_fpr95_increase": 0.02,
            "minimum_identity_auroc": 0.40,
            "maximum_identity_auroc_drop": 0.10,
            "directed_ddg_false_accept_not_above_d0": True,
            "d1_priority": True,
            "d2_minimum_mean_auroc_advantage_over_d1": 0.02,
            "failure_label": "decoupled_cssr_failed",
        },
        "pilot gate",
    )
    _require(
        errors,
        dict(_mapping(config.get("confirmation_gate"), "confirmation gate")),
        {
            "baseline": D0_R2_CLASS_CONDITIONAL_MLS,
            "minimum_mean_auroc_delta": 0.02,
            "minimum_positive_pair_count": 3,
            "minimum_mean_oscr_delta": 0.0,
            "maximum_mean_kccr_drop": 0.01,
            "maximum_mean_fpr95_increase": 0.02,
            "minimum_identity_auroc": 0.40,
            "maximum_identity_auroc_drop": 0.10,
            "pass_label": "decoupled_cssr_worth_full_validation",
            "failure_label": "decoupled_cssr_rejected",
        },
        "confirmation gate",
    )
    runtime = _mapping(config.get("runtime"), "runtime")
    for name, expected in {
        "formal_device": "cuda",
        "expected_gpu_model": "NVIDIA GeForce RTX 4090",
        "maximum_parallel_tasks": 4,
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "allow_tf32": False,
    }.items():
        _require(errors, runtime.get(name), expected, f"runtime.{name}")
    outputs = _mapping(config.get("outputs"), "outputs")
    _require(errors, outputs.get("namespace"), EXPERIMENT_ID, "output namespace")
    _require(errors, outputs.get("overwrite_existing"), False, "overwrite")
    _require(errors, outputs.get("final_unknown_test_authorized"), False, "final authorization")
    if errors:
        raise DataConfigError("invalid decoupled CSSR config:\n- " + "\n- ".join(errors))
    config["_config_path"] = str(config_path)
    config["_config_sha256"] = file_sha256(config_path)
    return config


def task_source_hashes(project_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in TASK_SOURCE_FILES:
        path = project_root / relative
        if not path.is_file():
            raise DataValidationError(f"missing task source: {relative}")
        hashes[relative] = file_sha256(path)
    return hashes


def _configure_runtime(config: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        raise DataValidationError("formal decoupled CSSR execution requires CUDA")
    runtime = _mapping(config["runtime"], "runtime")
    observed_name = torch.cuda.get_device_name(device)
    if observed_name != str(runtime["expected_gpu_model"]):
        raise DataValidationError(
            f"unexpected GPU: expected {runtime['expected_gpu_model']!r}, observed {observed_name!r}"
        )
    torch.use_deterministic_algorithms(bool(runtime["deterministic_algorithms"]))
    torch.backends.cudnn.benchmark = bool(runtime["cudnn_benchmark"])
    torch.backends.cuda.matmul.allow_tf32 = bool(runtime["allow_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(runtime["allow_tf32"])
    return {
        "device": str(device),
        "cuda_device_name": observed_name,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _git_environment(project_root: Path, device: torch.device) -> dict[str, Any]:
    def command(*parts: str) -> str:
        return subprocess.run(
            parts,
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "git_commit": command("git", "rev-parse", "HEAD"),
        "git_branch": command("git", "branch", "--show-current"),
        "git_status_porcelain": command("git", "status", "--porcelain"),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device),
    }


def _state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _parameter_vector(parameters: Iterable[nn.Parameter]) -> torch.Tensor:
    values = [parameter.detach().reshape(-1).cpu().to(torch.float64) for parameter in parameters]
    return torch.cat(values) if values else torch.zeros(0, dtype=torch.float64)


def _relative_update(before: torch.Tensor, parameters: Iterable[nn.Parameter]) -> float:
    after = _parameter_vector(parameters)
    if after.shape != before.shape:
        raise DataValidationError("parameter group shape changed during training")
    return float(torch.linalg.vector_norm(after - before) / max(float(torch.linalg.vector_norm(before)), 1.0e-12))


def _gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().square().sum())
    return math.sqrt(total)


def _learning_rate_factor(epoch: int, *, warmup_epochs: int, total_epochs: int) -> float:
    if epoch < 1 or epoch > total_epochs:
        raise DataValidationError("epoch is outside the frozen training range")
    if epoch <= warmup_epochs:
        return epoch / warmup_epochs
    return 0.5 * (
        1.0
        + math.cos(
            math.pi * (epoch - warmup_epochs - 1) / (total_epochs - warmup_epochs)
        )
    )


def _infer_decoupled(
    model: FGMVCSSRDecoupled1D,
    features: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    values = torch.from_numpy(np.asarray(features, dtype=np.float32))
    collected: dict[str, list[np.ndarray]] = {"u": [], "r": [], "probabilities": []}
    model.eval()
    with torch.no_grad():
        for start in range(0, values.shape[0], batch_size):
            output = model(values[start : start + batch_size].to(device))
            tensors = {
                "u": output.adapted_features,
                "r": output.normalized_reconstruction_errors,
                "probabilities": output.probabilities,
            }
            for name, tensor in tensors.items():
                collected[name].append(tensor.detach().cpu().numpy())
    result = {
        name: np.concatenate(parts, axis=0).astype(np.float32)
        for name, parts in collected.items()
    }
    expected = {
        "u": (features.shape[0], 128, 76),
        "r": (features.shape[0], 5),
        "probabilities": (features.shape[0], 5),
    }
    for name, shape in expected.items():
        if result[name].shape != shape or not np.isfinite(result[name]).all():
            raise DataValidationError(f"invalid decoupled inference array: {name}")
    return result


def _classification_diagnostics(
    model: FGMVCSSRDecoupled1D,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    output = _infer_decoupled(model, features, device=device, batch_size=batch_size)
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = output["probabilities"].astype(np.float64)
    r = output["r"].astype(np.float64)
    true_r = r[np.arange(labels.size), labels]
    masked = r.copy()
    masked[np.arange(labels.size), labels] = np.inf
    wrong_r = masked.min(axis=1)
    return {
        "accuracy": float(np.mean(probabilities.argmax(axis=1) == labels)),
        "relative_nll": float(-np.log(probabilities[np.arange(labels.size), labels] + 1.0e-8).mean()),
        "true_class_r": float(true_r.mean()),
        "nearest_wrong_r": float(wrong_r.mean()),
        "reconstruction_margin_wrong_minus_true": float((wrong_r - true_r).mean()),
    }


def _u_statistics(u: np.ndarray, *, config: Mapping[str, Any]) -> dict[str, Any]:
    values = np.asarray(u, dtype=np.float64)
    if values.shape != (720, 128, 76) or not np.isfinite(values).all():
        raise DataValidationError("U diagnostic population is not the sorted 720 train-known bases")
    channel_variance = values.transpose(0, 2, 1).reshape(-1, 128).var(axis=0)
    threshold = float(config["diagnostics"]["channel_variance_collapse_threshold"])
    if float(channel_variance.max()) <= threshold:
        raise DataValidationError("decoupled CSSR feature U collapsed to a constant")
    centered = values.transpose(0, 2, 1).reshape(-1, 128)
    centered -= centered.mean(axis=0, keepdims=True)
    gram = centered.T @ centered
    eigenvalues = np.linalg.eigvalsh(gram)
    energy = np.maximum(eigenvalues[::-1], 0.0)
    energy_sum = float(energy.sum())
    if energy_sum <= 0.0:
        raise DataValidationError("U singular-value energy is zero")
    proportions = energy / energy_sum
    epsilon = float(config["diagnostics"]["effective_rank_epsilon"])
    effective_rank = float(np.exp(-np.sum(proportions * np.log(proportions + epsilon))))
    top_k = int(config["diagnostics"]["singular_value_energy_top_k"])
    return {
        "mean_sample_frobenius_norm": float(np.linalg.norm(values, axis=(1, 2)).mean()),
        "channel_variance_min": float(channel_variance.min()),
        "channel_variance_mean": float(channel_variance.mean()),
        "channel_variance_max": float(channel_variance.max()),
        "effective_rank": effective_rank,
        "top_singular_value_energy_fraction": proportions[:top_k].tolist(),
        "collapsed": False,
    }


def train_decoupled_cssr(
    *,
    train_rows: Sequence[Mapping[str, Any]],
    train_features: np.ndarray,
    calibration_features: np.ndarray,
    calibration_labels: np.ndarray,
    pair_id: str,
    method: str,
    config: Mapping[str, Any],
    device: torch.device,
    smoke: bool,
) -> dict[str, Any]:
    if method not in DECOUPLED_METHODS:
        raise DataValidationError(f"unsupported decoupled method: {method}")
    if len(train_rows) != 720 or train_features.shape != (720, 128, 76):
        raise DataValidationError("decoupled CSSR training population is not 720 unique bases")
    train_labels = np.asarray([int(row["model_label"]) for row in train_rows], dtype=np.int64)
    if Counter(train_labels.tolist()) != Counter({index: 144 for index in range(5)}):
        raise DataValidationError("decoupled CSSR training classes are not 5 x 144")
    if len({str(row["sample_id"]) for row in train_rows}) != 720:
        raise DataValidationError("a train base sample is duplicated")
    if any(str(row["experiment_role"]) != "train_known" for row in train_rows):
        raise DataValidationError("non-train evidence entered decoupled training")

    training = _mapping(config["training"], "training")
    _set_determinism(CSSR_SEED, bool(config["runtime"]["deterministic_algorithms"]))
    model = FGMVCSSRDecoupled1D(
        num_classes=5,
        input_channels=128,
        latent_channels=32,
        residual_scale=0.1,
        gamma=0.1,
        clip_length=100.0,
        epsilon=1.0e-8,
        margin=0.2,
    ).to(device)
    initial_state_sha256 = _state_sha256(model.state_dict())
    groups = model.parameter_groups()
    adapter_parameters = tuple(groups["adapter"])
    ae_parameters = tuple(groups["autoencoders"])
    if set(map(id, adapter_parameters)) & set(map(id, ae_parameters)):
        raise DataValidationError("adapter and AE parameter groups overlap")
    optimizer = torch.optim.AdamW(
        [
            {
                "params": adapter_parameters,
                "lr": float(training["lr_adapter"]),
                "weight_decay": float(training["weight_decay_adapter"]),
                "group_name": "adapter",
            },
            {
                "params": ae_parameters,
                "lr": float(training["lr_autoencoders"]),
                "weight_decay": float(training["weight_decay_autoencoders"]),
                "group_name": "autoencoders",
            },
        ]
    )
    # Freeze the training RNG stream independently of construction/optimizer internals.
    _set_determinism(CSSR_SEED, bool(config["runtime"]["deterministic_algorithms"]))
    epochs = int(config["data"]["smoke"]["epochs"] if smoke else training["epochs"])
    batch_size = int(training["batch_size"])
    x_train = torch.from_numpy(np.asarray(train_features, dtype=np.float32))
    y_train = torch.from_numpy(train_labels)
    log: list[dict[str, Any]] = []
    schedules: list[list[dict[str, Any]]] = []
    schedule_audits: list[dict[str, Any]] = []
    epoch_hashes: list[str] = []
    all_parameters = tuple(model.parameters())

    for epoch in range(1, epochs + 1):
        adapter_trainable = epoch >= int(training["adapter_unfreeze_epoch"])
        model.configure_for_epoch(epoch)
        if any(parameter.requires_grad != adapter_trainable for parameter in adapter_parameters):
            raise DataValidationError("adapter freeze schedule changed")
        if not all(parameter.requires_grad for parameter in ae_parameters):
            raise DataValidationError("class autoencoders were unexpectedly frozen")
        factor = _learning_rate_factor(
            epoch,
            warmup_epochs=int(training["warmup_epochs"]),
            total_epochs=int(training["epochs"]),
        )
        optimizer.param_groups[0]["lr"] = (
            float(training["lr_adapter"]) * factor if adapter_trainable else 0.0
        )
        optimizer.param_groups[1]["lr"] = float(training["lr_autoencoders"]) * factor
        indices, schedule_audit = build_single_view_schedule(
            train_rows,
            pair_id=pair_id,
            angle_fold=ANGLE_FOLD,
            epoch=epoch,
            cssr_seed=CSSR_SEED,
        )
        if sorted(indices.tolist()) != list(range(720)):
            raise DataValidationError("single-view epoch does not use every base exactly once")
        schedule = [
            {
                "schedule_index": schedule_index,
                "source_row_index": int(source_index),
                "model_label": int(train_rows[int(source_index)]["model_label"]),
                "sample_id": str(train_rows[int(source_index)]["sample_id"]),
            }
            for schedule_index, source_index in enumerate(indices)
        ]
        schedules.append(schedule)
        schedule_audits.append(schedule_audit)
        epoch_hashes.append(str(schedule_audit["schedule_sha256"]))
        adapter_before = _parameter_vector(adapter_parameters)
        ae_before = _parameter_vector(ae_parameters)
        totals = {
            "relative": 0.0,
            "absolute": 0.0,
            "separation": 0.0,
            "total": 0.0,
            "correct": 0.0,
            "true_r": 0.0,
            "wrong_r": 0.0,
            "margin": 0.0,
            "adapter_grad": 0.0,
            "ae_grad": 0.0,
            "total_grad": 0.0,
            "clip": 0.0,
        }
        sample_count = 0
        batch_count = 0
        model.train()
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            x = x_train[batch_indices].to(device)
            y = y_train[batch_indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss_output = model.loss(x, y, method)
            if not bool(torch.isfinite(loss_output.total_loss)):
                raise DataValidationError("decoupled CSSR loss is non-finite")
            loss_output.total_loss.backward()
            adapter_grad = _gradient_norm(adapter_parameters)
            ae_grad = _gradient_norm(ae_parameters)
            total_grad = _gradient_norm(all_parameters)
            if not all(math.isfinite(value) for value in (adapter_grad, ae_grad, total_grad)):
                raise DataValidationError("decoupled CSSR gradient is non-finite")
            clip_scale = min(
                1.0,
                float(training["gradient_clip_norm"]) / (total_grad + 1.0e-6),
            )
            torch.nn.utils.clip_grad_norm_(all_parameters, float(training["gradient_clip_norm"]))
            optimizer.step()
            if not all(bool(torch.isfinite(parameter).all()) for parameter in all_parameters):
                raise DataValidationError("decoupled CSSR parameter is non-finite")
            count = int(y.numel())
            output = loss_output.output
            r = output.normalized_reconstruction_errors
            true_r = r.gather(1, y[:, None]).squeeze(1)
            wrong_r = r.masked_fill(torch.nn.functional.one_hot(y, 5).bool(), float("inf")).min(dim=1).values
            totals["relative"] += float(loss_output.relative_loss.detach()) * count
            totals["absolute"] += float(loss_output.absolute_loss.detach()) * count
            totals["separation"] += float(loss_output.separation_loss.detach()) * count
            totals["total"] += float(loss_output.total_loss.detach()) * count
            totals["correct"] += float((output.probabilities.argmax(dim=1) == y).sum())
            totals["true_r"] += float(true_r.detach().sum())
            totals["wrong_r"] += float(wrong_r.detach().sum())
            totals["margin"] += float((wrong_r - true_r).detach().sum())
            totals["adapter_grad"] += adapter_grad
            totals["ae_grad"] += ae_grad
            totals["total_grad"] += total_grad
            totals["clip"] += float(clip_scale < 1.0)
            sample_count += count
            batch_count += 1
        if sample_count != 720:
            raise DataValidationError("single-view epoch sample count changed")
        sorted_output = _infer_decoupled(model, train_features, device=device, batch_size=batch_size)
        u_stats = _u_statistics(sorted_output["u"], config=config)
        train_diag = _classification_diagnostics(
            model, train_features, train_labels, device=device, batch_size=batch_size
        )
        cal_diag = _classification_diagnostics(
            model,
            calibration_features,
            calibration_labels,
            device=device,
            batch_size=batch_size,
        )
        log.append(
            {
                "epoch": epoch,
                "method": method,
                "adapter_trainable": adapter_trainable,
                "learning_rate_factor": factor,
                "learning_rates": {
                    "adapter": float(optimizer.param_groups[0]["lr"]),
                    "autoencoders": float(optimizer.param_groups[1]["lr"]),
                },
                "train_total_loss": totals["total"] / sample_count,
                "train_relative_loss": totals["relative"] / sample_count,
                "train_absolute_loss_diagnostic": totals["absolute"] / sample_count,
                "train_separation_loss_diagnostic": totals["separation"] / sample_count,
                "train_cssr_accuracy": totals["correct"] / sample_count,
                "train_true_class_r": totals["true_r"] / sample_count,
                "train_nearest_wrong_r": totals["wrong_r"] / sample_count,
                "train_reconstruction_margin": totals["margin"] / sample_count,
                "mean_adapter_gradient_norm": totals["adapter_grad"] / batch_count,
                "mean_autoencoder_gradient_norm": totals["ae_grad"] / batch_count,
                "mean_total_gradient_norm": totals["total_grad"] / batch_count,
                "clipping_batch_fraction": totals["clip"] / batch_count,
                "adapter_parameter_relative_update": _relative_update(adapter_before, adapter_parameters),
                "autoencoder_parameter_relative_update": _relative_update(ae_before, ae_parameters),
                "sorted_full_train_diagnostics": train_diag,
                "known_calibration_diagnostics": cal_diag,
                "u_diagnostics": u_stats,
                "sample_schedule_sha256": epoch_hashes[-1],
                "sample_count": sample_count,
                "known_calibration_used_for_training": False,
                "surrogate_unknown_used_for_training": False,
                "final_unknown_used": False,
            }
        )
    model.eval()
    return {
        "model": model,
        "training_log": log,
        "schedules": schedules,
        "schedule_audits": schedule_audits,
        "audit": {
            "status": "passed",
            "epochs": epochs,
            "formal_checkpoint_epoch": None if smoke else int(training["formal_checkpoint_epoch"]),
            "checkpoint_selection": "fixed_final_epoch",
            "initial_state_sha256": initial_state_sha256,
            "final_state_sha256": _state_sha256(model.state_dict()),
            "epoch_schedule_sha256": epoch_hashes,
            "schedule_sha256": _sequence_sha256(epoch_hashes),
            "train_unique_sample_count": 720,
            "train_class_counts": [144] * 5,
            "adapter_frozen_epochs": list(range(1, min(epochs, 5) + 1)),
            "adapter_unfrozen_epochs": list(range(6, epochs + 1)),
            "all_parameters_finite": True,
            "pair_multiplicity_weight": False,
            "known_calibration_used_for_training": False,
            "surrogate_unknown_used_for_training": False,
            "final_unknown_used": False,
            "even_angle_test_used": False,
        },
    }


def _selected_unique_rows(
    unique_rows: Sequence[Mapping[str, Any]], role: str
) -> tuple[list[dict[str, Any]], np.ndarray]:
    indices = np.asarray(
        [index for index, row in enumerate(unique_rows) if str(row["experiment_role"]) == role],
        dtype=np.int64,
    )
    return [dict(unique_rows[int(index)]) for index in indices], indices


def evaluate_decoupled_unit(
    *,
    model: FGMVCSSRDecoupled1D,
    method: str,
    prepared: Any,
    unique_rows: Sequence[Mapping[str, Any]],
    unique_features: np.ndarray,
    frozen_r2_arrays: Mapping[str, Mapping[str, np.ndarray]],
    pair_id: str,
    config: Mapping[str, Any],
    device: torch.device,
    smoke: bool,
) -> dict[str, Any]:
    batch_size = int(config["training"]["batch_size"])
    unique_output = _infer_decoupled(model, unique_features, device=device, batch_size=batch_size)
    reference_arrays, references, reference_ids, reference_metadata = build_guided_reference_scores(
        unique_rows,
        unique_output["r"].astype(np.float64),
        epsilon=float(config["calibration"]["score_epsilon"]),
    )
    reference_arrays.update(
        {
            "full_calibration_logits": np.asarray(
                frozen_r2_arrays["known_calibration"]["global_logits"]
            ),
            "full_calibration_labels": np.asarray(
                prepared.labels["known_calibration"], dtype=np.int64
            ),
            "full_calibration_pair_ids": np.asarray(
                prepared.pair_ids["known_calibration"], dtype=np.str_
            ),
        }
    )
    role_indices = _evaluation_role_indices(prepared, smoke=smoke, config=config)
    role_pair_rows = {
        role: [
            _role_manifest_rows(prepared, role)[int(index)]
            for index in role_indices[role]
        ]
        for role in ("known_calibration", "surrogate_unknown")
    }
    role_logits = {
        role: np.asarray(frozen_r2_arrays[role]["global_logits"])[role_indices[role]]
        for role in role_indices
    }
    cc_mls = _class_conditional_mls_for_roles(
        full_calibration_logits=np.asarray(frozen_r2_arrays["known_calibration"]["global_logits"]),
        full_calibration_labels=np.asarray(prepared.labels["known_calibration"]),
        full_calibration_pair_ids=prepared.pair_ids["known_calibration"],
        role_logits=role_logits,
        role_pair_rows=role_pair_rows,
    )
    score_by_sample = reference_metadata["score_by_sample"]
    guided: dict[str, np.ndarray] = {}
    for role, rows in role_pair_rows.items():
        predictions = role_logits[role].argmax(axis=1)
        guided[role] = np.asarray(
            [
                0.5
                * (
                    score_by_sample[str(row["view1_sample_id"])][predictions[index]]
                    + score_by_sample[str(row["view2_sample_id"])][predictions[index]]
                )
                for index, row in enumerate(rows)
            ],
            dtype=np.float64,
        )
        swapped = np.asarray(
            [
                0.5
                * (
                    score_by_sample[str(row["view2_sample_id"])][predictions[index]]
                    + score_by_sample[str(row["view1_sample_id"])][predictions[index]]
                )
                for index, row in enumerate(rows)
            ],
            dtype=np.float64,
        )
        if not np.array_equal(guided[role], swapped):
            raise DataValidationError("guided score is not exactly view-swap invariant")
    method_scores = {
        role: {
            "known_prediction": role_logits[role].argmax(axis=1),
            "main_unknown_score": guided[role],
            "main_score_name": "fusion_guided_decoupled_cssr",
            "diagnostic_class_conditional_mls": cc_mls[role],
        }
        for role in role_indices
    }
    d0_scores = {
        role: {
            "known_prediction": role_logits[role].argmax(axis=1),
            "main_unknown_score": cc_mls[role],
            "main_score_name": "r2_class_conditional_mls",
            "diagnostic_class_conditional_mls": None,
        }
        for role in role_indices
    }
    global_scores = {
        role: {
            "known_prediction": role_logits[role].argmax(axis=1),
            "main_unknown_score": -role_logits[role].max(axis=1),
            "main_score_name": "r2_global_mls_background",
            "diagnostic_class_conditional_mls": None,
        }
        for role in role_indices
    }
    acceptance = float(config["calibration"]["threshold_known_acceptance_rate"])
    method_metrics = _evaluate_score_arrays(
        prepared=prepared,
        role_indices=role_indices,
        score_arrays=method_scores,
        acceptance_rate=acceptance,
    )
    d0_metrics = _evaluate_score_arrays(
        prepared=prepared,
        role_indices=role_indices,
        score_arrays=d0_scores,
        acceptance_rate=acceptance,
    )
    global_metrics = _evaluate_score_arrays(
        prepared=prepared,
        role_indices=role_indices,
        score_arrays=global_scores,
        acceptance_rate=acceptance,
    )
    method_rows = _build_method_prediction_rows(
        method=method,
        prepared=prepared,
        role_indices=role_indices,
        role_pair_rows=role_pair_rows,
        role_logits=role_logits,
        role_scores=method_scores,
        metrics=method_metrics,
        reference_metadata=reference_metadata,
    )
    d0_rows = _build_method_prediction_rows(
        method=D0_R2_CLASS_CONDITIONAL_MLS,
        prepared=prepared,
        role_indices=role_indices,
        role_pair_rows=role_pair_rows,
        role_logits=role_logits,
        role_scores=d0_scores,
        metrics=d0_metrics,
        reference_metadata=None,
    )
    _metrics_exact(
        method_metrics,
        recompute_method_metrics_from_prediction_rows(
            method_rows, known_acceptance_rate=acceptance
        ),
        context=method,
    )
    _metrics_exact(
        d0_metrics,
        recompute_method_metrics_from_prediction_rows(
            d0_rows, known_acceptance_rate=acceptance
        ),
        context=D0_R2_CLASS_CONDITIONAL_MLS,
    )
    shared_prediction_audit = {
        "status": "passed",
        "methods": [D0_R2_CLASS_CONDITIONAL_MLS, method],
        "known_logits_sha256": _array_sha256(role_logits["known_calibration"]),
        "known_prediction_sha256": _array_sha256(
            role_logits["known_calibration"].argmax(axis=1).astype(np.int64)
        ),
        "surrogate_logits_sha256": _array_sha256(role_logits["surrogate_unknown"]),
        "surrogate_prediction_sha256": _array_sha256(
            role_logits["surrogate_unknown"].argmax(axis=1).astype(np.int64)
        ),
        "logits_exactly_equal": True,
        "known_predictions_exactly_equal": True,
        "known_accuracy_exactly_equal": (
            method_metrics["known_accuracy"] == d0_metrics["known_accuracy"]
        ),
        "known_macro_f1_exactly_equal": (
            method_metrics["known_macro_f1"] == d0_metrics["known_macro_f1"]
        ),
    }
    if not all(
        shared_prediction_audit[name]
        for name in (
            "logits_exactly_equal",
            "known_predictions_exactly_equal",
            "known_accuracy_exactly_equal",
            "known_macro_f1_exactly_equal",
        )
    ):
        raise DataValidationError("decoupled CSSR changed the frozen R2 known prediction")
    method_identity, method_absorption, method_error = build_identity_and_absorption_rows(
        method_rows,
        method=method,
        pair_id=pair_id,
        train_class_order=prepared.train_class_order,
        acceptance_rate=acceptance,
    )
    d0_identity, d0_absorption, d0_error = build_identity_and_absorption_rows(
        d0_rows,
        method=D0_R2_CLASS_CONDITIONAL_MLS,
        pair_id=pair_id,
        train_class_order=prepared.train_class_order,
        acceptance_rate=acceptance,
    )
    return {
        "unique_output": unique_output,
        "reference_arrays": reference_arrays,
        "references": references,
        "reference_ids": reference_ids,
        "reference_metadata": reference_metadata,
        "role_indices": role_indices,
        "role_pair_rows": role_pair_rows,
        "role_logits": role_logits,
        "method_prediction_rows": method_rows,
        "d0_prediction_rows": d0_rows,
        "method_metrics": method_metrics,
        "d0_metrics": d0_metrics,
        "global_mls_metrics": global_metrics,
        "identity_rows": [*method_identity, *d0_identity],
        "absorption_rows": [*method_absorption, *d0_absorption],
        "error_analysis": {"method": method_error, "d0": d0_error},
        "shared_prediction_audit": shared_prediction_audit,
    }


def _unit_destination(root: Path, pair_id: str, method: str) -> Path:
    return root / pair_id / "fold_0" / f"seed_{CSSR_SEED}" / method


def _reference_metadata_for_json(
    metadata: Mapping[str, Any], reference_ids: Sequence[Sequence[str]]
) -> dict[str, Any]:
    result = {
        key: value
        for key, value in metadata.items()
        if key not in {"score_by_sample", "r_by_sample", "p_by_sample"}
    }
    result["reference_sample_ids"] = [list(values) for values in reference_ids]
    return result


def save_unit_result(
    destination: Path,
    *,
    phase: str,
    pair_id: str,
    method: str,
    config: Mapping[str, Any],
    prepared: Any,
    unique_rows: Sequence[Mapping[str, Any]],
    unique_features: np.ndarray,
    training_result: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    r2_audit: Mapping[str, Any],
    r2_state_before: str,
    r2_state_after: str,
    runtime_contract: Mapping[str, Any],
    environment: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    confirmation_authorization: Mapping[str, Any] | None,
    gradient_audit_authorization: Mapping[str, Any],
    smoke_authorization: Mapping[str, Any] | None,
    wall_time_seconds: float,
) -> dict[str, Any]:
    smoke = phase == "smoke"
    destination.mkdir(parents=True, exist_ok=False)
    _atomic_write_bytes(destination / "source_pair_manifest.csv", prepared.pair_manifest_bytes)
    _write_csv(destination / "unique_base_sample_manifest.csv", unique_rows)
    evaluation_rows = [
        {**row, "evaluation_subset_role": role, "evaluation_subset_index": index}
        for role in ("known_calibration", "surrogate_unknown")
        for index, row in enumerate(evaluation["role_pair_rows"][role])
    ]
    _write_csv(destination / "evaluation_pair_manifest.csv", evaluation_rows)
    for rows, audit in zip(
        training_result["schedules"], training_result["schedule_audits"], strict=True
    ):
        epoch = int(audit["epoch"])
        _write_csv(destination / "single_view_schedules" / f"epoch_{epoch:03d}.csv", rows)
        _write_json(destination / "single_view_schedule_audits" / f"epoch_{epoch:03d}.json", audit)
    _atomic_write_bytes(
        destination / "training_log.jsonl",
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in training_result["training_log"]
        ).encode("utf-8"),
    )
    _write_json(destination / "training_audit.json", training_result["audit"])
    _write_json(destination / "r2_reference_audit.json", dict(r2_audit))
    _write_json(
        destination / "r2_frozen_state_audit.json",
        {
            "before_sha256": r2_state_before,
            "after_sha256": r2_state_after,
            "unchanged": r2_state_before == r2_state_after,
            "all_parameters_requires_grad_false": True,
            "eval_mode": True,
        },
    )
    _write_json(destination / "normalization.json", _normalization_record(prepared))
    reference_arrays = dict(evaluation["reference_arrays"])
    for index, values in enumerate(evaluation["references"]):
        reference_arrays[f"class_{index}_reference_r"] = np.asarray(values)
    _save_npz(destination / "reference_scores.npz", reference_arrays)
    _write_json(
        destination / "reference_distribution.json",
        _reference_metadata_for_json(
            evaluation["reference_metadata"], evaluation["reference_ids"]
        ),
    )
    _write_csv(destination / "predictions_and_scores.csv", evaluation["method_prediction_rows"])
    _write_csv(destination / "d0_predictions_and_scores.csv", evaluation["d0_prediction_rows"])
    _write_json(destination / "metrics.json", evaluation["method_metrics"])
    _write_json(destination / "d0_metrics.json", evaluation["d0_metrics"])
    _write_json(destination / "global_mls_background_metrics.json", evaluation["global_mls_metrics"])
    _write_csv(destination / "identity_metrics.csv", evaluation["identity_rows"])
    _write_csv(destination / "absorption_by_known_class.csv", evaluation["absorption_rows"])
    _write_json(destination / "error_analysis.json", evaluation["error_analysis"])
    _write_json(destination / "shared_r2_prediction_audit.json", evaluation["shared_prediction_audit"])

    model: FGMVCSSRDecoupled1D = training_result["model"]
    checkpoint = {
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "mode": "smoke" if smoke else "full",
        "pair_id": pair_id,
        "method": method,
        "architecture": "fg_mv_cssr_decoupled_1d_v1",
        "model_state_dict": {
            name: value.detach().cpu().clone() for name, value in model.state_dict().items()
        },
        "checkpoint_epoch": int(training_result["audit"]["epochs"]),
        "formal_checkpoint": not smoke,
        "checkpoint_selection": "fixed_final_epoch",
        "cssr_seed": CSSR_SEED,
        "train_class_order": tuple(prepared.train_class_order),
        "source_pair_manifest_sha256": prepared.pair_manifest_sha256,
        "unique_base_manifest_sha256": hashlib.sha256(_render_csv(unique_rows)).hexdigest(),
        "unique_feature_map_sha256": _array_sha256(unique_features),
        "schedule_sha256": training_result["audit"]["schedule_sha256"],
        "config_sha256": config["_config_sha256"],
        "r2_checkpoint_sha256": r2_audit["checkpoint_sha256"],
        "r2_state_sha256": r2_state_after,
        "final_unknown_used": False,
        "even_angle_test_used": False,
    }
    checkpoint_buffer = io.BytesIO()
    torch.save(checkpoint, checkpoint_buffer)
    _atomic_write_bytes(destination / "checkpoint.pt", checkpoint_buffer.getvalue())
    _save_npz(
        destination / "checkpoint_replay.npz",
        {
            "unique_features": np.asarray(unique_features, dtype=np.float32),
            "expected_u": evaluation["unique_output"]["u"],
            "expected_r": evaluation["unique_output"]["r"],
            "expected_probabilities": evaluation["unique_output"]["probabilities"],
        },
    )
    restored = FGMVCSSRDecoupled1D(
        num_classes=5,
        input_channels=128,
        latent_channels=32,
        residual_scale=0.1,
        gamma=0.1,
        clip_length=100.0,
        epsilon=1.0e-8,
        margin=0.2,
    ).to(next(model.parameters()).device)
    incompatible = restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise DataValidationError("decoupled checkpoint did not strict-load")
    replay = _infer_decoupled(
        restored,
        unique_features,
        device=next(model.parameters()).device,
        batch_size=int(config["training"]["batch_size"]),
    )
    for name in ("u", "r", "probabilities"):
        if not np.array_equal(replay[name], evaluation["unique_output"][name]):
            raise DataValidationError(f"decoupled checkpoint replay is not bitwise exact: {name}")

    resolved = dict(config)
    resolved["_resolved"] = {
        "phase": phase,
        "mode": "smoke" if smoke else "full",
        "pair_id": pair_id,
        "method": method,
        "angle_fold": ANGLE_FOLD,
        "r2_seed": R2_SEED,
        "cssr_seed": CSSR_SEED,
        "source_pair_manifest_sha256": prepared.pair_manifest_sha256,
        "unique_base_manifest_sha256": checkpoint["unique_base_manifest_sha256"],
        "unique_feature_map_sha256": checkpoint["unique_feature_map_sha256"],
        "r2_checkpoint_sha256": r2_audit["checkpoint_sha256"],
        "confirmation_authorization": confirmation_authorization,
        "gradient_audit_authorization": dict(gradient_audit_authorization),
        "smoke_authorization": None
        if smoke_authorization is None
        else dict(smoke_authorization),
        "test_features_materialized": False,
    }
    _atomic_write_bytes(
        destination / "resolved_config.yaml",
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False).encode("utf-8"),
    )
    contract = {
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "mode": "smoke" if smoke else "full",
        "pair_id": pair_id,
        "method": method,
        "angle_fold": ANGLE_FOLD,
        "r2_seed": R2_SEED,
        "cssr_seed": CSSR_SEED,
        "config_sha256": config["_config_sha256"],
        "source_hashes": dict(source_hashes),
        "runtime_contract": dict(runtime_contract),
        "confirmation_authorization": confirmation_authorization,
        "gradient_audit_authorization": dict(gradient_audit_authorization),
        "smoke_authorization": None
        if smoke_authorization is None
        else dict(smoke_authorization),
        "known_prediction_source": "frozen_r2_fused_logits_argmax",
        "main_score": "fusion_guided_decoupled_cssr",
        "threshold_source": "known_calibration_only",
        "train_unique_single_view_bases": 720,
        "pair_multiplicity_weight": False,
        "r2_retrained_or_finetuned": False,
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
    _write_json(destination / "unit_contract.json", contract)
    _write_json(destination / "environment.json", dict(environment))
    summary = {
        "status": "complete",
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "mode": "smoke" if smoke else "full",
        "pair_id": pair_id,
        "method": method,
        "metrics": evaluation["method_metrics"],
        "d0_metrics": evaluation["d0_metrics"],
        "global_mls_background_metrics": evaluation["global_mls_metrics"],
        "identity_metrics": evaluation["identity_rows"],
        "schedule_sha256": training_result["audit"]["schedule_sha256"],
        "initial_state_sha256": training_result["audit"]["initial_state_sha256"],
        "r2_state_unchanged": r2_state_before == r2_state_after,
        "checkpoint_replay": "bitwise_exact",
        "gradient_audit_authorization": dict(gradient_audit_authorization),
        "confirmation_authorization": confirmation_authorization,
        "smoke_authorization": None
        if smoke_authorization is None
        else dict(smoke_authorization),
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
        raise DataValidationError("pilot phase lacks an audited success seal")
    success = _read_json(success_path)
    if (
        success.get("status") != "complete"
        or success.get("phase_summary_sha256") != file_sha256(root / "phase_summary.json")
        or success.get("artifact_hashes_sha256") != file_sha256(root / "artifact_hashes.json")
        or _read_json(root / "artifact_hashes.json") != _artifact_hashes(root)
    ):
        raise DataValidationError("pilot phase success seal is invalid")
    summary = _read_json(root / "phase_summary.json")
    gate = _read_json(root / "pilot_gate.json")
    selected = gate.get("selected_method")
    smoke_authorization = summary.get("smoke_authorization")
    if not isinstance(smoke_authorization, Mapping):
        raise DataValidationError("pilot summary lacks smoke authorization")
    if dict(smoke_authorization) != _read_authorized_smoke(
        str(smoke_authorization["smoke_root"]), config
    ):
        raise DataValidationError("pilot smoke authorization changed")
    if (
        summary.get("phase") != "pilot"
        or summary.get("config_sha256") != config["_config_sha256"]
        or summary.get("gate") != gate
        or gate.get("confirmation_allowed") is not True
        or selected not in DECOUPLED_METHODS
        or summary.get("final_unknown_used") is not False
    ):
        raise DataValidationError("audited pilot does not authorize confirmation")
    return {
        "pilot_root": str(root),
        "pilot_gate_sha256": file_sha256(root / "pilot_gate.json"),
        "selected_method": str(selected),
        "decision": str(gate["signal"]),
        "smoke_authorization": dict(smoke_authorization),
    }


def _read_authorized_gradient_audit(
    gradient_audit_root: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(gradient_audit_root).resolve()
    summary = audit_gradient_audit_phase(root, config=config)
    if (
        summary.get("status") != "complete"
        or summary.get("unit_count") != 2
        or summary.get("stage_b_allowed") is not True
        or summary.get("final_unknown_test_authorized") is not False
    ):
        raise DataValidationError("gradient audit does not authorize stage B")
    return {
        "gradient_audit_root": str(root),
        "phase_success_sha256": file_sha256(root / "_PHASE_SUCCESS.json"),
        "summary_sha256": file_sha256(root / "gradient_pathology_audit_summary.json"),
        "stage_b_allowed": True,
        "final_unknown_test_authorized": False,
    }


def _read_authorized_smoke(
    smoke_root: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(smoke_root).resolve()
    success_path = root / "_PHASE_SUCCESS.json"
    if not success_path.is_file():
        raise DataValidationError("smoke phase lacks an audited success seal")
    success = _read_json(success_path)
    if (
        success.get("status") != "complete"
        or success.get("phase_summary_sha256") != file_sha256(root / "phase_summary.json")
        or success.get("artifact_hashes_sha256") != file_sha256(root / "artifact_hashes.json")
        or _read_json(root / "artifact_hashes.json") != _artifact_hashes(root)
    ):
        raise DataValidationError("smoke phase success seal is invalid")
    summary = _read_json(root / "phase_summary.json")
    if (
        summary.get("phase") != "smoke"
        or summary.get("unit_count") != 2
        or summary.get("config_sha256") != config["_config_sha256"]
        or summary.get("decision") != "diagnostic_smoke_only"
        or summary.get("final_unknown_test_authorized") is not False
        or summary.get("final_unknown_used") is not False
    ):
        raise DataValidationError("audited smoke does not authorize pilot")
    return {
        "smoke_root": str(root),
        "phase_success_sha256": file_sha256(success_path),
        "phase_summary_sha256": file_sha256(root / "phase_summary.json"),
        "status": "passed",
        "final_unknown_test_authorized": False,
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
    gradient_audit_root: str | Path,
    device_request: str = "auto",
    pilot_root: str | Path | None = None,
    smoke_root: str | Path | None = None,
) -> dict[str, Any]:
    config = load_fg_mv_cssr_decoupled_config(config_path)
    gradient_audit_authorization = _read_authorized_gradient_audit(
        gradient_audit_root, config
    )
    authorization = None
    smoke_authorization = None
    selected_method = None
    if phase == "confirmation":
        if pilot_root is None:
            raise DataValidationError("confirmation requires an audited pilot root")
        authorization = _read_authorized_pilot(pilot_root, config)
        selected_method = str(authorization["selected_method"])
        if smoke_root is not None:
            raise DataValidationError("confirmation inherits smoke authorization from pilot")
    elif phase == "pilot":
        if smoke_root is None:
            raise DataValidationError("pilot requires an audited smoke root")
        smoke_authorization = _read_authorized_smoke(smoke_root, config)
        if pilot_root is not None:
            raise DataValidationError("pilot-root only applies to confirmation")
    elif phase == "smoke":
        if smoke_root is not None or pilot_root is not None:
            raise DataValidationError("smoke cannot consume later-phase authorization")
    elif pilot_root is not None:
        raise DataValidationError("pilot-root only applies to confirmation")
    plan = build_phase_plan(phase, selected_method=selected_method)
    if (pair_id, method) not in {
        (str(unit["pair_id"]), str(unit["method"])) for unit in plan
    }:
        raise DataValidationError("unit is outside the frozen phase plan")
    root = Path(phase_root).resolve()
    destination = _unit_destination(root, pair_id, method)
    if destination.exists():
        raise DataValidationError(f"decoupled CSSR output already exists: {destination}")
    staging = destination.parent / f".{method}.staging"
    if staging.exists():
        raise DataValidationError(f"stale decoupled staging output exists: {staging}")
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
    if r2_model.training or any(parameter.requires_grad for parameter in r2_model.parameters()):
        raise DataValidationError("R2 is not frozen in eval mode")
    r2_state_before = _state_sha256(r2_model.state_dict())
    unique_rows = build_unique_base_sample_manifest(prepared, bundle)
    unique_features, feature_audit = extract_frozen_feature_maps(
        model=r2_model,
        bundle=bundle,
        prepared=prepared,
        rows=unique_rows,
        device=device,
        batch_size=int(config["training"]["batch_size"]),
    )
    train_rows, train_indices = _selected_unique_rows(unique_rows, "train_known")
    cal_rows, cal_indices = _selected_unique_rows(unique_rows, "known_calibration")
    training_result = train_decoupled_cssr(
        train_rows=train_rows,
        train_features=unique_features[train_indices],
        calibration_features=unique_features[cal_indices],
        calibration_labels=np.asarray([int(row["model_label"]) for row in cal_rows]),
        pair_id=pair_id,
        method=method,
        config=config,
        device=device,
        smoke=phase == "smoke",
    )
    evaluation = evaluate_decoupled_unit(
        model=training_result["model"],
        method=method,
        prepared=prepared,
        unique_rows=unique_rows,
        unique_features=unique_features,
        frozen_r2_arrays=frozen_r2_arrays,
        pair_id=pair_id,
        config=config,
        device=device,
        smoke=phase == "smoke",
    )
    r2_state_after = _state_sha256(r2_model.state_dict())
    if r2_state_after != r2_state_before:
        raise DataValidationError("frozen R2 parameters or buffers changed")
    if task_source_hashes(project_root) != source_hashes:
        raise DataValidationError("task source changed while decoupled unit was running")
    environment = _git_environment(project_root, device)
    environment["runtime_contract"] = runtime_contract
    environment["task_source_hashes"] = source_hashes
    environment["feature_audit"] = feature_audit
    summary = save_unit_result(
        staging,
        phase=phase,
        pair_id=pair_id,
        method=method,
        config=config,
        prepared=prepared,
        unique_rows=unique_rows,
        unique_features=unique_features,
        training_result=training_result,
        evaluation=evaluation,
        r2_audit=r2_audit,
        r2_state_before=r2_state_before,
        r2_state_after=r2_state_after,
        runtime_contract=runtime_contract,
        environment=environment,
        source_hashes=source_hashes,
        confirmation_authorization=authorization,
        gradient_audit_authorization=gradient_audit_authorization,
        smoke_authorization=smoke_authorization,
        wall_time_seconds=time.perf_counter() - started,
    )
    staging.replace(destination)
    return {**summary, "destination": str(destination)}


def _frozen_class_orders(
    config: Mapping[str, Any], pair_id: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    source_order = tuple(str(value) for value in config["classes"]["source_known_order"])
    matches = [
        row
        for row in config["classes"]["identity_pairs"]
        if str(row["pair_id"]) == pair_id
    ]
    if len(matches) != 1:
        raise DataValidationError(f"pair {pair_id} is outside the frozen identity plan")
    spec = matches[0]
    train_order = tuple(source_order[int(index)] for index in spec["train_known_indices"])
    surrogate_order = tuple(
        source_order[int(index)] for index in spec["surrogate_unknown_indices"]
    )
    return train_order, surrogate_order


def _rebuild_evaluation_manifest(
    source_rows: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    phase: str,
    pair_id: str,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Rebuild the exact evaluation population from the frozen R2 manifest."""

    if phase not in {"smoke", "pilot", "confirmation"}:
        raise DataValidationError(f"unsupported decoupled phase: {phase}")
    train_order, surrogate_order = _frozen_class_orders(config, pair_id)
    class_orders = {
        "known_calibration": train_order,
        "surrogate_unknown": surrogate_order,
    }
    full_per_class = int(config["data"]["evaluation_pairs_per_class"])
    selected_per_class = (
        int(config["data"]["smoke"]["evaluation_pairs_per_class"])
        if phase == "smoke"
        else full_per_class
    )
    rebuilt: list[dict[str, Any]] = []
    indices_by_role: dict[str, np.ndarray] = {}
    for role in ("known_calibration", "surrogate_unknown"):
        role_rows = [
            dict(row) for row in source_rows if str(row.get("experiment_role")) == role
        ]
        grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        for source_index, row in enumerate(role_rows):
            grouped.setdefault(str(row["class_name"]), []).append((source_index, row))
        if tuple(grouped) != class_orders[role] or any(
            len(grouped[class_name]) != full_per_class
            for class_name in class_orders[role]
        ):
            raise DataValidationError(f"frozen R2 {role} population changed")
        selected = [
            value
            for class_name in class_orders[role]
            for value in grouped[class_name][:selected_per_class]
        ]
        indices_by_role[role] = np.asarray(
            [source_index for source_index, _ in selected], dtype=np.int64
        )
        for local_index, (_, row) in enumerate(selected):
            rebuilt.append(
                {
                    **row,
                    "evaluation_subset_role": role,
                    "evaluation_subset_index": local_index,
                }
            )
    if any(
        int(row[f"view{view}_angle_deg"]) % 2 == 0
        for row in rebuilt
        for view in (1, 2)
    ):
        raise DataValidationError("evaluation manifest contains an even-angle sample")
    return rebuilt, indices_by_role


def _audit_frozen_r2_evaluation_binding(
    root: Path,
    *,
    config: Mapping[str, Any],
    phase: str,
    pair_id: str,
    method: str,
    method_rows: Sequence[Mapping[str, Any]],
    d0_rows: Sequence[Mapping[str, Any]],
    reference_arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Bind Stage-B evaluation rows and logits to the immutable prior-R2 outputs."""

    r2_audit = _read_json(root / "r2_reference_audit.json")
    expected_hashes = {
        str(name): str(value)
        for name, value in config["prior_r2"]["unit_artifact_hashes"][pair_id].items()
    }
    train_order, surrogate_order = _frozen_class_orders(config, pair_id)
    expected_exact_keys = {
        f"{role}_{name}"
        for role in ("train", "known_calibration", "surrogate_unknown")
        for name in (
            "per_view_features",
            "fused_features",
            "per_view_logits",
            "global_logits",
            "unknown_score",
            "labels",
        )
    }
    exact_checks = r2_audit.get("old_output_exact_checks")
    maximum_errors = r2_audit.get("old_output_maximum_absolute_errors")
    if (
        r2_audit.get("status") != "passed"
        or r2_audit.get("pair_id") != pair_id
        or r2_audit.get("prior_formal_code_commit")
        != config["prior_r2"]["formal_code_commit"]
        or r2_audit.get("expected_unit_artifact_hashes") != expected_hashes
        or r2_audit.get("strict_load") is not True
        or r2_audit.get("all_parameters_frozen") is not True
        or r2_audit.get("arpl_module_instantiated") is not False
        or r2_audit.get("old_outputs_exact") is not True
        or set(exact_checks or {}) != expected_exact_keys
        or not all(value is True for value in (exact_checks or {}).values())
        or set(maximum_errors or {}) != expected_exact_keys
        or any(float(value) != 0.0 for value in (maximum_errors or {}).values())
        or r2_audit.get("train_class_order") != list(train_order)
        or r2_audit.get("surrogate_class_order") != list(surrogate_order)
        or r2_audit.get("final_unknown_used") is not False
        or r2_audit.get("even_angle_test_used") is not False
    ):
        raise DataValidationError("frozen R2 reference audit changed")

    prior_unit_root = Path(str(r2_audit.get("unit_root", ""))).resolve()
    relative = Path(
        str(config["prior_r2"]["unit_relative_template"]).format(pair_id=pair_id)
    )
    if relative.is_absolute() or tuple(prior_unit_root.parts[-len(relative.parts) :]) != relative.parts:
        raise DataValidationError("frozen R2 unit path is outside its configured layout")
    prior_checkpoint = prior_unit_root / "checkpoint.pt"
    prior_manifest = prior_unit_root / "pair_manifest.csv"
    prior_outputs = prior_unit_root / "features_logits_scores.npz"
    if Path(str(r2_audit.get("checkpoint_path", ""))).resolve() != prior_checkpoint:
        raise DataValidationError("frozen R2 checkpoint path changed")
    for name, path in (
        ("checkpoint.pt", prior_checkpoint),
        ("pair_manifest.csv", prior_manifest),
        ("features_logits_scores.npz", prior_outputs),
    ):
        if not path.is_file() or file_sha256(path) != expected_hashes[name]:
            raise DataValidationError(f"frozen R2 bound artifact changed: {name}")
    if (
        r2_audit.get("checkpoint_sha256") != expected_hashes["checkpoint.pt"]
        or r2_audit.get("pair_manifest_sha256") != expected_hashes["pair_manifest.csv"]
    ):
        raise DataValidationError("frozen R2 reference hashes changed")

    results_root = prior_unit_root
    for _ in relative.parts:
        results_root = results_root.parent
    root_hash_manifest = results_root / "artifact_hashes.json"
    if (
        not root_hash_manifest.is_file()
        or file_sha256(root_hash_manifest)
        != config["prior_r2"]["root_artifact_hash_manifest_sha256"]
        or r2_audit.get("root_artifact_hash_manifest_sha256")
        != config["prior_r2"]["root_artifact_hash_manifest_sha256"]
    ):
        raise DataValidationError("frozen R2 root artifact binding changed")
    root_hashes = _read_json(root_hash_manifest)
    for name, expected in expected_hashes.items():
        if root_hashes.get(str(relative / name)) != expected:
            raise DataValidationError(f"frozen R2 root binding changed: {name}")
    prior_hash_manifest = prior_unit_root / "artifact_hashes.json"
    if (
        not prior_hash_manifest.is_file()
        or file_sha256(prior_hash_manifest)
        != r2_audit.get("artifact_hash_manifest_sha256")
    ):
        raise DataValidationError("frozen R2 unit artifact manifest changed")
    prior_hashes = _read_json(prior_hash_manifest)
    if int(r2_audit.get("artifact_count", -1)) != len(prior_hashes):
        raise DataValidationError("frozen R2 artifact count changed")
    for name, expected in prior_hashes.items():
        path = prior_unit_root / str(name)
        if not path.is_file() or file_sha256(path) != str(expected):
            raise DataValidationError(f"frozen R2 unit artifact changed: {name}")

    source_bytes = (root / "source_pair_manifest.csv").read_bytes()
    if source_bytes != prior_manifest.read_bytes():
        raise DataValidationError("source pair manifest is not the frozen R2 manifest")
    source_rows = _read_csv(prior_manifest)
    expected_evaluation, indices_by_role = _rebuild_evaluation_manifest(
        source_rows,
        config=config,
        phase=phase,
        pair_id=pair_id,
    )
    if (root / "evaluation_pair_manifest.csv").read_bytes() != _render_csv(
        expected_evaluation
    ):
        raise DataValidationError("evaluation pair manifest does not derive from frozen R2")
    evaluation_by_role = {
        role: [
            row
            for row in expected_evaluation
            if str(row["evaluation_subset_role"]) == role
        ]
        for role in ("known_calibration", "surrogate_unknown")
    }

    with np.load(prior_outputs, allow_pickle=False) as sealed:
        missing = sorted(expected_exact_keys - set(sealed.files))
        if missing:
            raise DataValidationError(f"frozen R2 output lacks arrays: {missing}")
        role_logits = {
            role: np.asarray(sealed[f"{role}_global_logits"])[indices_by_role[role]]
            for role in ("known_calibration", "surrogate_unknown")
        }
        role_labels = {
            role: np.asarray(sealed[f"{role}_labels"], dtype=np.int64)[
                indices_by_role[role]
            ]
            for role in ("known_calibration", "surrogate_unknown")
        }
        full_calibration_logits = np.asarray(sealed["known_calibration_global_logits"])
        full_calibration_labels = np.asarray(
            sealed["known_calibration_labels"], dtype=np.int64
        )
    if (
        not np.array_equal(
            np.asarray(reference_arrays["full_calibration_logits"]),
            full_calibration_logits,
        )
        or not np.array_equal(
            np.asarray(reference_arrays["full_calibration_labels"], dtype=np.int64),
            full_calibration_labels,
        )
        or tuple(reference_arrays["full_calibration_pair_ids"].tolist())
        != tuple(
            row["pair_id"]
            for row in source_rows
            if row["experiment_role"] == "known_calibration"
        )
    ):
        raise DataValidationError("reference scores are not bound to frozen R2 outputs")

    if len(method_rows) != len(expected_evaluation) or len(d0_rows) != len(
        expected_evaluation
    ):
        raise DataValidationError("prediction population does not match evaluation manifest")
    role_offsets = {"known_calibration": 0, "surrogate_unknown": 0}
    for evaluation_row, method_row, d0_row in zip(
        expected_evaluation, method_rows, d0_rows, strict=True
    ):
        role = str(evaluation_row["evaluation_subset_role"])
        local_index = role_offsets[role]
        role_offsets[role] += 1
        manifest_bindings = {
            "pair_id": "pair_id",
            "class_name": "class_name",
            "view1_sample_id": "view1_sample_id",
            "view2_sample_id": "view2_sample_id",
            "view1_angle_deg": "view1_angle_deg",
            "view2_angle_deg": "view2_angle_deg",
            "view1_frame_id": "view1_frame_id",
            "view2_frame_id": "view2_frame_id",
        }
        for prediction_row in (method_row, d0_row):
            if str(prediction_row.get("evaluation_role")) != role or any(
                str(prediction_row.get(prediction_name))
                != str(evaluation_row[manifest_name])
                for prediction_name, manifest_name in manifest_bindings.items()
            ):
                raise DataValidationError("prediction rows do not match evaluation manifest")
            if int(prediction_row["true_label"]) != int(evaluation_row["model_label"]):
                raise DataValidationError("prediction label does not match evaluation manifest")
        if method_row.get("method") != method or d0_row.get("method") != D0_R2_CLASS_CONDITIONAL_MLS:
            raise DataValidationError("prediction method label changed")
        expected_logits = np.asarray(role_logits[role][local_index])
        for prediction_row in (method_row, d0_row):
            observed_logits = np.asarray(
                json.loads(str(prediction_row["fused_logits"])), dtype=expected_logits.dtype
            )
            if not np.array_equal(observed_logits, expected_logits):
                raise DataValidationError(
                    "prediction logits are not the frozen R2 outputs"
                )
            expected_prediction = int(expected_logits.argmax())
            if int(prediction_row["predicted_known_label"]) != expected_prediction:
                raise DataValidationError("prediction is not frozen R2 argmax")
        if int(role_labels[role][local_index]) != int(evaluation_row["model_label"]):
            raise DataValidationError("frozen R2 labels do not match its pair manifest")

    return {
        "role_logits": role_logits,
        "role_pair_rows": evaluation_by_role,
        "audit": {
            "status": "passed",
            "r2_reference_audit_sha256": file_sha256(root / "r2_reference_audit.json"),
            "evaluation_pair_manifest_sha256": file_sha256(
                root / "evaluation_pair_manifest.csv"
            ),
            "prior_pair_manifest_sha256": expected_hashes["pair_manifest.csv"],
            "prior_features_logits_scores_sha256": expected_hashes[
                "features_logits_scores.npz"
            ],
            "prediction_rows_bound_to_prior_r2": True,
            "final_unknown_used": False,
            "even_angle_test_used": False,
        },
    }


def audit_unit_result(
    unit_root: str | Path,
    *,
    config: Mapping[str, Any],
    phase: str,
    pair_id: str,
    method: str,
    device_request: str = "auto",
) -> dict[str, Any]:
    root = Path(unit_root).resolve()
    success = _read_json(root / "_SUCCESS.json")
    if (
        success.get("status") != "complete"
        or success.get("unit_summary_sha256") != file_sha256(root / "unit_summary.json")
        or success.get("artifact_hashes_sha256") != file_sha256(root / "artifact_hashes.json")
        or _read_json(root / "artifact_hashes.json") != _artifact_hashes(root)
    ):
        raise DataValidationError("decoupled unit artifact hash audit failed")
    expected_epochs = 6 if phase == "smoke" else 20
    unique_rows = _read_csv(root / "unique_base_sample_manifest.csv")
    train_rows = [row for row in unique_rows if row["experiment_role"] == "train_known"]
    schedule_hashes: list[str] = []
    for epoch in range(1, expected_epochs + 1):
        indices, audit = build_single_view_schedule(
            train_rows,
            pair_id=pair_id,
            angle_fold=ANGLE_FOLD,
            epoch=epoch,
            cssr_seed=CSSR_SEED,
        )
        rows = [
            {
                "schedule_index": schedule_index,
                "source_row_index": int(source_index),
                "model_label": int(train_rows[int(source_index)]["model_label"]),
                "sample_id": str(train_rows[int(source_index)]["sample_id"]),
            }
            for schedule_index, source_index in enumerate(indices)
        ]
        if (
            (root / "single_view_schedules" / f"epoch_{epoch:03d}.csv").read_bytes()
            != _render_csv(rows)
            or _read_json(root / "single_view_schedule_audits" / f"epoch_{epoch:03d}.json")
            != audit
        ):
            raise DataValidationError(f"single-view schedule {epoch} does not reproduce")
        schedule_hashes.append(str(audit["schedule_sha256"]))
    training_audit = _read_json(root / "training_audit.json")
    if (
        training_audit.get("status") != "passed"
        or int(training_audit.get("epochs", -1)) != expected_epochs
        or training_audit.get("schedule_sha256") != _sequence_sha256(schedule_hashes)
        or training_audit.get("all_parameters_finite") is not True
        or training_audit.get("pair_multiplicity_weight") is not False
        or training_audit.get("surrogate_unknown_used_for_training") is not False
        or training_audit.get("final_unknown_used") is not False
    ):
        raise DataValidationError("decoupled training audit changed")
    contract = _read_json(root / "unit_contract.json")
    for key, expected in {
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "mode": "smoke" if phase == "smoke" else "full",
        "pair_id": pair_id,
        "method": method,
        "angle_fold": ANGLE_FOLD,
        "r2_seed": R2_SEED,
        "cssr_seed": CSSR_SEED,
        "config_sha256": config["_config_sha256"],
        "known_prediction_source": "frozen_r2_fused_logits_argmax",
        "main_score": "fusion_guided_decoupled_cssr",
        "threshold_source": "known_calibration_only",
        "train_unique_single_view_bases": 720,
        "pair_multiplicity_weight": False,
        "r2_retrained_or_finetuned": False,
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
    }.items():
        if contract.get(key) != expected:
            raise DataValidationError(f"unit contract changed: {key}")
    project_root = Path(config["_config_path"]).parents[3]
    current_source_hashes = task_source_hashes(project_root)
    if contract.get("source_hashes") != current_source_hashes:
        raise DataValidationError("unit source hash binding changed")
    environment = _read_json(root / "environment.json")
    if environment.get("task_source_hashes") != current_source_hashes:
        raise DataValidationError("environment source hash binding changed")
    gradient_authorization = _mapping(
        contract.get("gradient_audit_authorization"),
        "gradient audit authorization",
    )
    observed_gradient_authorization = _read_authorized_gradient_audit(
        str(gradient_authorization["gradient_audit_root"]), config
    )
    if dict(gradient_authorization) != observed_gradient_authorization:
        raise DataValidationError("gradient-audit authorization binding changed")
    smoke_authorization = contract.get("smoke_authorization")
    if phase == "pilot":
        if not isinstance(smoke_authorization, Mapping):
            raise DataValidationError("pilot unit lacks smoke authorization")
        if dict(smoke_authorization) != _read_authorized_smoke(
            str(smoke_authorization["smoke_root"]), config
        ):
            raise DataValidationError("smoke authorization binding changed")
    elif smoke_authorization is not None:
        raise DataValidationError("non-pilot unit contains direct smoke authorization")
    confirmation_authorization = contract.get("confirmation_authorization")
    if phase == "confirmation":
        if not isinstance(confirmation_authorization, Mapping):
            raise DataValidationError("confirmation unit lacks pilot authorization")
        if dict(confirmation_authorization) != _read_authorized_pilot(
            str(confirmation_authorization["pilot_root"]), config
        ):
            raise DataValidationError("pilot authorization binding changed")
    elif confirmation_authorization is not None:
        raise DataValidationError("non-confirmation unit contains pilot authorization")
    frozen_audit = _read_json(root / "r2_frozen_state_audit.json")
    if (
        frozen_audit.get("unchanged") is not True
        or frozen_audit.get("before_sha256") != frozen_audit.get("after_sha256")
        or frozen_audit.get("all_parameters_requires_grad_false") is not True
        or frozen_audit.get("eval_mode") is not True
    ):
        raise DataValidationError("frozen R2 state audit changed")

    training_rows = [
        json.loads(line)
        for line in (root / "training_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(training_rows) != expected_epochs:
        raise DataValidationError("training log epoch count changed")
    for epoch, row in enumerate(training_rows, start=1):
        trainable = epoch >= 6
        factor = _learning_rate_factor(epoch, warmup_epochs=2, total_epochs=20)
        expected_lrs = {
            "adapter": 3.0e-4 * factor if trainable else 0.0,
            "autoencoders": 1.0e-3 * factor,
        }
        observed_lrs = {key: float(value) for key, value in row["learning_rates"].items()}
        if (
            int(row["epoch"]) != epoch
            or str(row["method"]) != method
            or bool(row["adapter_trainable"]) != trainable
            or float(row["learning_rate_factor"]) != factor
            or observed_lrs != expected_lrs
            or int(row["sample_count"]) != 720
            or str(row["sample_schedule_sha256"]) != schedule_hashes[epoch - 1]
            or row["known_calibration_used_for_training"] is not False
            or row["surrogate_unknown_used_for_training"] is not False
            or row["final_unknown_used"] is not False
        ):
            raise DataValidationError(f"training log contract changed at epoch {epoch}")
        relative = float(row["train_relative_loss"])
        absolute = float(row["train_absolute_loss_diagnostic"])
        separation = float(row["train_separation_loss_diagnostic"])
        expected_total = (
            relative
            if method == D1_DECOUPLED_REL_CSSR
            else relative + 0.25 * absolute + 0.5 * separation
        )
        if not math.isclose(
            float(row["train_total_loss"]), expected_total, rel_tol=1.0e-6, abs_tol=1.0e-7
        ):
            raise DataValidationError(f"loss composition changed at epoch {epoch}")
        numeric_fields = (
            "train_total_loss",
            "train_relative_loss",
            "train_absolute_loss_diagnostic",
            "train_separation_loss_diagnostic",
            "train_cssr_accuracy",
            "mean_adapter_gradient_norm",
            "mean_autoencoder_gradient_norm",
            "mean_total_gradient_norm",
            "clipping_batch_fraction",
            "adapter_parameter_relative_update",
            "autoencoder_parameter_relative_update",
        )
        if any(not math.isfinite(float(row[name])) for name in numeric_fields):
            raise DataValidationError("training diagnostics contain NaN or Inf")
        if epoch <= 5 and (
            float(row["mean_adapter_gradient_norm"]) != 0.0
            or float(row["adapter_parameter_relative_update"]) != 0.0
        ):
            raise DataValidationError("adapter changed during its frozen five epochs")
    if expected_epochs >= 6 and (
        float(training_rows[5]["mean_adapter_gradient_norm"]) <= 0.0
        or float(training_rows[5]["adapter_parameter_relative_update"]) <= 0.0
    ):
        raise DataValidationError("adapter did not receive an update after epoch 5")
    checkpoint = torch.load(root / "checkpoint.pt", map_location="cpu", weights_only=False)
    for key, expected in {
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "pair_id": pair_id,
        "method": method,
        "checkpoint_epoch": expected_epochs,
        "formal_checkpoint": phase != "smoke",
        "checkpoint_selection": "fixed_final_epoch",
        "cssr_seed": CSSR_SEED,
        "config_sha256": config["_config_sha256"],
        "final_unknown_used": False,
        "even_angle_test_used": False,
    }.items():
        if checkpoint.get(key) != expected:
            raise DataValidationError(f"checkpoint metadata changed: {key}")
    if hashlib.sha256((root / "unique_base_sample_manifest.csv").read_bytes()).hexdigest() != checkpoint["unique_base_manifest_sha256"]:
        raise DataValidationError("unique base manifest binding changed")
    if file_sha256(root / "source_pair_manifest.csv") != checkpoint["source_pair_manifest_sha256"]:
        raise DataValidationError("source pair manifest binding changed")
    if checkpoint.get("r2_checkpoint_sha256") != config["prior_r2"]["unit_artifact_hashes"][pair_id]["checkpoint.pt"]:
        raise DataValidationError("R2 checkpoint binding changed")
    if checkpoint.get("r2_state_sha256") != frozen_audit["after_sha256"]:
        raise DataValidationError("R2 state hash binding changed")
    device = _resolve_device(device_request)
    _configure_runtime(config, device)
    _set_determinism(CSSR_SEED, bool(config["runtime"]["deterministic_algorithms"]))
    restored = FGMVCSSRDecoupled1D(
        num_classes=5,
        input_channels=128,
        latent_channels=32,
        residual_scale=0.1,
        gamma=0.1,
        clip_length=100.0,
        epsilon=1.0e-8,
        margin=0.2,
    ).to(device)
    if _state_sha256(restored.state_dict()) != training_audit["initial_state_sha256"]:
        raise DataValidationError("decoupled initial state does not reproduce")
    incompatible = restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise DataValidationError("checkpoint strict load failed")
    with np.load(root / "checkpoint_replay.npz", allow_pickle=False) as saved:
        features = np.asarray(saved["unique_features"], dtype=np.float32)
        if _array_sha256(features) != checkpoint["unique_feature_map_sha256"]:
            raise DataValidationError("unique feature-map binding changed")
        replay = _infer_decoupled(
            restored,
            features,
            device=device,
            batch_size=int(config["training"]["batch_size"]),
        )
        for name in ("u", "r", "probabilities"):
            if not np.array_equal(replay[name], saved[f"expected_{name}"]):
                raise DataValidationError(f"checkpoint replay changed: {name}")
    if _state_sha256(restored.state_dict()) != training_audit["final_state_sha256"]:
        raise DataValidationError("decoupled final state hash changed")
    with np.load(root / "reference_scores.npz", allow_pickle=False) as saved:
        reference_arrays = {name: saved[name] for name in saved.files}
    if not np.array_equal(
        np.asarray(reference_arrays["r"], dtype=np.float64),
        replay["r"].astype(np.float64),
    ):
        raise DataValidationError("reference r values are not bound to checkpoint replay")
    rebuilt_arrays, rebuilt_references, rebuilt_ids, rebuilt_metadata = (
        build_guided_reference_scores(
            unique_rows,
            np.asarray(reference_arrays["r"], dtype=np.float64),
            epsilon=float(config["calibration"]["score_epsilon"]),
        )
    )
    for name, values in rebuilt_arrays.items():
        if not np.array_equal(values, reference_arrays[name]):
            raise DataValidationError(f"reference score does not reproduce: {name}")
    for index, values in enumerate(rebuilt_references):
        if not np.array_equal(values, reference_arrays[f"class_{index}_reference_r"]):
            raise DataValidationError("class reference distribution changed")
    reference_metadata = _read_json(root / "reference_distribution.json")
    if (
        reference_metadata.get("status") != "passed"
        or reference_metadata.get("reference_counts") != [36] * 5
        or reference_metadata.get("reference_sample_ids")
        != [list(values) for values in rebuilt_ids]
        or reference_metadata.get("calibration_leave_one_base_sample_out") is not True
        or reference_metadata.get("surrogate_unknown_in_reference") is not False
    ):
        raise DataValidationError("reference metadata changed")
    acceptance = float(config["calibration"]["threshold_known_acceptance_rate"])
    method_rows = _read_csv(root / "predictions_and_scores.csv")
    d0_rows = _read_csv(root / "d0_predictions_and_scores.csv")
    if len(method_rows) != len(d0_rows) or not method_rows:
        raise DataValidationError("method and D0 prediction populations differ")
    r2_binding = _audit_frozen_r2_evaluation_binding(
        root,
        config=config,
        phase=phase,
        pair_id=pair_id,
        method=method,
        method_rows=method_rows,
        d0_rows=d0_rows,
        reference_arrays=reference_arrays,
    )
    role_method_rows = {
        role: [row for row in method_rows if row["evaluation_role"] == role]
        for role in ("known_calibration", "surrogate_unknown")
    }
    role_logits = {
        role: np.asarray(values, dtype=np.float64)
        for role, values in r2_binding["role_logits"].items()
    }
    role_pair_rows = r2_binding["role_pair_rows"]
    cc_mls = _class_conditional_mls_for_roles(
        full_calibration_logits=np.asarray(reference_arrays["full_calibration_logits"]),
        full_calibration_labels=np.asarray(reference_arrays["full_calibration_labels"]),
        full_calibration_pair_ids=tuple(reference_arrays["full_calibration_pair_ids"].tolist()),
        role_logits=role_logits,
        role_pair_rows=role_pair_rows,
    )
    score_by_sample = rebuilt_metadata["score_by_sample"]
    r_by_sample = rebuilt_metadata["r_by_sample"]
    p_by_sample = rebuilt_metadata["p_by_sample"]
    role_offsets = {"known_calibration": 0, "surrogate_unknown": 0}
    for method_row, d0_row in zip(method_rows, d0_rows, strict=True):
        role = str(method_row["evaluation_role"])
        local_index = role_offsets[role]
        role_offsets[role] += 1
        if any(
            method_row[name] != d0_row[name]
            for name in (
                "pair_id",
                "evaluation_role",
                "class_name",
                "true_label",
                "predicted_known_label",
                "predicted_known_class_name",
                "fused_logits",
                "view1_sample_id",
                "view2_sample_id",
            )
        ):
            raise DataValidationError("D0 and decoupled rows do not share frozen R2 outputs")
        logits = np.asarray(json.loads(method_row["fused_logits"]), dtype=np.float64)
        prediction = int(logits.argmax())
        if prediction != int(method_row["predicted_known_label"]):
            raise DataValidationError("guided score does not use frozen R2 y_hat")
        view_scores: list[np.ndarray] = []
        for view in (1, 2):
            sample_id = str(method_row[f"view{view}_sample_id"])
            observed_r = np.asarray(json.loads(method_row[f"view{view}_r"]), dtype=np.float64)
            observed_p = np.asarray(json.loads(method_row[f"view{view}_p_value"]), dtype=np.float64)
            observed_a = np.asarray(json.loads(method_row[f"view{view}_a"]), dtype=np.float64)
            if (
                not np.array_equal(observed_r, r_by_sample[sample_id])
                or not np.array_equal(observed_p, p_by_sample[sample_id])
                or not np.array_equal(observed_a, score_by_sample[sample_id])
            ):
                raise DataValidationError("prediction row r/p/a does not reproduce")
            view_scores.append(observed_a)
        guided = 0.5 * (view_scores[0][prediction] + view_scores[1][prediction])
        if not math.isclose(
            float(method_row["unknown_score"]), float(guided), rel_tol=0.0, abs_tol=1.0e-15
        ):
            raise DataValidationError("guided pair score does not reproduce")
        if not math.isclose(
            float(method_row["diagnostic_class_conditional_mls"]),
            float(cc_mls[role][local_index]),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise DataValidationError("diagnostic class-conditional MLS changed")
        if not math.isclose(
            float(d0_row["unknown_score"]),
            float(cc_mls[role][local_index]),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise DataValidationError("D0 class-conditional MLS does not reproduce")
    method_metrics = _read_json(root / "metrics.json")
    d0_metrics = _read_json(root / "d0_metrics.json")
    global_mls_metrics = {
        key: float(value)
        for key, value in evaluate_open_set(
            known_true=np.asarray(
                [int(row["true_label"]) for row in role_method_rows["known_calibration"]],
                dtype=np.int64,
            ),
            known_pred=role_logits["known_calibration"].argmax(axis=1),
            known_unknown_scores=-role_logits["known_calibration"].max(axis=1),
            unknown_pred=role_logits["surrogate_unknown"].argmax(axis=1),
            unknown_unknown_scores=-role_logits["surrogate_unknown"].max(axis=1),
            known_validation_scores=-role_logits["known_calibration"].max(axis=1),
            known_class_count=5,
            known_acceptance_rate=acceptance,
        ).items()
    }
    if global_mls_metrics != _read_json(root / "global_mls_background_metrics.json"):
        raise DataValidationError("global MLS background metrics do not reproduce")
    _metrics_exact(
        method_metrics,
        recompute_method_metrics_from_prediction_rows(method_rows, known_acceptance_rate=acceptance),
        context=method,
    )
    _metrics_exact(
        d0_metrics,
        recompute_method_metrics_from_prediction_rows(d0_rows, known_acceptance_rate=acceptance),
        context=D0_R2_CLASS_CONDITIONAL_MLS,
    )
    method_identity, method_absorption, method_error = build_identity_and_absorption_rows(
        method_rows,
        method=method,
        pair_id=pair_id,
        train_class_order=tuple(checkpoint["train_class_order"]),
        acceptance_rate=acceptance,
    )
    d0_identity, d0_absorption, d0_error = build_identity_and_absorption_rows(
        d0_rows,
        method=D0_R2_CLASS_CONDITIONAL_MLS,
        pair_id=pair_id,
        train_class_order=tuple(checkpoint["train_class_order"]),
        acceptance_rate=acceptance,
    )
    identity_rows = [*method_identity, *d0_identity]
    absorption_rows = [*method_absorption, *d0_absorption]
    if (root / "identity_metrics.csv").read_bytes() != _render_csv(identity_rows):
        raise DataValidationError("identity metrics do not reproduce")
    if (root / "absorption_by_known_class.csv").read_bytes() != _render_csv(absorption_rows):
        raise DataValidationError("absorption analysis does not reproduce")
    if _read_json(root / "error_analysis.json") != {"method": method_error, "d0": d0_error}:
        raise DataValidationError("identity error analysis does not reproduce")
    shared = _read_json(root / "shared_r2_prediction_audit.json")
    if (
        shared.get("status") != "passed"
        or shared.get("methods") != [D0_R2_CLASS_CONDITIONAL_MLS, method]
        or shared.get("known_logits_sha256")
        != _array_sha256(role_logits["known_calibration"])
        or shared.get("known_prediction_sha256")
        != _array_sha256(role_logits["known_calibration"].argmax(axis=1).astype(np.int64))
        or shared.get("surrogate_logits_sha256")
        != _array_sha256(role_logits["surrogate_unknown"])
        or shared.get("surrogate_prediction_sha256")
        != _array_sha256(role_logits["surrogate_unknown"].argmax(axis=1).astype(np.int64))
        or shared.get("known_accuracy_exactly_equal") is not True
        or shared.get("known_macro_f1_exactly_equal") is not True
    ):
        raise DataValidationError("shared R2 prediction audit changed")
    summary = _read_json(root / "unit_summary.json")
    if (
        summary.get("status") != "complete"
        or summary.get("pair_id") != pair_id
        or summary.get("method") != method
        or summary.get("metrics") != method_metrics
        or summary.get("d0_metrics") != d0_metrics
        or summary.get("checkpoint_replay") != "bitwise_exact"
        or summary.get("gradient_audit_authorization") != dict(gradient_authorization)
        or summary.get("smoke_authorization") != smoke_authorization
        or summary.get("confirmation_authorization") != confirmation_authorization
        or summary.get("final_unknown_used") is not False
    ):
        raise DataValidationError("decoupled unit summary changed")
    return {
        "status": "passed",
        "phase": phase,
        "pair_id": pair_id,
        "method": method,
        "destination": str(root),
        "metrics": method_metrics,
        "d0_metrics": d0_metrics,
        "global_mls_background_metrics": global_mls_metrics,
        "identity_metrics": _read_csv(root / "identity_metrics.csv"),
        "absorption_rows": _read_csv(root / "absorption_by_known_class.csv"),
        "schedule_sha256": training_audit["schedule_sha256"],
        "initial_state_sha256": training_audit["initial_state_sha256"],
        "shared_r2_prediction_audit": shared,
        "r2_artifact_binding": r2_binding["audit"],
        "source_hashes": current_source_hashes,
        "gradient_audit_authorization": dict(gradient_authorization),
        "smoke_authorization": smoke_authorization,
        "confirmation_authorization": confirmation_authorization,
        "checkpoint_strict_load": True,
        "checkpoint_replay": "bitwise_exact",
        "metric_recomputation": "exact",
        "final_unknown_used": False,
        "even_angle_test_used": False,
    }


def _phase_authorization_and_plan(
    *, config: Mapping[str, Any], phase: str, pilot_root: str | Path | None
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if phase == "confirmation":
        if pilot_root is None:
            raise DataValidationError("confirmation requires an audited pilot root")
        authorization = _read_authorized_pilot(pilot_root, config)
        return authorization, build_phase_plan(
            phase, selected_method=str(authorization["selected_method"])
        )
    if pilot_root is not None:
        raise DataValidationError("pilot root only applies to confirmation")
    return None, build_phase_plan(phase)


def aggregate_phase_root(
    phase_root: str | Path,
    *,
    config: Mapping[str, Any],
    phase: str,
    pilot_root: str | Path | None = None,
    device_request: str = "auto",
) -> dict[str, Any]:
    root = Path(phase_root).resolve()
    if (root / "_PHASE_SUCCESS.json").exists():
        raise DataValidationError("phase was already aggregated")
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
            device_request=device_request,
        )
        for unit in plan
    ]
    audit_map = {(row["pair_id"], row["method"]): row for row in audits}
    source_hash_populations = {
        json.dumps(row["source_hashes"], sort_keys=True) for row in audits
    }
    gradient_authorizations = {
        json.dumps(row["gradient_audit_authorization"], sort_keys=True)
        for row in audits
    }
    smoke_authorizations = {
        json.dumps(row["smoke_authorization"], sort_keys=True) for row in audits
    }
    confirmation_authorizations = {
        json.dumps(row["confirmation_authorization"], sort_keys=True)
        for row in audits
    }
    if len(source_hash_populations) != 1 or len(gradient_authorizations) != 1:
        raise DataValidationError("phase mixes source code or gradient-audit authorization")
    if phase == "pilot":
        if len(smoke_authorizations) != 1 or next(iter(smoke_authorizations)) == "null":
            raise DataValidationError("pilot units do not share one audited smoke authorization")
    elif smoke_authorizations != {"null"}:
        raise DataValidationError("non-pilot phase contains direct smoke authorization")
    if phase == "confirmation":
        expected_authorization = json.dumps(authorization, sort_keys=True)
        if confirmation_authorizations != {expected_authorization}:
            raise DataValidationError(
                "confirmation units do not share the requested pilot authorization"
            )
    elif confirmation_authorizations != {"null"}:
        raise DataValidationError("non-confirmation phase contains pilot authorization")
    shared_gradient_authorization = json.loads(next(iter(gradient_authorizations)))
    shared_smoke_authorization = json.loads(next(iter(smoke_authorizations)))
    metric_rows: list[dict[str, Any]] = []
    global_metric_rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    absorption_rows: list[dict[str, Any]] = []
    integrity: dict[str, Any] = {}
    for pair_id in dict.fromkeys(str(unit["pair_id"]) for unit in plan):
        methods = [str(unit["method"]) for unit in plan if str(unit["pair_id"]) == pair_id]
        pair_audits = [audit_map[(pair_id, method)] for method in methods]
        d0 = pair_audits[0]["d0_metrics"]
        d0_identity = [
            row for row in pair_audits[0]["identity_metrics"]
            if row["method"] == D0_R2_CLASS_CONDITIONAL_MLS
        ]
        d0_absorption = [
            row for row in pair_audits[0]["absorption_rows"]
            if row["method"] == D0_R2_CLASS_CONDITIONAL_MLS
        ]
        for audit in pair_audits[1:]:
            if audit["d0_metrics"] != d0:
                raise DataValidationError(f"D0 metrics differ across {pair_id} candidate units")
            current_identity = [row for row in audit["identity_metrics"] if row["method"] == D0_R2_CLASS_CONDITIONAL_MLS]
            if current_identity != d0_identity:
                raise DataValidationError(f"D0 identity rows differ across {pair_id} units")
            current_absorption = [
                row
                for row in audit["absorption_rows"]
                if row["method"] == D0_R2_CLASS_CONDITIONAL_MLS
            ]
            if current_absorption != d0_absorption:
                raise DataValidationError(f"D0 absorption rows differ across {pair_id} units")
        metric_rows.append(
            {
                "pair_id": pair_id,
                "method": D0_R2_CLASS_CONDITIONAL_MLS,
                **{key: float(d0[key]) for key in REPORT_METRIC_KEYS},
                "threshold": float(d0["threshold"]),
            }
        )
        global_metrics = pair_audits[0]["global_mls_background_metrics"]
        if any(
            audit["global_mls_background_metrics"] != global_metrics
            for audit in pair_audits[1:]
        ):
            raise DataValidationError(f"global MLS differs across {pair_id} units")
        global_metric_rows.append(
            {
                "pair_id": pair_id,
                "method": GLOBAL_MLS,
                **{key: float(global_metrics[key]) for key in REPORT_METRIC_KEYS},
                "threshold": float(global_metrics["threshold"]),
            }
        )
        identity_rows.extend(d0_identity)
        absorption_rows.extend(d0_absorption)
        for audit in pair_audits:
            metrics = audit["metrics"]
            metric_rows.append(
                {
                    "pair_id": pair_id,
                    "method": audit["method"],
                    **{key: float(metrics[key]) for key in REPORT_METRIC_KEYS},
                    "threshold": float(metrics["threshold"]),
                }
            )
            identity_rows.extend(row for row in audit["identity_metrics"] if row["method"] == audit["method"])
            absorption_rows.extend(row for row in audit["absorption_rows"] if row["method"] == audit["method"])
        schedule_hashes = {str(audit["schedule_sha256"]) for audit in pair_audits}
        initial_hashes = {str(audit["initial_state_sha256"]) for audit in pair_audits}
        logits_hashes = {
            str(audit["shared_r2_prediction_audit"]["known_logits_sha256"])
            for audit in pair_audits
        }
        prediction_hashes = {
            str(audit["shared_r2_prediction_audit"]["known_prediction_sha256"])
            for audit in pair_audits
        }
        surrogate_logits_hashes = {
            str(audit["shared_r2_prediction_audit"]["surrogate_logits_sha256"])
            for audit in pair_audits
        }
        surrogate_prediction_hashes = {
            str(audit["shared_r2_prediction_audit"]["surrogate_prediction_sha256"])
            for audit in pair_audits
        }
        if (
            len(schedule_hashes) != 1
            or len(initial_hashes) != 1
            or len(logits_hashes) != 1
            or len(prediction_hashes) != 1
            or len(surrogate_logits_hashes) != 1
            or len(surrogate_prediction_hashes) != 1
        ):
            raise DataValidationError(f"paired D1/D2 protocol differs for {pair_id}")
        integrity[pair_id] = {
            "schedule_sha256": next(iter(schedule_hashes)),
            "initial_state_sha256": next(iter(initial_hashes)),
            "known_logits_sha256": next(iter(logits_hashes)),
            "known_prediction_sha256": next(iter(prediction_hashes)),
            "surrogate_logits_sha256": next(iter(surrogate_logits_hashes)),
            "surrogate_prediction_sha256": next(iter(surrogate_prediction_hashes)),
            "d0_reused_without_training": True,
            "r2_logits_predictions_identical": True,
        }
    _write_json(root / "task_plan.json", {"phase": phase, "units": plan})
    _write_csv(root / "metrics_by_pair.csv", metric_rows)
    _write_csv(root / "global_mls_background_by_pair.csv", global_metric_rows)
    _write_csv(root / "identity_metrics.csv", identity_rows)
    _write_csv(root / "absorption_by_known_class.csv", absorption_rows)
    _write_json(root / "phase_integrity_audit.json", {"status": "passed", "pairs": integrity})
    if phase == "pilot":
        gate = evaluate_pilot_gate(metric_rows, identity_rows, absorption_rows)
        _write_json(root / "pilot_gate.json", gate)
        decision = str(gate["signal"])
    elif phase == "confirmation":
        gate = evaluate_confirmation_gate(
            metric_rows,
            identity_rows,
            str(authorization["selected_method"]),
        )
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
        "gate": gate,
        "decision": decision,
        "confirmation_authorization": authorization,
        "gradient_audit_authorization": shared_gradient_authorization,
        "smoke_authorization": shared_smoke_authorization,
        "config_sha256": config["_config_sha256"],
        "diagnostic_only": phase == "smoke",
        "final_unknown_used": False,
        "even_angle_test_used": False,
        "final_unknown_test_authorized": False,
        "automatic_followon_authorized": phase == "pilot" and bool(gate and gate.get("confirmation_allowed")),
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
    device_request: str = "auto",
) -> dict[str, Any]:
    root = Path(phase_root).resolve()
    success = _read_json(root / "_PHASE_SUCCESS.json")
    if (
        success.get("status") != "complete"
        or success.get("phase_summary_sha256") != file_sha256(root / "phase_summary.json")
        or success.get("artifact_hashes_sha256") != file_sha256(root / "artifact_hashes.json")
        or _read_json(root / "artifact_hashes.json") != _artifact_hashes(root)
    ):
        raise DataValidationError("phase artifact hash audit failed")
    stored = _read_json(root / "phase_summary.json")
    if phase == "confirmation" and pilot_root is None:
        pilot_root = str(stored["confirmation_authorization"]["pilot_root"])
    authorization, plan = _phase_authorization_and_plan(
        config=config, phase=phase, pilot_root=pilot_root
    )
    audits = []
    for unit in plan:
        audits.append(audit_unit_result(
            _unit_destination(root, str(unit["pair_id"]), str(unit["method"])),
            config=config,
            phase=phase,
            pair_id=str(unit["pair_id"]),
            method=str(unit["method"]),
            device_request=device_request,
        ))
    audit_map = {(row["pair_id"], row["method"]): row for row in audits}
    expected_plan_payload = {"phase": phase, "units": plan}
    if _read_json(root / "task_plan.json") != expected_plan_payload:
        raise DataValidationError("phase task plan changed")
    source_hash_populations = {
        json.dumps(row["source_hashes"], sort_keys=True) for row in audits
    }
    gradient_authorizations = {
        json.dumps(row["gradient_audit_authorization"], sort_keys=True)
        for row in audits
    }
    smoke_authorizations = {
        json.dumps(row["smoke_authorization"], sort_keys=True) for row in audits
    }
    confirmation_authorizations = {
        json.dumps(row["confirmation_authorization"], sort_keys=True)
        for row in audits
    }
    if len(source_hash_populations) != 1 or len(gradient_authorizations) != 1:
        raise DataValidationError("phase mixes source code or gradient-audit authorization")
    if phase == "pilot":
        if len(smoke_authorizations) != 1 or next(iter(smoke_authorizations)) == "null":
            raise DataValidationError("pilot units do not share one audited smoke authorization")
    elif smoke_authorizations != {"null"}:
        raise DataValidationError("non-pilot phase contains direct smoke authorization")
    if phase == "confirmation":
        expected_authorization = json.dumps(authorization, sort_keys=True)
        if confirmation_authorizations != {expected_authorization}:
            raise DataValidationError(
                "confirmation units do not share the requested pilot authorization"
            )
    elif confirmation_authorizations != {"null"}:
        raise DataValidationError("non-confirmation phase contains pilot authorization")
    shared_gradient_authorization = json.loads(next(iter(gradient_authorizations)))
    shared_smoke_authorization = json.loads(next(iter(smoke_authorizations)))
    expected_metric_rows: list[dict[str, Any]] = []
    expected_global_rows: list[dict[str, Any]] = []
    expected_identity_rows: list[dict[str, Any]] = []
    expected_absorption_rows: list[dict[str, Any]] = []
    expected_integrity: dict[str, Any] = {}
    for pair_id in dict.fromkeys(str(unit["pair_id"]) for unit in plan):
        methods = [str(unit["method"]) for unit in plan if str(unit["pair_id"]) == pair_id]
        pair_audits = [audit_map[(pair_id, method)] for method in methods]
        d0 = pair_audits[0]["d0_metrics"]
        d0_identity = [
            row
            for row in pair_audits[0]["identity_metrics"]
            if row["method"] == D0_R2_CLASS_CONDITIONAL_MLS
        ]
        d0_absorption = [
            row
            for row in pair_audits[0]["absorption_rows"]
            if row["method"] == D0_R2_CLASS_CONDITIONAL_MLS
        ]
        global_metrics = pair_audits[0]["global_mls_background_metrics"]
        for audit in pair_audits[1:]:
            if audit["d0_metrics"] != d0:
                raise DataValidationError(f"D0 metrics differ across {pair_id} candidate units")
            if [
                row
                for row in audit["identity_metrics"]
                if row["method"] == D0_R2_CLASS_CONDITIONAL_MLS
            ] != d0_identity:
                raise DataValidationError(f"D0 identity rows differ across {pair_id} units")
            if [
                row
                for row in audit["absorption_rows"]
                if row["method"] == D0_R2_CLASS_CONDITIONAL_MLS
            ] != d0_absorption:
                raise DataValidationError(f"D0 absorption rows differ across {pair_id} units")
            if audit["global_mls_background_metrics"] != global_metrics:
                raise DataValidationError(f"global MLS differs across {pair_id} units")
        expected_metric_rows.append(
            {
                "pair_id": pair_id,
                "method": D0_R2_CLASS_CONDITIONAL_MLS,
                **{key: float(d0[key]) for key in REPORT_METRIC_KEYS},
                "threshold": float(d0["threshold"]),
            }
        )
        expected_global_rows.append(
            {
                "pair_id": pair_id,
                "method": GLOBAL_MLS,
                **{key: float(global_metrics[key]) for key in REPORT_METRIC_KEYS},
                "threshold": float(global_metrics["threshold"]),
            }
        )
        expected_identity_rows.extend(d0_identity)
        expected_absorption_rows.extend(d0_absorption)
        for audit in pair_audits:
            metrics = audit["metrics"]
            expected_metric_rows.append(
                {
                    "pair_id": pair_id,
                    "method": audit["method"],
                    **{key: float(metrics[key]) for key in REPORT_METRIC_KEYS},
                    "threshold": float(metrics["threshold"]),
                }
            )
            expected_identity_rows.extend(
                row for row in audit["identity_metrics"] if row["method"] == audit["method"]
            )
            expected_absorption_rows.extend(
                row for row in audit["absorption_rows"] if row["method"] == audit["method"]
            )
        schedule_hashes = {str(audit["schedule_sha256"]) for audit in pair_audits}
        initial_hashes = {str(audit["initial_state_sha256"]) for audit in pair_audits}
        known_logits_hashes = {
            str(audit["shared_r2_prediction_audit"]["known_logits_sha256"])
            for audit in pair_audits
        }
        known_prediction_hashes = {
            str(audit["shared_r2_prediction_audit"]["known_prediction_sha256"])
            for audit in pair_audits
        }
        surrogate_logits_hashes = {
            str(audit["shared_r2_prediction_audit"]["surrogate_logits_sha256"])
            for audit in pair_audits
        }
        surrogate_prediction_hashes = {
            str(audit["shared_r2_prediction_audit"]["surrogate_prediction_sha256"])
            for audit in pair_audits
        }
        if any(
            len(values) != 1
            for values in (
                schedule_hashes,
                initial_hashes,
                known_logits_hashes,
                known_prediction_hashes,
                surrogate_logits_hashes,
                surrogate_prediction_hashes,
            )
        ):
            raise DataValidationError(f"paired D1/D2 protocol differs for {pair_id}")
        expected_integrity[pair_id] = {
            "schedule_sha256": next(iter(schedule_hashes)),
            "initial_state_sha256": next(iter(initial_hashes)),
            "known_logits_sha256": next(iter(known_logits_hashes)),
            "known_prediction_sha256": next(iter(known_prediction_hashes)),
            "surrogate_logits_sha256": next(iter(surrogate_logits_hashes)),
            "surrogate_prediction_sha256": next(iter(surrogate_prediction_hashes)),
            "d0_reused_without_training": True,
            "r2_logits_predictions_identical": True,
        }
    if (root / "metrics_by_pair.csv").read_bytes() != _render_csv(expected_metric_rows):
        raise DataValidationError("phase metric table does not reproduce from units")
    if (root / "global_mls_background_by_pair.csv").read_bytes() != _render_csv(expected_global_rows):
        raise DataValidationError("global MLS table does not reproduce from units")
    if (root / "identity_metrics.csv").read_bytes() != _render_csv(expected_identity_rows):
        raise DataValidationError("phase identity table does not reproduce from units")
    if (root / "absorption_by_known_class.csv").read_bytes() != _render_csv(expected_absorption_rows):
        raise DataValidationError("phase absorption table does not reproduce from units")
    if _read_json(root / "phase_integrity_audit.json") != {
        "status": "passed",
        "pairs": expected_integrity,
    }:
        raise DataValidationError("phase integrity record does not reproduce")
    metric_rows = _read_csv(root / "metrics_by_pair.csv")
    identity_rows = _read_csv(root / "identity_metrics.csv")
    absorption_rows = _read_csv(root / "absorption_by_known_class.csv")
    if phase == "pilot":
        expected_gate = evaluate_pilot_gate(metric_rows, identity_rows, absorption_rows)
        if _read_json(root / "pilot_gate.json") != expected_gate:
            raise DataValidationError("pilot gate does not reproduce")
        expected_decision = str(expected_gate["signal"])
    elif phase == "confirmation":
        expected_gate = evaluate_confirmation_gate(
            metric_rows, identity_rows, str(authorization["selected_method"])
        )
        if _read_json(root / "confirmation_gate.json") != expected_gate:
            raise DataValidationError("confirmation gate does not reproduce")
        expected_decision = str(expected_gate["decision"])
    else:
        expected_gate = None
        expected_decision = "diagnostic_smoke_only"
    expected_summary = {
        "status": "complete",
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "pair_ids": list(dict.fromkeys(str(unit["pair_id"]) for unit in plan)),
        "unit_count": len(plan),
        "gate": expected_gate,
        "decision": expected_decision,
        "confirmation_authorization": authorization,
        "gradient_audit_authorization": shared_gradient_authorization,
        "smoke_authorization": shared_smoke_authorization,
        "config_sha256": config["_config_sha256"],
        "diagnostic_only": phase == "smoke",
        "final_unknown_used": False,
        "even_angle_test_used": False,
        "final_unknown_test_authorized": False,
        "automatic_followon_authorized": phase == "pilot"
        and bool(expected_gate and expected_gate.get("confirmation_allowed")),
    }
    if stored != expected_summary:
        raise DataValidationError("phase summary contract changed")
    return {
        "status": "passed",
        "phase": phase,
        "root": str(root),
        "unit_count": len(plan),
        "decision": stored["decision"],
        "gate": stored["gate"],
        "checkpoint_replay": "passed",
        "metric_recomputation": "exact",
        "authorization": authorization,
        "final_unknown_test_authorized": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preregistered decoupled FG-MV-CSSR runner")
    commands = parser.add_subparsers(dest="command", required=True)
    load = commands.add_parser("load-config")
    load.add_argument("--config", default=CONFIG_RELATIVE_PATH)
    plan = commands.add_parser("plan")
    plan.add_argument("--config", default=CONFIG_RELATIVE_PATH)
    plan.add_argument("--phase", choices=("smoke", "pilot", "confirmation"), required=True)
    plan.add_argument("--selected-method", choices=DECOUPLED_METHODS)
    plan.add_argument("--pilot-root")
    run = commands.add_parser("run-unit")
    run.add_argument("--config", default=CONFIG_RELATIVE_PATH)
    run.add_argument("--bundle-root", required=True)
    run.add_argument("--r2-results-root", required=True)
    run.add_argument("--phase-root", required=True)
    run.add_argument("--gradient-audit-root", required=True)
    run.add_argument("--phase", choices=("smoke", "pilot", "confirmation"), required=True)
    run.add_argument("--pair-id", required=True)
    run.add_argument("--method", choices=DECOUPLED_METHODS, required=True)
    run.add_argument("--device", default="auto")
    run.add_argument("--pilot-root")
    run.add_argument("--smoke-root")
    audit_unit_parser = commands.add_parser("audit-unit")
    audit_unit_parser.add_argument("--config", default=CONFIG_RELATIVE_PATH)
    audit_unit_parser.add_argument("--unit-root", required=True)
    audit_unit_parser.add_argument("--phase", choices=("smoke", "pilot", "confirmation"), required=True)
    audit_unit_parser.add_argument("--pair-id", required=True)
    audit_unit_parser.add_argument("--method", choices=DECOUPLED_METHODS, required=True)
    audit_unit_parser.add_argument("--device", default="auto")
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--config", default=CONFIG_RELATIVE_PATH)
    aggregate.add_argument("--phase-root", required=True)
    aggregate.add_argument("--phase", choices=("smoke", "pilot", "confirmation"), required=True)
    aggregate.add_argument("--pilot-root")
    aggregate.add_argument("--device", default="auto")
    audit_phase_parser = commands.add_parser("audit-phase")
    audit_phase_parser.add_argument("--config", default=CONFIG_RELATIVE_PATH)
    audit_phase_parser.add_argument("--phase-root", required=True)
    audit_phase_parser.add_argument("--phase", choices=("smoke", "pilot", "confirmation"), required=True)
    audit_phase_parser.add_argument("--pilot-root")
    audit_phase_parser.add_argument("--device", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = load_fg_mv_cssr_decoupled_config(arguments.config)
    if arguments.command == "load-config":
        result = {
            "status": "passed",
            "experiment_id": EXPERIMENT_ID,
            "config_path": config["_config_path"],
            "config_sha256": config["_config_sha256"],
            "final_unknown_test_authorized": False,
        }
    elif arguments.command == "plan":
        selected = arguments.selected_method
        authorization = None
        if arguments.pilot_root:
            if arguments.phase != "confirmation" or selected is not None:
                raise DataValidationError("pilot-root planning only applies to confirmation")
            authorization = _read_authorized_pilot(arguments.pilot_root, config)
            selected = str(authorization["selected_method"])
        result = {
            "status": "planned",
            "phase": arguments.phase,
            "authorization": authorization,
            "units": build_phase_plan(arguments.phase, selected_method=selected),
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
            gradient_audit_root=arguments.gradient_audit_root,
            device_request=arguments.device,
            pilot_root=arguments.pilot_root,
            smoke_root=arguments.smoke_root,
        )
    elif arguments.command == "audit-unit":
        result = audit_unit_result(
            arguments.unit_root,
            config=config,
            phase=arguments.phase,
            pair_id=arguments.pair_id,
            method=arguments.method,
            device_request=arguments.device,
        )
    elif arguments.command == "aggregate":
        result = aggregate_phase_root(
            arguments.phase_root,
            config=config,
            phase=arguments.phase,
            pilot_root=arguments.pilot_root,
            device_request=arguments.device,
        )
    elif arguments.command == "audit-phase":
        result = audit_phase_root(
            arguments.phase_root,
            config=config,
            phase=arguments.phase,
            pilot_root=arguments.pilot_root,
            device_request=arguments.device,
        )
    else:
        raise AssertionError("unreachable")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
