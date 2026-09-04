from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from hrrp_osr.data.errors import DataConfigError, DataValidationError


EXPERIMENT_ID = "official_cssr_hrrp_pilot_v1"
CONFIG_RELATIVE_PATH = (
    "configs/experiments/cssr/official_cssr_hrrp_pilot_v1.yaml"
)
CONFIG_FILE_SHA256 = (
    "11a92b15ca21f6b025f1fce0dfcb6411f3fa53ef1d26f22693c73cf5d46fa188"
)

O0_R2_CC_MLS = "O0_R2_CC_MLS"
O1_OFFICIAL_LINEAR_FT = "O1_OFFICIAL_LINEAR_FT"
O2_OFFICIAL_PCSSR_FT = "O2_OFFICIAL_PCSSR_FT"
O3_OFFICIAL_LINEAR_E2E = "O3_OFFICIAL_LINEAR_E2E"
O4_OFFICIAL_PCSSR_E2E = "O4_OFFICIAL_PCSSR_E2E"

METHODS = (
    O0_R2_CC_MLS,
    O1_OFFICIAL_LINEAR_FT,
    O2_OFFICIAL_PCSSR_FT,
    O3_OFFICIAL_LINEAR_E2E,
    O4_OFFICIAL_PCSSR_E2E,
)
TRAINABLE_METHODS = METHODS[1:]
PCSSR_METHODS = (O2_OFFICIAL_PCSSR_FT, O4_OFFICIAL_PCSSR_E2E)
PILOT_PAIRS = ("N1", "N4", "N2")
ANGLE_FOLD = 0
R2_SEED = 20260830
OFFICIAL_CSSR_SEED = 20260906
CLASS_COUNT = 5
TRAIN_BASES_PER_CLASS = 144
TRAIN_SAMPLE_COUNT = CLASS_COUNT * TRAIN_BASES_PER_CLASS
INPUT_LENGTH = 601
TRAIN_BATCH_SIZE = 128
TRAIN_STEPS_PER_EPOCH = 6
TRAIN_EPOCHS = 40
SCORE_NORM_VARIANTS = (1, 2, 3, 4)
GATE_TOLERANCE = 1.0e-12

IDENTITY_PAIRS = (
    ("N0", (0, 2), (1, 3, 4, 5, 6)),
    ("N1", (2, 5), (0, 1, 3, 4, 6)),
    ("N2", (3, 5), (0, 1, 2, 4, 6)),
    ("N3", (1, 3), (0, 2, 4, 5, 6)),
    ("N4", (1, 6), (0, 2, 3, 4, 5)),
    ("N5", (4, 6), (0, 1, 2, 3, 5)),
    ("N6", (0, 4), (1, 2, 3, 5, 6)),
)
SOURCE_KNOWN_ORDER = (
    "CVN77",
    "DDG-1000",
    "DDG-112",
    "油气轮MARVEL CRANE",
    "爱达魔都号",
    "迷你好望角型散货船",
    "集装箱船达飞罗尔多夫级",
)
SURROGATE_IDENTITIES = {
    pair_id: tuple(SOURCE_KNOWN_ORDER[index] for index in unknown_indices)
    for pair_id, unknown_indices, _ in IDENTITY_PAIRS
    if pair_id in PILOT_PAIRS
}

OFFICIAL_FILE_HASHES = {
    "methods/cssr.py": "0d23558c6a3cc4bf068036502a8ab43ee6278aecd91d96741f7375a142d9c5a3",
    "methods/cssr_ft.py": "31244f194d91f6cab0bdf34eb14a0ed3b58f25b6c49a44042bb96baa9977fb16",
    "configs/basic.json": "672375c6838004ae604509ba57098c7fefd17b6ac0f38e7c955fc8c09ba3192a",
    "configs/pcssr.json": "353b0768cc6ee60ac76c110a22da8bdb5c15179260d4abeb2f43fee422d24c6b",
    "configs/pcssr/cifar10.json": "ce5c7187cab1d8a7387526e459dc21c257f407e15e2304a91f618a8d8d34b0ab",
    "configs/pcssr/imagenet.json": "170b8b7f86a2bde8fd409feaa96edfbfbd4226cc7ed9d1a564db8ca8a783b505",
}

_EXPECTED_TOP_LEVEL_KEYS = {
    "schema_version",
    "stage",
    "experiment_id",
    "result_scope",
    "evidence_scope",
    "sealed_decoupled_source",
    "official_reference",
    "prior_r2",
    "bundle",
    "classes",
    "data",
    "normalization",
    "model",
    "training",
    "score",
    "evaluation",
    "pilot_gate",
    "runtime",
    "outputs",
}


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DataConfigError(f"{name} must be a mapping")
    return value


def _require(errors: list[str], observed: Any, expected: Any, name: str) -> None:
    if observed != expected:
        errors.append(f"{name} changed: expected {expected!r}, observed {observed!r}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sequence_sha256(values: Sequence[Any]) -> str:
    payload = json.dumps(
        list(values), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def derive_seed(material: str) -> int:
    """Apply the preregistered UTF-8 SHA-256 -> first-eight-bytes seed rule."""

    if not isinstance(material, str) or not material:
        raise DataValidationError("seed material must be a nonempty string")
    return int.from_bytes(
        hashlib.sha256(material.encode("utf-8")).digest()[:8],
        byteorder="big",
        signed=False,
    )


def _expected_pair_rows() -> list[dict[str, Any]]:
    return [
        {
            "pair_id": pair_id,
            "surrogate_unknown_indices": list(unknown),
            "train_known_indices": list(train),
        }
        for pair_id, unknown, train in IDENTITY_PAIRS
    ]


def validate_official_cssr_config(config: Mapping[str, Any]) -> None:
    """Validate every execution-critical preregistered protocol field."""

    config = _mapping(config, "official CSSR config")
    errors: list[str] = []
    runtime_loader_keys = {"_config_path", "_config_sha256"}
    _require(
        errors,
        set(config) - runtime_loader_keys,
        _EXPECTED_TOP_LEVEL_KEYS,
        "top-level keys",
    )
    for name, expected in {
        "schema_version": 1,
        "stage": "P3_cssr_mechanism_and_official_semantics_hrrp_pilot",
        "experiment_id": EXPERIMENT_ID,
        "result_scope": (
            "read_only_mechanism_audit_then_diagnostic_smoke_and_three_pair_pilot"
        ),
    }.items():
        _require(errors, config.get(name), expected, name)

    evidence = _mapping(config.get("evidence_scope"), "evidence_scope")
    _require(errors, evidence.get("source_known_odd_angle_only"), True, "odd-only")
    for name in (
        "final_unknown_classes_used",
        "even_angle_test_used",
        "surrogate_unknown_used_for_training",
        "surrogate_unknown_used_for_template",
        "surrogate_unknown_used_for_normalization",
        "surrogate_unknown_used_for_threshold",
        "known_calibration_used_for_training",
        "known_calibration_used_for_template",
        "known_calibration_used_for_normalization",
        "calibration_checkpoint_selection",
        "performance_hyperparameter_selection",
        "arpl_used",
        "pseudo_unknown_used",
        "angle_metadata_used_by_model",
    ):
        _require(errors, evidence.get(name), False, f"evidence_scope.{name}")

    sealed = _mapping(config.get("sealed_decoupled_source"), "sealed source")
    for name, expected in {
        "experiment_id": "fg_mv_cssr_decoupled_audit_v3",
        "source_code_commit": "eb17466ff41efaf15f555c545da4ce207f8ddb96",
        "results_commit": "105c313f436f20e57c6157e08e0afd737556302e",
        "source_config": "configs/experiments/cssr/fg_mv_cssr_decoupled_audit_v3.yaml",
        "source_config_sha256": "b67f84dda0754b9b628ce046beb1b02bc8d7e15e0764bb03889bc6865ece5f7c",
        "source_report": "docs/cssr/fg_mv_cssr_decoupled_results_2026-09-04.md",
        "source_report_sha256": "2db893a4ba568505e20b95eb07312dfea0401053b330b59672dd41153a359760",
        "phase": "stage_b_pilot",
        "pairs": list(PILOT_PAIRS),
        "methods": [
            "D0_R2_CLASS_CONDITIONAL_MLS",
            "D1_DECOUPLED_REL_CSSR",
            "D2_DECOUPLED_ABSREL_CSSR",
        ],
        "read_only": True,
        "performance_gate_eligible": False,
    }.items():
        _require(errors, sealed.get(name), expected, f"sealed source {name}")

    official = _mapping(config.get("official_reference"), "official reference")
    for name, expected in {
        "paper": "arXiv:2207.02158",
        "repository": "https://github.com/xyzedd/CSSR",
        "commit": "d5a99e91f310ec274c7bfe5796fb270719a07ab3",
        "runtime_root_required": True,
        "vendored_in_repository": False,
    }.items():
        _require(errors, official.get(name), expected, f"official reference {name}")
    _require(
        errors,
        dict(_mapping(official.get("files"), "official files")),
        OFFICIAL_FILE_HASHES,
        "official file hashes",
    )
    _require(
        errors,
        dict(_mapping(official.get("differential"), "official differential")),
        {
            "float32": {"rtol": 1.0e-5, "atol": 1.0e-6},
            "float64": {"rtol": 1.0e-9, "atol": 1.0e-11},
            "failure_status": "blocked_by_official_differential_failure",
        },
        "official differential",
    )

    prior = _mapping(config.get("prior_r2"), "prior_r2")
    for name, expected in {
        "result_commit": "edb05062d07be1984067f91759d6029cd9c0bf9a",
        "formal_code_commit": "62e318de82b4221b599e06b1166483673e9c1cd3",
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
        "frozen_encoder_source": "stem_stage1_stage2_stage3_only",
    }.items():
        _require(errors, prior.get(name), expected, f"prior_r2.{name}")
    expected_r2_hashes = {
        "N0": (
            "142a85b3a090213684126cf695b08fec259724a0bd8399dc1adb40b114aab192",
            "37dac18016223e08451c6551e279a6136ed494cb9c86edb5f0a938d71a2b115d",
            "a6bc7f4b1c095976964716e70c72666c0133f0bb3b5a63aacbc5612d9a888c93",
        ),
        "N1": (
            "a4f6fa3235fbb5cf74b712588a0318f614a05287adec4ee881820424cddbcbaa",
            "0b8a97dcfd744896bbae912c1363379201ced18a55107f80b2d2f3256fb5c5bc",
            "b43da73179b8ddb0e0ae1f97b3724e9fcffe9ce32f10aaa6466cc8f408a74275",
        ),
        "N2": (
            "14e2ac7b686c901112f969fe0bd7f53c29646e7c015bae794d30c39051f9c0b9",
            "1a7dc0031cf5b32a41131289fb4117a144463c025e93bc7a487e56a3c8c8bd2d",
            "58e8086e8ba27e2c4537d98d5ec1e6faaaa1bdf3d47cfec6ad9278227279114a",
        ),
        "N3": (
            "6427a09f3e4a5e67ff652fea6e44c8364b62381acc8338099dccc818ac284bc9",
            "53fead93617851f8646dc7c76ff3773b6c55a720d3be17feda462535994e7d27",
            "05ef84488ee515e09afbdf4504fdd1dd8597347faa63442a49abc32265caa6e8",
        ),
        "N4": (
            "169387ad7a87463110ac7a2cd45afd7dac49428538c93c84975162e425d94ff5",
            "8b0202d1e08ae83eec4bf07fc1dbb6a3f39fef2378ac15e57635709d8872b41a",
            "942e6c14d2237120ca9937a23df7f095ce718ea072933ee52d0ab2d3c3c79e95",
        ),
        "N5": (
            "74cde2c6b30f1fa96219fe20777dfc632575c8c3c0281706ca016ef2497642df",
            "a706c63e47f8522510c2926e70a8072ca8ca183c5ef74957b8451d28d2c47c80",
            "ab748cce5fbb8f1299fe720311e2c1da3805bda070b3b47b5db46f886805eee5",
        ),
        "N6": (
            "178dbaa9e461d28825124b688752ed5c1005a8f0265963ef57e5c27a0a65e86e",
            "46b454fc313573121fcf6ad214b91f9e21a2cb996a38d3beaf9c83d8321ce140",
            "b11a125b08a236f182857030fc07770d22a1890f5d20e68efa0f3fade4a4b20b",
        ),
    }
    observed_hashes = _mapping(prior.get("unit_artifact_hashes"), "R2 hashes")
    _require(errors, set(observed_hashes), set(expected_r2_hashes), "R2 pair hash keys")
    for pair_id, expected in expected_r2_hashes.items():
        observed = _mapping(observed_hashes.get(pair_id), f"R2 hashes {pair_id}")
        for filename, expected_hash in zip(
            ("checkpoint.pt", "pair_manifest.csv", "features_logits_scores.npz"),
            expected,
        ):
            _require(
                errors,
                observed.get(filename),
                expected_hash,
                f"R2 {pair_id}/{filename}",
            )

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
    _require(errors, list(classes.get("source_known_order", [])), list(SOURCE_KNOWN_ORDER), "class order")
    _require(errors, list(classes.get("identity_pairs", [])), _expected_pair_rows(), "identity pairs")
    _require(errors, list(classes.get("pilot_pairs", [])), list(PILOT_PAIRS), "pilot pairs")

    data = _mapping(config.get("data"), "data")
    for name, expected in {
        "angle_fold": ANGLE_FOLD,
        "development_angle_parity": "odd",
        "profile_value_access_policy": (
            "enforced_source_known_odd_index_allowlist_v1"
        ),
        "full_manifest_metadata_read_for_integrity": True,
        "full_profile_file_hashed_as_opaque_bytes": True,
        "final_unknown_profile_values_read": False,
        "even_angle_profile_values_read": False,
        "view_count": 2,
        "train_unique_base_samples_per_class": TRAIN_BASES_PER_CLASS,
        "known_calibration_unique_base_samples_per_class": 36,
        "surrogate_unique_base_samples_per_identity": 36,
        "evaluation_pairs_per_class": 500,
        "final_test_pairs_generated": False,
    }.items():
        _require(errors, data.get(name), expected, f"data.{name}")
    _require(
        errors,
        dict(_mapping(data.get("smoke"), "data.smoke")),
        {
            "pair_id": "N1",
            "methods": list(TRAINABLE_METHODS),
            "epochs": 1,
            "optimizer_updates": 6,
            "full_train_unique_base_schedule": True,
            "evaluation_pairs_per_class_or_identity": 2,
            "diagnostic_only": True,
        },
        "smoke",
    )
    _require(
        errors,
        dict(_mapping(data.get("pilot"), "data.pilot")),
        {
            "pairs": list(PILOT_PAIRS),
            "methods": list(TRAINABLE_METHODS),
            "epochs": TRAIN_EPOCHS,
            "task_count": 12,
        },
        "pilot",
    )

    _require(
        errors,
        dict(_mapping(config.get("normalization"), "normalization")),
        {
            "method": "global_scalar_zscore",
            "fit_population": "unique_train_known_base_samples_only",
            "accumulation_dtype": "float64",
            "population_ddof": 0,
            "epsilon": 1.0e-12,
        },
        "normalization",
    )
    model = _mapping(config.get("model"), "model")
    for name, expected in {
        "class_count": CLASS_COUNT,
        "input_length": INPUT_LENGTH,
        "feature_map_shape": [128, 76],
        "encoder_source": "prior_r2_stem_stage1_stage2_stage3",
        "encoder_initial_state_shared_across_o1_o4": True,
    }.items():
        _require(errors, model.get(name), expected, f"model.{name}")
    _require(
        errors,
        dict(_mapping(model.get("pcssr"), "model.pcssr")),
        {
            "independent_class_autoencoders": 5,
            "encoder": {
                "type": "Conv1d",
                "in_channels": 128,
                "out_channels": 64,
                "kernel_size": 1,
                "bias": False,
            },
            "activation": "Tanh",
            "decoder": {
                "type": "Conv1d",
                "in_channels": 64,
                "out_channels": 128,
                "kernel_size": 1,
                "bias": False,
            },
            "gamma": 0.1,
            "reconstruction_error": "channel_sum_L1",
            "clip": [-100.0, 100.0],
            "probability": "class_softmax_per_position_then_position_mean",
            "classification_loss": "negative_log_true_class_mean_probability",
        },
        "pCSSR model",
    )
    _require(
        errors,
        dict(_mapping(model.get("matched_linear"), "model.matched_linear")),
        {
            "type": "Conv1d",
            "in_channels": 128,
            "out_channels": 5,
            "kernel_size": 1,
            "bias": False,
            "gamma": 0.1,
            "probability": "class_softmax_per_position_then_position_mean",
            "classification_loss": "negative_log_true_class_mean_probability",
        },
        "matched linear",
    )

    training = _mapping(config.get("training"), "training")
    expected_training = {
        "optimizer": "SGD",
        "momentum": 0.0,
        "nesterov": False,
        "batch_size": TRAIN_BATCH_SIZE,
        "steps_per_epoch": TRAIN_STEPS_PER_EPOCH,
        "epochs": TRAIN_EPOCHS,
        "early_stopping": False,
        "formal_checkpoint_epoch": TRAIN_EPOCHS,
        "checkpoint_selection": "fixed_final_epoch",
        "gradient_clipping": "none",
        "head_base_lr": 0.05,
        "head_weight_decay": 1.0e-4,
        "encoder_base_lr": 0.005,
        "encoder_weight_decay": 5.0e-4,
        "warmup_epochs": 2,
        "milestone_epochs": [25, 35],
        "milestone_decay": 0.1,
        "e2e_encoder_frozen_epochs": 5,
        "e2e_encoder_unfreeze_epoch": 6,
        "ft_encoder_frozen_all_epochs": True,
        "each_unique_base_once_per_epoch": True,
        "drop_last": False,
        "official_cssr_seed": OFFICIAL_CSSR_SEED,
        "schedule_material": "official_cssr_hrrp_schedule_v1|20260906|phase|pair_id|fold_0|epoch|purpose",
        "schedule_purposes": ["base_order", "gain", "noise"],
        "seed_hash": "sha256_first_8_bytes_big_endian_unsigned",
        "random_generator": "numpy_PCG64",
        "gain": {"distribution": "uniform", "low": 0.9, "high": 1.1},
        "noise": {
            "distribution": "gaussian",
            "mean": 0.0,
            "std": 0.02,
            "independent_per_position": True,
        },
        "augmentation_construction_dtype": "float64",
        "model_input_dtype": "float32",
        "dataloader_shuffle": False,
    }
    _require(errors, dict(training), expected_training, "training")

    score = _mapping(config.get("score"), "score")
    for name, expected in {
        "s1_expression": "R0_div_R1_div_R1_then_spatial_mean",
        "s2_template_feature": "absolute",
        "s2_test_feature": "raw_signed",
        "s2_grouping": "model_single_view_predicted_class",
        "s2_cross_class_channel_normalization": "sum",
        "s3_g_p_pro_p": 8,
        "s3_grouping": "model_single_view_predicted_class",
        "train_template_population": "raw_unique_train_known_single_view",
        "empty_predicted_class": "hard_failure",
        "pair_prediction": "arithmetic_mean_of_view_probabilities_argmax",
        "pair_score": "arithmetic_mean_same_common_predicted_class",
        "integrated_score": "standardized_S1_plus_S2_plus_S3",
        "unknown_score_direction": "larger_is_more_unknown",
        "linear_unknown_score": "negative_maximum_pair_probability",
        "pcssr_unknown_score": "negative_integrated_pair_score",
    }.items():
        _require(errors, score.get(name), expected, f"score.{name}")
    _require(
        errors,
        dict(_mapping(score.get("score_norm"), "score norm")),
        {
            "variants_per_base": 4,
            "material": "official_cssr_hrrp_score_norm_v1|20260906|pair_id|fold_0|sample_id|variant|purpose",
            "purposes": ["gain", "noise"],
            "population": "augmented_train_known_only",
            "accumulation_dtype": "float64",
            "population_ddof": 0,
            "minimum_std_exclusive": 1.0e-12,
            "epsilon": 1.0e-8,
        },
        "score normalization",
    )

    evaluation = _mapping(config.get("evaluation"), "evaluation")
    _require(
        errors,
        dict(evaluation),
        {
            "threshold_source": "known_calibration_pairs_only",
            "threshold_known_acceptance_rate": 0.95,
            "threshold_semantics": "reuse_existing_evaluator_quantile_and_ties",
            "report_metrics": [
                "known_accuracy",
                "known_macro_f1",
                "auroc",
                "oscr",
                "fpr95",
                "known_correct_acceptance_rate",
                "unknown_rejection_rate",
                "open_set_harmonic_score",
                "k_plus_1_macro_f1",
            ],
            "per_surrogate_identity": True,
            "false_accept_destination": True,
            "view_swap_invariance_required": True,
        },
        "evaluation",
    )
    outputs = _mapping(config.get("outputs"), "outputs")
    for name, expected in {
        "namespace": EXPERIMENT_ID,
        "overwrite_existing": False,
        "save_resolved_config": True,
        "save_manifests": True,
        "save_predictions": True,
        "save_checkpoint_replay": True,
        "save_artifact_hashes": True,
        "confirmation_allowed": False,
        "automatic_followon_authorized": False,
        "final_unknown_test_authorized": False,
    }.items():
        _require(errors, outputs.get(name), expected, f"outputs.{name}")

    gate = _mapping(config.get("pilot_gate"), "pilot gate")
    _require(
        errors,
        list(gate.get("label_priority", [])),
        [
            "official_cssr_strong_signal",
            "official_cssr_method_signal_only",
            "official_cssr_ft_signal_only",
            "official_cssr_score_integration_only",
            "official_cssr_no_signal",
        ],
        "gate priority",
    )
    for section, expected in {
        "strong": {
            "o4_minus_o3_minimum_mean_auroc": 0.02,
            "o4_minus_o3_minimum_positive_pairs": 2,
            "o4_minus_o0_minimum_mean_auroc": 0.01,
            "o4_minus_o0_minimum_positive_pairs": 2,
        },
        "ft": {
            "o2_minus_o1_minimum_mean_auroc": 0.02,
            "o2_minus_o1_minimum_positive_pairs": 2,
            "minimum_mean_oscr_delta": 0.0,
            "minimum_mean_kccr_delta": -0.01,
            "maximum_mean_fpr95_delta": 0.02,
        },
        "score_integration": {
            "full_minus_s1_minimum_mean_auroc": 0.01,
            "minimum_positive_pairs": 2,
        },
        "safe_vs_o0": {
            "minimum_mean_oscr_delta": 0.0,
            "minimum_mean_kccr_delta": -0.01,
            "maximum_mean_fpr95_delta": 0.02,
        },
        "safe_identity": {
            "minimum_auroc": 0.40,
            "minimum_delta_vs_o0": -0.10,
        },
        "hard_failure": {
            "pilot_status": "hard_failed_incomplete",
            "pilot_gate": "not_evaluated",
            "selected_method": None,
        },
    }.items():
        _require(
            errors,
            dict(_mapping(gate.get(section), f"gate.{section}")),
            expected,
            f"gate.{section}",
        )

    runtime = _mapping(config.get("runtime"), "runtime")
    _require(
        errors,
        dict(runtime),
        {
            "formal_device": "cuda",
            "expected_gpu_model": "NVIDIA GeForce RTX 4090",
            "maximum_parallel_tasks": 4,
            "deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "allow_tf32": False,
        },
        "runtime",
    )
    if errors:
        raise DataConfigError("; ".join(errors))


def load_official_cssr_config(path: str | Path) -> dict[str, Any]:
    """Load only the byte-locked preregistered configuration."""

    config_path = Path(path).resolve()
    observed_hash = _file_sha256(config_path)
    if observed_hash != CONFIG_FILE_SHA256:
        raise DataConfigError(
            "official CSSR config bytes changed: "
            f"expected {CONFIG_FILE_SHA256}, observed {observed_hash}"
        )
    with config_path.open("r", encoding="utf-8") as handle:
        config = dict(_mapping(yaml.safe_load(handle), "official CSSR config"))
    config["_config_path"] = str(config_path)
    config["_config_sha256"] = observed_hash
    validate_official_cssr_config(config)
    return config


def build_phase_plan(
    config: Mapping[str, Any],
    phase: str,
) -> list[dict[str, Any]]:
    """Return only the frozen smoke or pilot training tasks; never confirmation."""

    validate_official_cssr_config(config)
    if phase == "smoke":
        pair_ids = ("N1",)
        epochs = 1
        diagnostic_only = True
    elif phase == "pilot":
        pair_ids = PILOT_PAIRS
        epochs = TRAIN_EPOCHS
        diagnostic_only = False
    elif phase in {"confirmation", "final", "final_test", "even_angle_test"}:
        raise DataValidationError(
            "confirmation and final/even-angle tests are not authorized"
        )
    else:
        raise DataValidationError("phase must be smoke or pilot")
    return [
        {
            "experiment_id": EXPERIMENT_ID,
            "phase": phase,
            "pair_id": pair_id,
            "method": method,
            "angle_fold": ANGLE_FOLD,
            "r2_seed": R2_SEED,
            "official_cssr_seed": OFFICIAL_CSSR_SEED,
            "epochs": epochs,
            "diagnostic_only": diagnostic_only,
            "reused_baseline": O0_R2_CC_MLS,
            "confirmation_allowed": False,
            "automatic_followon_authorized": False,
            "final_unknown_test_authorized": False,
            "even_angle_test_authorized": False,
        }
        for pair_id in pair_ids
        for method in TRAINABLE_METHODS
    ]


def _train_population(
    train_rows: Sequence[Mapping[str, Any]],
    train_inputs: np.ndarray,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    values = np.asarray(train_inputs)
    if values.shape != (len(train_rows), INPUT_LENGTH):
        raise DataValidationError("train inputs must align with rows and have shape N x 601")
    if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
        raise DataValidationError("train inputs must be finite floating-point values")
    if len(train_rows) != TRAIN_SAMPLE_COUNT:
        raise DataValidationError("training population is not 5 x 144 unique bases")

    sample_ids: list[str] = []
    labels: list[int] = []
    for row in train_rows:
        if str(row.get("experiment_role")) != "train_known":
            raise DataValidationError("non-train-known sample entered training material")
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            raise DataValidationError("train-known sample ID is empty")
        angle = int(row.get("angle_deg", -1))
        if angle < 0 or angle % 2 != 1:
            raise DataValidationError("even or invalid angle entered training material")
        label = int(row.get("model_label", -1))
        sample_ids.append(sample_id)
        labels.append(label)
    if len(set(sample_ids)) != TRAIN_SAMPLE_COUNT:
        raise DataValidationError("training sample IDs are not unique")
    if Counter(labels) != Counter({index: TRAIN_BASES_PER_CLASS for index in range(CLASS_COUNT)}):
        raise DataValidationError("training labels are not balanced 5 x 144")
    canonical_indices = np.asarray(
        sorted(
            range(TRAIN_SAMPLE_COUNT),
            key=lambda index: (labels[index], sample_ids[index]),
        ),
        dtype=np.int64,
    )
    return canonical_indices, sample_ids, values


def _validate_phase_epoch(phase: str, pair_id: str, epoch: int) -> None:
    if phase == "smoke":
        if pair_id != "N1" or epoch != 1:
            raise DataValidationError("smoke schedule is frozen to N1 epoch 1")
    elif phase == "pilot":
        if pair_id not in PILOT_PAIRS or not 1 <= epoch <= TRAIN_EPOCHS:
            raise DataValidationError("pilot schedule is outside the frozen plan")
    else:
        raise DataValidationError("training schedule phase must be smoke or pilot")


def build_training_epoch_material(
    train_rows: Sequence[Mapping[str, Any]],
    train_inputs: np.ndarray,
    phase: str,
    pair_id: str,
    epoch: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build shared, sample-bound order and HRRP augmentation for one epoch."""

    validate_official_cssr_config(config)
    epoch = int(epoch)
    pair_id = str(pair_id)
    phase = str(phase)
    _validate_phase_epoch(phase, pair_id, epoch)
    canonical_indices, sample_ids, values = _train_population(train_rows, train_inputs)

    materials = {
        purpose: (
            f"official_cssr_hrrp_schedule_v1|{OFFICIAL_CSSR_SEED}|{phase}|"
            f"{pair_id}|fold_0|{epoch}|{purpose}"
        )
        for purpose in ("base_order", "gain", "noise")
    }
    seeds = {purpose: derive_seed(material) for purpose, material in materials.items()}
    order_rng = np.random.Generator(np.random.PCG64(seeds["base_order"]))
    order_within_canonical = order_rng.permutation(TRAIN_SAMPLE_COUNT)
    indices = canonical_indices[order_within_canonical]

    # Gain and noise are generated in canonical sample-ID order, then reordered for
    # batches.  This binds augmentation to sample identity, not model execution order.
    gain_rng = np.random.Generator(np.random.PCG64(seeds["gain"]))
    noise_rng = np.random.Generator(np.random.PCG64(seeds["noise"]))
    canonical_gain = gain_rng.uniform(0.9, 1.1, size=TRAIN_SAMPLE_COUNT).astype(
        np.float64, copy=False
    )
    canonical_noise = noise_rng.normal(
        0.0, 0.02, size=(TRAIN_SAMPLE_COUNT, INPUT_LENGTH)
    ).astype(np.float64, copy=False)
    gain = np.empty(TRAIN_SAMPLE_COUNT, dtype=np.float64)
    noise = np.empty((TRAIN_SAMPLE_COUNT, INPUT_LENGTH), dtype=np.float64)
    gain[canonical_indices] = canonical_gain
    noise[canonical_indices] = canonical_noise
    augmented = (
        gain[:, None] * np.asarray(values, dtype=np.float64) + noise
    ).astype(np.float32)
    scheduled_ids = [sample_ids[int(index)] for index in indices]
    if len(set(scheduled_ids)) != TRAIN_SAMPLE_COUNT:
        raise DataValidationError("epoch schedule does not use each base exactly once")

    batch_sizes = [
        min(TRAIN_BATCH_SIZE, TRAIN_SAMPLE_COUNT - start)
        for start in range(0, TRAIN_SAMPLE_COUNT, TRAIN_BATCH_SIZE)
    ]
    if batch_sizes != [128, 128, 128, 128, 128, 80]:
        raise DataValidationError("training batch boundaries changed")
    audit = {
        "status": "passed",
        "phase": phase,
        "pair_id": pair_id,
        "angle_fold": ANGLE_FOLD,
        "epoch": epoch,
        "sample_count": TRAIN_SAMPLE_COUNT,
        "batch_sizes": batch_sizes,
        "optimizer_updates": len(batch_sizes),
        "sample_usage_exactly_once": True,
        "pair_multiplicity_used": False,
        "known_calibration_used": False,
        "surrogate_unknown_used": False,
        "final_unknown_used": False,
        "even_angle_test_used": False,
        "materials": materials,
        "seeds": seeds,
        "scheduled_sample_ids": scheduled_ids,
        "schedule_sha256": _sequence_sha256(scheduled_ids),
        "sample_id_population_sha256": _sequence_sha256(
            sorted(sample_ids)
        ),
        "gain_sha256": _array_sha256(gain[canonical_indices]),
        "noise_sha256": _array_sha256(noise[canonical_indices]),
        "augmented_inputs_sha256": _array_sha256(augmented[canonical_indices]),
    }
    return {
        "indices": indices,
        "gain": gain,
        "noise": noise,
        "augmented_inputs": augmented,
        "schedule_sha256": audit["schedule_sha256"],
        "gain_sha256": audit["gain_sha256"],
        "noise_sha256": audit["noise_sha256"],
        "augmented_inputs_sha256": audit["augmented_inputs_sha256"],
        "audit": audit,
    }


def build_score_norm_augmentation(
    train_rows: Sequence[Mapping[str, Any]],
    train_inputs: np.ndarray,
    pair_id: str,
    variant: int,
    config: Mapping[str, Any],
    namespace: str = "official",
) -> dict[str, Any]:
    """Build one deterministic, per-sample Stage-B score-normalization variant."""

    validate_official_cssr_config(config)
    pair_id = str(pair_id)
    variant = int(variant)
    if namespace != "official":
        raise DataValidationError("score-normalization namespace must be official")
    if pair_id not in PILOT_PAIRS or variant not in SCORE_NORM_VARIANTS:
        raise DataValidationError("score-normalization request is outside the frozen plan")
    canonical_indices, sample_ids, values = _train_population(train_rows, train_inputs)
    canonical_gains = np.empty(TRAIN_SAMPLE_COUNT, dtype=np.float64)
    canonical_noise = np.empty((TRAIN_SAMPLE_COUNT, INPUT_LENGTH), dtype=np.float64)
    gain_seeds: list[int] = []
    noise_seeds: list[int] = []
    material_rows: list[dict[str, Any]] = []
    for output_index, source_index in enumerate(canonical_indices):
        sample_id = sample_ids[int(source_index)]
        prefix = (
            f"official_cssr_hrrp_score_norm_v1|{OFFICIAL_CSSR_SEED}|{pair_id}|"
            f"fold_0|{sample_id}|{variant}"
        )
        gain_material = f"{prefix}|gain"
        noise_material = f"{prefix}|noise"
        gain_seed = derive_seed(gain_material)
        noise_seed = derive_seed(noise_material)
        gain_seeds.append(gain_seed)
        noise_seeds.append(noise_seed)
        canonical_gains[output_index] = np.random.Generator(
            np.random.PCG64(gain_seed)
        ).uniform(0.9, 1.1)
        canonical_noise[output_index] = np.random.Generator(
            np.random.PCG64(noise_seed)
        ).normal(0.0, 0.02, size=INPUT_LENGTH)
        material_rows.append(
            {
                "sample_id": sample_id,
                "gain": gain_material,
                "noise": noise_material,
            }
        )
    gains = np.empty(TRAIN_SAMPLE_COUNT, dtype=np.float64)
    noise = np.empty((TRAIN_SAMPLE_COUNT, INPUT_LENGTH), dtype=np.float64)
    gains[canonical_indices] = canonical_gains
    noise[canonical_indices] = canonical_noise
    augmented = (
        gains[:, None] * np.asarray(values, dtype=np.float64) + noise
    ).astype(np.float32)
    canonical_ids = [sample_ids[int(index)] for index in canonical_indices]
    audit = {
        "status": "passed",
        "namespace": namespace,
        "pair_id": pair_id,
        "angle_fold": ANGLE_FOLD,
        "variant": variant,
        "sample_count": TRAIN_SAMPLE_COUNT,
        "sample_order": "model_label_then_sample_id",
        "sample_id_population_sha256": _sequence_sha256(canonical_ids),
        "material_sha256": _sequence_sha256(material_rows),
        "gain_seed_sha256": _sequence_sha256(gain_seeds),
        "noise_seed_sha256": _sequence_sha256(noise_seeds),
        "gain_sha256": _array_sha256(gains[canonical_indices]),
        "noise_sha256": _array_sha256(noise[canonical_indices]),
        "augmented_inputs_sha256": _array_sha256(augmented[canonical_indices]),
        "known_calibration_used": False,
        "surrogate_unknown_used": False,
        "final_unknown_used": False,
        "even_angle_test_used": False,
    }
    return {
        "indices": canonical_indices,
        "gain": gains,
        "noise": noise,
        "augmented_inputs": augmented,
        "gain_sha256": audit["gain_sha256"],
        "noise_sha256": audit["noise_sha256"],
        "augmented_inputs_sha256": audit["augmented_inputs_sha256"],
        "audit": audit,
    }


def model_initialization_seed(
    kind: str,
    pair_id: str,
    config: Mapping[str, Any],
) -> int:
    """Return the shared O1/O3 or O2/O4 initialization seed."""

    validate_official_cssr_config(config)
    if pair_id not in PILOT_PAIRS:
        raise DataValidationError("initialization pair is outside the frozen pilot")
    versions = {
        "linear": "official_cssr_hrrp_linear_init_v1",
        "pcssr": "official_cssr_hrrp_pcssr_init_v1",
    }
    if kind not in versions:
        raise DataValidationError("initialization kind must be linear or pcssr")
    # PyTorch accepts the preregistered unsigned 64-bit value directly.  NumPy
    # schedules use their own PCG64 streams and therefore must not truncate it.
    return derive_seed(f"{versions[kind]}|{OFFICIAL_CSSR_SEED}|{pair_id}")


def learning_rates_for_update(
    epoch: int,
    batch_index: int,
    steps_per_epoch: int,
    *,
    head_base_lr: float = 0.05,
    encoder_base_lr: float = 0.005,
    warmup_epochs: int = 2,
    milestone_epochs: Sequence[int] = (25, 35),
    milestone_decay: float = 0.1,
    encoder_unfreeze_epoch: int = 6,
    method: str | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Return head/encoder LR for a zero-based batch index before optimizer.step."""

    if config is not None:
        validate_official_cssr_config(config)
        training = _mapping(config.get("training"), "training")
        head_base_lr = float(training["head_base_lr"])
        encoder_base_lr = float(training["encoder_base_lr"])
        warmup_epochs = int(training["warmup_epochs"])
        milestone_epochs = tuple(int(value) for value in training["milestone_epochs"])
        milestone_decay = float(training["milestone_decay"])
        encoder_unfreeze_epoch = int(training["e2e_encoder_unfreeze_epoch"])
    if method is not None and method not in TRAINABLE_METHODS:
        raise DataValidationError("learning-rate method is outside O1-O4")
    epoch = int(epoch)
    batch_index = int(batch_index)
    steps_per_epoch = int(steps_per_epoch)
    milestones = tuple(int(value) for value in milestone_epochs)
    if (
        not 1 <= epoch <= TRAIN_EPOCHS
        or steps_per_epoch != TRAIN_STEPS_PER_EPOCH
        or not 0 <= batch_index < steps_per_epoch
        or warmup_epochs != 2
        or milestones != (25, 35)
        or not math.isclose(float(milestone_decay), 0.1)
        or encoder_unfreeze_epoch != 6
        or not math.isclose(float(head_base_lr), 0.05)
        or not math.isclose(float(encoder_base_lr), 0.005)
    ):
        raise DataValidationError("learning-rate request changed the frozen schedule")

    if epoch <= warmup_epochs:
        update = (epoch - 1) * steps_per_epoch + batch_index + 1
        head_factor = update / float(warmup_epochs * steps_per_epoch)
    else:
        head_factor = 1.0
        for milestone in milestones:
            if epoch >= milestone:
                head_factor *= float(milestone_decay)
    encoder_factor = 0.0
    encoder_is_e2e = method is None or method in {
        O3_OFFICIAL_LINEAR_E2E,
        O4_OFFICIAL_PCSSR_E2E,
    }
    if encoder_is_e2e and epoch >= encoder_unfreeze_epoch:
        encoder_factor = 1.0
        for milestone in milestones:
            if epoch >= milestone:
                encoder_factor *= float(milestone_decay)
    return {
        "head": float(head_base_lr) * head_factor,
        "encoder": float(encoder_base_lr) * encoder_factor,
    }


def _finite_metric(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise DataValidationError(f"{name} must be finite and in [0, 1]")
    return result


def _hard_failure(reasons: Sequence[str]) -> dict[str, Any]:
    return {
        "pilot_status": "hard_failed_incomplete",
        "pilot_gate": "not_evaluated",
        "result_label": None,
        "selected_method": None,
        "hard_failure_reasons": sorted(set(str(reason) for reason in reasons)),
        "confirmation_allowed": False,
        "automatic_followon_authorized": False,
        "final_unknown_test_authorized": False,
        "even_angle_test_authorized": False,
    }


def _audit_task_rows(task_rows: Sequence[Mapping[str, Any]] | None) -> list[str]:
    if task_rows is None:
        return []
    expected = {(pair_id, method) for pair_id in PILOT_PAIRS for method in TRAINABLE_METHODS}
    observed: dict[tuple[str, str], Mapping[str, Any]] = {}
    reasons: list[str] = []
    for row in task_rows:
        key = (str(row.get("pair_id")), str(row.get("method")))
        if key not in expected:
            reasons.append(f"unexpected_task:{key[0]}:{key[1]}")
            continue
        if key in observed:
            reasons.append(f"duplicate_task:{key[0]}:{key[1]}")
            continue
        observed[key] = row
    for key in sorted(expected - set(observed)):
        reasons.append(f"missing_task:{key[0]}:{key[1]}")
    for key, row in observed.items():
        if str(row.get("status")) != "success" or row.get("audit_passed") is not True:
            reasons.append(f"failed_task:{key[0]}:{key[1]}")
    return reasons


def _metric_maps(
    metric_rows: Sequence[Mapping[str, Any]],
    identity_rows: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[tuple[str, str], dict[str, float]],
    dict[tuple[str, str, str], float],
    list[str],
]:
    metrics: dict[tuple[str, str], dict[str, float]] = {}
    identities: dict[tuple[str, str, str], float] = {}
    reasons: list[str] = []
    expected_metrics = {(pair_id, method) for pair_id in PILOT_PAIRS for method in METHODS}
    for row in metric_rows:
        key = (str(row.get("pair_id")), str(row.get("method")))
        if key not in expected_metrics:
            reasons.append(f"unexpected_metric:{key[0]}:{key[1]}")
            continue
        if key in metrics:
            reasons.append(f"duplicate_metric:{key[0]}:{key[1]}")
            continue
        try:
            metrics[key] = {
                name: _finite_metric(row[name], f"{key}/{name}")
                for name in (
                    "auroc",
                    "oscr",
                    "known_correct_acceptance_rate",
                    "fpr95",
                )
            }
        except (KeyError, TypeError, ValueError, DataValidationError) as error:
            reasons.append(f"invalid_metric:{key[0]}:{key[1]}:{error}")
    for key in sorted(expected_metrics - set(metrics)):
        reasons.append(f"missing_metric:{key[0]}:{key[1]}")

    expected_identities = {
        (pair_id, method, identity)
        for pair_id in PILOT_PAIRS
        for method in METHODS
        for identity in SURROGATE_IDENTITIES[pair_id]
    }
    for row in identity_rows:
        key = (
            str(row.get("pair_id")),
            str(row.get("method")),
            str(row.get("surrogate_identity")),
        )
        if key not in expected_identities:
            reasons.append(f"unexpected_identity:{key[0]}:{key[1]}:{key[2]}")
            continue
        if key in identities:
            reasons.append(f"duplicate_identity:{key[0]}:{key[1]}:{key[2]}")
            continue
        try:
            identities[key] = _finite_metric(row["auroc"], f"{key}/auroc")
        except (KeyError, TypeError, ValueError, DataValidationError) as error:
            reasons.append(f"invalid_identity:{key[0]}:{key[1]}:{key[2]}:{error}")
    for key in sorted(expected_identities - set(identities)):
        reasons.append(f"missing_identity:{key[0]}:{key[1]}:{key[2]}")
    return metrics, identities, reasons


def _comparison(
    metrics: Mapping[tuple[str, str], Mapping[str, float]],
    left: str,
    right: str,
) -> dict[str, Any]:
    rows = []
    for pair_id in PILOT_PAIRS:
        row = {"pair_id": pair_id}
        for metric in ("auroc", "oscr", "known_correct_acceptance_rate", "fpr95"):
            row[f"delta_{metric}"] = float(
                metrics[(pair_id, left)][metric] - metrics[(pair_id, right)][metric]
            )
        rows.append(row)
    means = {
        metric: float(np.mean([row[f"delta_{metric}"] for row in rows]))
        for metric in ("auroc", "oscr", "known_correct_acceptance_rate", "fpr95")
    }
    return {
        "left": left,
        "right": right,
        "pair_deltas": rows,
        "mean_deltas": means,
        "positive_auroc_pair_count": sum(row["delta_auroc"] > 0.0 for row in rows),
    }


def _safe_identity(
    identities: Mapping[tuple[str, str, str], float],
    method: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for pair_id in PILOT_PAIRS:
        for identity in SURROGATE_IDENTITIES[pair_id]:
            candidate = identities[(pair_id, method, identity)]
            baseline = identities[(pair_id, O0_R2_CC_MLS, identity)]
            rows.append(
                {
                    "pair_id": pair_id,
                    "surrogate_identity": identity,
                    "auroc": candidate,
                    "o0_auroc": baseline,
                    "delta_vs_o0": candidate - baseline,
                }
            )
    minimum_auroc = min(row["auroc"] for row in rows)
    minimum_delta = min(row["delta_vs_o0"] for row in rows)
    checks = {
        "minimum_identity_auroc": minimum_auroc + GATE_TOLERANCE >= 0.40,
        "minimum_identity_delta_vs_o0": minimum_delta + GATE_TOLERANCE >= -0.10,
    }
    return {
        "method": method,
        "rows": rows,
        "minimum_auroc": minimum_auroc,
        "minimum_delta_vs_o0": minimum_delta,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _stable(comparison: Mapping[str, Any], margin: float) -> bool:
    return (
        float(comparison["mean_deltas"]["auroc"]) + GATE_TOLERANCE >= margin
        and int(comparison["positive_auroc_pair_count"]) >= 2
    )


def _score_integration_evidence(
    rows: Sequence[Mapping[str, Any]] | None,
) -> tuple[dict[str, Any], list[str]]:
    if rows is None:
        return {
            "supplied": False,
            "qualifying_variants_before_identity_guard": [],
            "variants": {},
        }, []
    expected = {
        (pair_id, method, variant)
        for pair_id in PILOT_PAIRS
        for method in PCSSR_METHODS
        for variant in ("S1", "full")
    }
    observed: dict[tuple[str, str, str], float] = {}
    reasons: list[str] = []
    for row in rows:
        key = (
            str(row.get("pair_id")),
            str(row.get("method")),
            str(row.get("score_variant")),
        )
        if key not in expected:
            reasons.append(f"unexpected_score_ablation:{key[0]}:{key[1]}:{key[2]}")
            continue
        if key in observed:
            reasons.append(f"duplicate_score_ablation:{key[0]}:{key[1]}:{key[2]}")
            continue
        try:
            observed[key] = _finite_metric(row["auroc"], f"{key}/auroc")
        except (KeyError, TypeError, ValueError, DataValidationError) as error:
            reasons.append(f"invalid_score_ablation:{key[0]}:{key[1]}:{key[2]}:{error}")
    for key in sorted(expected - set(observed)):
        reasons.append(f"missing_score_ablation:{key[0]}:{key[1]}:{key[2]}")
    variants: dict[str, Any] = {}
    qualifying: list[str] = []
    if not reasons:
        for method in PCSSR_METHODS:
            pair_deltas = [
                {
                    "pair_id": pair_id,
                    "delta_auroc": observed[(pair_id, method, "full")]
                    - observed[(pair_id, method, "S1")],
                }
                for pair_id in PILOT_PAIRS
            ]
            mean_delta = float(np.mean([row["delta_auroc"] for row in pair_deltas]))
            positive_count = sum(row["delta_auroc"] > 0.0 for row in pair_deltas)
            passed = mean_delta + GATE_TOLERANCE >= 0.01 and positive_count >= 2
            variants[method] = {
                "pair_deltas": pair_deltas,
                "mean_auroc_delta": mean_delta,
                "positive_pair_count": positive_count,
                "passed_before_identity_guard": passed,
            }
            if passed:
                qualifying.append(method)
    return {
        "supplied": True,
        "qualifying_variants_before_identity_guard": qualifying,
        "variants": variants,
    }, reasons


def evaluate_pilot_gate(
    metric_rows: Sequence[Mapping[str, Any]],
    identity_rows: Sequence[Mapping[str, Any]],
    score_ablation_rows: Sequence[Mapping[str, Any]] | None = None,
    *,
    task_rows: Sequence[Mapping[str, Any]] | None = None,
    audit_passed: bool = True,
) -> dict[str, Any]:
    """Apply the preregistered priority gate and never authorize follow-on work."""

    metrics, identities, reasons = _metric_maps(metric_rows, identity_rows)
    score_evidence, score_reasons = _score_integration_evidence(score_ablation_rows)
    reasons.extend(score_reasons)
    reasons.extend(_audit_task_rows(task_rows))
    if audit_passed is not True:
        reasons.append("aggregate_audit_failed")
    if reasons:
        return _hard_failure(reasons)

    comparisons = {
        "O2_minus_O1": _comparison(metrics, O2_OFFICIAL_PCSSR_FT, O1_OFFICIAL_LINEAR_FT),
        "O4_minus_O3": _comparison(metrics, O4_OFFICIAL_PCSSR_E2E, O3_OFFICIAL_LINEAR_E2E),
        "O4_minus_O2": _comparison(metrics, O4_OFFICIAL_PCSSR_E2E, O2_OFFICIAL_PCSSR_FT),
        "O4_minus_O0": _comparison(metrics, O4_OFFICIAL_PCSSR_E2E, O0_R2_CC_MLS),
    }
    safety = {method: _safe_identity(identities, method) for method in PCSSR_METHODS}
    o4_vs_o0 = comparisons["O4_minus_O0"]
    safe_vs_o0_checks = {
        "mean_oscr_not_lower": o4_vs_o0["mean_deltas"]["oscr"] + GATE_TOLERANCE >= 0.0,
        "mean_kccr_drop_at_most_1pp": o4_vs_o0["mean_deltas"]["known_correct_acceptance_rate"]
        + GATE_TOLERANCE
        >= -0.01,
        "mean_fpr95_increase_at_most_2pp": o4_vs_o0["mean_deltas"]["fpr95"]
        <= 0.02 + GATE_TOLERANCE,
        "safe_identity": safety[O4_OFFICIAL_PCSSR_E2E]["passed"],
    }
    safe_vs_o0 = all(safe_vs_o0_checks.values())
    stable_o4_o3 = _stable(comparisons["O4_minus_O3"], 0.02)
    stable_o4_o0 = _stable(o4_vs_o0, 0.01)

    ft = comparisons["O2_minus_O1"]
    ft_checks = {
        "mean_auroc_delta": ft["mean_deltas"]["auroc"] + GATE_TOLERANCE >= 0.02,
        "positive_pair_count": ft["positive_auroc_pair_count"] >= 2,
        "mean_oscr_delta": ft["mean_deltas"]["oscr"] + GATE_TOLERANCE >= 0.0,
        "mean_kccr_delta": ft["mean_deltas"]["known_correct_acceptance_rate"]
        + GATE_TOLERANCE
        >= -0.01,
        "mean_fpr95_delta": ft["mean_deltas"]["fpr95"]
        <= 0.02 + GATE_TOLERANCE,
        "safe_identity": safety[O2_OFFICIAL_PCSSR_FT]["passed"],
    }
    ft_passed = all(ft_checks.values())

    integration_qualifying = [
        method
        for method in score_evidence["qualifying_variants_before_identity_guard"]
        if safety[method]["passed"]
    ]
    score_evidence["qualifying_variants"] = integration_qualifying

    if stable_o4_o3 and stable_o4_o0 and safe_vs_o0:
        label = "official_cssr_strong_signal"
        selected_method: str | None = O4_OFFICIAL_PCSSR_E2E
    elif stable_o4_o3 and safe_vs_o0 and not stable_o4_o0:
        label = "official_cssr_method_signal_only"
        selected_method = O4_OFFICIAL_PCSSR_E2E
    elif ft_passed:
        label = "official_cssr_ft_signal_only"
        selected_method = O2_OFFICIAL_PCSSR_FT
    elif integration_qualifying:
        label = "official_cssr_score_integration_only"
        selected_method = None
    else:
        label = "official_cssr_no_signal"
        selected_method = None

    return {
        "pilot_status": "completed",
        "pilot_gate": "evaluated",
        "result_label": label,
        "selected_method": selected_method,
        "comparisons": comparisons,
        "safe_identity": safety,
        "safe_vs_o0": {
            "checks": safe_vs_o0_checks,
            "passed": safe_vs_o0,
        },
        "ft_checks": {"checks": ft_checks, "passed": ft_passed},
        "score_integration": score_evidence,
        "label_priority_applied": [
            "official_cssr_strong_signal",
            "official_cssr_method_signal_only",
            "official_cssr_ft_signal_only",
            "official_cssr_score_integration_only",
            "official_cssr_no_signal",
        ],
        "confirmation_allowed": False,
        "automatic_followon_authorized": False,
        "final_unknown_test_authorized": False,
        "even_angle_test_authorized": False,
    }
