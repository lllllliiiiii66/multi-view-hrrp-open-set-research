from __future__ import annotations

import argparse
import concurrent.futures
import csv
import fcntl
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from hrrp_osr.data.errors import DataConfigError, DataValidationError
from hrrp_osr.data.manifest import file_sha256
from hrrp_osr.data.processed import load_processed_bundle
from hrrp_osr.evaluation.metrics import binary_auroc, evaluate_open_set
from hrrp_osr.evaluation.ms_mean_factorial import (
    FACTORIAL_COMPARISONS,
    extract_report_metrics,
    summarize_factorial_results,
)
from hrrp_osr.models.arpl import ARPLReciprocalHead
from hrrp_osr.models.ms_mean_factorial import (
    ARPL_METHODS,
    CE_METHODS,
    METHODS,
    MSMeanHeadFactorialModel,
    clone_state_dict,
)
from hrrp_osr.training.arpl_mv_evidence import _length_padding_audit
from hrrp_osr.training.arpl_pilot import (
    SOURCE_KNOWN_ORDER,
    PreparedSurrogateSplit,
    _environment,
    _is_finite_model,
    _resolve_device,
    _set_determinism,
    _state_sha256,
    prepare_surrogate_split,
)
from hrrp_osr.training.mv_rpformer import (
    IndexedPairDataset,
    _calibration_diagnostics,
    _configure_torch_runtime,
    _head_diagnostics,
    learning_rate_for_epoch,
)


EXPERIMENT_ID = "ms_mean_head_factorial_surrogate_v1"
CONFIG_RELATIVE_PATH = (
    "configs/experiments/arpl/ms_mean_head_factorial_surrogate_v1.yaml"
)
IDENTITY_PAIRS = (
    ("N0", (0, 2), (1, 3, 4, 5, 6)),
    ("N1", (2, 5), (0, 1, 3, 4, 6)),
    ("N2", (3, 5), (0, 1, 2, 4, 6)),
    ("N3", (1, 3), (0, 2, 4, 5, 6)),
    ("N4", (1, 6), (0, 2, 3, 4, 5)),
    ("N5", (4, 6), (0, 1, 2, 3, 5)),
    ("N6", (0, 4), (1, 2, 3, 5, 6)),
)
PROHIBITED_PAIRS = frozenset(
    {
        (0, 1),
        (2, 3),
        (4, 5),
        (0, 6),
        (1, 5),
        (2, 4),
        (3, 6),
    }
)
ANGLE_FOLDS = (0, 4)
CONFIRMATION_SEEDS = (20260830, 20260831, 20260832)
METRIC_KEYS = (
    "known_accuracy",
    "known_macro_f1",
    "auroc",
    "oscr",
    "fpr95",
    "known_correct_acceptance_rate",
    "unknown_rejection_rate",
    "open_set_harmonic_score",
    "k_plus_1_macro_f1",
)
TASK_SOURCE_FILES = (
    CONFIG_RELATIVE_PATH,
    "src/hrrp_osr/amdr/data.py",
    "src/hrrp_osr/data/manifest.py",
    "src/hrrp_osr/data/processed.py",
    "src/hrrp_osr/evaluation/metrics.py",
    "src/hrrp_osr/evaluation/ms_mean_factorial.py",
    "src/hrrp_osr/models/arpl.py",
    "src/hrrp_osr/models/cnn1d.py",
    "src/hrrp_osr/models/hrrp_ms_resnet.py",
    "src/hrrp_osr/models/mv_rpformer.py",
    "src/hrrp_osr/models/ms_mean_factorial.py",
    "src/hrrp_osr/training/arpl_pilot.py",
    "src/hrrp_osr/training/arpl_mv_evidence.py",
    "src/hrrp_osr/training/mv_rpformer.py",
    "src/hrrp_osr/training/ms_mean_head_factorial.py",
)


class NumericalInstabilityError(RuntimeError):
    pass


class IntentionalTrainingInterruption(RuntimeError):
    """Test-only fault injection after a recoverable epoch checkpoint."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DataConfigError(f"{name} must be a mapping")
    return value


def _write_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    _atomic_write_bytes(path, payload)


def _atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _render_csv(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise DataValidationError("cannot render an empty CSV")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _render_csv_with_fields(
    rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]
) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fieldnames))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_write_bytes(path, _render_csv(rows))


ABSORPTION_FIELDS = (
    "pair_id",
    "method",
    "surrogate_identity",
    "absorbed_as_known_identity",
    "false_accept_count",
    "rate_over_all_surrogate",
    "composition_within_false_accepts",
    "total_surrogate_count",
    "total_false_accept_count",
)
ABSORPTION_OVERALL_FIELDS = (
    "method",
    "surrogate_identity",
    "absorbed_as_known_identity",
    "false_accept_count",
    "rate_over_all_surrogate",
    "composition_within_false_accepts",
    "total_surrogate_count",
    "total_false_accept_count",
    "identity_pair_context_count",
)


def _artifact_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }


def task_source_hashes(project_root: Path) -> dict[str, str]:
    result = {}
    for relative in TASK_SOURCE_FILES:
        path = project_root / relative
        if not path.is_file():
            raise DataValidationError(f"task source file is missing: {relative}")
        result[relative] = file_sha256(path)
    return result


def _expected_pair_rows() -> list[dict[str, Any]]:
    return [
        {
            "pair_id": pair_id,
            "surrogate_unknown_indices": list(unknown),
            "train_known_indices": list(train),
        }
        for pair_id, unknown, train in IDENTITY_PAIRS
    ]


def load_ms_mean_head_factorial_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = dict(_mapping(yaml.safe_load(handle), "factorial config"))
    errors: list[str] = []
    if (
        config.get("schema_version") != 1
        or config.get("stage") != "P3_surrogate_ms_mean_head_factorial"
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("result_scope")
        != "diagnostic_smoke_then_mandatory_confirmation_known_source_only"
    ):
        errors.append("experiment identity changed")
    scope = _mapping(config.get("evidence_scope"), "evidence_scope")
    if scope.get("source_known_odd_angle_only") is not True or any(
        scope.get(name) is not False
        for name in (
            "final_unknown_classes_used",
            "even_angle_test_used",
            "surrogate_unknown_used_for_training",
            "surrogate_unknown_used_for_normalization",
            "surrogate_unknown_used_for_threshold",
            "surrogate_unknown_used_for_checkpoint_selection",
            "angle_or_view_metadata_used_by_model",
        )
    ):
        errors.append("evidence isolation changed")
    prior = _mapping(config.get("prior_experiment"), "prior_experiment")
    prohibited = {
        tuple(sorted(int(value) for value in pair))
        for pair in prior.get("prohibited_surrogate_pairs", [])
    }
    if (
        prior.get("source_commit")
        != "ccb30e18b4aa9e78e136ba330dddefb11aab11ae"
        or prohibited != PROHIBITED_PAIRS
    ):
        errors.append("prior experiment boundary changed")
    reference = _mapping(config.get("official_reference"), "official_reference")
    if (
        reference.get("repository") != "https://github.com/gary23ai/ARPL"
        or reference.get("commit")
        != "3ede8b38e1cfb9d70e106cc19d563453110c36ab"
        or reference.get("dist_sha256")
        != "a05fc01c9051d8cb8d87cc7183e0a3d9fd1a11ca9de38d58a4870cb70ad4dc62"
        or reference.get("arploss_sha256")
        != "6dec41f0265b6665e8c66a27f506f176a0a7b0b2e4426760c09c203ab0c327ec"
    ):
        errors.append("official ARPL reference changed")
    bundle = _mapping(config.get("bundle"), "bundle")
    if (
        bundle.get("dataset_id") != "hrrp_10class_theta83_hh_v1"
        or bundle.get("preprocessing_id") != "hrrp_padding_complex_gaussian_v1"
        or bundle.get("profiles_sha256")
        != "2dd92282c125f0f677cf1f2dfce828781c8ba4385cf9ae552c4a2c56033c3f5b"
        or bundle.get("manifest_sha256")
        != "748b9f30629c3b3cbe66c6a1dac30863fdab2d81a214e46d8bc3ef7c6022a08a"
        or bundle.get("bundle_sha256")
        != "79176091b5b745e5df5957f1bb4ade7304c1403a4a32e64622e012db67d5a0c5"
    ):
        errors.append("processed bundle contract changed")
    classes = _mapping(config.get("classes"), "classes")
    if list(classes.get("source_known_order", [])) != list(SOURCE_KNOWN_ORDER):
        errors.append("source known class order changed")
    if classes.get("final_unknown_count") != 3:
        errors.append("final unknown count changed")
    if list(classes.get("identity_pairs", [])) != _expected_pair_rows():
        errors.append("identity pair list changed")
    if list(classes.get("angle_folds", [])) != list(ANGLE_FOLDS):
        errors.append("angle folds changed")
    observed_new = {
        tuple(sorted(row["surrogate_unknown_indices"]))
        for row in classes.get("identity_pairs", [])
        if isinstance(row, Mapping)
    }
    occurrences = [0] * 7
    for pair in observed_new:
        for index in pair:
            if 0 <= int(index) < 7:
                occurrences[int(index)] += 1
    if observed_new & PROHIBITED_PAIRS or occurrences != [2] * 7:
        errors.append("identity pair balance or novelty failed")
    sampling = _mapping(config.get("sampling"), "sampling")
    if (
        sampling.get("pair_protocol_source") != "arpl_lite_surrogate_osr_v1"
        or sampling.get("pair_protocol_scope")
        != "experiment_angle_fold_shared_across_identity_pairs"
        or sampling.get("identity_pair_only_filters_class_roles") is not True
        or
        sampling.get("fold_count") != 5
        or sampling.get("development_angle_parity") != "odd"
        or sampling.get("base_seed") != 20260830
        or sampling.get("slot_order") != "randomized_seeded"
        or sampling.get("distinct_frames") is not True
        or dict(_mapping(sampling.get("pairs_per_class"), "pairs_per_class"))
        != {"smoke": 10, "full": 500}
        or sampling.get("final_test_pairs_generated") is not False
    ):
        errors.append("sampling protocol changed")
    normalization = _mapping(config.get("normalization"), "normalization")
    if (
        normalization.get("method") != "global_scalar_zscore"
        or normalization.get("fit_population")
        != "unique_train_known_base_samples_only"
        or float(normalization.get("epsilon", -1)) != 1e-8
    ):
        errors.append("normalization protocol changed")
    model = _mapping(config.get("model"), "model")
    shallow = _mapping(model.get("shallow_encoder"), "model.shallow_encoder")
    multiscale = _mapping(model.get("multiscale_encoder"), "multiscale_encoder")
    ce = _mapping(model.get("ce"), "model.ce")
    arpl = _mapping(model.get("arpl"), "model.arpl")
    if (
        list(model.get("methods", [])) != list(METHODS)
        or list(model.get("input_shape", [])) != [2, 601]
        or model.get("feature_dim") != 128
        or model.get("fusion") != "arithmetic_mean"
        or shallow.get("architecture") != "shared_hrrp_encoder_1d_v1"
        or shallow.get("shared_between_views") is not True
        or multiscale.get("architecture") != "hrrp_ms_resnet_v1"
        or multiscale.get("shared_between_views") is not True
        or multiscale.get("input_length") != 601
        or multiscale.get("stem_channels") != 32
        or multiscale.get("stem_kernel_size") != 31
        or list(multiscale.get("stage_channels", [])) != [32, 64, 128]
        or list(multiscale.get("branch_kernel_sizes", [])) != [3, 7, 15]
        or multiscale.get("branch_convolution") != "standard_conv1d"
        or multiscale.get("activation") != "GELU"
        or float(multiscale.get("dropout", -1)) != 0.1
        or list(multiscale.get("pooling", [])) != ["global_average", "global_max"]
        or multiscale.get("projection_dim") != 128
        or ce.get("head") != "linear"
        or float(ce.get("weight_init_std", -1)) != 0.01
        or float(ce.get("bias_init", -1)) != 0.0
        or arpl.get("algorithm_id")
        != "arpl_official_loss_one_center_device_safe_v1"
        or arpl.get("num_centers_per_class") != 1
        or float(arpl.get("temperature", -1)) != 1.0
        or float(arpl.get("weight_pl", -1)) != 0.1
        or float(arpl.get("margin", -1)) != 1.0
        or float(arpl.get("reciprocal_init_std", -1)) != 0.1
        or float(arpl.get("initial_radius", -1)) != 0.0
    ):
        errors.append("model contract changed")
    forbidden = set(model.get("forbidden_components", []))
    if forbidden != {
        "SAB",
        "PMA",
        "attention",
        "view_level_head",
        "reject_token",
        "learned_rejector",
        "pseudo_unknown",
        "confusing_samples",
        "GAN",
    }:
        errors.append("forbidden component list changed")
    training = _mapping(config.get("training"), "training")
    if (
        list(training.get("methods", [])) != list(METHODS)
        or list(training.get("confirmation_seeds", [])) != list(CONFIRMATION_SEEDS)
        or training.get("optimizer") != "AdamW"
        or float(training.get("learning_rate", -1)) != 3e-4
        or float(training.get("weight_decay", -1)) != 1e-4
        or int(training.get("batch_size", 0)) != 64
        or int(training.get("total_epochs", 0)) != 100
        or int(training.get("smoke_epochs", 0)) != 1
        or training.get("scheduler") != "warmup_cosine"
        or int(training.get("warmup_epochs", 0)) != 5
        or int(training.get("formal_checkpoint_epoch", 0)) != 100
        or training.get("early_stopping") is not False
        or training.get("calibration_checkpoint_selection") is not False
        or training.get("performance_fallback") is not False
        or training.get("data_augmentation") != "none"
        or int(training.get("dataloader_seed_offset", -1)) != 1
        or training.get("deterministic_algorithms") is not True
        or int(training.get("num_workers", -1)) != 0
    ):
        errors.append("training contract changed")
    evaluation = _mapping(config.get("evaluation"), "evaluation")
    if (
        evaluation.get("unknown_score") != "negative_maximum_global_raw_logit"
        or evaluation.get("unknown_score_direction") != "larger_is_more_unknown"
        or evaluation.get("threshold_source") != "known_calibration_only"
        or float(evaluation.get("threshold_known_acceptance_rate", -1)) != 0.95
        or list(evaluation.get("report_metrics", [])) != list(METRIC_KEYS)
    ):
        errors.append("evaluation contract changed")
    decision = _mapping(config.get("decision"), "decision")
    backbone_gate = _mapping(decision.get("backbone_gate"), "backbone_gate")
    head_gate = _mapping(decision.get("head_gate"), "head_gate")
    if (
        float(backbone_gate.get("minimum_mean_auroc_delta", -1)) != 0.03
        or int(backbone_gate.get("minimum_positive_pair_count", 0)) != 6
        or float(backbone_gate.get("minimum_mean_oscr_delta", -1)) != 0.0
        or float(backbone_gate.get("maximum_mean_known_accuracy_drop", -1)) != 0.005
        or float(backbone_gate.get("maximum_mean_fpr95_increase", -1)) != 0.02
        or float(head_gate.get("minimum_mean_auroc_delta", -1)) != 0.01
        or int(head_gate.get("minimum_positive_pair_count", 0)) != 5
        or float(head_gate.get("minimum_mean_oscr_delta", -1)) != 0.0
        or float(head_gate.get("maximum_mean_known_accuracy_drop", -1)) != 0.005
        or float(head_gate.get("maximum_mean_fpr95_increase", -1)) != 0.02
        or decision.get("final_unknown_test_authorized") is not False
    ):
        errors.append("decision rule changed")
    uncertainty = _mapping(config.get("uncertainty"), "uncertainty")
    if (
        uncertainty.get("bootstrap_statistical_unit") != "identity_pair"
        or uncertainty.get("bootstrap_resamples") != 10000
        or uncertainty.get("bootstrap_seed") != 20260903
        or float(uncertainty.get("confidence_level", -1)) != 0.95
    ):
        errors.append("uncertainty protocol changed")
    length_audit = _mapping(config.get("length_padding_audit"), "length_padding_audit")
    if (
        length_audit.get("diagnostic_only") is not True
        or length_audit.get("used_for_gate") is not False
        or length_audit.get("delete_or_replace_pair") is not False
        or length_audit.get("length_safe_definition")
        != "both_folds_all_surrogate_original_lengths_within_inclusive_train_known_range"
        or length_audit.get("length_risk_definition")
        != "at_least_one_fold_has_surrogate_original_length_outside_inclusive_train_known_range"
        or length_audit.get("report_length_only_auroc") is not True
        or length_audit.get("report_all_safe_and_risk_subsets") is not True
    ):
        errors.append("length/padding audit contract changed")
    comparisons = _mapping(config.get("comparisons"), "comparisons")
    if (
        list(comparisons.get("backbone_arpl", [])) != [METHODS[3], METHODS[1]]
        or list(comparisons.get("backbone_ce", [])) != [METHODS[2], METHODS[0]]
        or list(comparisons.get("head_multiscale", [])) != [METHODS[3], METHODS[2]]
        or list(comparisons.get("head_shallow", [])) != [METHODS[1], METHODS[0]]
        or comparisons.get("interaction") != "(R3-R2)-(R1-R0)"
        or comparisons.get("primary_statistical_unit") != "identity_pair"
        or comparisons.get("unit_aggregation_order")
        != "mean_over_2_folds_x_3_seeds_then_aggregate_7_pairs"
    ):
        errors.append("factorial comparison contract changed")
    runtime = _mapping(config.get("runtime"), "runtime")
    if (
        runtime.get("formal_device") != "cuda"
        or runtime.get("expected_gpu_model") != "NVIDIA GeForce RTX 4090"
        or int(runtime.get("expected_gpu_count", 0)) != 4
        or int(runtime.get("jobs_per_gpu", 0)) != 4
        or int(runtime.get("total_parallel_jobs", 0)) != 16
        or runtime.get("gpu_assignment") != "rotating_method_balanced_v1"
        or int(runtime.get("torch_intraop_threads", 0)) != 4
        or int(runtime.get("torch_interop_threads", 0)) != 1
        or runtime.get("cublas_workspace_config") != ":4096:8"
        or runtime.get("amp") is not False
        or runtime.get("tf32") is not False
        or runtime.get("torch_compile") is not False
    ):
        errors.append("runtime contract changed")
    outputs = _mapping(config.get("outputs"), "outputs")
    if (
        outputs.get("namespace")
        != "artifacts/arpl/ms_mean_head_factorial_surrogate_v1"
        or outputs.get("fail_if_output_nonempty") is not True
        or any(value is not True for key, value in outputs.items() if key != "namespace")
    ):
        errors.append("output contract changed")
    if errors:
        raise DataConfigError("Invalid factorial config:\n- " + "\n- ".join(errors))
    config["_config_path"] = str(config_path)
    config["_config_sha256"] = file_sha256(config_path)
    return config


def _specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            **dict(pair),
            "angle_fold": int(angle_fold),
            "unit_id": f"{pair['pair_id']}_F{angle_fold}",
        }
        for pair in config["classes"]["identity_pairs"]
        for angle_fold in config["classes"]["angle_folds"]
    ]


def build_phase_plan(config: Mapping[str, Any], phase: str) -> list[dict[str, Any]]:
    specs = _specs(config)
    if phase == "smoke":
        specs = [spec for spec in specs if spec["pair_id"] == "N0" and spec["angle_fold"] == 0]
        seeds = [CONFIRMATION_SEEDS[0]]
        mode = "smoke"
    elif phase == "confirmation":
        seeds = list(CONFIRMATION_SEEDS)
        mode = "full"
    else:
        raise DataConfigError("phase must be smoke or confirmation")
    return [
        {
            "phase": phase,
            "mode": mode,
            "spec": spec,
            "pair_id": spec["pair_id"],
            "angle_fold": spec["angle_fold"],
            "seed": seed,
            "methods": list(METHODS),
        }
        for spec in specs
        for seed in seeds
    ]


def _prepare_split(
    bundle: Any,
    config: Mapping[str, Any],
    spec: Mapping[str, Any],
    mode: str,
) -> PreparedSurrogateSplit:
    # Freeze pair sampling by experiment + angle fold, not by identity pair.
    # Consequently every source class shared by two N-pairs uses the same
    # sampled base pairs in a fold; N only changes the 5-known/2-surrogate role.
    shared = prepare_surrogate_split(
        bundle,
        source_known_order=config["classes"]["source_known_order"],
        split_id=f"{EXPERIMENT_ID}_F{int(spec['angle_fold'])}",
        angle_fold=int(spec["angle_fold"]),
        train_known_indices=spec["train_known_indices"],
        surrogate_unknown_indices=spec["surrogate_unknown_indices"],
        pairs_per_class=int(config["sampling"]["pairs_per_class"][mode]),
        base_seed=int(config["sampling"]["base_seed"]),
        fold_count=int(config["sampling"]["fold_count"]),
        normalization_epsilon=float(config["normalization"]["epsilon"]),
    )
    unit_id = str(spec["unit_id"])
    manifest_rows = []
    for source_row in shared.pair_manifest_rows:
        row = dict(source_row)
        row["surrogate_split_id"] = unit_id
        manifest_rows.append(row)
    manifest_bytes = _render_csv(manifest_rows)
    pair_audit = dict(shared.pair_audit)
    pair_audit.update(
        {
            "split_id": unit_id,
            "pair_protocol_scope": "experiment_angle_fold_shared_across_identity_pairs",
            "pair_protocol_split_id": shared.split_id,
            "identity_pair_only_filters_class_roles": True,
        }
    )
    return PreparedSurrogateSplit(
        split_id=unit_id,
        angle_fold=shared.angle_fold,
        train_class_order=shared.train_class_order,
        surrogate_class_order=shared.surrogate_class_order,
        pair_manifest_rows=tuple(manifest_rows),
        pair_manifest_bytes=manifest_bytes,
        pair_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        pair_audit=pair_audit,
        normalization=shared.normalization,
        inputs=shared.inputs,
        labels=shared.labels,
        pair_ids=shared.pair_ids,
        class_names=shared.class_names,
    )


def _build_model(
    method: str, known_class_count: int, config: Mapping[str, Any]
) -> MSMeanHeadFactorialModel:
    arpl = config["model"]["arpl"]
    return MSMeanHeadFactorialModel(
        method,
        known_class_count,
        feature_dim=int(config["model"]["feature_dim"]),
        dropout=float(config["model"]["multiscale_encoder"]["dropout"]),
        temperature=float(arpl["temperature"]),
        weight_pl=float(arpl["weight_pl"]),
        margin=float(arpl["margin"]),
        reciprocal_init_std=float(arpl["reciprocal_init_std"]),
        initial_radius=float(arpl["initial_radius"]),
        ce_weight_init_std=float(config["model"]["ce"]["weight_init_std"]),
    )


def build_initialized_method_group(
    known_class_count: int,
    *,
    seed: int,
    config: Mapping[str, Any],
) -> tuple[dict[str, MSMeanHeadFactorialModel], dict[str, Any]]:
    models = {}
    for method in METHODS:
        _set_determinism(seed, bool(config["training"]["deterministic_algorithms"]))
        models[method] = _build_model(method, known_class_count, config)
    models["R1_SHALLOW_MEAN_ARPL"].encoder.load_state_dict(
        clone_state_dict(models["R0_SHALLOW_MEAN_CE"].encoder.state_dict())
    )
    models["R3_MS_MEAN_ARPL"].encoder.load_state_dict(
        clone_state_dict(models["R2_MS_MEAN_CE"].encoder.state_dict())
    )
    models["R2_MS_MEAN_CE"].global_head.load_state_dict(
        clone_state_dict(models["R0_SHALLOW_MEAN_CE"].global_head.state_dict())
    )
    models["R3_MS_MEAN_ARPL"].global_head.load_state_dict(
        clone_state_dict(models["R1_SHALLOW_MEAN_ARPL"].global_head.state_dict())
    )
    hashes = {
        method: {
            "backbone": _state_sha256(model.encoder.state_dict()),
            "head": _state_sha256(model.global_head.state_dict()),
        }
        for method, model in models.items()
    }
    checks = {
        "shallow_backbone_equal": hashes[METHODS[0]]["backbone"]
        == hashes[METHODS[1]]["backbone"],
        "multiscale_backbone_equal": hashes[METHODS[2]]["backbone"]
        == hashes[METHODS[3]]["backbone"],
        "ce_head_equal": hashes[METHODS[0]]["head"] == hashes[METHODS[2]]["head"],
        "arpl_head_equal": hashes[METHODS[1]]["head"] == hashes[METHODS[3]]["head"],
        "independent_model_objects": len({id(model) for model in models.values()}) == 4,
        "forbidden_components_absent": all(
            not any(model.forbidden_component_status.values()) for model in models.values()
        ),
    }
    if not all(checks.values()):
        raise DataValidationError("factorial initialization audit failed")
    return models, {"seed": seed, "component_hashes": hashes, "checks": checks}


def _runtime_contract(
    device: torch.device, config: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "device": str(device),
        "device_type": device.type,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_version": torch.version.cuda,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": bool(config["training"]["deterministic_algorithms"]),
        "amp": False,
        "tf32": False,
        "torch_compile": False,
    }


def _training_order_sha256(epoch_hashes: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(epoch_hashes), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def train_one_method(
    model: MSMeanHeadFactorialModel,
    *,
    method: str,
    prepared: PreparedSurrogateSplit,
    seed: int,
    config: Mapping[str, Any],
    mode: str,
    device: torch.device,
    resume_checkpoint: Path | None = None,
    _interrupt_after_epoch: int | None = None,
) -> dict[str, Any]:
    if method not in METHODS or model.method != method:
        raise DataValidationError("factorial method/model mismatch")
    training = config["training"]
    _set_determinism(seed, bool(training["deterministic_algorithms"]))
    model = model.to(device)
    train_dataset = IndexedPairDataset(prepared.inputs["train"], prepared.labels["train"])
    calibration_dataset = IndexedPairDataset(
        prepared.inputs["known_calibration"], prepared.labels["known_calibration"]
    )
    dataloader_generator = torch.Generator().manual_seed(
        seed + int(training["dataloader_seed_offset"])
    )
    loader = DataLoader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        generator=dataloader_generator,
        num_workers=int(training["num_workers"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    project_root = Path(config["_config_path"]).parents[3]
    source_hashes = task_source_hashes(project_root)
    runtime = _runtime_contract(device, config)
    epochs = int(training["smoke_epochs"] if mode == "smoke" else training["total_epochs"])
    log: list[dict[str, Any]] = []
    epoch_order_hashes: list[str] = []
    start_epoch = 1
    if resume_checkpoint is not None and resume_checkpoint.exists():
        state = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        expected = (
            EXPERIMENT_ID,
            method,
            seed,
            mode,
            prepared.pair_manifest_sha256,
            config["_config_sha256"],
            tuple(prepared.train_class_order),
            source_hashes,
            runtime,
        )
        observed = (
            state.get("experiment_id"),
            state.get("method"),
            int(state.get("seed", -1)),
            state.get("mode"),
            state.get("pair_manifest_sha256"),
            state.get("config_sha256"),
            tuple(state.get("train_class_order", ())),
            state.get("source_hashes"),
            state.get("runtime_contract"),
        )
        if observed != expected:
            raise DataValidationError("resume checkpoint contract differs")
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        dataloader_generator.set_state(state["dataloader_generator_state"])
        torch.set_rng_state(state["torch_rng_state"])
        if device.type == "cuda" and state.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
        log = [dict(row) for row in state["training_log"]]
        epoch_order_hashes = list(state["epoch_order_hashes"])
        start_epoch = int(state["completed_epoch"]) + 1
    for epoch in range(start_epoch, epochs + 1):
        started = time.perf_counter()
        learning_rate = learning_rate_for_epoch(
            epoch,
            base_learning_rate=float(training["learning_rate"]),
            warmup_epochs=int(training["warmup_epochs"]),
            total_epochs=int(training["total_epochs"]),
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train()
        totals = {
            "count": 0,
            "loss": 0.0,
            "classification": 0.0,
            "margin": 0.0,
            "correct": 0,
        }
        order_hasher = hashlib.sha256()
        order_hasher.update(f"epoch:{epoch}\n".encode("ascii"))
        for batch_inputs, labels, indices in loader:
            order_hasher.update(indices.numpy().astype("<i8", copy=False).tobytes())
            batch_inputs = batch_inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch_inputs, compute_rejector=False)
            loss = model.representation_loss(output, labels, lambda_view=0.0)
            total_loss = loss["total"]
            if not torch.isfinite(total_loss):
                raise NumericalInstabilityError(f"{method} loss became NaN or Inf")
            total_loss.backward()
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise NumericalInstabilityError(f"{method} gradient became NaN or Inf")
            optimizer.step()
            if not _is_finite_model(model):
                raise NumericalInstabilityError(f"{method} parameter became NaN or Inf")
            count = int(labels.numel())
            totals["count"] += count
            totals["loss"] += float(total_loss.item()) * count
            totals["classification"] += float(loss["global_classification"].item()) * count
            totals["margin"] += float(loss["global_margin"].item()) * count
            totals["correct"] += int(
                (output.global_logits.argmax(dim=1) == labels).sum().item()
            )
        epoch_order_hashes.append(order_hasher.hexdigest())
        calibration_accuracy, calibration_macro_f1 = _calibration_diagnostics(
            model,
            calibration_dataset,
            device=device,
            batch_size=int(training["batch_size"]),
        )
        count = int(totals["count"])
        log.append(
            {
                "epoch": epoch,
                "method": method,
                "learning_rate": learning_rate,
                "train_loss": totals["loss"] / count,
                "train_classification_loss": totals["classification"] / count,
                "train_margin_loss": totals["margin"] / count,
                "train_accuracy": totals["correct"] / count,
                "known_calibration_accuracy_diagnostic": calibration_accuracy,
                "known_calibration_macro_f1_diagnostic": calibration_macro_f1,
                "checkpoint_selected_for_open_set_performance": False,
                "pseudo_unknown_generated": False,
                "train_order_epoch_sha256": epoch_order_hashes[-1],
                "elapsed_seconds": time.perf_counter() - started,
                **_head_diagnostics(model),
            }
        )
        if resume_checkpoint is not None:
            _atomic_torch_save(
                resume_checkpoint,
                {
                    "experiment_id": EXPERIMENT_ID,
                    "method": method,
                    "seed": seed,
                    "mode": mode,
                    "pair_manifest_sha256": prepared.pair_manifest_sha256,
                    "config_sha256": config["_config_sha256"],
                    "train_class_order": prepared.train_class_order,
                    "source_hashes": source_hashes,
                    "runtime_contract": runtime,
                    "completed_epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "dataloader_generator_state": dataloader_generator.get_state(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state_all": (
                        torch.cuda.get_rng_state_all() if device.type == "cuda" else None
                    ),
                    "training_log": log,
                    "epoch_order_hashes": epoch_order_hashes,
                    "pseudo_unknown_generated": False,
                },
            )
        if _interrupt_after_epoch == epoch:
            if resume_checkpoint is None:
                raise ValueError("fault injection requires a resume checkpoint path")
            raise IntentionalTrainingInterruption(
                f"intentional interruption after checkpointed epoch {epoch}"
            )
    model.eval()
    return {
        "model": model,
        "final_state": clone_state_dict(model.state_dict()),
        "checkpoint_epoch": epochs,
        "formal_checkpoint": mode == "full"
        and epochs == int(training["formal_checkpoint_epoch"]),
        "training_log": log,
        "source_hashes": source_hashes,
        "runtime_contract": runtime,
        "training_order_sha256": _training_order_sha256(epoch_order_hashes),
        "epoch_order_hashes": epoch_order_hashes,
        "pseudo_audit": {
            "status": "not_applicable",
            "pseudo_unknown_generated": False,
            "pseudo_count": 0,
            "surrogate_unknown_used": False,
            "final_unknown_used": False,
            "even_angle_test_used": False,
        },
    }


def infer_model(
    model: MSMeanHeadFactorialModel,
    inputs: np.ndarray,
    labels: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    dataset = IndexedPairDataset(inputs, labels)
    names = (
        "per_view_features",
        "fused_features",
        "per_view_logits",
        "global_logits",
        "unknown_score",
        "labels",
    )
    collected: dict[str, list[np.ndarray]] = {name: [] for name in names}
    model.eval()
    with torch.no_grad():
        for batch_inputs, batch_labels, _ in DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        ):
            output = model(batch_inputs.to(device), compute_rejector=False)
            tensors = {
                "per_view_features": output.raw_view_tokens,
                "fused_features": output.global_class_token,
                "per_view_logits": output.per_view_logits,
                "global_logits": output.global_logits,
                "unknown_score": -output.global_logits.max(dim=1).values,
            }
            for name, tensor in tensors.items():
                dtype = np.float64 if name in {"per_view_logits", "global_logits", "unknown_score"} else np.float32
                collected[name].append(tensor.detach().cpu().numpy().astype(dtype))
            collected["labels"].append(batch_labels.numpy().astype(np.int64))
    result = {name: np.concatenate(values, axis=0) for name, values in collected.items()}
    if not all(np.isfinite(value).all() for value in result.values()):
        raise NumericalInstabilityError("inference produced NaN or Inf")
    return result


def permutation_audit(
    model: MSMeanHeadFactorialModel,
    inputs: np.ndarray,
    *,
    device: torch.device,
) -> dict[str, Any]:
    values = torch.from_numpy(np.asarray(inputs[: min(16, len(inputs))], dtype=np.float32)).to(device)
    model.eval()
    with torch.no_grad():
        original = model(values, compute_rejector=False)
        swapped = model(values[:, [1, 0]], compute_rejector=False)
    comparisons = {
        "per_view_features_equivariant": (
            original.raw_view_tokens[:, [1, 0]],
            swapped.raw_view_tokens,
        ),
        "fused_features_invariant": (
            original.global_class_token,
            swapped.global_class_token,
        ),
        "per_view_logits_equivariant": (
            original.per_view_logits[:, [1, 0]],
            swapped.per_view_logits,
        ),
        "global_logits_invariant": (original.global_logits, swapped.global_logits),
    }
    maximum = {
        name: float((left - right).abs().max().cpu())
        for name, (left, right) in comparisons.items()
    }
    if not all(
        torch.allclose(left, right, rtol=1e-5, atol=1e-6)
        for left, right in comparisons.values()
    ):
        raise DataValidationError("factorial model failed permutation audit")
    return {
        "status": "passed",
        "eval_mode": True,
        "rtol": 1e-5,
        "atol": 1e-6,
        "maximum_absolute_errors": maximum,
    }


def evaluate_inference_arrays(
    arrays: Mapping[str, Mapping[str, np.ndarray]],
    *,
    prepared: PreparedSurrogateSplit,
    config: Mapping[str, Any],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    known = arrays["known_calibration"]
    unknown = arrays["surrogate_unknown"]
    known_pred = known["global_logits"].argmax(axis=1)
    unknown_pred = unknown["global_logits"].argmax(axis=1)
    metrics = evaluate_open_set(
        known_true=known["labels"],
        known_pred=known_pred,
        known_unknown_scores=known["unknown_score"],
        unknown_pred=unknown_pred,
        unknown_unknown_scores=unknown["unknown_score"],
        known_validation_scores=known["unknown_score"],
        known_class_count=len(prepared.train_class_order),
        known_acceptance_rate=float(config["evaluation"]["threshold_known_acceptance_rate"]),
    )
    if any(key not in metrics for key in METRIC_KEYS):
        raise DataValidationError("open-set evaluator omitted a frozen metric")
    rows: list[dict[str, Any]] = []
    threshold = float(metrics["threshold"])
    for role, values, predictions in (
        ("known_calibration", known, known_pred),
        ("surrogate_unknown", unknown, unknown_pred),
    ):
        for index, (pair_id, class_name) in enumerate(
            zip(prepared.pair_ids[role], prepared.class_names[role], strict=True)
        ):
            rejected = bool(values["unknown_score"][index] > threshold)
            true_label = int(values["labels"][index])
            predicted = int(predictions[index])
            is_known = role == "known_calibration"
            correct = bool(is_known and predicted == true_label)
            rows.append(
                {
                    "pair_id": pair_id,
                    "evaluation_role": role,
                    "class_name": class_name,
                    "true_label": true_label,
                    "predicted_known_label": predicted,
                    "predicted_known_class_name": prepared.train_class_order[predicted],
                    "unknown_score": float(values["unknown_score"][index]),
                    "score_source": "negative_maximum_global_raw_logit",
                    "global_logits": json.dumps(
                        values["global_logits"][index].tolist(), separators=(",", ":")
                    ),
                    "view1_logits": json.dumps(
                        values["per_view_logits"][index, 0].tolist(), separators=(",", ":")
                    ),
                    "view2_logits": json.dumps(
                        values["per_view_logits"][index, 1].tolist(), separators=(",", ":")
                    ),
                    "threshold": threshold,
                    "rejected": rejected,
                    "correctly_classified": correct,
                    "known_correctly_accepted": bool(correct and not rejected),
                }
            )
    return metrics, rows


def recompute_unit_metrics_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    known_class_count: int = 5,
    known_acceptance_rate: float = 0.95,
) -> dict[str, float]:
    known = [row for row in rows if row["evaluation_role"] == "known_calibration"]
    unknown = [row for row in rows if row["evaluation_role"] == "surrogate_unknown"]
    if not known or not unknown:
        raise DataValidationError("prediction rows are missing an evaluation role")
    return evaluate_open_set(
        known_true=np.asarray([int(row["true_label"]) for row in known]),
        known_pred=np.asarray([int(row["predicted_known_label"]) for row in known]),
        known_unknown_scores=np.asarray([float(row["unknown_score"]) for row in known]),
        unknown_pred=np.asarray([int(row["predicted_known_label"]) for row in unknown]),
        unknown_unknown_scores=np.asarray([float(row["unknown_score"]) for row in unknown]),
        known_validation_scores=np.asarray([float(row["unknown_score"]) for row in known]),
        known_class_count=known_class_count,
        known_acceptance_rate=known_acceptance_rate,
    )


def _head_arrays(model: MSMeanHeadFactorialModel) -> dict[str, np.ndarray]:
    if isinstance(model.global_head, ARPLReciprocalHead):
        return {
            "global_reciprocal_points": model.global_head.reciprocal_points.detach().cpu().numpy().astype(np.float32),
            "global_radius": model.global_head.radius.detach().cpu().numpy().astype(np.float32),
        }
    if isinstance(model.global_head, nn.Linear):
        return {
            "global_ce_weight": model.global_head.weight.detach().cpu().numpy().astype(np.float32),
            "global_ce_bias": model.global_head.bias.detach().cpu().numpy().astype(np.float32),
        }
    raise DataValidationError("unknown factorial head type")


def length_support_audit(
    bundle: Any, prepared: PreparedSurrogateSplit
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    length_metrics, rows = _length_padding_audit(bundle, prepared.pair_manifest_rows)
    metadata = {str(row["sample_id"]): row for row in bundle.rows}
    train_ids = {
        str(row[key])
        for row in prepared.pair_manifest_rows
        if row["experiment_role"] == "train_known"
        for key in ("view1_sample_id", "view2_sample_id")
    }
    surrogate_ids_by_class: dict[str, set[str]] = {}
    for row in prepared.pair_manifest_rows:
        if row["experiment_role"] == "surrogate_unknown":
            ids = surrogate_ids_by_class.setdefault(str(row["class_name"]), set())
            ids.update((str(row["view1_sample_id"]), str(row["view2_sample_id"])))
    train_lengths = np.asarray(
        [float(metadata[sample_id]["profile_length"]) for sample_id in sorted(train_ids)]
    )
    train_min = float(train_lengths.min())
    train_max = float(train_lengths.max())
    per_class = {}
    outside_total = 0
    for class_name, sample_ids in sorted(surrogate_ids_by_class.items()):
        values = np.asarray(
            [float(metadata[sample_id]["profile_length"]) for sample_id in sorted(sample_ids)]
        )
        outside = int(np.count_nonzero((values < train_min) | (values > train_max)))
        outside_total += outside
        per_class[class_name] = {
            "unique_base_samples": int(values.size),
            "minimum_original_profile_length": float(values.min()),
            "maximum_original_profile_length": float(values.max()),
            "outside_train_known_inclusive_range_count": outside,
        }
    return (
        {
            "diagnostic_only": True,
            "used_for_gate": False,
            "train_known_original_length_min": train_min,
            "train_known_original_length_max": train_max,
            "surrogate_outside_support_count": outside_total,
            "fold_length_safe": outside_total == 0,
            "per_surrogate_class": per_class,
            "length_only_auroc": length_metrics,
        },
        rows,
    )


def _configure_factorial_runtime(config: Mapping[str, Any]) -> dict[str, int]:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != config["runtime"][
        "cublas_workspace_config"
    ]:
        raise DataValidationError("CUBLAS workspace configuration changed")
    thread_runtime = _configure_torch_runtime(config)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    return thread_runtime


def _spec_for_single_unit(
    config: Mapping[str, Any],
    *,
    phase: str,
    pair_id: str,
    angle_fold: int,
    seed: int,
    method: str,
) -> tuple[Mapping[str, Any], str]:
    matches = [
        unit
        for unit in build_phase_plan(config, phase)
        if str(unit["pair_id"]) == pair_id
        and int(unit["angle_fold"]) == angle_fold
        and int(unit["seed"]) == seed
        and method in unit["methods"]
    ]
    if len(matches) != 1:
        raise DataConfigError(
            "requested pair/fold/seed/method is outside the frozen phase plan"
        )
    return matches[0]["spec"], str(matches[0]["mode"])


def _unit_destination(
    root: Path, *, pair_id: str, angle_fold: int, seed: int, method: str
) -> Path:
    return root / pair_id / f"fold_{angle_fold}" / f"seed_{seed}" / method


def _gpu_worker_assignment(
    task_index: int, *, gpu_count: int, jobs_per_gpu: int
) -> tuple[int, int]:
    # The plan has four consecutive methods per experimental unit. Rotate the
    # method-to-GPU mapping between units so no method is tied to one device,
    # while every unit contributes exactly one task to every GPU.
    unit_index, method_index = divmod(task_index, len(METHODS))
    gpu = (unit_index + method_index) % gpu_count
    local_worker = unit_index % jobs_per_gpu
    return gpu * jobs_per_gpu + local_worker, gpu


def _quarantine_path(path: Path, *, phase_root: Path, reason: str) -> Path:
    quarantine = (
        phase_root.parent
        / "_quarantine"
        / phase_root.name
        / f"{reason}_{time.time_ns()}_{path.name}"
    )
    quarantine.parent.mkdir(parents=True, exist_ok=True)
    path.replace(quarantine)
    return quarantine


def save_method_result(
    destination: Path,
    *,
    phase: str,
    pair_id: str,
    angle_fold: int,
    method: str,
    seed: int,
    prepared: PreparedSurrogateSplit,
    trained: Mapping[str, Any],
    arrays: Mapping[str, Mapping[str, np.ndarray]],
    metrics: Mapping[str, float],
    prediction_rows: Sequence[Mapping[str, Any]],
    permutation: Mapping[str, Any],
    length_audit: Mapping[str, Any],
    length_rows: Sequence[Mapping[str, Any]],
    initialization: Mapping[str, Any],
    config: Mapping[str, Any],
    wall_time_seconds: float,
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=False)
    _write_csv(destination / "predictions.csv", prediction_rows)
    _write_json(destination / "metrics.json", dict(metrics))
    _write_json(destination / "permutation_audit.json", permutation)
    _write_json(destination / "pseudo_unknown_audit.json", trained["pseudo_audit"])
    _write_json(destination / "initialization_audit.json", initialization)
    _atomic_write_bytes(
        destination / "training_log.jsonl",
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in trained["training_log"]
        ).encode("utf-8"),
    )
    np.savez_compressed(
        destination / "features_logits_scores.npz",
        **{
            f"{role}_{name}": value
            for role, role_arrays in arrays.items()
            for name, value in role_arrays.items()
        },
    )
    np.savez_compressed(
        destination / "head_parameters.npz", **_head_arrays(trained["model"])
    )
    checkpoint = {
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "pair_id": pair_id,
        "angle_fold": angle_fold,
        "method": method,
        "architecture": trained["model"].architecture_id,
        "head_type": trained["model"].head_type,
        "model_state_dict": trained["final_state"],
        "checkpoint_epoch": trained["checkpoint_epoch"],
        "formal_checkpoint": trained["formal_checkpoint"],
        "checkpoint_selection": "fixed_final_epoch",
        "train_class_order": prepared.train_class_order,
        "surrogate_class_order": prepared.surrogate_class_order,
        "normalization": asdict(prepared.normalization),
        "pair_manifest_sha256": prepared.pair_manifest_sha256,
        "initialization_seed": seed,
        "config_sha256": config["_config_sha256"],
        "task_source_hashes": trained["source_hashes"],
        "execution_runtime": trained["runtime_contract"],
        "pseudo_unknown_generated": False,
    }
    _atomic_torch_save(destination / "checkpoint.pt", checkpoint)
    resolved = dict(config)
    resolved["_resolved"] = {
        "phase": phase,
        "mode": "full" if trained["formal_checkpoint"] else "smoke",
        "pair_id": pair_id,
        "angle_fold": angle_fold,
        "unit_id": prepared.split_id,
        "method": method,
        "seed": seed,
        "dataloader_seed": seed
        + int(config["training"]["dataloader_seed_offset"]),
        "pair_manifest_sha256": prepared.pair_manifest_sha256,
        "train_class_order": list(prepared.train_class_order),
        "surrogate_class_order": list(prepared.surrogate_class_order),
        "checkpoint_epoch": trained["checkpoint_epoch"],
        "checkpoint_selection": "fixed_final_epoch",
        "training_order_sha256": trained["training_order_sha256"],
        "final_unknown_used": False,
        "even_angle_test_used": False,
    }
    _atomic_write_bytes(
        destination / "resolved_config.yaml",
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False).encode("utf-8"),
    )
    (destination / "pair_manifest.csv").write_bytes(prepared.pair_manifest_bytes)
    _write_json(destination / "pair_audit.json", prepared.pair_audit)
    _write_json(destination / "normalization.json", asdict(prepared.normalization))
    _write_json(destination / "length_padding_audit.json", length_audit)
    _write_csv(destination / "length_padding_rows.csv", length_rows)
    project_root = Path(config["_config_path"]).parents[3]
    model_device = next(trained["model"].parameters()).device
    environment = _environment(project_root, model_device)
    environment["execution_runtime"] = trained["runtime_contract"]
    environment["task_source_hashes"] = trained["source_hashes"]
    _write_json(destination / "environment.json", environment)
    contract = {
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "mode": "full" if trained["formal_checkpoint"] else "smoke",
        "pair_id": pair_id,
        "angle_fold": angle_fold,
        "unit_id": prepared.split_id,
        "seed": seed,
        "method": method,
        "config_sha256": config["_config_sha256"],
        "pair_manifest_sha256": prepared.pair_manifest_sha256,
        "train_class_order": list(prepared.train_class_order),
        "surrogate_class_order": list(prepared.surrogate_class_order),
        "task_source_hashes": trained["source_hashes"],
        "execution_runtime": trained["runtime_contract"],
        "initialization_audit": initialization,
        "training_order_sha256": trained["training_order_sha256"],
        "epoch_order_hashes": list(trained["epoch_order_hashes"]),
        "unknown_score": "negative_maximum_global_raw_logit",
        "threshold_source": "known_calibration_only",
        "surrogate_unknown_used_for_training": False,
        "surrogate_unknown_used_for_normalization": False,
        "surrogate_unknown_used_for_threshold": False,
        "final_unknown_used": False,
        "even_angle_test_used": False,
        "test_features_materialized": False,
    }
    _write_json(destination / "unit_contract.json", contract)
    summary = {
        "status": "complete",
        "phase": phase,
        "pair_id": pair_id,
        "angle_fold": angle_fold,
        "seed": seed,
        "method": method,
        "metrics": dict(metrics),
        "checkpoint_epoch": trained["checkpoint_epoch"],
        "formal_checkpoint": trained["formal_checkpoint"],
        "pair_manifest_sha256": prepared.pair_manifest_sha256,
        "training_order_sha256": trained["training_order_sha256"],
        "wall_time_seconds": wall_time_seconds,
    }
    _write_json(destination / "method_summary.json", summary)
    _write_json(
        destination / "_SUCCESS.json",
        {
            "status": "complete",
            "method_summary_sha256": file_sha256(destination / "method_summary.json"),
        },
    )
    _write_json(destination / "artifact_hashes.json", _artifact_hashes(destination))
    return summary


def _run_single_method_unlocked(
    config_path: str | Path,
    bundle_root: str | Path,
    phase_root: str | Path,
    *,
    phase: str,
    pair_id: str,
    angle_fold: int,
    seed: int,
    method: str,
    device_request: str = "auto",
    resume: bool = False,
    _interrupt_after_epoch: int | None = None,
) -> dict[str, Any]:
    config = load_ms_mean_head_factorial_config(config_path)
    spec, mode = _spec_for_single_unit(
        config,
        phase=phase,
        pair_id=pair_id,
        angle_fold=angle_fold,
        seed=seed,
        method=method,
    )
    root = Path(phase_root).resolve()
    destination = _unit_destination(
        root,
        pair_id=pair_id,
        angle_fold=angle_fold,
        seed=seed,
        method=method,
    )
    work_root = destination.parent / f".{method}.work"
    if destination.exists():
        if resume and (destination / "_SUCCESS.json").is_file():
            result = audit_method_result(
                destination,
                config=config,
                phase=phase,
                pair_id=pair_id,
                angle_fold=angle_fold,
                seed=seed,
                method=method,
                require_formal=mode == "full",
            )
            if work_root.exists():
                _quarantine_path(
                    work_root, phase_root=root, reason="redundant_completed_work"
                )
            return {**result, "status": "already_complete"}
        raise DataValidationError(f"method output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if work_root.exists() and not resume:
        raise DataValidationError(f"resume work exists; pass --resume: {work_root}")
    work_root.mkdir(exist_ok=True)
    resume_checkpoint = work_root / "latest_checkpoint.pt"
    _configure_factorial_runtime(config)
    device = _resolve_device(device_request)
    if mode == "full":
        expected_gpu = str(config["runtime"]["expected_gpu_model"])
        if device.type != "cuda" or torch.cuda.get_device_name(device) != expected_gpu:
            raise DataValidationError(
                f"formal unit requires {expected_gpu}; observed {device}"
            )
    bundle_config = config["bundle"]
    bundle = load_processed_bundle(
        bundle_root,
        expected_profiles_sha256=bundle_config["profiles_sha256"],
        expected_manifest_sha256=bundle_config["manifest_sha256"],
        expected_bundle_sha256=bundle_config["bundle_sha256"],
    )
    prepared = _prepare_split(bundle, config, spec, mode=mode)
    models, initialization = build_initialized_method_group(
        len(prepared.train_class_order), seed=seed, config=config
    )
    started = time.perf_counter()
    trained = train_one_method(
        models[method],
        method=method,
        prepared=prepared,
        seed=seed,
        config=config,
        mode=mode,
        device=device,
        resume_checkpoint=resume_checkpoint,
        _interrupt_after_epoch=_interrupt_after_epoch,
    )
    arrays = {
        role: infer_model(
            trained["model"],
            prepared.inputs[role],
            prepared.labels[role],
            device=device,
            batch_size=int(config["training"]["batch_size"]),
        )
        for role in ("train", "known_calibration", "surrogate_unknown")
    }
    metrics, prediction_rows = evaluate_inference_arrays(
        arrays, prepared=prepared, config=config
    )
    recomputed = recompute_unit_metrics_from_rows(
        prediction_rows,
        known_class_count=len(prepared.train_class_order),
        known_acceptance_rate=float(
            config["evaluation"]["threshold_known_acceptance_rate"]
        ),
    )
    for key in (*METRIC_KEYS, "threshold"):
        if not math.isclose(
            float(metrics[key]), float(recomputed[key]), rel_tol=0.0, abs_tol=1e-15
        ):
            raise DataValidationError(
                "prediction rows do not reproduce the frozen method metrics"
            )
    permutation = permutation_audit(
        trained["model"], prepared.inputs["known_calibration"], device=device
    )
    length_audit, length_rows = length_support_audit(bundle, prepared)
    project_root = Path(config["_config_path"]).parents[3]
    if task_source_hashes(project_root) != trained["source_hashes"]:
        raise DataValidationError("task source changed while the method was training")
    staging = destination.parent / f".{method}.staging"
    if staging.exists():
        if not resume:
            raise DataValidationError(f"stale staging output exists: {staging}")
        _quarantine_path(staging, phase_root=root, reason="interrupted_staging")
    summary = save_method_result(
        staging,
        phase=phase,
        pair_id=pair_id,
        angle_fold=angle_fold,
        method=method,
        seed=seed,
        prepared=prepared,
        trained=trained,
        arrays=arrays,
        metrics=metrics,
        prediction_rows=prediction_rows,
        permutation=permutation,
        length_audit=length_audit,
        length_rows=length_rows,
        initialization=initialization,
        config=config,
        wall_time_seconds=time.perf_counter() - started,
    )
    staging.replace(destination)
    if resume_checkpoint.exists():
        resume_checkpoint.unlink()
    if work_root.exists() and not any(work_root.iterdir()):
        work_root.rmdir()
    return summary


def run_single_method(
    config_path: str | Path,
    bundle_root: str | Path,
    phase_root: str | Path,
    *,
    phase: str,
    pair_id: str,
    angle_fold: int,
    seed: int,
    method: str,
    device_request: str = "auto",
    resume: bool = False,
) -> dict[str, Any]:
    root = Path(phase_root).resolve()
    lock_path = (
        root.parent
        / "_locks"
        / root.name
        / pair_id
        / f"fold_{angle_fold}"
        / f"seed_{seed}"
        / f"{method}.lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise DataValidationError(
                "the same pair/fold/seed/method is already running"
            ) from error
        try:
            return _run_single_method_unlocked(
                config_path,
                bundle_root,
                phase_root,
                phase=phase,
                pair_id=pair_id,
                angle_fold=angle_fold,
                seed=seed,
                method=method,
                device_request=device_request,
                resume=resume,
            )
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _csv_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise DataValidationError(f"invalid CSV boolean: {value}")


def _sequence_sha256(values: Sequence[Any]) -> str:
    return hashlib.sha256(
        json.dumps(list(values), ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def audit_method_result(
    destination: Path,
    *,
    config: Mapping[str, Any],
    phase: str,
    pair_id: str,
    angle_fold: int,
    seed: int,
    method: str,
    require_formal: bool,
) -> dict[str, Any]:
    required = {
        "_SUCCESS.json",
        "artifact_hashes.json",
        "checkpoint.pt",
        "environment.json",
        "features_logits_scores.npz",
        "head_parameters.npz",
        "initialization_audit.json",
        "length_padding_audit.json",
        "length_padding_rows.csv",
        "method_summary.json",
        "metrics.json",
        "normalization.json",
        "pair_audit.json",
        "pair_manifest.csv",
        "permutation_audit.json",
        "predictions.csv",
        "pseudo_unknown_audit.json",
        "resolved_config.yaml",
        "training_log.jsonl",
        "unit_contract.json",
    }
    missing = sorted(name for name in required if not (destination / name).is_file())
    if missing:
        raise DataValidationError(f"method artifact is incomplete: {missing}")
    if _read_json(destination / "artifact_hashes.json") != _artifact_hashes(
        destination
    ):
        raise DataValidationError("method artifact hashes do not match")
    success = _read_json(destination / "_SUCCESS.json")
    if (
        success.get("status") != "complete"
        or success.get("method_summary_sha256")
        != file_sha256(destination / "method_summary.json")
    ):
        raise DataValidationError("method success seal is invalid")
    contract = _read_json(destination / "unit_contract.json")
    expected_identity = (
        EXPERIMENT_ID,
        phase,
        pair_id,
        angle_fold,
        f"{pair_id}_F{angle_fold}",
        seed,
        method,
        config["_config_sha256"],
    )
    observed_identity = (
        contract.get("experiment_id"),
        contract.get("phase"),
        contract.get("pair_id"),
        int(contract.get("angle_fold", -1)),
        contract.get("unit_id"),
        int(contract.get("seed", -1)),
        contract.get("method"),
        contract.get("config_sha256"),
    )
    if observed_identity != expected_identity:
        raise DataValidationError("method unit contract identity differs")
    project_root = Path(config["_config_path"]).parents[3]
    current_source_hashes = task_source_hashes(project_root)
    if contract.get("task_source_hashes") != current_source_hashes:
        raise DataValidationError("method source hashes differ from frozen code")
    execution_runtime = _mapping(
        contract.get("execution_runtime"), "unit execution_runtime"
    )
    allowed_device_types = {"cuda"} if require_formal else {"cpu", "cuda", "mps"}
    if (
        execution_runtime.get("device_type") not in allowed_device_types
        or (
            require_formal
            and execution_runtime.get("device_name")
            != config["runtime"]["expected_gpu_model"]
        )
        or int(execution_runtime.get("torch_intraop_threads", -1))
        != int(config["runtime"]["torch_intraop_threads"])
        or int(execution_runtime.get("torch_interop_threads", -1))
        != int(config["runtime"]["torch_interop_threads"])
        or execution_runtime.get("cublas_workspace_config")
        != config["runtime"]["cublas_workspace_config"]
        or execution_runtime.get("deterministic_algorithms") is not True
        or execution_runtime.get("amp") is not False
        or execution_runtime.get("tf32") is not False
        or execution_runtime.get("torch_compile") is not False
    ):
        raise DataValidationError("method execution runtime contract failed")
    if any(
        contract.get(name) is not False
        for name in (
            "surrogate_unknown_used_for_training",
            "surrogate_unknown_used_for_normalization",
            "surrogate_unknown_used_for_threshold",
            "final_unknown_used",
            "even_angle_test_used",
            "test_features_materialized",
        )
    ):
        raise DataValidationError("forbidden evidence entered the method unit")
    initialization = _read_json(destination / "initialization_audit.json")
    if initialization != contract.get("initialization_audit") or not all(
        initialization.get("checks", {}).values()
    ):
        raise DataValidationError("factorial initialization audit failed")
    component_hashes = initialization.get("component_hashes", {})
    if (
        component_hashes.get(METHODS[0], {}).get("backbone")
        != component_hashes.get(METHODS[1], {}).get("backbone")
        or component_hashes.get(METHODS[2], {}).get("backbone")
        != component_hashes.get(METHODS[3], {}).get("backbone")
        or component_hashes.get(METHODS[0], {}).get("head")
        != component_hashes.get(METHODS[2], {}).get("head")
        or component_hashes.get(METHODS[1], {}).get("head")
        != component_hashes.get(METHODS[3], {}).get("head")
    ):
        raise DataValidationError("paired initialization hashes differ")
    pair_bytes = (destination / "pair_manifest.csv").read_bytes()
    pair_sha = hashlib.sha256(pair_bytes).hexdigest()
    if pair_sha != contract.get("pair_manifest_sha256"):
        raise DataValidationError("method pair manifest hash differs")
    pair_rows = _read_csv(destination / "pair_manifest.csv")
    pair_audit = _read_json(destination / "pair_audit.json")
    expected_per_class = int(
        config["sampling"]["pairs_per_class"]["full" if require_formal else "smoke"]
    )
    expected_counts = {
        "train_known": 5 * expected_per_class,
        "known_calibration": 5 * expected_per_class,
        "surrogate_unknown": 2 * expected_per_class,
    }
    observed_counts = {
        role: sum(row["experiment_role"] == role for row in pair_rows)
        for role in expected_counts
    }
    if (
        pair_audit.get("status") != "passed"
        or pair_audit.get("train_evaluation_base_overlap") != 0
        or pair_audit.get("final_unknown_pairs") != 0
        or pair_audit.get("even_angle_pairs") != 0
        or pair_audit.get("test_pairs_generated") is not False
        or pair_audit.get("test_features_materialized") is not False
        or pair_audit.get("surrogate_train_pairs_materialized") is not False
        or observed_counts != expected_counts
        or any(
            int(row[field]) % 2 == 0
            for row in pair_rows
            for field in ("view1_angle_deg", "view2_angle_deg")
        )
        or any(row["view1_frame_id"] == row["view2_frame_id"] for row in pair_rows)
    ):
        raise DataValidationError("method pair isolation audit failed")
    train_base = {
        row[field]
        for row in pair_rows
        if row["experiment_role"] == "train_known"
        for field in ("view1_sample_id", "view2_sample_id")
    }
    eval_base = {
        row[field]
        for row in pair_rows
        if row["experiment_role"] != "train_known"
        for field in ("view1_sample_id", "view2_sample_id")
    }
    if train_base & eval_base:
        raise DataValidationError("base HRRP leaked between training and evaluation")
    checkpoint = torch.load(
        destination / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    expected_epoch = int(
        config["training"]["total_epochs"]
        if require_formal
        else config["training"]["smoke_epochs"]
    )
    if (
        checkpoint.get("experiment_id") != EXPERIMENT_ID
        or checkpoint.get("phase") != phase
        or checkpoint.get("pair_id") != pair_id
        or int(checkpoint.get("angle_fold", -1)) != angle_fold
        or checkpoint.get("method") != method
        or int(checkpoint.get("initialization_seed", -1)) != seed
        or checkpoint.get("pair_manifest_sha256") != pair_sha
        or checkpoint.get("config_sha256") != config["_config_sha256"]
        or checkpoint.get("task_source_hashes") != current_source_hashes
        or int(checkpoint.get("checkpoint_epoch", -1)) != expected_epoch
        or bool(checkpoint.get("formal_checkpoint")) != require_formal
        or checkpoint.get("checkpoint_selection") != "fixed_final_epoch"
        or checkpoint.get("execution_runtime") != execution_runtime
        or checkpoint.get("pseudo_unknown_generated") is not False
    ):
        raise DataValidationError("method checkpoint contract failed")
    log_lines = [
        json.loads(line)
        for line in (destination / "training_log.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    if (
        len(log_lines) != expected_epoch
        or [int(row["epoch"]) for row in log_lines]
        != list(range(1, expected_epoch + 1))
        or any(row.get("method") != method for row in log_lines)
        or any(row.get("pseudo_unknown_generated") is not False for row in log_lines)
        or any(
            row.get("checkpoint_selected_for_open_set_performance") is not False
            for row in log_lines
        )
        or _training_order_sha256(
            [str(row["train_order_epoch_sha256"]) for row in log_lines]
        )
        != contract.get("training_order_sha256")
    ):
        raise DataValidationError("fixed-epoch training log contract failed")
    pseudo = _read_json(destination / "pseudo_unknown_audit.json")
    if (
        pseudo.get("status") != "not_applicable"
        or pseudo.get("pseudo_unknown_generated") is not False
        or pseudo.get("pseudo_count") != 0
        or pseudo.get("surrogate_unknown_used") is not False
        or pseudo.get("final_unknown_used") is not False
        or pseudo.get("even_angle_test_used") is not False
    ):
        raise DataValidationError("pseudo-unknown path was not absent")
    permutation = _read_json(destination / "permutation_audit.json")
    if permutation.get("status") != "passed" or permutation.get("eval_mode") is not True:
        raise DataValidationError("method permutation audit failed")
    length_audit = _read_json(destination / "length_padding_audit.json")
    if (
        length_audit.get("diagnostic_only") is not True
        or length_audit.get("used_for_gate") is not False
        or not isinstance(length_audit.get("fold_length_safe"), bool)
    ):
        raise DataValidationError("length/padding audit contract failed")
    metrics = _read_json(destination / "metrics.json")
    extract_report_metrics(metrics)
    prediction_rows = _read_csv(destination / "predictions.csv")
    expected_prediction_count = expected_counts["known_calibration"] + expected_counts[
        "surrogate_unknown"
    ]
    if len(prediction_rows) != expected_prediction_count:
        raise DataValidationError("prediction row count differs from frozen protocol")
    recomputed = recompute_unit_metrics_from_rows(
        prediction_rows,
        known_class_count=5,
        known_acceptance_rate=float(
            config["evaluation"]["threshold_known_acceptance_rate"]
        ),
    )
    for key in (*METRIC_KEYS, "threshold", "known_acceptance_rate"):
        if not math.isclose(
            float(metrics[key]), float(recomputed[key]), rel_tol=0.0, abs_tol=1e-15
        ):
            raise DataValidationError("saved predictions do not reproduce all metrics")
    threshold = float(metrics["threshold"])
    for row in prediction_rows:
        rejected = float(row["unknown_score"]) > threshold
        is_known = row["evaluation_role"] == "known_calibration"
        correct = is_known and int(row["true_label"]) == int(
            row["predicted_known_label"]
        )
        if (
            not math.isclose(
                float(row["threshold"]), threshold, rel_tol=0.0, abs_tol=0.0
            )
            or
            _csv_bool(row["rejected"]) != rejected
            or _csv_bool(row["correctly_classified"]) != correct
            or _csv_bool(row["known_correctly_accepted"])
            != (correct and not rejected)
            or row["score_source"] != "negative_maximum_global_raw_logit"
        ):
            raise DataValidationError("per-sample operating-point fields are inconsistent")
    with np.load(destination / "features_logits_scores.npz") as arrays:
        keys = set(arrays.files)
        if any("reject" in key or "pseudo" in key for key in keys):
            raise DataValidationError("forbidden rejector/pseudo array was saved")
        for role in ("known_calibration", "surrogate_unknown"):
            role_rows = [
                row for row in prediction_rows if row["evaluation_role"] == role
            ]
            global_logits = arrays[f"{role}_global_logits"]
            per_view_logits = arrays[f"{role}_per_view_logits"]
            scores = arrays[f"{role}_unknown_score"]
            labels = arrays[f"{role}_labels"]
            csv_global = np.asarray(
                [json.loads(row["global_logits"]) for row in role_rows],
                dtype=np.float64,
            )
            csv_per_view = np.asarray(
                [
                    [json.loads(row["view1_logits"]), json.loads(row["view2_logits"])]
                    for row in role_rows
                ],
                dtype=np.float64,
            )
            csv_scores = np.asarray(
                [float(row["unknown_score"]) for row in role_rows], dtype=np.float64
            )
            csv_labels = np.asarray(
                [int(row["true_label"]) for row in role_rows], dtype=np.int64
            )
            csv_predictions = np.asarray(
                [int(row["predicted_known_label"]) for row in role_rows],
                dtype=np.int64,
            )
            if (
                global_logits.shape != csv_global.shape
                or per_view_logits.shape != csv_per_view.shape
                or scores.shape != csv_scores.shape
                or not np.array_equal(labels, csv_labels)
                or not np.array_equal(global_logits.argmax(axis=1), csv_predictions)
                or not np.array_equal(global_logits, csv_global)
                or not np.array_equal(per_view_logits, csv_per_view)
                or not np.array_equal(scores, csv_scores)
            ):
                raise DataValidationError("saved NPZ arrays differ from predictions.csv")
    with np.load(destination / "head_parameters.npz") as head:
        expected_head_keys = (
            {"global_ce_weight", "global_ce_bias"}
            if method in CE_METHODS
            else {"global_reciprocal_points", "global_radius"}
        )
        if set(head.files) != expected_head_keys:
            raise DataValidationError("saved head parameters differ from method definition")
    summary = _read_json(destination / "method_summary.json")
    if (
        summary.get("status") != "complete"
        or summary.get("phase") != phase
        or summary.get("pair_id") != pair_id
        or int(summary.get("angle_fold", -1)) != angle_fold
        or int(summary.get("seed", -1)) != seed
        or summary.get("method") != method
        or summary.get("pair_manifest_sha256") != pair_sha
        or summary.get("training_order_sha256")
        != contract.get("training_order_sha256")
    ):
        raise DataValidationError("method summary contract failed")
    return {
        "status": "passed",
        "phase": phase,
        "pair_id": pair_id,
        "angle_fold": angle_fold,
        "seed": seed,
        "method": method,
        "metrics": extract_report_metrics(metrics),
        "threshold": float(metrics["threshold"]),
        "pair_manifest_sha256": pair_sha,
        "prediction_order_sha256": _sequence_sha256(
            [row["pair_id"] for row in prediction_rows]
        ),
        "prediction_label_order_sha256": _sequence_sha256(
            [int(row["true_label"]) for row in prediction_rows]
        ),
        "initialization_audit": initialization,
        "training_order_sha256": contract["training_order_sha256"],
        "task_source_hashes": current_source_hashes,
        "execution_runtime": dict(execution_runtime),
        "train_class_order": list(contract["train_class_order"]),
        "surrogate_class_order": list(contract["surrogate_class_order"]),
        "fold_length_safe": bool(length_audit["fold_length_safe"]),
        "length_audit": length_audit,
        "npz_predictions_crosschecked": True,
        "all_nine_metrics_recomputed": True,
        "pseudo_unknown_absent": True,
        "final_unknown_used": False,
        "even_angle_test_used": False,
    }


def _expected_unit_identities(
    config: Mapping[str, Any], phase: str
) -> list[tuple[str, int, int, str]]:
    return [
        (
            str(unit["pair_id"]),
            int(unit["angle_fold"]),
            int(unit["seed"]),
            method,
        )
        for unit in build_phase_plan(config, phase)
        for method in unit["methods"]
    ]


def _assert_exact_method_directory_matrix(
    root: Path, expected: Sequence[tuple[str, int, int, str]]
) -> None:
    expected_paths = {
        _unit_destination(
            root,
            pair_id=pair_id,
            angle_fold=fold,
            seed=seed,
            method=method,
        ).relative_to(root)
        for pair_id, fold, seed, method in expected
    }
    observed_paths = {
        path.parent.relative_to(root) for path in root.rglob("_SUCCESS.json")
    }
    if observed_paths != expected_paths:
        missing = sorted(str(path) for path in expected_paths - observed_paths)
        extra = sorted(str(path) for path in observed_paths - expected_paths)
        raise DataValidationError(
            "phase method directory matrix differs from the frozen plan; "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    incomplete = [
        path
        for path in root.rglob(".*")
        if path.is_dir() and (path.name.endswith(".work") or path.name.endswith(".staging"))
    ]
    if incomplete:
        raise DataValidationError(f"phase contains incomplete work: {incomplete[0]}")


def _audit_confirmation_launcher(
    root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    manifest_path = root / "launcher" / "launch_manifest.json"
    results_path = root / "launcher" / "launch_results.json"
    if not manifest_path.is_file() or not results_path.is_file():
        raise DataValidationError("confirmation lacks the frozen GPU launcher records")
    manifest = _read_json(manifest_path)
    results = _read_json(results_path)
    expected_tasks = _plan_payload(config, "confirmation")["tasks"]
    assignments = manifest.get("assignments")
    result_rows = results.get("results")
    if not isinstance(assignments, list) or not isinstance(result_rows, list):
        raise DataValidationError("GPU launcher records have invalid row types")
    worker_count = int(config["runtime"]["total_parallel_jobs"])
    gpu_count = int(config["runtime"]["expected_gpu_count"])
    jobs_per_gpu = int(config["runtime"]["jobs_per_gpu"])
    tokens = manifest.get("child_visible_gpu_tokens")
    task_keys = ("phase", "pair_id", "angle_fold", "seed", "method")
    expected_identities = [tuple(task[key] for key in task_keys) for task in expected_tasks]
    assignment_identities = [
        tuple(row.get(key) for key in task_keys) for row in assignments
    ]
    result_identities = [tuple(row.get(key) for key in task_keys) for row in result_rows]
    assignment_indices = [int(row.get("task_index", -1)) for row in assignments]
    worker_slots = [int(row.get("worker_slot", -1)) for row in assignments]
    gpu_indices = [int(row.get("physical_gpu_index", -1)) for row in assignments]
    if (
        manifest.get("experiment_id") != EXPERIMENT_ID
        or manifest.get("phase") != "confirmation"
        or manifest.get("config_sha256") != config["_config_sha256"]
        or manifest.get("gpu_names")
        != [config["runtime"]["expected_gpu_model"]] * gpu_count
        or not isinstance(tokens, list)
        or len(tokens) != gpu_count
        or len(set(tokens)) != gpu_count
        or int(manifest.get("jobs_per_gpu", -1)) != jobs_per_gpu
        or int(manifest.get("worker_count", -1)) != worker_count
        or manifest.get("gpu_assignment") != "rotating_method_balanced_v1"
        or int(manifest.get("training_task_count", -1)) != 168
        or manifest.get("final_unknown_test_authorized") is not False
        or len(assignments) != 168
        or assignment_indices != list(range(168))
        or assignment_identities != expected_identities
        or len(set(assignment_identities)) != 168
        or any(
            (slot, gpu)
            != _gpu_worker_assignment(
                index, gpu_count=gpu_count, jobs_per_gpu=jobs_per_gpu
            )
            for index, (slot, gpu) in enumerate(
                zip(worker_slots, gpu_indices, strict=True)
            )
        )
        or any(
            row.get("visible_gpu_token") != tokens[int(row["physical_gpu_index"])]
            for row in assignments
        )
    ):
        raise DataValidationError("GPU launch manifest differs from the frozen 4x4 plan")
    workers_per_gpu = {
        str(gpu): len(
            {
                int(row["worker_slot"])
                for row in assignments
                if int(row["physical_gpu_index"]) == gpu
            }
        )
        for gpu in range(gpu_count)
    }
    tasks_per_gpu = {
        str(gpu): sum(
            int(row["physical_gpu_index"]) == gpu for row in assignments
        )
        for gpu in range(gpu_count)
    }
    unit_gpu_sets: dict[tuple[str, int, int], set[int]] = {}
    method_gpu_counts = {
        method: {str(gpu): 0 for gpu in range(gpu_count)} for method in METHODS
    }
    for row in assignments:
        unit = (str(row["pair_id"]), int(row["angle_fold"]), int(row["seed"]))
        gpu = int(row["physical_gpu_index"])
        unit_gpu_sets.setdefault(unit, set()).add(gpu)
        method_gpu_counts[str(row["method"])][str(gpu)] += 1
    every_unit_covers_four_gpus = all(
        values == set(range(gpu_count)) for values in unit_gpu_sets.values()
    ) and len(unit_gpu_sets) == 42
    every_method_balanced_across_gpus = all(
        max(counts.values()) - min(counts.values()) <= 1
        for counts in method_gpu_counts.values()
    )
    result_lookup = {
        tuple(row.get(key) for key in task_keys): row for row in result_rows
    }
    if (
        workers_per_gpu != {str(gpu): jobs_per_gpu for gpu in range(gpu_count)}
        or tasks_per_gpu != {str(gpu): 42 for gpu in range(gpu_count)}
        or not every_unit_covers_four_gpus
        or not every_method_balanced_across_gpus
        or len(result_rows) != 168
        or len(result_lookup) != 168
        or set(result_identities) != set(expected_identities)
        or int(results.get("training_task_count", -1)) != 168
        or int(results.get("successful_task_count", -1)) != 168
        or int(results.get("failed_task_count", -1)) != 0
    ):
        raise DataValidationError("GPU launch completion matrix is incomplete")
    for assignment in assignments:
        identity = tuple(assignment[key] for key in task_keys)
        result = result_lookup[identity]
        log_path = root / str(result.get("log", ""))
        if (
            int(result.get("exit_code", -1)) != 0
            or int(result.get("worker_slot", -1))
            != int(assignment["worker_slot"])
            or int(result.get("physical_gpu_index", -1))
            != int(assignment["physical_gpu_index"])
            or result.get("visible_gpu_token") != assignment["visible_gpu_token"]
            or not log_path.is_file()
            or log_path.stat().st_size == 0
        ):
            raise DataValidationError("GPU launch result differs from its assignment")
    return {
        "status": "passed",
        "gpu_count": gpu_count,
        "jobs_per_gpu": jobs_per_gpu,
        "worker_count": worker_count,
        "tasks_per_gpu": tasks_per_gpu,
        "method_tasks_per_gpu": method_gpu_counts,
        "every_unit_covers_four_gpus": every_unit_covers_four_gpus,
        "every_method_balanced_across_gpus": every_method_balanced_across_gpus,
        "training_task_count": 168,
        "successful_task_count": 168,
        "failed_task_count": 0,
        "all_logs_present": True,
        "all_assignments_exact": True,
    }


def _collect_phase_audit(
    config: Mapping[str, Any], root: Path, *, phase: str
) -> dict[str, Any]:
    expected = _expected_unit_identities(config, phase)
    _assert_exact_method_directory_matrix(root, expected)
    require_formal = phase == "confirmation"
    launcher_audit = (
        _audit_confirmation_launcher(root, config) if require_formal else None
    )
    audited = []
    for pair_id, fold, seed, method in expected:
        destination = _unit_destination(
            root,
            pair_id=pair_id,
            angle_fold=fold,
            seed=seed,
            method=method,
        )
        audited.append(
            audit_method_result(
                destination,
                config=config,
                phase=phase,
                pair_id=pair_id,
                angle_fold=fold,
                seed=seed,
                method=method,
                require_formal=require_formal,
            )
        )
    fairness_rows = []
    for unit in build_phase_plan(config, phase):
        members = [
            row
            for row in audited
            if row["pair_id"] == unit["pair_id"]
            and row["angle_fold"] == unit["angle_fold"]
            and row["seed"] == unit["seed"]
        ]
        if len(members) != 4 or {row["method"] for row in members} != set(METHODS):
            raise DataValidationError("factorial unit lacks exactly four methods")
        exact_fields = (
            "pair_manifest_sha256",
            "prediction_order_sha256",
            "prediction_label_order_sha256",
            "initialization_audit",
            "training_order_sha256",
            "task_source_hashes",
            "train_class_order",
            "surrogate_class_order",
            "fold_length_safe",
        )
        equality = {
            field: all(row[field] == members[0][field] for row in members[1:])
            for field in exact_fields
        }
        runtime_values = [dict(row["execution_runtime"]) for row in members]
        runtime_equal = all(value == runtime_values[0] for value in runtime_values[1:])
        if not all(equality.values()) or not runtime_equal:
            raise DataValidationError(
                "four methods do not share the frozen data/init/order/runtime contract"
            )
        fairness_rows.append(
            {
                "pair_id": unit["pair_id"],
                "angle_fold": unit["angle_fold"],
                "seed": unit["seed"],
                "methods": list(METHODS),
                "checks": {**equality, "execution_runtime": runtime_equal},
                "pair_manifest_sha256": members[0]["pair_manifest_sha256"],
                "training_order_sha256": members[0]["training_order_sha256"],
                "initialization_audit": members[0]["initialization_audit"],
                "fold_length_safe": members[0]["fold_length_safe"],
            }
        )
    source_hashes = [row["task_source_hashes"] for row in audited]
    if not all(value == source_hashes[0] for value in source_hashes[1:]):
        raise DataValidationError("formal tasks used different source revisions")
    formal_runtime_contracts = {
        json.dumps(row["execution_runtime"], sort_keys=True) for row in audited
    }
    if require_formal and len(formal_runtime_contracts) != 1:
        raise DataValidationError("formal tasks used different execution runtimes")
    metric_rows = [
        {
            "pair_id": row["pair_id"],
            "angle_fold": row["angle_fold"],
            "seed": row["seed"],
            "method": row["method"],
            **row["metrics"],
        }
        for row in audited
    ]
    return {
        "status": "passed",
        "phase": phase,
        "experimental_unit_count": len(build_phase_plan(config, phase)),
        "training_task_count": len(audited),
        "expected_training_task_count": 168 if require_formal else 4,
        "all_method_artifacts_passed": True,
        "all_pair_manifests_shared_within_units": True,
        "all_prediction_orders_shared_within_units": True,
        "all_dataloader_orders_shared_within_units": True,
        "all_paired_initializations_passed": True,
        "all_nine_metrics_recomputed": True,
        "all_pseudo_unknown_paths_absent": True,
        "all_final_unknown_excluded": True,
        "all_even_angle_test_excluded": True,
        "all_task_source_hashes_equal": True,
        "all_formal_runtime_contracts_equal": len(formal_runtime_contracts) == 1,
        "task_source_hashes": source_hashes[0],
        "formal_runtime_contract": (
            audited[0]["execution_runtime"] if require_formal else None
        ),
        "confirmation_launcher_audit": launcher_audit,
        "metric_rows": metric_rows,
        "fairness_rows": fairness_rows,
        "audited_rows": audited,
    }


def _pair_delta_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "comparison": comparison,
            "name": value["name"],
            "formula": value["formula"],
            **row,
        }
        for comparison, value in summary["comparisons"].items()
        for row in value["pair_deltas"]
    ]


def _unit_delta_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "comparison": comparison,
            "name": value["name"],
            "formula": value["formula"],
            **row,
        }
        for comparison, value in summary["comparisons"].items()
        for row in value["unit_deltas"]
    ]


def _absorption_rows(
    root: Path, expected: Sequence[tuple[str, int, int, str]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    counts: dict[tuple[str, str, str, str], int] = {}
    totals: dict[tuple[str, str, str], int] = {}
    rejected: dict[tuple[str, str, str], int] = {}
    for pair_id, fold, seed, method in expected:
        rows = _read_csv(
            _unit_destination(
                root,
                pair_id=pair_id,
                angle_fold=fold,
                seed=seed,
                method=method,
            )
            / "predictions.csv"
        )
        for row in rows:
            if row["evaluation_role"] != "surrogate_unknown":
                continue
            base = (pair_id, method, row["class_name"])
            totals[base] = totals.get(base, 0) + 1
            if _csv_bool(row["rejected"]):
                rejected[base] = rejected.get(base, 0) + 1
            else:
                key = (*base, row["predicted_known_class_name"])
                counts[key] = counts.get(key, 0) + 1
    summary_rows = [
        {
            "pair_id": pair_id,
            "method": method,
            "surrogate_identity": surrogate,
            "total_surrogate_count": total,
            "rejected_count": rejected.get((pair_id, method, surrogate), 0),
            "false_accept_count": total
            - rejected.get((pair_id, method, surrogate), 0),
            "unknown_rejection_rate": rejected.get(
                (pair_id, method, surrogate), 0
            )
            / total,
            "false_accept_rate": 1.0
            - rejected.get((pair_id, method, surrogate), 0) / total,
        }
        for (pair_id, method, surrogate), total in sorted(totals.items())
    ]
    false_accept_totals = {
        (row["pair_id"], row["method"], row["surrogate_identity"]): row[
            "false_accept_count"
        ]
        for row in summary_rows
    }
    detail_rows = [
        {
            "pair_id": pair_id,
            "method": method,
            "surrogate_identity": surrogate,
            "absorbed_as_known_identity": predicted,
            "false_accept_count": count,
            "rate_over_all_surrogate": count / totals[(pair_id, method, surrogate)],
            "composition_within_false_accepts": count
            / false_accept_totals[(pair_id, method, surrogate)],
            "total_surrogate_count": totals[(pair_id, method, surrogate)],
            "total_false_accept_count": false_accept_totals[
                (pair_id, method, surrogate)
            ],
        }
        for (pair_id, method, surrogate, predicted), count in sorted(counts.items())
    ]
    return detail_rows, summary_rows


def _absorption_overall_rows(
    pair_context_rows: Sequence[Mapping[str, Any]],
    pair_context_summary: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    counts: dict[tuple[str, str, str], int] = {}
    for row in pair_context_rows:
        base = (str(row["method"]), str(row["surrogate_identity"]))
        key = (*base, str(row["absorbed_as_known_identity"]))
        count = int(row["false_accept_count"])
        counts[key] = counts.get(key, 0) + count
    totals: dict[tuple[str, str], int] = {}
    rejected: dict[tuple[str, str], int] = {}
    for row in pair_context_summary:
        base = (str(row["method"]), str(row["surrogate_identity"]))
        totals[base] = totals.get(base, 0) + int(row["total_surrogate_count"])
        rejected[base] = rejected.get(base, 0) + int(row["rejected_count"])
    summary_rows = [
        {
            "method": method,
            "surrogate_identity": surrogate,
            "identity_pair_context_count": 2,
            "total_surrogate_count": total,
            "rejected_count": rejected[(method, surrogate)],
            "false_accept_count": total - rejected[(method, surrogate)],
            "unknown_rejection_rate": rejected[(method, surrogate)] / total,
            "false_accept_rate": 1.0 - rejected[(method, surrogate)] / total,
        }
        for (method, surrogate), total in sorted(totals.items())
    ]
    false_accept_totals = {
        (row["method"], row["surrogate_identity"]): row["false_accept_count"]
        for row in summary_rows
    }
    detail_rows = [
        {
            "method": method,
            "surrogate_identity": surrogate,
            "absorbed_as_known_identity": predicted,
            "false_accept_count": count,
            "rate_over_all_surrogate": count / totals[(method, surrogate)],
            "composition_within_false_accepts": count
            / false_accept_totals[(method, surrogate)],
            "total_surrogate_count": totals[(method, surrogate)],
            "total_false_accept_count": false_accept_totals[(method, surrogate)],
            "identity_pair_context_count": 2,
        }
        for (method, surrogate, predicted), count in sorted(counts.items())
    ]
    return detail_rows, summary_rows


def _surrogate_identity_metric_rows(
    root: Path, expected: Sequence[tuple[str, int, int, str]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    unit_rows: list[dict[str, Any]] = []
    for pair_id, fold, seed, method in expected:
        rows = _read_csv(
            _unit_destination(
                root,
                pair_id=pair_id,
                angle_fold=fold,
                seed=seed,
                method=method,
            )
            / "predictions.csv"
        )
        known_scores = np.asarray(
            [
                float(row["unknown_score"])
                for row in rows
                if row["evaluation_role"] == "known_calibration"
            ],
            dtype=np.float64,
        )
        surrogate_names = sorted(
            {
                row["class_name"]
                for row in rows
                if row["evaluation_role"] == "surrogate_unknown"
            }
        )
        if len(surrogate_names) != 2:
            raise DataValidationError("unit does not contain two surrogate identities")
        for class_name in surrogate_names:
            class_rows = [
                row
                for row in rows
                if row["evaluation_role"] == "surrogate_unknown"
                and row["class_name"] == class_name
            ]
            unknown_scores = np.asarray(
                [float(row["unknown_score"]) for row in class_rows], dtype=np.float64
            )
            unit_rows.append(
                {
                    "pair_id": pair_id,
                    "angle_fold": fold,
                    "seed": seed,
                    "method": method,
                    "surrogate_identity": class_name,
                    "known_calibration_count": int(known_scores.size),
                    "surrogate_sample_count": int(unknown_scores.size),
                    "auroc": binary_auroc(known_scores, unknown_scores),
                    "unknown_rejection_rate": float(
                        np.mean([_csv_bool(row["rejected"]) for row in class_rows])
                    ),
                    "threshold": float(class_rows[0]["threshold"]),
                }
            )
    aggregate_rows = []
    groups = sorted(
        {
            (row["pair_id"], row["method"], row["surrogate_identity"])
            for row in unit_rows
        }
    )
    for pair_id, method, surrogate_identity in groups:
        values = [
            row
            for row in unit_rows
            if (
                row["pair_id"],
                row["method"],
                row["surrogate_identity"],
            )
            == (pair_id, method, surrogate_identity)
        ]
        if len(values) != len(ANGLE_FOLDS) * len(CONFIRMATION_SEEDS):
            raise DataValidationError(
                "surrogate identity aggregate lacks two folds x three seeds"
            )
        aggregate_rows.append(
            {
                "pair_id": pair_id,
                "method": method,
                "surrogate_identity": surrogate_identity,
                "unit_count": len(values),
                "mean_auroc": float(np.mean([row["auroc"] for row in values])),
                "mean_unknown_rejection_rate": float(
                    np.mean([row["unknown_rejection_rate"] for row in values])
                ),
                "minimum_auroc": float(np.min([row["auroc"] for row in values])),
                "maximum_auroc": float(np.max([row["auroc"] for row in values])),
            }
        )
    overall_rows = []
    overall_groups = sorted(
        {(row["method"], row["surrogate_identity"]) for row in unit_rows}
    )
    for method, surrogate_identity in overall_groups:
        values = [
            row
            for row in unit_rows
            if (row["method"], row["surrogate_identity"])
            == (method, surrogate_identity)
        ]
        if len(values) != 2 * len(ANGLE_FOLDS) * len(CONFIRMATION_SEEDS):
            raise DataValidationError(
                "surrogate identity overall aggregate lacks two pair contexts x two folds x three seeds"
            )
        overall_rows.append(
            {
                "method": method,
                "surrogate_identity": surrogate_identity,
                "identity_pair_context_count": 2,
                "unit_count": len(values),
                "mean_auroc": float(np.mean([row["auroc"] for row in values])),
                "mean_unknown_rejection_rate": float(
                    np.mean([row["unknown_rejection_rate"] for row in values])
                ),
                "minimum_auroc": float(np.min([row["auroc"] for row in values])),
                "maximum_auroc": float(np.max([row["auroc"] for row in values])),
            }
        )
    return unit_rows, aggregate_rows, overall_rows


def _length_pair_summary(audited_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_pair = []
    for pair_id, _, _ in IDENTITY_PAIRS:
        fold_rows = []
        for fold in ANGLE_FOLDS:
            values = [
                row
                for row in audited_rows
                if row["pair_id"] == pair_id and row["angle_fold"] == fold
            ]
            safe_values = {bool(row["fold_length_safe"]) for row in values}
            outside_values = {
                int(row["length_audit"]["surrogate_outside_support_count"])
                for row in values
            }
            metric_values = {
                json.dumps(row["length_audit"]["length_only_auroc"], sort_keys=True)
                for row in values
            }
            if len(values) != len(CONFIRMATION_SEEDS) * len(METHODS) or any(
                len(group) != 1
                for group in (safe_values, outside_values, metric_values)
            ):
                raise DataValidationError(
                    "length/padding diagnostic differs across methods or seeds"
                )
            fold_rows.append(
                {
                    "angle_fold": fold,
                    "fold_length_safe": safe_values.pop(),
                    "surrogate_outside_support_count": outside_values.pop(),
                    "length_only_auroc": json.loads(metric_values.pop()),
                }
            )
        pair_safe = all(row["fold_length_safe"] for row in fold_rows)
        by_pair.append(
            {
                "pair_id": pair_id,
                "classification": "length_safe" if pair_safe else "length_risk",
                "both_folds_safe": pair_safe,
                "folds": fold_rows,
            }
        )
    return {
        "diagnostic_only": True,
        "used_for_gate": False,
        "pair_definition": config_safe_definition(),
        "by_pair": by_pair,
        "length_safe_pairs": [
            row["pair_id"] for row in by_pair if row["both_folds_safe"]
        ],
        "length_risk_pairs": [
            row["pair_id"] for row in by_pair if not row["both_folds_safe"]
        ],
    }


def config_safe_definition() -> str:
    return "both_folds_all_surrogate_original_lengths_within_inclusive_train_known_range"


def _descriptive_subset_summary(
    metric_rows: Sequence[Mapping[str, Any]], pair_ids: Sequence[str]
) -> dict[str, Any]:
    selected_pairs = tuple(pair_ids)
    if not selected_pairs:
        return {"status": "no_pairs", "pair_ids": [], "used_for_gate": False}
    rows = [row for row in metric_rows if row["pair_id"] in selected_pairs]
    method_rows = []
    for method in METHODS:
        method_values = [row for row in rows if row["method"] == method]
        method_rows.append(
            {
                "method": method,
                "unit_count": len(method_values),
                **{
                    metric: float(
                        np.mean([float(row[metric]) for row in method_values])
                    )
                    for metric in METRIC_KEYS
                },
            }
        )
    lookup = {
        (row["pair_id"], row["angle_fold"], row["seed"], row["method"]): row
        for row in rows
    }
    comparisons: dict[str, Any] = {}
    for comparison, specification in FACTORIAL_COMPARISONS.items():
        pair_rows = []
        for pair_id in selected_pairs:
            unit_deltas = []
            for fold in ANGLE_FOLDS:
                for seed in CONFIRMATION_SEEDS:
                    unit = {
                        method: lookup[(pair_id, fold, seed, method)]
                        for method in METHODS
                    }
                    unit_deltas.append(
                        {
                            metric: sum(
                                float(coefficient) * float(unit[method][metric])
                                for method, coefficient in specification[
                                    "coefficients"
                                ].items()
                            )
                            for metric in METRIC_KEYS
                        }
                    )
            pair_rows.append(
                {
                    "pair_id": pair_id,
                    **{
                        f"delta_{metric}": float(
                            np.mean([row[metric] for row in unit_deltas])
                        )
                        for metric in METRIC_KEYS
                    },
                }
            )
        comparisons[comparison] = {
            "formula": specification["formula"],
            "pair_deltas": pair_rows,
            "mean_deltas": {
                metric: float(
                    np.mean([row[f"delta_{metric}"] for row in pair_rows])
                )
                for metric in METRIC_KEYS
            },
            "positive_auroc_pair_count": sum(
                row["delta_auroc"] > 0.0 for row in pair_rows
            ),
        }
    return {
        "status": "reported",
        "pair_ids": list(selected_pairs),
        "pair_count": len(selected_pairs),
        "used_for_gate": False,
        "method_aggregates": method_rows,
        "comparisons": comparisons,
    }


def _difficulty_ranking(metric_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for pair_id, _, _ in IDENTITY_PAIRS:
        selected = [row for row in metric_rows if row["pair_id"] == pair_id]
        rows.append(
            {
                "pair_id": pair_id,
                "mean_auroc_across_four_methods": float(
                    np.mean([float(row["auroc"]) for row in selected])
                ),
                "mean_urr_across_four_methods": float(
                    np.mean([float(row["unknown_rejection_rate"]) for row in selected])
                ),
                "mean_known_accuracy_across_four_methods": float(
                    np.mean([float(row["known_accuracy"]) for row in selected])
                ),
            }
        )
    return sorted(rows, key=lambda row: (row["mean_auroc_across_four_methods"], row["pair_id"]))


def _phase_artifact_hashes(root: Path) -> dict[str, str]:
    excluded = {"artifact_hashes.json", "_PHASE_SUCCESS.json"}
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in excluded
    }


def _smoke_summary(metric_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "status": "diagnostic_only",
        "performance_used_for_decision": False,
        "training_task_count": len(metric_rows),
        "methods": [
            {
                "method": row["method"],
                **{metric: float(row[metric]) for metric in METRIC_KEYS},
            }
            for row in metric_rows
        ],
        "candidate_decision": None,
        "final_unknown_test_authorized": False,
    }


def _build_confirmation_derived(
    root: Path,
    metric_rows: Sequence[Mapping[str, Any]],
    audited_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    summary = summarize_factorial_results(
        metric_rows,
        bootstrap_resamples=int(config["uncertainty"]["bootstrap_resamples"]),
        bootstrap_seed=int(config["uncertainty"]["bootstrap_seed"]),
        confidence_level=float(config["uncertainty"]["confidence_level"]),
    )
    length = _length_pair_summary(audited_rows)
    subset = {
        "all_pairs": _descriptive_subset_summary(
            metric_rows, [pair_id for pair_id, _, _ in IDENTITY_PAIRS]
        ),
        "length_safe": _descriptive_subset_summary(
            metric_rows, length["length_safe_pairs"]
        ),
        "length_risk": _descriptive_subset_summary(
            metric_rows, length["length_risk_pairs"]
        ),
        "diagnostic_only": True,
        "used_for_gate": False,
    }
    expected = _expected_unit_identities(config, "confirmation")
    surrogate_unit_rows, surrogate_aggregate_rows, surrogate_overall_rows = (
        _surrogate_identity_metric_rows(root, expected)
    )
    absorption_rows, absorption_summary_rows = _absorption_rows(root, expected)
    absorption_overall_rows, absorption_overall_summary_rows = (
        _absorption_overall_rows(absorption_rows, absorption_summary_rows)
    )
    return {
        "factorial_summary": summary,
        "pair_delta_rows": _pair_delta_rows(summary),
        "unit_delta_rows": _unit_delta_rows(summary),
        "absorption_rows": absorption_rows,
        "absorption_summary_rows": absorption_summary_rows,
        "absorption_overall_rows": absorption_overall_rows,
        "absorption_overall_summary_rows": absorption_overall_summary_rows,
        "surrogate_identity_unit_rows": surrogate_unit_rows,
        "surrogate_identity_aggregate_rows": surrogate_aggregate_rows,
        "surrogate_identity_overall_rows": surrogate_overall_rows,
        "length_summary": length,
        "length_subset_summary": subset,
        "difficulty_ranking": _difficulty_ranking(metric_rows),
    }


def aggregate_phase_root(
    config_path: str | Path,
    phase_root: str | Path,
    *,
    phase: str,
    resume: bool = False,
) -> dict[str, Any]:
    if phase not in {"smoke", "confirmation"}:
        raise DataConfigError("phase must be smoke or confirmation")
    config = load_ms_mean_head_factorial_config(config_path)
    root = Path(phase_root).resolve()
    if (root / "_PHASE_SUCCESS.json").is_file():
        if not resume:
            raise DataValidationError("phase aggregate is already complete")
        audit_phase_root(config_path, root, phase=phase, verify_root_hashes=True)
        return _read_json(root / "summary.json")
    aggregate_names = {
        "metrics_by_unit.csv",
        "fairness_by_unit.json",
        "integrity.json",
        "summary.json",
        "environment.json",
        "paired_deltas.csv",
        "unit_deltas.csv",
        "comparison_summary.json",
        "factorial_decision.json",
        "surrogate_absorption.csv",
        "surrogate_absorption_summary.csv",
        "surrogate_absorption_overall.csv",
        "surrogate_absorption_overall_summary.csv",
        "surrogate_identity_metrics_by_unit.csv",
        "surrogate_identity_metrics_aggregate.csv",
        "surrogate_identity_metrics_overall.csv",
        "length_pair_classification.json",
        "length_subset_summary.json",
        "difficulty_ranking.json",
        "artifact_hashes.json",
    }
    existing = [root / name for name in sorted(aggregate_names) if (root / name).exists()]
    if existing and not resume:
        raise DataValidationError(f"incomplete aggregate output exists: {existing[0]}")
    if existing:
        quarantine = (
            root.parent
            / "_quarantine"
            / root.name
            / f"interrupted_aggregate_{time.time_ns()}"
        )
        quarantine.mkdir(parents=True, exist_ok=False)
        for path in existing:
            path.replace(quarantine / path.name)
    audited = _collect_phase_audit(config, root, phase=phase)
    metric_rows = audited.pop("metric_rows")
    fairness_rows = audited.pop("fairness_rows")
    audited_rows = audited.pop("audited_rows")
    _write_csv(root / "metrics_by_unit.csv", metric_rows)
    _write_json(root / "fairness_by_unit.json", fairness_rows)
    integrity = {
        **audited,
        "config_sha256": config["_config_sha256"],
        "performance_used_for_smoke_gate": False,
        "length_padding_used_for_gate": False,
        "bootstrap_used_for_gate": False,
        "final_unknown_test_authorized": False,
    }
    _write_json(root / "integrity.json", integrity)
    if phase == "confirmation":
        derived = _build_confirmation_derived(
            root, metric_rows, audited_rows, config
        )
        factorial_summary = derived["factorial_summary"]
        _write_csv(root / "paired_deltas.csv", derived["pair_delta_rows"])
        _write_csv(root / "unit_deltas.csv", derived["unit_delta_rows"])
        _write_json(root / "comparison_summary.json", factorial_summary)
        _write_json(root / "factorial_decision.json", factorial_summary["decision"])
        _atomic_write_bytes(
            root / "surrogate_absorption.csv",
            _render_csv_with_fields(derived["absorption_rows"], ABSORPTION_FIELDS),
        )
        _write_csv(
            root / "surrogate_absorption_summary.csv",
            derived["absorption_summary_rows"],
        )
        _atomic_write_bytes(
            root / "surrogate_absorption_overall.csv",
            _render_csv_with_fields(
                derived["absorption_overall_rows"], ABSORPTION_OVERALL_FIELDS
            ),
        )
        _write_csv(
            root / "surrogate_absorption_overall_summary.csv",
            derived["absorption_overall_summary_rows"],
        )
        _write_csv(
            root / "surrogate_identity_metrics_by_unit.csv",
            derived["surrogate_identity_unit_rows"],
        )
        _write_csv(
            root / "surrogate_identity_metrics_aggregate.csv",
            derived["surrogate_identity_aggregate_rows"],
        )
        _write_csv(
            root / "surrogate_identity_metrics_overall.csv",
            derived["surrogate_identity_overall_rows"],
        )
        _write_json(
            root / "length_pair_classification.json", derived["length_summary"]
        )
        _write_json(
            root / "length_subset_summary.json", derived["length_subset_summary"]
        )
        _write_json(root / "difficulty_ranking.json", derived["difficulty_ranking"])
        analysis = factorial_summary
    else:
        analysis = _smoke_summary(metric_rows)
    project_root = Path(config["_config_path"]).parents[3]
    environment = _environment(project_root, torch.device("cpu"))
    environment["task_source_hashes"] = task_source_hashes(project_root)
    _write_json(root / "environment.json", environment)
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "status": "complete",
        "integrity": integrity,
        "analysis": analysis,
        "final_unknown_used": False,
        "even_angle_test_used": False,
        "final_unknown_test_authorized": False,
    }
    _write_json(root / "summary.json", summary)
    _write_json(root / "artifact_hashes.json", _phase_artifact_hashes(root))
    _write_json(
        root / "_PHASE_SUCCESS.json",
        {
            "status": "complete",
            "phase": phase,
            "config_sha256": config["_config_sha256"],
            "artifact_hash_manifest_sha256": file_sha256(
                root / "artifact_hashes.json"
            ),
            "summary_sha256": file_sha256(root / "summary.json"),
            "integrity_sha256": file_sha256(root / "integrity.json"),
            "final_unknown_test_authorized": False,
        },
    )
    return summary


def audit_phase_root(
    config_path: str | Path,
    phase_root: str | Path,
    *,
    phase: str,
    verify_root_hashes: bool = True,
) -> dict[str, Any]:
    config = load_ms_mean_head_factorial_config(config_path)
    root = Path(phase_root).resolve()
    audited = _collect_phase_audit(config, root, phase=phase)
    metric_rows = audited.pop("metric_rows")
    fairness_rows = audited.pop("fairness_rows")
    audited_rows = audited.pop("audited_rows")
    expected_integrity = {
        **audited,
        "config_sha256": config["_config_sha256"],
        "performance_used_for_smoke_gate": False,
        "length_padding_used_for_gate": False,
        "bootstrap_used_for_gate": False,
        "final_unknown_test_authorized": False,
    }
    if _read_json(root / "integrity.json") != expected_integrity:
        raise DataValidationError("stored phase integrity differs from full re-audit")
    if (root / "metrics_by_unit.csv").read_bytes() != _render_csv(metric_rows):
        raise DataValidationError("stored phase metric matrix differs")
    if _read_json(root / "fairness_by_unit.json") != fairness_rows:
        raise DataValidationError("stored phase fairness rows differ")
    if phase == "confirmation":
        derived = _build_confirmation_derived(root, metric_rows, audited_rows, config)
        factorial_summary = derived["factorial_summary"]
        if (
            (root / "paired_deltas.csv").read_bytes()
            != _render_csv(derived["pair_delta_rows"])
            or (root / "unit_deltas.csv").read_bytes()
            != _render_csv(derived["unit_delta_rows"])
            or _read_json(root / "comparison_summary.json") != factorial_summary
            or _read_json(root / "factorial_decision.json")
            != factorial_summary["decision"]
            or (root / "surrogate_absorption.csv").read_bytes()
            != _render_csv_with_fields(derived["absorption_rows"], ABSORPTION_FIELDS)
            or (root / "surrogate_absorption_summary.csv").read_bytes()
            != _render_csv(derived["absorption_summary_rows"])
            or (root / "surrogate_absorption_overall.csv").read_bytes()
            != _render_csv_with_fields(
                derived["absorption_overall_rows"], ABSORPTION_OVERALL_FIELDS
            )
            or (root / "surrogate_absorption_overall_summary.csv").read_bytes()
            != _render_csv(derived["absorption_overall_summary_rows"])
            or (root / "surrogate_identity_metrics_by_unit.csv").read_bytes()
            != _render_csv(derived["surrogate_identity_unit_rows"])
            or (root / "surrogate_identity_metrics_aggregate.csv").read_bytes()
            != _render_csv(derived["surrogate_identity_aggregate_rows"])
            or (root / "surrogate_identity_metrics_overall.csv").read_bytes()
            != _render_csv(derived["surrogate_identity_overall_rows"])
            or _read_json(root / "length_pair_classification.json")
            != derived["length_summary"]
            or _read_json(root / "length_subset_summary.json")
            != derived["length_subset_summary"]
            or _read_json(root / "difficulty_ranking.json")
            != derived["difficulty_ranking"]
        ):
            raise DataValidationError("stored confirmation analysis differs from re-audit")
        analysis = factorial_summary
    else:
        analysis = _smoke_summary(metric_rows)
    expected_summary = {
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "status": "complete",
        "integrity": expected_integrity,
        "analysis": analysis,
        "final_unknown_used": False,
        "even_angle_test_used": False,
        "final_unknown_test_authorized": False,
    }
    if _read_json(root / "summary.json") != expected_summary:
        raise DataValidationError("stored phase summary differs from full re-audit")
    if verify_root_hashes and _read_json(
        root / "artifact_hashes.json"
    ) != _phase_artifact_hashes(root):
        raise DataValidationError("phase root artifact hashes do not match")
    success = _read_json(root / "_PHASE_SUCCESS.json")
    expected_success = {
        "status": "complete",
        "phase": phase,
        "config_sha256": config["_config_sha256"],
        "artifact_hash_manifest_sha256": file_sha256(root / "artifact_hashes.json"),
        "summary_sha256": file_sha256(root / "summary.json"),
        "integrity_sha256": file_sha256(root / "integrity.json"),
        "final_unknown_test_authorized": False,
    }
    if success != expected_success:
        raise DataValidationError("phase success seal is invalid")
    return expected_integrity


def run_phase(
    config_path: str | Path,
    bundle_root: str | Path,
    output_root: str | Path,
    *,
    phase: str,
    device_request: str = "auto",
    resume: bool = False,
) -> dict[str, Any]:
    if phase == "confirmation":
        raise DataConfigError(
            "formal confirmation must use launch_confirmation_gpu to preserve the frozen 4x4 schedule"
        )
    config = load_ms_mean_head_factorial_config(config_path)
    output = Path(output_root).resolve()
    if (output / "_PHASE_SUCCESS.json").is_file() and resume:
        audit_phase_root(config_path, output, phase=phase, verify_root_hashes=True)
        return _read_json(output / "summary.json")
    if output.exists() and any(output.iterdir()) and not resume:
        raise DataValidationError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for unit in build_phase_plan(config, phase):
        for method in unit["methods"]:
            run_single_method(
                config_path,
                bundle_root,
                output,
                phase=phase,
                pair_id=str(unit["pair_id"]),
                angle_fold=int(unit["angle_fold"]),
                seed=int(unit["seed"]),
                method=method,
                device_request=device_request,
                resume=resume,
            )
    return aggregate_phase_root(config_path, output, phase=phase, resume=resume)


def launch_confirmation_gpu(
    config_path: str | Path,
    bundle_root: str | Path,
    output_root: str | Path,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Run the frozen 168 tasks with four persistent slots per physical GPU."""

    config = load_ms_mean_head_factorial_config(config_path)
    output = Path(output_root).resolve()
    if (output / "_PHASE_SUCCESS.json").is_file():
        if not resume:
            raise DataValidationError("confirmation phase is already complete")
        audit_phase_root(config_path, output, phase="confirmation")
        return _read_json(output / "summary.json")
    if output.exists() and any(output.iterdir()) and not resume:
        raise DataValidationError(f"confirmation output is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    expected_gpu_count = int(config["runtime"]["expected_gpu_count"])
    if torch.cuda.device_count() != expected_gpu_count:
        raise DataValidationError(
            f"launcher requires exactly {expected_gpu_count} visible GPUs"
        )
    observed_names = [
        torch.cuda.get_device_name(index) for index in range(expected_gpu_count)
    ]
    if any(name != config["runtime"]["expected_gpu_model"] for name in observed_names):
        raise DataValidationError(f"unexpected GPU model list: {observed_names}")
    jobs_per_gpu = int(config["runtime"]["jobs_per_gpu"])
    worker_count = expected_gpu_count * jobs_per_gpu
    if worker_count != int(config["runtime"]["total_parallel_jobs"]):
        raise DataValidationError("GPU worker count differs from frozen runtime")
    inherited_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    visible_tokens = (
        [token.strip() for token in inherited_visible.split(",")]
        if inherited_visible
        else [str(index) for index in range(expected_gpu_count)]
    )
    if len(visible_tokens) != expected_gpu_count or any(not token for token in visible_tokens):
        raise DataValidationError(
            "CUDA_VISIBLE_DEVICES must expose exactly four GPU indices or UUIDs"
        )
    payload = _plan_payload(config, "confirmation")
    tasks = list(payload["tasks"])
    if len(tasks) != 168:
        raise DataValidationError("confirmation launcher plan is not 168 tasks")
    worker_buckets: list[list[dict[str, Any]]] = [[] for _ in range(worker_count)]
    assignment_rows = []
    for index, task in enumerate(tasks):
        worker, gpu = _gpu_worker_assignment(
            index, gpu_count=expected_gpu_count, jobs_per_gpu=jobs_per_gpu
        )
        worker_buckets[worker].append(task)
        assignment_rows.append(
            {
                "task_index": index,
                "worker_slot": worker,
                "physical_gpu_index": gpu,
                "visible_gpu_token": visible_tokens[gpu],
                **task,
            }
        )
    launch_root = output / "launcher"
    launch_root.mkdir(exist_ok=True)
    _write_json(
        launch_root / "launch_manifest.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "phase": "confirmation",
            "config_sha256": config["_config_sha256"],
            "gpu_names": observed_names,
            "inherited_cuda_visible_devices": inherited_visible or None,
            "child_visible_gpu_tokens": visible_tokens,
            "jobs_per_gpu": jobs_per_gpu,
            "worker_count": worker_count,
            "gpu_assignment": "rotating_method_balanced_v1",
            "training_task_count": len(tasks),
            "assignments": assignment_rows,
            "resume": resume,
            "final_unknown_test_authorized": False,
        },
    )
    config_path = Path(config_path).resolve()
    bundle_root = Path(bundle_root).resolve()

    def worker(worker_slot: int) -> list[dict[str, Any]]:
        gpu = worker_slot // jobs_per_gpu
        results = []
        for task in worker_buckets[worker_slot]:
            label = (
                f"{task['pair_id']}_F{task['angle_fold']}_S{task['seed']}_"
                f"{task['method']}"
            )
            log_path = launch_root / f"worker_{worker_slot:02d}_{label}.log"
            command = [
                sys.executable,
                "-m",
                "hrrp_osr.training.ms_mean_head_factorial",
                "run-unit",
                "--config",
                str(config_path),
                "--bundle-root",
                str(bundle_root),
                "--output",
                str(output),
                "--phase",
                "confirmation",
                "--pair-id",
                str(task["pair_id"]),
                "--angle-fold",
                str(task["angle_fold"]),
                "--seed",
                str(task["seed"]),
                "--method",
                str(task["method"]),
                "--device",
                "cuda",
                "--resume",
            ]
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = visible_tokens[gpu]
            source_root = str(project_root / "src")
            inherited_pythonpath = environment.get("PYTHONPATH", "")
            environment["PYTHONPATH"] = (
                source_root
                if not inherited_pythonpath
                else source_root + os.pathsep + inherited_pythonpath
            )
            started = time.time()
            with log_path.open("wb") as handle:
                completed = subprocess.run(
                    command,
                    cwd=project_root,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            results.append(
                {
                    "worker_slot": worker_slot,
                    "physical_gpu_index": gpu,
                    "visible_gpu_token": visible_tokens[gpu],
                    **task,
                    "exit_code": completed.returncode,
                    "elapsed_seconds": time.time() - started,
                    "log": str(log_path.relative_to(output)),
                }
            )
        return results

    project_root = Path(config["_config_path"]).parents[3]
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(worker, slot) for slot in range(worker_count)]
        completed_rows = [
            row
            for future in concurrent.futures.as_completed(futures)
            for row in future.result()
        ]
    completed_rows.sort(
        key=lambda row: next(
            item["task_index"]
            for item in assignment_rows
            if all(
                item[key] == row[key]
                for key in ("pair_id", "angle_fold", "seed", "method")
            )
        )
    )
    _write_json(
        launch_root / "launch_results.json",
        {
            "training_task_count": len(completed_rows),
            "successful_task_count": sum(row["exit_code"] == 0 for row in completed_rows),
            "failed_task_count": sum(row["exit_code"] != 0 for row in completed_rows),
            "results": completed_rows,
        },
    )
    failures = [row for row in completed_rows if row["exit_code"] != 0]
    if failures:
        raise DataValidationError(
            f"{len(failures)} confirmation tasks failed; rerun the same launcher with --resume"
        )
    return aggregate_phase_root(
        config_path, output, phase="confirmation", resume=resume
    )


def _plan_payload(config: Mapping[str, Any], phase: str) -> dict[str, Any]:
    units = build_phase_plan(config, phase)
    tasks = [
        {
            "phase": phase,
            "pair_id": str(unit["pair_id"]),
            "angle_fold": int(unit["angle_fold"]),
            "seed": int(unit["seed"]),
            "method": method,
        }
        for unit in units
        for method in unit["methods"]
    ]
    return {
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "experimental_unit_count": len(units),
        "training_task_count": len(tasks),
        "tasks": tasks,
        "final_unknown_test_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the preregistered multiscale-mean x CE/ARPL factorial"
    )
    parser.add_argument(
        "action",
        choices=(
            "validate",
            "plan",
            "smoke",
            "confirmation",
            "launch-confirmation-gpu",
            "run-unit",
            "aggregate",
            "audit",
        ),
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--phase", choices=("smoke", "confirmation"))
    parser.add_argument("--pair-id", choices=tuple(row[0] for row in IDENTITY_PAIRS))
    parser.add_argument("--angle-fold", type=int, choices=ANGLE_FOLDS)
    parser.add_argument("--seed", type=int, choices=CONFIRMATION_SEEDS)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "validate":
        config = load_ms_mean_head_factorial_config(args.config)
        result: Any = {
            "status": "passed",
            "experiment_id": EXPERIMENT_ID,
            "config_sha256": config["_config_sha256"],
            "confirmation_training_task_count": len(
                _plan_payload(config, "confirmation")["tasks"]
            ),
            "final_unknown_test_authorized": False,
        }
    elif args.action == "plan":
        if args.phase is None:
            parser.error("plan requires --phase")
        result = _plan_payload(
            load_ms_mean_head_factorial_config(args.config), args.phase
        )
    elif args.action == "smoke":
        if args.bundle_root is None or args.output is None:
            parser.error("training actions require --bundle-root and --output")
        result = run_phase(
            args.config,
            args.bundle_root,
            args.output,
            phase=args.action,
            device_request=args.device,
            resume=args.resume,
        )
    elif args.action in {"confirmation", "launch-confirmation-gpu"}:
        if args.bundle_root is None or args.output is None:
            parser.error("GPU launcher requires --bundle-root and --output")
        result = launch_confirmation_gpu(
            args.config,
            args.bundle_root,
            args.output,
            resume=args.resume,
        )
    elif args.action == "run-unit":
        required_values = (
            args.bundle_root,
            args.output,
            args.phase,
            args.pair_id,
            args.angle_fold,
            args.seed,
            args.method,
        )
        if any(value is None for value in required_values):
            parser.error(
                "run-unit requires bundle-root, output, phase, pair-id, angle-fold, seed, and method"
            )
        result = run_single_method(
            args.config,
            args.bundle_root,
            args.output,
            phase=args.phase,
            pair_id=args.pair_id,
            angle_fold=args.angle_fold,
            seed=args.seed,
            method=args.method,
            device_request=args.device,
            resume=args.resume,
        )
    else:
        if args.output is None or args.phase is None:
            parser.error("aggregate/audit require --output and --phase")
        if args.action == "aggregate":
            result = aggregate_phase_root(
                args.config, args.output, phase=args.phase, resume=args.resume
            )
        else:
            result = audit_phase_root(
                args.config,
                args.output,
                phase=args.phase,
                verify_root_hashes=True,
            )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
