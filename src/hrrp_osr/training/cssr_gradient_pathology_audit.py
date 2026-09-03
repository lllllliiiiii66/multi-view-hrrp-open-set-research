from __future__ import annotations

import argparse
import io
import json
import math
import os
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
from hrrp_osr.models.cssr_e2e_1d import (
    FGMVCSSRE2EModel,
    Q2_E2E_REL_CSSR_1X1,
)
from hrrp_osr.training.arpl_pilot import _resolve_device, _set_determinism
from hrrp_osr.training.fg_mv_cssr_e2e_redesign import (
    ANGLE_FOLD,
    FINETUNE_SEED,
    MODEL_SEED,
    TASK_SOURCE_FILES as LEGACY_TASK_SOURCE_FILES,
    _build_optimizer,
    _common_r2_state_sha256,
    _configure_runtime,
    _frozen_prefix_sha256,
    _git_environment,
    _learning_rate_factor,
    _materialize_schedule_inputs,
    _normalized_unique_base_inputs,
    _set_optimizer_lrs,
    _state_sha256,
    build_epoch_pair_schedule,
    calibration_diagnostics,
    infer_e2e_model,
    load_fg_mv_cssr_e2e_config,
)
from hrrp_osr.training.fg_mv_cssr_pilot import (
    _atomic_write_bytes,
    _load_bundle,
    _load_prior_config,
    _prepare_frozen_split,
    _read_csv,
    _read_json,
    _sequence_sha256,
    _write_csv,
    _write_json,
    build_unique_base_sample_manifest,
    load_and_audit_frozen_r2,
)


EXPERIMENT_ID = "fg_mv_cssr_decoupled_audit_v3"
CONFIG_RELATIVE_PATH = (
    "configs/experiments/cssr/fg_mv_cssr_decoupled_audit_v3.yaml"
)
LEGACY_CONFIG_RELATIVE_PATH = (
    "configs/experiments/cssr/fg_mv_cssr_e2e_redesign_v2.yaml"
)
LEGACY_CONFIG_SHA256 = (
    "5c227c00a7ac5a88c9bf5d66618964bc05c67f45c51c2a880731f6753626512e"
)
AUDIT_PAIRS = ("N1", "N4")
AUDIT_METHOD = Q2_E2E_REL_CSSR_1X1
AUDIT_EPOCHS = 5
AUDIT_SEED = 20260904
RELATIVE_WEIGHT = 0.5
RATIO_DENOMINATOR_FLOOR = 1.0e-12
COSINE_PRODUCT_FLOOR = 1.0e-24
RELATIVE_DISCONNECTED_TOLERANCE = 1.0e-12
ORIGINAL_RATIO_LIMIT = 100.0
ORIGINAL_STREAK_LIMIT = 3
CLIP_EPSILON = 1.0e-6

EXPECTED_R2_CHECKPOINT_SHA256 = {
    "N1": "a4f6fa3235fbb5cf74b712588a0318f614a05287adec4ee881820424cddbcbaa",
    "N4": "169387ad7a87463110ac7a2cd45afd7dac49428538c93c84975162e425d94ff5",
}
EXPECTED_R2_PAIR_MANIFEST_SHA256 = {
    "N1": "0b8a97dcfd744896bbae912c1363379201ced18a55107f80b2d2f3256fb5c5bc",
    "N4": "8b0202d1e08ae83eec4bf07fc1dbb6a3f39fef2378ac15e57635709d8872b41a",
}
EXPECTED_EPOCH0_R2_STATE_SHA256 = {
    "N1": "3a7e74e89d11812877409d415c505c087a913e3b08098ab1fa583fa1151aeb07",
    "N4": "67825e2bc143b32ed52beb5778e1190bc809269dfaedfb516a692eac64fa31f2",
}
EXPECTED_Q2_AE_INITIAL_SHA256 = (
    "4c3257678eaca1bd20ea2e97b09aeaf87fdd23c71ff65dcc1696dfe61963d6fb"
)
EXPECTED_SCHEDULE_SHA256 = {
    "N1": (
        "c2583a5ef3bea986a97fb14aac738e03a16ffc0f794a13ccc3951aaad4468922",
        "3e89d9a8cc76685d8dead18614818cf5480b0f308c348e744e773b9e17d8f498",
        "0e591a578934bbaaebf42437870b328a92923f5799975a43e1943272e9601b39",
        "85a8bd03d420797559afba991931f83dbd3189ccc17d59f54fe6644729c0fcd6",
        "b07a1347d955ca466736e570b370fcf719c16f2ed1aff404cd2f7871e53f53fe",
    ),
    "N4": (
        "c6090d55d3500feb6e578d29c5738e8696ddd87eb68d01207a8dc3f0d1acd6f1",
        "9321d11bbc7d15ea3965d9fb3dfc84181f48877a972955696b8b1f35fa2ef582",
        "82244ed649e5f4c4cb192feb1a227118889521729431945e968767f6465fac10",
        "ec11524bcf16eb0001878de2a2d5cb82c6c11d9f95c7b636bb3973feb96f50f1",
        "85673d1df05cc1190aadafcfc93d1782363288467ec27c28f7110f54dc06951a",
    ),
}

AUDIT_SOURCE_FILES = tuple(
    dict.fromkeys(
        (
            *LEGACY_TASK_SOURCE_FILES,
            CONFIG_RELATIVE_PATH,
            "src/hrrp_osr/training/cssr_gradient_pathology_audit.py",
        )
    )
)

PARAMETER_GROUP_NAMES = (
    "last_residual_stage",
    "projection",
    "ce_head",
    "cssr_autoencoders",
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DataConfigError(f"{name} must be a mapping")
    return value


def _require(errors: list[str], observed: Any, expected: Any, name: str) -> None:
    if observed != expected:
        errors.append(f"{name} changed: expected {expected!r}, observed {observed!r}")


def _gradient_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the independently preregistered stage-A configuration block."""

    return _mapping(
        config.get("gradient_pathology_audit"), "gradient_pathology_audit"
    )


def _legacy_q2_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(config.get("legacy_e2e_q2"), "legacy_e2e_q2")


def load_gradient_pathology_config(path: str | Path) -> dict[str, Any]:
    """Load the v3 config and fail closed on every stage-A invariant."""

    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = dict(_mapping(yaml.safe_load(handle), "decoupled CSSR config"))
    audit = _gradient_section(config)
    legacy = _legacy_q2_section(config)
    errors: list[str] = []
    _require(errors, config.get("schema_version"), 1, "schema_version")
    _require(
        errors,
        config.get("stage"),
        "P3_fg_mv_cssr_gradient_audit_and_decoupled_validation",
        "stage",
    )
    _require(errors, config.get("experiment_id"), EXPERIMENT_ID, "experiment_id")
    _require(errors, legacy.get("experiment_id"), "fg_mv_cssr_e2e_redesign_v2", "legacy experiment")
    _require(
        errors,
        legacy.get("source_config"),
        LEGACY_CONFIG_RELATIVE_PATH,
        "legacy config path",
    )
    _require(errors, legacy.get("method"), AUDIT_METHOD, "audit method")
    _require(errors, list(legacy.get("audit_pairs", [])), list(AUDIT_PAIRS), "audit pairs")
    _require(errors, int(legacy.get("audit_epochs", -1)), AUDIT_EPOCHS, "audit epochs")
    _require(errors, int(legacy.get("audit_seed", -1)), AUDIT_SEED, "audit seed")
    _require(
        errors,
        legacy.get("source_config_sha256"),
        LEGACY_CONFIG_SHA256,
        "legacy config SHA-256",
    )
    _require(errors, legacy.get("expected_r2_checkpoint_sha256"), EXPECTED_R2_CHECKPOINT_SHA256, "R2 checkpoint hashes")
    _require(errors, legacy.get("expected_epoch0_common_r2_state_sha256"), EXPECTED_EPOCH0_R2_STATE_SHA256, "epoch-0 R2 state hashes")
    _require(errors, legacy.get("expected_ae_initial_state_sha256"), EXPECTED_Q2_AE_INITIAL_SHA256, "Q2 AE initial hash")
    _require(
        errors,
        {
            key: tuple(value)
            for key, value in _mapping(
                legacy.get("expected_epoch_schedule_sha256"),
                "legacy_e2e_q2.expected_epoch_schedule_sha256",
            ).items()
        },
        EXPECTED_SCHEDULE_SHA256,
        "epoch schedule hashes",
    )
    prior_units = _mapping(
        _mapping(config.get("prior_r2"), "prior_r2").get("unit_artifact_hashes"),
        "prior_r2.unit_artifact_hashes",
    )
    for pair_id in AUDIT_PAIRS:
        unit = _mapping(prior_units.get(pair_id), f"prior_r2 unit {pair_id}")
        _require(
            errors,
            unit.get("checkpoint.pt"),
            EXPECTED_R2_CHECKPOINT_SHA256[pair_id],
            f"{pair_id} R2 checkpoint hash",
        )
        _require(
            errors,
            unit.get("pair_manifest.csv"),
            EXPECTED_R2_PAIR_MANIFEST_SHA256[pair_id],
            f"{pair_id} R2 pair manifest hash",
        )
    for name, expected in {
        "diagnostic_only": True,
        "disable_only_original_100x_exception": True,
        "denominator_floor": RATIO_DENOMINATOR_FLOOR,
        "clip_epsilon": CLIP_EPSILON,
        "cosine_undefined_norm_product_max": COSINE_PRODUCT_FLOOR,
        "quantile_method": "linear",
        "original_gate_maximum_ratio": ORIGINAL_RATIO_LIMIT,
        "original_gate_consecutive_epochs": ORIGINAL_STREAK_LIMIT,
        "atomic_batch_diagnostics": True,
    }.items():
        _require(errors, audit.get(name), expected, f"gradient_pathology_audit.{name}")
    _require(
        errors,
        [float(value) for value in audit.get("ce_small_gradient_thresholds", [])],
        [1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8],
        "CE small-gradient thresholds",
    )
    label_thresholds = _mapping(
        audit.get("label_thresholds"), "gradient_pathology_audit.label_thresholds"
    )
    for name, expected in {
        "small_ce_batch_fraction": 0.5,
        "large_ratio_epoch_count": 3,
        "large_absolute_weighted_relative_to_ce": 5.0,
        "frequent_clipping_fraction": 0.5,
        "abnormal_relative_update": 0.01,
        "strong_conflict_median_cosine_max": -0.25,
        "strong_conflict_negative_fraction_min": 0.75,
        "rapid_calibration_accuracy_drop": 0.02,
        "rapid_calibration_nll_increase": 0.1,
    }.items():
        _require(errors, label_thresholds.get(name), expected, f"label threshold {name}")
    _require(
        errors,
        list(audit.get("label_priority", [])),
        [
            "mixed_gradient_conflict",
            "true_auxiliary_domination",
            "ratio_denominator_collapse_likely",
            "inconclusive",
        ],
        "pathology label priority",
    )
    evidence = _mapping(config.get("evidence_scope"), "evidence_scope")
    for name in (
        "final_unknown_classes_used",
        "even_angle_test_used",
        "surrogate_unknown_used_for_training",
    ):
        _require(errors, evidence.get(name), False, f"evidence_scope.{name}")
    _require(
        errors,
        _mapping(config.get("data"), "data").get("final_test_pairs_generated"),
        False,
        "final test pair generation",
    )
    _require(
        errors,
        _mapping(config.get("outputs"), "outputs").get(
            "final_unknown_test_authorized"
        ),
        False,
        "final unknown authorization",
    )
    if errors:
        raise DataConfigError(
            "Invalid CSSR gradient pathology audit config:\n- " + "\n- ".join(errors)
        )
    config["_config_path"] = str(config_path)
    config["_config_sha256"] = file_sha256(config_path)
    return config


def load_legacy_q2_config(project_root: Path) -> dict[str, Any]:
    path = project_root / LEGACY_CONFIG_RELATIVE_PATH
    if file_sha256(path) != LEGACY_CONFIG_SHA256:
        raise DataValidationError("legacy Q2 config SHA-256 changed")
    config = load_fg_mv_cssr_e2e_config(path)
    weights = config["loss"]["weights"][AUDIT_METHOD]
    if dict(weights) != {
        "classification": 1.0,
        "relative": RELATIVE_WEIGHT,
        "absolute": 0.0,
        "separation": 0.0,
    }:
        raise DataValidationError("legacy Q2 loss semantics changed")
    return config


def build_gradient_audit_plan(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    audit = _legacy_q2_section(config)
    pairs = tuple(audit.get("audit_pairs", ()))
    method = str(audit.get("method", ""))
    epochs = int(audit.get("audit_epochs", -1))
    seed = int(audit.get("audit_seed", -1))
    if pairs != AUDIT_PAIRS or method != AUDIT_METHOD or epochs != AUDIT_EPOCHS or seed != AUDIT_SEED:
        raise DataValidationError("gradient audit plan differs from preregistration")
    return [
        {
            "experiment_id": EXPERIMENT_ID,
            "phase": "gradient_pathology_audit",
            "pair_id": pair_id,
            "method": AUDIT_METHOD,
            "angle_fold": ANGLE_FOLD,
            "model_seed": MODEL_SEED,
            "audit_seed": AUDIT_SEED,
            "epochs": AUDIT_EPOCHS,
            "performance_gate_eligible": False,
            "final_unknown_test_authorized": False,
        }
        for pair_id in AUDIT_PAIRS
    ]


def audit_task_source_hashes(project_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in AUDIT_SOURCE_FILES:
        path = project_root / relative
        if not path.is_file():
            raise DataValidationError(f"missing gradient-audit source: {relative}")
        result[relative] = file_sha256(path)
    return result


def _parameter_groups(
    model: FGMVCSSRE2EModel,
) -> dict[str, tuple[nn.Parameter, ...]]:
    if model.cssr_core is None:
        raise DataValidationError("Q2 gradient audit requires class autoencoders")
    groups = {
        "last_residual_stage": tuple(model.encoder.stages[2].parameters()),
        "projection": tuple(model.encoder.projection.parameters()),
        "ce_head": tuple(model.global_head.parameters()),
        "cssr_autoencoders": tuple(model.cssr_core.parameters()),
    }
    flattened = [parameter for values in groups.values() for parameter in values]
    if not all(groups.values()) or len(flattened) != len({id(value) for value in flattened}):
        raise DataValidationError("gradient-audit parameter groups are empty or overlap")
    if set(map(id, flattened)) != {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }:
        raise DataValidationError("gradient-audit groups do not cover trainable parameters")
    return groups


def _loss_gradients(
    loss: torch.Tensor,
    parameters: Sequence[nn.Parameter],
    *,
    retain_graph: bool,
) -> tuple[torch.Tensor | None, ...]:
    if not loss.requires_grad:
        return tuple(None for _ in parameters)
    return tuple(
        torch.autograd.grad(
            loss,
            tuple(parameters),
            retain_graph=retain_graph,
            allow_unused=True,
        )
    )


def _gradient_tuple_norm(
    gradients: Sequence[torch.Tensor | None],
    *,
    reference: torch.Tensor,
) -> float:
    squared = reference.new_zeros(())
    for gradient in gradients:
        if gradient is not None:
            squared = squared + gradient.detach().square().sum()
    return float(torch.sqrt(squared).item())


def _parameter_gradient_norm(
    parameters: Sequence[nn.Parameter],
    *,
    reference: torch.Tensor,
) -> float:
    return _gradient_tuple_norm(
        tuple(parameter.grad for parameter in parameters), reference=reference
    )


def gradient_pair_statistics(
    classification_gradients: Sequence[torch.Tensor | None],
    relative_gradients: Sequence[torch.Tensor | None],
    *,
    reference: torch.Tensor,
    relative_weight: float = RELATIVE_WEIGHT,
) -> dict[str, float | None]:
    """Compute the norm, dot, ratio and cosine on one ordered parameter vector."""

    if len(classification_gradients) != len(relative_gradients):
        raise DataValidationError("classification and relative gradients do not align")
    cls_norm = _gradient_tuple_norm(classification_gradients, reference=reference)
    rel_raw_norm = _gradient_tuple_norm(relative_gradients, reference=reference)
    rel_weighted_norm = abs(float(relative_weight)) * rel_raw_norm
    dot_tensor = reference.new_zeros(())
    for cls_gradient, rel_gradient in zip(
        classification_gradients, relative_gradients, strict=True
    ):
        if cls_gradient is not None and rel_gradient is not None:
            dot_tensor = dot_tensor + (
                cls_gradient.detach() * (float(relative_weight) * rel_gradient.detach())
            ).sum()
    dot = float(dot_tensor.item())
    product = cls_norm * rel_weighted_norm
    cosine = None if product <= COSINE_PRODUCT_FLOOR else dot / product
    return {
        "classification_gradient_norm": cls_norm,
        "relative_raw_gradient_norm": rel_raw_norm,
        "relative_weighted_gradient_norm": rel_weighted_norm,
        "weighted_relative_to_classification_ratio": rel_weighted_norm
        / max(cls_norm, RATIO_DENOMINATOR_FLOOR),
        "classification_weighted_relative_dot": dot,
        "classification_weighted_relative_cosine": cosine,
    }


def clip_diagnostics(pre_clip_total_norm: float, clip_norm: float) -> dict[str, Any]:
    if not math.isfinite(pre_clip_total_norm) or pre_clip_total_norm < 0.0:
        raise DataValidationError("pre-clip total gradient norm is invalid")
    if not math.isfinite(clip_norm) or clip_norm <= 0.0:
        raise DataValidationError("gradient clip norm is invalid")
    scale = min(1.0, clip_norm / (pre_clip_total_norm + CLIP_EPSILON))
    return {
        "pre_clip_total_gradient_norm": float(pre_clip_total_norm),
        "gradient_clipping_scale": float(scale),
        "post_clip_estimated_gradient_norm": float(pre_clip_total_norm * scale),
        "gradient_clipped": bool(pre_clip_total_norm > clip_norm),
    }


def snapshot_parameter_groups(
    groups: Mapping[str, Sequence[nn.Parameter]],
) -> dict[str, tuple[torch.Tensor, ...]]:
    return {
        name: tuple(parameter.detach().clone() for parameter in parameters)
        for name, parameters in groups.items()
    }


def relative_parameter_updates(
    before: Mapping[str, Sequence[torch.Tensor]],
    groups: Mapping[str, Sequence[nn.Parameter]],
) -> dict[str, float]:
    result: dict[str, float] = {}
    if set(before) != set(groups):
        raise DataValidationError("parameter-update groups changed")
    for name in groups:
        old_values = tuple(before[name])
        new_values = tuple(groups[name])
        if len(old_values) != len(new_values):
            raise DataValidationError("parameter-update tensor population changed")
        numerator = 0.0
        denominator = 0.0
        for old, new in zip(old_values, new_values, strict=True):
            old64 = old.detach().to(dtype=torch.float64)
            new64 = new.detach().to(dtype=torch.float64)
            numerator += float((new64 - old64).square().sum().item())
            denominator += float(old64.square().sum().item())
        result[name] = math.sqrt(numerator) / max(
            math.sqrt(denominator), RATIO_DENOMINATOR_FLOOR
        )
    return result


def summarize_gradient_epoch(
    batch_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute every preregistered epoch statistic from batch diagnostics."""

    if not batch_rows:
        raise DataValidationError("cannot summarize an empty gradient-audit epoch")
    cls = np.asarray(
        [float(row["classification_gradient_norm"]) for row in batch_rows],
        dtype=np.float64,
    )
    rel = np.asarray(
        [float(row["relative_weighted_gradient_norm"]) for row in batch_rows],
        dtype=np.float64,
    )
    ratios = np.asarray(
        [float(row["weighted_relative_to_classification_ratio"]) for row in batch_rows],
        dtype=np.float64,
    )
    if not all(np.isfinite(values).all() for values in (cls, rel, ratios)):
        raise DataValidationError("gradient epoch contains NaN or Inf")
    cosine_values = np.asarray(
        [
            float(row["classification_weighted_relative_cosine"])
            for row in batch_rows
            if row.get("classification_weighted_relative_cosine") is not None
        ],
        dtype=np.float64,
    )
    if not np.isfinite(cosine_values).all():
        raise DataValidationError("gradient epoch contains non-finite cosine")
    batch_count = len(batch_rows)
    cosine_defined = int(cosine_values.size)
    return {
        "batch_count": batch_count,
        "mean_of_batch_ratios": float(ratios.mean()),
        "ratio_of_mean_norms": float(rel.mean())
        / max(float(cls.mean()), RATIO_DENOMINATOR_FLOOR),
        "rms_norm_ratio": float(np.sqrt(np.mean(np.square(rel))))
        / max(float(np.sqrt(np.mean(np.square(cls)))), RATIO_DENOMINATOR_FLOOR),
        "ratio_min": float(ratios.min()),
        "ratio_median": float(np.quantile(ratios, 0.5, method="linear")),
        "ratio_p90": float(np.quantile(ratios, 0.9, method="linear")),
        "ratio_p95": float(np.quantile(ratios, 0.95, method="linear")),
        "ratio_max": float(ratios.max()),
        "weighted_relative_gradient_norm_median": float(
            np.quantile(rel, 0.5, method="linear")
        ),
        "cosine_mean": None if not cosine_defined else float(cosine_values.mean()),
        "cosine_median": None
        if not cosine_defined
        else float(np.quantile(cosine_values, 0.5, method="linear")),
        "cosine_positive_fraction": None
        if not cosine_defined
        else float(np.count_nonzero(cosine_values > 0.0)) / cosine_defined,
        "cosine_negative_fraction": None
        if not cosine_defined
        else float(np.count_nonzero(cosine_values < 0.0)) / cosine_defined,
        "cosine_undefined_fraction": float(batch_count - cosine_defined) / batch_count,
        **{
            f"classification_gradient_below_{threshold:.0e}_fraction": float(
                np.count_nonzero(cls < threshold)
            )
            / batch_count
            for threshold in (1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8)
        },
        "gradient_clipping_fraction": float(
            np.mean([bool(row["gradient_clipped"]) for row in batch_rows])
        ),
    }


def annotate_original_gate(epoch_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    streak = 0
    first_trigger: int | None = None
    annotated: list[dict[str, Any]] = []
    for expected_epoch, source in enumerate(epoch_rows, start=1):
        row = dict(source)
        if int(row["epoch"]) != expected_epoch:
            raise DataValidationError("gradient-audit epochs are not contiguous")
        violation = float(row["mean_of_batch_ratios"]) > ORIGINAL_RATIO_LIMIT
        streak = streak + 1 if violation else 0
        triggered = streak >= ORIGINAL_STREAK_LIMIT
        if triggered and first_trigger is None:
            first_trigger = expected_epoch
        row.update(
            {
                "original_100x_epoch_violation": violation,
                "original_100x_violation_streak": streak,
                "would_have_triggered_original_100x_gate": triggered,
            }
        )
        annotated.append(row)
    return {
        "epochs": annotated,
        "would_have_triggered_original_100x_gate": first_trigger is not None,
        "first_original_100x_trigger_epoch": first_trigger,
        "not_triggered_within_5_epochs": first_trigger is None
        and len(annotated) == AUDIT_EPOCHS,
    }


def classify_gradient_pathology(
    epoch_rows: Sequence[Mapping[str, Any]],
    *,
    epoch0_calibration: Mapping[str, Any],
    numerical_anomaly: bool = False,
) -> dict[str, Any]:
    """Apply the frozen explanatory rubric; it never authorizes stage B."""

    if len(epoch_rows) != AUDIT_EPOCHS:
        raise DataValidationError("pathology label requires all five audit epochs")
    total_batches = sum(int(row["batch_count"]) for row in epoch_rows)
    small_batches = sum(
        float(row["classification_gradient_below_1e-04_fraction"])
        * int(row["batch_count"])
        for row in epoch_rows
    )
    small_ce = small_batches / total_batches >= 0.5
    robust = (
        sum(
            float(row["ratio_of_mean_norms"]) > ORIGINAL_RATIO_LIMIT
            or float(row["rms_norm_ratio"]) > ORIGINAL_RATIO_LIMIT
            for row in epoch_rows
        )
        >= 3
    )
    large_absolute = (
        sum(
            float(row["weighted_relative_gradient_norm_median"]) >= 5.0
            for row in epoch_rows
        )
        >= 3
    )
    frequent_clipping = (
        sum(float(row["gradient_clipping_fraction"]) >= 0.5 for row in epoch_rows)
        >= 3
    )
    large_update = any(
        float(row["parameter_relative_updates"]["last_residual_stage"]) >= 0.01
        for row in epoch_rows
    )
    strong_conflict = (
        sum(
            row.get("cosine_median") is not None
            and float(row["cosine_median"]) <= -0.25
            and float(row["cosine_negative_fraction"]) >= 0.75
            for row in epoch_rows
        )
        >= 3
    )
    initial_accuracy = float(epoch0_calibration["accuracy"])
    initial_nll = float(epoch0_calibration["nll"])
    rapid_drift = any(
        float(row["calibration"]["accuracy"]) <= initial_accuracy - 0.02
        or float(row["calibration"]["nll"]) >= initial_nll + 0.1
        for row in epoch_rows
    )
    all_low_clipping = all(
        float(row["gradient_clipping_fraction"]) < 0.1 for row in epoch_rows
    )
    all_small_update = all(
        float(row["parameter_relative_updates"]["last_residual_stage"]) < 0.01
        for row in epoch_rows
    )
    if small_ce and (strong_conflict or frequent_clipping or rapid_drift):
        label = "mixed_gradient_conflict"
    elif (
        not small_ce
        and robust
        and (
            large_absolute
            or frequent_clipping
            or large_update
            or strong_conflict
            or rapid_drift
        )
    ):
        label = "true_auxiliary_domination"
    elif (
        small_ce
        and not robust
        and all_low_clipping
        and all_small_update
        and not numerical_anomaly
    ):
        label = "ratio_denominator_collapse_likely"
    else:
        label = "inconclusive"
    return {
        "label": label,
        "evidence": {
            "small_ce_denominator_evidence": small_ce,
            "robust_auxiliary_domination_evidence": robust,
            "large_absolute_auxiliary_evidence": large_absolute,
            "frequent_clipping_evidence": frequent_clipping,
            "large_parameter_update_evidence": large_update,
            "strong_conflict_evidence": strong_conflict,
            "rapid_calibration_drift": rapid_drift,
            "numerical_anomaly": bool(numerical_anomaly),
            "all_epoch_clipping_fraction_below_0_1": all_low_clipping,
            "all_epoch_last_stage_relative_update_below_0_01": all_small_update,
            "classification_gradient_below_1e_4_all_batch_fraction": small_batches
            / total_batches,
        },
        "performance_gate_eligible": False,
        "stage_b_decision": "continue_if_no_code_or_numerical_failure",
        "final_unknown_test_authorized": False,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _render_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(_json_safe(dict(row)), ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n"
        for row in rows
    ).encode("utf-8")


class AtomicGradientAuditSink:
    """Latest-complete snapshots for normal progress and all failure paths."""

    def __init__(self, staging: Path) -> None:
        self.staging = staging
        self.batch_rows: list[dict[str, Any]] = []
        self.epoch_rows: list[dict[str, Any]] = []
        self.current_state: dict[str, Any] = {"status": "initializing"}
        staging.mkdir(parents=True, exist_ok=False)

    def set_current_state(self, **values: Any) -> None:
        self.current_state.update(_json_safe(values))
        _write_json(self.staging / "latest_state.json", self.current_state)

    def append_batch(self, row: Mapping[str, Any]) -> None:
        self.batch_rows.append(dict(_json_safe(row)))
        _atomic_write_bytes(
            self.staging / "batch_diagnostics.jsonl",
            _render_jsonl(self.batch_rows),
        )
        self.set_current_state(
            status="training",
            last_completed_epoch=int(row["epoch"]),
            last_completed_batch=int(row["batch_index"]),
            completed_batch_count=len(self.batch_rows),
        )

    def append_epoch(self, row: Mapping[str, Any]) -> None:
        self.epoch_rows.append(dict(_json_safe(row)))
        _atomic_write_bytes(
            self.staging / "epoch_diagnostics.jsonl",
            _render_jsonl(self.epoch_rows),
        )
        self.set_current_state(
            status="training",
            last_completed_full_epoch=int(row["epoch"]),
            completed_epoch_count=len(self.epoch_rows),
        )

    def save_failure(self, error: BaseException, **context: Any) -> None:
        state = {
            **self.current_state,
            **_json_safe(context),
            "status": "failed",
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "completed_batch_count": len(self.batch_rows),
            "completed_epoch_count": len(self.epoch_rows),
        }
        pending = state.get("pending_batch")
        if isinstance(pending, Mapping):
            _write_json(self.staging / "failure_batch_diagnostic.json", pending)
        self.current_state = dict(state)
        _write_json(self.staging / "failure_state.json", state)
        _write_json(self.staging / "latest_state.json", state)


class _BatchDiagnosticFailure(DataValidationError):
    """Carry the largest finite diagnostic prefix to the atomic failure path."""

    def __init__(self, message: str, diagnostic: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.diagnostic = dict(_json_safe(diagnostic))


def _audit_artifact_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and str(path.relative_to(root))
        not in {
            "artifact_hashes.json",
            "_SUCCESS.json",
            "_FAILED.json",
            "_PHASE_SUCCESS.json",
        }
    }


def _finalize_marker(root: Path, *, success: bool, summary_name: str) -> None:
    _write_json(root / "artifact_hashes.json", _audit_artifact_hashes(root))
    marker = "_SUCCESS.json" if success else "_FAILED.json"
    payload = {
        "status": "complete" if success else "failed",
        "artifact_hashes_sha256": file_sha256(root / "artifact_hashes.json"),
    }
    summary_path = root / summary_name
    if summary_path.is_file():
        payload[f"{summary_path.stem}_sha256"] = file_sha256(summary_path)
    _write_json(root / marker, payload)


def _finite_parameters(model: nn.Module) -> bool:
    return all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters())


def _record_batch_diagnostics(
    *,
    model: FGMVCSSRE2EModel,
    losses: Any,
    output: Any,
    labels: torch.Tensor,
    groups: Mapping[str, Sequence[nn.Parameter]],
    epoch: int,
    batch_index: int,
    batch_start: int,
    pair_ids: Sequence[str],
    clip_norm: float,
) -> tuple[dict[str, Any], torch.Tensor]:
    partial: dict[str, Any] = {
        "epoch": int(epoch),
        "batch_index": int(batch_index),
        "batch_start": int(batch_start),
        "batch_size": int(labels.numel()),
        "batch_pair_id_sequence_sha256": _sequence_sha256(pair_ids),
        "classification_loss": float(losses.classification_loss.detach().item()),
        "relative_loss": float(losses.relative_loss.detach().item()),
        "total_loss": float(losses.total_loss.detach().item()),
    }
    stage = groups["last_residual_stage"]
    classification_gradients = _loss_gradients(
        losses.classification_loss, stage, retain_graph=True
    )
    relative_parameters = (
        *stage,
        *groups["projection"],
        *groups["ce_head"],
    )
    relative_gradients = _loss_gradients(
        losses.relative_loss, relative_parameters, retain_graph=True
    )
    stage_count = len(stage)
    projection_count = len(groups["projection"])
    relative_stage = relative_gradients[:stage_count]
    relative_projection = relative_gradients[
        stage_count : stage_count + projection_count
    ]
    relative_head = relative_gradients[stage_count + projection_count :]
    statistics = gradient_pair_statistics(
        classification_gradients,
        relative_stage,
        reference=losses.classification_loss,
    )
    relative_projection_norm = _gradient_tuple_norm(
        relative_projection, reference=losses.classification_loss
    )
    relative_head_norm = _gradient_tuple_norm(
        relative_head, reference=losses.classification_loss
    )
    partial.update(
        {
            **statistics,
            "relative_projection_gradient_norm": relative_projection_norm,
            "relative_ce_head_gradient_norm": relative_head_norm,
        }
    )
    if (
        relative_projection_norm > RELATIVE_DISCONNECTED_TOLERANCE
        or relative_head_norm > RELATIVE_DISCONNECTED_TOLERANCE
    ):
        raise _BatchDiagnosticFailure(
            "L_rel unexpectedly reaches projection or CE head", partial
        )

    losses.total_loss.backward()
    if any(
        parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    ):
        raise _BatchDiagnosticFailure(
            "gradient-audit training gradient is NaN or Inf",
            {**partial, "all_training_gradients_finite": False},
        )
    total_group_norms = {
        name: _parameter_gradient_norm(parameters, reference=losses.total_loss)
        for name, parameters in groups.items()
    }
    returned_pre_clip = torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        max_norm=float(clip_norm),
    )
    if not bool(torch.isfinite(returned_pre_clip)):
        raise _BatchDiagnosticFailure(
            "gradient-audit clipped norm is NaN or Inf",
            {
                **partial,
                "total_gradient_norm_by_parameter_group": total_group_norms,
                "clip_grad_norm_returned_pre_clip_norm": float(
                    returned_pre_clip.item()
                ),
            },
        )
    pre_clip = float(returned_pre_clip.item())
    clipping = clip_diagnostics(pre_clip, clip_norm)
    post_clip_observed = math.sqrt(
        sum(
            _parameter_gradient_norm(parameters, reference=losses.total_loss) ** 2
            for parameters in groups.values()
        )
    )
    row = {
        "epoch": int(epoch),
        "batch_index": int(batch_index),
        "batch_start": int(batch_start),
        "batch_size": int(labels.numel()),
        "batch_pair_id_sequence_sha256": _sequence_sha256(pair_ids),
        **statistics,
        "total_gradient_norm": total_group_norms["last_residual_stage"],
        "total_last_residual_stage_gradient_norm": total_group_norms[
            "last_residual_stage"
        ],
        "total_gradient_norm_by_parameter_group": total_group_norms,
        "relative_projection_gradient_norm": relative_projection_norm,
        "relative_ce_head_gradient_norm": relative_head_norm,
        **clipping,
        "post_clip_observed_gradient_norm": post_clip_observed,
        "clip_grad_norm_returned_pre_clip_norm": float(returned_pre_clip.item()),
        "classification_loss": float(losses.classification_loss.detach().item()),
        "relative_loss": float(losses.relative_loss.detach().item()),
        "total_loss": float(losses.total_loss.detach().item()),
        "train_accuracy": float(
            (output.fused_logits.argmax(dim=1) == labels).float().mean().item()
        ),
        "ce_mean_max_confidence": float(
            torch.softmax(output.fused_logits.detach(), dim=1).max(dim=1).values.mean().item()
        ),
        "original_100x_exception_suppressed": True,
        "final_unknown_used": False,
        "even_angle_test_used": False,
    }
    if not all(
        math.isfinite(float(value))
        for key, value in row.items()
        if isinstance(value, (float, int)) and key not in {"epoch", "batch_index", "batch_start", "batch_size"}
    ):
        raise _BatchDiagnosticFailure(
            "gradient-audit batch diagnostic is NaN or Inf", row
        )
    return row, returned_pre_clip


def train_q2_gradient_audit(
    model: FGMVCSSRE2EModel,
    *,
    unique_rows: Sequence[Mapping[str, Any]],
    unique_inputs: np.ndarray,
    prepared: Any,
    pair_id: str,
    legacy_config: Mapping[str, Any],
    device: torch.device,
    frozen_r2_arrays: Mapping[str, Mapping[str, np.ndarray]],
    sink: AtomicGradientAuditSink,
) -> dict[str, Any]:
    """Run exactly the first five legacy Q2 epochs with observation-only diagnostics."""

    if pair_id not in AUDIT_PAIRS or model.variant != AUDIT_METHOD:
        raise DataValidationError("gradient audit received an unauthorized unit")
    training = legacy_config["training"]
    batch_size = int(training["batch_size_pairs"])
    clip_norm = float(training["gradient_clip_norm"])
    model = model.to(device)
    optimizer = _build_optimizer(model, legacy_config)
    _set_determinism(AUDIT_SEED, bool(legacy_config["runtime"]["deterministic_algorithms"]))
    groups = _parameter_groups(model)
    epoch0_common_hash = _common_r2_state_sha256(model)
    ae_initial_hash = _state_sha256(model.cssr_core.state_dict())
    if epoch0_common_hash != EXPECTED_EPOCH0_R2_STATE_SHA256[pair_id]:
        raise DataValidationError("epoch-0 R2 common-state hash changed")
    if ae_initial_hash != EXPECTED_Q2_AE_INITIAL_SHA256:
        raise DataValidationError("Q2 AE initial-state hash changed")
    frozen_before = _frozen_prefix_sha256(model)
    epoch0_arrays = infer_e2e_model(
        model,
        prepared.inputs["known_calibration"],
        device=device,
        batch_size=batch_size,
    )
    if not np.array_equal(
        epoch0_arrays["fused_logits"],
        np.asarray(frozen_r2_arrays["known_calibration"]["global_logits"]),
    ):
        raise DataValidationError("audit epoch-0 logits differ from frozen R2")
    if not np.array_equal(
        epoch0_arrays["fused_features"],
        np.asarray(frozen_r2_arrays["known_calibration"]["fused_features"]),
    ):
        raise DataValidationError("audit epoch-0 features differ from frozen R2")
    epoch0_calibration = calibration_diagnostics(
        epoch0_arrays,
        prepared.labels["known_calibration"],
        model=model,
        epoch0_arrays=epoch0_arrays,
        ece_bins=int(legacy_config["diagnostics"]["ece_bins"]),
    )
    _write_json(sink.staging / "epoch0_calibration.json", epoch0_calibration)
    schedules: list[list[dict[str, Any]]] = []
    schedule_audits: list[dict[str, Any]] = []

    for epoch in range(1, AUDIT_EPOCHS + 1):
        sink.set_current_state(status="training", current_epoch=epoch, current_batch=0)
        schedule, schedule_audit = build_epoch_pair_schedule(
            unique_rows,
            pair_id=pair_id,
            angle_fold=ANGLE_FOLD,
            epoch=epoch,
            finetune_seed=AUDIT_SEED,
        )
        if schedule_audit["epoch_manifest_sha256"] != EXPECTED_SCHEDULE_SHA256[pair_id][epoch - 1]:
            raise DataValidationError(f"{pair_id} epoch {epoch} schedule hash changed")
        schedules.append(schedule)
        schedule_audits.append(schedule_audit)
        _write_csv(sink.staging / "pair_schedules" / f"epoch_{epoch:03d}.csv", schedule)
        _write_json(
            sink.staging / "pair_schedule_audits" / f"epoch_{epoch:03d}.json",
            schedule_audit,
        )
        epoch_inputs, epoch_labels = _materialize_schedule_inputs(
            schedule, unique_rows=unique_rows, unique_inputs=unique_inputs
        )
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
        before = snapshot_parameter_groups(groups)
        epoch_batch_rows: list[dict[str, Any]] = []
        totals = defaultdict(float)
        example_count = 0
        started = time.perf_counter()
        for batch_index, start in enumerate(range(0, len(epoch_labels), batch_size), start=1):
            sink.set_current_state(
                status="training",
                current_epoch=epoch,
                current_batch=batch_index,
                current_batch_start=start,
            )
            inputs = torch.from_numpy(epoch_inputs[start : start + batch_size]).to(device)
            labels = torch.from_numpy(epoch_labels[start : start + batch_size]).to(device)
            current_pair_ids = [
                str(item["pair_id"])
                for item in schedule[start : start + len(labels)]
            ]
            pending_batch: dict[str, Any] = {
                "status": "started",
                "epoch": epoch,
                "batch_index": batch_index,
                "batch_start": start,
                "batch_size": int(labels.numel()),
                "batch_pair_id_sequence_sha256": _sequence_sha256(current_pair_ids),
            }
            failed_operation = "optimizer.zero_grad"
            try:
                optimizer.zero_grad(set_to_none=True)
                failed_operation = "model.forward"
                output = model(inputs)
                failed_operation = "model.loss"
                losses = model.loss(output, labels)
                pending_batch.update(
                    {
                        "status": "loss_computed",
                        "classification_loss": float(
                            losses.classification_loss.detach().item()
                        ),
                        "relative_loss": float(losses.relative_loss.detach().item()),
                        "total_loss": float(losses.total_loss.detach().item()),
                    }
                )
                if not bool(torch.isfinite(losses.total_loss)):
                    raise DataValidationError(
                        "gradient-audit training loss is NaN or Inf"
                    )
                failed_operation = "gradient_diagnostics_and_backward"
                row, _ = _record_batch_diagnostics(
                    model=model,
                    losses=losses,
                    output=output,
                    labels=labels,
                    groups=groups,
                    epoch=epoch,
                    batch_index=batch_index,
                    batch_start=start,
                    pair_ids=current_pair_ids,
                    clip_norm=clip_norm,
                )
                pending_batch = {**row, "status": "gradients_computed"}
                failed_operation = "optimizer.step"
                optimizer.step()
                pending_batch["status"] = "optimizer_stepped"
            except BaseException as error:
                if isinstance(error, _BatchDiagnosticFailure):
                    pending_batch.update(error.diagnostic)
                sink.save_failure(
                    error,
                    failed_operation=failed_operation,
                    pending_batch=pending_batch,
                )
                raise
            if not _finite_parameters(model):
                error = DataValidationError(
                    "gradient-audit parameter is NaN or Inf"
                )
                sink.save_failure(
                    error,
                    failed_operation="parameter_finiteness_check",
                    pending_batch=pending_batch,
                )
                raise error
            sink.append_batch(row)
            epoch_batch_rows.append(row)
            count = int(labels.numel())
            example_count += count
            totals["classification_loss"] += row["classification_loss"] * count
            totals["relative_loss"] += row["relative_loss"] * count
            totals["total_loss"] += row["total_loss"] * count
            totals["train_accuracy"] += row["train_accuracy"] * count
        if example_count != len(epoch_labels):
            raise DataValidationError("gradient audit did not consume the complete schedule")
        if _frozen_prefix_sha256(model) != frozen_before:
            raise DataValidationError("frozen early R2 state changed during gradient audit")
        current_arrays = infer_e2e_model(
            model,
            prepared.inputs["known_calibration"],
            device=device,
            batch_size=batch_size,
        )
        calibration = calibration_diagnostics(
            current_arrays,
            prepared.labels["known_calibration"],
            model=model,
            epoch0_arrays=epoch0_arrays,
            ece_bins=int(legacy_config["diagnostics"]["ece_bins"]),
        )
        epoch_summary = {
            "epoch": epoch,
            "method": AUDIT_METHOD,
            "learning_rate_factor": factor,
            "learning_rates": learning_rates,
            "train_classification_loss": totals["classification_loss"] / example_count,
            "train_relative_loss": totals["relative_loss"] / example_count,
            "train_total_loss": totals["total_loss"] / example_count,
            "train_accuracy": totals["train_accuracy"] / example_count,
            **summarize_gradient_epoch(epoch_batch_rows),
            "parameter_relative_updates": relative_parameter_updates(before, groups),
            "calibration": {
                key: calibration[key]
                for key in (
                    "accuracy",
                    "nll",
                    "ece",
                    "mean_max_logit",
                    "mean_single_view_feature_norm",
                    "mean_fused_feature_norm",
                )
            },
            "pair_schedule_sha256": schedule_audit["epoch_manifest_sha256"],
            "elapsed_seconds": time.perf_counter() - started,
            "performance_gate_eligible": False,
            "surrogate_unknown_used": False,
            "final_unknown_used": False,
            "even_angle_test_used": False,
        }
        provisional = annotate_original_gate([*sink.epoch_rows, epoch_summary])
        epoch_summary.update(provisional["epochs"][-1])
        sink.append_epoch(epoch_summary)

    gate = annotate_original_gate(sink.epoch_rows)
    # Rewrite once so all epoch rows contain the canonical independently recomputed state.
    sink.epoch_rows = [dict(row) for row in gate["epochs"]]
    _atomic_write_bytes(
        sink.staging / "epoch_diagnostics.jsonl", _render_jsonl(sink.epoch_rows)
    )
    classification = classify_gradient_pathology(
        sink.epoch_rows,
        epoch0_calibration=epoch0_calibration,
    )
    return {
        "model": model.eval(),
        "epoch0_calibration": epoch0_calibration,
        "batch_rows": list(sink.batch_rows),
        "epoch_rows": list(sink.epoch_rows),
        "schedules": schedules,
        "schedule_audits": schedule_audits,
        "original_gate": {key: value for key, value in gate.items() if key != "epochs"},
        "pathology_classification": classification,
        "audit": {
            "status": "passed",
            "pair_id": pair_id,
            "method": AUDIT_METHOD,
            "epochs": AUDIT_EPOCHS,
            "batch_count": len(sink.batch_rows),
            "schedule_sha256": _sequence_sha256(
                item["epoch_manifest_sha256"] for item in schedule_audits
            ),
            "epoch_schedule_sha256": [
                item["epoch_manifest_sha256"] for item in schedule_audits
            ],
            "epoch0_common_r2_state_sha256": epoch0_common_hash,
            "ae_initial_state_sha256": ae_initial_hash,
            "frozen_prefix_unchanged": _frozen_prefix_sha256(model) == frozen_before,
            "all_parameters_finite": _finite_parameters(model),
            "original_100x_exception_suppressed_only": True,
            "performance_gate_eligible": False,
            "surrogate_unknown_used": False,
            "final_unknown_used": False,
            "even_angle_test_used": False,
        },
    }


def _unit_destination(root: Path, pair_id: str) -> Path:
    return (
        root
        / "gradient_pathology_audit"
        / pair_id
        / "fold_0"
        / f"seed_{AUDIT_SEED}"
        / AUDIT_METHOD
    )


def run_gradient_audit_unit(
    config_path: str | Path,
    bundle_root: str | Path,
    r2_results_root: str | Path,
    phase_root: str | Path,
    *,
    pair_id: str,
    device_request: str = "auto",
) -> dict[str, Any]:
    config = load_gradient_pathology_config(config_path)
    if pair_id not in AUDIT_PAIRS:
        raise DataValidationError("pair is outside the frozen gradient-audit plan")
    project_root = Path(config["_config_path"]).parents[3]
    legacy_config = load_legacy_q2_config(project_root)
    root = Path(phase_root).resolve()
    destination = _unit_destination(root, pair_id)
    staging = destination.parent / f".{AUDIT_METHOD}.staging"
    if destination.exists() or staging.exists():
        raise DataValidationError("gradient-audit output already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sink = AtomicGradientAuditSink(staging)
    started = time.perf_counter()
    try:
        source_hashes = audit_task_source_hashes(project_root)
        device = _resolve_device(device_request)
        runtime_contract = _configure_runtime(legacy_config, device)
        _write_json(
            staging / "unit_contract.json",
            {
                "experiment_id": EXPERIMENT_ID,
                "phase": "gradient_pathology_audit",
                "pair_id": pair_id,
                "method": AUDIT_METHOD,
                "angle_fold": ANGLE_FOLD,
                "model_seed": MODEL_SEED,
                "audit_seed": AUDIT_SEED,
                "epochs": AUDIT_EPOCHS,
                "config_sha256": config["_config_sha256"],
                "legacy_config_sha256": legacy_config["_config_sha256"],
                "source_hashes": source_hashes,
                "runtime_contract": runtime_contract,
                "performance_gate_eligible": False,
                "surrogate_unknown_used": False,
                "final_unknown_used": False,
                "even_angle_test_used": False,
            },
        )
        _atomic_write_bytes(
            staging / "resolved_config.yaml",
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False).encode("utf-8"),
        )
        prior_config = _load_prior_config(project_root, legacy_config)
        bundle = _load_bundle(bundle_root, legacy_config)
        prepared = _prepare_frozen_split(
            bundle, prior_config, legacy_config, pair_id
        )
        model, frozen_r2_arrays, r2_audit = load_and_audit_frozen_r2(
            project_root=project_root,
            r2_results_root=r2_results_root,
            pair_id=pair_id,
            config=legacy_config,
            prepared=prepared,
            prior_config=prior_config,
            device=device,
        )
        if r2_audit["checkpoint_sha256"] != EXPECTED_R2_CHECKPOINT_SHA256[pair_id]:
            raise DataValidationError("gradient-audit R2 checkpoint hash changed")
        unique_rows = build_unique_base_sample_manifest(prepared, bundle)
        unique_inputs = _normalized_unique_base_inputs(bundle, prepared, unique_rows)
        wrapper = FGMVCSSRE2EModel(
            model,
            AUDIT_METHOD,
            autoencoder_seed=AUDIT_SEED,
        )
        _atomic_write_bytes(staging / "source_pair_manifest.csv", prepared.pair_manifest_bytes)
        _write_csv(staging / "unique_base_sample_manifest.csv", unique_rows)
        _write_json(staging / "r2_reference_audit.json", r2_audit)
        result = train_q2_gradient_audit(
            wrapper,
            unique_rows=unique_rows,
            unique_inputs=unique_inputs,
            prepared=prepared,
            pair_id=pair_id,
            legacy_config=legacy_config,
            device=device,
            frozen_r2_arrays=frozen_r2_arrays,
            sink=sink,
        )
        if audit_task_source_hashes(project_root) != source_hashes:
            raise DataValidationError("gradient-audit source changed during execution")
        checkpoint = {
            "experiment_id": EXPERIMENT_ID,
            "phase": "gradient_pathology_audit",
            "pair_id": pair_id,
            "method": AUDIT_METHOD,
            "checkpoint_epoch": AUDIT_EPOCHS,
            "diagnostic_only": True,
            "model_state_dict": {
                name: value.detach().cpu().clone()
                for name, value in result["model"].state_dict().items()
            },
            "config_sha256": config["_config_sha256"],
            "legacy_config_sha256": LEGACY_CONFIG_SHA256,
            "schedule_sha256": result["audit"]["schedule_sha256"],
            "final_unknown_used": False,
            "even_angle_test_used": False,
        }
        checkpoint_buffer = io.BytesIO()
        torch.save(checkpoint, checkpoint_buffer)
        _atomic_write_bytes(staging / "checkpoint.pt", checkpoint_buffer.getvalue())
        _write_json(staging / "training_audit.json", result["audit"])
        _write_json(staging / "original_100x_gate.json", result["original_gate"])
        _write_json(
            staging / "pathology_classification.json",
            result["pathology_classification"],
        )
        environment = _git_environment(project_root, device)
        environment["runtime_contract"] = runtime_contract
        environment["task_source_hashes"] = source_hashes
        _write_json(staging / "environment.json", environment)
        summary = {
            "status": "complete",
            "experiment_id": EXPERIMENT_ID,
            "phase": "gradient_pathology_audit",
            "pair_id": pair_id,
            "method": AUDIT_METHOD,
            "epochs": AUDIT_EPOCHS,
            "diagnosis": result["pathology_classification"],
            "original_gate": result["original_gate"],
            "wall_time_seconds": time.perf_counter() - started,
            "diagnostic_only": True,
            "performance_gate_eligible": False,
            "final_unknown_used": False,
            "even_angle_test_used": False,
        }
        _write_json(staging / "unit_summary.json", summary)
        sink.set_current_state(status="complete", completed_epoch_count=AUDIT_EPOCHS)
        _finalize_marker(staging, success=True, summary_name="unit_summary.json")
        staging.replace(destination)
        return {**summary, "destination": str(destination)}
    except BaseException as error:
        if staging.exists():
            sink.save_failure(error, wall_time_seconds=time.perf_counter() - started)
            _finalize_marker(staging, success=False, summary_name="failure_state.json")
            staging.replace(destination)
        raise


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _finite_float(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise DataValidationError(f"{name} is not numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise DataValidationError(f"{name} is not numeric") from error
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise DataValidationError(f"{name} is invalid")
    return result


def _close(left: float, right: float, *, scale: float = 1.0) -> bool:
    return math.isclose(
        float(left),
        float(right),
        rel_tol=1.0e-6,
        abs_tol=1.0e-12 * max(1.0, float(scale)),
    )


def _audit_batch_diagnostic(
    row: Mapping[str, Any],
    *,
    expected_epoch: int,
    expected_batch_index: int,
    expected_start: int,
    expected_size: int,
    expected_pair_hash: str,
    clip_norm: float,
) -> None:
    context = f"epoch {expected_epoch} batch {expected_batch_index}"
    if (
        int(row.get("epoch", -1)) != expected_epoch
        or int(row.get("batch_index", -1)) != expected_batch_index
        or int(row.get("batch_start", -1)) != expected_start
        or int(row.get("batch_size", -1)) != expected_size
        or row.get("batch_pair_id_sequence_sha256") != expected_pair_hash
    ):
        raise DataValidationError(f"gradient-audit batch identity changed: {context}")
    nonnegative_names = (
        "classification_gradient_norm",
        "relative_raw_gradient_norm",
        "relative_weighted_gradient_norm",
        "weighted_relative_to_classification_ratio",
        "total_gradient_norm",
        "total_last_residual_stage_gradient_norm",
        "relative_projection_gradient_norm",
        "relative_ce_head_gradient_norm",
        "pre_clip_total_gradient_norm",
        "post_clip_estimated_gradient_norm",
        "post_clip_observed_gradient_norm",
        "clip_grad_norm_returned_pre_clip_norm",
    )
    values = {
        name: _finite_float(row.get(name), f"{context}.{name}", nonnegative=True)
        for name in nonnegative_names
    }
    for name in (
        "classification_weighted_relative_dot",
        "classification_loss",
        "relative_loss",
        "total_loss",
        "train_accuracy",
        "ce_mean_max_confidence",
        "gradient_clipping_scale",
    ):
        values[name] = _finite_float(row.get(name), f"{context}.{name}")
    groups = row.get("total_gradient_norm_by_parameter_group")
    if not isinstance(groups, Mapping) or set(groups) != set(PARAMETER_GROUP_NAMES):
        raise DataValidationError(f"gradient-audit parameter groups changed: {context}")
    for name, value in groups.items():
        _finite_float(value, f"{context}.{name} gradient", nonnegative=True)
    if not _close(
        values["relative_weighted_gradient_norm"],
        RELATIVE_WEIGHT * values["relative_raw_gradient_norm"],
    ):
        raise DataValidationError(f"weighted relative gradient changed: {context}")
    expected_ratio = values["relative_weighted_gradient_norm"] / max(
        values["classification_gradient_norm"], RATIO_DENOMINATOR_FLOOR
    )
    if not _close(values["weighted_relative_to_classification_ratio"], expected_ratio):
        raise DataValidationError(f"gradient ratio changed: {context}")
    product = (
        values["classification_gradient_norm"]
        * values["relative_weighted_gradient_norm"]
    )
    cosine = row.get("classification_weighted_relative_cosine")
    if product <= COSINE_PRODUCT_FLOOR:
        if cosine is not None:
            raise DataValidationError(f"undefined cosine changed: {context}")
    else:
        observed_cosine = _finite_float(cosine, f"{context}.cosine")
        expected_cosine = values["classification_weighted_relative_dot"] / product
        if not _close(observed_cosine, expected_cosine) or abs(observed_cosine) > 1.000001:
            raise DataValidationError(f"gradient cosine changed: {context}")
    if (
        values["relative_projection_gradient_norm"]
        > RELATIVE_DISCONNECTED_TOLERANCE
        or values["relative_ce_head_gradient_norm"]
        > RELATIVE_DISCONNECTED_TOLERANCE
    ):
        raise DataValidationError(f"L_rel disconnection changed: {context}")
    if not _close(
        values["total_gradient_norm"],
        values["total_last_residual_stage_gradient_norm"],
    ) or not _close(
        values["total_gradient_norm"],
        float(groups["last_residual_stage"]),
    ):
        raise DataValidationError(f"last-stage total gradient changed: {context}")
    expected_pre_clip = math.sqrt(
        sum(float(value) ** 2 for value in groups.values())
    )
    if not _close(
        values["pre_clip_total_gradient_norm"],
        expected_pre_clip,
        scale=expected_pre_clip,
    ) or not _close(
        values["pre_clip_total_gradient_norm"],
        values["clip_grad_norm_returned_pre_clip_norm"],
    ):
        raise DataValidationError(f"pre-clip gradient norm changed: {context}")
    expected_clipping = clip_diagnostics(
        values["pre_clip_total_gradient_norm"], clip_norm
    )
    for name in (
        "gradient_clipping_scale",
        "post_clip_estimated_gradient_norm",
    ):
        if not _close(values[name], float(expected_clipping[name])):
            raise DataValidationError(f"gradient clipping formula changed: {context}")
    if row.get("gradient_clipped") is not expected_clipping["gradient_clipped"]:
        raise DataValidationError(f"gradient clipping state changed: {context}")
    if not _close(
        values["post_clip_observed_gradient_norm"],
        values["post_clip_estimated_gradient_norm"],
        scale=values["post_clip_estimated_gradient_norm"],
    ):
        raise DataValidationError(f"post-clip gradient norm changed: {context}")
    if not _close(
        values["total_loss"],
        values["classification_loss"] + RELATIVE_WEIGHT * values["relative_loss"],
        scale=abs(values["total_loss"]),
    ):
        raise DataValidationError(f"Q2 loss composition changed: {context}")
    if not 0.0 <= values["train_accuracy"] <= 1.0 or not 0.0 <= values[
        "ce_mean_max_confidence"
    ] <= 1.0:
        raise DataValidationError(f"batch prediction diagnostic changed: {context}")
    for name in (
        "original_100x_exception_suppressed",
        "final_unknown_used",
        "even_angle_test_used",
    ):
        expected = name == "original_100x_exception_suppressed"
        if row.get(name) is not expected:
            raise DataValidationError(f"batch evidence boundary changed: {context}.{name}")


def _audit_saved_schedules_and_batches(
    root: Path,
    *,
    pair_id: str,
    batch_rows: Sequence[Mapping[str, Any]],
    clip_norm: float,
) -> None:
    expected_global_index = 0
    for epoch in range(1, AUDIT_EPOCHS + 1):
        schedule_path = root / "pair_schedules" / f"epoch_{epoch:03d}.csv"
        schedule_audit_path = (
            root / "pair_schedule_audits" / f"epoch_{epoch:03d}.json"
        )
        if file_sha256(schedule_path) != EXPECTED_SCHEDULE_SHA256[pair_id][epoch - 1]:
            raise DataValidationError(f"{pair_id} epoch {epoch} schedule changed")
        schedule = _read_csv(schedule_path)
        schedule_audit = _read_json(schedule_audit_path)
        if (
            len(schedule) != 720
            or schedule_audit.get("epoch_manifest_sha256")
            != EXPECTED_SCHEDULE_SHA256[pair_id][epoch - 1]
            or int(schedule_audit.get("pair_count", -1)) != 720
            or schedule_audit.get("all_constraints_passed") is not True
        ):
            raise DataValidationError(f"{pair_id} epoch {epoch} schedule audit changed")
        if any(
            str(row.get("identity_pair_id")) != pair_id
            or int(row.get("epoch", -1)) != epoch
            or int(row.get("view1_angle_deg", -1)) % 2 != 1
            or int(row.get("view2_angle_deg", -1)) % 2 != 1
            or int(row.get("view1_frame_id", -1))
            == int(row.get("view2_frame_id", -1))
            for row in schedule
        ):
            raise DataValidationError(f"{pair_id} epoch {epoch} schedule protocol changed")
        epoch_batches = batch_rows[expected_global_index : expected_global_index + 12]
        if len(epoch_batches) != 12:
            raise DataValidationError(f"{pair_id} epoch {epoch} batch population changed")
        for batch_index, start in enumerate(range(0, 720, 64), start=1):
            size = min(64, 720 - start)
            expected_pair_hash = _sequence_sha256(
                row["pair_id"] for row in schedule[start : start + size]
            )
            _audit_batch_diagnostic(
                epoch_batches[batch_index - 1],
                expected_epoch=epoch,
                expected_batch_index=batch_index,
                expected_start=start,
                expected_size=size,
                expected_pair_hash=expected_pair_hash,
                clip_norm=clip_norm,
            )
        expected_global_index += 12
    if expected_global_index != len(batch_rows):
        raise DataValidationError("gradient-audit batch population changed")


def audit_gradient_audit_unit(
    unit_root: str | Path,
    *,
    config: Mapping[str, Any],
    pair_id: str,
) -> dict[str, Any]:
    root = Path(unit_root).resolve()
    required_files = {
        "_SUCCESS.json",
        "artifact_hashes.json",
        "batch_diagnostics.jsonl",
        "checkpoint.pt",
        "environment.json",
        "epoch0_calibration.json",
        "epoch_diagnostics.jsonl",
        "latest_state.json",
        "original_100x_gate.json",
        "pathology_classification.json",
        "r2_reference_audit.json",
        "resolved_config.yaml",
        "source_pair_manifest.csv",
        "training_audit.json",
        "unique_base_sample_manifest.csv",
        "unit_contract.json",
        "unit_summary.json",
        *{
            f"pair_schedules/epoch_{epoch:03d}.csv"
            for epoch in range(1, AUDIT_EPOCHS + 1)
        },
        *{
            f"pair_schedule_audits/epoch_{epoch:03d}.json"
            for epoch in range(1, AUDIT_EPOCHS + 1)
        },
    }
    observed_files = {
        str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()
    }
    if not required_files <= observed_files:
        missing = sorted(required_files - observed_files)
        raise DataValidationError(f"gradient-audit unit lacks required artifacts: {missing}")
    if any(
        "final_test" in relative.lower() or "unknown_test" in relative.lower()
        for relative in observed_files
    ):
        raise DataValidationError("gradient-audit unit contains a prohibited test artifact")
    success = root / "_SUCCESS.json"
    if not success.is_file() or (root / "_FAILED.json").exists():
        raise DataValidationError("gradient-audit unit is not a successful complete unit")
    marker = _read_json(success)
    if (
        marker.get("status") != "complete"
        or marker.get("artifact_hashes_sha256")
        != file_sha256(root / "artifact_hashes.json")
        or marker.get("unit_summary_sha256")
        != file_sha256(root / "unit_summary.json")
    ):
        raise DataValidationError("gradient-audit success marker changed")
    if _read_json(root / "artifact_hashes.json") != _audit_artifact_hashes(root):
        raise DataValidationError("gradient-audit artifact hashes changed")
    resolved = yaml.safe_load((root / "resolved_config.yaml").read_text(encoding="utf-8"))
    if resolved != dict(config):
        raise DataValidationError("gradient-audit resolved config changed")
    project_root = Path(str(config["_config_path"])).parents[3]
    source_hashes = audit_task_source_hashes(project_root)
    legacy_config = load_legacy_q2_config(project_root)
    contract = _read_json(root / "unit_contract.json")
    if (
        contract.get("experiment_id") != EXPERIMENT_ID
        or contract.get("phase") != "gradient_pathology_audit"
        or contract.get("pair_id") != pair_id
        or contract.get("method") != AUDIT_METHOD
        or int(contract.get("angle_fold", -1)) != ANGLE_FOLD
        or int(contract.get("model_seed", -1)) != MODEL_SEED
        or int(contract.get("audit_seed", -1)) != AUDIT_SEED
        or int(contract.get("epochs", -1)) != AUDIT_EPOCHS
        or contract.get("config_sha256") != config.get("_config_sha256")
        or contract.get("legacy_config_sha256") != LEGACY_CONFIG_SHA256
        or contract.get("source_hashes") != source_hashes
        or contract.get("performance_gate_eligible") is not False
        or contract.get("surrogate_unknown_used") is not False
        or contract.get("final_unknown_used") is not False
        or contract.get("even_angle_test_used") is not False
    ):
        raise DataValidationError("gradient-audit unit contract changed")
    r2_audit = _read_json(root / "r2_reference_audit.json")
    if (
        r2_audit.get("status") != "passed"
        or r2_audit.get("pair_id") != pair_id
        or r2_audit.get("checkpoint_sha256")
        != EXPECTED_R2_CHECKPOINT_SHA256[pair_id]
        or r2_audit.get("pair_manifest_sha256")
        != file_sha256(root / "source_pair_manifest.csv")
        or r2_audit.get("pair_manifest_sha256")
        != EXPECTED_R2_PAIR_MANIFEST_SHA256[pair_id]
        or r2_audit.get("strict_load") is not True
        or r2_audit.get("old_outputs_exact") is not True
        or r2_audit.get("all_parameters_frozen") is not True
        or r2_audit.get("final_unknown_used") is not False
        or r2_audit.get("even_angle_test_used") is not False
    ):
        raise DataValidationError("gradient-audit R2 reference changed")
    unique_rows = _read_csv(root / "unique_base_sample_manifest.csv")
    role_counts = Counter(str(row.get("experiment_role")) for row in unique_rows)
    train_rows = [row for row in unique_rows if row.get("experiment_role") == "train_known"]
    if role_counts != Counter(
        {"train_known": 720, "known_calibration": 180, "surrogate_unknown": 72}
    ) or Counter(int(row["model_label"]) for row in train_rows) != Counter(
        {index: 144 for index in range(5)}
    ):
        raise DataValidationError("gradient-audit unique-base population changed")
    if any(int(row["angle_deg"]) % 2 != 1 for row in unique_rows):
        raise DataValidationError("even-angle base entered gradient-audit artifacts")
    role_ids = {
        role: {
            str(row["sample_id"])
            for row in unique_rows
            if row["experiment_role"] == role
        }
        for role in role_counts
    }
    if (
        role_ids["train_known"] & role_ids["known_calibration"]
        or role_ids["train_known"] & role_ids["surrogate_unknown"]
        or role_ids["known_calibration"] & role_ids["surrogate_unknown"]
    ):
        raise DataValidationError("gradient-audit unique-base roles overlap")
    batch_rows = _read_jsonl(root / "batch_diagnostics.jsonl")
    epoch_rows = _read_jsonl(root / "epoch_diagnostics.jsonl")
    if len(epoch_rows) != AUDIT_EPOCHS or len(batch_rows) != 60:
        raise DataValidationError("gradient-audit row population changed")
    _audit_saved_schedules_and_batches(
        root,
        pair_id=pair_id,
        batch_rows=batch_rows,
        clip_norm=float(legacy_config["training"]["gradient_clip_norm"]),
    )
    recomputed_epochs: list[dict[str, Any]] = []
    for epoch in range(1, AUDIT_EPOCHS + 1):
        selected = [row for row in batch_rows if int(row["epoch"]) == epoch]
        expected = summarize_gradient_epoch(selected)
        observed = epoch_rows[epoch - 1]
        if (
            int(observed.get("epoch", -1)) != epoch
            or observed.get("method") != AUDIT_METHOD
            or observed.get("pair_schedule_sha256")
            != EXPECTED_SCHEDULE_SHA256[pair_id][epoch - 1]
            or observed.get("performance_gate_eligible") is not False
            or observed.get("surrogate_unknown_used") is not False
            or observed.get("final_unknown_used") is not False
            or observed.get("even_angle_test_used") is not False
        ):
            raise DataValidationError(f"gradient epoch contract changed: {epoch}")
        for key, value in expected.items():
            if isinstance(value, float):
                if not math.isclose(float(observed[key]), value, rel_tol=0.0, abs_tol=1.0e-15):
                    raise DataValidationError(f"gradient epoch statistic changed: {epoch}.{key}")
            elif observed[key] != value:
                raise DataValidationError(f"gradient epoch statistic changed: {epoch}.{key}")
        updates = observed.get("parameter_relative_updates")
        if not isinstance(updates, Mapping) or set(updates) != set(PARAMETER_GROUP_NAMES):
            raise DataValidationError(f"gradient epoch updates changed: {epoch}")
        for name, value in updates.items():
            _finite_float(value, f"epoch {epoch}.{name} update", nonnegative=True)
        calibration = observed.get("calibration")
        required_calibration = (
            "accuracy",
            "nll",
            "ece",
            "mean_max_logit",
            "mean_single_view_feature_norm",
            "mean_fused_feature_norm",
        )
        if not isinstance(calibration, Mapping):
            raise DataValidationError(f"gradient epoch calibration changed: {epoch}")
        for name in required_calibration:
            _finite_float(calibration.get(name), f"epoch {epoch}.calibration.{name}")
        recomputed_epochs.append(dict(observed))
    epoch0_calibration = _read_json(root / "epoch0_calibration.json")
    for name in (
        "accuracy",
        "nll",
        "ece",
        "mean_max_logit",
        "mean_single_view_feature_norm",
        "mean_fused_feature_norm",
    ):
        _finite_float(epoch0_calibration.get(name), f"epoch0_calibration.{name}")
    gate = annotate_original_gate(recomputed_epochs)
    if _read_json(root / "original_100x_gate.json") != {
        key: value for key, value in gate.items() if key != "epochs"
    }:
        raise DataValidationError("original 100x gate state changed")
    classification = classify_gradient_pathology(
        gate["epochs"],
        epoch0_calibration=epoch0_calibration,
    )
    if _read_json(root / "pathology_classification.json") != classification:
        raise DataValidationError("gradient pathology label changed")
    summary = _read_json(root / "unit_summary.json")
    if (
        summary.get("status") != "complete"
        or summary.get("experiment_id") != EXPERIMENT_ID
        or summary.get("phase") != "gradient_pathology_audit"
        or summary.get("pair_id") != pair_id
        or summary.get("method") != AUDIT_METHOD
        or int(summary.get("epochs", -1)) != AUDIT_EPOCHS
        or summary.get("diagnosis") != classification
        or summary.get("original_gate")
        != {key: value for key, value in gate.items() if key != "epochs"}
        or summary.get("diagnostic_only") is not True
        or summary.get("performance_gate_eligible") is not False
        or summary.get("final_unknown_used") is not False
        or summary.get("even_angle_test_used") is not False
    ):
        raise DataValidationError("gradient-audit unit summary changed")
    training_audit = _read_json(root / "training_audit.json")
    expected_schedule_hashes = list(EXPECTED_SCHEDULE_SHA256[pair_id])
    if (
        training_audit.get("status") != "passed"
        or training_audit.get("pair_id") != pair_id
        or training_audit.get("method") != AUDIT_METHOD
        or int(training_audit.get("epochs", -1)) != AUDIT_EPOCHS
        or int(training_audit.get("batch_count", -1)) != 60
        or training_audit.get("epoch_schedule_sha256") != expected_schedule_hashes
        or training_audit.get("schedule_sha256")
        != _sequence_sha256(expected_schedule_hashes)
        or training_audit.get("epoch0_common_r2_state_sha256")
        != EXPECTED_EPOCH0_R2_STATE_SHA256[pair_id]
        or training_audit.get("ae_initial_state_sha256")
        != EXPECTED_Q2_AE_INITIAL_SHA256
        or training_audit.get("frozen_prefix_unchanged") is not True
        or training_audit.get("all_parameters_finite") is not True
        or training_audit.get("original_100x_exception_suppressed_only") is not True
        or training_audit.get("performance_gate_eligible") is not False
        or training_audit.get("surrogate_unknown_used") is not False
        or training_audit.get("final_unknown_used") is not False
        or training_audit.get("even_angle_test_used") is not False
    ):
        raise DataValidationError("gradient-audit training audit changed")
    checkpoint = torch.load(root / "checkpoint.pt", map_location="cpu", weights_only=False)
    if (
        checkpoint.get("experiment_id") != EXPERIMENT_ID
        or checkpoint.get("phase") != "gradient_pathology_audit"
        or checkpoint.get("pair_id") != pair_id
        or checkpoint.get("method") != AUDIT_METHOD
        or int(checkpoint.get("checkpoint_epoch", -1)) != AUDIT_EPOCHS
        or checkpoint.get("diagnostic_only") is not True
        or checkpoint.get("config_sha256") != config.get("_config_sha256")
        or checkpoint.get("legacy_config_sha256") != LEGACY_CONFIG_SHA256
        or checkpoint.get("schedule_sha256")
        != _sequence_sha256(expected_schedule_hashes)
        or checkpoint.get("final_unknown_used") is not False
        or checkpoint.get("even_angle_test_used") is not False
    ):
        raise DataValidationError("gradient-audit checkpoint contract changed")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping) or not state or any(
        not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all())
        for value in state.values()
    ):
        raise DataValidationError("gradient-audit checkpoint state is invalid")
    r2_state = {
        str(name).removeprefix("r2_model."): value
        for name, value in state.items()
        if str(name).startswith("r2_model.")
    }
    if not r2_state:
        raise DataValidationError("gradient-audit checkpoint lacks the R2 state")
    try:
        with torch.random.fork_rng(devices=[]):
            replay_model = FGMVCSSRE2EModel.from_r2_state_dict(
                r2_state,
                AUDIT_METHOD,
                known_class_count=5,
                autoencoder_seed=AUDIT_SEED,
            )
        incompatible = replay_model.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError("strict load reported incompatible keys")
        _parameter_groups(replay_model)
    except Exception as error:
        raise DataValidationError(
            "gradient-audit checkpoint is not a strict Q2 model state"
        ) from error
    environment = _read_json(root / "environment.json")
    if environment.get("task_source_hashes") != source_hashes:
        raise DataValidationError("gradient-audit environment source hashes changed")
    latest = _read_json(root / "latest_state.json")
    if (
        latest.get("status") != "complete"
        or int(latest.get("completed_epoch_count", -1)) != AUDIT_EPOCHS
        or int(latest.get("completed_batch_count", -1)) != 60
    ):
        raise DataValidationError("gradient-audit latest state changed")
    return {
        "status": "passed",
        "pair_id": pair_id,
        "method": AUDIT_METHOD,
        "diagnosis": classification,
        "original_gate": summary["original_gate"],
        "batch_count": len(batch_rows),
        "epoch_count": len(epoch_rows),
        "artifact_count": len(_read_json(root / "artifact_hashes.json")),
        "batch_to_epoch_recomputation": "exact",
        "final_unknown_used": False,
        "even_angle_test_used": False,
    }


def aggregate_gradient_audit_phase(
    phase_root: str | Path,
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(phase_root).resolve()
    if (root / "_PHASE_SUCCESS.json").exists():
        raise DataValidationError("gradient-audit phase is already aggregated")
    audits = [
        audit_gradient_audit_unit(
            _unit_destination(root, pair_id), config=config, pair_id=pair_id
        )
        for pair_id in AUDIT_PAIRS
    ]
    by_pair = {str(row["pair_id"]): row for row in audits}
    if set(by_pair) != set(AUDIT_PAIRS):
        raise DataValidationError("gradient-audit phase population changed")
    summary = {
        "status": "complete",
        "experiment_id": EXPERIMENT_ID,
        "phase": "gradient_pathology_audit",
        "unit_count": len(audits),
        "units": audits,
        "n4_primary_diagnosis": by_pair["N4"]["diagnosis"],
        "n1_control_diagnosis": by_pair["N1"]["diagnosis"],
        "config_sha256": config["_config_sha256"],
        "stage_b_allowed": True,
        "performance_gate_eligible": False,
        "final_unknown_test_authorized": False,
    }
    _write_json(
        root / "gradient_pathology_audit_plan.json",
        {"units": build_gradient_audit_plan(config)},
    )
    _write_json(root / "gradient_pathology_audit_summary.json", summary)
    _write_json(root / "artifact_hashes.json", _audit_artifact_hashes(root))
    _write_json(
        root / "_PHASE_SUCCESS.json",
        {
            "status": "complete",
            "phase": "gradient_pathology_audit",
            "config_sha256": config["_config_sha256"],
            "summary_sha256": file_sha256(
                root / "gradient_pathology_audit_summary.json"
            ),
            "artifact_hashes_sha256": file_sha256(root / "artifact_hashes.json"),
            "stage_b_allowed": True,
            "final_unknown_test_authorized": False,
        },
    )
    return summary


def audit_gradient_audit_phase(
    phase_root: str | Path,
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-audit the sealed two-unit phase before stage B consumes it."""

    root = Path(phase_root).resolve()
    marker_path = root / "_PHASE_SUCCESS.json"
    summary_path = root / "gradient_pathology_audit_summary.json"
    hashes_path = root / "artifact_hashes.json"
    if not marker_path.is_file() or not summary_path.is_file() or not hashes_path.is_file():
        raise DataValidationError("gradient-audit phase seal is incomplete")
    marker = _read_json(marker_path)
    if (
        marker.get("status") != "complete"
        or marker.get("phase") != "gradient_pathology_audit"
        or marker.get("config_sha256") != config.get("_config_sha256")
        or marker.get("summary_sha256") != file_sha256(summary_path)
        or marker.get("artifact_hashes_sha256") != file_sha256(hashes_path)
        or marker.get("stage_b_allowed") is not True
        or marker.get("final_unknown_test_authorized") is not False
    ):
        raise DataValidationError("gradient-audit phase success marker changed")
    if _read_json(hashes_path) != _audit_artifact_hashes(root):
        raise DataValidationError("gradient-audit phase artifact hashes changed")
    expected_plan = {"units": build_gradient_audit_plan(config)}
    if _read_json(root / "gradient_pathology_audit_plan.json") != expected_plan:
        raise DataValidationError("gradient-audit phase plan changed")
    audits = [
        audit_gradient_audit_unit(
            _unit_destination(root, pair_id), config=config, pair_id=pair_id
        )
        for pair_id in AUDIT_PAIRS
    ]
    by_pair = {str(row["pair_id"]): row for row in audits}
    expected_summary = {
        "status": "complete",
        "experiment_id": EXPERIMENT_ID,
        "phase": "gradient_pathology_audit",
        "unit_count": len(audits),
        "units": audits,
        "n4_primary_diagnosis": by_pair["N4"]["diagnosis"],
        "n1_control_diagnosis": by_pair["N1"]["diagnosis"],
        "config_sha256": config["_config_sha256"],
        "stage_b_allowed": True,
        "performance_gate_eligible": False,
        "final_unknown_test_authorized": False,
    }
    if _read_json(summary_path) != expected_summary:
        raise DataValidationError("gradient-audit phase summary changed")
    return expected_summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preregistered five-epoch legacy-Q2 gradient pathology audit"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    load = commands.add_parser("load-config")
    load.add_argument("--config", default=CONFIG_RELATIVE_PATH)
    plan = commands.add_parser("plan")
    plan.add_argument("--config", default=CONFIG_RELATIVE_PATH)
    run = commands.add_parser("run-unit")
    run.add_argument("--config", default=CONFIG_RELATIVE_PATH)
    run.add_argument("--bundle-root", required=True)
    run.add_argument("--r2-results-root", required=True)
    run.add_argument("--phase-root", required=True)
    run.add_argument("--pair-id", choices=AUDIT_PAIRS, required=True)
    run.add_argument("--device", default="auto")
    audit = commands.add_parser("audit-unit")
    audit.add_argument("--config", default=CONFIG_RELATIVE_PATH)
    audit.add_argument("--unit-root", required=True)
    audit.add_argument("--pair-id", choices=AUDIT_PAIRS, required=True)
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--config", default=CONFIG_RELATIVE_PATH)
    aggregate.add_argument("--phase-root", required=True)
    audit_phase = commands.add_parser("audit-phase")
    audit_phase.add_argument("--config", default=CONFIG_RELATIVE_PATH)
    audit_phase.add_argument("--phase-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = load_gradient_pathology_config(arguments.config)
    if arguments.command == "load-config":
        result = {
            "status": "passed",
            "experiment_id": EXPERIMENT_ID,
            "config_sha256": config["_config_sha256"],
            "final_unknown_test_authorized": False,
        }
    elif arguments.command == "plan":
        result = {
            "status": "planned",
            "units": build_gradient_audit_plan(config),
            "final_unknown_test_authorized": False,
        }
    elif arguments.command == "run-unit":
        result = run_gradient_audit_unit(
            arguments.config,
            arguments.bundle_root,
            arguments.r2_results_root,
            arguments.phase_root,
            pair_id=arguments.pair_id,
            device_request=arguments.device,
        )
    elif arguments.command == "audit-unit":
        result = audit_gradient_audit_unit(
            arguments.unit_root, config=config, pair_id=arguments.pair_id
        )
    elif arguments.command == "aggregate":
        result = aggregate_gradient_audit_phase(
            arguments.phase_root, config=config
        )
    elif arguments.command == "audit-phase":
        result = audit_gradient_audit_phase(arguments.phase_root, config=config)
    else:
        raise AssertionError("unreachable command")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
