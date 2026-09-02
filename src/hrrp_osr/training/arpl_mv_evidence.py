from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader

from hrrp_osr.data.errors import DataConfigError, DataValidationError
from hrrp_osr.data.manifest import file_sha256
from hrrp_osr.data.processed import ProcessedBundle, load_processed_bundle
from hrrp_osr.evaluation.metrics import (
    binary_auroc,
    evaluate_open_set,
    summarize_metric_repeats,
)
from hrrp_osr.models.arpl import TwoViewARPLClassifier, TwoViewCEClassifier
from hrrp_osr.training.arpl_pilot import (
    METHODS as BASELINE_METHODS,
    SOURCE_KNOWN_ORDER,
    NumericalInstabilityError,
    PairTensorDataset,
    PreparedSurrogateSplit,
    _artifact_hashes,
    _environment,
    _is_finite_model,
    _render_csv,
    _resolve_device,
    _set_determinism,
    _state_sha256,
    _write_csv,
    _write_json,
    prepare_surrogate_split,
)


EXPERIMENT_ID = "arpl_mv_evidence_surrogate_v1"
METHODS = ("CE_MLS", "CE_VIEW_AUX", "ARPL_LITE", "ARPL_VIEW_AUX")
VIEW_AUX_METHODS = ("CE_VIEW_AUX", "ARPL_VIEW_AUX")
RULES = (
    "F0_FUSED",
    "F1_WORST_VIEW",
    "F2_EVIDENCE_UNION",
    "F3_DISAGREEMENT_AWARE",
)
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


@dataclass(frozen=True)
class KnownCalibrationECDF:
    sorted_reference: np.ndarray

    @classmethod
    def fit(cls, scores: np.ndarray, *, role: str) -> "KnownCalibrationECDF":
        if role != "known_calibration":
            raise DataValidationError("ECDF may only be fitted on known_calibration")
        values = np.asarray(scores, dtype=np.float64)
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise DataValidationError("ECDF reference must be finite and one-dimensional")
        return cls(np.sort(values, kind="mergesort"))

    def transform(self, values: np.ndarray) -> np.ndarray:
        query = np.asarray(values, dtype=np.float64)
        if not np.isfinite(query).all():
            raise DataValidationError("ECDF query contains NaN or Inf")
        return np.searchsorted(self.sorted_reference, query, side="right") / float(
            self.sorted_reference.size
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "fit_role": "known_calibration",
            "sample_count": int(self.sorted_reference.size),
            "sorted_reference": self.sorted_reference.tolist(),
        }


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DataConfigError(f"{name} must be a mapping")
    return value


def load_mv_evidence_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = dict(_require_mapping(yaml.safe_load(handle), "config"))
    errors: list[str] = []
    if (
        config.get("schema_version") != 1
        or config.get("stage") != "P1_surrogate_mv_evidence"
        or config.get("experiment_id") != EXPERIMENT_ID
    ):
        errors.append("experiment identity changed")
    scope = _require_mapping(config.get("evidence_scope"), "evidence_scope")
    if scope.get("source_known_odd_angle_only") is not True or any(
        scope.get(key) is not False
        for key in (
            "final_unknown_classes_used",
            "even_angle_test_used",
            "amdr_used",
            "confusing_samples_used",
            "other_open_set_methods_used",
        )
    ):
        errors.append("evidence scope changed")
    reference = _require_mapping(config.get("official_reference"), "official_reference")
    if (
        reference.get("commit") != "3ede8b38e1cfb9d70e106cc19d563453110c36ab"
        or reference.get("dist_sha256")
        != "a05fc01c9051d8cb8d87cc7183e0a3d9fd1a11ca9de38d58a4870cb70ad4dc62"
        or reference.get("arploss_sha256")
        != "6dec41f0265b6665e8c66a27f506f176a0a7b0b2e4426760c09c203ab0c327ec"
    ):
        errors.append("official ARPL reference changed")
    classes = _require_mapping(config.get("classes"), "classes")
    if list(classes.get("source_known_order", [])) != list(SOURCE_KNOWN_ORDER):
        errors.append("source known order changed")
    expected_development = (
        ("S0", 1, [2, 3, 4, 5, 6], [0, 1]),
        ("S1", 2, [0, 1, 4, 5, 6], [2, 3]),
        ("S2", 3, [0, 1, 2, 3, 6], [4, 5]),
    )
    expected_confirmation = (
        ("C0", 0, [1, 2, 3, 4, 5], [0, 6]),
        ("C1", 4, [0, 2, 3, 4, 6], [1, 5]),
        ("C2", 0, [0, 1, 3, 5, 6], [2, 4]),
        ("C3", 4, [0, 1, 2, 4, 5], [3, 6]),
    )
    for key, expected in (
        ("development_splits", expected_development),
        ("confirmation_splits", expected_confirmation),
    ):
        rows = classes.get(key)
        if not isinstance(rows, Sequence) or len(rows) != len(expected):
            errors.append(f"{key} count changed")
            continue
        observed = tuple(
            (
                row.get("split_id"),
                row.get("angle_fold"),
                list(row.get("train_known_indices", [])),
                list(row.get("surrogate_unknown_indices", [])),
            )
            for row in rows
        )
        if observed != expected:
            errors.append(f"{key} changed")
    model = _require_mapping(config.get("model"), "model")
    arpl = _require_mapping(model.get("arpl"), "model.arpl")
    if (
        list(model.get("methods", [])) != list(METHODS)
        or float(model.get("lambda_view", -1)) != 0.5
        or model.get("view_specific_encoder") is not False
        or model.get("position_encoding") is not False
        or model.get("permutation_invariant") is not True
        or arpl.get("margin_scope") != "fused_only"
        or float(arpl.get("temperature", -1)) != 1.0
        or float(arpl.get("weight_pl", -1)) != 0.1
        or float(arpl.get("margin", -1)) != 1.0
    ):
        errors.append("model contract changed")
    training = _require_mapping(config.get("training"), "training")
    if (
        list(training.get("development_seeds", [])) != [20260830]
        or list(training.get("confirmation_seeds", []))
        != [20260830, 20260831, 20260832]
        or float(training.get("learning_rate", -1)) != 1e-3
        or int(training.get("batch_size", 0)) != 64
        or dict(training.get("max_epochs", {})) != {"smoke": 3, "full": 100}
        or int(training.get("early_stopping_patience", 0)) != 15
    ):
        errors.append("training contract changed")
    selection = _require_mapping(
        config.get("development_selection"), "development_selection"
    )
    if (
        list(selection.get("eligible_methods", [])) != list(VIEW_AUX_METHODS)
        or list(selection.get("candidate_rules", [])) != list(RULES[1:])
        or int(selection.get("minimum_positive_auroc_units", 0)) != 4
        or float(selection.get("minimum_mean_auroc_delta", -1)) != 0.02
        or float(selection.get("minimum_mean_oscr_delta", -1)) != 0.0
    ):
        errors.append("development gate changed")
    if errors:
        raise DataConfigError("Invalid MV evidence config:\n- " + "\n- ".join(errors))
    config["_config_path"] = str(config_path)
    config["_config_sha256"] = file_sha256(config_path)
    return config


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def jensen_shannon_divergence(logits1: np.ndarray, logits2: np.ndarray) -> np.ndarray:
    p = _softmax(np.asarray(logits1, dtype=np.float64))
    q = _softmax(np.asarray(logits2, dtype=np.float64))
    midpoint = 0.5 * (p + q)
    epsilon = np.finfo(np.float64).tiny
    return 0.5 * np.sum(p * np.log((p + epsilon) / (midpoint + epsilon)), axis=1) + 0.5 * np.sum(
        q * np.log((q + epsilon) / (midpoint + epsilon)), axis=1
    )


def raw_view_evidence(
    per_view_features: np.ndarray,
    per_view_logits: np.ndarray,
    fused_logits: np.ndarray,
) -> dict[str, np.ndarray]:
    features = np.asarray(per_view_features, dtype=np.float64)
    logits = np.asarray(per_view_logits, dtype=np.float64)
    fused = np.asarray(fused_logits, dtype=np.float64)
    if features.ndim != 3 or features.shape[1] != 2:
        raise DataValidationError("per-view features must have shape [n,2,d]")
    if logits.ndim != 3 or logits.shape[:2] != features.shape[:2]:
        raise DataValidationError("per-view logits do not align")
    if fused.shape != (features.shape[0], logits.shape[2]):
        raise DataValidationError("fused logits do not align")
    u_views = -np.max(logits, axis=2)
    norms = np.linalg.norm(features, axis=2)
    dot = np.sum(features[:, 0] * features[:, 1], axis=1)
    denominator = norms[:, 0] * norms[:, 1]
    cosine = np.divide(dot, denominator, out=np.zeros_like(dot), where=denominator > 0)
    return {
        "u_f": -np.max(fused, axis=1),
        "u_1": u_views[:, 0],
        "u_2": u_views[:, 1],
        "u_worst": np.max(u_views, axis=1),
        "u_mean": np.mean(u_views, axis=1),
        "u_gap": np.abs(u_views[:, 0] - u_views[:, 1]),
        "view_prediction_disagreement": (
            np.argmax(logits[:, 0], axis=1) != np.argmax(logits[:, 1], axis=1)
        ).astype(np.float64),
        "js": jensen_shannon_divergence(logits[:, 0], logits[:, 1]),
        "feature_cosine_distance": 1.0 - cosine,
        "feature_l2_mean": np.mean((features[:, 0] - features[:, 1]) ** 2, axis=1),
        "view1_norm": norms[:, 0],
        "view2_norm": norms[:, 1],
        "max_view_norm": np.max(norms, axis=1),
        "mean_view_norm": np.mean(norms, axis=1),
        "fused_norm": np.linalg.norm(np.mean(features, axis=1), axis=1),
    }


def fit_and_apply_evidence_ecdfs(
    known: Mapping[str, np.ndarray],
    unknown: Mapping[str, np.ndarray],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    fused_ecdf = KnownCalibrationECDF.fit(known["u_f"], role="known_calibration")
    view_ecdf = KnownCalibrationECDF.fit(
        np.concatenate([known["u_1"], known["u_2"]]), role="known_calibration"
    )
    js_ecdf = KnownCalibrationECDF.fit(known["js"], role="known_calibration")
    transformed: dict[str, dict[str, np.ndarray]] = {}
    for role, values in (("known_calibration", known), ("surrogate_unknown", unknown)):
        qf = fused_ecdf.transform(values["u_f"])
        q1 = view_ecdf.transform(values["u_1"])
        q2 = view_ecdf.transform(values["u_2"])
        qjs = js_ecdf.transform(values["js"])
        transformed[role] = {
            "q_f": qf,
            "q_1": q1,
            "q_2": q2,
            "q_js": qjs,
            "F0_FUSED": qf,
            "F1_WORST_VIEW": np.maximum(q1, q2),
            "F2_EVIDENCE_UNION": np.maximum.reduce([qf, q1, q2]),
            "F3_DISAGREEMENT_AWARE": np.maximum.reduce([qf, q1, q2, qjs]),
            "mean_q1_q2": 0.5 * (q1 + q2),
        }
    parameters = {
        "fused": fused_ecdf.to_json(),
        "per_view_pooled": view_ecdf.to_json(),
        "js": js_ecdf.to_json(),
        "surrogate_unknown_used_for_fit": False,
    }
    return transformed, parameters


def select_development_fusion(
    metric_rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    selection = config["development_selection"]
    eligible_methods = set(selection["eligible_methods"])
    candidates: list[dict[str, Any]] = []
    for complexity, rule in enumerate(selection["candidate_rules"], start=1):
        deltas = []
        for row in metric_rows:
            if row["method"] not in eligible_methods or row["rule"] != rule:
                continue
            baseline = next(
                item
                for item in metric_rows
                if item["split_id"] == row["split_id"]
                and item["seed"] == row["seed"]
                and item["method"] == row["method"]
                and item["rule"] == "F0_FUSED"
            )
            deltas.append(
                {
                    "split_id": row["split_id"],
                    "method": row["method"],
                    "auroc_delta": float(row["auroc"] - baseline["auroc"]),
                    "oscr_delta": float(row["oscr"] - baseline["oscr"]),
                }
            )
        if len(deltas) != int(selection["unit_count"]):
            raise DataValidationError("development selection does not contain six units")
        deltas.sort(key=lambda item: (item["method"], item["split_id"]))
        summary = {
            "rule": rule,
            "positive_auroc_units": sum(item["auroc_delta"] > 0 for item in deltas),
            "mean_auroc_delta": float(np.mean([item["auroc_delta"] for item in deltas])),
            "mean_oscr_delta": float(np.mean([item["oscr_delta"] for item in deltas])),
            "complexity_rank": complexity,
            "paired_deltas": deltas,
        }
        summary["eligible"] = bool(
            summary["positive_auroc_units"]
            >= int(selection["minimum_positive_auroc_units"])
            and summary["mean_auroc_delta"]
            >= float(selection["minimum_mean_auroc_delta"])
            and summary["mean_oscr_delta"]
            >= float(selection["minimum_mean_oscr_delta"])
        )
        candidates.append(summary)
    eligible = [item for item in candidates if item["eligible"]]
    selected = (
        sorted(
            eligible,
            key=lambda item: (
                -item["mean_auroc_delta"],
                -item["mean_oscr_delta"],
                item["complexity_rank"],
            ),
        )[0]["rule"]
        if eligible
        else None
    )
    return {
        "gate_passed": selected is not None,
        "selected_rule": selected,
        "candidates": candidates,
        "confirmation_allowed": selected is not None,
    }


def require_confirmation_gate(selection: Mapping[str, Any]) -> str:
    if selection.get("gate_passed") is not True or not selection.get("selected_rule"):
        raise DataValidationError("development gate failed; confirmation is forbidden")
    return str(selection["selected_rule"])


def _build_model(method: str, known_class_count: int, config: Mapping[str, Any]) -> nn.Module:
    if method.startswith("CE_"):
        return TwoViewCEClassifier(known_class_count)
    arpl = config["model"]["arpl"]
    return TwoViewARPLClassifier(
        known_class_count,
        temperature=float(arpl["temperature"]),
        weight_pl=float(arpl["weight_pl"]),
        margin=float(arpl["margin"]),
        reciprocal_init_std=float(arpl["reciprocal_init_std"]),
        initial_radius=float(arpl["initial_radius"]),
    )


def view_auxiliary_loss(
    model: nn.Module,
    method: str,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    lambda_view: float,
) -> tuple[Any, dict[str, torch.Tensor]]:
    output = model.forward_all_views(inputs)
    view_logits = output.per_view_logits.reshape(-1, output.per_view_logits.shape[-1])
    repeated_labels = labels[:, None].expand(-1, 2).reshape(-1)
    if method.startswith("CE_"):
        fused_classification = F.cross_entropy(output.fused_logits, labels)
        view_classification = F.cross_entropy(view_logits, repeated_labels)
        margin_loss = torch.zeros((), device=inputs.device)
        true_distance = torch.zeros(labels.shape[0], device=inputs.device)
        active_margin = torch.zeros(labels.shape[0], device=inputs.device)
        total = fused_classification
    else:
        fused_loss = model.head.loss(output.fused_features, labels)
        fused_classification = fused_loss.classification_loss
        view_classification = F.cross_entropy(
            view_logits / model.head.temperature, repeated_labels
        )
        margin_loss = fused_loss.margin_loss
        true_distance = fused_loss.true_class_reciprocal_distance
        active_margin = (
            true_distance - model.head.radius + model.head.margin > 0
        ).to(inputs.dtype)
        total = fused_loss.total_loss
    if method.endswith("VIEW_AUX"):
        total = total + float(lambda_view) * view_classification
    return output, {
        "total_loss": total,
        "fused_classification_loss": fused_classification,
        "view_classification_loss": view_classification,
        "margin_loss": margin_loss,
        "true_class_reciprocal_distance": true_distance,
        "active_margin": active_margin,
    }


def _infer_detailed(
    model: nn.Module,
    dataset: PairTensorDataset,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    collected: dict[str, list[np.ndarray]] = {
        "per_view_features": [],
        "fused_features": [],
        "per_view_logits": [],
        "fused_logits": [],
        "labels": [],
    }
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            output = model.forward_all_views(inputs.to(device))
            collected["per_view_features"].append(
                output.per_view_features.cpu().numpy().astype(np.float32)
            )
            collected["fused_features"].append(
                output.fused_features.cpu().numpy().astype(np.float32)
            )
            collected["per_view_logits"].append(
                output.per_view_logits.cpu().numpy().astype(np.float64)
            )
            collected["fused_logits"].append(
                output.fused_logits.cpu().numpy().astype(np.float64)
            )
            collected["labels"].append(labels.numpy().astype(np.int64))
    result = {key: np.concatenate(values) for key, values in collected.items()}
    if not all(np.isfinite(value).all() for value in result.values()):
        raise NumericalInstabilityError("detailed inference produced NaN or Inf")
    return result


def _train_model(
    model: nn.Module,
    *,
    method: str,
    prepared: PreparedSurrogateSplit,
    seed: int,
    learning_rate: float,
    config: Mapping[str, Any],
    mode: str,
    device: torch.device,
) -> dict[str, Any]:
    training = config["training"]
    batch_size = int(training["batch_size"])
    train_dataset = PairTensorDataset(prepared.inputs["train"], prepared.labels["train"])
    calibration_dataset = PairTensorDataset(
        prepared.inputs["known_calibration"], prepared.labels["known_calibration"]
    )
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(
            seed + int(training["dataloader_seed_offset"])
        ),
        num_workers=int(training["num_workers"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(training["weight_decay"]),
    )
    best_key = (-np.inf, -np.inf)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    stale = 0
    log: list[dict[str, Any]] = []
    lambda_view = float(config["model"]["lambda_view"])
    for epoch in range(1, int(training["max_epochs"][mode]) + 1):
        start = time.perf_counter()
        model.train()
        totals = {
            "count": 0,
            "total_loss": 0.0,
            "fused_classification_loss": 0.0,
            "view_classification_loss": 0.0,
            "margin_loss": 0.0,
            "true_class_reciprocal_distance": 0.0,
            "active_margin": 0.0,
            "correct": 0,
        }
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            output, losses = view_auxiliary_loss(
                model,
                method,
                inputs,
                labels,
                lambda_view=lambda_view,
            )
            total_loss = losses["total_loss"]
            if not torch.isfinite(total_loss):
                raise NumericalInstabilityError(f"{method} loss became non-finite")
            total_loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise NumericalInstabilityError(f"{method} gradient became non-finite")
            optimizer.step()
            if not _is_finite_model(model):
                raise NumericalInstabilityError(f"{method} parameter became non-finite")
            count = int(labels.numel())
            totals["count"] += count
            for key in (
                "total_loss",
                "fused_classification_loss",
                "view_classification_loss",
                "margin_loss",
            ):
                totals[key] += float(losses[key].item()) * count
            totals["true_class_reciprocal_distance"] += float(
                losses["true_class_reciprocal_distance"].sum().item()
            )
            totals["active_margin"] += float(losses["active_margin"].sum().item())
            totals["correct"] += int(
                (output.fused_logits.argmax(dim=1) == labels).sum().item()
            )
        calibration = _infer_detailed(
            model, calibration_dataset, device=device, batch_size=batch_size
        )
        predicted = calibration["fused_logits"].argmax(axis=1)
        true = calibration["labels"]
        accuracy = float(np.mean(predicted == true))
        f1_values = []
        for label in range(len(prepared.train_class_order)):
            tp = int(np.count_nonzero((true == label) & (predicted == label)))
            fp = int(np.count_nonzero((true != label) & (predicted == label)))
            fn = int(np.count_nonzero((true == label) & (predicted != label)))
            denominator = 2 * tp + fp + fn
            f1_values.append(0.0 if denominator == 0 else 2 * tp / denominator)
        macro_f1 = float(np.mean(f1_values))
        key = (accuracy, macro_f1)
        improved = key > best_key
        if improved:
            best_key = key
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        count = int(totals["count"])
        row: dict[str, Any] = {
            "epoch": epoch,
            "method": method,
            "train_accuracy": totals["correct"] / count,
            "known_calibration_accuracy": accuracy,
            "known_calibration_macro_f1": macro_f1,
            "checkpoint_improved": improved,
            "epochs_without_improvement": stale,
            "elapsed_seconds": time.perf_counter() - start,
        }
        for key_name in (
            "total_loss",
            "fused_classification_loss",
            "view_classification_loss",
            "margin_loss",
            "true_class_reciprocal_distance",
            "active_margin",
        ):
            row[f"train_{key_name}"] = totals[key_name] / count
        if method.startswith("ARPL_"):
            row["radius"] = float(model.head.radius.detach().cpu().item())
            norms = model.head.reciprocal_points.detach().norm(dim=2).cpu().numpy()
            row["reciprocal_point_norm_mean"] = float(norms.mean())
        log.append(row)
        if stale >= int(training["early_stopping_patience"]):
            row["early_stopping_triggered"] = True
            break
    if best_state is None:
        raise NumericalInstabilityError(f"{method} produced no checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    return {
        "model": model,
        "best_state": best_state,
        "best_epoch": best_epoch,
        "stopped_epoch": len(log),
        "best_known_calibration_accuracy": best_key[0],
        "best_known_calibration_macro_f1": best_key[1],
        "learning_rate": float(learning_rate),
        "training_log": log,
    }


def _load_prior_model(
    prior_method_root: Path,
    *,
    method: str,
    known_class_count: int,
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(
        prior_method_root / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    model = _build_model(method, known_class_count, config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


def _evaluate_arrays(
    arrays: Mapping[str, Mapping[str, np.ndarray]],
    *,
    prepared: PreparedSurrogateSplit,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    known_arrays = arrays["known_calibration"]
    unknown_arrays = arrays["surrogate_unknown"]
    known_raw = raw_view_evidence(
        known_arrays["per_view_features"],
        known_arrays["per_view_logits"],
        known_arrays["fused_logits"],
    )
    unknown_raw = raw_view_evidence(
        unknown_arrays["per_view_features"],
        unknown_arrays["per_view_logits"],
        unknown_arrays["fused_logits"],
    )
    quantiles, ecdf_parameters = fit_and_apply_evidence_ecdfs(known_raw, unknown_raw)
    known_pred = known_arrays["fused_logits"].argmax(axis=1)
    unknown_pred = unknown_arrays["fused_logits"].argmax(axis=1)
    metrics = {}
    for rule in RULES:
        metrics[rule] = evaluate_open_set(
            known_true=known_arrays["labels"],
            known_pred=known_pred,
            known_unknown_scores=quantiles["known_calibration"][rule],
            unknown_pred=unknown_pred,
            unknown_unknown_scores=quantiles["surrogate_unknown"][rule],
            known_validation_scores=quantiles["known_calibration"][rule],
            known_class_count=len(prepared.train_class_order),
            known_acceptance_rate=float(
                config["evidence"]["threshold_known_acceptance_rate"]
            ),
        )
    prediction_rows: list[dict[str, Any]] = []
    for role, values, raw in (
        ("known_calibration", known_arrays, known_raw),
        ("surrogate_unknown", unknown_arrays, unknown_raw),
    ):
        q = quantiles[role]
        predictions = values["fused_logits"].argmax(axis=1)
        for index, (pair_id, class_name) in enumerate(
            zip(prepared.pair_ids[role], prepared.class_names[role], strict=True)
        ):
            row: dict[str, Any] = {
                "pair_id": pair_id,
                "evaluation_role": role,
                "class_name": class_name,
                "true_label": int(values["labels"][index]),
                "predicted_known_label": int(predictions[index]),
                "fused_logits": json.dumps(
                    values["fused_logits"][index].tolist(), separators=(",", ":")
                ),
                "view1_logits": json.dumps(
                    values["per_view_logits"][index, 0].tolist(), separators=(",", ":")
                ),
                "view2_logits": json.dumps(
                    values["per_view_logits"][index, 1].tolist(), separators=(",", ":")
                ),
            }
            for name in (
                "u_f",
                "u_1",
                "u_2",
                "js",
                "feature_cosine_distance",
                "feature_l2_mean",
                "fused_norm",
            ):
                row[name] = float(raw[name][index])
            for name in ("q_f", "q_1", "q_2", "q_js", *RULES, "mean_q1_q2"):
                row[name] = float(q[name][index])
            for rule in RULES:
                row[f"threshold_{rule}"] = float(metrics[rule]["threshold"])
                row[f"rejected_{rule}"] = bool(q[rule][index] > metrics[rule]["threshold"])
            prediction_rows.append(row)
    return {
        "arrays": arrays,
        "raw": {"known_calibration": known_raw, "surrogate_unknown": unknown_raw},
        "quantiles": quantiles,
        "ecdf_parameters": ecdf_parameters,
        "metrics": metrics,
        "prediction_rows": prediction_rows,
    }


def recompute_rule_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    rule: str,
    known_class_count: int,
    known_acceptance_rate: float = 0.95,
) -> dict[str, float]:
    known = [row for row in rows if row["evaluation_role"] == "known_calibration"]
    unknown = [row for row in rows if row["evaluation_role"] == "surrogate_unknown"]
    return evaluate_open_set(
        known_true=np.asarray([int(row["true_label"]) for row in known]),
        known_pred=np.asarray([int(row["predicted_known_label"]) for row in known]),
        known_unknown_scores=np.asarray([float(row[rule]) for row in known]),
        unknown_pred=np.asarray([int(row["predicted_known_label"]) for row in unknown]),
        unknown_unknown_scores=np.asarray([float(row[rule]) for row in unknown]),
        known_validation_scores=np.asarray([float(row[rule]) for row in known]),
        known_class_count=known_class_count,
        known_acceptance_rate=known_acceptance_rate,
    )


def _save_evaluated(
    destination: Path,
    *,
    method: str,
    seed: int,
    prepared: PreparedSurrogateSplit,
    evaluated: Mapping[str, Any],
    config: Mapping[str, Any],
    trained: Mapping[str, Any] | None,
    source_checkpoint: Path | None = None,
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=False)
    _write_csv(destination / "predictions.csv", evaluated["prediction_rows"])
    _write_json(destination / "metrics_by_rule.json", evaluated["metrics"])
    _write_json(destination / "ecdf_parameters.json", evaluated["ecdf_parameters"])
    np.savez_compressed(
        destination / "features_logits_scores.npz",
        **{
            f"{role}_{name}": array
            for role, values in evaluated["arrays"].items()
            for name, array in values.items()
        },
        **{
            f"{role}_{name}": array
            for role, values in evaluated["quantiles"].items()
            for name, array in values.items()
        },
    )
    if trained is not None:
        checkpoint = {
            "method": method,
            "model_state_dict": trained["best_state"],
            "train_class_order": prepared.train_class_order,
            "surrogate_class_order": prepared.surrogate_class_order,
            "normalization": asdict(prepared.normalization),
            "best_epoch": trained["best_epoch"],
            "pair_manifest_sha256": prepared.pair_manifest_sha256,
            "initialization_seed": seed,
            "config_sha256": config["_config_sha256"],
        }
        torch.save(checkpoint, destination / "checkpoint.pt")
        (destination / "training_log.jsonl").write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in trained["training_log"]
            ),
            encoding="utf-8",
        )
        training_summary = {
            key: trained[key]
            for key in (
                "best_epoch",
                "stopped_epoch",
                "best_known_calibration_accuracy",
                "best_known_calibration_macro_f1",
                "learning_rate",
            )
        }
    else:
        if source_checkpoint is None:
            raise DataValidationError("baseline source checkpoint is missing")
        _write_json(
            destination / "baseline_reference.json",
            {
                "source_checkpoint": str(source_checkpoint),
                "source_checkpoint_sha256": file_sha256(source_checkpoint),
                "source_pair_manifest_sha256": prepared.pair_manifest_sha256,
            },
        )
        training_summary = {"reused_prior_checkpoint": True}
    resolved = dict(config)
    resolved["_resolved"] = {
        "method": method,
        "seed": seed,
        "split_id": prepared.split_id,
        "pair_manifest_sha256": prepared.pair_manifest_sha256,
        "train_class_order": list(prepared.train_class_order),
        "surrogate_class_order": list(prepared.surrogate_class_order),
    }
    (destination / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    _write_json(destination / "artifact_hashes.json", _artifact_hashes(destination))
    return {
        "method": method,
        "seed": seed,
        "metrics": evaluated["metrics"],
        "training": training_summary,
        "artifact_hashes_sha256": file_sha256(destination / "artifact_hashes.json"),
    }


def _arrays_from_saved_features(
    model: nn.Module, features_path: Path
) -> dict[str, dict[str, np.ndarray]]:
    stored = np.load(features_path)
    result: dict[str, dict[str, np.ndarray]] = {}
    for role in ("known_calibration", "surrogate_unknown"):
        per_view = np.asarray(stored[f"{role}_per_view"], dtype=np.float32)
        fused = np.asarray(stored[f"{role}_fused"], dtype=np.float32)
        if not np.allclose(fused, per_view.mean(axis=1), rtol=1e-6, atol=1e-6):
            raise DataValidationError("saved fused feature is not the mean of two views")
        with torch.no_grad():
            tensor = torch.from_numpy(per_view)
            if isinstance(model, TwoViewCEClassifier):
                per_view_logits = model.classifier(tensor).cpu().numpy()
                fused_logits = model.classifier(torch.from_numpy(fused)).cpu().numpy()
            else:
                shape = tensor.shape
                per_view_logits = (
                    model.head.logits(tensor.reshape(-1, shape[-1]))
                    .reshape(shape[0], shape[1], -1)
                    .cpu()
                    .numpy()
                )
                fused_logits = model.head.logits(torch.from_numpy(fused)).cpu().numpy()
        result[role] = {
            "per_view_features": per_view,
            "fused_features": fused,
            "per_view_logits": np.asarray(per_view_logits, dtype=np.float64),
            "fused_logits": np.asarray(fused_logits, dtype=np.float64),
        }
    return result


def _average_ranks(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    sorted_values = array[order]
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        stop = start + 1
        while stop < array.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if np.std(x) == 0 or np.std(y) == 0:
        return {"pearson": 0.0, "spearman": 0.0}
    return {
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "spearman": float(np.corrcoef(_average_ranks(x), _average_ranks(y))[0, 1]),
    }


def _direct_score_metrics(
    known_arrays: Mapping[str, np.ndarray],
    unknown_arrays: Mapping[str, np.ndarray],
    known_scores: Mapping[str, np.ndarray],
    unknown_scores: Mapping[str, np.ndarray],
    known_class_count: int,
) -> dict[str, dict[str, float]]:
    names = (
        "u_f",
        "u_worst",
        "u_mean",
        "u_gap",
        "view_prediction_disagreement",
        "js",
        "feature_cosine_distance",
        "feature_l2_mean",
        "view1_norm",
        "view2_norm",
        "max_view_norm",
        "mean_view_norm",
        "fused_norm",
    )
    return {
        name: {
            key: value
            for key, value in evaluate_open_set(
                known_true=known_arrays["labels"],
                known_pred=known_arrays["fused_logits"].argmax(axis=1),
                known_unknown_scores=known_scores[name],
                unknown_pred=unknown_arrays["fused_logits"].argmax(axis=1),
                unknown_unknown_scores=unknown_scores[name],
                known_validation_scores=known_scores[name],
                known_class_count=known_class_count,
                known_acceptance_rate=0.95,
            ).items()
            if key in {"auroc", "oscr", "fpr95"}
        }
        for name in names
    }


def _nearest_training_support_score(
    train_values: np.ndarray, evaluation_values: np.ndarray
) -> np.ndarray:
    train = np.atleast_2d(np.asarray(train_values, dtype=np.float64))
    evaluation = np.atleast_2d(np.asarray(evaluation_values, dtype=np.float64))
    if train.shape[1] != evaluation.shape[1]:
        raise DataValidationError("shortcut feature dimensions differ")
    scale = train.std(axis=0)
    scale[scale == 0] = 1.0
    normalized_train = train / scale
    normalized_evaluation = evaluation / scale
    distances = np.sum(
        (normalized_evaluation[:, None, :] - normalized_train[None, :, :]) ** 2,
        axis=2,
    )
    return distances.min(axis=1)


def _length_padding_audit(
    bundle: ProcessedBundle,
    manifest_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    metadata = {str(row["sample_id"]): row for row in bundle.rows}
    rows: list[dict[str, Any]] = []
    for pair in manifest_rows:
        values = []
        for sample_key in ("view1_sample_id", "view2_sample_id"):
            sample = metadata[str(pair[sample_key])]
            values.append(
                (
                    float(sample["profile_length"]),
                    float(sample["left_padding_bins"]),
                    float(sample["right_padding_bins"]),
                )
            )
        vector = np.mean(np.asarray(values), axis=0)
        rows.append(
            {
                "pair_id": pair["pair_id"],
                "evaluation_role": pair["experiment_role"],
                "class_name": pair["class_name"],
                "original_profile_length": float(vector[0]),
                "left_padding_bins": float(vector[1]),
                "right_padding_bins": float(vector[2]),
            }
        )
    train = [row for row in rows if row["evaluation_role"] == "train_known"]
    known = [row for row in rows if row["evaluation_role"] == "known_calibration"]
    unknown = [row for row in rows if row["evaluation_role"] == "surrogate_unknown"]
    feature_sets = {
        "original_profile_length": ("original_profile_length",),
        "left_padding_bins": ("left_padding_bins",),
        "right_padding_bins": ("right_padding_bins",),
        "combined_length_padding": (
            "original_profile_length",
            "left_padding_bins",
            "right_padding_bins",
        ),
    }
    metrics = {}
    for name, fields in feature_sets.items():
        train_values = np.asarray([[row[field] for field in fields] for row in train])
        known_score = _nearest_training_support_score(
            train_values, np.asarray([[row[field] for field in fields] for row in known])
        )
        unknown_score = _nearest_training_support_score(
            train_values, np.asarray([[row[field] for field in fields] for row in unknown])
        )
        metrics[name] = binary_auroc(known_score, unknown_score)
        for row, score in zip(known, known_score, strict=True):
            row[f"score_{name}"] = float(score)
        for row, score in zip(unknown, unknown_score, strict=True):
            row[f"score_{name}"] = float(score)
    return metrics, [*known, *unknown]


def audit_prior_artifacts(
    config_path: str | Path,
    bundle_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    config = load_mv_evidence_config(config_path)
    project_root = Path(config["_config_path"]).parents[3]
    prior_root = project_root / config["prior_baseline"]["artifact_root"]
    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise DataValidationError(f"audit output is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if file_sha256(prior_root / "artifact_hashes.json") != config["prior_baseline"][
        "root_hash_manifest_sha256"
    ]:
        raise DataValidationError("prior root hash manifest changed")
    root_hashes = json.loads((prior_root / "artifact_hashes.json").read_text())
    for relative, expected in root_hashes.items():
        if file_sha256(prior_root / relative) != expected:
            raise DataValidationError(f"prior artifact hash mismatch: {relative}")
    bundle_config = config["bundle"]
    bundle = load_processed_bundle(
        bundle_root,
        expected_profiles_sha256=bundle_config["profiles_sha256"],
        expected_manifest_sha256=bundle_config["manifest_sha256"],
        expected_bundle_sha256=bundle_config["bundle_sha256"],
    )
    summary: dict[str, Any] = {
        "prior_hashes_verified": len(root_hashes),
        "splits": {},
        "final_unknown_used": False,
        "even_angle_test_used": False,
        "posthoc_only": True,
    }
    for spec in config["classes"]["development_splits"]:
        split_id = spec["split_id"]
        prepared = prepare_surrogate_split(
            bundle,
            source_known_order=config["classes"]["source_known_order"],
            split_id=split_id,
            angle_fold=int(spec["angle_fold"]),
            train_known_indices=spec["train_known_indices"],
            surrogate_unknown_indices=spec["surrogate_unknown_indices"],
            pairs_per_class=500,
            base_seed=int(config["sampling"]["base_seed"]),
            fold_count=int(config["sampling"]["fold_count"]),
            normalization_epsilon=float(config["normalization"]["epsilon"]),
        )
        expected_pair_hash = config["prior_baseline"]["development_pair_manifest_sha256"][
            split_id
        ]
        if prepared.pair_manifest_sha256 != expected_pair_hash:
            raise DataValidationError(f"{split_id} prior pair manifest changed")
        split_root = output / split_id
        split_root.mkdir(parents=True, exist_ok=False)
        (split_root / "pair_manifest.csv").write_bytes(prepared.pair_manifest_bytes)
        length_metrics, length_rows = _length_padding_audit(
            bundle, prepared.pair_manifest_rows
        )
        _write_json(split_root / "length_padding_metrics.json", length_metrics)
        _write_csv(split_root / "length_padding_samples.csv", length_rows)
        split_summary: dict[str, Any] = {
            "pair_manifest_sha256": prepared.pair_manifest_sha256,
            "length_padding_auroc": length_metrics,
            "methods": {},
        }
        for method in BASELINE_METHODS:
            prior_method = prior_root / split_id / method
            model, checkpoint = _load_prior_model(
                prior_method,
                method=method,
                known_class_count=5,
                config=config,
                device=torch.device("cpu"),
            )
            arrays = _arrays_from_saved_features(model, prior_method / "features.npz")
            prediction_rows = list(
                csv.DictReader(
                    (prior_method / "predictions.csv").open(
                        newline="", encoding="utf-8"
                    )
                )
            )
            for role in ("known_calibration", "surrogate_unknown"):
                role_rows = [row for row in prediction_rows if row["evaluation_role"] == role]
                arrays[role]["labels"] = np.asarray(
                    [int(row["true_label"]) for row in role_rows], dtype=np.int64
                )
            known_raw = raw_view_evidence(
                arrays["known_calibration"]["per_view_features"],
                arrays["known_calibration"]["per_view_logits"],
                arrays["known_calibration"]["fused_logits"],
            )
            unknown_raw = raw_view_evidence(
                arrays["surrogate_unknown"]["per_view_features"],
                arrays["surrogate_unknown"]["per_view_logits"],
                arrays["surrogate_unknown"]["fused_logits"],
            )
            direct_metrics = _direct_score_metrics(
                arrays["known_calibration"],
                arrays["surrogate_unknown"],
                known_raw,
                unknown_raw,
                5,
            )
            correlations = {
                role: {
                    norm_name: _correlation(raw["u_f"], raw[norm_name])
                    for norm_name in (
                        "view1_norm",
                        "view2_norm",
                        "max_view_norm",
                        "mean_view_norm",
                        "fused_norm",
                    )
                }
                for role, raw in (
                    ("known_calibration", known_raw),
                    ("surrogate_unknown", unknown_raw),
                )
            }
            identity: dict[str, Any] | None = None
            if method == "ARPL_LITE":
                residuals = []
                class_spreads = []
                allclose_checks = []
                for role, values in arrays.items():
                    correction = np.mean(
                        (values["per_view_features"][:, 0] - values["per_view_features"][:, 1])
                        ** 2,
                        axis=1,
                    ) / 4.0
                    left = values["per_view_logits"].mean(axis=1)
                    right = values["fused_logits"] + correction[:, None]
                    residual = left - right
                    residuals.append(np.abs(residual).max())
                    class_spreads.append(np.ptp(residual, axis=1).max())
                    allclose_checks.append(
                        np.allclose(left, right, rtol=1e-5, atol=1e-6)
                    )
                identity = {
                    "max_abs_residual": float(max(residuals)),
                    "max_classwise_spread": float(max(class_spreads)),
                    "rtol": 1e-5,
                    "atol": 1e-6,
                    "passed": bool(all(allclose_checks)),
                }
                if not identity["passed"]:
                    raise DataValidationError("ARPL mean-view logit identity failed")
            method_root = split_root / method
            method_root.mkdir()
            _write_json(method_root / "direct_score_metrics.json", direct_metrics)
            _write_json(method_root / "score_norm_correlations.json", correlations)
            if identity is not None:
                _write_json(method_root / "arpl_logit_identity.json", identity)
            sample_rows = []
            for role, raw in (
                ("known_calibration", known_raw),
                ("surrogate_unknown", unknown_raw),
            ):
                role_predictions = [
                    row for row in prediction_rows if row["evaluation_role"] == role
                ]
                for index, row in enumerate(role_predictions):
                    sample_rows.append(
                        {
                            "pair_id": row["pair_id"],
                            "evaluation_role": role,
                            "class_name": row["class_name"],
                            **{key: float(value[index]) for key, value in raw.items()},
                        }
                    )
            _write_csv(method_root / "posthoc_samples.csv", sample_rows)
            _write_json(
                method_root / "source_audit.json",
                {
                    "checkpoint_sha256": file_sha256(prior_method / "checkpoint.pt"),
                    "features_sha256": file_sha256(prior_method / "features.npz"),
                    "predictions_sha256": file_sha256(prior_method / "predictions.csv"),
                    "checkpoint_pair_manifest_sha256": checkpoint[
                        "pair_manifest_sha256"
                    ],
                },
            )
            split_summary["methods"][method] = {
                "direct_score_metrics": direct_metrics,
                "correlations": correlations,
                "arpl_logit_identity": identity,
            }
        summary["splits"][split_id] = split_summary
    _write_json(output / "summary.json", summary)
    _write_json(output / "artifact_hashes.json", _artifact_hashes(output))
    return summary


def _prepare_split(
    bundle: ProcessedBundle,
    config: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    mode: str,
) -> PreparedSurrogateSplit:
    return prepare_surrogate_split(
        bundle,
        source_known_order=config["classes"]["source_known_order"],
        split_id=str(spec["split_id"]),
        angle_fold=int(spec["angle_fold"]),
        train_known_indices=spec["train_known_indices"],
        surrogate_unknown_indices=spec["surrogate_unknown_indices"],
        pairs_per_class=int(config["sampling"]["pairs_per_class"][mode]),
        base_seed=int(config["sampling"]["base_seed"]),
        fold_count=int(config["sampling"]["fold_count"]),
        normalization_epsilon=float(config["normalization"]["epsilon"]),
    )


def _train_method_group(
    prepared: PreparedSurrogateSplit,
    *,
    seed: int,
    config: Mapping[str, Any],
    mode: str,
    device: torch.device,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    primary_lr = float(config["training"]["learning_rate"])
    fallback_lr = float(config["training"]["numerical_fallback"]["learning_rate"])
    fallback_used = False
    learning_rate = primary_lr
    while True:
        models: dict[str, nn.Module] = {}
        initial_hashes: dict[str, str] = {}
        full_initial_hashes: dict[str, str] = {}
        for method in METHODS:
            _set_determinism(seed, bool(config["training"]["deterministic_algorithms"]))
            model = _build_model(method, len(prepared.train_class_order), config).to(device)
            models[method] = model
            initial_hashes[method] = _state_sha256(model.backbone.state_dict())
            full_initial_hashes[method] = _state_sha256(model.state_dict())
        if len(set(initial_hashes.values())) != 1:
            raise DataValidationError("four methods do not share backbone initialization")
        if full_initial_hashes["CE_MLS"] != full_initial_hashes["CE_VIEW_AUX"]:
            raise DataValidationError("CE baseline and auxiliary initialization differ")
        if full_initial_hashes["ARPL_LITE"] != full_initial_hashes["ARPL_VIEW_AUX"]:
            raise DataValidationError("ARPL baseline and auxiliary initialization differ")
        try:
            trained = {
                method: _train_model(
                    models[method],
                    method=method,
                    prepared=prepared,
                    seed=seed,
                    learning_rate=learning_rate,
                    config=config,
                    mode=mode,
                    device=device,
                )
                for method in METHODS
            }
            return trained, {
                "fallback_used": fallback_used,
                "learning_rate": learning_rate,
                "shared_backbone_initialization_sha256": next(
                    iter(initial_hashes.values())
                ),
                "ce_shared_full_initialization": True,
                "arpl_shared_full_initialization": True,
            }
        except NumericalInstabilityError:
            if fallback_used:
                raise
            fallback_used = True
            learning_rate = fallback_lr


def _phase_metric_rows(
    result_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for result in result_rows:
        for rule, metrics in result["metrics"].items():
            rows.append(
                {
                    "split_id": result["split_id"],
                    "seed": result["seed"],
                    "method": result["method"],
                    "rule": rule,
                    **metrics,
                }
            )
    return rows


def run_training_phase(
    config_path: str | Path,
    bundle_root: str | Path,
    output_root: str | Path,
    *,
    phase: str,
    mode: str,
    device_request: str,
) -> dict[str, Any]:
    if phase not in {"smoke", "development", "confirmation"}:
        raise DataConfigError("invalid MV evidence phase")
    if mode not in {"smoke", "full"}:
        raise DataConfigError("mode must be smoke or full")
    config = load_mv_evidence_config(config_path)
    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise DataValidationError(f"phase output is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(device_request)
    bundle_config = config["bundle"]
    bundle = load_processed_bundle(
        bundle_root,
        expected_profiles_sha256=bundle_config["profiles_sha256"],
        expected_manifest_sha256=bundle_config["manifest_sha256"],
        expected_bundle_sha256=bundle_config["bundle_sha256"],
    )
    if phase == "smoke":
        specs = [config["classes"]["development_splits"][0]]
        seeds = [int(config["training"]["development_seeds"][0])]
        if mode != "smoke":
            raise DataConfigError("smoke phase requires smoke mode")
    elif phase == "development":
        specs = config["classes"]["development_splits"]
        seeds = [int(value) for value in config["training"]["development_seeds"]]
        if mode != "full":
            raise DataConfigError("development requires full mode")
    else:
        specs = config["classes"]["confirmation_splits"]
        seeds = [int(value) for value in config["training"]["confirmation_seeds"]]
        if mode != "full":
            raise DataConfigError("confirmation requires full mode")
    started = time.perf_counter()
    result_rows: list[dict[str, Any]] = []
    fairness_rows: list[dict[str, Any]] = []
    for spec in specs:
        prepared = _prepare_split(bundle, config, spec, mode=mode)
        split_root = output / str(spec["split_id"])
        split_root.mkdir(parents=True, exist_ok=False)
        (split_root / "pair_manifest.csv").write_bytes(prepared.pair_manifest_bytes)
        _write_json(split_root / "pair_audit.json", prepared.pair_audit)
        _write_json(split_root / "normalization.json", asdict(prepared.normalization))
        if phase == "development":
            expected = config["prior_baseline"]["development_pair_manifest_sha256"][
                str(spec["split_id"])
            ]
            if prepared.pair_manifest_sha256 != expected:
                raise DataValidationError("development pair manifest differs from baseline")
        for seed in seeds:
            seed_root = split_root / f"seed_{seed}"
            seed_root.mkdir()
            trained, fairness = _train_method_group(
                prepared,
                seed=seed,
                config=config,
                mode=mode,
                device=device,
            )
            prediction_pair_orders: dict[str, tuple[str, ...]] = {}
            for method in METHODS:
                model = trained[method]["model"]
                arrays = {
                    role: _infer_detailed(
                        model,
                        PairTensorDataset(prepared.inputs[role], prepared.labels[role]),
                        device=device,
                        batch_size=int(config["training"]["batch_size"]),
                    )
                    for role in ("train", "known_calibration", "surrogate_unknown")
                }
                evaluated = _evaluate_arrays(
                    arrays, prepared=prepared, config=config
                )
                for rule in RULES:
                    recomputed = recompute_rule_metrics(
                        evaluated["prediction_rows"],
                        rule=rule,
                        known_class_count=5,
                        known_acceptance_rate=float(
                            config["evidence"]["threshold_known_acceptance_rate"]
                        ),
                    )
                    if any(
                        not np.isclose(
                            recomputed[key], evaluated["metrics"][rule][key], atol=1e-12
                        )
                        for key in recomputed
                    ):
                        raise DataValidationError("prediction rows do not reproduce metrics")
                saved = _save_evaluated(
                    seed_root / method,
                    method=method,
                    seed=seed,
                    prepared=prepared,
                    evaluated=evaluated,
                    config=config,
                    trained=trained[method],
                )
                result_rows.append(
                    {
                        "split_id": str(spec["split_id"]),
                        "seed": seed,
                        "method": method,
                        "metrics": saved["metrics"],
                        "training": saved["training"],
                    }
                )
                prediction_pair_orders[method] = tuple(
                    row["pair_id"] for row in evaluated["prediction_rows"]
                )
            if len(set(prediction_pair_orders.values())) != 1:
                raise DataValidationError("four methods use different pair order")
            fairness_row = {
                "split_id": str(spec["split_id"]),
                "seed": seed,
                "pair_manifest_sha256": prepared.pair_manifest_sha256,
                "same_pair_manifest_and_label_order": True,
                "final_unknown_used": False,
                "even_angle_test_used": False,
                **fairness,
            }
            fairness_rows.append(fairness_row)
            _write_json(seed_root / "fairness_audit.json", fairness_row)
    metric_rows = _phase_metric_rows(result_rows)
    _write_csv(output / "metrics_by_unit.csv", metric_rows)
    _write_csv(output / "fairness_by_unit.csv", fairness_rows)
    summary: dict[str, Any] = {
        "phase": phase,
        "mode": mode,
        "result_count": len(result_rows),
        "metric_row_count": len(metric_rows),
        "results": result_rows,
        "final_unknown_used": False,
        "even_angle_test_used": False,
        "wall_time_seconds": time.perf_counter() - started,
    }
    if phase == "development":
        selection = select_development_fusion(metric_rows, config)
        summary["development_selection"] = selection
        _write_json(output / "development_selection.json", selection)
    _write_json(output / "summary.json", summary)
    project_root = Path(config["_config_path"]).parents[3]
    _write_json(output / "environment.json", _environment(project_root, device))
    _write_json(output / "artifact_hashes.json", _artifact_hashes(output))
    return summary


def _find_metric_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    split_id: str,
    seed: int,
    method: str,
    rule: str,
) -> Mapping[str, Any]:
    matches = [
        row
        for row in rows
        if row["split_id"] == split_id
        and int(row["seed"]) == seed
        and row["method"] == method
        and row["rule"] == rule
    ]
    if len(matches) != 1:
        raise DataValidationError("metric unit lookup is not unique")
    return matches[0]


def summarize_confirmation(
    metric_rows: Sequence[Mapping[str, Any]],
    *,
    selected_rule: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    comparisons = {
        "view_aux_training": {
            "CE": ("CE_VIEW_AUX", "F0_FUSED", "CE_MLS", "F0_FUSED"),
            "ARPL": ("ARPL_VIEW_AUX", "F0_FUSED", "ARPL_LITE", "F0_FUSED"),
        },
        "evidence_fusion": {
            "CE": ("CE_VIEW_AUX", selected_rule, "CE_VIEW_AUX", "F0_FUSED"),
            "ARPL": (
                "ARPL_VIEW_AUX",
                selected_rule,
                "ARPL_VIEW_AUX",
                "F0_FUSED",
            ),
        },
        "complete_method": {
            "CE": ("CE_VIEW_AUX", selected_rule, "CE_MLS", "F0_FUSED"),
            "ARPL": (
                "ARPL_VIEW_AUX",
                selected_rule,
                "ARPL_LITE",
                "F0_FUSED",
            ),
        },
    }
    split_seed_units = sorted(
        {(str(row["split_id"]), int(row["seed"])) for row in metric_rows}
    )
    if len(split_seed_units) != 12:
        raise DataValidationError("confirmation must contain 12 split-seed units")
    output: dict[str, Any] = {"selected_rule": selected_rule, "comparisons": {}}
    for comparison_name, heads in comparisons.items():
        output["comparisons"][comparison_name] = {}
        for head, (left_method, left_rule, right_method, right_rule) in heads.items():
            paired = []
            for split_id, seed in split_seed_units:
                left = _find_metric_row(
                    metric_rows,
                    split_id=split_id,
                    seed=seed,
                    method=left_method,
                    rule=left_rule,
                )
                right = _find_metric_row(
                    metric_rows,
                    split_id=split_id,
                    seed=seed,
                    method=right_method,
                    rule=right_rule,
                )
                paired.append(
                    {
                        "split_id": split_id,
                        "seed": seed,
                        **{
                            f"delta_{metric}": float(left[metric]) - float(right[metric])
                            for metric in METRIC_KEYS
                        },
                    }
                )
            aggregate = {
                metric: {
                    "mean_delta": float(
                        np.mean([row[f"delta_{metric}"] for row in paired])
                    ),
                    "std_delta": float(
                        np.std([row[f"delta_{metric}"] for row in paired], ddof=0)
                    ),
                    "positive_units": sum(
                        row[f"delta_{metric}"] > 0 for row in paired
                    ),
                }
                for metric in METRIC_KEYS
            }
            output["comparisons"][comparison_name][head] = {
                "left": f"{left_method}+{left_rule}",
                "right": f"{right_method}+{right_rule}",
                "paired_deltas": paired,
                "aggregate": aggregate,
            }
    gate = config["confirmation_decision"]
    success: dict[str, bool] = {}
    for head in ("CE", "ARPL"):
        aggregate = output["comparisons"]["complete_method"][head]["aggregate"]
        success[head] = bool(
            aggregate["auroc"]["mean_delta"]
            >= float(gate["minimum_mean_auroc_delta"])
            and aggregate["auroc"]["positive_units"]
            >= int(gate["minimum_positive_auroc_units"])
            and aggregate["oscr"]["mean_delta"]
            >= float(gate["minimum_mean_oscr_delta"])
            and aggregate["known_accuracy"]["mean_delta"]
            >= -float(gate["maximum_mean_known_accuracy_drop"])
            and aggregate["fpr95"]["mean_delta"]
            <= float(gate["maximum_mean_fpr95_increase"])
        )
    if success == {"CE": True, "ARPL": True}:
        decision = "common_success"
    elif success == {"CE": False, "ARPL": True}:
        decision = "arpl_specific_success"
    elif success == {"CE": True, "ARPL": False}:
        decision = "ce_specific_success"
    else:
        decision = "no_stable_gain"
    output["head_success"] = success
    output["final_decision"] = decision
    return output


def finalize_confirmation(
    config_path: str | Path,
    development_root: str | Path,
    confirmation_root: str | Path,
) -> dict[str, Any]:
    config = load_mv_evidence_config(config_path)
    development = Path(development_root).resolve()
    confirmation = Path(confirmation_root).resolve()
    selection = json.loads((development / "development_selection.json").read_text())
    selected_rule = require_confirmation_gate(selection)
    with (confirmation / "metrics_by_unit.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["seed"] = int(row["seed"])
        for key in row:
            if key in METRIC_KEYS:
                row[key] = float(row[key])
    summary = summarize_confirmation(
        rows, selected_rule=selected_rule, config=config
    )
    _write_json(confirmation / "confirmation_decision.json", summary)
    _write_json(confirmation / "artifact_hashes.json", _artifact_hashes(confirmation))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run preregistered multi-view evidence ARPL/CE experiment"
    )
    parser.add_argument(
        "phase", choices=("audit", "smoke", "development", "confirmation", "finalize")
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--development-root", type=Path)
    parser.add_argument("--confirmation-root", type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if args.phase == "audit":
        if args.bundle_root is None or args.output is None:
            parser.error("audit requires --bundle-root and --output")
        result = audit_prior_artifacts(args.config, args.bundle_root, args.output)
    elif args.phase in {"smoke", "development", "confirmation"}:
        if args.bundle_root is None or args.output is None:
            parser.error("training phases require --bundle-root and --output")
        if args.phase == "confirmation":
            if args.development_root is None:
                parser.error("confirmation requires --development-root")
            selection = json.loads(
                (args.development_root / "development_selection.json").read_text()
            )
            require_confirmation_gate(selection)
        result = run_training_phase(
            args.config,
            args.bundle_root,
            args.output,
            phase=args.phase,
            mode="smoke" if args.phase == "smoke" else "full",
            device_request=args.device,
        )
    else:
        if args.development_root is None or args.confirmation_root is None:
            parser.error("finalize requires development and confirmation roots")
        result = finalize_confirmation(
            args.config, args.development_root, args.confirmation_root
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
