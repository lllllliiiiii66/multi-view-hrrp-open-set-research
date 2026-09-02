from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from hrrp_osr.amdr.data import (
    RANDOMIZED_SLOT_ORDER,
    TwoViewPair,
    build_fold_pairs,
)
from hrrp_osr.amdr.smoke import _git_state, _resource_limits
from hrrp_osr.data.errors import DataConfigError, DataValidationError
from hrrp_osr.data.manifest import file_sha256
from hrrp_osr.data.processed import ProcessedBundle, load_processed_bundle
from hrrp_osr.evaluation.metrics import (
    accuracy_score,
    evaluate_open_set,
    macro_f1_score,
    summarize_metric_repeats,
)
from hrrp_osr.models.arpl import (
    TwoViewARPLClassifier,
    TwoViewCEClassifier,
    maximum_logit_unknown_score,
)


EXPERIMENT_ID = "arpl_lite_surrogate_osr_v1"
RESULT_SCOPE = "diagnostic_surrogate_known_only"
METHODS = ("CE_MLS", "ARPL_LITE")
SOURCE_KNOWN_ORDER = (
    "CVN77",
    "DDG-1000",
    "DDG-112",
    "油气轮MARVEL CRANE",
    "爱达魔都号",
    "迷你好望角型散货船",
    "集装箱船达飞罗尔多夫级",
)


class NumericalInstabilityError(RuntimeError):
    pass


@dataclass(frozen=True)
class ScalarNormalization:
    mean: float
    std: float
    epsilon: float
    unique_base_sample_count: int


@dataclass(frozen=True)
class PreparedSurrogateSplit:
    split_id: str
    angle_fold: int
    train_class_order: tuple[str, ...]
    surrogate_class_order: tuple[str, ...]
    pair_manifest_rows: tuple[dict[str, Any], ...]
    pair_manifest_bytes: bytes
    pair_manifest_sha256: str
    pair_audit: dict[str, Any]
    normalization: ScalarNormalization
    inputs: Mapping[str, np.ndarray]
    labels: Mapping[str, np.ndarray]
    pair_ids: Mapping[str, tuple[str, ...]]
    class_names: Mapping[str, tuple[str, ...]]


class PairTensorDataset(Dataset):
    def __init__(self, inputs: np.ndarray, labels: np.ndarray) -> None:
        if inputs.ndim != 3 or inputs.shape[1:] != (2, 601):
            raise DataValidationError("pair inputs must have shape [n, 2, 601]")
        if labels.ndim != 1 or labels.shape[0] != inputs.shape[0]:
            raise DataValidationError("pair labels do not match inputs")
        self.inputs = np.asarray(inputs, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.labels.size)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.inputs[index]), torch.tensor(
            self.labels[index], dtype=torch.long
        )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DataConfigError(f"{name} must be a mapping")
    return value


def _exact_sequence(value: Any, expected: Sequence[Any], name: str) -> None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DataConfigError(f"{name} must be a sequence")
    if list(value) != list(expected):
        raise DataConfigError(f"{name} must remain {list(expected)}")


def load_arpl_pilot_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = dict(_mapping(yaml.safe_load(handle), "ARPL pilot config"))
    errors: list[str] = []
    if config.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if (
        config.get("stage") != "P1_surrogate"
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("result_scope") != RESULT_SCOPE
    ):
        errors.append("ARPL pilot identity changed")
    evidence = _mapping(config.get("evidence_scope"), "evidence_scope")
    if any(value is not False for value in evidence.values()):
        errors.append("all forbidden evidence-scope flags must remain false")
    reference = _mapping(config.get("official_reference"), "official_reference")
    if (
        reference.get("repository") != "https://github.com/gary23ai/ARPL"
        or reference.get("commit")
        != "3ede8b38e1cfb9d70e106cc19d563453110c36ab"
    ):
        errors.append("official ARPL reference changed")
    bundle = _mapping(config.get("bundle"), "bundle")
    for name in ("profiles_sha256", "manifest_sha256", "bundle_sha256"):
        if not isinstance(bundle.get(name), str) or len(str(bundle.get(name))) != 64:
            errors.append(f"bundle.{name} must be a SHA-256")
    classes = _mapping(config.get("classes"), "classes")
    try:
        _exact_sequence(
            classes.get("source_known_order"), SOURCE_KNOWN_ORDER, "source_known_order"
        )
    except DataConfigError as exc:
        errors.append(str(exc))
    expected_splits = (
        ("S0", 1, [2, 3, 4, 5, 6], [0, 1]),
        ("S1", 2, [0, 1, 4, 5, 6], [2, 3]),
        ("S2", 3, [0, 1, 2, 3, 6], [4, 5]),
    )
    split_rows = classes.get("surrogate_splits")
    if not isinstance(split_rows, Sequence) or len(split_rows) != 3:
        errors.append("exactly three surrogate splits are required")
    else:
        for row, expected in zip(split_rows, expected_splits, strict=True):
            item = _mapping(row, "surrogate split")
            observed = (
                item.get("split_id"),
                item.get("angle_fold"),
                list(item.get("train_known_indices", [])),
                list(item.get("surrogate_unknown_indices", [])),
            )
            if observed != expected:
                errors.append(f"surrogate split must remain {expected}")
    sampling = _mapping(config.get("sampling"), "sampling")
    if (
        sampling.get("development_angle_parity") != "odd"
        or sampling.get("slot_order") != RANDOMIZED_SLOT_ORDER
        or sampling.get("final_test_pairs_generated") is not False
        or dict(_mapping(sampling.get("pairs_per_class"), "pairs_per_class"))
        != {"full": 500, "smoke": 50}
    ):
        errors.append("sampling protocol changed")
    normalization = _mapping(config.get("normalization"), "normalization")
    if (
        normalization.get("method") != "global_scalar_zscore"
        or normalization.get("fit_population")
        != "unique_train_known_base_samples_only"
    ):
        errors.append("normalization protocol changed")
    model = _mapping(config.get("model"), "model")
    arpl = _mapping(model.get("arpl"), "model.arpl")
    if (
        model.get("view_count") != 2
        or model.get("input_length") != 601
        or model.get("feature_dim") != 128
        or model.get("pooling") != "mean"
        or model.get("angle_or_position_encoding") is not False
        or model.get("permutation_invariant") is not True
        or list(model.get("methods", [])) != list(METHODS)
        or arpl.get("num_centers_per_class") != 1
        or float(arpl.get("temperature", -1)) != 1.0
        or float(arpl.get("weight_pl", -1)) != 0.1
        or float(arpl.get("margin", -1)) != 1.0
    ):
        errors.append("model or ARPL math contract changed")
    training = _mapping(config.get("training"), "training")
    if (
        list(training.get("initialization_seeds", [])) != [20260830]
        or training.get("optimizer") != "AdamW"
        or float(training.get("learning_rate", -1)) != 1e-3
        or float(training.get("weight_decay", -1)) != 1e-4
        or int(training.get("batch_size", 0)) != 64
        or dict(_mapping(training.get("max_epochs"), "max_epochs"))
        != {"full": 100, "smoke": 3}
        or int(training.get("early_stopping_patience", 0)) != 15
        or training.get("selection_primary") != "known_calibration_accuracy"
        or training.get("selection_secondary")
        != "known_calibration_macro_f1"
    ):
        errors.append("shared training budget changed")
    fallback = _mapping(training.get("numerical_fallback"), "numerical_fallback")
    if (
        fallback.get("trigger") != "nan_inf_or_optimizer_numerical_error_only"
        or float(fallback.get("learning_rate", -1)) != 3e-4
        or fallback.get("rerun_both_methods") is not True
    ):
        errors.append("numerical fallback changed")
    evaluation = _mapping(config.get("evaluation"), "evaluation")
    if (
        evaluation.get("unknown_score") != "negative_maximum_raw_logit"
        or evaluation.get("unknown_score_direction") != "larger_is_more_unknown"
        or evaluation.get("threshold_source") != "known_calibration_only"
        or float(evaluation.get("threshold_known_acceptance_rate", -1)) != 0.95
    ):
        errors.append("evaluation protocol changed")
    if errors:
        raise DataConfigError("Invalid ARPL pilot config:\n- " + "\n- ".join(errors))
    config["_config_path"] = str(config_path)
    config["_config_sha256"] = file_sha256(config_path)
    return config


def _render_csv(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise DataValidationError("cannot render an empty CSV")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _fit_normalization(
    bundle: ProcessedBundle,
    train_pairs: Sequence[TwoViewPair],
    epsilon: float,
) -> ScalarNormalization:
    indices = sorted(
        {
            index
            for pair in train_pairs
            for index in (pair.view1_row_index, pair.view2_row_index)
        }
    )
    values = np.asarray(bundle.profiles[np.asarray(indices, dtype=np.int64)])
    mean = float(values.mean())
    std = float(values.std())
    if not np.isfinite(mean) or not np.isfinite(std) or std <= epsilon:
        raise DataValidationError("train-known normalization is not finite")
    return ScalarNormalization(mean, std, float(epsilon), len(indices))


def _materialize_inputs(
    bundle: ProcessedBundle,
    pairs: Sequence[TwoViewPair],
    normalization: ScalarNormalization,
) -> np.ndarray:
    view1 = np.asarray(
        bundle.profiles[np.asarray([pair.view1_row_index for pair in pairs])]
    )
    view2 = np.asarray(
        bundle.profiles[np.asarray([pair.view2_row_index for pair in pairs])]
    )
    result = (
        np.stack([view1, view2], axis=1) - normalization.mean
    ) / normalization.std
    result = np.asarray(result, dtype=np.float32)
    if not np.isfinite(result).all():
        raise DataValidationError("normalized pair inputs contain NaN or Inf")
    return result


def prepare_surrogate_split(
    bundle: ProcessedBundle,
    *,
    source_known_order: Sequence[str],
    split_id: str,
    angle_fold: int,
    train_known_indices: Sequence[int],
    surrogate_unknown_indices: Sequence[int],
    pairs_per_class: int,
    base_seed: int,
    fold_count: int = 5,
    normalization_epsilon: float = 1e-8,
) -> PreparedSurrogateSplit:
    source_order = tuple(str(name) for name in source_known_order)
    if len(source_order) != 7 or set(source_order) != set(bundle.known_classes):
        raise DataValidationError("source known class order does not match bundle")
    train_indices = tuple(int(index) for index in train_known_indices)
    surrogate_indices = tuple(int(index) for index in surrogate_unknown_indices)
    if (
        len(train_indices) != 5
        or len(surrogate_indices) != 2
        or set(train_indices) & set(surrogate_indices)
        or set(train_indices) | set(surrogate_indices) != set(range(7))
    ):
        raise DataValidationError("surrogate split must partition seven classes as 5/2")
    train_order = tuple(source_order[index] for index in train_indices)
    surrogate_order = tuple(source_order[index] for index in surrogate_indices)
    all_pairs, source_audit = build_fold_pairs(
        bundle,
        protocol_id=f"{EXPERIMENT_ID}_{split_id}",
        fold_index=int(angle_fold),
        fold_count=int(fold_count),
        base_seed=int(base_seed),
        pairs_per_class={"train": int(pairs_per_class), "calibration": int(pairs_per_class)},
        slot_order=RANDOMIZED_SLOT_ORDER,
        included_splits=("train", "calibration"),
    )
    roles: dict[str, list[TwoViewPair]] = {
        "train_known": [],
        "known_calibration": [],
        "surrogate_unknown": [],
    }
    for pair in all_pairs:
        if pair.split == "train" and pair.class_name in train_order:
            roles["train_known"].append(pair)
        elif pair.split == "calibration" and pair.class_name in train_order:
            roles["known_calibration"].append(pair)
        elif pair.split == "calibration" and pair.class_name in surrogate_order:
            roles["surrogate_unknown"].append(pair)
    expected_counts = {
        "train_known": 5 * pairs_per_class,
        "known_calibration": 5 * pairs_per_class,
        "surrogate_unknown": 2 * pairs_per_class,
    }
    if {role: len(values) for role, values in roles.items()} != expected_counts:
        raise DataValidationError("surrogate pair role counts are invalid")
    source_unknown_names = {
        str(row["class_name"])
        for row in bundle.rows
        if str(row["class_role"]) == "unknown"
    }
    selected_pairs = [pair for values in roles.values() for pair in values]
    if any(pair.class_name in source_unknown_names for pair in selected_pairs):
        raise DataValidationError("final unknown class entered surrogate protocol")
    if any(
        angle % 2 == 0
        for pair in selected_pairs
        for angle in (pair.view1_angle_deg, pair.view2_angle_deg)
    ):
        raise DataValidationError("even-angle test sample entered surrogate protocol")
    train_base_ids = {
        sample_id
        for pair in roles["train_known"]
        for sample_id in (pair.view1_sample_id, pair.view2_sample_id)
    }
    evaluation_base_ids = {
        sample_id
        for role in ("known_calibration", "surrogate_unknown")
        for pair in roles[role]
        for sample_id in (pair.view1_sample_id, pair.view2_sample_id)
    }
    if train_base_ids & evaluation_base_ids:
        raise DataValidationError("base HRRP leaked between training and evaluation")
    class_to_label = {name: index for index, name in enumerate(train_order)}
    manifest_rows: list[dict[str, Any]] = []
    for role in ("train_known", "known_calibration", "surrogate_unknown"):
        for pair in roles[role]:
            row = asdict(pair)
            row["experiment_role"] = role
            row["surrogate_split_id"] = split_id
            row["model_label"] = (
                class_to_label[pair.class_name]
                if role != "surrogate_unknown"
                else len(train_order)
            )
            manifest_rows.append(row)
    manifest_bytes = _render_csv(manifest_rows)
    normalization = _fit_normalization(
        bundle, roles["train_known"], normalization_epsilon
    )
    role_keys = {
        "train_known": "train",
        "known_calibration": "known_calibration",
        "surrogate_unknown": "surrogate_unknown",
    }
    inputs: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    pair_ids: dict[str, tuple[str, ...]] = {}
    class_names: dict[str, tuple[str, ...]] = {}
    for role, key in role_keys.items():
        pairs = roles[role]
        inputs[key] = _materialize_inputs(bundle, pairs, normalization)
        labels[key] = np.asarray(
            [
                class_to_label[pair.class_name]
                if role != "surrogate_unknown"
                else len(train_order)
                for pair in pairs
            ],
            dtype=np.int64,
        )
        pair_ids[key] = tuple(pair.pair_id for pair in pairs)
        class_names[key] = tuple(pair.class_name for pair in pairs)
    audit = {
        "status": "passed",
        "split_id": split_id,
        "angle_fold": int(angle_fold),
        "source_pair_audit": source_audit,
        "selected_pair_counts": expected_counts,
        "train_class_order": list(train_order),
        "surrogate_class_order": list(surrogate_order),
        "train_unique_base_samples": len(train_base_ids),
        "evaluation_unique_base_samples": len(evaluation_base_ids),
        "train_evaluation_base_overlap": 0,
        "final_unknown_pairs": 0,
        "even_angle_pairs": 0,
        "test_pairs_generated": False,
        "test_features_materialized": False,
        "surrogate_train_pairs_materialized": False,
    }
    return PreparedSurrogateSplit(
        split_id=split_id,
        angle_fold=int(angle_fold),
        train_class_order=train_order,
        surrogate_class_order=surrogate_order,
        pair_manifest_rows=tuple(manifest_rows),
        pair_manifest_bytes=manifest_bytes,
        pair_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        pair_audit=audit,
        normalization=normalization,
        inputs=inputs,
        labels=labels,
        pair_ids=pair_ids,
        class_names=class_names,
    )


def _set_determinism(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)


def _resolve_device(request: str) -> torch.device:
    if request == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(request)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise DataValidationError("CUDA was requested but is not available")
    return device


def _state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _is_finite_model(model: nn.Module) -> bool:
    return all(torch.isfinite(parameter).all().item() for parameter in model.parameters())


def _infer(
    model: nn.Module,
    dataset: PairTensorDataset,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    per_view: list[np.ndarray] = []
    fused: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for inputs, batch_labels in loader:
            output = model.forward_representation(inputs.to(device))
            per_view.append(output.per_view_features.cpu().numpy().astype(np.float32))
            fused.append(output.fused_features.cpu().numpy().astype(np.float32))
            logits.append(output.logits.cpu().numpy().astype(np.float64))
            labels.append(batch_labels.numpy().astype(np.int64))
    result = {
        "per_view_features": np.concatenate(per_view),
        "fused_features": np.concatenate(fused),
        "logits": np.concatenate(logits),
        "labels": np.concatenate(labels),
    }
    if not all(np.isfinite(value).all() for value in result.values()):
        raise NumericalInstabilityError("inference produced NaN or Inf")
    return result


def _train_one(
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
    generator = torch.Generator().manual_seed(
        seed + int(training["dataloader_seed_offset"])
    )
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=int(training["num_workers"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(training["weight_decay"]),
    )
    best_key = (-np.inf, -np.inf)
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    log: list[dict[str, Any]] = []
    max_epochs = int(training["max_epochs"][mode])
    patience = int(training["early_stopping_patience"])
    for epoch in range(1, max_epochs + 1):
        started = time.perf_counter()
        model.train()
        totals = {
            "count": 0,
            "total_loss": 0.0,
            "classification_loss": 0.0,
            "margin_loss": 0.0,
            "feature_norm": 0.0,
            "true_class_reciprocal_distance": 0.0,
            "correct": 0,
            "logit_sum": 0.0,
            "logit_count": 0,
        }
        logit_min = np.inf
        logit_max = -np.inf
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            if method == "CE_MLS":
                output = model.forward_representation(inputs)
                classification_loss = torch.nn.functional.cross_entropy(
                    output.logits, labels
                )
                total_loss = classification_loss
                margin_loss = torch.zeros((), device=device)
                true_distance = torch.zeros(labels.shape[0], device=device)
            else:
                output, loss_output = model.loss(inputs, labels)
                classification_loss = loss_output.classification_loss
                total_loss = loss_output.total_loss
                margin_loss = loss_output.margin_loss
                true_distance = loss_output.true_class_reciprocal_distance
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
            totals["total_loss"] += float(total_loss.item()) * count
            totals["classification_loss"] += float(classification_loss.item()) * count
            totals["margin_loss"] += float(margin_loss.item()) * count
            totals["feature_norm"] += float(output.fused_features.norm(dim=1).sum().item())
            totals["true_class_reciprocal_distance"] += float(true_distance.sum().item())
            totals["correct"] += int((output.logits.argmax(dim=1) == labels).sum().item())
            detached_logits = output.logits.detach()
            logit_min = min(logit_min, float(detached_logits.min().item()))
            logit_max = max(logit_max, float(detached_logits.max().item()))
            totals["logit_sum"] += float(detached_logits.sum().item())
            totals["logit_count"] += int(detached_logits.numel())
        calibration = _infer(
            model,
            calibration_dataset,
            device=device,
            batch_size=batch_size,
        )
        predictions = calibration["logits"].argmax(axis=1)
        calibration_accuracy = accuracy_score(calibration["labels"], predictions)
        calibration_macro_f1 = macro_f1_score(
            calibration["labels"], predictions, labels=range(len(prepared.train_class_order))
        )
        selection_key = (calibration_accuracy, calibration_macro_f1)
        improved = selection_key > best_key
        if improved:
            best_key = selection_key
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        count = int(totals["count"])
        record: dict[str, Any] = {
            "epoch": epoch,
            "method": method,
            "train_total_loss": totals["total_loss"] / count,
            "train_classification_loss": totals["classification_loss"] / count,
            "train_margin_loss": totals["margin_loss"] / count,
            "train_accuracy": totals["correct"] / count,
            "mean_fused_feature_norm": totals["feature_norm"] / count,
            "mean_true_class_reciprocal_distance": totals[
                "true_class_reciprocal_distance"
            ]
            / count,
            "logits_min": logit_min,
            "logits_max": logit_max,
            "logits_mean": totals["logit_sum"] / totals["logit_count"],
            "known_calibration_accuracy": calibration_accuracy,
            "known_calibration_macro_f1": calibration_macro_f1,
            "checkpoint_improved": improved,
            "epochs_without_improvement": epochs_without_improvement,
            "elapsed_seconds": time.perf_counter() - started,
        }
        if method == "ARPL_LITE":
            record["radius"] = float(model.head.radius.detach().cpu().item())
            point_norms = model.head.reciprocal_points.detach().norm(dim=2).cpu().numpy()
            record["reciprocal_point_norm_mean"] = float(point_norms.mean())
            record["reciprocal_point_norm_min"] = float(point_norms.min())
            record["reciprocal_point_norm_max"] = float(point_norms.max())
        log.append(record)
        if epochs_without_improvement >= patience:
            record["early_stopping_triggered"] = True
            break
    if best_state is None:
        raise NumericalInstabilityError(f"{method} produced no finite checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    return {
        "model": model,
        "best_state": best_state,
        "best_epoch": best_epoch,
        "best_known_calibration_accuracy": best_key[0],
        "best_known_calibration_macro_f1": best_key[1],
        "stopped_epoch": len(log),
        "training_log": log,
        "learning_rate": float(learning_rate),
    }


def _permutation_audit(
    model: nn.Module,
    inputs: np.ndarray,
    *,
    device: torch.device,
    atol: float,
) -> dict[str, Any]:
    batch = torch.from_numpy(np.asarray(inputs[: min(128, len(inputs))])).to(device)
    model.eval()
    with torch.no_grad():
        reference = model.forward_representation(batch)
        swapped = model.forward_representation(batch[:, [1, 0], :])
    per_view_difference = float(
        (reference.per_view_features[:, [1, 0], :] - swapped.per_view_features)
        .abs()
        .max()
        .item()
    )
    fused_difference = float(
        (reference.fused_features - swapped.fused_features).abs().max().item()
    )
    logit_difference = float((reference.logits - swapped.logits).abs().max().item())
    if max(per_view_difference, fused_difference, logit_difference) > atol:
        raise DataValidationError("two-view permutation audit failed")
    return {
        "status": "passed",
        "sample_count": int(batch.shape[0]),
        "atol": float(atol),
        "per_view_swap_max_abs": per_view_difference,
        "fused_max_abs": fused_difference,
        "logits_max_abs": logit_difference,
    }


def _evaluate_best_model(
    result: Mapping[str, Any],
    *,
    method: str,
    prepared: PreparedSurrogateSplit,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    model = result["model"]
    batch_size = int(config["training"]["batch_size"])
    arrays = {
        role: _infer(
            model,
            PairTensorDataset(prepared.inputs[role], prepared.labels[role]),
            device=device,
            batch_size=batch_size,
        )
        for role in ("train", "known_calibration", "surrogate_unknown")
    }
    known = arrays["known_calibration"]
    unknown = arrays["surrogate_unknown"]
    known_predictions = known["logits"].argmax(axis=1)
    unknown_predictions = unknown["logits"].argmax(axis=1)
    known_scores = -known["logits"].max(axis=1)
    unknown_scores = -unknown["logits"].max(axis=1)
    metrics = evaluate_open_set(
        known_true=known["labels"],
        known_pred=known_predictions,
        known_unknown_scores=known_scores,
        unknown_pred=unknown_predictions,
        unknown_unknown_scores=unknown_scores,
        known_validation_scores=known_scores,
        known_class_count=len(prepared.train_class_order),
        known_acceptance_rate=float(
            config["evaluation"]["threshold_known_acceptance_rate"]
        ),
    )
    threshold = float(metrics["threshold"])
    rows: list[dict[str, Any]] = []
    for role, values, predictions, scores in (
        ("known_calibration", known, known_predictions, known_scores),
        ("surrogate_unknown", unknown, unknown_predictions, unknown_scores),
    ):
        for pair_id, class_name, true_label, predicted, score, logits in zip(
            prepared.pair_ids[role],
            prepared.class_names[role],
            values["labels"],
            predictions,
            scores,
            values["logits"],
            strict=True,
        ):
            rows.append(
                {
                    "pair_id": pair_id,
                    "split_id": prepared.split_id,
                    "evaluation_role": role,
                    "class_name": class_name,
                    "true_label": int(true_label),
                    "predicted_known_label": int(predicted),
                    "predicted_known_class": prepared.train_class_order[int(predicted)],
                    "logits": json.dumps(logits.tolist(), separators=(",", ":")),
                    "unknown_score": float(score),
                    "threshold": threshold,
                    "rejected_as_unknown": bool(score > threshold),
                    "method": method,
                }
            )
    diagnostics: dict[str, Any] = {
        "feature_norm": {
            role: {
                "mean": float(np.linalg.norm(values["fused_features"], axis=1).mean()),
                "std": float(np.linalg.norm(values["fused_features"], axis=1).std()),
                "min": float(np.linalg.norm(values["fused_features"], axis=1).min()),
                "max": float(np.linalg.norm(values["fused_features"], axis=1).max()),
            }
            for role, values in arrays.items()
        },
        "logits": {
            role: {
                "min": float(values["logits"].min()),
                "max": float(values["logits"].max()),
                "mean": float(values["logits"].mean()),
            }
            for role, values in arrays.items()
        },
        "nan_or_inf_detected": False,
    }
    if method == "ARPL_LITE":
        points = model.head.reciprocal_points.detach().cpu().numpy()[:, 0, :]
        train_features = arrays["train"]["fused_features"]
        train_labels = arrays["train"]["labels"]
        distances = np.mean((train_features - points[train_labels]) ** 2, axis=1)
        diagnostics.update(
            {
                "radius": float(model.head.radius.detach().cpu().item()),
                "reciprocal_point_norms": np.linalg.norm(points, axis=1).tolist(),
                "true_class_reciprocal_distance": {
                    "mean": float(distances.mean()),
                    "std": float(distances.std()),
                    "min": float(distances.min()),
                    "max": float(distances.max()),
                },
            }
        )
    permutation = _permutation_audit(
        model,
        prepared.inputs["known_calibration"],
        device=device,
        atol=float(config["evaluation"]["permutation_atol"]),
    )
    return {
        "metrics": metrics,
        "predictions": rows,
        "arrays": arrays,
        "diagnostics": diagnostics,
        "permutation_audit": permutation,
    }


def recompute_metrics_from_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    known_class_count: int,
    known_acceptance_rate: float = 0.95,
) -> dict[str, float]:
    known = [row for row in rows if row["evaluation_role"] == "known_calibration"]
    unknown = [row for row in rows if row["evaluation_role"] == "surrogate_unknown"]
    if not known or not unknown:
        raise DataValidationError("prediction rows are missing known or surrogate data")
    return evaluate_open_set(
        known_true=np.asarray([int(row["true_label"]) for row in known]),
        known_pred=np.asarray([int(row["predicted_known_label"]) for row in known]),
        known_unknown_scores=np.asarray([float(row["unknown_score"]) for row in known]),
        unknown_pred=np.asarray([int(row["predicted_known_label"]) for row in unknown]),
        unknown_unknown_scores=np.asarray(
            [float(row["unknown_score"]) for row in unknown]
        ),
        known_validation_scores=np.asarray(
            [float(row["unknown_score"]) for row in known]
        ),
        known_class_count=known_class_count,
        known_acceptance_rate=known_acceptance_rate,
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_render_csv(rows))


def _artifact_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }


def _environment(project_root: Path, device: torch.device) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "git": _git_state(project_root),
        "resource_limits": _resource_limits(),
    }


def _save_method_result(
    destination: Path,
    *,
    method: str,
    trained: Mapping[str, Any],
    evaluated: Mapping[str, Any],
    prepared: PreparedSurrogateSplit,
    config: Mapping[str, Any],
    mode: str,
    seed: int,
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=False)
    checkpoint = {
        "method": method,
        "architecture": trained["model"].architecture_id,
        "model_state_dict": trained["best_state"],
        "train_class_order": prepared.train_class_order,
        "surrogate_class_order": prepared.surrogate_class_order,
        "normalization": asdict(prepared.normalization),
        "best_epoch": trained["best_epoch"],
        "best_known_calibration_accuracy": trained[
            "best_known_calibration_accuracy"
        ],
        "best_known_calibration_macro_f1": trained[
            "best_known_calibration_macro_f1"
        ],
        "pair_manifest_sha256": prepared.pair_manifest_sha256,
        "initialization_seed": seed,
        "learning_rate": trained["learning_rate"],
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
    _write_csv(destination / "predictions.csv", evaluated["predictions"])
    _write_json(destination / "metrics.json", evaluated["metrics"])
    _write_json(destination / "numerical_diagnostics.json", evaluated["diagnostics"])
    _write_json(destination / "permutation_audit.json", evaluated["permutation_audit"])
    np.savez_compressed(
        destination / "features.npz",
        train_per_view=evaluated["arrays"]["train"]["per_view_features"],
        train_fused=evaluated["arrays"]["train"]["fused_features"],
        known_calibration_per_view=evaluated["arrays"]["known_calibration"][
            "per_view_features"
        ],
        known_calibration_fused=evaluated["arrays"]["known_calibration"][
            "fused_features"
        ],
        surrogate_unknown_per_view=evaluated["arrays"]["surrogate_unknown"][
            "per_view_features"
        ],
        surrogate_unknown_fused=evaluated["arrays"]["surrogate_unknown"][
            "fused_features"
        ],
    )
    resolved = dict(config)
    resolved["_resolved"] = {
        "mode": mode,
        "method": method,
        "split_id": prepared.split_id,
        "angle_fold": prepared.angle_fold,
        "train_class_order": list(prepared.train_class_order),
        "surrogate_class_order": list(prepared.surrogate_class_order),
        "pair_manifest_sha256": prepared.pair_manifest_sha256,
        "initialization_seed": seed,
        "dataloader_seed": seed + int(config["training"]["dataloader_seed_offset"]),
        "learning_rate_used": trained["learning_rate"],
        "best_epoch": trained["best_epoch"],
    }
    (destination / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    _write_json(destination / "artifact_hashes.json", _artifact_hashes(destination))
    return {
        "method": method,
        "best_epoch": trained["best_epoch"],
        "stopped_epoch": trained["stopped_epoch"],
        "learning_rate": trained["learning_rate"],
        "metrics": evaluated["metrics"],
        "diagnostics": evaluated["diagnostics"],
        "permutation_audit": evaluated["permutation_audit"],
        "checkpoint_sha256": file_sha256(destination / "checkpoint.pt"),
    }


def run_arpl_pilot(
    config_path: str | Path,
    bundle_root: str | Path,
    output_root: str | Path,
    *,
    mode: str = "full",
    device_request: str = "auto",
    selected_splits: Sequence[str] | None = None,
) -> dict[str, Any]:
    if mode not in {"smoke", "full"}:
        raise DataConfigError("mode must be smoke or full")
    config_path = Path(config_path).resolve()
    config = load_arpl_pilot_config(config_path)
    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise DataValidationError(f"ARPL output is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(device_request)
    bundle_config = config["bundle"]
    bundle = load_processed_bundle(
        bundle_root,
        expected_profiles_sha256=bundle_config["profiles_sha256"],
        expected_manifest_sha256=bundle_config["manifest_sha256"],
        expected_bundle_sha256=bundle_config["bundle_sha256"],
    )
    split_registry = {
        str(row["split_id"]): row for row in config["classes"]["surrogate_splits"]
    }
    requested = (
        ["S0"]
        if mode == "smoke" and selected_splits is None
        else list(split_registry)
        if selected_splits is None
        else [str(value) for value in selected_splits]
    )
    if not requested or len(set(requested)) != len(requested) or not set(requested) <= set(split_registry):
        raise DataConfigError("selected splits must be a unique subset of S0/S1/S2")
    if mode == "smoke" and requested != ["S0"]:
        raise DataConfigError("smoke is frozen to S0 only")
    seed = int(config["training"]["initialization_seeds"][0])
    results: dict[str, Any] = {}
    aggregate_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for split_id in requested:
        split_spec = split_registry[split_id]
        split_root = output / split_id
        split_root.mkdir(parents=True, exist_ok=False)
        prepared = prepare_surrogate_split(
            bundle,
            source_known_order=config["classes"]["source_known_order"],
            split_id=split_id,
            angle_fold=int(split_spec["angle_fold"]),
            train_known_indices=split_spec["train_known_indices"],
            surrogate_unknown_indices=split_spec["surrogate_unknown_indices"],
            pairs_per_class=int(config["sampling"]["pairs_per_class"][mode]),
            base_seed=int(config["sampling"]["base_seed"]),
            fold_count=int(config["sampling"]["fold_count"]),
            normalization_epsilon=float(config["normalization"]["epsilon"]),
        )
        (split_root / "pair_manifest.csv").write_bytes(prepared.pair_manifest_bytes)
        _write_json(split_root / "pair_audit.json", prepared.pair_audit)
        _write_json(split_root / "normalization.json", asdict(prepared.normalization))
        learning_rate = float(config["training"]["learning_rate"])
        fallback_used = False
        while True:
            _set_determinism(seed, bool(config["training"]["deterministic_algorithms"]))
            ce_model = TwoViewCEClassifier(len(prepared.train_class_order)).to(device)
            ce_backbone_hash = _state_sha256(ce_model.backbone.state_dict())
            _set_determinism(seed, bool(config["training"]["deterministic_algorithms"]))
            arpl_model = TwoViewARPLClassifier(
                len(prepared.train_class_order),
                temperature=float(config["model"]["arpl"]["temperature"]),
                weight_pl=float(config["model"]["arpl"]["weight_pl"]),
                margin=float(config["model"]["arpl"]["margin"]),
                reciprocal_init_std=float(
                    config["model"]["arpl"]["reciprocal_init_std"]
                ),
                initial_radius=float(config["model"]["arpl"]["initial_radius"]),
            ).to(device)
            arpl_backbone_hash = _state_sha256(arpl_model.backbone.state_dict())
            if ce_backbone_hash != arpl_backbone_hash:
                raise DataValidationError("CE and ARPL backbone initialization differs")
            try:
                trained = {
                    "CE_MLS": _train_one(
                        ce_model,
                        method="CE_MLS",
                        prepared=prepared,
                        seed=seed,
                        learning_rate=learning_rate,
                        config=config,
                        mode=mode,
                        device=device,
                    ),
                    "ARPL_LITE": _train_one(
                        arpl_model,
                        method="ARPL_LITE",
                        prepared=prepared,
                        seed=seed,
                        learning_rate=learning_rate,
                        config=config,
                        mode=mode,
                        device=device,
                    ),
                }
                break
            except NumericalInstabilityError:
                if fallback_used:
                    raise
                fallback_used = True
                learning_rate = float(
                    config["training"]["numerical_fallback"]["learning_rate"]
                )
        method_results: dict[str, Any] = {}
        for method in METHODS:
            evaluated = _evaluate_best_model(
                trained[method],
                method=method,
                prepared=prepared,
                config=config,
                device=device,
            )
            recomputed = recompute_metrics_from_prediction_rows(
                evaluated["predictions"],
                known_class_count=len(prepared.train_class_order),
                known_acceptance_rate=float(
                    config["evaluation"]["threshold_known_acceptance_rate"]
                ),
            )
            if any(
                not np.isclose(recomputed[key], evaluated["metrics"][key], atol=1e-12)
                for key in recomputed
            ):
                raise DataValidationError("saved predictions do not reproduce metrics")
            method_result = _save_method_result(
                split_root / method,
                method=method,
                trained=trained[method],
                evaluated=evaluated,
                prepared=prepared,
                config=config,
                mode=mode,
                seed=seed,
            )
            method_results[method] = method_result
            aggregate_rows.append(
                {
                    "split_id": split_id,
                    "method": method,
                    "best_epoch": method_result["best_epoch"],
                    **method_result["metrics"],
                }
            )
        fairness = {
            "status": "passed",
            "same_pair_manifest": True,
            "pair_manifest_sha256": prepared.pair_manifest_sha256,
            "same_train_class_order": True,
            "train_class_order": list(prepared.train_class_order),
            "same_backbone_architecture": True,
            "same_backbone_initialization": True,
            "backbone_initialization_sha256": ce_backbone_hash,
            "same_optimizer_and_budget": True,
            "same_initialization_seed": True,
            "same_dataloader_seed": True,
            "fallback_used_for_both_methods": fallback_used,
            "surrogate_unknown_used_for_training": False,
            "surrogate_unknown_used_for_model_selection": False,
            "surrogate_unknown_used_for_threshold": False,
            "final_unknown_used": False,
            "even_angle_test_used": False,
        }
        _write_json(split_root / "fairness_audit.json", fairness)
        results[split_id] = {
            "pair_manifest_sha256": prepared.pair_manifest_sha256,
            "train_class_order": list(prepared.train_class_order),
            "surrogate_class_order": list(prepared.surrogate_class_order),
            "fallback_used": fallback_used,
            "methods": method_results,
        }
    _write_csv(output / "aggregate_rows.csv", aggregate_rows)
    aggregate = {
        method: summarize_metric_repeats(
            [
                {
                    key: float(row[key])
                    for key in config["evaluation"]["metrics"]
                }
                for row in aggregate_rows
                if row["method"] == method
            ]
        )
        for method in METHODS
    }
    comparisons = []
    for split_id in requested:
        ce = results[split_id]["methods"]["CE_MLS"]["metrics"]
        arpl = results[split_id]["methods"]["ARPL_LITE"]["metrics"]
        comparisons.append(
            {
                "split_id": split_id,
                **{
                    f"arpl_minus_ce_{metric}": float(arpl[metric] - ce[metric])
                    for metric in config["evaluation"]["metrics"]
                },
            }
        )
    _write_json(output / "aggregate_metrics.json", aggregate)
    _write_json(output / "method_comparisons.json", comparisons)
    project_root = config_path.parents[3]
    _write_json(output / "environment.json", _environment(project_root, device))
    final = {
        "experiment_id": EXPERIMENT_ID,
        "result_scope": RESULT_SCOPE,
        "mode": mode,
        "splits": requested,
        "results": results,
        "aggregate": aggregate,
        "comparisons": comparisons,
        "final_unknown_used": False,
        "even_angle_test_used": False,
        "arpl_cs_started": False,
        "per_view_reciprocal_fusion_started": False,
        "cssr_started": False,
        "costarr_started": False,
        "wall_time_seconds": time.perf_counter() - started,
    }
    _write_json(output / "summary.json", final)
    _write_json(output / "artifact_hashes.json", _artifact_hashes(output))
    return final


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run preregistered CE-MLS vs ARPL-lite surrogate HRRP pilot"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--bundle-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--splits", nargs="*", choices=("S0", "S1", "S2"))
    args = parser.parse_args(argv)
    result = run_arpl_pilot(
        args.config,
        args.bundle_root,
        args.output,
        mode=args.mode,
        device_request=args.device,
        selected_splits=args.splits or None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
