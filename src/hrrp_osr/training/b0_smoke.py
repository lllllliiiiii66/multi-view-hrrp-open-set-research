from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from hrrp_osr.baselines.b0 import energy_unknown_score, msp_unknown_score
from hrrp_osr.data.errors import DataConfigError, DataValidationError
from hrrp_osr.data.processed import ProcessedBundle, load_processed_bundle
from hrrp_osr.data.sets import (
    B0_SELECTION_ALGORITHM_VERSION,
    SET_ALGORITHM_VERSION,
    ViewSet,
    build_v3_evaluation_sets,
    render_set_manifest_csv,
    select_b0_single_view,
)
from hrrp_osr.evaluation.metrics import (
    evaluate_open_set,
    summarize_metric_repeats,
    threshold_for_known_acceptance,
)
from hrrp_osr.models.cnn1d import HRRPClassifier1D


EXPECTED_KNOWN_CLASSES = (
    "CVN77",
    "DDG-1000",
    "DDG-112",
    "油气轮MARVEL CRANE",
    "爱达魔都号",
    "迷你好望角型散货船",
    "集装箱船达飞罗尔多夫级",
)


@dataclass(frozen=True)
class ScalarNormalization:
    mean: float
    std: float
    fitted_sample_count: int
    fitted_value_count: int
    fit_population: str = "known_train_only"


class ProfileDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        bundle: ProcessedBundle,
        indices: list[int],
        labels: list[int],
        normalization: ScalarNormalization,
    ) -> None:
        if len(indices) != len(labels):
            raise DataValidationError("dataset indices and labels have different lengths")
        self.bundle = bundle
        self.indices = indices
        self.labels = labels
        self.normalization = normalization

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        profile = np.asarray(
            self.bundle.profiles[self.indices[item]], dtype=np.float32
        )
        normalized = (profile - self.normalization.mean) / self.normalization.std
        return (
            torch.from_numpy(np.asarray(normalized, dtype=np.float32)),
            torch.tensor(self.labels[item], dtype=torch.long),
        )


def _mapping_section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise DataConfigError(f"missing or invalid B0 config section: {name}")
    return value


def load_b0_smoke_config(path: str | Path) -> dict[str, Any]:
    config = load_b0_config(path)
    if config.get("run_kind") != "diagnostic_smoke":
        raise DataConfigError("B0 smoke config must use diagnostic_smoke run_kind")
    return config


def load_b0_main_config(path: str | Path) -> dict[str, Any]:
    config = load_b0_config(path)
    if config.get("run_kind") != "main":
        raise DataConfigError("B0 main config must use main run_kind")
    return config


def _deep_merge_config(
    base: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge_config(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _config_source_paths(path: Path) -> tuple[Path, ...]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise DataConfigError("B0 config must be a YAML mapping")
    extends = raw.get("extends")
    if extends is None:
        return (path,)
    if not isinstance(extends, str) or not extends:
        raise DataConfigError("B0 config extends must be a non-empty filename")
    base_path = (path.parent / extends).resolve()
    if base_path.parent != path.parent.resolve():
        raise DataConfigError("B0 config extends must remain in the same directory")
    base_sources = _config_source_paths(base_path)
    if len(base_sources) != 1:
        raise DataConfigError("nested B0 config overlays are not supported")
    return (*base_sources, path)


def load_b0_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    sources = _config_source_paths(config_path)
    with sources[0].open("r", encoding="utf-8") as handle:
        base_raw = yaml.safe_load(handle)
    if not isinstance(base_raw, Mapping):
        raise DataConfigError("B0 base config must be a YAML mapping")
    config = copy.deepcopy(dict(base_raw))
    if len(sources) == 2:
        with sources[1].open("r", encoding="utf-8") as handle:
            overlay_raw = yaml.safe_load(handle)
        if not isinstance(overlay_raw, Mapping):
            raise DataConfigError("B0 overlay config must be a YAML mapping")
        unexpected = set(overlay_raw) - {"extends", "overrides"}
        if unexpected or not isinstance(overlay_raw.get("overrides"), Mapping):
            raise DataConfigError(
                "B0 overlay may contain only extends and mapping overrides"
            )
        config = _deep_merge_config(config, overlay_raw["overrides"])
    data = _mapping_section(config, "data")
    classes = _mapping_section(config, "classes")
    normalization = _mapping_section(config, "normalization")
    set_protocol = _mapping_section(config, "set_protocol")
    selection = _mapping_section(config, "b0_view_selection")
    model = _mapping_section(config, "model")
    training = _mapping_section(config, "training")
    evaluation = _mapping_section(config, "evaluation")
    errors: list[str] = []
    if int(config.get("schema_version", -1)) != 1:
        errors.append("schema_version must be 1")
    run_kind = config.get("run_kind")
    if (config.get("stage"), config.get("baseline")) != ("P0", "B0"):
        errors.append("config must remain P0/B0")
    if run_kind not in {"diagnostic_smoke", "main"}:
        errors.append("run_kind must be diagnostic_smoke or main")
    if run_kind == "diagnostic_smoke" and config.get("result_scope") != "diagnostic":
        errors.append("smoke result_scope must be diagnostic")
    if run_kind == "main" and config.get("result_scope") != "main_v3":
        errors.append("main result_scope must be main_v3")
    if data.get("accepted_risk_status") != "accepted_for_first_round":
        errors.append("accepted profile-length risk is missing")
    if (int(data.get("input_length", -1)), data.get("input_dtype_model")) != (
        601,
        "float32",
    ):
        errors.append("model input contract must be float32 length 601")
    if tuple(classes.get("known_order", [])) != EXPECTED_KNOWN_CLASSES:
        errors.append("known class order has changed")
    if int(classes.get("unknown_label", -1)) != 7:
        errors.append("unknown label must be 7")
    if (
        normalization.get("method"),
        normalization.get("fit_population"),
    ) != ("global_scalar_zscore", "known_train_only"):
        errors.append("normalization must be global scalar z-score fit on known train only")
    if (
        int(set_protocol.get("view_count", -1)),
        bool(set_protocol.get("distinct_domains")),
        set_protocol.get("set_algorithm_version"),
        bool(set_protocol.get("random_input_order")),
    ) != (3, True, SET_ALGORITHM_VERSION, True):
        errors.append("V=3 distinct-domain random-order set protocol has changed")
    if (
        selection.get("algorithm_version"),
        bool(selection.get("average_metrics_not_predictions")),
    ) != (B0_SELECTION_ALGORITHM_VERSION, True):
        errors.append("B0 view selection protocol has changed")
    if int(selection.get("repeats", 0)) < 2:
        errors.append("B0 must use multiple view-selection repeats")
    if (
        model.get("architecture"),
        list(model.get("channels", [])),
        list(model.get("kernels", [])),
        int(model.get("known_class_count", -1)),
        bool(model.get("angle_or_position_encoding")),
    ) != (
        "shared_hrrp_encoder_1d_v1",
        [32, 64, 128],
        [7, 5, 3],
        7,
        False,
    ):
        errors.append("frozen shared 1D-CNN definition has changed")
    if training.get("unknown_data_used") is not False:
        errors.append("unknown data must not be used for training")
    if run_kind == "diagnostic_smoke":
        if (
            int(training.get("epochs", 0)) != 2
            or training.get("early_stopping") is not False
        ):
            errors.append(
                "diagnostic smoke budget must remain two epochs without early stopping"
            )
    if run_kind == "main":
        if training.get("budget_id") != "neural_budget_v1":
            errors.append("main run must use neural_budget_v1")
        if (
            int(training.get("epochs", 0)) != 100
            or training.get("early_stopping") is not True
            or int(training.get("early_stopping_patience", -1)) != 15
            or float(training.get("early_stopping_min_delta", -1.0)) != 0.0
        ):
            errors.append("main neural budget must be 100 epochs with patience 15")
        if int(selection.get("repeats", 0)) != 30:
            errors.append("main B0 must use exactly 30 view-selection repeats")
        expected_seeds = [20260810, 20260820, 20260830, 20260840, 20260850]
        if list(training.get("planned_initialization_seeds", [])) != expected_seeds:
            errors.append("main initialization seed registry has changed")
        seed_index = int(training.get("active_seed_index", -1))
        if not 0 <= seed_index < len(expected_seeds):
            errors.append("active_seed_index is outside the registered seed set")
        elif int(training.get("initialization_seed", -1)) != expected_seeds[seed_index]:
            errors.append("active initialization seed does not match seed registry")
        if training.get("data_augmentation") != "none":
            errors.append("main neural budget v1 must use no data augmentation")
        artifact_policy = _mapping_section(config, "artifact_policy")
        if artifact_policy.get("fail_if_output_nonempty") is not True:
            errors.append("main output must refuse to overwrite a non-empty directory")
    if (
        evaluation.get("unknown_score_direction"),
        evaluation.get("threshold_source"),
    ) != ("larger_is_more_unknown", "known_validation_only"):
        errors.append("unknown score direction or threshold source has changed")
    if list(evaluation.get("scores", [])) != ["msp", "energy"]:
        errors.append("B0 smoke must evaluate MSP and Energy")
    if errors:
        raise DataConfigError("Invalid B0 smoke config:\n- " + "\n- ".join(errors))
    return config


def _set_determinism(seed: int, deterministic_algorithms: bool) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms)


def _resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise DataValidationError("CUDA was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _row_indices(
    bundle: ProcessedBundle,
    predicate,
) -> list[int]:
    return [
        int(row["processed_row_index"])
        for row in bundle.rows
        if predicate(row)
    ]


def _fit_normalization(
    bundle: ProcessedBundle,
    train_indices: list[int],
    epsilon: float,
) -> ScalarNormalization:
    if not train_indices:
        raise DataValidationError("known training index set is empty")
    values = np.asarray(bundle.profiles[train_indices], dtype=np.float64)
    mean = float(np.mean(values, dtype=np.float64))
    std = float(np.std(values, dtype=np.float64))
    if not np.isfinite(mean) or not np.isfinite(std) or std <= epsilon:
        raise DataValidationError("known-training normalization statistics are invalid")
    return ScalarNormalization(
        mean=mean,
        std=std,
        fitted_sample_count=len(train_indices),
        fitted_value_count=int(values.size),
    )


def _labels_for_indices(
    bundle: ProcessedBundle,
    indices: list[int],
    class_to_index: Mapping[str, int],
) -> list[int]:
    labels: list[int] = []
    for index in indices:
        class_name = str(bundle.rows[index]["class_name"])
        if class_name not in class_to_index:
            raise DataValidationError("unknown class entered a supervised dataset")
        labels.append(class_to_index[class_name])
    return labels


def _evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    loss_function = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    count = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            total_loss += float(loss_function(logits, labels).item())
            correct += int(torch.count_nonzero(logits.argmax(dim=1) == labels).item())
            count += int(labels.numel())
    return total_loss / count, correct / count


def _infer_all_base_logits(
    model: nn.Module,
    bundle: ProcessedBundle,
    indices: list[int],
    normalization: ScalarNormalization,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    dummy_labels = [0] * len(indices)
    dataset = ProfileDataset(bundle, indices, dummy_labels, normalization)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    logits_by_sample: dict[str, np.ndarray] = {}
    cursor = 0
    model.eval()
    with torch.no_grad():
        for inputs, _ in loader:
            logits = model(inputs.to(device)).cpu().numpy().astype(np.float64, copy=False)
            for batch_index in range(logits.shape[0]):
                row_index = indices[cursor + batch_index]
                sample_id = str(bundle.rows[row_index]["sample_id"])
                logits_by_sample[sample_id] = logits[batch_index].copy()
            cursor += logits.shape[0]
    if len(logits_by_sample) != len(indices):
        raise DataValidationError("base-logit inference did not cover every requested sample")
    return logits_by_sample


def _set_arrays(
    sets: tuple[ViewSet, ...],
    logits_by_sample: Mapping[str, np.ndarray],
    class_to_index: Mapping[str, int],
    selection_seed: int,
    selection_repeat: int,
) -> tuple[np.ndarray, np.ndarray, list[Any]]:
    logits: list[np.ndarray] = []
    labels: list[int] = []
    selections = []
    for item in sets:
        selection = select_b0_single_view(
            item,
            base_seed=selection_seed,
            selection_repeat=selection_repeat,
        )
        logits.append(logits_by_sample[selection.selected_sample_id])
        labels.append(class_to_index.get(item.class_name, len(class_to_index)))
        selections.append(selection)
    return np.stack(logits), np.asarray(labels, dtype=int), selections


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    included: list[Path] = []
    for relative_root in (Path("src"), Path("configs"), Path("tests")):
        candidate = root / relative_root
        if candidate.is_dir():
            included.extend(
                path
                for path in candidate.rglob("*")
                if path.is_file() and path.suffix in {".py", ".yaml"}
            )
    included.extend(
        path
        for path in (root / "pyproject.toml", root / "AGENTS.md", root / "RESEARCH_CONTEXT.md")
        if path.is_file()
    )
    for path in sorted(included):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git_state(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _complexity() -> dict[str, int | str]:
    macs = (
        601 * 32 * 1 * 7
        + 300 * 64 * 32 * 5
        + 150 * 128 * 64 * 3
        + 128 * 7
    )
    return {
        "multiply_accumulates_per_profile": macs,
        "approximate_flops_per_profile": 2 * macs,
        "convention": "conv_and_linear_only_one_multiply_plus_one_add_equals_two_flops",
    }


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure_single_profile_latency(
    model: nn.Module,
    bundle: ProcessedBundle,
    row_index: int,
    normalization: ScalarNormalization,
    device: torch.device,
    *,
    warmup_iterations: int = 20,
    measured_iterations: int = 100,
) -> dict[str, Any]:
    profile = np.asarray(bundle.profiles[row_index], dtype=np.float32)
    normalized = (profile - normalization.mean) / normalization.std
    inputs = torch.from_numpy(np.asarray(normalized, dtype=np.float32)).unsqueeze(0)
    inputs = inputs.to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup_iterations):
            model(inputs)
        _synchronize_device(device)
        started = time.perf_counter()
        for _ in range(measured_iterations):
            model(inputs)
        _synchronize_device(device)
    elapsed = time.perf_counter() - started
    return {
        "device": str(device),
        "batch_size": 1,
        "warmup_iterations": warmup_iterations,
        "measured_iterations": measured_iterations,
        "total_seconds": elapsed,
        "mean_seconds_per_profile": elapsed / measured_iterations,
        "scope": "model_forward_only_single_selected_view",
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_b0(
    config_path: str | Path,
    bundle_root: str | Path,
    output_dir: str | Path,
    *,
    device_request: str = "auto",
    expected_run_kind: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    config_path = Path(config_path).resolve()
    config = load_b0_config(config_path)
    if config.get("run_kind") != expected_run_kind:
        raise DataConfigError(
            f"expected run_kind={expected_run_kind}, got {config.get('run_kind')}"
        )
    data_config = _mapping_section(config, "data")
    training_config = _mapping_section(config, "training")
    normalization_config = _mapping_section(config, "normalization")
    set_config = _mapping_section(config, "set_protocol")
    selection_config = _mapping_section(config, "b0_view_selection")
    evaluation_config = _mapping_section(config, "evaluation")
    destination = Path(output_dir).resolve()
    if (
        config.get("run_kind") == "main"
        and destination.exists()
        and any(destination.iterdir())
    ):
        raise DataValidationError(
            f"main output directory is not empty and will not be overwritten: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    bundle = load_processed_bundle(
        bundle_root,
        expected_profiles_sha256=str(data_config["profiles_sha256"]),
        expected_manifest_sha256=str(data_config["processed_manifest_sha256"]),
        expected_bundle_sha256=str(data_config["bundle_sha256"]),
    )
    class_order = tuple(config["classes"]["known_order"])
    if bundle.known_classes != class_order:
        raise DataValidationError("processed bundle known classes do not match config order")
    class_to_index = {name: index for index, name in enumerate(class_order)}

    train_indices = _row_indices(
        bundle,
        lambda row: int(row["eligible_for_training"]) == 1,
    )
    validation_indices = _row_indices(
        bundle,
        lambda row: int(row["eligible_for_validation"]) == 1,
    )
    test_indices = _row_indices(
        bundle,
        lambda row: int(row["eligible_for_evaluation"]) == 1,
    )
    if (len(train_indices), len(validation_indices), len(test_indices)) != (
        1512,
        504,
        720,
    ):
        raise DataValidationError("B0 train/validation/test base-pool counts are invalid")
    normalization = _fit_normalization(
        bundle,
        train_indices,
        epsilon=float(normalization_config["epsilon"]),
    )
    train_labels = _labels_for_indices(bundle, train_indices, class_to_index)
    validation_labels = _labels_for_indices(bundle, validation_indices, class_to_index)

    seed = int(training_config["initialization_seed"])
    _set_determinism(seed, bool(training_config["deterministic_algorithms"]))
    device = _resolve_device(device_request)
    model = HRRPClassifier1D().to(device)
    train_dataset = ProfileDataset(bundle, train_indices, train_labels, normalization)
    validation_dataset = ProfileDataset(
        bundle,
        validation_indices,
        validation_labels,
        normalization,
    )
    generator = torch.Generator()
    generator.manual_seed(int(training_config["dataloader_seed"]))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=True,
        num_workers=int(training_config["num_workers"]),
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    loss_function = nn.CrossEntropyLoss()
    best_validation_accuracy = -np.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    training_log: list[dict[str, Any]] = []
    epochs_without_improvement = 0
    early_stopping_enabled = bool(training_config["early_stopping"])
    early_stopping_patience = int(
        training_config.get("early_stopping_patience", int(training_config["epochs"]))
    )
    early_stopping_min_delta = float(
        training_config.get("early_stopping_min_delta", 0.0)
    )
    stopped_early = False
    for epoch in range(int(training_config["epochs"])):
        model.train()
        total_loss = 0.0
        correct = 0
        count = 0
        epoch_started = time.perf_counter()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * labels.numel()
            correct += int(torch.count_nonzero(logits.argmax(dim=1) == labels).item())
            count += int(labels.numel())
        validation_loss, validation_accuracy = _evaluate_loader(
            model, validation_loader, device
        )
        record = {
            "epoch": epoch + 1,
            "train_loss": total_loss / count,
            "train_accuracy": correct / count,
            "known_validation_loss": validation_loss,
            "known_validation_accuracy": validation_accuracy,
            "elapsed_seconds": time.perf_counter() - epoch_started,
        }
        training_log.append(record)
        if validation_accuracy > best_validation_accuracy + early_stopping_min_delta:
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        record["epochs_without_improvement"] = epochs_without_improvement
        if early_stopping_enabled and epochs_without_improvement >= early_stopping_patience:
            stopped_early = True
            record["early_stopping_triggered"] = True
            break
    if best_state is None:
        raise DataValidationError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)

    checkpoint_path = destination / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "model_architecture": HRRPClassifier1D().encoder.architecture_id,
            "class_order": class_order,
            "normalization": normalization.__dict__,
            "best_epoch": best_epoch,
            "best_known_validation_accuracy": best_validation_accuracy,
            "stopped_epoch": len(training_log),
            "stopped_early": stopped_early,
            "input_bundle_sha256": data_config["bundle_sha256"],
            "config": config,
        },
        checkpoint_path,
    )

    validation_sets = build_v3_evaluation_sets(
        bundle.rows,
        split="validation",
        base_seed=int(set_config["set_seed"]),
        set_repeat=int(set_config["set_repeat"]),
    )
    test_sets = build_v3_evaluation_sets(
        bundle.rows,
        split="test",
        base_seed=int(set_config["set_seed"]),
        set_repeat=int(set_config["set_repeat"]),
    )
    set_manifest_bytes = render_set_manifest_csv((*validation_sets, *test_sets))
    (destination / "set_manifest.csv").write_bytes(set_manifest_bytes)
    set_manifest_hash = hashlib.sha256(set_manifest_bytes).hexdigest()

    inference_indices = sorted(set(validation_indices + test_indices))
    logits_by_sample = _infer_all_base_logits(
        model,
        bundle,
        inference_indices,
        normalization,
        device,
        batch_size=int(training_config["batch_size"]),
    )
    score_metric_repeats: dict[str, list[dict[str, float]]] = {
        "msp": [],
        "energy": [],
    }
    prediction_rows: list[dict[str, Any]] = []
    selection_seed = int(selection_config["seed"])
    selection_repeat_count = int(selection_config["repeats"])
    energy_temperature = float(evaluation_config["energy_temperature"])
    acceptance_rate = float(evaluation_config["threshold_known_acceptance_rate"])
    for selection_repeat in range(selection_repeat_count):
        validation_logits, validation_labels_for_sets, validation_selections = _set_arrays(
            validation_sets,
            logits_by_sample,
            class_to_index,
            selection_seed,
            selection_repeat,
        )
        test_logits, test_labels, test_selections = _set_arrays(
            test_sets,
            logits_by_sample,
            class_to_index,
            selection_seed,
            selection_repeat,
        )
        validation_scores = {
            "msp": msp_unknown_score(validation_logits),
            "energy": energy_unknown_score(
                validation_logits, temperature=energy_temperature
            ),
        }
        test_scores = {
            "msp": msp_unknown_score(test_logits),
            "energy": energy_unknown_score(test_logits, temperature=energy_temperature),
        }
        test_predictions = np.argmax(test_logits, axis=1)
        validation_predictions = np.argmax(validation_logits, axis=1)
        known_mask = test_labels < len(class_order)
        unknown_mask = ~known_mask
        thresholds = {
            score_name: threshold_for_known_acceptance(scores, acceptance_rate)
            for score_name, scores in validation_scores.items()
        }
        for score_name in ("msp", "energy"):
            metrics = evaluate_open_set(
                known_true=test_labels[known_mask],
                known_pred=test_predictions[known_mask],
                known_unknown_scores=test_scores[score_name][known_mask],
                unknown_pred=test_predictions[unknown_mask],
                unknown_unknown_scores=test_scores[score_name][unknown_mask],
                known_validation_scores=validation_scores[score_name],
                known_class_count=len(class_order),
                known_acceptance_rate=acceptance_rate,
            )
            score_metric_repeats[score_name].append(metrics)

        for item, selection, logits, true_label, prediction, msp, energy in zip(
            validation_sets,
            validation_selections,
            validation_logits,
            validation_labels_for_sets,
            validation_predictions,
            validation_scores["msp"],
            validation_scores["energy"],
            strict=True,
        ):
            prediction_rows.append(
                {
                    "selection_repeat": selection_repeat,
                    "set_id": item.set_id,
                    "split": "validation",
                    "class_name": item.class_name,
                    "class_role": item.class_role,
                    "true_label": int(true_label),
                    "selected_sample_id": selection.selected_sample_id,
                    "selected_index": selection.selected_index,
                    "predicted_known_label": int(prediction),
                    "predicted_known_class": class_order[int(prediction)],
                    "logits": json.dumps(logits.tolist(), separators=(",", ":")),
                    "msp_unknown_score": float(msp),
                    "energy_unknown_score": float(energy),
                    "msp_threshold": thresholds["msp"],
                    "energy_threshold": thresholds["energy"],
                    "msp_rejected": int(msp > thresholds["msp"]),
                    "energy_rejected": int(energy > thresholds["energy"]),
                }
            )

        for item, selection, logits, true_label, prediction, msp, energy in zip(
            test_sets,
            test_selections,
            test_logits,
            test_labels,
            test_predictions,
            test_scores["msp"],
            test_scores["energy"],
            strict=True,
        ):
            prediction_rows.append(
                {
                    "selection_repeat": selection_repeat,
                    "set_id": item.set_id,
                    "split": "test",
                    "class_name": item.class_name,
                    "class_role": item.class_role,
                    "true_label": int(true_label),
                    "selected_sample_id": selection.selected_sample_id,
                    "selected_index": selection.selected_index,
                    "predicted_known_label": int(prediction),
                    "predicted_known_class": class_order[int(prediction)],
                    "logits": json.dumps(logits.tolist(), separators=(",", ":")),
                    "msp_unknown_score": float(msp),
                    "energy_unknown_score": float(energy),
                    "msp_threshold": thresholds["msp"],
                    "energy_threshold": thresholds["energy"],
                    "msp_rejected": int(msp > thresholds["msp"]),
                    "energy_rejected": int(energy > thresholds["energy"]),
                }
            )

    run_kind = str(config["run_kind"])
    metrics_document = {
        "run_kind": run_kind,
        "result_scope": str(config["result_scope"]),
        "not_for_main_result_table": run_kind != "main",
        "best_epoch": best_epoch,
        "best_known_validation_accuracy": best_validation_accuracy,
        "stopped_epoch": len(training_log),
        "stopped_early": stopped_early,
        "per_selection_repeat": score_metric_repeats,
        "aggregate_across_selection_repeats": {
            score_name: summarize_metric_repeats(repeats)
            for score_name, repeats in score_metric_repeats.items()
        },
        "threshold_source": "known_validation_only",
        "unknown_score_direction": "larger_is_more_unknown",
        "accepted_risks": [
            {
                "risk_id": data_config["accepted_risk_id"],
                "status": data_config["accepted_risk_status"],
            }
        ],
    }
    (destination / "metrics.json").write_text(
        json.dumps(metrics_document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (destination / "predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    (destination / "training_log.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in training_log),
        encoding="utf-8",
    )
    (destination / "normalization.json").write_text(
        json.dumps(normalization.__dict__, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    project_root = config_path.parents[3]
    environment = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor()
        ),
        "git": _git_state(project_root),
        "code_tree_sha256": _tree_hash(project_root),
    }
    (destination / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    resolved_config = copy.deepcopy(config)
    resolved_config["_resolved"] = {
        "bundle_root": str(bundle.root),
        "output_dir": str(destination),
        "device": str(device),
        "known_class_to_index": class_to_index,
        "normalization": normalization.__dict__,
        "set_manifest_sha256": set_manifest_hash,
        "checkpoint": str(checkpoint_path),
        "checkpoint_selected_epoch": best_epoch,
        "code_tree_sha256": environment["code_tree_sha256"],
    }
    (destination / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    complexity = {
        "parameter_count": model.parameter_count,
        **_complexity(),
    }
    (destination / "complexity.json").write_text(
        json.dumps(complexity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    inference_timing = _measure_single_profile_latency(
        model,
        bundle,
        test_indices[0],
        normalization,
        device,
    )
    (destination / "inference_timing.json").write_text(
        json.dumps(inference_timing, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifact_names = [
        "checkpoint.pt",
        "complexity.json",
        "environment.json",
        "inference_timing.json",
        "metrics.json",
        "normalization.json",
        "predictions.csv",
        "resolved_config.yaml",
        "set_manifest.csv",
        "training_log.jsonl",
    ]
    artifact_manifest = {
        "config_source": str(config_path),
        "config_source_sha256": _sha256_file(config_path),
        "config_sources": [
            {"path": str(path), "sha256": _sha256_file(path)}
            for path in _config_source_paths(config_path)
        ],
        "data_bundle_sha256": str(data_config["bundle_sha256"]),
        "artifacts": {
            name: _sha256_file(destination / name) for name in artifact_names
        },
    }
    (destination / "artifact_manifest.json").write_text(
        json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    elapsed = time.perf_counter() - started
    return {
        "validation_status": "passed",
        "run_kind": run_kind,
        "output_dir": str(destination),
        "device": str(device),
        "train_sample_count": len(train_indices),
        "known_validation_sample_count": len(validation_indices),
        "evaluation_set_count": len(test_sets),
        "selection_repeats": selection_repeat_count,
        "best_epoch": best_epoch,
        "best_known_validation_accuracy": best_validation_accuracy,
        "stopped_epoch": len(training_log),
        "stopped_early": stopped_early,
        "parameter_count": model.parameter_count,
        "set_manifest_sha256": set_manifest_hash,
        "elapsed_seconds": elapsed,
        "accepted_risk_id": data_config["accepted_risk_id"],
        "metrics": metrics_document["aggregate_across_selection_repeats"],
    }


def run_b0_smoke(
    config_path: str | Path,
    bundle_root: str | Path,
    output_dir: str | Path,
    *,
    device_request: str = "auto",
) -> dict[str, Any]:
    return _run_b0(
        config_path,
        bundle_root,
        output_dir,
        device_request=device_request,
        expected_run_kind="diagnostic_smoke",
    )


def run_b0_main(
    config_path: str | Path,
    bundle_root: str | Path,
    output_dir: str | Path,
    *,
    device_request: str = "auto",
) -> dict[str, Any]:
    return _run_b0(
        config_path,
        bundle_root,
        output_dir,
        device_request=device_request,
        expected_run_kind="main",
    )
