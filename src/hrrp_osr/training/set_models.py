from __future__ import annotations

import csv
import hashlib
import itertools
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from hrrp_osr.baselines.b0 import energy_unknown_score, msp_unknown_score
from hrrp_osr.data.errors import DataConfigError, DataValidationError
from hrrp_osr.data.processed import ProcessedBundle, load_processed_bundle
from hrrp_osr.data.sets import SET_ALGORITHM_VERSION, ViewSet, build_v3_sets, render_set_manifest_csv
from hrrp_osr.evaluation.metrics import evaluate_open_set, threshold_for_known_acceptance
from hrrp_osr.models.sets import DeepSetsClassifier, SetTransformerClassifier
from hrrp_osr.training.b0_smoke import (
    EXPECTED_KNOWN_CLASSES,
    ScalarNormalization,
    _fit_normalization,
    _git_state,
    _resolve_device,
    _row_indices,
    _set_determinism,
    _sha256_file,
    _synchronize_device,
    _tree_hash,
)


class ViewSetDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        bundle: ProcessedBundle,
        sets: Sequence[ViewSet],
        normalization: ScalarNormalization,
        class_to_index: Mapping[str, int],
        permutation: tuple[int, ...] | None = None,
    ) -> None:
        self.bundle = bundle
        self.sets = tuple(sets)
        self.normalization = normalization
        self.class_to_index = class_to_index
        self.permutation = permutation
        self.sample_to_row = {
            str(row["sample_id"]): int(row["processed_row_index"])
            for row in bundle.rows
        }

    def __len__(self) -> int:
        return len(self.sets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.sets[index]
        member_ids = item.member_sample_ids
        if self.permutation is not None:
            member_ids = tuple(member_ids[position] for position in self.permutation)
        profiles = np.stack([
            np.asarray(self.bundle.profiles[self.sample_to_row[sample_id]], dtype=np.float32)
            for sample_id in member_ids
        ])
        profiles = (profiles - self.normalization.mean) / self.normalization.std
        label = self.class_to_index.get(item.class_name, len(self.class_to_index))
        return torch.from_numpy(np.asarray(profiles, dtype=np.float32)), torch.tensor(label, dtype=torch.long)


def load_set_model_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise DataConfigError("set-model config must be a mapping")
    config = dict(raw)
    errors: list[str] = []
    baseline = config.get("baseline")
    expected_architecture = {"B2": "deep_sets_mean_v1", "B3": "set_transformer_sab2_pma_v1"}.get(baseline)
    if config.get("stage") != "P1" or expected_architecture is None:
        errors.append("set model must be P1/B2 or P1/B3")
    if config.get("result_scope") != "main_v3":
        errors.append("set model scope must be main_v3")
    if tuple(config["classes"]["known_order"]) != EXPECTED_KNOWN_CLASSES:
        errors.append("known class order changed")
    if config["model"]["architecture"] != expected_architecture:
        errors.append("baseline architecture mismatch")
    if config["model"]["encoder_architecture"] != "shared_hrrp_encoder_1d_v1" or config["model"]["angle_or_position_encoding"] is not False:
        errors.append("shared angle-free encoder contract changed")
    if (int(config["set_protocol"]["view_count"]), config["set_protocol"]["set_algorithm_version"]) != (3, SET_ALGORITHM_VERSION):
        errors.append("main V=3 set protocol changed")
    training = config["training"]
    if (
        training["budget_id"], int(training["epochs"]), int(training["early_stopping_patience"]),
        training["data_augmentation"], training["unknown_data_used"]
    ) != ("neural_budget_v1", 100, 15, "none", False):
        errors.append("shared neural budget changed")
    if list(training["initialization_seeds"]) != [20260810, 20260820, 20260830, 20260840, 20260850]:
        errors.append("model seed registry changed")
    if errors:
        raise DataConfigError("Invalid set-model config:\n- " + "\n- ".join(errors))
    return config


def _make_model(baseline: str) -> nn.Module:
    if baseline == "B2":
        return DeepSetsClassifier()
    if baseline == "B3":
        return SetTransformerClassifier()
    raise DataConfigError(f"unsupported set baseline {baseline}")


def _infer_sets(
    model: nn.Module,
    dataset: ViewSetDataset,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for inputs, batch_labels in loader:
            logits.append(model(inputs.to(device)).cpu().numpy().astype(np.float64))
            labels.append(batch_labels.numpy())
    return np.concatenate(logits), np.concatenate(labels).astype(int)


def _permutation_audit(
    model: nn.Module,
    bundle: ProcessedBundle,
    sets: Sequence[ViewSet],
    normalization: ScalarNormalization,
    class_to_index: Mapping[str, int],
    device: torch.device,
    batch_size: int,
    atol: float,
) -> dict[str, Any]:
    reference, _ = _infer_sets(
        model, ViewSetDataset(bundle, sets, normalization, class_to_index), device, batch_size
    )
    maximum = 0.0
    for permutation in itertools.permutations(range(3)):
        actual, _ = _infer_sets(
            model,
            ViewSetDataset(bundle, sets, normalization, class_to_index, permutation=permutation),
            device,
            batch_size,
        )
        maximum = max(maximum, float(np.max(np.abs(actual - reference))))
    if maximum > atol:
        raise DataValidationError(f"set-model permutation audit failed: {maximum} > {atol}")
    return {"status": "passed", "set_count": len(sets), "permutations_per_set": 6, "atol": atol, "maximum_absolute_logit_difference": maximum}


def _complexity(baseline: str, parameter_count: int) -> dict[str, Any]:
    encoder_macs_one = 601 * 32 * 7 + 300 * 64 * 32 * 5 + 150 * 128 * 64 * 3
    if baseline == "B2":
        extra_macs = 128 * 7
    else:
        # Two SAB plus one PMA: Q/K/V/out projections and two-layer FFNs.
        attention_block = 4 * 128 * 128 + 2 * 128 * 256
        extra_macs = 2 * 3 * attention_block + 128 * 7
    macs = 3 * encoder_macs_one + extra_macs
    return {"parameter_count": parameter_count, "multiply_accumulates_per_v3_set": macs, "approximate_flops_per_v3_set": 2 * macs, "convention": "conv_linear_attention_projection_only"}


def run_set_model_all(
    config_path: str | Path,
    bundle_root: str | Path,
    output_root: str | Path,
    *,
    device_request: str = "auto",
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = load_set_model_config(config_path)
    baseline = str(config["baseline"])
    bundle = load_processed_bundle(
        bundle_root,
        expected_profiles_sha256=config["data"]["profiles_sha256"],
        expected_manifest_sha256=config["data"]["processed_manifest_sha256"],
        expected_bundle_sha256=config["data"]["bundle_sha256"],
    )
    class_order = tuple(config["classes"]["known_order"])
    class_to_index = {name: index for index, name in enumerate(class_order)}
    protocol = config["set_protocol"]
    sets = {
        split: build_v3_sets(bundle.rows, split=split, base_seed=int(protocol["set_seed"]), set_repeat=int(protocol["set_repeat"]))
        for split in ("train", "validation", "test")
    }
    manifest = render_set_manifest_csv((*sets["train"], *sets["validation"], *sets["test"]))
    manifest_sha = hashlib.sha256(manifest).hexdigest()
    train_indices = _row_indices(bundle, lambda row: int(row["eligible_for_training"]) == 1)
    normalization = _fit_normalization(bundle, train_indices, float(config["normalization"]["epsilon"]))
    device = _resolve_device(device_request)
    root = Path(output_root).resolve()
    seed_results: dict[str, Any] = {}
    for seed in config["training"]["initialization_seeds"]:
        destination = root / f"seed_{seed}"
        if destination.exists() and any(destination.iterdir()):
            raise DataValidationError(f"{baseline} output is non-empty: {destination}")
        destination.mkdir(parents=True, exist_ok=True)
        _set_determinism(int(seed), bool(config["training"]["deterministic_algorithms"]))
        model = _make_model(baseline).to(device)
        datasets = {
            split: ViewSetDataset(bundle, split_sets, normalization, class_to_index)
            for split, split_sets in sets.items()
        }
        generator = torch.Generator().manual_seed(int(seed) + 1)
        train_loader = DataLoader(datasets["train"], batch_size=int(config["training"]["batch_size"]), shuffle=True, generator=generator, num_workers=0)
        val_loader = DataLoader(datasets["validation"], batch_size=int(config["training"]["batch_size"]), shuffle=False, num_workers=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
        loss_fn = nn.CrossEntropyLoss()
        best_accuracy = -np.inf; best_epoch = -1; best_state = None; no_improvement = 0; log = []
        for epoch in range(int(config["training"]["epochs"])):
            started = time.perf_counter(); model.train(); total_loss = 0.0; correct = count = 0
            for inputs, labels in train_loader:
                inputs = inputs.to(device); labels = labels.to(device)
                optimizer.zero_grad(set_to_none=True); logits = model(inputs); loss = loss_fn(logits, labels); loss.backward(); optimizer.step()
                total_loss += float(loss.item()) * labels.numel(); correct += int((logits.argmax(1) == labels).sum().item()); count += labels.numel()
            val_logits, val_labels = _infer_sets(model, datasets["validation"], device, int(config["training"]["batch_size"]))
            val_accuracy = float(np.mean(val_logits.argmax(1) == val_labels))
            record = {"epoch": epoch + 1, "train_loss": total_loss / count, "train_accuracy": correct / count, "known_validation_accuracy": val_accuracy, "elapsed_seconds": time.perf_counter() - started}
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy; best_epoch = epoch + 1; best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}; no_improvement = 0
            else:
                no_improvement += 1
            record["epochs_without_improvement"] = no_improvement; log.append(record)
            if no_improvement >= int(config["training"]["early_stopping_patience"]):
                record["early_stopping_triggered"] = True; break
        if best_state is None:
            raise DataValidationError(f"{baseline} training produced no checkpoint")
        model.load_state_dict(best_state); model.to(device); model.eval()
        checkpoint = {"model_state_dict": best_state, "architecture": model.architecture_id, "class_order": class_order, "normalization": normalization.__dict__, "best_epoch": best_epoch, "best_known_validation_accuracy": best_accuracy, "stopped_epoch": len(log), "input_bundle_sha256": config["data"]["bundle_sha256"], "config": config}
        torch.save(checkpoint, destination / "checkpoint.pt")
        val_logits, val_labels = _infer_sets(model, datasets["validation"], device, 64)
        test_logits, test_labels = _infer_sets(model, datasets["test"], device, 64)
        val_scores = {"msp": msp_unknown_score(val_logits), "energy": energy_unknown_score(val_logits, float(config["evaluation"]["energy_temperature"]))}
        test_scores = {"msp": msp_unknown_score(test_logits), "energy": energy_unknown_score(test_logits, float(config["evaluation"]["energy_temperature"]))}
        test_predictions = test_logits.argmax(1); val_predictions = val_logits.argmax(1); known_mask = test_labels < len(class_order); acceptance = float(config["evaluation"]["threshold_known_acceptance_rate"])
        metrics = {score: evaluate_open_set(known_true=test_labels[known_mask], known_pred=test_predictions[known_mask], known_unknown_scores=values[known_mask], unknown_pred=test_predictions[~known_mask], unknown_unknown_scores=values[~known_mask], known_validation_scores=val_scores[score], known_class_count=len(class_order), known_acceptance_rate=acceptance) for score, values in test_scores.items()}
        audit = _permutation_audit(model, bundle, (*sets["validation"], *sets["test"]), normalization, class_to_index, device, 64, float(config["evaluation"]["permutation_atol"]))
        rows = []
        for split, split_sets, logits_values, labels, predictions, score_map in (("validation", sets["validation"], val_logits, val_labels, val_predictions, val_scores), ("test", sets["test"], test_logits, test_labels, test_predictions, test_scores)):
            thresholds = {score: threshold_for_known_acceptance(val_scores[score], acceptance) for score in val_scores}
            for item, logits, label, prediction, msp, energy in zip(split_sets, logits_values, labels, predictions, score_map["msp"], score_map["energy"], strict=True):
                rows.append({"set_id": item.set_id, "split": split, "class_name": item.class_name, "class_role": item.class_role, "true_label": int(label), "predicted_known_label": int(prediction), "predicted_known_class": class_order[int(prediction)], "logits": json.dumps(logits.tolist(), separators=(",", ":")), "msp_unknown_score": float(msp), "energy_unknown_score": float(energy), "msp_threshold": thresholds["msp"], "energy_threshold": thresholds["energy"]})
        with (destination / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        (destination / "set_manifest.csv").write_bytes(manifest)
        (destination / "training_log.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in log), encoding="utf-8")
        (destination / "normalization.json").write_text(json.dumps(normalization.__dict__, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (destination / "permutation_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        metrics_doc = {"stage": "P1", "baseline": baseline, "result_scope": "main_v3", "model_seed": seed, "best_epoch": best_epoch, "best_known_validation_accuracy": best_accuracy, "stopped_epoch": len(log), "metrics": metrics, "threshold_source": "known_validation_only", "accepted_risks": [{"risk_id": config["data"]["accepted_risk_id"], "status": config["data"]["accepted_risk_status"]}]}
        (destination / "metrics.json").write_text(json.dumps(metrics_doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        complexity = _complexity(baseline, model.parameter_count)
        (destination / "complexity.json").write_text(json.dumps(complexity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        sample_inputs, _ = datasets["test"][0]; sample_inputs = sample_inputs.unsqueeze(0).to(device)
        with torch.no_grad():
            for _ in range(20): model(sample_inputs)
            _synchronize_device(device); latency_started = time.perf_counter()
            for _ in range(100): model(sample_inputs)
            _synchronize_device(device)
        timing = {"device": str(device), "batch_size": 1, "view_count": 3, "mean_seconds_per_set": (time.perf_counter() - latency_started) / 100}
        (destination / "inference_timing.json").write_text(json.dumps(timing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        project_root = config_path.parents[3]
        environment = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "torch": torch.__version__, "device": str(device), "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda, "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(), "git": _git_state(project_root), "code_tree_sha256": _tree_hash(project_root)}
        (destination / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        resolved = dict(config); resolved["_resolved"] = {"bundle_root": str(bundle.root), "set_manifest_sha256": manifest_sha, "device": str(device), "initialization_seed": seed, "dataloader_seed": seed + 1}
        (destination / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False), encoding="utf-8")
        artifact_names = ["checkpoint.pt", "complexity.json", "environment.json", "inference_timing.json", "metrics.json", "normalization.json", "permutation_audit.json", "predictions.csv", "resolved_config.yaml", "set_manifest.csv", "training_log.jsonl"]
        (destination / "artifact_manifest.json").write_text(json.dumps({"artifacts": {name: _sha256_file(destination / name) for name in artifact_names}}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        seed_results[str(seed)] = {"best_epoch": best_epoch, "best_known_validation_accuracy": best_accuracy, "stopped_epoch": len(log), "metrics": metrics, "permutation_max_abs": audit["maximum_absolute_logit_difference"], "checkpoint_sha256": _sha256_file(destination / "checkpoint.pt")}
    return {"validation_status": "passed", "baseline": baseline, "seed_results": seed_results}
