from __future__ import annotations

import csv
import hashlib
import itertools
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from hrrp_osr.baselines.b0 import energy_unknown_score, softmax_probabilities
from hrrp_osr.data.errors import DataConfigError, DataValidationError
from hrrp_osr.data.processed import load_processed_bundle
from hrrp_osr.data.sets import SET_ALGORITHM_VERSION, ViewSet, build_v3_evaluation_sets, render_set_manifest_csv
from hrrp_osr.evaluation.metrics import evaluate_open_set, threshold_for_known_acceptance
from hrrp_osr.models.cnn1d import HRRPClassifier1D
from hrrp_osr.training.b0_smoke import (
    EXPECTED_KNOWN_CLASSES,
    ScalarNormalization,
    _git_state,
    _infer_all_base_logits,
    _resolve_device,
    _row_indices,
    _sha256_file,
    _synchronize_device,
    _tree_hash,
)


def load_b1_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise DataConfigError("B1 config must be a mapping")
    config = dict(raw)
    errors: list[str] = []
    if (config.get("stage"), config.get("baseline"), config.get("result_scope")) != (
        "P1", "B1", "main_v3"
    ):
        errors.append("B1 must remain P1/main_v3")
    if tuple(config["classes"]["known_order"]) != EXPECTED_KNOWN_CLASSES:
        errors.append("known class order changed")
    protocol = config["set_protocol"]
    if (
        int(protocol["view_count"]),
        protocol["set_algorithm_version"],
        bool(protocol["angle_or_position_inputs"]),
    ) != (3, SET_ALGORITHM_VERSION, False):
        errors.append("B1 V=3 angle-free set protocol changed")
    reuse = config["checkpoint_reuse"]
    if reuse["source_baseline"] != "B0" or reuse["training_allowed"] is not False:
        errors.append("B1 must reuse B0 without training")
    if list(reuse["seeds"]) != [20260810, 20260820, 20260830, 20260840, 20260850]:
        errors.append("B1 seed registry changed")
    fusion = config["fusion"]
    if fusion != {
        "msp": "mean_per_view_known_posterior_then_msp",
        "energy": "mean_per_view_energy_score",
        "mean_logits_before_energy": False,
    }:
        errors.append("B1 fusion definition changed")
    if errors:
        raise DataConfigError("Invalid B1 config:\n- " + "\n- ".join(errors))
    return config


def _fuse_member_logits(member_logits: np.ndarray, temperature: float) -> tuple[np.ndarray, float, float]:
    if member_logits.ndim != 2 or member_logits.shape[0] != 3:
        raise DataValidationError("B1 requires exactly three member logits")
    mean_posterior = softmax_probabilities(member_logits).mean(axis=0)
    msp_score = float(1.0 - np.max(mean_posterior))
    energy_score = float(np.mean(energy_unknown_score(member_logits, temperature)))
    return mean_posterior, msp_score, energy_score


def _set_outputs(
    sets: tuple[ViewSet, ...],
    logits_by_sample: Mapping[str, np.ndarray],
    class_to_index: Mapping[str, int],
    temperature: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    posteriors: list[np.ndarray] = []
    msp_scores: list[float] = []
    energy_scores: list[float] = []
    labels: list[int] = []
    for item in sets:
        member_logits = np.stack([logits_by_sample[sample_id] for sample_id in item.member_sample_ids])
        posterior, msp, energy = _fuse_member_logits(member_logits, temperature)
        posteriors.append(posterior)
        msp_scores.append(msp)
        energy_scores.append(energy)
        labels.append(class_to_index.get(item.class_name, len(class_to_index)))
    return (
        np.stack(posteriors),
        np.asarray(msp_scores),
        np.asarray(energy_scores),
        np.asarray(labels, dtype=int),
    )


def _permutation_audit(
    sets: tuple[ViewSet, ...],
    logits_by_sample: Mapping[str, np.ndarray],
    temperature: float,
    atol: float,
) -> dict[str, Any]:
    maximum = 0.0
    for item in sets:
        member_logits = np.stack([logits_by_sample[sample_id] for sample_id in item.member_sample_ids])
        reference = _fuse_member_logits(member_logits, temperature)
        for permutation in itertools.permutations(range(3)):
            actual = _fuse_member_logits(member_logits[list(permutation)], temperature)
            maximum = max(
                maximum,
                float(np.max(np.abs(reference[0] - actual[0]))),
                abs(reference[1] - actual[1]),
                abs(reference[2] - actual[2]),
            )
    if maximum > atol:
        raise DataValidationError(f"B1 permutation audit failed: {maximum} > {atol}")
    return {
        "status": "passed",
        "set_count": len(sets),
        "permutations_per_set": 6,
        "atol": atol,
        "maximum_absolute_difference": maximum,
    }


def run_b1_all(
    config_path: str | Path,
    bundle_root: str | Path,
    b0_root: str | Path,
    output_root: str | Path,
    *,
    device_request: str = "auto",
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = load_b1_config(config_path)
    bundle = load_processed_bundle(
        bundle_root,
        expected_profiles_sha256=config["data"]["profiles_sha256"],
        expected_manifest_sha256=config["data"]["processed_manifest_sha256"],
        expected_bundle_sha256=config["data"]["bundle_sha256"],
    )
    class_order = tuple(config["classes"]["known_order"])
    class_to_index = {name: index for index, name in enumerate(class_order)}
    protocol = config["set_protocol"]
    validation_sets = build_v3_evaluation_sets(
        bundle.rows, split="validation", base_seed=int(protocol["set_seed"]), set_repeat=int(protocol["set_repeat"])
    )
    test_sets = build_v3_evaluation_sets(
        bundle.rows, split="test", base_seed=int(protocol["set_seed"]), set_repeat=int(protocol["set_repeat"])
    )
    set_manifest = render_set_manifest_csv((*validation_sets, *test_sets))
    set_manifest_sha = hashlib.sha256(set_manifest).hexdigest()
    inference_indices = sorted(set(_row_indices(bundle, lambda row: str(row["split"]) in {"validation", "test"})))
    device = _resolve_device(device_request)
    root = Path(output_root).resolve()
    results: dict[str, Any] = {}
    for seed in config["checkpoint_reuse"]["seeds"]:
        destination = root / f"seed_{seed}"
        if destination.exists() and any(destination.iterdir()):
            raise DataValidationError(f"B1 output is non-empty: {destination}")
        destination.mkdir(parents=True, exist_ok=True)
        source = Path(b0_root).resolve() / f"seed_{seed}" / "checkpoint.pt"
        expected_checkpoint_sha = str(config["checkpoint_reuse"]["checkpoint_sha256"][str(seed)])
        if _sha256_file(source) != expected_checkpoint_sha:
            raise DataValidationError(f"B0 checkpoint hash mismatch for seed {seed}")
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        if tuple(checkpoint["class_order"]) != class_order:
            raise DataValidationError("B0 checkpoint class order mismatch")
        normalization = ScalarNormalization(**checkpoint["normalization"])
        if normalization.fit_population != "known_train_only":
            raise DataValidationError("B0 normalization was not fitted on known train")
        model = HRRPClassifier1D().to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        logits_by_sample = _infer_all_base_logits(
            model, bundle, inference_indices, normalization, device, batch_size=64
        )
        temperature = float(config["evaluation"]["energy_temperature"])
        val_p, val_msp, val_energy, val_labels = _set_outputs(
            validation_sets, logits_by_sample, class_to_index, temperature
        )
        test_p, test_msp, test_energy, test_labels = _set_outputs(
            test_sets, logits_by_sample, class_to_index, temperature
        )
        val_predictions = val_p.argmax(axis=1)
        test_predictions = test_p.argmax(axis=1)
        acceptance = float(config["evaluation"]["threshold_known_acceptance_rate"])
        thresholds = {
            "msp": threshold_for_known_acceptance(val_msp, acceptance),
            "energy": threshold_for_known_acceptance(val_energy, acceptance),
        }
        known_mask = test_labels < len(class_order)
        score_arrays = {"msp": test_msp, "energy": test_energy}
        val_score_arrays = {"msp": val_msp, "energy": val_energy}
        metrics = {
            score: evaluate_open_set(
                known_true=test_labels[known_mask],
                known_pred=test_predictions[known_mask],
                known_unknown_scores=values[known_mask],
                unknown_pred=test_predictions[~known_mask],
                unknown_unknown_scores=values[~known_mask],
                known_validation_scores=val_score_arrays[score],
                known_class_count=len(class_order),
                known_acceptance_rate=acceptance,
            )
            for score, values in score_arrays.items()
        }
        audit = _permutation_audit(
            (*validation_sets, *test_sets),
            logits_by_sample,
            temperature,
            float(config["evaluation"]["permutation_atol"]),
        )
        prediction_rows: list[dict[str, Any]] = []
        for split, sets, posteriors, msp, energy, labels, predictions in (
            ("validation", validation_sets, val_p, val_msp, val_energy, val_labels, val_predictions),
            ("test", test_sets, test_p, test_msp, test_energy, test_labels, test_predictions),
        ):
            for item, posterior, msp_value, energy_value, label, prediction in zip(
                sets, posteriors, msp, energy, labels, predictions, strict=True
            ):
                prediction_rows.append({
                    "set_id": item.set_id,
                    "split": split,
                    "class_name": item.class_name,
                    "class_role": item.class_role,
                    "true_label": int(label),
                    "predicted_known_label": int(prediction),
                    "predicted_known_class": class_order[int(prediction)],
                    "mean_known_posterior": json.dumps(posterior.tolist(), separators=(",", ":")),
                    "msp_unknown_score": float(msp_value),
                    "energy_unknown_score": float(energy_value),
                    "msp_threshold": thresholds["msp"],
                    "energy_threshold": thresholds["energy"],
                })
        with (destination / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
            writer.writeheader(); writer.writerows(prediction_rows)
        (destination / "set_manifest.csv").write_bytes(set_manifest)
        metrics_doc = {
            "stage": "P1", "baseline": "B1", "result_scope": "main_v3",
            "model_seed": seed, "metrics": metrics,
            "threshold_source": "known_validation_only",
            "checkpoint_reuse": {"baseline": "B0", "path": str(source), "sha256": expected_checkpoint_sha},
            "accepted_risks": [{"risk_id": config["data"]["accepted_risk_id"], "status": config["data"]["accepted_risk_status"]}],
        }
        (destination / "metrics.json").write_text(json.dumps(metrics_doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (destination / "permutation_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (destination / "checkpoint_reference.json").write_text(json.dumps(metrics_doc["checkpoint_reuse"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        project_root = config_path.parents[3]
        environment = {
            "python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "torch": torch.__version__,
            "device": str(device), "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "git": _git_state(project_root), "code_tree_sha256": _tree_hash(project_root),
        }
        (destination / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        resolved = dict(config)
        resolved["_resolved"] = {"bundle_root": str(bundle.root), "b0_checkpoint": str(source), "b0_checkpoint_sha256": expected_checkpoint_sha, "set_manifest_sha256": set_manifest_sha, "device": str(device)}
        (destination / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False), encoding="utf-8")
        artifact_names = ["checkpoint_reference.json", "environment.json", "metrics.json", "permutation_audit.json", "predictions.csv", "resolved_config.yaml", "set_manifest.csv"]
        artifact_manifest = {"artifacts": {name: _sha256_file(destination / name) for name in artifact_names}}
        (destination / "artifact_manifest.json").write_text(json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        results[str(seed)] = {"status": "passed", "metrics": metrics, "permutation_max_abs": audit["maximum_absolute_difference"]}
    return {"validation_status": "passed", "baseline": "B1", "seed_results": results}
