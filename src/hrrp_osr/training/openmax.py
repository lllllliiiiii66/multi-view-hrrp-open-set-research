from __future__ import annotations

import csv
import hashlib
import itertools
import json
import platform
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import yaml

from hrrp_osr.data.errors import DataConfigError, DataValidationError
from hrrp_osr.data.processed import ProcessedBundle, load_processed_bundle
from hrrp_osr.data.sets import (
    SET_ALGORITHM_VERSION,
    V5_SET_ALGORITHM_VERSION,
    ViewSet,
    build_v3_sets,
    build_v5_sets,
    render_set_manifest_csv,
)
from hrrp_osr.evaluation.metrics import evaluate_open_set, threshold_for_known_acceptance
from hrrp_osr.models.cnn1d import HRRPClassifier1D
from hrrp_osr.models.sets import DeepSetsClassifier
from hrrp_osr.openmax import OpenMaxModel, fit_openmax, openmax_probabilities
from hrrp_osr.training.b0_smoke import (
    EXPECTED_KNOWN_CLASSES,
    ScalarNormalization,
    _git_state,
    _infer_all_base_logits,
    _resolve_device,
    _row_indices,
    _sha256_file,
    _tree_hash,
)
from hrrp_osr.training.set_models import ViewSetDataset, _infer_sets


EXPECTED_SEEDS = [20260810, 20260820, 20260830, 20260840, 20260850]
OPENMAX_SELECTION_RULE = "five_fold_known_validation_acceptance_stability_v1"
OPENMAX_TAIL_SIZE_CANDIDATES = [10, 20, 40]
OPENMAX_ALPHA_RANK_CANDIDATES = [3, 5, 7]
OPENMAX_DISTANCE_TYPE_CANDIDATES = ["euclidean", "cosine", "eucos"]


def load_openmax_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise DataConfigError("OpenMax config must be a mapping")
    config = dict(raw)
    errors: list[str] = []
    baseline = config.get("baseline")
    expected_source = {"B4": "B0", "B5": "B2"}.get(baseline)
    scope_contract = {
        ("P2", "main_v3"): (3, SET_ALGORITHM_VERSION),
        ("P3", "paper_aligned_v5"): (5, V5_SET_ALGORITHM_VERSION),
    }.get((config.get("stage"), config.get("result_scope")))
    if scope_contract is None or expected_source is None:
        errors.append("OpenMax must be P2/main_v3 or P3/paper_aligned_v5")
    if tuple(config["classes"]["known_order"]) != EXPECTED_KNOWN_CLASSES:
        errors.append("known class order changed")
    protocol = config["set_protocol"]
    if scope_contract is not None and (
        int(protocol["view_count"]), protocol["set_algorithm_version"], bool(protocol["angle_or_position_inputs"])
    ) != (*scope_contract, False):
        errors.append("OpenMax set protocol changed")
    reuse = config["checkpoint_reuse"]
    if reuse["source_baseline"] != expected_source or reuse["training_allowed"] is not False:
        errors.append(f"{baseline} must reuse {expected_source} without training")
    if list(reuse["seeds"]) != EXPECTED_SEEDS:
        errors.append("OpenMax seed registry changed")
    fitting = config["openmax_fitting"]
    if (
        fitting["population"] != "correctly_classified_known_train_only"
        or fitting["activation"] != "pre_softmax_known_logits"
        or bool(fitting["unknown_data_used"])
    ):
        errors.append("OpenMax fit isolation or activation definition changed")
    parameters = config["openmax_parameters"]
    selection_rule = parameters.get("selection_rule")
    if selection_rule == "fixed_preregistered_original_style_v1":
        if (
            int(parameters["tail_size"]) != 20
            or int(parameters["alpha_rank"]) != 7
            or parameters["distance_type"] != "eucos"
            or float(parameters["eucos_euclidean_scale"]) != 200.0
        ):
            errors.append("legacy OpenMax parameters changed")
    elif selection_rule == OPENMAX_SELECTION_RULE:
        if (
            list(parameters.get("tail_size_candidates", [])) != OPENMAX_TAIL_SIZE_CANDIDATES
            or list(parameters.get("alpha_rank_candidates", [])) != OPENMAX_ALPHA_RANK_CANDIDATES
            or list(parameters.get("distance_type_candidates", [])) != OPENMAX_DISTANCE_TYPE_CANDIDATES
            or int(parameters.get("calibration_folds", 0)) != 5
            or float(parameters.get("stability_weight", -1.0)) != 0.01
            or float(parameters["eucos_euclidean_scale"]) != 200.0
        ):
            errors.append("OpenMax validation-selection grid changed")
    else:
        errors.append("unknown OpenMax parameter selection rule")
    if baseline == "B4" and config["fusion"]["method"] != "per_view_openmax_soft_k_plus_1_posterior_mean":
        errors.append("B4 must average per-view soft K+1 OpenMax posteriors")
    if baseline == "B5" and config["fusion"]["method"] != "single_set_activation_openmax":
        errors.append("B5 must fit and apply OpenMax on set activations")
    if errors:
        raise DataConfigError("Invalid OpenMax config:\n- " + "\n- ".join(errors))
    return config


def _parameter_candidates(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = config["openmax_parameters"]
    if values["selection_rule"] == "fixed_preregistered_original_style_v1":
        grid = [(values["tail_size"], values["alpha_rank"], values["distance_type"])]
    else:
        grid = itertools.product(
            values["tail_size_candidates"],
            values["alpha_rank_candidates"],
            values["distance_type_candidates"],
        )
    return [
        {
            "class_count": 7,
            "tail_size": int(tail_size),
            "alpha_rank": int(alpha_rank),
            "distance_type": str(distance_type),
            "eucos_euclidean_scale": float(values["eucos_euclidean_scale"]),
        }
        for tail_size, alpha_rank, distance_type in grid
    ]


def _select_openmax_model(
    train_activations: np.ndarray,
    train_labels: np.ndarray,
    train_predictions: np.ndarray,
    fit_entity_ids: Sequence[str],
    config: Mapping[str, Any],
    validation_builder: Callable[[OpenMaxModel], tuple[np.ndarray, np.ndarray]],
) -> tuple[OpenMaxModel, np.ndarray, np.ndarray, dict[str, Any]]:
    values = config["openmax_parameters"]
    acceptance = float(config["evaluation"]["threshold_known_acceptance_rate"])
    folds = int(values.get("calibration_folds", 5))
    stability_weight = float(values.get("stability_weight", 0.01))
    candidates: list[dict[str, Any]] = []
    fitted_models: list[OpenMaxModel] = []
    validation_probabilities: list[np.ndarray] = []
    validation_labels: list[np.ndarray] = []
    for ordinal, parameters in enumerate(_parameter_candidates(config)):
        fitted = fit_openmax(
            train_activations,
            train_labels,
            train_predictions,
            fit_entity_ids,
            **parameters,
        )
        probabilities, labels = validation_builder(fitted)
        if probabilities.ndim != 2 or probabilities.shape[1] != parameters["class_count"] + 1:
            raise DataValidationError("OpenMax validation probabilities have the wrong shape")
        if labels.shape != (len(probabilities),) or np.any(labels >= parameters["class_count"]):
            raise DataValidationError("OpenMax parameter selection received non-known validation data")
        scores = probabilities[:, -1]
        fold_records: list[dict[str, Any]] = []
        for fold in range(folds):
            held_out = np.arange(len(scores)) % folds == fold
            calibration = ~held_out
            if not np.any(held_out) or not np.any(calibration):
                raise DataValidationError("OpenMax validation fold is empty")
            threshold = threshold_for_known_acceptance(scores[calibration], acceptance)
            held_out_acceptance = float(np.mean(scores[held_out] <= threshold))
            fold_records.append({
                "fold": fold,
                "calibration_entity_count": int(np.sum(calibration)),
                "held_out_entity_count": int(np.sum(held_out)),
                "threshold": threshold,
                "held_out_known_acceptance": held_out_acceptance,
                "absolute_acceptance_error": abs(held_out_acceptance - acceptance),
            })
        thresholds = np.asarray([record["threshold"] for record in fold_records])
        score_iqr = float(np.percentile(scores, 75) - np.percentile(scores, 25))
        threshold_stability = float(np.std(thresholds) / max(score_iqr, 1.0e-12))
        acceptance_error = float(np.mean([record["absolute_acceptance_error"] for record in fold_records]))
        objective = acceptance_error + stability_weight * threshold_stability
        candidates.append({
            "ordinal": ordinal,
            "parameters": {key: value for key, value in parameters.items() if key != "class_count"},
            "known_validation_entity_count": len(scores),
            "mean_absolute_acceptance_error": acceptance_error,
            "iqr_normalized_threshold_std": threshold_stability,
            "objective": objective,
            "folds": fold_records,
        })
        fitted_models.append(fitted)
        validation_probabilities.append(probabilities)
        validation_labels.append(labels)
    if values["selection_rule"] == "fixed_preregistered_original_style_v1":
        selected_index = 0
        selection_basis = "legacy_fixed_parameter_reproduction_only"
    else:
        selected_index = min(range(len(candidates)), key=lambda index: (candidates[index]["objective"], index))
        selection_basis = "minimum_known_validation_calibration_instability"
    candidates[selected_index]["selected"] = True
    audit = {
        "status": "passed",
        "selection_rule": values["selection_rule"],
        "selection_basis": selection_basis,
        "known_validation_only": True,
        "unknown_validation_entity_count": 0,
        "candidate_count": len(candidates),
        "selected_ordinal": selected_index,
        "selected_parameters": candidates[selected_index]["parameters"],
        "acceptance_target": acceptance,
        "calibration_folds": folds,
        "stability_weight": stability_weight,
        "candidates": candidates,
    }
    return (
        fitted_models[selected_index],
        validation_probabilities[selected_index],
        validation_labels[selected_index],
        audit,
    )


def _b4_outputs(
    sets: Sequence[ViewSet],
    logits_by_sample: Mapping[str, np.ndarray],
    fitted: OpenMaxModel,
    class_to_index: Mapping[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = []
    labels = []
    for item in sets:
        member_logits = np.stack([logits_by_sample[sample_id] for sample_id in item.member_sample_ids])
        probabilities.append(openmax_probabilities(member_logits, fitted).mean(axis=0))
        labels.append(class_to_index.get(item.class_name, len(class_to_index)))
    return np.stack(probabilities), np.asarray(labels, dtype=int)


def _b4_permutation_audit(
    sets: Sequence[ViewSet],
    logits_by_sample: Mapping[str, np.ndarray],
    fitted: OpenMaxModel,
    atol: float,
) -> dict[str, Any]:
    view_count = len(sets[0].member_sample_ids)
    permutations = _audit_permutations(view_count)
    maximum = 0.0
    for item in sets:
        logits = np.stack([logits_by_sample[sample_id] for sample_id in item.member_sample_ids])
        reference = openmax_probabilities(logits, fitted).mean(axis=0)
        for permutation in permutations:
            actual = openmax_probabilities(logits[list(permutation)], fitted).mean(axis=0)
            maximum = max(maximum, float(np.max(np.abs(actual - reference))))
    if maximum > atol:
        raise DataValidationError(f"B4 permutation audit failed: {maximum} > {atol}")
    return {
        "status": "passed", "set_count": len(sets), "permutations_per_set": len(permutations),
        "atol": atol, "maximum_absolute_probability_difference": maximum,
    }


def _b5_permutation_audit(
    model: DeepSetsClassifier,
    bundle: ProcessedBundle,
    sets: Sequence[ViewSet],
    normalization: ScalarNormalization,
    class_to_index: Mapping[str, int],
    device: torch.device,
    fitted: OpenMaxModel,
    atol: float,
) -> dict[str, Any]:
    reference_logits, _ = _infer_sets(
        model, ViewSetDataset(bundle, sets, normalization, class_to_index), device, 64
    )
    reference = openmax_probabilities(reference_logits, fitted)
    permutations = _audit_permutations(len(sets[0].member_sample_ids))
    maximum = 0.0
    for permutation in permutations:
        logits, _ = _infer_sets(
            model,
            ViewSetDataset(bundle, sets, normalization, class_to_index, permutation=permutation),
            device,
            64,
        )
        actual = openmax_probabilities(logits, fitted)
        maximum = max(maximum, float(np.max(np.abs(actual - reference))))
    if maximum > atol:
        raise DataValidationError(f"B5 permutation audit failed: {maximum} > {atol}")
    return {
        "status": "passed", "set_count": len(sets), "permutations_per_set": len(permutations),
        "atol": atol, "maximum_absolute_probability_difference": maximum,
    }


def _audit_permutations(view_count: int) -> tuple[tuple[int, ...], ...]:
    if view_count == 3:
        return tuple(itertools.permutations(range(3)))
    if view_count != 5:
        raise DataValidationError("OpenMax permutation audit supports V=3 or V=5")
    base = tuple(range(5))
    reverse = tuple(reversed(base))
    values = []
    for shift in range(5):
        values.append(base[shift:] + base[:shift])
        values.append(reverse[shift:] + reverse[:shift])
    return tuple(dict.fromkeys(values))


def _write_predictions(
    destination: Path,
    split_sets: Mapping[str, Sequence[ViewSet]],
    split_probabilities: Mapping[str, np.ndarray],
    split_labels: Mapping[str, np.ndarray],
    class_order: Sequence[str],
    threshold: float,
) -> None:
    rows: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        probabilities = split_probabilities[split]
        predictions = probabilities[:, :-1].argmax(axis=1)
        for item, values, label, prediction in zip(
            split_sets[split], probabilities, split_labels[split], predictions, strict=True
        ):
            rows.append({
                "set_id": item.set_id,
                "split": split,
                "class_name": item.class_name,
                "class_role": item.class_role,
                "true_label": int(label),
                "predicted_known_label": int(prediction),
                "predicted_known_class": class_order[int(prediction)],
                "openmax_k_plus_1_posterior": json.dumps(values.tolist(), separators=(",", ":")),
                "openmax_unknown_score": float(values[-1]),
                "openmax_threshold": threshold,
            })
    with (destination / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_openmax_all(
    config_path: str | Path,
    bundle_root: str | Path,
    source_root: str | Path,
    output_root: str | Path,
    *,
    device_request: str = "auto",
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = load_openmax_config(config_path)
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
    set_builder = build_v3_sets if int(protocol["view_count"]) == 3 else build_v5_sets
    sets = {
        split: set_builder(
            bundle.rows, split=split, base_seed=int(protocol["set_seed"]),
            set_repeat=int(protocol["set_repeat"]),
        )
        for split in ("train", "validation", "test")
    }
    set_manifest = render_set_manifest_csv((*sets["train"], *sets["validation"], *sets["test"]))
    set_manifest_sha = hashlib.sha256(set_manifest).hexdigest()
    device = _resolve_device(device_request)
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    results: dict[str, Any] = {}
    for seed in config["checkpoint_reuse"]["seeds"]:
        destination = output_root / f"seed_{seed}"
        if destination.exists() and any(destination.iterdir()):
            raise DataValidationError(f"{baseline} output is non-empty: {destination}")
        destination.mkdir(parents=True, exist_ok=True)
        source = source_root / f"seed_{seed}" / "checkpoint.pt"
        expected_sha = str(config["checkpoint_reuse"]["checkpoint_sha256"][str(seed)])
        if _sha256_file(source) != expected_sha:
            raise DataValidationError(f"{baseline} source checkpoint hash mismatch for seed {seed}")
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        if tuple(checkpoint["class_order"]) != class_order:
            raise DataValidationError("OpenMax source checkpoint class order mismatch")
        normalization = ScalarNormalization(**checkpoint["normalization"])
        if normalization.fit_population != "known_train_only":
            raise DataValidationError("OpenMax source normalization is not known-train-only")
        if any(item.class_role != "known" for item in sets["validation"]):
            raise DataValidationError("OpenMax validation sets contain an unknown class")
        if baseline == "B4":
            model = HRRPClassifier1D().to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            all_indices = _row_indices(bundle, lambda row: True)
            logits_by_sample = _infer_all_base_logits(model, bundle, all_indices, normalization, device, 64)
            train_rows = [row for row in bundle.rows if int(row["eligible_for_training"]) == 1]
            if any(str(row["split"]) != "train" or str(row["class_role"]) != "known" for row in train_rows):
                raise DataValidationError("B4 fit population contains a forbidden row")
            train_logits = np.stack([logits_by_sample[str(row["sample_id"])] for row in train_rows])
            train_labels = np.asarray([class_to_index[str(row["class_name"])] for row in train_rows], dtype=int)
            fit_entity_ids = [str(row["sample_id"]) for row in train_rows]
            fitted, validation_probabilities, validation_labels, selection_audit = _select_openmax_model(
                train_logits,
                train_labels,
                train_logits.argmax(axis=1),
                fit_entity_ids,
                config,
                lambda candidate: _b4_outputs(
                    sets["validation"], logits_by_sample, candidate, class_to_index
                ),
            )
            test_probabilities, test_labels = _b4_outputs(
                sets["test"], logits_by_sample, fitted, class_to_index
            )
            split_probabilities = {
                "validation": validation_probabilities,
                "test": test_probabilities,
            }
            split_labels = {"validation": validation_labels, "test": test_labels}
            audit = _b4_permutation_audit(
                (*sets["validation"], *sets["test"]), logits_by_sample, fitted,
                float(config["evaluation"]["permutation_atol"]),
            )
            fit_entity_type = "base_hrrp_sample"
        else:
            model = DeepSetsClassifier().to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            datasets = {
                split: ViewSetDataset(bundle, values, normalization, class_to_index)
                for split, values in sets.items()
            }
            train_logits, train_labels = _infer_sets(model, datasets["train"], device, 64)
            if np.any(train_labels >= len(class_order)) or any(item.class_role != "known" for item in sets["train"]):
                raise DataValidationError("B5 fit population contains unknown sets")
            fit_entity_ids = [item.set_id for item in sets["train"]]
            validation_logits, expected_validation_labels = _infer_sets(
                model, datasets["validation"], device, 64
            )
            fitted, validation_probabilities, validation_labels, selection_audit = _select_openmax_model(
                train_logits,
                train_labels,
                train_logits.argmax(axis=1),
                fit_entity_ids,
                config,
                lambda candidate: (
                    openmax_probabilities(validation_logits, candidate),
                    expected_validation_labels,
                ),
            )
            test_logits, test_labels = _infer_sets(model, datasets["test"], device, 64)
            split_probabilities = {
                "validation": validation_probabilities,
                "test": openmax_probabilities(test_logits, fitted),
            }
            split_labels = {"validation": validation_labels, "test": test_labels}
            audit = _b5_permutation_audit(
                model, bundle, (*sets["validation"], *sets["test"]), normalization,
                class_to_index, device, fitted, float(config["evaluation"]["permutation_atol"]),
            )
            fit_entity_type = f"v{int(protocol['view_count'])}_set"
        validation_scores = split_probabilities["validation"][:, -1]
        test_probabilities = split_probabilities["test"]
        test_scores = test_probabilities[:, -1]
        test_predictions = test_probabilities[:, :-1].argmax(axis=1)
        test_labels = split_labels["test"]
        known_mask = test_labels < len(class_order)
        acceptance = float(config["evaluation"]["threshold_known_acceptance_rate"])
        threshold = threshold_for_known_acceptance(validation_scores, acceptance)
        metrics = evaluate_open_set(
            known_true=test_labels[known_mask],
            known_pred=test_predictions[known_mask],
            known_unknown_scores=test_scores[known_mask],
            unknown_pred=test_predictions[~known_mask],
            unknown_unknown_scores=test_scores[~known_mask],
            known_validation_scores=validation_scores,
            known_class_count=len(class_order),
            known_acceptance_rate=acceptance,
        )
        _write_predictions(
            destination, sets, split_probabilities, split_labels, class_order, threshold
        )
        (destination / "set_manifest.csv").write_bytes(set_manifest)
        fit_audit = {
            "status": "passed",
            "fit_population": "correctly_classified_known_train_only",
            "fit_entity_type": fit_entity_type,
            "candidate_entity_count": len(fit_entity_ids),
            "fitted_entity_count_by_class": [len(values) for values in fitted.fit_sample_ids_by_class],
            "fit_entity_ids_by_class": [list(values) for values in fitted.fit_sample_ids_by_class],
            "unknown_entity_count": 0,
            "validation_entity_count": 0,
            "test_entity_count": 0,
        }
        (destination / "openmax_fit.json").write_text(
            json.dumps({"model": fitted.to_dict(), "audit": fit_audit}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (destination / "openmax_selection.json").write_text(
            json.dumps(selection_audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (destination / "permutation_audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        checkpoint_reference = {
            "baseline": config["checkpoint_reuse"]["source_baseline"],
            "path": str(source), "sha256": expected_sha, "training_performed": False,
        }
        (destination / "checkpoint_reference.json").write_text(
            json.dumps(checkpoint_reference, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        metrics_doc = {
            "stage": config["stage"], "baseline": baseline, "result_scope": config["result_scope"],
            "model_seed": seed, "metrics": {"openmax": metrics},
            "threshold_source": "known_validation_only",
            "openmax_parameter_source": config["openmax_parameters"]["selection_rule"],
            "selected_openmax_parameters": selection_audit["selected_parameters"],
            "accepted_risks": [{"risk_id": config["data"]["accepted_risk_id"], "status": config["data"]["accepted_risk_status"]}],
        }
        (destination / "metrics.json").write_text(
            json.dumps(metrics_doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        project_root = config_path.parents[3]
        environment = {
            "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
            "torch": torch.__version__, "device": str(device), "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "git": _git_state(project_root), "code_tree_sha256": _tree_hash(project_root),
        }
        (destination / "environment.json").write_text(
            json.dumps(environment, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        resolved = dict(config)
        resolved["_resolved"] = {
            "bundle_root": str(bundle.root), "source_checkpoint": str(source),
            "source_checkpoint_sha256": expected_sha, "set_manifest_sha256": set_manifest_sha,
            "device": str(device), "model_seed": seed,
        }
        (destination / "resolved_config.yaml").write_text(
            yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        artifact_names = [
            "checkpoint_reference.json", "environment.json", "metrics.json", "openmax_fit.json",
            "openmax_selection.json", "permutation_audit.json", "predictions.csv",
            "resolved_config.yaml", "set_manifest.csv",
        ]
        (destination / "artifact_manifest.json").write_text(
            json.dumps({"artifacts": {name: _sha256_file(destination / name) for name in artifact_names}}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        results[str(seed)] = {
            "metrics": {"openmax": metrics},
            "permutation_max_abs": audit["maximum_absolute_probability_difference"],
            "source_checkpoint_sha256": expected_sha,
            "fitted_entity_count_by_class": fit_audit["fitted_entity_count_by_class"],
            "selected_openmax_parameters": selection_audit["selected_parameters"],
        }
    return {"validation_status": "passed", "baseline": baseline, "seed_results": results}
