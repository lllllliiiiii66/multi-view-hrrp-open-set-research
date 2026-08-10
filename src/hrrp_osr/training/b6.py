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
from hrrp_osr.evaluation.metrics import evaluate_open_set, macro_f1_score, threshold_for_known_acceptance
from hrrp_osr.jdsr import (
    JDSRGPDModel,
    dual_tail_unknown_scores,
    fit_dual_tail_gpd,
    jdsr_reconstruction_errors,
)
from hrrp_osr.training.b0_smoke import (
    EXPECTED_KNOWN_CLASSES,
    _git_state,
    _resolve_device,
    _sha256_file,
    _tree_hash,
)


def load_b6_main_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise DataConfigError("B6 config must be a mapping")
    config = dict(raw)
    errors: list[str] = []
    if (config.get("stage"), config.get("baseline"), config.get("result_scope")) != ("P3", "B6", "main_v3"):
        errors.append("B6 main must remain P3/B6/main_v3")
    if config.get("method_claim") != "core_algorithm_reproduction_and_v3_adaptation":
        errors.append("B6 evidence boundary changed")
    if tuple(config["classes"]["known_order"]) != EXPECTED_KNOWN_CLASSES:
        errors.append("known class order changed")
    protocol = config["set_protocol"]
    if (int(protocol["view_count"]), protocol["set_algorithm_version"], bool(protocol["angle_or_position_inputs"])) != (
        3, SET_ALGORITHM_VERSION, False
    ):
        errors.append("B6 main V=3 protocol changed")
    dictionary = config["dictionary"]
    if (dictionary["population"], int(dictionary["atoms_per_class"]), int(dictionary["class_count"])) != (
        "known_train_hrrp_only", 216, 7
    ):
        errors.append("B6 dictionary isolation changed")
    if config["solver"]["algorithm"] != "greedy_dynamic_active_set_group_omp_v1":
        errors.append("B6 JDSR core solver changed")
    if list(config["solver"]["sparsity_candidates"]) != [1, 2, 3]:
        errors.append("B6 main sparsity selection grid changed")
    gpd = config["gpd"]
    if (
        gpd["fit_population"] != "known_train_sets_only"
        or bool(gpd["unknown_data_used"])
        or list(gpd["rho_candidates"]) != [0.5, 0.6, 0.7, 0.8]
        or list(gpd["nonmatching_weight_candidates"]) != [0.0, 0.25, 0.5, 0.75, 1.0]
    ):
        errors.append("B6 dual-tail fitting isolation or grid changed")
    if config["evaluation"]["paper_fixed_delta_used"] is not False:
        errors.append("paper fixed threshold is forbidden in main_v3")
    if errors:
        raise DataConfigError("Invalid B6 main config:\n- " + "\n- ".join(errors))
    return config


def load_b6_paper_aligned_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise DataConfigError("B6 paper-aligned config must be a mapping")
    config = dict(raw)
    errors: list[str] = []
    if (config.get("stage"), config.get("baseline"), config.get("result_scope")) != (
        "P3", "B6", "paper_aligned_v5"
    ):
        errors.append("B6 paper-aligned run must remain P3/B6/paper_aligned_v5")
    protocol = config["set_protocol"]
    if (int(protocol["view_count"]), protocol["set_algorithm_version"], bool(protocol["angle_or_position_inputs"])) != (
        5, V5_SET_ALGORITHM_VERSION, False
    ):
        errors.append("B6 paper-aligned V=5 protocol changed")
    if tuple(config["classes"]["known_order"]) != EXPECTED_KNOWN_CLASSES:
        errors.append("known class order changed")
    if config["dictionary"]["population"] != "known_train_hrrp_only":
        errors.append("B6 paper-aligned dictionary isolation changed")
    if int(config["solver"]["sparsity"]) != 2 or config["solver"]["parameter_source"] != "paper_fixed_K":
        errors.append("paper-aligned K must remain 2")
    if float(config["gpd"]["rho"]) != 0.7 or config["gpd"]["rho_source"] != "paper_fixed_rho":
        errors.append("paper-aligned rho must remain 0.7")
    if bool(config["gpd"]["unknown_data_used"]):
        errors.append("unknown data cannot enter paper-aligned GPD fitting")
    if float(config["evaluation"]["paper_fixed_delta"]) != 0.3:
        errors.append("paper-aligned diagnostic delta must remain 0.3")
    if errors:
        raise DataConfigError("Invalid B6 paper-aligned config:\n- " + "\n- ".join(errors))
    return config


def _dictionary(
    bundle: ProcessedBundle,
    class_order: Sequence[str],
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, int]]:
    blocks = []
    manifest_rows = []
    sample_to_flat_index: dict[str, int] = {}
    atoms_per_class = None
    for class_index, class_name in enumerate(class_order):
        rows = [
            row for row in bundle.rows
            if str(row["class_name"]) == class_name
            and str(row["class_role"]) == "known"
            and str(row["split"]) == "train"
            and int(row["eligible_for_training"]) == 1
        ]
        rows.sort(key=lambda row: int(row["processed_row_index"]))
        if atoms_per_class is None:
            atoms_per_class = len(rows)
        if len(rows) != atoms_per_class or len(rows) != 216:
            raise DataValidationError("B6 dictionary does not have exactly 216 atoms per known class")
        blocks.append(np.stack([bundle.profiles[int(row["processed_row_index"])] for row in rows]))
        for atom_index, row in enumerate(rows):
            flat_index = class_index * len(rows) + atom_index
            sample_id = str(row["sample_id"])
            sample_to_flat_index[sample_id] = flat_index
            manifest_rows.append({
                "flat_atom_index": flat_index, "class_index": class_index,
                "class_name": class_name, "class_atom_index": atom_index,
                "sample_id": sample_id, "split": row["split"], "class_role": row["class_role"],
                "angle_deg": row["angle_deg"], "domain_id": row["domain_id"],
            })
    return np.asarray(np.stack(blocks), dtype=np.float32), manifest_rows, sample_to_flat_index


def _set_arrays(
    bundle: ProcessedBundle,
    sets: Sequence[ViewSet],
    class_to_index: Mapping[str, int],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    sample_to_row = {str(row["sample_id"]): int(row["processed_row_index"]) for row in bundle.rows}
    profiles = np.stack([
        np.stack([bundle.profiles[sample_to_row[sample_id]] for sample_id in item.member_sample_ids])
        for item in sets
    ])
    labels = np.asarray([class_to_index.get(item.class_name, len(class_to_index)) for item in sets], dtype=int)
    return np.asarray(profiles, dtype=np.float32), labels, [item.set_id for item in sets]


def _gpd_selection(
    train_errors: np.ndarray,
    train_labels: np.ndarray,
    train_ids: Sequence[str],
    validation_errors: np.ndarray,
    rho_candidates: Sequence[float],
    weight_candidates: Sequence[float],
    acceptance_target: float,
) -> tuple[JDSRGPDModel, list[dict[str, Any]]]:
    records = []
    best: tuple[float, float, float, JDSRGPDModel] | None = None
    indices = np.arange(len(validation_errors))
    for rho in rho_candidates:
        for weight in weight_candidates:
            model = fit_dual_tail_gpd(
                train_errors, train_labels, train_ids, class_count=train_errors.shape[1],
                rho=float(rho), nonmatching_weight=float(weight),
            )
            _, scores, _, _ = dual_tail_unknown_scores(validation_errors, model)
            fold_thresholds = []
            fold_acceptance_errors = []
            for fold in range(5):
                held_out = indices % 5 == fold
                calibration = ~held_out
                threshold = threshold_for_known_acceptance(scores[calibration], acceptance_target)
                held_acceptance = float(np.mean(scores[held_out] <= threshold))
                fold_thresholds.append(float(threshold))
                fold_acceptance_errors.append(abs(held_acceptance - acceptance_target))
            spread = float(np.percentile(scores, 75) - np.percentile(scores, 25))
            normalized_threshold_std = float(np.std(fold_thresholds)) / max(spread, 1.0e-12)
            objective = float(np.mean(fold_acceptance_errors)) + 0.01 * normalized_threshold_std
            record = {
                "rho": float(rho), "nonmatching_weight": float(weight),
                "mean_absolute_heldout_acceptance_error": float(np.mean(fold_acceptance_errors)),
                "normalized_threshold_std": normalized_threshold_std,
                "objective": objective, "fold_thresholds": fold_thresholds,
            }
            records.append(record)
            key = (objective, float(rho), float(weight), model)
            if best is None or key[:3] < best[:3]:
                best = key
    if best is None:
        raise DataValidationError("B6 GPD selection produced no candidate")
    return best[3], records


def _permutation_audit(
    profiles: np.ndarray,
    reference_errors: np.ndarray,
    dictionary: np.ndarray,
    sparsity: int,
    device: torch.device,
    gpd_model: JDSRGPDModel,
    atol: float,
) -> dict[str, Any]:
    reference_candidates, reference_scores, _, _ = dual_tail_unknown_scores(reference_errors, gpd_model)
    maximum_error = 0.0
    maximum_score = 0.0
    candidate_mismatches = 0
    view_count = profiles.shape[1]
    if view_count == 3:
        permutations = tuple(itertools.permutations(range(3)))
    elif view_count == 5:
        base = tuple(range(5)); reverse = tuple(reversed(base)); values = []
        for shift in range(5):
            values.extend((base[shift:] + base[:shift], reverse[shift:] + reverse[:shift]))
        permutations = tuple(dict.fromkeys(values))
    else:
        raise DataValidationError("B6 permutation audit supports V=3 or V=5")
    for permutation in permutations:
        actual_errors = jdsr_reconstruction_errors(
            profiles[:, permutation, :], dictionary, sparsity=sparsity, device=device
        )
        actual_candidates, actual_scores, _, _ = dual_tail_unknown_scores(actual_errors, gpd_model)
        maximum_error = max(maximum_error, float(np.max(np.abs(actual_errors - reference_errors))))
        maximum_score = max(maximum_score, float(np.max(np.abs(actual_scores - reference_scores))))
        candidate_mismatches += int(np.count_nonzero(actual_candidates != reference_candidates))
    if maximum_error > atol or maximum_score > atol or candidate_mismatches:
        raise DataValidationError(
            f"B6 permutation audit failed: error={maximum_error}, score={maximum_score}, mismatches={candidate_mismatches}"
        )
    return {
        "status": "passed", "set_count": len(profiles), "permutations_per_set": len(permutations),
        "atol": atol, "maximum_absolute_error_difference": maximum_error,
        "maximum_absolute_score_difference": maximum_score,
        "candidate_mismatch_count": candidate_mismatches,
    }


def run_b6_main(
    config_path: str | Path,
    bundle_root: str | Path,
    output: str | Path,
    *,
    device_request: str = "auto",
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = load_b6_main_config(config_path)
    destination = Path(output).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise DataValidationError(f"B6 output is non-empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
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
        split: build_v3_sets(
            bundle.rows, split=split, base_seed=int(protocol["set_seed"]),
            set_repeat=int(protocol["set_repeat"]),
        )
        for split in ("train", "validation", "test")
    }
    if any(item.class_role != "known" for item in (*sets["train"], *sets["validation"])):
        raise DataValidationError("B6 train or validation sets contain unknown classes")
    set_manifest = render_set_manifest_csv((*sets["train"], *sets["validation"], *sets["test"]))
    dictionary, dictionary_rows, sample_to_atom = _dictionary(bundle, class_order)
    arrays = {split: _set_arrays(bundle, values, class_to_index) for split, values in sets.items()}
    train_exclusions = np.asarray([
        [sample_to_atom[sample_id] for sample_id in item.member_sample_ids]
        for item in sets["train"]
    ], dtype=int)
    device = _resolve_device(device_request)
    solver_records = []
    cached_train: dict[int, np.ndarray] = {}
    cached_validation: dict[int, np.ndarray] = {}
    for sparsity in config["solver"]["sparsity_candidates"]:
        started = time.perf_counter()
        train_errors = jdsr_reconstruction_errors(
            arrays["train"][0], dictionary, sparsity=int(sparsity), device=device,
            excluded_atom_indices=train_exclusions, ridge=float(config["solver"]["ridge"]),
        )
        validation_errors = jdsr_reconstruction_errors(
            arrays["validation"][0], dictionary, sparsity=int(sparsity), device=device,
            ridge=float(config["solver"]["ridge"]),
        )
        validation_accuracy = float(np.mean(validation_errors.argmin(axis=1) == arrays["validation"][1]))
        cached_train[int(sparsity)] = train_errors
        cached_validation[int(sparsity)] = validation_errors
        solver_records.append({
            "sparsity": int(sparsity), "known_validation_closed_set_accuracy": validation_accuracy,
            "elapsed_seconds": time.perf_counter() - started,
        })
    selected_sparsity = min(
        solver_records, key=lambda row: (-row["known_validation_closed_set_accuracy"], row["sparsity"])
    )["sparsity"]
    train_errors = cached_train[selected_sparsity]
    validation_errors = cached_validation[selected_sparsity]
    gpd_model, gpd_grid = _gpd_selection(
        train_errors, arrays["train"][1], arrays["train"][2], validation_errors,
        config["gpd"]["rho_candidates"], config["gpd"]["nonmatching_weight_candidates"],
        float(config["evaluation"]["threshold_known_acceptance_rate"]),
    )
    test_started = time.perf_counter()
    test_errors = jdsr_reconstruction_errors(
        arrays["test"][0], dictionary, sparsity=selected_sparsity, device=device,
        ridge=float(config["solver"]["ridge"]),
    )
    test_solver_seconds = time.perf_counter() - test_started
    validation_predictions, validation_scores, validation_match, validation_nonmatch = dual_tail_unknown_scores(
        validation_errors, gpd_model
    )
    test_predictions, test_scores, test_match, test_nonmatch = dual_tail_unknown_scores(test_errors, gpd_model)
    acceptance = float(config["evaluation"]["threshold_known_acceptance_rate"])
    threshold = threshold_for_known_acceptance(validation_scores, acceptance)
    test_labels = arrays["test"][1]
    known_mask = test_labels < len(class_order)
    metrics = evaluate_open_set(
        known_true=test_labels[known_mask], known_pred=test_predictions[known_mask],
        known_unknown_scores=test_scores[known_mask], unknown_pred=test_predictions[~known_mask],
        unknown_unknown_scores=test_scores[~known_mask], known_validation_scores=validation_scores,
        known_class_count=len(class_order), known_acceptance_rate=acceptance,
    )
    audit_profiles = np.concatenate([arrays["validation"][0], arrays["test"][0]])
    audit_errors = np.concatenate([validation_errors, test_errors])
    permutation_audit = _permutation_audit(
        audit_profiles, audit_errors, dictionary, selected_sparsity, device, gpd_model,
        float(config["evaluation"]["permutation_atol"]),
    )
    prediction_rows = []
    for split, split_sets, labels, errors, predictions, scores, matching, nonmatching in (
        ("validation", sets["validation"], arrays["validation"][1], validation_errors, validation_predictions, validation_scores, validation_match, validation_nonmatch),
        ("test", sets["test"], test_labels, test_errors, test_predictions, test_scores, test_match, test_nonmatch),
    ):
        for item, label, row_errors, prediction, score, match_score, nonmatch_score in zip(
            split_sets, labels, errors, predictions, scores, matching, nonmatching, strict=True
        ):
            prediction_rows.append({
                "set_id": item.set_id, "split": split, "class_name": item.class_name,
                "class_role": item.class_role, "true_label": int(label),
                "predicted_known_label": int(prediction), "predicted_known_class": class_order[int(prediction)],
                "reconstruction_errors": json.dumps(row_errors.tolist(), separators=(",", ":")),
                "matching_tail_score": float(match_score), "nonmatching_tail_score": float(nonmatch_score),
                "dual_tail_unknown_score": float(score), "known_validation_threshold": threshold,
            })
    with (destination / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader(); writer.writerows(prediction_rows)
    with (destination / "dictionary_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dictionary_rows[0]))
        writer.writeheader(); writer.writerows(dictionary_rows)
    (destination / "set_manifest.csv").write_bytes(set_manifest)
    solver_doc = {
        "algorithm": config["solver"]["algorithm"], "selection_rule": config["solver"]["selection_rule"],
        "candidates": solver_records, "selected_sparsity": selected_sparsity,
        "dictionary_shape": list(dictionary.shape), "dictionary_atom_count": int(np.prod(dictionary.shape[:2])),
        "test_set_count": len(test_errors), "test_solver_seconds": test_solver_seconds,
        "mean_test_solver_seconds_per_set": test_solver_seconds / len(test_errors),
    }
    (destination / "solver_selection.json").write_text(
        json.dumps(solver_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    gpd_doc = {
        "selection_rule": config["gpd"]["selection_rule"], "selected_model": gpd_model.to_dict(),
        "candidate_grid_audit": gpd_grid, "fit_population": "known_train_sets_only",
        "fit_set_count": len(arrays["train"][2]), "unknown_fit_set_count": 0,
        "validation_fit_set_count": 0, "test_fit_set_count": 0,
    }
    (destination / "gpd_fit.json").write_text(
        json.dumps(gpd_doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (destination / "permutation_audit.json").write_text(
        json.dumps(permutation_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metrics_doc = {
        "stage": "P3", "baseline": "B6", "result_scope": "main_v3",
        "method_claim": config["method_claim"], "metrics": {"dual_tail_gpd": metrics},
        "threshold_source": "known_validation_only", "paper_fixed_delta_used": False,
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
        "bundle_root": str(bundle.root), "device": str(device), "selected_sparsity": selected_sparsity,
        "selected_rho": gpd_model.rho, "selected_nonmatching_weight": gpd_model.nonmatching_weight,
        "set_manifest_sha256": hashlib.sha256(set_manifest).hexdigest(),
    }
    (destination / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    artifact_names = [
        "dictionary_manifest.csv", "environment.json", "gpd_fit.json", "metrics.json",
        "permutation_audit.json", "predictions.csv", "resolved_config.yaml", "set_manifest.csv",
        "solver_selection.json",
    ]
    (destination / "artifact_manifest.json").write_text(
        json.dumps({"artifacts": {name: _sha256_file(destination / name) for name in artifact_names}}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "validation_status": "passed", "baseline": "B6", "result_scope": "main_v3",
        "selected_sparsity": selected_sparsity, "selected_rho": gpd_model.rho,
        "selected_nonmatching_weight": gpd_model.nonmatching_weight, "metrics": {"dual_tail_gpd": metrics},
        "permutation_max_abs": max(
            permutation_audit["maximum_absolute_error_difference"],
            permutation_audit["maximum_absolute_score_difference"],
        ),
    }


def run_b6_paper_aligned_v5(
    config_path: str | Path,
    bundle_root: str | Path,
    output: str | Path,
    *,
    device_request: str = "auto",
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = load_b6_paper_aligned_config(config_path)
    destination = Path(output).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise DataValidationError(f"B6 V=5 output is non-empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
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
        split: build_v5_sets(
            bundle.rows, split=split, base_seed=int(protocol["set_seed"]),
            set_repeat=int(protocol["set_repeat"]),
        )
        for split in ("train", "validation", "test")
    }
    if any(item.class_role != "known" for item in (*sets["train"], *sets["validation"])):
        raise DataValidationError("B6 V=5 train or validation sets contain unknown classes")
    set_manifest = render_set_manifest_csv((*sets["train"], *sets["validation"], *sets["test"]))
    dictionary, dictionary_rows, sample_to_atom = _dictionary(bundle, class_order)
    arrays = {split: _set_arrays(bundle, values, class_to_index) for split, values in sets.items()}
    train_exclusions = np.asarray([
        [sample_to_atom[sample_id] for sample_id in item.member_sample_ids]
        for item in sets["train"]
    ], dtype=int)
    device = _resolve_device(device_request)
    sparsity = int(config["solver"]["sparsity"])
    ridge = float(config["solver"]["ridge"])
    started = time.perf_counter()
    train_errors = jdsr_reconstruction_errors(
        arrays["train"][0], dictionary, sparsity=sparsity, device=device,
        excluded_atom_indices=train_exclusions, ridge=ridge,
    )
    validation_errors = jdsr_reconstruction_errors(
        arrays["validation"][0], dictionary, sparsity=sparsity, device=device, ridge=ridge
    )
    test_errors = jdsr_reconstruction_errors(
        arrays["test"][0], dictionary, sparsity=sparsity, device=device, ridge=ridge
    )
    solver_seconds = time.perf_counter() - started
    known_count = int(config["gpd"]["openness_known_class_count"])
    total_count = int(config["gpd"]["openness_total_class_count"])
    openness = 1.0 - float(np.sqrt(2.0 * known_count / (known_count + total_count)))
    nonmatching_weight = float(config["gpd"]["p"]) * (1.0 - openness)
    gpd_model = fit_dual_tail_gpd(
        train_errors, arrays["train"][1], arrays["train"][2], class_count=len(class_order),
        rho=float(config["gpd"]["rho"]), nonmatching_weight=nonmatching_weight,
    )
    validation_predictions, validation_scores, validation_match, validation_nonmatch = dual_tail_unknown_scores(
        validation_errors, gpd_model
    )
    test_predictions, test_scores, test_match, test_nonmatch = dual_tail_unknown_scores(test_errors, gpd_model)
    acceptance = float(config["evaluation"]["threshold_known_acceptance_rate"])
    fair_threshold = threshold_for_known_acceptance(validation_scores, acceptance)
    test_labels = arrays["test"][1]
    known_mask = test_labels < len(class_order)
    fair_metrics = evaluate_open_set(
        known_true=test_labels[known_mask], known_pred=test_predictions[known_mask],
        known_unknown_scores=test_scores[known_mask], unknown_pred=test_predictions[~known_mask],
        unknown_unknown_scores=test_scores[~known_mask], known_validation_scores=validation_scores,
        known_class_count=len(class_order), known_acceptance_rate=acceptance,
    )
    fixed_delta = float(config["evaluation"]["paper_fixed_delta"])
    unknown_label = len(class_order)
    open_true = np.concatenate([
        test_labels[known_mask], np.full(np.count_nonzero(~known_mask), unknown_label, dtype=int)
    ])
    open_pred = np.concatenate([
        np.where(test_scores[known_mask] > fixed_delta, unknown_label, test_predictions[known_mask]),
        np.where(test_scores[~known_mask] > fixed_delta, unknown_label, test_predictions[~known_mask]),
    ])
    fixed_operating_point = {
        "threshold": fixed_delta,
        "threshold_source": "paper_fixed_delta_diagnostic_only",
        "known_acceptance_rate": float(np.mean(test_scores[known_mask] <= fixed_delta)),
        "unknown_rejection_rate": float(np.mean(test_scores[~known_mask] > fixed_delta)),
        "k_plus_1_macro_f1": macro_f1_score(open_true, open_pred, labels=range(len(class_order) + 1)),
    }
    permutation_audit = _permutation_audit(
        np.concatenate([arrays["validation"][0], arrays["test"][0]]),
        np.concatenate([validation_errors, test_errors]), dictionary, sparsity, device, gpd_model,
        float(config["evaluation"]["permutation_atol"]),
    )
    prediction_rows = []
    for split, split_sets, labels, errors, predictions, scores, matching, nonmatching in (
        ("validation", sets["validation"], arrays["validation"][1], validation_errors, validation_predictions, validation_scores, validation_match, validation_nonmatch),
        ("test", sets["test"], test_labels, test_errors, test_predictions, test_scores, test_match, test_nonmatch),
    ):
        for item, label, row_errors, prediction, score, match_score, nonmatch_score in zip(
            split_sets, labels, errors, predictions, scores, matching, nonmatching, strict=True
        ):
            prediction_rows.append({
                "set_id": item.set_id, "split": split, "class_name": item.class_name,
                "class_role": item.class_role, "true_label": int(label),
                "predicted_known_label": int(prediction), "predicted_known_class": class_order[int(prediction)],
                "reconstruction_errors": json.dumps(row_errors.tolist(), separators=(",", ":")),
                "matching_tail_score": float(match_score), "nonmatching_tail_score": float(nonmatch_score),
                "dual_tail_unknown_score": float(score), "known_validation_threshold": fair_threshold,
                "paper_fixed_delta": fixed_delta,
            })
    with (destination / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0])); writer.writeheader(); writer.writerows(prediction_rows)
    with (destination / "dictionary_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dictionary_rows[0])); writer.writeheader(); writer.writerows(dictionary_rows)
    (destination / "set_manifest.csv").write_bytes(set_manifest)
    (destination / "gpd_fit.json").write_text(
        json.dumps({
            "parameter_source": "paper_fixed_K_rho_with_dataset_openness_weight",
            "selected_model": gpd_model.to_dict(), "openness": openness,
            "nonmatching_weight": nonmatching_weight, "fit_population": "known_train_sets_only",
            "fit_set_count": len(arrays["train"][2]), "unknown_fit_set_count": 0,
            "validation_fit_set_count": 0, "test_fit_set_count": 0,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    solver_doc = {
        "algorithm": config["solver"]["algorithm"], "sparsity": sparsity,
        "sparsity_source": "paper_fixed_K", "dictionary_shape": list(dictionary.shape),
        "dictionary_atom_count": int(np.prod(dictionary.shape[:2])),
        "all_train_validation_test_solver_seconds": solver_seconds,
        "set_counts": {split: len(values) for split, values in sets.items()},
    }
    (destination / "solver.json").write_text(json.dumps(solver_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (destination / "permutation_audit.json").write_text(
        json.dumps(permutation_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metrics_doc = {
        "stage": "P3", "baseline": "B6", "result_scope": "paper_aligned_v5",
        "method_claim": config["method_claim"], "metrics": {"dual_tail_gpd": fair_metrics},
        "primary_auxiliary_threshold_source": "known_validation_only",
        "paper_fixed_delta_operating_point": fixed_operating_point,
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
        "bundle_root": str(bundle.root), "device": str(device), "openness": openness,
        "nonmatching_weight": nonmatching_weight,
        "set_manifest_sha256": hashlib.sha256(set_manifest).hexdigest(),
    }
    (destination / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    artifact_names = [
        "dictionary_manifest.csv", "environment.json", "gpd_fit.json", "metrics.json",
        "permutation_audit.json", "predictions.csv", "resolved_config.yaml", "set_manifest.csv", "solver.json",
    ]
    (destination / "artifact_manifest.json").write_text(
        json.dumps({"artifacts": {name: _sha256_file(destination / name) for name in artifact_names}}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "validation_status": "passed", "baseline": "B6", "result_scope": "paper_aligned_v5",
        "sparsity": sparsity, "rho": gpd_model.rho, "nonmatching_weight": nonmatching_weight,
        "metrics": {"dual_tail_gpd": fair_metrics},
        "paper_fixed_delta_operating_point": fixed_operating_point,
        "permutation_max_abs": max(
            permutation_audit["maximum_absolute_error_difference"],
            permutation_audit["maximum_absolute_score_difference"],
        ),
    }
