from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import resource
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
import yaml

from hrrp_osr.amdr.data import (
    PEAK_RELATIVE_AMPLITUDE_TRANSFORM_ID,
    RANDOMIZED_SLOT_ORDER,
    TwoViewPair,
    build_fold_pairs,
    materialize_pair_views,
    write_pair_manifest,
)
from hrrp_osr.amdr.model import (
    ALLOW_SAME_BASE_GRAPH,
    AMDR_ALGORITHM_VERSION,
    COMPLETE_SAME_CLASS_GRAPH,
    FIXED_INITIAL_L21_REWEIGHTING,
    RELATIVE_STATE_CHANGE,
    SAMPLE_CLASS_MEAN_OBJECTIVE,
    AMDRFitResult,
    AMDRModelConfig,
    fit_amdr,
    knn_predict_and_score,
    load_amdr_checkpoint,
    project_views,
    save_amdr_checkpoint,
)
from hrrp_osr.amdr.smoke import _git_state, _resource_limits
from hrrp_osr.data.errors import DataConfigError, DataValidationError
from hrrp_osr.data.manifest import file_sha256
from hrrp_osr.data.processed import ProcessedBundle, load_processed_bundle
from hrrp_osr.evaluation.metrics import accuracy_score, macro_f1_score


CLOSURE_SCOPE = "confirmatory_known_only"
CLOSURE_ID = "amdr_p0_closure_known_only_v1"
PRIMARY = "primary"
AUXILIARY = "auxiliary"
REJECT = "reject"


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DataConfigError(f"{name} must be a mapping")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise DataValidationError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _recursive_hashes(root: Path, excluded_names: set[str]) -> dict[str, str]:
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in excluded_names
    }


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _validate_exact_sequence(value: Any, expected: Sequence[Any], name: str) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise DataConfigError(f"{name} must be a sequence")
    if list(value) != list(expected):
        raise DataConfigError(f"{name} must remain {list(expected)}")


def load_closure_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = dict(_mapping(yaml.safe_load(handle), "P0 closure config"))
    errors: list[str] = []
    if int(config.get("schema_version", 0)) != 1:
        errors.append("schema_version must be 1")
    if config.get("stage") != "P0" or config.get("result_scope") != CLOSURE_SCOPE:
        errors.append("closure must remain P0 confirmatory_known_only")
    if config.get("closure_id") != CLOSURE_ID:
        errors.append(f"closure_id must be {CLOSURE_ID}")

    evidence = _mapping(config.get("evidence_scope"), "evidence_scope")
    required_false = (
        "fold0_used",
        "unknown_used",
        "even_angle_test_pairs_generated",
        "test_features_materialized",
        "test_metrics_used",
        "p1_started",
    )
    if any(evidence.get(name) is not False for name in required_false):
        errors.append("fold0, unknown, even-angle test and P1 must remain unused")

    bundle = _mapping(config.get("bundle"), "bundle")
    for name in ("profiles_sha256", "manifest_sha256", "bundle_sha256"):
        if not isinstance(bundle.get(name), str) or len(str(bundle.get(name))) != 64:
            errors.append(f"bundle.{name} must be a SHA-256")

    protocol = _mapping(config.get("protocol"), "protocol")
    try:
        _validate_exact_sequence(protocol.get("forbidden_folds"), [0], "forbidden_folds")
        _validate_exact_sequence(
            protocol.get("selection_folds"), [1, 2, 3], "selection_folds"
        )
    except DataConfigError as exc:
        errors.append(str(exc))
    if int(protocol.get("confirmation_fold", -1)) != 4:
        errors.append("confirmation_fold must be 4")
    if (
        int(protocol.get("frame_width_deg", 0)),
        int(protocol.get("frame_count", 0)),
        int(protocol.get("fold_count", 0)),
        int(protocol.get("known_class_count", 0)),
        int(protocol.get("unknown_class_count", 0)),
    ) != (15, 24, 5, 7, 3):
        errors.append("protocol dimensions must remain 15/24/5/7/3")
    if protocol.get("development_angle_parity") != "odd":
        errors.append("development angles must remain odd")

    sampling = _mapping(config.get("sampling"), "sampling")
    try:
        _validate_exact_sequence(
            sampling.get("included_splits"),
            ["train", "calibration"],
            "sampling.included_splits",
        )
    except DataConfigError as exc:
        errors.append(str(exc))
    pair_counts = _mapping(sampling.get("pairs_per_class"), "pairs_per_class")
    if dict(pair_counts) != {"train": 500, "calibration": 500}:
        errors.append("train/calibration must each use 500 pairs per class")
    if (
        sampling.get("slot_order") != RANDOMIZED_SLOT_ORDER
        or sampling.get("distinct_frames") is not True
        or sampling.get("duplicate_unordered_pairs") is not False
    ):
        errors.append("sampling invariants changed")

    preprocessing = _mapping(config.get("preprocessing"), "preprocessing")
    if preprocessing.get("transform") != PEAK_RELATIVE_AMPLITUDE_TRANSFORM_ID:
        errors.append("closure input must be peak-relative amplitude")
    if preprocessing.get("dimension_reduction") != "none":
        errors.append("closure must not use dimension reduction")

    model = _mapping(config.get("model"), "model")
    exact_model = {
        "algorithm_version": AMDR_ALGORITHM_VERSION,
        "graph_neighborhood": COMPLETE_SAME_CLASS_GRAPH,
        "graph_same_base_policy": ALLOW_SAME_BASE_GRAPH,
        "l21_reweighting": FIXED_INITIAL_L21_REWEIGHTING,
        "objective_scaling": SAMPLE_CLASS_MEAN_OBJECTIVE,
        "max_iterations": 300,
        "minimum_iterations": 3,
        "tolerance": 3.0e-5,
        "convergence_metric": RELATIVE_STATE_CHANGE,
        "numerical_epsilon": 1.0e-10,
        "solve_ridge": 1.0e-10,
        "initialization_seed": 20260830,
        "post_training_row_prune_squared_norm_threshold": 0.0,
    }
    for name, expected in exact_model.items():
        if model.get(name) != expected:
            errors.append(f"model.{name} must remain {expected}")

    selection = _mapping(config.get("selection"), "selection")
    if selection.get("require_converged") is not True:
        errors.append("selection.require_converged must be true")
    if float(selection.get("minimum_alpha", -1.0)) != 0.05:
        errors.append("selection.minimum_alpha must be 0.05")
    try:
        _validate_exact_sequence(
            selection.get("lambda_manifold"),
            [10.0, 100.0, 1000.0],
            "selection.lambda_manifold",
        )
        _validate_exact_sequence(
            selection.get("lambda_sparse"),
            [2.857142857142857e-4],
            "selection.lambda_sparse",
        )
    except DataConfigError as exc:
        errors.append(str(exc))
    if selection.get("expand_grid") is not False:
        errors.append("selection grid expansion must remain disabled")

    knn = _mapping(config.get("knn"), "knn")
    if (
        int(knn.get("k", 0)) != 3
        or knn.get("distance") != "squared_euclidean"
        or knn.get("raw_reference") != "concatenate_view1_view2"
        or knn.get("normalization") != "none"
    ):
        errors.append("KNN/Raw reference definition changed")

    decision = _mapping(config.get("decision"), "decision")
    expected_decision = {
        "primary_minimum_accuracy_gain": 0.02,
        "primary_minimum_macro_f1_gain": 0.0,
        "auxiliary_accuracy_delta_strictly_greater_than": -0.02,
        "auxiliary_macro_f1_delta_strictly_greater_than": -0.02,
        "alpha_collapse_threshold": 0.05,
    }
    for name, expected in expected_decision.items():
        if float(decision.get(name, np.nan)) != expected:
            errors.append(f"decision.{name} must remain {expected}")
    if errors:
        raise DataConfigError(
            "Invalid P0 AMDR closure configuration:\n- " + "\n- ".join(errors)
        )
    config["_config_path"] = str(config_path)
    config["_config_sha256"] = file_sha256(config_path)
    return config


def select_global_candidate(
    rows: Sequence[Mapping[str, Any]],
    *,
    folds: Sequence[int] = (1, 2, 3),
    require_converged: bool = True,
    minimum_alpha: float = 0.05,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[tuple[float, float], dict[int, Mapping[str, Any]]] = {}
    for row in rows:
        key = (float(row["lambda_manifold"]), float(row["lambda_sparse"]))
        fold = int(row["fold_index"])
        if fold in grouped.setdefault(key, {}):
            raise DataValidationError(f"duplicate candidate result for fold {fold}: {key}")
        grouped[key][fold] = row

    aggregate: list[dict[str, Any]] = []
    expected_folds = tuple(int(fold) for fold in folds)
    for (lambda_manifold, lambda_sparse), by_fold in sorted(grouped.items()):
        if tuple(sorted(by_fold)) != expected_folds:
            raise DataValidationError(
                f"candidate {lambda_manifold}/{lambda_sparse} lacks required folds"
            )
        fold_rows = [by_fold[fold] for fold in expected_folds]
        all_converged = all(row.get("converged") is True for row in fold_rows)
        minimum_observed_alpha = min(
            min(float(value) for value in row["alpha"]) for row in fold_rows
        )
        no_alpha_collapse = minimum_observed_alpha >= float(minimum_alpha)
        eligible = (all_converged or not require_converged) and no_alpha_collapse
        aggregate.append(
            {
                "candidate_id": (
                    f"lambda_manifold_{lambda_manifold:g}"
                    f"__lambda_sparse_{lambda_sparse:g}"
                ),
                "lambda_manifold": lambda_manifold,
                "lambda_sparse": lambda_sparse,
                "folds": list(expected_folds),
                "all_folds_converged": all_converged,
                "minimum_alpha_across_folds": minimum_observed_alpha,
                "alpha_collapse": not no_alpha_collapse,
                "eligible": eligible,
                "mean_calibration_accuracy": float(
                    np.mean([float(row["calibration_accuracy"]) for row in fold_rows])
                ),
                "mean_calibration_macro_f1": float(
                    np.mean([float(row["calibration_macro_f1"]) for row in fold_rows])
                ),
            }
        )
    eligible_rows = [row for row in aggregate if row["eligible"]]
    if not eligible_rows:
        raise DataValidationError(
            "no P0 closure candidate converged on every selection fold without alpha collapse"
        )
    eligible_rows.sort(
        key=lambda row: (
            -float(row["mean_calibration_accuracy"]),
            -float(row["mean_calibration_macro_f1"]),
            float(row["lambda_manifold"]),
            float(row["lambda_sparse"]),
        )
    )
    return dict(eligible_rows[0]), aggregate


def decide_backbone_status(
    *,
    converged: bool,
    alpha: Sequence[float],
    amdr_accuracy: float,
    amdr_macro_f1: float,
    raw_accuracy: float,
    raw_macro_f1: float,
    alpha_threshold: float = 0.05,
) -> dict[str, Any]:
    minimum_alpha = min(float(value) for value in alpha)
    accuracy_delta = float(amdr_accuracy) - float(raw_accuracy)
    macro_f1_delta = float(amdr_macro_f1) - float(raw_macro_f1)
    stable = bool(converged) and minimum_alpha >= float(alpha_threshold)
    primary_accuracy = accuracy_delta > 0.02 or np.isclose(
        accuracy_delta, 0.02, rtol=0.0, atol=1.0e-12
    )
    primary_macro_f1 = macro_f1_delta > 0.0 or np.isclose(
        macro_f1_delta, 0.0, rtol=0.0, atol=1.0e-12
    )
    auxiliary_accuracy = accuracy_delta - (-0.02) > 1.0e-12
    auxiliary_macro_f1 = macro_f1_delta - (-0.02) > 1.0e-12
    if stable and primary_accuracy and primary_macro_f1:
        status = PRIMARY
        reason = "AMDR met the preregistered primary-backbone gains"
    elif stable and auxiliary_accuracy and auxiliary_macro_f1:
        status = AUXILIARY
        reason = "AMDR was stable and within the preregistered auxiliary margins"
    else:
        status = REJECT
        reason = "AMDR failed convergence, alpha, or Raw-KNN noninferiority gates"
    return {
        "decision": status,
        "reason": reason,
        "converged": bool(converged),
        "minimum_alpha": minimum_alpha,
        "alpha_collapse": minimum_alpha < float(alpha_threshold),
        "amdr_accuracy": float(amdr_accuracy),
        "raw_accuracy": float(raw_accuracy),
        "accuracy_delta": accuracy_delta,
        "amdr_macro_f1": float(amdr_macro_f1),
        "raw_macro_f1": float(raw_macro_f1),
        "macro_f1_delta": macro_f1_delta,
        "decision_threshold_is_statistical_significance": False,
    }


def _labels(
    pairs: Sequence[TwoViewPair], class_to_label: Mapping[str, int]
) -> np.ndarray:
    labels = np.asarray(
        [class_to_label.get(pair.class_name, -1) for pair in pairs], dtype=np.int64
    )
    if np.any(labels < 0):
        raise DataValidationError("unknown class entered the P0 closure")
    return labels


def _split_pairs(
    pairs: Sequence[TwoViewPair], split: str
) -> tuple[TwoViewPair, ...]:
    selected = tuple(pair for pair in pairs if pair.split == split)
    if not selected:
        raise DataValidationError(f"P0 closure split is empty: {split}")
    return selected


def _prepare_fold(
    bundle: ProcessedBundle,
    config: Mapping[str, Any],
    fold_index: int,
    fold_root: Path,
) -> dict[str, Any]:
    protocol = _mapping(config["protocol"], "protocol")
    sampling = _mapping(config["sampling"], "sampling")
    pair_counts = {
        key: int(value)
        for key, value in _mapping(sampling["pairs_per_class"], "pairs_per_class").items()
    }
    protocol_id = f"{protocol['protocol_id_prefix']}_fold{fold_index}"
    pairs, audit = build_fold_pairs(
        bundle,
        protocol_id=protocol_id,
        fold_index=fold_index,
        fold_count=int(protocol["fold_count"]),
        base_seed=int(sampling["base_seed"]),
        pairs_per_class=pair_counts,
        slot_order=str(sampling["slot_order"]),
        included_splits=("train", "calibration"),
    )
    if any(pair.split == "test" for pair in pairs):
        raise DataValidationError("P0 closure generated a forbidden test pair")
    fold_root.mkdir(parents=True, exist_ok=True)
    manifest_path = fold_root / "pair_manifest.csv"
    if manifest_path.exists():
        existing_hash = file_sha256(manifest_path)
        temporary = fold_root / "pair_manifest.expected.csv"
        write_pair_manifest(temporary, pairs)
        expected_hash = file_sha256(temporary)
        temporary.unlink()
        if existing_hash != expected_hash:
            raise DataValidationError(f"fold {fold_index} pair manifest changed")
    else:
        write_pair_manifest(manifest_path, pairs)
    _write_json(fold_root / "pair_audit.json", audit)

    split_pairs = {
        split: _split_pairs(pairs, split) for split in ("train", "calibration")
    }
    transform = str(_mapping(config["preprocessing"], "preprocessing")["transform"])
    split_views = {
        split: materialize_pair_views(bundle, split_pairs[split], transform=transform)
        for split in split_pairs
    }
    known_classes = tuple(sorted(bundle.known_classes))
    class_to_label = {name: index for index, name in enumerate(known_classes)}
    split_labels = {
        split: _labels(split_pairs[split], class_to_label) for split in split_pairs
    }
    return {
        "fold_index": fold_index,
        "pairs": pairs,
        "split_pairs": split_pairs,
        "split_views": split_views,
        "split_labels": split_labels,
        "known_classes": known_classes,
        "pair_manifest_path": manifest_path,
        "pair_manifest_sha256": file_sha256(manifest_path),
        "train_labels_sha256": _array_sha256(split_labels["train"]),
        "calibration_labels_sha256": _array_sha256(split_labels["calibration"]),
    }


def _model_config(
    config: Mapping[str, Any], lambda_manifold: float, lambda_sparse: float
) -> AMDRModelConfig:
    model = _mapping(config["model"], "model")
    return AMDRModelConfig(
        lambda_manifold=float(lambda_manifold),
        lambda_sparse=float(lambda_sparse),
        max_iterations=int(model["max_iterations"]),
        minimum_iterations=int(model["minimum_iterations"]),
        tolerance=float(model["tolerance"]),
        numerical_epsilon=float(model["numerical_epsilon"]),
        solve_ridge=float(model["solve_ridge"]),
        initialization_seed=int(model["initialization_seed"]),
        convergence_metric=str(model["convergence_metric"]),
        graph_same_base_policy=str(model["graph_same_base_policy"]),
        graph_neighborhood=str(model["graph_neighborhood"]),
        l21_reweighting=str(model["l21_reweighting"]),
        objective_scaling=str(model["objective_scaling"]),
    )


def _candidate_id(lambda_manifold: float, lambda_sparse: float) -> str:
    return (
        f"lambda_manifold_{lambda_manifold:g}".replace(".", "p")
        + "__"
        + f"lambda_sparse_{lambda_sparse:g}".replace(".", "p")
    )


def _prediction_rows(
    pairs: Sequence[TwoViewPair],
    true_labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    *,
    method: str,
) -> list[dict[str, Any]]:
    return [
        {
            "pair_id": pair.pair_id,
            "split": pair.split,
            "class_name": pair.class_name,
            "true_label": int(true_label),
            "predicted_label": int(predicted),
            "mean_k_neighbor_squared_distance": float(score),
            "method": method,
        }
        for pair, true_label, predicted, score in zip(
            pairs, true_labels, predictions, scores, strict=True
        )
    ]


def _load_saved_fit(candidate_root: Path, metrics: Mapping[str, Any]) -> AMDRFitResult:
    with np.load(candidate_root / "model.npz", allow_pickle=False) as archive:
        weights = np.asarray(archive["weights"], dtype=np.float64).copy()
        alpha = np.asarray(archive["alpha"], dtype=np.float64).copy()
        view_dimensions = tuple(int(value) for value in archive["view_dimensions"])
        class_count = int(archive["class_count"].item())
    history = tuple(
        json.loads(line)
        for line in (candidate_root / "training_log.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    )
    return AMDRFitResult(
        weights=weights,
        alpha=alpha,
        view_dimensions=view_dimensions,
        class_count=class_count,
        history=history,
        converged=bool(metrics["converged"]),
        stop_reason=str(metrics["stop_reason"]),
    )


def _fit_candidate(
    *,
    config: Mapping[str, Any],
    fold_data: Mapping[str, Any],
    candidate_root: Path,
    lambda_manifold: float,
    lambda_sparse: float,
    phase: str,
) -> tuple[dict[str, Any], AMDRFitResult]:
    metrics_path = candidate_root / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (
            metrics.get("closure_config_sha256") != config["_config_sha256"]
            or metrics.get("pair_manifest_sha256")
            != fold_data["pair_manifest_sha256"]
        ):
            raise DataValidationError("saved P0 closure candidate identity changed")
        return metrics, _load_saved_fit(candidate_root, metrics)

    candidate_root.mkdir(parents=True, exist_ok=True)
    model_config = _model_config(config, lambda_manifold, lambda_sparse)
    resolved = {
        "closure_id": config["closure_id"],
        "closure_config_sha256": config["_config_sha256"],
        "phase": phase,
        "fold_index": int(fold_data["fold_index"]),
        "result_scope": CLOSURE_SCOPE,
        "test_features_materialized": False,
        "unknown_used": False,
        "pair_manifest_sha256": fold_data["pair_manifest_sha256"],
        "train_labels_sha256": fold_data["train_labels_sha256"],
        "calibration_labels_sha256": fold_data["calibration_labels_sha256"],
        "preprocessing": dict(_mapping(config["preprocessing"], "preprocessing")),
        "sampling": dict(_mapping(config["sampling"], "sampling")),
        "model": asdict(model_config),
        "knn": dict(_mapping(config["knn"], "knn")),
    }
    (candidate_root / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    checkpoint_path = candidate_root / "checkpoint_latest.npz"
    resume_checkpoint = (
        load_amdr_checkpoint(checkpoint_path) if checkpoint_path.exists() else None
    )
    every = int(_mapping(config["checkpoint"], "checkpoint")["every_iterations"])

    def checkpoint_callback(checkpoint: Any) -> None:
        if checkpoint.iteration_completed % every == 0:
            save_amdr_checkpoint(checkpoint_path, checkpoint)

    started = time.perf_counter()
    fit = fit_amdr(
        fold_data["split_views"]["train"],
        fold_data["split_labels"]["train"],
        model_config,
        resume_checkpoint=resume_checkpoint,
        checkpoint_callback=checkpoint_callback,
    )
    train_projection = project_views(fold_data["split_views"]["train"], fit)
    calibration_projection = project_views(
        fold_data["split_views"]["calibration"], fit
    )
    prediction, scores = knn_predict_and_score(
        train_projection,
        fold_data["split_labels"]["train"],
        calibration_projection,
        k=int(_mapping(config["knn"], "knn")["k"]),
    )
    labels = fold_data["split_labels"]["calibration"]
    metrics = {
        "closure_config_sha256": config["_config_sha256"],
        "phase": phase,
        "fold_index": int(fold_data["fold_index"]),
        "candidate_id": _candidate_id(lambda_manifold, lambda_sparse),
        "lambda_manifold": float(lambda_manifold),
        "lambda_sparse": float(lambda_sparse),
        "converged": fit.converged,
        "stop_reason": fit.stop_reason,
        "iterations_ran": len(fit.history),
        "alpha": fit.alpha.tolist(),
        "minimum_alpha": float(np.min(fit.alpha)),
        "alpha_collapse": float(np.min(fit.alpha)) < 0.05,
        "weight_frobenius_norm": float(np.linalg.norm(fit.weights)),
        "calibration_accuracy": accuracy_score(labels, prediction),
        "calibration_macro_f1": macro_f1_score(
            labels, prediction, labels=range(len(fold_data["known_classes"]))
        ),
        "pair_manifest_sha256": fold_data["pair_manifest_sha256"],
        "train_labels_sha256": fold_data["train_labels_sha256"],
        "calibration_labels_sha256": fold_data["calibration_labels_sha256"],
        "test_pairs_generated": False,
        "test_features_materialized": False,
        "test_metrics_used": False,
        "wall_time_seconds": time.perf_counter() - started,
    }
    np.savez_compressed(
        candidate_root / "model.npz",
        weights=fit.weights,
        alpha=fit.alpha,
        view_dimensions=np.asarray(fit.view_dimensions, dtype=np.int64),
        class_count=np.asarray(fit.class_count, dtype=np.int64),
        known_classes=np.asarray(fold_data["known_classes"]),
    )
    with (candidate_root / "training_log.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in fit.history:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    _write_csv(
        candidate_root / "calibration_predictions.csv",
        _prediction_rows(
            fold_data["split_pairs"]["calibration"],
            labels,
            prediction,
            scores,
            method="amdr",
        ),
    )
    _write_json(metrics_path, metrics)
    _write_json(
        candidate_root / "artifact_hashes.json",
        _recursive_hashes(candidate_root, {"artifact_hashes.json"}),
    )
    return metrics, fit


def _environment() -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[3]
    peak_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    peak_rss_mib = (
        peak_rss / (1024.0 * 1024.0)
        if platform.system() == "Darwin"
        else peak_rss / 1024.0
    )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "resource_limits": _resource_limits(),
        "peak_rss_mib": peak_rss_mib,
        "git": _git_state(project_root),
    }


def run_p0_closure(
    *,
    config_path: str | Path,
    bundle_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_closure_config(config_path)
    destination = Path(output_root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    identity_path = destination / "closure_identity.json"
    identity = {
        "closure_id": config["closure_id"],
        "closure_config_sha256": config["_config_sha256"],
    }
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise DataValidationError("P0 closure output belongs to another config")
    else:
        if any(destination.iterdir()):
            raise DataValidationError("non-empty P0 closure output has no identity")
        _write_json(identity_path, identity)

    final_path = destination / "final_decision.json"
    if final_path.exists():
        return json.loads(final_path.read_text(encoding="utf-8"))

    bundle_config = _mapping(config["bundle"], "bundle")
    bundle = load_processed_bundle(
        bundle_root,
        expected_profiles_sha256=str(bundle_config["profiles_sha256"]),
        expected_manifest_sha256=str(bundle_config["manifest_sha256"]),
        expected_bundle_sha256=str(bundle_config["bundle_sha256"]),
    )
    selection = _mapping(config["selection"], "selection")
    lambda_manifold_values = [float(value) for value in selection["lambda_manifold"]]
    lambda_sparse_values = [float(value) for value in selection["lambda_sparse"]]
    stage_a_rows: list[dict[str, Any]] = []

    for fold_index in (1, 2, 3):
        fold_root = destination / "stage_a" / f"fold_{fold_index}" / "diagnostic_known_only"
        fold_data = _prepare_fold(bundle, config, fold_index, fold_root)
        for lambda_manifold in lambda_manifold_values:
            for lambda_sparse in lambda_sparse_values:
                candidate_root = (
                    fold_root
                    / "candidates"
                    / _candidate_id(lambda_manifold, lambda_sparse)
                )
                metrics, _ = _fit_candidate(
                    config=config,
                    fold_data=fold_data,
                    candidate_root=candidate_root,
                    lambda_manifold=lambda_manifold,
                    lambda_sparse=lambda_sparse,
                    phase="stage_a_diagnostic_known_only",
                )
                stage_a_rows.append(metrics)

    try:
        selected, aggregate = select_global_candidate(
            stage_a_rows,
            folds=(1, 2, 3),
            require_converged=bool(selection["require_converged"]),
            minimum_alpha=float(selection["minimum_alpha"]),
        )
    except DataValidationError as exc:
        aggregate = []
        grouped_keys = sorted(
            {
                (float(row["lambda_manifold"]), float(row["lambda_sparse"]))
                for row in stage_a_rows
            }
        )
        for lambda_manifold, lambda_sparse in grouped_keys:
            candidate_rows = [
                row
                for row in stage_a_rows
                if float(row["lambda_manifold"]) == lambda_manifold
                and float(row["lambda_sparse"]) == lambda_sparse
            ]
            aggregate.append(
                {
                    "candidate_id": _candidate_id(lambda_manifold, lambda_sparse),
                    "lambda_manifold": lambda_manifold,
                    "lambda_sparse": lambda_sparse,
                    "all_folds_converged": all(
                        row.get("converged") is True for row in candidate_rows
                    ),
                    "minimum_alpha_across_folds": min(
                        min(float(value) for value in row["alpha"])
                        for row in candidate_rows
                    ),
                    "eligible": False,
                }
            )
        _write_csv(destination / "stage_a" / "candidate_aggregate.csv", aggregate)
        _write_json(destination / "stage_a" / "candidate_aggregate.json", aggregate)
        final = {
            "closure_id": config["closure_id"],
            "decision": REJECT,
            "decision_stage": "stage_a",
            "reason": str(exc),
            "fold0_used": False,
            "unknown_used": False,
            "test_pairs_generated": False,
            "test_features_materialized": False,
            "p1_started": False,
            "wall_time_seconds": time.perf_counter() - started,
        }
        _write_json(destination / "environment.json", _environment())
        _write_json(final_path, final)
        _write_json(
            destination / "artifact_hashes.json",
            _recursive_hashes(destination, {"artifact_hashes.json"}),
        )
        return final

    _write_csv(destination / "stage_a" / "candidate_aggregate.csv", aggregate)
    _write_json(destination / "stage_a" / "candidate_aggregate.json", aggregate)
    _write_json(destination / "stage_a" / "selected_candidate.json", selected)

    fold4_root = destination / "stage_b" / "fold_4" / "confirmatory_known_only"
    fold4_data = _prepare_fold(bundle, config, 4, fold4_root)
    amdr_root = fold4_root / "amdr"
    amdr_metrics, amdr_fit = _fit_candidate(
        config=config,
        fold_data=fold4_data,
        candidate_root=amdr_root,
        lambda_manifold=float(selected["lambda_manifold"]),
        lambda_sparse=float(selected["lambda_sparse"]),
        phase="stage_b_confirmatory_known_only",
    )

    raw_train = np.concatenate(fold4_data["split_views"]["train"], axis=1)
    raw_calibration = np.concatenate(
        fold4_data["split_views"]["calibration"], axis=1
    )
    raw_prediction, raw_scores = knn_predict_and_score(
        raw_train,
        fold4_data["split_labels"]["train"],
        raw_calibration,
        k=int(_mapping(config["knn"], "knn")["k"]),
    )
    calibration_labels = fold4_data["split_labels"]["calibration"]
    raw_accuracy = accuracy_score(calibration_labels, raw_prediction)
    raw_macro_f1 = macro_f1_score(
        calibration_labels,
        raw_prediction,
        labels=range(len(fold4_data["known_classes"])),
    )
    raw_root = fold4_root / "raw_two_view_concatenation_knn"
    raw_root.mkdir(parents=True, exist_ok=True)
    _write_csv(
        raw_root / "calibration_predictions.csv",
        _prediction_rows(
            fold4_data["split_pairs"]["calibration"],
            calibration_labels,
            raw_prediction,
            raw_scores,
            method="raw_two_view_concatenation_knn",
        ),
    )
    raw_metrics = {
        "fold_index": 4,
        "calibration_accuracy": raw_accuracy,
        "calibration_macro_f1": raw_macro_f1,
        "pair_manifest_sha256": fold4_data["pair_manifest_sha256"],
        "train_labels_sha256": fold4_data["train_labels_sha256"],
        "calibration_labels_sha256": fold4_data["calibration_labels_sha256"],
        "k": 3,
        "distance": "squared_euclidean",
        "normalization": "none",
        "test_pairs_generated": False,
        "test_features_materialized": False,
    }
    _write_json(raw_root / "metrics.json", raw_metrics)
    (raw_root / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            {
                "closure_config_sha256": config["_config_sha256"],
                "phase": "stage_b_confirmatory_known_only",
                "fold_index": 4,
                "method": "raw_two_view_concatenation_knn",
                "input": "concatenate_view1_view2",
                "transform": PEAK_RELATIVE_AMPLITUDE_TRANSFORM_ID,
                "k": 3,
                "distance": "squared_euclidean",
                "normalization": "none",
                "pair_manifest_sha256": fold4_data["pair_manifest_sha256"],
                "train_labels_sha256": fold4_data["train_labels_sha256"],
                "calibration_labels_sha256": fold4_data[
                    "calibration_labels_sha256"
                ],
                "test_features_materialized": False,
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _write_json(
        raw_root / "artifact_hashes.json",
        _recursive_hashes(raw_root, {"artifact_hashes.json"}),
    )

    decision = decide_backbone_status(
        converged=bool(amdr_metrics["converged"]),
        alpha=amdr_fit.alpha,
        amdr_accuracy=float(amdr_metrics["calibration_accuracy"]),
        amdr_macro_f1=float(amdr_metrics["calibration_macro_f1"]),
        raw_accuracy=raw_accuracy,
        raw_macro_f1=raw_macro_f1,
        alpha_threshold=float(_mapping(config["decision"], "decision")["alpha_collapse_threshold"]),
    )
    comparison = {
        "fold_index": 4,
        "selected_candidate": selected,
        "amdr": amdr_metrics,
        "raw": raw_metrics,
        "fairness_checks": {
            "same_pair_manifest": (
                amdr_metrics["pair_manifest_sha256"]
                == raw_metrics["pair_manifest_sha256"]
            ),
            "same_train_label_order": (
                amdr_metrics["train_labels_sha256"]
                == raw_metrics["train_labels_sha256"]
            ),
            "same_calibration_label_order": (
                amdr_metrics["calibration_labels_sha256"]
                == raw_metrics["calibration_labels_sha256"]
            ),
            "same_knn_k": int(_mapping(config["knn"], "knn")["k"]) == 3,
            "test_pairs_generated": False,
            "test_features_materialized": False,
        },
        "decision": decision,
    }
    if not all(
        value is True
        for name, value in comparison["fairness_checks"].items()
        if name.startswith("same_")
    ):
        raise DataValidationError("fold 4 AMDR/Raw fairness identity check failed")
    _write_json(fold4_root / "comparison.json", comparison)

    final = {
        "closure_id": config["closure_id"],
        "closure_config_sha256": config["_config_sha256"],
        "decision": decision["decision"],
        "decision_stage": "stage_b_fold4",
        "selected_candidate": selected,
        "fold4_comparison": comparison,
        "fold0_used": False,
        "unknown_used": False,
        "test_pairs_generated": False,
        "test_features_materialized": False,
        "test_metrics_used": False,
        "p1_started": False,
        "wall_time_seconds": time.perf_counter() - started,
    }
    _write_json(destination / "environment.json", _environment())
    _write_json(final_path, final)
    _write_json(
        destination / "artifact_hashes.json",
        _recursive_hashes(destination, {"artifact_hashes.json"}),
    )
    return final


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the preregistered AMDR P0 closure on known odd-angle folds"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--bundle-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = run_p0_closure(
        config_path=args.config,
        bundle_root=args.bundle_root,
        output_root=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
