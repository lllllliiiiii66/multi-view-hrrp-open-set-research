from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
import yaml

from hrrp_osr.amdr.data import (
    CANONICAL_SLOT_ORDER,
    PEAK_RELATIVE_AMPLITUDE_TRANSFORM_ID,
    PEAK_RELATIVE_POWER_TRANSFORM_ID,
    RANDOMIZED_SLOT_ORDER,
    TwoViewPair,
    build_fold_pairs,
    materialize_pair_views,
    write_pair_manifest,
)
from hrrp_osr.amdr.model import (
    AMDR_ALGORITHM_VERSION,
    RELATIVE_STATE_CHANGE,
    AMDRCheckpoint,
    AMDRModelConfig,
    fit_amdr,
    knn_predict_and_score,
    load_amdr_checkpoint,
    prune_amdr_weight_rows,
    project_views,
    save_amdr_checkpoint,
)
from hrrp_osr.data.errors import DataConfigError, DataValidationError
from hrrp_osr.data.manifest import file_sha256
from hrrp_osr.data.processed import load_processed_bundle
from hrrp_osr.evaluation.metrics import (
    accuracy_score,
    evaluate_open_set,
    macro_f1_score,
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DataConfigError(f"{name} must be a mapping")
    return value


def load_smoke_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    config = dict(_mapping(raw, "AMDR smoke config"))
    errors: list[str] = []
    if int(config.get("schema_version", 0)) != 1:
        errors.append("schema_version must be 1")
    result_scope = str(config.get("result_scope"))
    if config.get("stage") != "P0" or config.get(
        "protocol_family"
    ) != "amdr_odd_even_two_view_crossfit_v1":
        errors.append("stage/protocol_family do not match the P0 AMDR diagnostic")
    if result_scope not in {
        "diagnostic_smoke",
        "diagnostic_convergence",
        "diagnostic_pilot",
    }:
        errors.append("result_scope is not an allowed P0 AMDR diagnostic")
    bundle = _mapping(config.get("bundle"), "bundle")
    for name in ("profiles_sha256", "manifest_sha256", "bundle_sha256"):
        value = bundle.get(name)
        if not isinstance(value, str) or len(value) != 64:
            errors.append(f"bundle.{name} must be a SHA-256")
    protocol = _mapping(config.get("protocol"), "protocol")
    if int(protocol.get("frame_width_deg", 0)) != 15:
        errors.append("frame_width_deg must be 15")
    if int(protocol.get("fold_count", 0)) != 5:
        errors.append("fold_count must be 5")
    if int(protocol.get("fold_index", -1)) not in range(5):
        errors.append("fold_index must be in 0..4")
    if (
        protocol.get("development_angle_parity"),
        protocol.get("test_angle_parity"),
        int(protocol.get("known_class_count", 0)),
        int(protocol.get("unknown_class_count", 0)),
    ) != ("odd", "even", 7, 3):
        errors.append("odd/even and 7/3 protocol invariants are invalid")
    sampling = _mapping(config.get("sampling"), "sampling")
    counts = _mapping(sampling.get("pairs_per_class"), "sampling.pairs_per_class")
    expected_pairs = 500 if result_scope == "diagnostic_pilot" else 100
    for split in ("train", "calibration", "test"):
        if int(counts.get(split, 0)) != expected_pairs:
            errors.append(
                f"{result_scope} requires {expected_pairs} {split} pairs per class"
            )
    sampling_algorithm = sampling.get("algorithm")
    expected_slot_order = {
        "uniform_ordered_cross_frame_balanced_base_usage": RANDOMIZED_SLOT_ORDER,
        "uniform_canonical_cross_frame_balanced_base_usage": CANONICAL_SLOT_ORDER,
    }.get(sampling_algorithm)
    slot_order = sampling.get("slot_order", RANDOMIZED_SLOT_ORDER)
    if (
        expected_slot_order is None
        or slot_order != expected_slot_order
        or sampling.get("duplicate_unordered_pairs") is not False
    ):
        errors.append("sampling invariants are invalid")
    preprocessing = _mapping(config.get("preprocessing"), "preprocessing")
    if preprocessing.get("transform") not in {
        PEAK_RELATIVE_POWER_TRANSFORM_ID,
        PEAK_RELATIVE_AMPLITUDE_TRANSFORM_ID,
    }:
        errors.append("smoke preprocessing transform is invalid")
    model = _mapping(config.get("model"), "model")
    if model.get("algorithm_version") != AMDR_ALGORITHM_VERSION:
        errors.append("model algorithm_version is invalid")
    expected_implementation_scope = {
        "diagnostic_smoke": "amdr_research_v1_diagnostic_budget",
        "diagnostic_convergence": "amdr_research_v1_convergence_diagnostic",
        "diagnostic_pilot": {
            "amdr_research_v1_pilot",
            "amdr_research_v1_alignment_diagnostic",
        },
    }.get(result_scope)
    implementation_scope = model.get("implementation_scope")
    if isinstance(expected_implementation_scope, set):
        scope_valid = implementation_scope in expected_implementation_scope
    else:
        scope_valid = implementation_scope == expected_implementation_scope
    if not scope_valid:
        errors.append("model implementation_scope is invalid")
    if model.get("convergence_metric") != RELATIVE_STATE_CHANGE:
        errors.append("model convergence_metric is invalid")
    max_iterations = int(model.get("max_iterations", 0))
    max_allowed = 5 if result_scope == "diagnostic_smoke" else 300
    if not 1 <= max_iterations <= max_allowed:
        errors.append(f"max_iterations must be in 1..{max_allowed}")
    if not 1 <= int(model.get("minimum_iterations", 3)) <= int(
        model.get("max_iterations", 0)
    ):
        errors.append("model.minimum_iterations is invalid")
    row_prune_threshold = float(
        model.get("post_training_row_prune_squared_norm_threshold", 0.0)
    )
    if row_prune_threshold not in {0.0, 1.0e-5}:
        errors.append("post-training row-prune threshold must be 0 or 1e-5")
    checkpoint = _mapping(config.get("checkpoint", {}), "checkpoint")
    if checkpoint:
        if checkpoint.get("strategy") != "latest_only_atomic_replace":
            errors.append("checkpoint.strategy must be latest_only_atomic_replace")
        if int(checkpoint.get("every_iterations", 0)) < 1:
            errors.append("checkpoint.every_iterations must be positive")
        if checkpoint.get("selection") != "none_no_validation_best":
            errors.append("checkpoint selection must not use validation performance")
    knn = _mapping(config.get("knn"), "knn")
    if int(knn.get("k", 0)) not in {1, 3, 5, 7, 9}:
        errors.append("KNN k must be one of 1/3/5/7/9")
    if float(knn.get("known_acceptance_rate", 0.0)) != 0.95:
        errors.append("known acceptance rate must be 0.95")
    if errors:
        raise DataConfigError(
            "Invalid AMDR smoke configuration:\n- " + "\n- ".join(errors)
        )
    config["_config_path"] = str(config_path)
    config["_config_sha256"] = file_sha256(config_path)
    return config


def _git_state(project_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "dirty": bool(run("status", "--short")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None}


def _read_system_text(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None


def _resource_limits() -> dict[str, Any]:
    host_cpu_count = os.cpu_count()
    quota_cores: float | None = None
    memory_limit_bytes: int | None = None
    cgroup_version: int | None = None

    quota_v1 = _read_system_text("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period_v1 = _read_system_text("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    memory_v1 = _read_system_text("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if quota_v1 is not None and period_v1 is not None:
        cgroup_version = 1
        quota = int(quota_v1)
        period = int(period_v1)
        if quota > 0 and period > 0:
            quota_cores = quota / period
    if memory_v1 is not None:
        parsed = int(memory_v1)
        if 0 < parsed < 2**60:
            memory_limit_bytes = parsed

    cpu_v2 = _read_system_text("/sys/fs/cgroup/cpu.max")
    memory_v2 = _read_system_text("/sys/fs/cgroup/memory.max")
    if cpu_v2 is not None:
        cgroup_version = 2
        quota_text, period_text = cpu_v2.split()
        if quota_text != "max":
            quota = int(quota_text)
            period = int(period_text)
            if quota > 0 and period > 0:
                quota_cores = quota / period
    if memory_v2 not in (None, "max"):
        parsed = int(memory_v2)
        if parsed > 0:
            memory_limit_bytes = parsed

    effective_cpu_limit = (
        min(float(host_cpu_count), quota_cores)
        if host_cpu_count is not None and quota_cores is not None
        else quota_cores or host_cpu_count
    )
    return {
        "cgroup_version": cgroup_version,
        "host_logical_cpu_count": host_cpu_count,
        "cgroup_cpu_quota_cores": quota_cores,
        "effective_cpu_limit": effective_cpu_limit,
        "cgroup_memory_limit_bytes": memory_limit_bytes,
        "cgroup_memory_limit_gib": (
            memory_limit_bytes / (1024.0**3)
            if memory_limit_bytes is not None
            else None
        ),
    }


def _split_pairs(
    pairs: Sequence[TwoViewPair], split: str
) -> tuple[TwoViewPair, ...]:
    selected = tuple(pair for pair in pairs if pair.split == split)
    if not selected:
        raise DataValidationError(f"no pairs found for split {split}")
    return selected


def _labels(
    pairs: Sequence[TwoViewPair], class_to_label: Mapping[str, int]
) -> np.ndarray:
    return np.asarray(
        [class_to_label.get(pair.class_name, -1) for pair in pairs],
        dtype=np.int64,
    )


def _predictions_rows(
    pairs: Sequence[TwoViewPair],
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    scores: np.ndarray,
    *,
    label_to_class: Mapping[int, str],
    threshold: float,
) -> list[dict[str, Any]]:
    rows = []
    for pair, true_label, predicted_label, score in zip(
        pairs,
        true_labels,
        predicted_labels,
        scores,
        strict=True,
    ):
        rows.append(
            {
                "pair_id": pair.pair_id,
                "split": pair.split,
                "class_name": pair.class_name,
                "class_role": pair.class_role,
                "true_label": int(true_label),
                "predicted_label": int(predicted_label),
                "predicted_class": label_to_class[int(predicted_label)],
                "unknown_score": float(score),
                "threshold": float(threshold),
                "rejected": int(float(score) > threshold),
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise DataValidationError(f"cannot write empty CSV {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _artifact_hashes(destination: Path, excluded: set[str]) -> dict[str, str]:
    return {
        path.name: file_sha256(path)
        for path in sorted(destination.iterdir())
        if path.is_file() and path.name not in excluded
    }


def run_smoke(
    *,
    config_path: str | Path,
    bundle_root: str | Path,
    output_root: str | Path,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_smoke_config(config_path)
    destination = Path(output_root).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise DataValidationError(f"output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    bundle_config = _mapping(config["bundle"], "bundle")
    bundle = load_processed_bundle(
        bundle_root,
        expected_profiles_sha256=str(bundle_config["profiles_sha256"]),
        expected_manifest_sha256=str(bundle_config["manifest_sha256"]),
        expected_bundle_sha256=str(bundle_config["bundle_sha256"]),
    )
    protocol = _mapping(config["protocol"], "protocol")
    sampling = _mapping(config["sampling"], "sampling")
    pair_counts = {
        key: int(value)
        for key, value in _mapping(
            sampling["pairs_per_class"], "sampling.pairs_per_class"
        ).items()
    }
    pairs, pair_audit = build_fold_pairs(
        bundle,
        protocol_id=str(config["protocol_id"]),
        fold_index=int(protocol["fold_index"]),
        fold_count=int(protocol["fold_count"]),
        base_seed=int(sampling["base_seed"]),
        pairs_per_class=pair_counts,
        slot_order=str(sampling.get("slot_order", RANDOMIZED_SLOT_ORDER)),
    )
    write_pair_manifest(destination / "pair_manifest.csv", pairs)

    split_pairs = {
        split: _split_pairs(pairs, split)
        for split in ("train", "calibration", "test")
    }
    preprocessing = _mapping(config["preprocessing"], "preprocessing")
    profile_transform = str(preprocessing["transform"])
    split_views = {
        split: materialize_pair_views(
            bundle,
            split_pairs[split],
            transform=profile_transform,
        )
        for split in split_pairs
    }
    known_classes = tuple(sorted(bundle.known_classes))
    class_to_label = {class_name: index for index, class_name in enumerate(known_classes)}
    label_to_class = {index: class_name for class_name, index in class_to_label.items()}
    split_labels = {
        split: _labels(split_pairs[split], class_to_label)
        for split in split_pairs
    }
    if np.any(split_labels["train"] < 0) or np.any(split_labels["calibration"] < 0):
        raise DataValidationError("unknown class entered AMDR fit or calibration")

    model_raw = _mapping(config["model"], "model")
    model_config = AMDRModelConfig(
        lambda_manifold=float(model_raw["lambda_manifold"]),
        lambda_sparse=float(model_raw["lambda_sparse"]),
        max_iterations=int(model_raw["max_iterations"]),
        tolerance=float(model_raw["tolerance"]),
        numerical_epsilon=float(model_raw["numerical_epsilon"]),
        solve_ridge=float(model_raw["solve_ridge"]),
        initialization_seed=int(model_raw["initialization_seed"]),
        minimum_iterations=int(model_raw.get("minimum_iterations", 3)),
        convergence_metric=str(
            model_raw.get("convergence_metric", "relative_state_change_v1")
        ),
    )
    checkpoint_raw = _mapping(config.get("checkpoint", {}), "checkpoint")
    checkpoint_every = int(checkpoint_raw.get("every_iterations", 0))
    latest_checkpoint_path = destination / "checkpoint_latest.npz"
    resume_checkpoint = (
        load_amdr_checkpoint(resume_from) if resume_from is not None else None
    )

    def checkpoint_callback(checkpoint: AMDRCheckpoint) -> None:
        if not checkpoint_raw:
            return
        is_final_budget_iteration = (
            checkpoint.iteration_completed == model_config.max_iterations
        )
        is_converged = (
            checkpoint.iteration_completed >= model_config.minimum_iterations
            and float(checkpoint.history[-1]["convergence_value"])
            < model_config.tolerance
        )
        if (
            checkpoint.iteration_completed % checkpoint_every == 0
            or is_final_budget_iteration
            or is_converged
        ):
            save_amdr_checkpoint(latest_checkpoint_path, checkpoint)

    fit = fit_amdr(
        split_views["train"],
        split_labels["train"],
        model_config,
        resume_checkpoint=resume_checkpoint,
        checkpoint_callback=checkpoint_callback if checkpoint_raw else None,
    )
    row_prune_threshold = float(
        model_raw.get("post_training_row_prune_squared_norm_threshold", 0.0)
    )
    fit, pruned_weight_row_count = prune_amdr_weight_rows(
        fit,
        squared_row_norm_threshold=row_prune_threshold,
    )
    projections = {
        split: project_views(split_views[split], fit)
        for split in split_views
    }
    knn_raw = _mapping(config["knn"], "knn")
    k = int(knn_raw["k"])
    predictions: dict[str, np.ndarray] = {}
    scores: dict[str, np.ndarray] = {}
    for split in ("calibration", "test"):
        predictions[split], scores[split] = knn_predict_and_score(
            projections["train"],
            split_labels["train"],
            projections[split],
            k=k,
        )
    test_known_mask = split_labels["test"] >= 0
    test_unknown_mask = ~test_known_mask
    metrics = evaluate_open_set(
        known_true=split_labels["test"][test_known_mask],
        known_pred=predictions["test"][test_known_mask],
        known_unknown_scores=scores["test"][test_known_mask],
        unknown_pred=predictions["test"][test_unknown_mask],
        unknown_unknown_scores=scores["test"][test_unknown_mask],
        known_validation_scores=scores["calibration"],
        known_class_count=len(known_classes),
        known_acceptance_rate=float(knn_raw["known_acceptance_rate"]),
    )
    metrics.update(
        {
            "result_scope": str(config["result_scope"]),
            "formal_experiment": False,
            "fold_index": int(protocol["fold_index"]),
            "iterations_ran": len(fit.history),
            "converged_within_configured_budget": fit.converged,
            "optimization_stop_reason": fit.stop_reason,
            "resumed_from_iteration": (
                resume_checkpoint.iteration_completed
                if resume_checkpoint is not None
                else 0
            ),
            "checkpoint_policy": (
                str(checkpoint_raw.get("strategy")) if checkpoint_raw else "disabled"
            ),
            "calibration_accuracy": accuracy_score(
                split_labels["calibration"], predictions["calibration"]
            ),
            "calibration_macro_f1": macro_f1_score(
                split_labels["calibration"],
                predictions["calibration"],
                labels=range(len(known_classes)),
            ),
            "pair_counts": {
                split: len(split_pairs[split]) for split in split_pairs
            },
            "base_profile_transform": profile_transform,
            "post_training_row_prune_squared_norm_threshold": row_prune_threshold,
            "pruned_weight_row_count": pruned_weight_row_count,
            "slot_order": str(sampling.get("slot_order", RANDOMIZED_SLOT_ORDER)),
        }
    )
    _write_json(destination / "metrics.json", metrics)
    _write_json(destination / "pair_audit.json", pair_audit)
    with (destination / "training_log.jsonl").open("w", encoding="utf-8") as handle:
        for row in fit.history:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    np.savez_compressed(
        destination / "model.npz",
        weights=fit.weights,
        alpha=fit.alpha,
        known_classes=np.asarray(known_classes),
        post_training_row_prune_squared_norm_threshold=np.asarray(
            row_prune_threshold
        ),
        pruned_weight_row_count=np.asarray(pruned_weight_row_count),
    )
    np.savez_compressed(
        destination / "projections.npz",
        train=projections["train"],
        calibration=projections["calibration"],
        test=projections["test"],
        train_labels=split_labels["train"],
        calibration_labels=split_labels["calibration"],
        test_labels=split_labels["test"],
    )
    threshold = float(metrics["threshold"])
    prediction_rows = []
    prediction_rows.extend(
        _predictions_rows(
            split_pairs["calibration"],
            split_labels["calibration"],
            predictions["calibration"],
            scores["calibration"],
            label_to_class=label_to_class,
            threshold=threshold,
        )
    )
    prediction_rows.extend(
        _predictions_rows(
            split_pairs["test"],
            split_labels["test"],
            predictions["test"],
            scores["test"],
            label_to_class=label_to_class,
            threshold=threshold,
        )
    )
    _write_csv(destination / "predictions.csv", prediction_rows)

    project_root = Path(__file__).resolve().parents[3]
    peak_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        peak_rss_mib = peak_rss / (1024.0 * 1024.0)
    else:
        peak_rss_mib = peak_rss / 1024.0
    numpy_config = getattr(np.__config__, "CONFIG", {})
    blas = (
        numpy_config.get("Build Dependencies", {}).get("blas", {})
        if isinstance(numpy_config, Mapping)
        else {}
    )
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "resource_limits": _resource_limits(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numpy_blas": dict(blas) if isinstance(blas, Mapping) else {},
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "peak_rss_mib": peak_rss_mib,
        "git": _git_state(project_root),
        "wall_time_seconds": time.perf_counter() - started,
    }
    _write_json(destination / "environment.json", environment)
    resolved = dict(config)
    resolved["_resolved"] = {
        "bundle_root": str(Path(bundle_root).resolve()),
        "output_root": str(destination),
        "profiles_sha256": bundle.profiles_sha256,
        "manifest_sha256": bundle.manifest_sha256,
        "bundle_sha256": bundle.bundle_sha256,
        "known_classes": list(known_classes),
    }
    (destination / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    hashes = _artifact_hashes(destination, {"artifact_hashes.json"})
    _write_json(destination / "artifact_hashes.json", hashes)
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the diagnostic Python AMDR smoke")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--bundle-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Resume deterministically from a latest AMDR checkpoint into a new output directory",
    )
    args = parser.parse_args(argv)
    metrics = run_smoke(
        config_path=args.config,
        bundle_root=args.bundle_root,
        output_root=args.output,
        resume_from=args.resume_from,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
