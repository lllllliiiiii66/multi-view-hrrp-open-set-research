from __future__ import annotations

import argparse
import csv
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
from scipy.io import loadmat

from hrrp_osr.amdr.model import (
    ALLOW_SAME_BASE_GRAPH,
    AMDR_ALGORITHM_VERSION,
    RELATIVE_STATE_CHANGE,
    AMDRCheckpoint,
    AMDRModelConfig,
    fit_amdr,
    load_amdr_checkpoint,
    pairwise_squared_distances,
    project_views,
    prune_amdr_weight_rows,
    save_amdr_checkpoint,
)
from hrrp_osr.data.errors import DataConfigError, DataValidationError
from hrrp_osr.data.manifest import file_sha256
from hrrp_osr.evaluation.metrics import accuracy_score, macro_f1_score


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DataConfigError(f"{name} must be a mapping")
    return value


def load_attachment_ship_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as handle:
        config = dict(_mapping(yaml.safe_load(handle), "ship attachment config"))
    errors: list[str] = []
    if int(config.get("schema_version", 0)) != 1:
        errors.append("schema_version must be 1")
    if config.get("stage") != "P0" or config.get("result_scope") != (
        "diagnostic_attachment_ship"
    ):
        errors.append("stage/result_scope do not identify the ship diagnostic")
    dataset = _mapping(config.get("dataset"), "dataset")
    if not isinstance(dataset.get("file_sha256"), str) or len(
        str(dataset.get("file_sha256"))
    ) != 64:
        errors.append("dataset.file_sha256 must be a SHA-256")
    if (
        int(dataset.get("view_count", 0)),
        list(dataset.get("feature_dimensions", [])),
        int(dataset.get("class_count", 0)),
        int(dataset.get("train_sample_count", 0)),
        int(dataset.get("test_sample_count", 0)),
    ) != (2, [200, 200], 5, 12015, 8010):
        errors.append("ship attachment dimensions or counts changed")
    model = _mapping(config.get("model"), "model")
    if model.get("algorithm_version") != AMDR_ALGORITHM_VERSION:
        errors.append("model algorithm version is invalid")
    if model.get("implementation_scope") != (
        "python_method_on_attachment_ship_diagnostic"
    ):
        errors.append("model implementation scope is invalid")
    if model.get("convergence_metric") != RELATIVE_STATE_CHANGE:
        errors.append("model convergence metric is invalid")
    if model.get("graph_same_base_policy") != ALLOW_SAME_BASE_GRAPH:
        errors.append("attachment diagnostic must preserve the existing graph")
    if not 1 <= int(model.get("max_iterations", 0)) <= 300:
        errors.append("max_iterations must be in 1..300")
    if float(model.get("post_training_row_prune_squared_norm_threshold", -1)) not in {
        0.0,
        1.0e-5,
    }:
        errors.append("row-prune threshold is invalid")
    checkpoint = _mapping(config.get("checkpoint"), "checkpoint")
    if checkpoint.get("strategy") != "latest_only_atomic_replace":
        errors.append("checkpoint strategy is invalid")
    if int(checkpoint.get("every_iterations", 0)) < 1:
        errors.append("checkpoint interval must be positive")
    if checkpoint.get("selection") != "none_no_test_best":
        errors.append("checkpoint must not select by test performance")
    knn = _mapping(config.get("knn"), "knn")
    if list(knn.get("k_values", [])) != [1, 3, 5, 7, 9]:
        errors.append("ship diagnostic must report K=1/3/5/7/9")
    if knn.get("selection") != "none_report_all_test_results":
        errors.append("KNN test results must not be used for selection")
    if errors:
        raise DataConfigError(
            "Invalid ship attachment configuration:\n- " + "\n- ".join(errors)
        )
    config["_config_path"] = str(source)
    config["_config_sha256"] = file_sha256(source)
    return config


def load_attachment_ship_data(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_train_count: int,
    expected_test_count: int,
    expected_dimensions: Sequence[int],
    expected_class_count: int,
) -> tuple[tuple[np.ndarray, ...], np.ndarray, tuple[np.ndarray, ...], np.ndarray]:
    source = Path(path).resolve()
    if not source.is_file():
        raise DataValidationError(f"ship attachment MAT does not exist: {source}")
    if file_sha256(source) != expected_sha256:
        raise DataValidationError("ship attachment MAT SHA-256 changed")
    try:
        raw = loadmat(source)
        train_cell = raw["train_data"]
        test_cell = raw["test_data"]
        train_labels = np.ravel(raw["trainlabel"]).astype(np.int64) - 1
        test_labels = np.ravel(raw["testlabel"]).astype(np.int64) - 1
        train_views = tuple(
            np.asarray(train_cell[0, index], dtype=np.float64).T.copy()
            for index in range(train_cell.shape[1])
        )
        test_views = tuple(
            np.asarray(test_cell[0, index], dtype=np.float64).T.copy()
            for index in range(test_cell.shape[1])
        )
    except (KeyError, IndexError, TypeError, ValueError, OSError) as exc:
        raise DataValidationError("invalid ship attachment MAT structure") from exc

    errors: list[str] = []
    dimensions = tuple(int(value) for value in expected_dimensions)
    if len(train_views) != len(dimensions) or len(test_views) != len(dimensions):
        errors.append("ship attachment view count changed")
    if any(
        view.shape != (expected_train_count, dimension)
        for view, dimension in zip(train_views, dimensions, strict=True)
    ):
        errors.append("ship training view shape changed")
    if any(
        view.shape != (expected_test_count, dimension)
        for view, dimension in zip(test_views, dimensions, strict=True)
    ):
        errors.append("ship test view shape changed")
    if train_labels.shape != (expected_train_count,) or test_labels.shape != (
        expected_test_count,
    ):
        errors.append("ship label shape changed")
    expected_labels = np.arange(expected_class_count)
    if not np.array_equal(np.unique(train_labels), expected_labels) or not np.array_equal(
        np.unique(test_labels), expected_labels
    ):
        errors.append("ship labels are not contiguous classes 1..5")
    if any(not np.isfinite(view).all() for view in (*train_views, *test_views)):
        errors.append("ship attachment contains NaN or Inf")
    if errors:
        raise DataValidationError(
            "Invalid ship attachment data:\n- " + "\n- ".join(errors)
        )
    return train_views, train_labels, test_views, test_labels


def knn_predict_multiple_k(
    train_projection: np.ndarray,
    train_labels: np.ndarray,
    query_projection: np.ndarray,
    *,
    k_values: Sequence[int],
) -> dict[int, np.ndarray]:
    train = np.asarray(train_projection, dtype=np.float64)
    query = np.asarray(query_projection, dtype=np.float64)
    labels = np.asarray(train_labels, dtype=np.int64)
    values = tuple(int(value) for value in k_values)
    if (
        train.ndim != 2
        or query.ndim != 2
        or train.shape[1] != query.shape[1]
        or labels.shape != (train.shape[0],)
        or not values
        or values != tuple(sorted(set(values)))
        or values[0] < 1
        or values[-1] > train.shape[0]
    ):
        raise DataValidationError("invalid multi-K KNN inputs")
    distances = pairwise_squared_distances(query, train)
    maximum_k = values[-1]
    neighbors = np.argpartition(distances, kth=maximum_k - 1, axis=1)[:, :maximum_k]
    neighbor_distances = np.take_along_axis(distances, neighbors, axis=1)
    order = np.argsort(neighbor_distances, axis=1)
    neighbors = np.take_along_axis(neighbors, order, axis=1)
    class_count = int(labels.max()) + 1
    predictions = {
        k: np.asarray(
            [
                np.bincount(labels[row[:k]], minlength=class_count).argmax()
                for row in neighbors
            ],
            dtype=np.int64,
        )
        for k in values
    }
    return predictions


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git_state(project_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "dirty": bool(run("status", "--short")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None}


def run_attachment_ship_reference(
    *,
    config_path: str | Path,
    mat_path: str | Path,
    output_root: str | Path,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_attachment_ship_config(config_path)
    destination = Path(output_root).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise DataValidationError(f"output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    dataset = _mapping(config["dataset"], "dataset")
    train_views, train_labels, test_views, test_labels = load_attachment_ship_data(
        mat_path,
        expected_sha256=str(dataset["file_sha256"]),
        expected_train_count=int(dataset["train_sample_count"]),
        expected_test_count=int(dataset["test_sample_count"]),
        expected_dimensions=[int(value) for value in dataset["feature_dimensions"]],
        expected_class_count=int(dataset["class_count"]),
    )
    model_raw = _mapping(config["model"], "model")
    model_config = AMDRModelConfig(
        lambda_manifold=float(model_raw["lambda_manifold"]),
        lambda_sparse=float(model_raw["lambda_sparse"]),
        max_iterations=int(model_raw["max_iterations"]),
        minimum_iterations=int(model_raw["minimum_iterations"]),
        tolerance=float(model_raw["tolerance"]),
        numerical_epsilon=float(model_raw["numerical_epsilon"]),
        solve_ridge=float(model_raw["solve_ridge"]),
        initialization_seed=int(model_raw["initialization_seed"]),
        convergence_metric=str(model_raw["convergence_metric"]),
        graph_same_base_policy=str(model_raw["graph_same_base_policy"]),
    )
    checkpoint_raw = _mapping(config["checkpoint"], "checkpoint")
    checkpoint_interval = int(checkpoint_raw["every_iterations"])
    checkpoint_path = destination / "checkpoint_latest.npz"
    resume_checkpoint = (
        None if resume_from is None else load_amdr_checkpoint(resume_from)
    )

    def checkpoint_callback(checkpoint: AMDRCheckpoint) -> None:
        final_iteration = checkpoint.iteration_completed == model_config.max_iterations
        converged = (
            checkpoint.iteration_completed >= model_config.minimum_iterations
            and float(checkpoint.history[-1]["convergence_value"])
            < model_config.tolerance
        )
        if (
            checkpoint.iteration_completed % checkpoint_interval == 0
            or final_iteration
            or converged
        ):
            save_amdr_checkpoint(checkpoint_path, checkpoint)

    fit = fit_amdr(
        train_views,
        train_labels,
        model_config,
        resume_checkpoint=resume_checkpoint,
        checkpoint_callback=checkpoint_callback,
    )
    prune_threshold = float(
        model_raw["post_training_row_prune_squared_norm_threshold"]
    )
    fit, pruned_count = prune_amdr_weight_rows(
        fit,
        squared_row_norm_threshold=prune_threshold,
    )
    train_projection = project_views(train_views, fit)
    test_projection = project_views(test_views, fit)
    k_values = tuple(int(value) for value in config["knn"]["k_values"])
    predictions = knn_predict_multiple_k(
        train_projection,
        train_labels,
        test_projection,
        k_values=k_values,
    )
    class_count = int(dataset["class_count"])
    by_k: dict[str, Any] = {}
    for k, predicted in predictions.items():
        by_k[str(k)] = {
            "accuracy": accuracy_score(test_labels, predicted),
            "macro_f1": macro_f1_score(
                test_labels,
                predicted,
                labels=range(class_count),
            ),
            "per_class_accuracy": {
                str(label + 1): float(
                    np.mean(predicted[test_labels == label] == label)
                )
                for label in range(class_count)
            },
        }
    history = fit.history
    metrics = {
        "result_scope": str(config["result_scope"]),
        "formal_experiment": False,
        "dataset_id": str(dataset["dataset_id"]),
        "dataset_file_sha256": str(dataset["file_sha256"]),
        "train_sample_count": int(train_labels.size),
        "test_sample_count": int(test_labels.size),
        "view_dimensions": list(fit.view_dimensions),
        "class_count": class_count,
        "iterations_ran": len(history),
        "converged_within_configured_budget": fit.converged,
        "optimization_stop_reason": fit.stop_reason,
        "resumed_from_iteration": (
            0 if resume_checkpoint is None else resume_checkpoint.iteration_completed
        ),
        "alpha": fit.alpha.tolist(),
        "weight_frobenius_norm": float(np.linalg.norm(fit.weights)),
        "post_training_row_prune_squared_norm_threshold": prune_threshold,
        "pruned_weight_row_count": pruned_count,
        "test_knn_results": by_k,
        "test_knn_selection": "none_report_all",
        "direct_argmax_train_accuracy": accuracy_score(
            train_labels, np.argmax(train_projection, axis=1)
        ),
        "direct_argmax_test_accuracy": accuracy_score(
            test_labels, np.argmax(test_projection, axis=1)
        ),
    }
    _write_json(destination / "metrics.json", metrics)
    with (destination / "training_log.jsonl").open("w", encoding="utf-8") as handle:
        for row in history:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    np.savez_compressed(
        destination / "model.npz",
        weights=fit.weights,
        alpha=fit.alpha,
        class_labels=np.arange(1, class_count + 1),
        pruned_weight_row_count=np.asarray(pruned_count),
    )
    np.savez_compressed(
        destination / "projections.npz",
        train=train_projection,
        test=test_projection,
        train_labels=train_labels,
        test_labels=test_labels,
    )
    with (destination / "predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = ["sample_index", "true_label", *[f"predicted_k{k}" for k in k_values]]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, true_label in enumerate(test_labels):
            row: dict[str, int] = {
                "sample_index": index,
                "true_label": int(true_label + 1),
            }
            row.update(
                {
                    f"predicted_k{k}": int(predictions[k][index] + 1)
                    for k in k_values
                }
            )
            writer.writerow(row)
    project_root = Path(__file__).resolve().parents[3]
    peak_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    peak_rss_mib = (
        peak_rss / (1024.0 * 1024.0)
        if platform.system() == "Darwin"
        else peak_rss / 1024.0
    )
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "peak_rss_mib": peak_rss_mib,
        "wall_time_seconds": time.perf_counter() - started,
        "git": _git_state(project_root),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }
    _write_json(destination / "environment.json", environment)
    resolved = dict(config)
    resolved["_resolved"] = {
        "mat_path": str(Path(mat_path).resolve()),
        "output_root": str(destination),
    }
    (destination / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    hashes = {
        path.name: file_sha256(path)
        for path in sorted(destination.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    _write_json(destination / "artifact_hashes.json", hashes)
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Python AMDR on the supplied ship attachment"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mat", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args(argv)
    metrics = run_attachment_ship_reference(
        config_path=args.config,
        mat_path=args.mat,
        output_root=args.output,
        resume_from=args.resume_from,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
