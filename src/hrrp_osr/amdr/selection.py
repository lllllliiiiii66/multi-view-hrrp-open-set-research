from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
import yaml

from hrrp_osr.amdr.data import (
    RANDOMIZED_SLOT_ORDER,
    build_fold_pairs,
    materialize_pair_views,
    write_pair_manifest,
)
from hrrp_osr.amdr.model import (
    ALLOW_SAME_BASE_GRAPH,
    COMPLETE_SAME_CLASS_GRAPH,
    EXCLUDE_SAME_BASE_GRAPH,
    FIXED_INITIAL_L21_REWEIGHTING,
    LEGACY_UNNORMALIZED_OBJECTIVE,
    UPDATE_EACH_ITERATION_L21_REWEIGHTING,
    AMDRModelConfig,
    fit_amdr,
    knn_predict_and_score,
    prune_amdr_weight_rows,
    project_views,
)
from hrrp_osr.amdr.reduction import (
    SHARED_TRAIN_BASE_PCA,
    apply_shared_pca,
)
from hrrp_osr.amdr.smoke import (
    _fit_shared_train_base_pca,
    _git_state,
    _labels,
    _mapping,
    _resource_limits,
    _split_pairs,
    load_smoke_config,
)
from hrrp_osr.data.errors import DataConfigError, DataValidationError
from hrrp_osr.data.manifest import file_sha256
from hrrp_osr.data.processed import load_processed_bundle
from hrrp_osr.evaluation.metrics import accuracy_score, macro_f1_score


SELECTION_SCOPE = "diagnostic_parameter_selection"
SUPPORTED_REWEIGHTING = (
    FIXED_INITIAL_L21_REWEIGHTING,
    UPDATE_EACH_ITERATION_L21_REWEIGHTING,
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        _write_json(temporary, value)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _positive_grid(values: Any, name: str) -> tuple[float, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise DataConfigError(f"selection.{name} must be a sequence")
    result = tuple(float(value) for value in values)
    if not result or any(not np.isfinite(value) or value <= 0.0 for value in result):
        raise DataConfigError(f"selection.{name} must contain positive finite values")
    if tuple(sorted(set(result))) != result:
        raise DataConfigError(f"selection.{name} must be unique and ascending")
    return result


def load_selection_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = dict(_mapping(yaml.safe_load(handle), "AMDR selection config"))
    errors: list[str] = []
    if int(config.get("schema_version", 0)) != 1:
        errors.append("schema_version must be 1")
    if config.get("stage") != "P0" or config.get("result_scope") != SELECTION_SCOPE:
        errors.append("selection must remain a P0 diagnostic")
    base_path = config.get("base_experiment_config")
    if not isinstance(base_path, str) or not base_path:
        errors.append("base_experiment_config must be a project-relative path")
    selection = _mapping(config.get("selection"), "selection")
    if selection.get("split") != "calibration":
        errors.append("parameters must be selected on calibration")
    if selection.get("test_features_materialized") is not False:
        errors.append("test features must not be materialized during selection")
    if selection.get("test_metrics_used") is not False:
        errors.append("test metrics must not be used during selection")
    if selection.get("primary_metric") != "calibration_accuracy":
        errors.append("primary metric must be calibration_accuracy")
    if selection.get("secondary_metric") != "calibration_macro_f1":
        errors.append("secondary metric must be calibration_macro_f1")
    if int(selection.get("knn_k", 0)) != 3:
        errors.append("selection KNN k must remain 3")
    if "require_converged" in selection and not isinstance(
        selection["require_converged"], bool
    ):
        errors.append("selection.require_converged must be boolean")
    selected_reweighting = tuple(selection.get("l21_reweighting", ()))
    if (
        not selected_reweighting
        or len(set(selected_reweighting)) != len(selected_reweighting)
        or any(value not in SUPPORTED_REWEIGHTING for value in selected_reweighting)
    ):
        errors.append("selection l21_reweighting must be a unique supported subset")
    expected_ties = (
        "higher_calibration_accuracy",
        "higher_calibration_macro_f1",
        "lower_lambda_manifold",
        "lower_lambda_sparse",
    )
    if tuple(selection.get("tie_break_order", ())) != expected_ties:
        errors.append("tie-break order changed")
    if selection.get("boundary_rule") not in {
        "extend_once_before_test_if_selected_lambda_is_grid_boundary",
        "report_boundary_without_automatic_extension",
    }:
        errors.append("boundary rule changed")
    try:
        _positive_grid(selection.get("lambda_manifold"), "lambda_manifold")
        _positive_grid(selection.get("lambda_sparse"), "lambda_sparse")
    except DataConfigError as exc:
        errors.append(str(exc))
    if errors:
        raise DataConfigError(
            "Invalid AMDR parameter-selection configuration:\n- "
            + "\n- ".join(errors)
        )
    config["_config_path"] = str(config_path)
    config["_config_sha256"] = file_sha256(config_path)
    return config


def _candidate_id(strategy: str, lambda_manifold: float, lambda_sparse: float) -> str:
    def token(value: float) -> str:
        return f"{value:g}".replace(".", "p")

    return (
        f"{strategy}__lambda_manifold_{token(lambda_manifold)}"
        f"__lambda_sparse_{token(lambda_sparse)}"
    )


def select_best_candidates(
    rows: Sequence[Mapping[str, Any]],
    strategies: Sequence[str] = SUPPORTED_REWEIGHTING,
    *,
    require_converged: bool = False,
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for strategy in strategies:
        completed = [
            dict(row) for row in rows if row["l21_reweighting"] == strategy
        ]
        eligible = (
            [row for row in completed if row.get("converged") is True]
            if require_converged
            else completed
        )
        if not eligible:
            reason = "eligible converged" if require_converged else "completed"
            raise DataValidationError(f"no {reason} candidates for {strategy}")
        eligible.sort(
            key=lambda row: (
                -float(row["calibration_accuracy"]),
                -float(row["calibration_macro_f1"]),
                float(row["lambda_manifold"]),
                float(row["lambda_sparse"]),
            )
        )
        selected[strategy] = eligible[0]
    return selected


def _is_grid_boundary(value: float, grid: Sequence[float]) -> bool:
    return value == float(grid[0]) or value == float(grid[-1])


def run_parameter_selection(
    *,
    config_path: str | Path,
    bundle_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_selection_config(config_path)
    project_root = Path(__file__).resolve().parents[3]
    base_config_path = project_root / str(config["base_experiment_config"])
    base = load_smoke_config(base_config_path)
    if base["result_scope"] != "diagnostic_pilot":
        raise DataConfigError("base experiment must be a pilot")
    if int(base["knn"]["k"]) != int(config["selection"]["knn_k"]):
        raise DataConfigError("base and selection KNN k differ")

    destination = Path(output_root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    identity_path = destination / "selection_identity.json"
    identity = {
        "selection_config_sha256": config["_config_sha256"],
        "base_config_sha256": base["_config_sha256"],
    }
    if identity_path.exists():
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing != identity:
            raise DataValidationError("selection output belongs to different configs")
    else:
        if any(destination.iterdir()):
            raise DataValidationError("non-empty selection output has no identity")
        _write_json(identity_path, identity)

    bundle_config = _mapping(base["bundle"], "bundle")
    bundle = load_processed_bundle(
        bundle_root,
        expected_profiles_sha256=str(bundle_config["profiles_sha256"]),
        expected_manifest_sha256=str(bundle_config["manifest_sha256"]),
        expected_bundle_sha256=str(bundle_config["bundle_sha256"]),
    )
    protocol = _mapping(base["protocol"], "protocol")
    sampling = _mapping(base["sampling"], "sampling")
    pair_counts = {
        key: int(value)
        for key, value in _mapping(
            sampling["pairs_per_class"], "sampling.pairs_per_class"
        ).items()
    }
    pairs, pair_audit = build_fold_pairs(
        bundle,
        protocol_id=str(base["protocol_id"]),
        fold_index=int(protocol["fold_index"]),
        fold_count=int(protocol["fold_count"]),
        base_seed=int(sampling["base_seed"]),
        pairs_per_class=pair_counts,
        slot_order=str(sampling.get("slot_order", RANDOMIZED_SLOT_ORDER)),
    )
    pair_manifest_path = destination / "pair_manifest.csv"
    if not pair_manifest_path.exists():
        write_pair_manifest(pair_manifest_path, pairs)
    _write_json(destination / "pair_audit.json", pair_audit)

    split_pairs = {
        split: _split_pairs(pairs, split) for split in ("train", "calibration")
    }
    preprocessing = _mapping(base["preprocessing"], "preprocessing")
    transform = str(preprocessing["transform"])
    split_views = {
        split: materialize_pair_views(bundle, split_pairs[split], transform=transform)
        for split in split_pairs
    }
    pca_model = None
    reduction_raw = preprocessing.get("dimension_reduction")
    if reduction_raw is not None:
        reduction = _mapping(reduction_raw, "preprocessing.dimension_reduction")
        pca_model = _fit_shared_train_base_pca(
            split_pairs["train"],
            split_views["train"],
            output_dimension=int(reduction["output_dimension"]),
        )
        split_views = {
            split: tuple(apply_shared_pca(view, pca_model) for view in views)
            for split, views in split_views.items()
        }
    known_classes = tuple(sorted(bundle.known_classes))
    class_to_label = {name: index for index, name in enumerate(known_classes)}
    split_labels = {
        split: _labels(split_pairs[split], class_to_label) for split in split_pairs
    }
    if any(np.any(labels < 0) for labels in split_labels.values()):
        raise DataValidationError("unknown class entered train or calibration")

    model_raw = _mapping(base["model"], "model")
    row_prune_threshold = float(
        model_raw.get("post_training_row_prune_squared_norm_threshold", 0.0)
    )
    selection = _mapping(config["selection"], "selection")
    selected_reweighting = tuple(
        str(value) for value in selection["l21_reweighting"]
    )
    manifold_grid = _positive_grid(selection["lambda_manifold"], "lambda_manifold")
    sparse_grid = _positive_grid(selection["lambda_sparse"], "lambda_sparse")
    candidate_root = destination / "candidates"
    candidate_root.mkdir(exist_ok=True)
    completed: list[dict[str, Any]] = []

    for strategy in selected_reweighting:
        for lambda_manifold in manifold_grid:
            for lambda_sparse in sparse_grid:
                candidate_id = _candidate_id(
                    strategy, lambda_manifold, lambda_sparse
                )
                candidate_dir = candidate_root / candidate_id
                metrics_path = candidate_dir / "metrics.json"
                if metrics_path.exists():
                    row = json.loads(metrics_path.read_text(encoding="utf-8"))
                    if (
                        row.get("selection_config_sha256")
                        != config["_config_sha256"]
                        or row.get("base_config_sha256") != base["_config_sha256"]
                    ):
                        raise DataValidationError(
                            f"candidate identity mismatch: {candidate_id}"
                        )
                    completed.append(row)
                    continue

                candidate_dir.mkdir(parents=True, exist_ok=False)
                candidate_started = time.perf_counter()
                model_config = AMDRModelConfig(
                    lambda_manifold=lambda_manifold,
                    lambda_sparse=lambda_sparse,
                    max_iterations=int(model_raw["max_iterations"]),
                    tolerance=float(model_raw["tolerance"]),
                    numerical_epsilon=float(model_raw["numerical_epsilon"]),
                    solve_ridge=float(model_raw["solve_ridge"]),
                    initialization_seed=int(model_raw["initialization_seed"]),
                    minimum_iterations=int(model_raw.get("minimum_iterations", 3)),
                    convergence_metric=str(model_raw["convergence_metric"]),
                    graph_same_base_policy=str(
                        model_raw.get(
                            "graph_same_base_policy", ALLOW_SAME_BASE_GRAPH
                        )
                    ),
                    graph_neighborhood=str(
                        model_raw.get(
                            "graph_neighborhood", COMPLETE_SAME_CLASS_GRAPH
                        )
                    ),
                    graph_neighbor_count=int(
                        model_raw.get("graph_neighbor_count", 10)
                    ),
                    l21_reweighting=strategy,
                    objective_scaling=str(
                        model_raw.get(
                            "objective_scaling", LEGACY_UNNORMALIZED_OBJECTIVE
                        )
                    ),
                )
                view_group_ids = (
                    (
                        tuple(pair.view1_sample_id for pair in split_pairs["train"]),
                        tuple(pair.view2_sample_id for pair in split_pairs["train"]),
                    )
                    if model_config.graph_same_base_policy
                    == EXCLUDE_SAME_BASE_GRAPH
                    else None
                )
                fit = fit_amdr(
                    split_views["train"],
                    split_labels["train"],
                    model_config,
                    view_group_ids=view_group_ids,
                )
                fit, pruned_count = prune_amdr_weight_rows(
                    fit,
                    squared_row_norm_threshold=row_prune_threshold,
                )
                train_projection = project_views(split_views["train"], fit)
                calibration_projection = project_views(
                    split_views["calibration"], fit
                )
                prediction, _ = knn_predict_and_score(
                    train_projection,
                    split_labels["train"],
                    calibration_projection,
                    k=int(selection["knn_k"]),
                )
                row = {
                    "candidate_id": candidate_id,
                    "selection_config_sha256": config["_config_sha256"],
                    "base_config_sha256": base["_config_sha256"],
                    "l21_reweighting": strategy,
                    "objective_scaling": model_config.objective_scaling,
                    "graph_neighborhood": model_config.graph_neighborhood,
                    "graph_neighbor_count": model_config.graph_neighbor_count,
                    "lambda_manifold": lambda_manifold,
                    "lambda_sparse": lambda_sparse,
                    "knn_k": int(selection["knn_k"]),
                    "calibration_accuracy": accuracy_score(
                        split_labels["calibration"], prediction
                    ),
                    "calibration_macro_f1": macro_f1_score(
                        split_labels["calibration"],
                        prediction,
                        labels=range(len(known_classes)),
                    ),
                    "iterations_ran": len(fit.history),
                    "converged": fit.converged,
                    "stop_reason": fit.stop_reason,
                    "weight_frobenius_norm": float(np.linalg.norm(fit.weights)),
                    "alpha": fit.alpha.tolist(),
                    "pruned_weight_row_count": pruned_count,
                    "wall_time_seconds": time.perf_counter() - candidate_started,
                    "test_features_materialized": False,
                    "test_metrics_used": False,
                }
                np.savez_compressed(
                    candidate_dir / "model.npz",
                    weights=fit.weights,
                    alpha=fit.alpha,
                    known_classes=np.asarray(known_classes),
                )
                with (candidate_dir / "training_log.jsonl").open(
                    "w", encoding="utf-8"
                ) as handle:
                    for history_row in fit.history:
                        handle.write(json.dumps(history_row, sort_keys=True) + "\n")
                _write_json(metrics_path, row)
                completed.append(row)
                _atomic_write_json(
                    destination / "selection_progress.json",
                    {
                        "completed_candidate_count": len(completed),
                        "expected_candidate_count": (
                            len(selected_reweighting)
                            * len(manifold_grid)
                            * len(sparse_grid)
                        ),
                        "last_completed_candidate_id": candidate_id,
                    },
                )

    require_converged = bool(selection.get("require_converged", False))
    selected = select_best_candidates(
        completed,
        selected_reweighting,
        require_converged=require_converged,
    )
    boundary = {
        strategy: {
            "lambda_manifold": _is_grid_boundary(
                float(row["lambda_manifold"]), manifold_grid
            ),
            "lambda_sparse": _is_grid_boundary(
                float(row["lambda_sparse"]), sparse_grid
            ),
        }
        for strategy, row in selected.items()
    }
    summary = {
        "selection_id": config["selection_id"],
        "selection_config_sha256": config["_config_sha256"],
        "base_config_path": str(base_config_path),
        "base_config_sha256": base["_config_sha256"],
        "bundle_sha256": bundle.bundle_sha256,
        "pair_manifest_sha256": file_sha256(pair_manifest_path),
        "candidate_count": len(completed),
        "selected": selected,
        "selected_parameter_on_grid_boundary": boundary,
        "requires_boundary_extension_before_test": (
            selection["boundary_rule"]
            == "extend_once_before_test_if_selected_lambda_is_grid_boundary"
            and any(any(flags.values()) for flags in boundary.values())
        ),
        "boundary_rule": selection["boundary_rule"],
        "require_converged": require_converged,
        "graph_neighborhood": str(
            model_raw.get("graph_neighborhood", COMPLETE_SAME_CLASS_GRAPH)
        ),
        "graph_neighbor_count": int(model_raw.get("graph_neighbor_count", 10)),
        "test_features_materialized": False,
        "test_metrics_used": False,
        "train_pair_count": len(split_pairs["train"]),
        "calibration_pair_count": len(split_pairs["calibration"]),
        "test_pair_count_registered_not_materialized": sum(
            pair.split == "test" for pair in pairs
        ),
        "base_profile_transform": transform,
        "dimension_reduction": (
            "none" if pca_model is None else SHARED_TRAIN_BASE_PCA
        ),
        "wall_time_seconds": time.perf_counter() - started,
    }
    _write_json(destination / "selection.json", summary)
    _write_json(
        destination / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "resource_limits": _resource_limits(),
            "peak_rss_mib": (
                float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                / (1024.0 * 1024.0 if platform.system() == "Darwin" else 1024.0)
            ),
            "git": _git_state(project_root),
        },
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Select AMDR D strategy parameters using known calibration only"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--bundle-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    summary = run_parameter_selection(
        config_path=args.config,
        bundle_root=args.bundle_root,
        output_root=args.output,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
