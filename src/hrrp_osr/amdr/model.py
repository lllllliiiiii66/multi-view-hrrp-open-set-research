from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.linalg import block_diag

from hrrp_osr.data.errors import DataValidationError


AMDR_ALGORITHM_VERSION = "amdr_research_v1"
RELATIVE_STATE_CHANGE = "relative_state_change_v1"
ABSOLUTE_STATE_DELTA = "absolute_state_delta_v1"
ALLOW_SAME_BASE_GRAPH = "allow_same_base_v1"
EXCLUDE_SAME_BASE_GRAPH = "exclude_same_base_v1"
COMPLETE_SAME_CLASS_GRAPH = "complete_same_class_inverse_distance_v1"
LOCAL_KNN_GAUSSIAN_GRAPH = "local_knn_gaussian_v1"
FIXED_INITIAL_L21_REWEIGHTING = "fixed_initial"
UPDATE_EACH_ITERATION_L21_REWEIGHTING = "update_each_iteration"


@dataclass(frozen=True)
class AMDRModelConfig:
    lambda_manifold: float
    lambda_sparse: float
    max_iterations: int
    tolerance: float
    numerical_epsilon: float
    solve_ridge: float
    initialization_seed: int
    minimum_iterations: int = 3
    convergence_metric: str = RELATIVE_STATE_CHANGE
    graph_same_base_policy: str = ALLOW_SAME_BASE_GRAPH
    graph_neighborhood: str = COMPLETE_SAME_CLASS_GRAPH
    graph_neighbor_count: int = 10
    l21_reweighting: str = UPDATE_EACH_ITERATION_L21_REWEIGHTING


@dataclass(frozen=True)
class AMDRCheckpoint:
    iteration_completed: int
    weights: np.ndarray
    alpha: np.ndarray
    adjusted_target: np.ndarray
    graphs: tuple[tuple[np.ndarray, ...], ...]
    history: tuple[dict[str, Any], ...]
    view_dimensions: tuple[int, ...]
    class_count: int
    sample_count: int
    labels_sha256: str
    training_views_sha256: str
    config_signature: dict[str, Any]


@dataclass(frozen=True)
class AMDRFitResult:
    weights: np.ndarray
    alpha: np.ndarray
    view_dimensions: tuple[int, ...]
    class_count: int
    history: tuple[dict[str, Any], ...]
    converged: bool
    stop_reason: str


def pairwise_squared_distances(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise DataValidationError("distance inputs must be 2D with equal feature size")
    distances = (
        np.sum(a * a, axis=1)[:, None]
        + np.sum(b * b, axis=1)[None, :]
        - 2.0 * (a @ b.T)
    )
    return np.maximum(np.real(distances), 0.0)


def _validate_fit_inputs(
    views: Sequence[np.ndarray], labels: np.ndarray, config: AMDRModelConfig
) -> tuple[tuple[np.ndarray, ...], np.ndarray, int]:
    materialized = tuple(np.asarray(view, dtype=np.float64) for view in views)
    y = np.asarray(labels, dtype=np.int64)
    errors: list[str] = []
    if len(materialized) != 2:
        errors.append("the first AMDR protocol requires exactly two views")
    if y.ndim != 1 or y.size == 0:
        errors.append("labels must be a non-empty 1D array")
    if any(view.ndim != 2 or view.shape[0] != y.size for view in materialized):
        errors.append("every view must have shape [n_samples, n_features]")
    if any(not np.isfinite(view).all() for view in materialized):
        errors.append("AMDR input contains NaN or Inf")
    unique = np.unique(y)
    if unique.size < 2 or not np.array_equal(unique, np.arange(unique.size)):
        errors.append("labels must be contiguous integers starting at zero")
    if any(np.count_nonzero(y == label) < 2 for label in unique):
        errors.append("every class needs at least two training samples")
    if config.lambda_manifold < 0 or config.lambda_sparse < 0:
        errors.append("regularization weights must be nonnegative")
    if config.max_iterations < 1 or config.tolerance <= 0:
        errors.append("iteration count and tolerance must be positive")
    if not 1 <= config.minimum_iterations <= config.max_iterations:
        errors.append("minimum_iterations must be within 1..max_iterations")
    if config.numerical_epsilon <= 0 or config.solve_ridge <= 0:
        errors.append("numerical stabilizers must be positive")
    if config.convergence_metric not in {
        RELATIVE_STATE_CHANGE,
        ABSOLUTE_STATE_DELTA,
    }:
        errors.append("unsupported AMDR convergence metric")
    if config.graph_same_base_policy not in {
        ALLOW_SAME_BASE_GRAPH,
        EXCLUDE_SAME_BASE_GRAPH,
    }:
        errors.append("unsupported same-base graph policy")
    if config.graph_neighborhood not in {
        COMPLETE_SAME_CLASS_GRAPH,
        LOCAL_KNN_GAUSSIAN_GRAPH,
    }:
        errors.append("unsupported graph neighborhood")
    if config.graph_neighbor_count < 1:
        errors.append("graph_neighbor_count must be positive")
    if config.l21_reweighting not in {
        FIXED_INITIAL_L21_REWEIGHTING,
        UPDATE_EACH_ITERATION_L21_REWEIGHTING,
    }:
        errors.append("unsupported l21 reweighting policy")
    if errors:
        raise DataValidationError("Invalid AMDR fit input:\n- " + "\n- ".join(errors))
    return materialized, y, int(unique.size)


def _labels_sha256(labels: np.ndarray) -> str:
    materialized = np.ascontiguousarray(labels, dtype=np.int64)
    return hashlib.sha256(materialized.tobytes()).hexdigest()


def _training_views_sha256(
    views: Sequence[np.ndarray],
    view_group_ids: Sequence[np.ndarray] | None = None,
) -> str:
    digest = hashlib.sha256()
    for view in views:
        materialized = np.ascontiguousarray(view, dtype=np.float64)
        digest.update(np.asarray(materialized.shape, dtype=np.int64).tobytes())
        digest.update(materialized.tobytes())
    if view_group_ids is not None:
        for group_ids in view_group_ids:
            for value in group_ids:
                encoded = str(value).encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "little"))
                digest.update(encoded)
    return digest.hexdigest()


def _config_signature(config: AMDRModelConfig) -> dict[str, Any]:
    """Return resume-critical settings; max_iterations may grow on resume."""

    signature = {
        "lambda_manifold": config.lambda_manifold,
        "lambda_sparse": config.lambda_sparse,
        "tolerance": config.tolerance,
        "numerical_epsilon": config.numerical_epsilon,
        "solve_ridge": config.solve_ridge,
        "initialization_seed": config.initialization_seed,
        "minimum_iterations": config.minimum_iterations,
        "algorithm_version": AMDR_ALGORITHM_VERSION,
        "convergence_metric": config.convergence_metric,
    }
    # Preserve compatibility with checkpoints created before the diagnostic
    # same-base exclusion policy existed.
    if config.graph_same_base_policy != ALLOW_SAME_BASE_GRAPH:
        signature["graph_same_base_policy"] = config.graph_same_base_policy
    if config.graph_neighborhood != COMPLETE_SAME_CLASS_GRAPH:
        signature["graph_neighborhood"] = config.graph_neighborhood
        signature["graph_neighbor_count"] = config.graph_neighbor_count
    if config.l21_reweighting != UPDATE_EACH_ITERATION_L21_REWEIGHTING:
        signature["l21_reweighting"] = config.l21_reweighting
    return signature


def save_amdr_checkpoint(path: str | Path, checkpoint: AMDRCheckpoint) -> None:
    """Atomically replace one latest-only checkpoint for interruption recovery."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    metadata = {
        "schema_version": 2,
        "iteration_completed": checkpoint.iteration_completed,
        "view_dimensions": list(checkpoint.view_dimensions),
        "class_count": checkpoint.class_count,
        "sample_count": checkpoint.sample_count,
        "labels_sha256": checkpoint.labels_sha256,
        "training_views_sha256": checkpoint.training_views_sha256,
        "config_signature": checkpoint.config_signature,
        "view_count": len(checkpoint.graphs),
        "graph_counts": [len(view_graphs) for view_graphs in checkpoint.graphs],
    }
    arrays: dict[str, np.ndarray] = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        "history_json": np.asarray(json.dumps(checkpoint.history, sort_keys=True)),
        "weights": checkpoint.weights,
        "alpha": checkpoint.alpha,
        "adjusted_target": checkpoint.adjusted_target,
    }
    for view_index, view_graphs in enumerate(checkpoint.graphs):
        for class_index, graph in enumerate(view_graphs):
            arrays[f"graph_{view_index}_{class_index}"] = graph
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_amdr_checkpoint(path: str | Path) -> AMDRCheckpoint:
    source = Path(path)
    if not source.is_file():
        raise DataValidationError(f"AMDR checkpoint does not exist: {source}")
    try:
        with np.load(source, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            history = tuple(json.loads(str(archive["history_json"].item())))
            if int(metadata.get("schema_version", 0)) != 2:
                raise DataValidationError("unsupported AMDR checkpoint schema")
            graph_counts = tuple(int(value) for value in metadata["graph_counts"])
            graphs = tuple(
                tuple(
                    np.asarray(
                        archive[f"graph_{view_index}_{class_index}"],
                        dtype=np.float64,
                    ).copy()
                    for class_index in range(graph_count)
                )
                for view_index, graph_count in enumerate(graph_counts)
            )
            return AMDRCheckpoint(
                iteration_completed=int(metadata["iteration_completed"]),
                weights=np.asarray(archive["weights"], dtype=np.float64).copy(),
                alpha=np.asarray(archive["alpha"], dtype=np.float64).copy(),
                adjusted_target=np.asarray(
                    archive["adjusted_target"], dtype=np.float64
                ).copy(),
                graphs=graphs,
                history=history,
                view_dimensions=tuple(int(value) for value in metadata["view_dimensions"]),
                class_count=int(metadata["class_count"]),
                sample_count=int(metadata["sample_count"]),
                labels_sha256=str(metadata["labels_sha256"]),
                training_views_sha256=str(metadata["training_views_sha256"]),
                config_signature=dict(metadata["config_signature"]),
            )
    except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"invalid AMDR checkpoint: {source}") from exc


def _materialize_view_group_ids(
    view_group_ids: Sequence[Sequence[str]] | None,
    *,
    sample_count: int,
    view_count: int,
    policy: str,
) -> tuple[np.ndarray, ...] | None:
    if policy == ALLOW_SAME_BASE_GRAPH:
        return None
    if view_group_ids is None or len(view_group_ids) != view_count:
        raise DataValidationError(
            "same-base graph exclusion requires one sample-ID vector per view"
        )
    materialized = tuple(np.asarray(ids, dtype=str) for ids in view_group_ids)
    if any(ids.shape != (sample_count,) for ids in materialized):
        raise DataValidationError("view sample-ID vectors have an invalid shape")
    if any(np.any(ids == "") for ids in materialized):
        raise DataValidationError("view sample-ID vectors contain an empty ID")
    return materialized


def _graph_exclusion_mask(group_ids: np.ndarray | None, count: int) -> np.ndarray:
    if group_ids is None:
        return np.eye(count, dtype=bool)
    return group_ids[:, None] == group_ids[None, :]


def _local_knn_gaussian_graph(
    squared_distances: np.ndarray,
    *,
    neighbor_count: int,
    exclusion_mask: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Build a directed row-normalized local graph from squared distances.

    ``pairwise_squared_distances`` already returns squared Euclidean distance.
    The row-local median therefore estimates sigma squared and must not be
    squared a second time in the Gaussian exponent.
    """

    distances = np.asarray(squared_distances, dtype=np.float64).copy()
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise DataValidationError("local graph distances must be square")
    count = distances.shape[0]
    distances[exclusion_mask] = np.inf
    available = np.sum(~exclusion_mask, axis=1)
    if np.any(available < 1):
        raise DataValidationError("local graph left a sample without a neighbor")
    effective_k = min(int(neighbor_count), int(available.min()))
    # Full row-wise sorting is unnecessarily expensive for the 500-pair
    # pilot: only the nearest ``effective_k`` entries are needed.  Use a
    # partial selection, then sort that small candidate set for deterministic
    # distance order.  Excluded entries are +inf and cannot enter unless the
    # protocol has fewer valid neighbors than requested (validated above).
    partition_indices = np.argpartition(
        distances, effective_k - 1, axis=1
    )[:, :effective_k]
    partition_distances = np.take_along_axis(
        distances, partition_indices, axis=1
    )
    order = np.argsort(partition_distances, axis=1, kind="stable")
    neighbor_indices = np.take_along_axis(partition_indices, order, axis=1)
    neighbor_distances = np.take_along_axis(
        distances, neighbor_indices, axis=1
    )
    sigma_squared = np.maximum(
        np.median(neighbor_distances, axis=1), epsilon
    )
    exponent = neighbor_distances / (2.0 * sigma_squared[:, None])
    weights = np.exp(-np.minimum(exponent, 700.0))
    graph = np.zeros((count, count), dtype=np.float64)
    graph[np.arange(count)[:, None], neighbor_indices] = weights
    graph /= np.maximum(graph.sum(axis=1, keepdims=True), epsilon)
    return graph


def _initialize_class_graphs(
    views: Sequence[np.ndarray],
    labels: np.ndarray,
    class_count: int,
    view_group_ids: tuple[np.ndarray, ...] | None,
    config: AMDRModelConfig,
) -> list[list[np.ndarray]]:
    graphs: list[list[np.ndarray]] = []
    for view_index, view in enumerate(views):
        view_graphs: list[np.ndarray] = []
        for label in range(class_count):
            indices = np.flatnonzero(labels == label)
            count = int(indices.size)
            graph = np.ones((count, count), dtype=np.float64)
            class_group_ids = (
                None
                if view_group_ids is None
                else view_group_ids[view_index][indices]
            )
            exclusion_mask = _graph_exclusion_mask(class_group_ids, count)
            if config.graph_neighborhood == LOCAL_KNN_GAUSSIAN_GRAPH:
                graph = _local_knn_gaussian_graph(
                    pairwise_squared_distances(view[indices], view[indices]),
                    neighbor_count=config.graph_neighbor_count,
                    exclusion_mask=exclusion_mask,
                    epsilon=config.numerical_epsilon,
                )
            else:
                graph[exclusion_mask] = 0.0
                row_sums = graph.sum(axis=1, keepdims=True)
                if np.any(row_sums <= 0.0):
                    raise DataValidationError(
                        "same-base graph exclusion left a sample without a neighbor"
                    )
                graph /= row_sums
            view_graphs.append(graph)
        graphs.append(view_graphs)
    return graphs


def _view_slices(dimensions: Sequence[int]) -> tuple[slice, ...]:
    slices: list[slice] = []
    offset = 0
    for dimension in dimensions:
        slices.append(slice(offset, offset + int(dimension)))
        offset += int(dimension)
    return tuple(slices)


def _alpha_from_weights(
    weights: np.ndarray,
    slices: Sequence[slice],
    epsilon: float,
) -> np.ndarray:
    raw = []
    for view_slice in slices:
        block = weights[view_slice]
        l21 = float(np.sum(np.linalg.norm(block, axis=1)))
        raw.append(np.sqrt(max(l21, epsilon)))
    alpha = np.asarray(raw, dtype=np.float64)
    total = float(alpha.sum())
    if not np.isfinite(total) or total <= 0:
        raise DataValidationError("AMDR alpha update is invalid")
    return alpha / total


def _validate_checkpoint(
    checkpoint: AMDRCheckpoint,
    *,
    views: Sequence[np.ndarray],
    labels: np.ndarray,
    class_count: int,
    config: AMDRModelConfig,
    view_group_ids: tuple[np.ndarray, ...] | None,
) -> None:
    dimensions = tuple(view.shape[1] for view in views)
    class_sizes = tuple(
        int(np.count_nonzero(labels == label)) for label in range(class_count)
    )
    errors: list[str] = []
    if checkpoint.iteration_completed < 1:
        errors.append("iteration_completed must be positive")
    if checkpoint.iteration_completed > config.max_iterations:
        errors.append("checkpoint iteration exceeds configured max_iterations")
    if checkpoint.view_dimensions != dimensions:
        errors.append("view dimensions differ from checkpoint")
    if checkpoint.class_count != class_count:
        errors.append("class count differs from checkpoint")
    if checkpoint.sample_count != labels.size:
        errors.append("sample count differs from checkpoint")
    if checkpoint.labels_sha256 != _labels_sha256(labels):
        errors.append("training label order differs from checkpoint")
    if checkpoint.training_views_sha256 != _training_views_sha256(
        views, view_group_ids
    ):
        errors.append(
            "training view values or order differ from checkpoint, or graph sample IDs changed"
        )
    if checkpoint.config_signature != _config_signature(config):
        errors.append("resume-critical model config differs from checkpoint")
    if checkpoint.weights.shape != (sum(dimensions), class_count):
        errors.append("checkpoint weights have an invalid shape")
    if checkpoint.alpha.shape != (len(views),):
        errors.append("checkpoint alpha has an invalid shape")
    if checkpoint.adjusted_target.shape != (labels.size, class_count):
        errors.append("checkpoint adjusted target has an invalid shape")
    if len(checkpoint.graphs) != len(views) or any(
        len(view_graphs) != class_count for view_graphs in checkpoint.graphs
    ):
        errors.append("checkpoint graph structure is invalid")
    elif any(
        graph.shape != (class_sizes[class_index], class_sizes[class_index])
        for view_graphs in checkpoint.graphs
        for class_index, graph in enumerate(view_graphs)
    ):
        errors.append("checkpoint graph shape is invalid")
    if len(checkpoint.history) != checkpoint.iteration_completed:
        errors.append("checkpoint history length differs from iteration count")
    elif any(
        int(row.get("iteration", -1)) != index
        for index, row in enumerate(checkpoint.history, start=1)
    ):
        errors.append("checkpoint history iteration sequence is invalid")
    numeric_arrays = [
        checkpoint.weights,
        checkpoint.alpha,
        checkpoint.adjusted_target,
        *(graph for view_graphs in checkpoint.graphs for graph in view_graphs),
    ]
    if any(not np.isfinite(array).all() for array in numeric_arrays):
        errors.append("checkpoint contains NaN or Inf")
    if not np.isclose(float(checkpoint.alpha.sum()), 1.0):
        errors.append("checkpoint alpha does not sum to one")
    if errors:
        raise DataValidationError(
            "Invalid AMDR checkpoint for this run:\n- " + "\n- ".join(errors)
        )


def _checkpoint_state(
    *,
    iteration_completed: int,
    weights: np.ndarray,
    alpha: np.ndarray,
    adjusted_target: np.ndarray,
    graphs: Sequence[Sequence[np.ndarray]],
    history: Sequence[dict[str, Any]],
    dimensions: tuple[int, ...],
    class_count: int,
    training_views_sha256: str,
    labels: np.ndarray,
    config: AMDRModelConfig,
) -> AMDRCheckpoint:
    return AMDRCheckpoint(
        iteration_completed=iteration_completed,
        weights=weights,
        alpha=alpha,
        adjusted_target=adjusted_target,
        graphs=tuple(tuple(view_graphs) for view_graphs in graphs),
        history=tuple(history),
        view_dimensions=dimensions,
        class_count=class_count,
        sample_count=int(labels.size),
        labels_sha256=_labels_sha256(labels),
        training_views_sha256=training_views_sha256,
        config_signature=_config_signature(config),
    )


def fit_amdr(
    views: Sequence[np.ndarray],
    labels: np.ndarray,
    config: AMDRModelConfig,
    *,
    view_group_ids: Sequence[Sequence[str]] | None = None,
    resume_checkpoint: AMDRCheckpoint | None = None,
    checkpoint_callback: Callable[[AMDRCheckpoint], None] | None = None,
) -> AMDRFitResult:
    materialized, y, class_count = _validate_fit_inputs(views, labels, config)
    dimensions = tuple(view.shape[1] for view in materialized)
    materialized_group_ids = _materialize_view_group_ids(
        view_group_ids,
        sample_count=y.size,
        view_count=len(materialized),
        policy=config.graph_same_base_policy,
    )
    training_views_sha256 = _training_views_sha256(
        materialized, materialized_group_ids
    )
    slices = _view_slices(dimensions)
    combined = np.concatenate(materialized, axis=1)
    sample_count, total_dimension = combined.shape
    target = -np.ones((sample_count, class_count), dtype=np.float64)
    target[np.arange(sample_count), y] = 1.0
    rng = np.random.default_rng(config.initialization_seed)
    initial_weights = rng.random(
        (total_dimension, class_count), dtype=np.float64
    )
    fixed_l21_diagonal = 1.0 / (
        np.linalg.norm(initial_weights, axis=1) + config.numerical_epsilon
    )
    if resume_checkpoint is None:
        adjusted_target = target.copy()
        weights = initial_weights.copy()
        alpha = np.full(len(materialized), 1.0 / len(materialized), dtype=np.float64)
        graphs = _initialize_class_graphs(
            materialized,
            y,
            class_count,
            materialized_group_ids,
            config,
        )
        history: list[dict[str, Any]] = []
        start_iteration = 0
    else:
        _validate_checkpoint(
            resume_checkpoint,
            views=materialized,
            labels=y,
            class_count=class_count,
            config=config,
            view_group_ids=materialized_group_ids,
        )
        adjusted_target = resume_checkpoint.adjusted_target.copy()
        weights = resume_checkpoint.weights.copy()
        alpha = resume_checkpoint.alpha.copy()
        graphs = [
            [graph.copy() for graph in view_graphs]
            for view_graphs in resume_checkpoint.graphs
        ]
        history = [dict(row) for row in resume_checkpoint.history]
        start_iteration = resume_checkpoint.iteration_completed
    class_indices = [np.flatnonzero(y == label) for label in range(class_count)]
    converged = bool(
        history
        and start_iteration >= config.minimum_iterations
        and float(history[-1]["convergence_value"]) < config.tolerance
    )

    for iteration in range(start_iteration, config.max_iterations):
        if converged:
            break
        old_weights = weights.copy()
        old_target = adjusted_target.copy()
        old_alpha = alpha.copy()
        old_graphs = [
            [graph.copy() for graph in view_graphs] for view_graphs in graphs
        ]

        residual = combined @ old_weights - target
        margin = np.maximum(target * residual, 0.0)
        adjusted_target = target + target * margin

        manifold_blocks: list[np.ndarray] = []
        for view_index, view in enumerate(materialized):
            manifold = np.zeros(
                (dimensions[view_index], dimensions[view_index]), dtype=np.float64
            )
            for label, indices in enumerate(class_indices):
                graph = graphs[view_index][label]
                squared_graph = graph * graph
                laplacian = (
                    np.diag(squared_graph.sum(axis=1))
                    + np.diag(squared_graph.sum(axis=0))
                    - squared_graph
                    - squared_graph.T
                )
                class_view = view[indices]
                manifold += len(indices) * (class_view.T @ laplacian @ class_view)
            manifold_blocks.append(manifold)
        manifold_all = block_diag(*manifold_blocks)

        if config.l21_reweighting == FIXED_INITIAL_L21_REWEIGHTING:
            l21_diagonal = fixed_l21_diagonal
        else:
            row_norm = np.linalg.norm(old_weights, axis=1)
            l21_diagonal = 1.0 / (row_norm + config.numerical_epsilon)
        alpha_all = np.concatenate(
            [
                np.full(dimension, alpha[index], dtype=np.float64)
                for index, dimension in enumerate(dimensions)
            ]
        )
        system = (
            combined.T @ combined
            + config.lambda_manifold * manifold_all
        )
        diagonal_regularizer = (
            0.5
            * config.lambda_sparse
            * l21_diagonal
            / np.maximum(alpha_all, config.numerical_epsilon)
        )
        diagonal = np.diag_indices_from(system)
        system[diagonal] += diagonal_regularizer + config.solve_ridge
        right_hand_side = combined.T @ adjusted_target
        weights = np.linalg.solve(system, right_hand_side)
        if not np.isfinite(weights).all():
            raise DataValidationError("AMDR W update produced NaN or Inf")

        manifold_raw = 0.0
        for view_index, view in enumerate(materialized):
            projected = view @ weights[slices[view_index]]
            for label, indices in enumerate(class_indices):
                distances = pairwise_squared_distances(
                    projected[indices], projected[indices]
                )
                class_group_ids = (
                    None
                    if materialized_group_ids is None
                    else materialized_group_ids[view_index][indices]
                )
                exclusion_mask = _graph_exclusion_mask(
                    class_group_ids, len(indices)
                )
                if config.graph_neighborhood == LOCAL_KNN_GAUSSIAN_GRAPH:
                    updated_graph = _local_knn_gaussian_graph(
                        distances,
                        neighbor_count=config.graph_neighbor_count,
                        exclusion_mask=exclusion_mask,
                        epsilon=config.numerical_epsilon,
                    )
                else:
                    inverse = 1.0 / (distances + config.numerical_epsilon)
                    inverse[exclusion_mask] = 0.0
                    row_sums = inverse.sum(axis=1, keepdims=True)
                    if np.any(row_sums <= 0.0):
                        raise DataValidationError(
                            "AMDR graph update left a sample without a neighbor"
                        )
                    updated_graph = inverse / np.maximum(
                        row_sums, config.numerical_epsilon
                    )
                graphs[view_index][label] = updated_graph
                manifold_raw += len(indices) * float(
                    np.sum(updated_graph * updated_graph * distances)
                )
        alpha = _alpha_from_weights(
            weights,
            slices,
            config.numerical_epsilon,
        )

        graph_delta = sum(
            float(np.sum((graphs[v][c] - old_graphs[v][c]) ** 2))
            for v in range(len(graphs))
            for c in range(class_count)
        )
        weight_delta = float(np.sum((weights - old_weights) ** 2))
        target_delta = float(np.sum((adjusted_target - old_target) ** 2))
        alpha_delta = float(np.sum((alpha - old_alpha) ** 2))
        absolute_delta = weight_delta + graph_delta + target_delta + alpha_delta
        old_graph_norm = sum(
            float(np.sum(old_graphs[v][c] ** 2))
            for v in range(len(old_graphs))
            for c in range(class_count)
        )
        relative_delta = (
            weight_delta
            / max(float(np.sum(old_weights * old_weights)), config.numerical_epsilon)
            + graph_delta / max(old_graph_norm, config.numerical_epsilon)
            + target_delta
            / max(float(np.sum(old_target * old_target)), config.numerical_epsilon)
            + alpha_delta
            / max(float(np.sum(old_alpha * old_alpha)), config.numerical_epsilon)
        )
        convergence_value = (
            relative_delta
            if config.convergence_metric == RELATIVE_STATE_CHANGE
            else absolute_delta
        )
        regression_loss = float(
            np.sum((combined @ weights - adjusted_target) ** 2)
        )
        sparse_raw = sum(
            float(np.sum(np.linalg.norm(weights[view_slice], axis=1)))
            / max(float(alpha[view_index]), config.numerical_epsilon)
            for view_index, view_slice in enumerate(slices)
        )
        manifold_loss = config.lambda_manifold * manifold_raw
        sparse_loss = config.lambda_sparse * sparse_raw
        objective = regression_loss + manifold_loss + sparse_loss
        if not np.isfinite(objective) or not np.isfinite(convergence_value):
            raise DataValidationError("AMDR convergence diagnostic is NaN or Inf")
        previous_objective = float(history[-1]["objective"]) if history else None
        history.append(
            {
                "iteration": iteration + 1,
                "delta": absolute_delta,
                "relative_state_change": relative_delta,
                "convergence_metric": config.convergence_metric,
                "graph_neighborhood": config.graph_neighborhood,
                "graph_neighbor_count": config.graph_neighbor_count,
                "l21_reweighting": config.l21_reweighting,
                "convergence_value": convergence_value,
                "weight_frobenius_norm": float(np.linalg.norm(weights)),
                "alpha": alpha.tolist(),
                "objective": objective,
                "objective_regression": regression_loss,
                "objective_manifold": manifold_loss,
                "objective_sparse": sparse_loss,
                "objective_change": (
                    None if previous_objective is None else objective - previous_objective
                ),
            }
        )
        if checkpoint_callback is not None:
            checkpoint = _checkpoint_state(
                iteration_completed=iteration + 1,
                weights=weights,
                alpha=alpha,
                adjusted_target=adjusted_target,
                graphs=graphs,
                history=history,
                dimensions=dimensions,
                class_count=class_count,
                training_views_sha256=training_views_sha256,
                labels=y,
                config=config,
            )
            checkpoint_callback(checkpoint)
        if (
            iteration + 1 >= config.minimum_iterations
            and convergence_value < config.tolerance
        ):
            converged = True
            break

    return AMDRFitResult(
        weights=weights,
        alpha=alpha,
        view_dimensions=dimensions,
        class_count=class_count,
        history=tuple(history),
        converged=converged,
        stop_reason="converged_tolerance" if converged else "max_iterations",
    )


def project_views(
    views: Sequence[np.ndarray],
    fit: AMDRFitResult,
) -> np.ndarray:
    materialized = tuple(np.asarray(view, dtype=np.float64) for view in views)
    if len(materialized) != len(fit.view_dimensions):
        raise DataValidationError("projection view count differs from fitted AMDR")
    if any(
        view.ndim != 2 or view.shape[1] != dimension
        for view, dimension in zip(
            materialized, fit.view_dimensions, strict=True
        )
    ):
        raise DataValidationError("projection view dimensions are invalid")
    combined = np.concatenate(materialized, axis=1)
    projected = combined @ fit.weights
    if not np.isfinite(projected).all():
        raise DataValidationError("AMDR projection contains NaN or Inf")
    return projected


def prune_amdr_weight_rows(
    fit: AMDRFitResult,
    *,
    squared_row_norm_threshold: float,
) -> tuple[AMDRFitResult, int]:
    """Apply the reference MATLAB demo's post-fit row threshold exactly once."""

    threshold = float(squared_row_norm_threshold)
    if not np.isfinite(threshold) or threshold < 0:
        raise DataValidationError("AMDR row-prune threshold must be finite and nonnegative")
    if threshold == 0:
        return fit, 0
    weights = fit.weights.copy()
    pruned_mask = np.sum(weights * weights, axis=1) < threshold
    weights[pruned_mask] = 0.0
    return replace(fit, weights=weights), int(np.count_nonzero(pruned_mask))


def knn_predict_and_score(
    train_projection: np.ndarray,
    train_labels: np.ndarray,
    query_projection: np.ndarray,
    *,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train_projection, dtype=np.float64)
    query = np.asarray(query_projection, dtype=np.float64)
    labels = np.asarray(train_labels, dtype=np.int64)
    if (
        train.ndim != 2
        or query.ndim != 2
        or train.shape[1] != query.shape[1]
        or labels.shape != (train.shape[0],)
    ):
        raise DataValidationError("invalid KNN inputs")
    if not 1 <= k <= train.shape[0]:
        raise DataValidationError("KNN k is outside the training sample range")
    distances = pairwise_squared_distances(query, train)
    neighbor_indices = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    neighbor_distances = np.take_along_axis(distances, neighbor_indices, axis=1)
    predictions = np.empty(query.shape[0], dtype=np.int64)
    class_count = int(labels.max()) + 1
    for index, neighbors in enumerate(neighbor_indices):
        predictions[index] = int(
            np.bincount(labels[neighbors], minlength=class_count).argmax()
        )
    scores = neighbor_distances.mean(axis=1)
    return predictions, scores
