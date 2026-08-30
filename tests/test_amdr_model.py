from __future__ import annotations

import numpy as np
import pytest

from hrrp_osr.data.errors import DataValidationError
from hrrp_osr.amdr.model import (
    EXCLUDE_SAME_BASE_GRAPH,
    AMDRFitResult,
    AMDRModelConfig,
    fit_amdr,
    knn_predict_and_score,
    load_amdr_checkpoint,
    pairwise_squared_distances,
    project_views,
    prune_amdr_weight_rows,
    save_amdr_checkpoint,
)


def test_reference_row_prune_uses_squared_norm_strict_threshold() -> None:
    weights = np.array(
        [
            [1.0e-3, 2.0e-3],
            [4.0e-3, 0.0],
            [1.0, 0.0],
        ]
    )
    fit = AMDRFitResult(
        weights=weights,
        alpha=np.array([0.5, 0.5]),
        view_dimensions=(1, 2),
        class_count=2,
        history=(),
        converged=False,
        stop_reason="fixture",
    )
    pruned, count = prune_amdr_weight_rows(
        fit,
        squared_row_norm_threshold=1.0e-5,
    )
    assert count == 1
    np.testing.assert_array_equal(pruned.weights[0], np.zeros(2))
    np.testing.assert_array_equal(pruned.weights[1:], weights[1:])
    np.testing.assert_array_equal(fit.weights, weights)


def test_pairwise_squared_distances_matches_hand_calculation() -> None:
    first = np.array([[0.0, 0.0], [1.0, 2.0]])
    second = np.array([[1.0, 0.0], [3.0, 4.0]])
    assert pairwise_squared_distances(first, second) == pytest.approx(
        np.array([[1.0, 25.0], [4.0, 8.0]])
    )


def _synthetic_views(
    seed: int,
    samples_per_class: int,
) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(3), samples_per_class)
    centers1 = np.array(
        [[1.0, 0.0, 0.0, 0.2], [0.0, 1.0, 0.0, 0.4], [0.0, 0.0, 1.0, 0.6]]
    )
    centers2 = np.array(
        [[0.8, 0.0, 0.1, 0.0], [0.0, 0.8, 0.2, 0.0], [0.1, 0.0, 0.8, 0.0]]
    )
    view1 = centers1[labels] + 0.02 * rng.normal(size=(labels.size, 4))
    view2 = centers2[labels] + 0.02 * rng.normal(size=(labels.size, 4))
    return (view1, view2), labels


def test_amdr_fit_project_and_knn_smoke_are_finite_and_directional() -> None:
    train_views, train_labels = _synthetic_views(1, 12)
    fit = fit_amdr(
        train_views,
        train_labels,
        AMDRModelConfig(
            lambda_manifold=0.01,
            lambda_sparse=0.01,
            max_iterations=3,
            tolerance=1.0e-8,
            numerical_epsilon=1.0e-10,
            solve_ridge=1.0e-6,
            initialization_seed=7,
        ),
    )
    assert fit.weights.shape == (8, 3)
    assert fit.alpha.shape == (2,)
    assert fit.alpha.sum() == pytest.approx(1.0)
    assert np.isfinite(fit.weights).all()
    assert len(fit.history) == 3
    assert fit.stop_reason == "max_iterations"
    assert all(row["objective"] > 0.0 for row in fit.history)
    assert all(row["relative_state_change"] >= 0.0 for row in fit.history)
    assert all(
        row["convergence_value"] == row["relative_state_change"]
        for row in fit.history
    )
    for row in fit.history:
        assert row["objective"] == pytest.approx(
            row["objective_regression"]
            + row["objective_manifold"]
            + row["objective_sparse"]
        )

    train_projection = project_views(train_views, fit)
    known_views, known_labels = _synthetic_views(2, 4)
    known_projection = project_views(known_views, fit)
    known_pred, known_scores = knn_predict_and_score(
        train_projection,
        train_labels,
        known_projection,
        k=3,
    )
    assert np.mean(known_pred == known_labels) >= 0.9

    unknown_views = (
        np.full((6, 4), 5.0, dtype=np.float64),
        np.full((6, 4), -5.0, dtype=np.float64),
    )
    unknown_projection = project_views(unknown_views, fit)
    _, unknown_scores = knn_predict_and_score(
        train_projection,
        train_labels,
        unknown_projection,
        k=3,
    )
    assert float(np.median(unknown_scores)) > float(np.median(known_scores))


def test_latest_checkpoint_resume_matches_uninterrupted_fit(tmp_path) -> None:
    train_views, train_labels = _synthetic_views(11, 8)

    def config(max_iterations: int) -> AMDRModelConfig:
        return AMDRModelConfig(
            lambda_manifold=0.01,
            lambda_sparse=0.01,
            max_iterations=max_iterations,
            minimum_iterations=1,
            tolerance=1.0e-30,
            numerical_epsilon=1.0e-10,
            solve_ridge=1.0e-6,
            initialization_seed=19,
        )

    uninterrupted = fit_amdr(train_views, train_labels, config(5))
    checkpoint_path = tmp_path / "checkpoint_latest.npz"
    partial = fit_amdr(
        train_views,
        train_labels,
        config(2),
        checkpoint_callback=lambda checkpoint: save_amdr_checkpoint(
            checkpoint_path, checkpoint
        ),
    )
    assert len(partial.history) == 2
    loaded = load_amdr_checkpoint(checkpoint_path)
    assert loaded.iteration_completed == 2

    resumed = fit_amdr(
        train_views,
        train_labels,
        config(5),
        resume_checkpoint=loaded,
    )
    assert resumed.stop_reason == uninterrupted.stop_reason == "max_iterations"
    assert resumed.history == uninterrupted.history
    assert np.array_equal(resumed.weights, uninterrupted.weights)
    assert np.array_equal(resumed.alpha, uninterrupted.alpha)

    changed_views = (train_views[0].copy(), train_views[1].copy())
    changed_views[0][0, 0] += 1.0e-6
    with pytest.raises(
        DataValidationError,
        match="training view values or order differ",
    ):
        fit_amdr(
            changed_views,
            train_labels,
            config(5),
            resume_checkpoint=loaded,
        )


def test_amdr_rejects_unknown_convergence_metric() -> None:
    train_views, train_labels = _synthetic_views(5, 4)
    with pytest.raises(DataValidationError, match="convergence metric"):
        fit_amdr(
            train_views,
            train_labels,
            AMDRModelConfig(
                lambda_manifold=0.01,
                lambda_sparse=0.01,
                max_iterations=3,
                tolerance=1.0e-5,
                numerical_epsilon=1.0e-10,
                solve_ridge=1.0e-6,
                initialization_seed=3,
                convergence_metric="unknown",
            ),
        )


def test_same_base_graph_policy_excludes_repeated_endpoint_copies() -> None:
    train_views, train_labels = _synthetic_views(23, 4)
    ids = np.asarray(
        [
            "a", "a", "b", "c",
            "d", "d", "e", "f",
            "g", "g", "h", "i",
        ]
    )
    checkpoints = []
    fit_amdr(
        train_views,
        train_labels,
        AMDRModelConfig(
            lambda_manifold=0.01,
            lambda_sparse=0.01,
            max_iterations=1,
            minimum_iterations=1,
            tolerance=1.0e-30,
            numerical_epsilon=1.0e-10,
            solve_ridge=1.0e-6,
            initialization_seed=29,
            graph_same_base_policy=EXCLUDE_SAME_BASE_GRAPH,
        ),
        view_group_ids=(ids, ids),
        checkpoint_callback=checkpoints.append,
    )
    graph = checkpoints[-1].graphs[0][0]
    assert graph[0, 1] == 0.0
    assert graph[1, 0] == 0.0
    np.testing.assert_allclose(graph.sum(axis=1), np.ones(4))
