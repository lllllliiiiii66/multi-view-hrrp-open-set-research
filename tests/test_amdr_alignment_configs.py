from __future__ import annotations

from pathlib import Path

from hrrp_osr.amdr.data import (
    CANONICAL_SLOT_ORDER,
    PEAK_RELATIVE_AMPLITUDE_TRANSFORM_ID,
    RANDOMIZED_SLOT_ORDER,
)
from hrrp_osr.amdr.model import (
    ALLOW_SAME_BASE_GRAPH,
    EXCLUDE_SAME_BASE_GRAPH,
    FIXED_INITIAL_L21_REWEIGHTING,
    LOCAL_KNN_GAUSSIAN_GRAPH,
    UPDATE_EACH_ITERATION_L21_REWEIGHTING,
)
from hrrp_osr.amdr.reduction import SHARED_TRAIN_BASE_PCA
from hrrp_osr.amdr.smoke import load_smoke_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_alignment_pilot_configs_are_cumulative_and_protocol_matched() -> None:
    names = (
        "pilot_fold0_amplitude_v1.yaml",
        "pilot_fold0_amplitude_pruned_v1.yaml",
        "pilot_fold0_amplitude_pruned_canonical_v1.yaml",
    )
    configs = [
        load_smoke_config(PROJECT_ROOT / "configs" / "amdr" / name)
        for name in names
    ]
    assert {config["protocol_id"] for config in configs} == {
        "amdr_10class_odd_even_crossfit_pilot_fold0_v1"
    }
    assert {config["bundle"]["bundle_sha256"] for config in configs} == {
        "79176091b5b745e5df5957f1bb4ade7304c1403a4a32e64622e012db67d5a0c5"
    }
    assert all(
        config["preprocessing"]["transform"]
        == PEAK_RELATIVE_AMPLITUDE_TRANSFORM_ID
        for config in configs
    )
    assert [
        float(config["model"]["post_training_row_prune_squared_norm_threshold"])
        for config in configs
    ] == [0.0, 1.0e-5, 1.0e-5]
    assert [config["sampling"]["slot_order"] for config in configs] == [
        RANDOMIZED_SLOT_ORDER,
        RANDOMIZED_SLOT_ORDER,
        CANONICAL_SLOT_ORDER,
    ]


def test_overfit_diagnostic_configs_change_only_the_registered_mechanism() -> None:
    names = (
        "pilot_fold0_amplitude_pca200_v1.yaml",
        "pilot_fold0_amplitude_pca200_nosamebase_v1.yaml",
    )
    configs = [
        load_smoke_config(PROJECT_ROOT / "configs" / "amdr" / name)
        for name in names
    ]
    assert {config["protocol_id"] for config in configs} == {
        "amdr_10class_odd_even_crossfit_pilot_fold0_v1"
    }
    assert all(
        config["preprocessing"]["dimension_reduction"]["algorithm"]
        == SHARED_TRAIN_BASE_PCA
        for config in configs
    )
    assert all(
        config["preprocessing"]["dimension_reduction"]["output_dimension"] == 200
        for config in configs
    )
    assert [config["model"]["graph_same_base_policy"] for config in configs] == [
        ALLOW_SAME_BASE_GRAPH,
        EXCLUDE_SAME_BASE_GRAPH,
    ]


def test_calibration_selected_d_strategy_configs_are_matched() -> None:
    names = (
        "pilot_fold0_amplitude_pruned_fixed_d_selected_v1.yaml",
        "pilot_fold0_amplitude_pruned_dynamic_d_selected_v1.yaml",
    )
    configs = [
        load_smoke_config(PROJECT_ROOT / "configs" / "amdr" / name)
        for name in names
    ]
    assert {config["protocol_id"] for config in configs} == {
        "amdr_10class_odd_even_crossfit_pilot_fold0_v1"
    }
    assert [config["model"]["l21_reweighting"] for config in configs] == [
        FIXED_INITIAL_L21_REWEIGHTING,
        UPDATE_EACH_ITERATION_L21_REWEIGHTING,
    ]
    assert all(config["model"]["lambda_manifold"] == 1.0 for config in configs)
    assert all(config["model"]["lambda_sparse"] == 1.0 for config in configs)
    assert all(
        config["parameter_selection"]["test_metrics_used"] is False
        for config in configs
    )


def test_local_graph_lambda2_base_changes_only_registered_graph_mechanism() -> None:
    baseline = load_smoke_config(
        PROJECT_ROOT
        / "configs"
        / "amdr"
        / "pilot_fold0_amplitude_pruned_fixed_d_selected_v1.yaml"
    )
    local = load_smoke_config(
        PROJECT_ROOT
        / "configs"
        / "amdr"
        / "pilot_fold0_amplitude_pruned_fixed_d_local_knn_v1.yaml"
    )
    assert local["protocol_id"] == baseline["protocol_id"]
    assert local["sampling"] == baseline["sampling"]
    assert local["preprocessing"] == baseline["preprocessing"]
    assert local["model"]["lambda_manifold"] == 1.0
    assert local["model"]["lambda_sparse"] == 1.0
    assert local["model"]["l21_reweighting"] == FIXED_INITIAL_L21_REWEIGHTING
    assert local["model"]["graph_neighborhood"] == LOCAL_KNN_GAUSSIAN_GRAPH
    assert local["model"]["graph_neighbor_count"] == 10


def test_posthoc_local_graph_selected_config_is_explicitly_diagnostic() -> None:
    config = load_smoke_config(
        PROJECT_ROOT
        / "configs"
        / "amdr"
        / "pilot_fold0_amplitude_pruned_fixed_d_local_knn_lambda2_posthoc_selected_v1.yaml"
    )
    assert config["model"]["lambda_manifold"] == 1.0
    assert config["model"]["lambda_sparse"] == 50.0
    assert config["model"]["l21_reweighting"] == FIXED_INITIAL_L21_REWEIGHTING
    assert config["diagnostic_parent"]["posthoc_after_fold0_test_seen"] is True
    assert config["diagnostic_parent"]["confirmatory_claim_allowed"] is False
    assert config["parameter_selection"]["test_metrics_used"] is False
