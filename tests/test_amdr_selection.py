from __future__ import annotations

from pathlib import Path

from hrrp_osr.amdr.model import (
    FIXED_INITIAL_L21_REWEIGHTING,
    UPDATE_EACH_ITERATION_L21_REWEIGHTING,
)
from hrrp_osr.amdr.selection import (
    load_selection_config,
    select_best_candidates,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_selection_config_uses_calibration_and_never_test() -> None:
    config = load_selection_config(
        PROJECT_ROOT
        / "configs"
        / "amdr"
        / "pilot_fold0_d_strategy_parameter_selection_v1.yaml"
    )
    selection = config["selection"]
    assert selection["split"] == "calibration"
    assert selection["test_features_materialized"] is False
    assert selection["test_metrics_used"] is False
    assert selection["knn_k"] == 3


def test_local_graph_lambda2_selection_is_fixed_d_and_bounded() -> None:
    config = load_selection_config(
        PROJECT_ROOT
        / "configs"
        / "amdr"
        / "pilot_fold0_local_knn_lambda2_selection_v1.yaml"
    )
    selection = config["selection"]
    assert selection["l21_reweighting"] == [FIXED_INITIAL_L21_REWEIGHTING]
    assert selection["lambda_manifold"] == [1.0]
    assert selection["lambda_sparse"] == list(range(1, 11))
    assert selection["boundary_rule"] == (
        "report_boundary_without_automatic_extension"
    )


def test_select_best_candidates_uses_registered_tie_breaks() -> None:
    rows = []
    for strategy in (
        FIXED_INITIAL_L21_REWEIGHTING,
        UPDATE_EACH_ITERATION_L21_REWEIGHTING,
    ):
        rows.extend(
            [
                {
                    "candidate_id": f"{strategy}-large",
                    "l21_reweighting": strategy,
                    "lambda_manifold": 1.0,
                    "lambda_sparse": 1.0,
                    "calibration_accuracy": 0.8,
                    "calibration_macro_f1": 0.7,
                },
                {
                    "candidate_id": f"{strategy}-small",
                    "l21_reweighting": strategy,
                    "lambda_manifold": 0.1,
                    "lambda_sparse": 0.1,
                    "calibration_accuracy": 0.8,
                    "calibration_macro_f1": 0.7,
                },
            ]
        )

    selected = select_best_candidates(rows)
    assert all(row["candidate_id"].endswith("-small") for row in selected.values())
