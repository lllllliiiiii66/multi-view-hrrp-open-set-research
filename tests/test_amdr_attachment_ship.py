from __future__ import annotations

from pathlib import Path

import numpy as np

from hrrp_osr.amdr.attachment_ship import (
    knn_predict_multiple_k,
    load_attachment_ship_config,
)
from hrrp_osr.amdr.model import knn_predict_and_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_attachment_ship_config_is_frozen_and_diagnostic() -> None:
    config = load_attachment_ship_config(
        PROJECT_ROOT
        / "configs"
        / "amdr"
        / "attachment_ship_python_reference_v1.yaml"
    )
    assert config["result_scope"] == "diagnostic_attachment_ship"
    assert config["dataset"]["train_sample_count"] == 12015
    assert config["dataset"]["test_sample_count"] == 8010
    assert config["knn"]["selection"] == "none_report_all_test_results"


def test_multi_k_knn_matches_individual_reference_calls() -> None:
    rng = np.random.default_rng(31)
    train = rng.normal(size=(30, 4))
    labels = np.repeat(np.arange(3), 10)
    query = rng.normal(size=(7, 4))
    result = knn_predict_multiple_k(
        train,
        labels,
        query,
        k_values=(1, 3, 5),
    )
    for k in (1, 3, 5):
        expected, _ = knn_predict_and_score(
            train,
            labels,
            query,
            k=k,
        )
        np.testing.assert_array_equal(result[k], expected)
