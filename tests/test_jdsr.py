from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from hrrp_osr.data.errors import DataConfigError, DataValidationError
from hrrp_osr.jdsr import (
    dual_tail_unknown_scores,
    fit_dual_tail_gpd,
    jdsr_reconstruction_errors,
)
from hrrp_osr.training.b6 import load_b6_main_config, load_b6_paper_aligned_config


def test_jdsr_shared_class_block_with_view_specific_atoms_and_permutation() -> None:
    dictionary = np.asarray([
        [[1, 0, 0, 0], [0.9, 0.1, 0, 0], [1, 0.1, 0, 0]],
        [[0, 0, 1, 0], [0, 0, 0.9, 0.1], [0, 0, 1, 0.1]],
    ], dtype=np.float32)
    profiles = np.asarray([
        [[1, 0, 0, 0], [0.9, 0.1, 0, 0], [1, 0.1, 0, 0]],
        [[0, 0, 1, 0], [0, 0, 0.9, 0.1], [0, 0, 1, 0.1]],
    ], dtype=np.float32)
    reference = jdsr_reconstruction_errors(
        profiles, dictionary, sparsity=1, device=torch.device("cpu")
    )
    permuted = jdsr_reconstruction_errors(
        profiles[:, [2, 0, 1]], dictionary, sparsity=1, device=torch.device("cpu")
    )
    np.testing.assert_allclose(reference, permuted, atol=1e-7)
    assert reference.argmin(axis=1).tolist() == [0, 1]


def _gpd_fixture() -> tuple[np.ndarray, np.ndarray, list[str]]:
    class_zero = np.asarray([[0.05 + index * 0.002, 0.95] for index in range(12)])
    class_one = np.asarray([[0.95, 0.05 + index * 0.002] for index in range(12)])
    errors = np.concatenate([class_zero, class_one])
    labels = np.asarray([0] * 12 + [1] * 12)
    return errors, labels, [f"train-set-{index}" for index in range(24)]


def test_dual_tail_gpd_records_fit_ids_and_larger_means_more_unknown() -> None:
    errors, labels, ids = _gpd_fixture()
    fitted = fit_dual_tail_gpd(
        errors, labels, ids, class_count=2, rho=0.7, nonmatching_weight=1.0
    )
    assert fitted.fit_set_ids_by_class[0][0] == "train-set-0"
    assert fitted.fit_set_ids_by_class[1][-1] == "train-set-23"
    _, scores, matching, nonmatching = dual_tail_unknown_scores(
        np.asarray([[0.055, 0.95], [0.8, 0.81]]), fitted
    )
    assert scores[1] > scores[0]
    assert matching[1] >= matching[0]
    assert nonmatching[1] >= nonmatching[0]


def test_dual_tail_gpd_rejects_unknown_fit_labels() -> None:
    errors, labels, ids = _gpd_fixture()
    labels[0] = 2
    with pytest.raises(DataValidationError, match="unknown labels"):
        fit_dual_tail_gpd(
            errors, labels, ids, class_count=2, rho=0.7, nonmatching_weight=1.0
        )


def test_b6_main_config_freezes_dictionary_isolation_and_no_paper_delta() -> None:
    path = Path(__file__).resolve().parents[1] / "configs/experiments/p3/b6_main_v1.yaml"
    config = load_b6_main_config(path)
    assert config["dictionary"]["population"] == "known_train_hrrp_only"
    assert config["gpd"]["unknown_data_used"] is False
    assert config["evaluation"]["paper_fixed_delta_used"] is False
    assert config["set_protocol"]["view_count"] == 3


def test_b6_main_rejects_fixed_paper_threshold(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "configs/experiments/p3/b6_main_v1.yaml"
    config = copy.deepcopy(load_b6_main_config(source))
    config["evaluation"]["paper_fixed_delta_used"] = True
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    with pytest.raises(DataConfigError, match="fixed threshold"):
        load_b6_main_config(path)


def test_b6_paper_aligned_config_is_v5_and_separate() -> None:
    path = Path(__file__).resolve().parents[1] / "configs/experiments/p3/b6_paper_aligned_v5_v1.yaml"
    config = load_b6_paper_aligned_config(path)
    assert config["result_scope"] == "paper_aligned_v5"
    assert config["set_protocol"]["view_count"] == 5
    assert config["solver"]["sparsity"] == 2
    assert config["gpd"]["rho"] == 0.7
    assert config["evaluation"]["paper_fixed_delta_role"] == "diagnostic_operating_point_only"
    assert config["evaluation"]["permutation_atol"] == 1.0e-4
