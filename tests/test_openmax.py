from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from hrrp_osr.data.errors import DataConfigError, DataValidationError
from hrrp_osr.openmax import fit_openmax, openmax_probabilities
from hrrp_osr.training.openmax import (
    OPENMAX_ALPHA_RANK_CANDIDATES,
    OPENMAX_DISTANCE_TYPE_CANDIDATES,
    OPENMAX_SELECTION_RULE,
    OPENMAX_TAIL_SIZE_CANDIDATES,
    _audit_permutations,
    _select_openmax_model,
    load_openmax_config,
)


def _fit_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    activations = np.asarray([
        [9.8, 0.2], [10.0, 0.0], [10.2, -0.1], [9.9, 0.1],
        [0.2, 9.8], [0.0, 10.0], [-0.1, 10.2], [0.1, 9.9],
    ])
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    predictions = labels.copy()
    return activations, labels, predictions, [f"train-{index}" for index in range(8)]


def test_openmax_fits_only_correct_training_entities() -> None:
    activations, labels, predictions, ids = _fit_fixture()
    predictions[3] = 1
    fitted = fit_openmax(
        activations, labels, predictions, ids, class_count=2, tail_size=20,
        alpha_rank=2, distance_type="euclidean", eucos_euclidean_scale=200.0,
    )
    assert fitted.fit_sample_ids_by_class[0] == ("train-0", "train-1", "train-2")
    assert "train-3" not in fitted.fit_sample_ids_by_class[0]
    assert fitted.tails[0].effective_tail_size == 3
    assert fitted.tails[1].effective_tail_size == 4


def test_openmax_rejects_empty_class() -> None:
    activations, labels, predictions, ids = _fit_fixture()
    predictions[labels == 1] = 0
    with pytest.raises(DataValidationError, match="no correctly classified"):
        fit_openmax(
            activations, labels, predictions, ids, class_count=2, tail_size=20,
            alpha_rank=2, distance_type="euclidean", eucos_euclidean_scale=200.0,
        )


def test_openmax_probabilities_are_finite_normalized_and_outlier_directed() -> None:
    activations, labels, predictions, ids = _fit_fixture()
    fitted = fit_openmax(
        activations, labels, predictions, ids, class_count=2, tail_size=20,
        alpha_rank=2, distance_type="euclidean", eucos_euclidean_scale=200.0,
    )
    probabilities = openmax_probabilities(np.asarray([[10.0, 0.0], [100.0, -20.0]]), fitted)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(np.isfinite(probabilities))
    assert probabilities[1, -1] > probabilities[0, -1]


@pytest.mark.parametrize(("name", "baseline", "source"), [
    ("b4_main_v1.yaml", "B4", "B0"),
    ("b5_main_v1.yaml", "B5", "B2"),
])
def test_openmax_configs_freeze_checkpoint_and_fit_isolation(name: str, baseline: str, source: str) -> None:
    path = Path(__file__).resolve().parents[1] / "configs/experiments/p2" / name
    config = load_openmax_config(path)
    assert config["baseline"] == baseline
    assert config["checkpoint_reuse"]["source_baseline"] == source
    assert config["checkpoint_reuse"]["training_allowed"] is False
    assert config["openmax_fitting"]["unknown_data_used"] is False


def test_b5_config_rejects_b4_checkpoint_reuse(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "configs/experiments/p2/b5_main_v1.yaml"
    config = copy.deepcopy(load_openmax_config(source))
    config["checkpoint_reuse"]["source_baseline"] = "B0"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    with pytest.raises(DataConfigError, match="reuse B2"):
        load_openmax_config(path)


@pytest.mark.parametrize("name", ["b4_paper_aligned_v5_v1.yaml", "b5_paper_aligned_v5_v1.yaml"])
def test_openmax_v5_configs_are_separate_auxiliary_results(name: str) -> None:
    path = Path(__file__).resolve().parents[1] / "configs/experiments/p3" / name
    config = load_openmax_config(path)
    assert config["stage"] == "P3"
    assert config["result_scope"] == "paper_aligned_v5"
    assert config["set_protocol"]["view_count"] == 5
    assert len(_audit_permutations(5)) == 10


@pytest.mark.parametrize(("directory", "name", "baseline", "source", "view_count"), [
    ("p2", "b4_main_v2.yaml", "B4", "B0", 3),
    ("p2", "b5_main_v2.yaml", "B5", "B2", 3),
    ("p3", "b4_paper_aligned_v5_v2.yaml", "B4", "B0", 5),
    ("p3", "b5_paper_aligned_v5_v2.yaml", "B5", "B2", 5),
])
def test_openmax_v2_configs_share_known_validation_candidate_grid(
    directory: str, name: str, baseline: str, source: str, view_count: int
) -> None:
    path = Path(__file__).resolve().parents[1] / "configs/experiments" / directory / name
    config = load_openmax_config(path)
    parameters = config["openmax_parameters"]
    assert config["baseline"] == baseline
    assert config["checkpoint_reuse"]["source_baseline"] == source
    assert config["set_protocol"]["view_count"] == view_count
    assert parameters["selection_rule"] == OPENMAX_SELECTION_RULE
    assert parameters["tail_size_candidates"] == OPENMAX_TAIL_SIZE_CANDIDATES
    assert parameters["alpha_rank_candidates"] == OPENMAX_ALPHA_RANK_CANDIDATES
    assert parameters["distance_type_candidates"] == OPENMAX_DISTANCE_TYPE_CANDIDATES
    assert config["openmax_fitting"]["unknown_data_used"] is False


def _seven_class_selection_fixture() -> tuple[np.ndarray, np.ndarray, list[str]]:
    activations = []
    labels = []
    for class_index in range(7):
        for offset in (-0.15, -0.05, 0.05, 0.15):
            row = np.full(7, -1.0)
            row[class_index] = 8.0 + offset
            activations.append(row)
            labels.append(class_index)
    return np.asarray(activations), np.asarray(labels), [f"train-{index}" for index in range(28)]


def test_openmax_v2_selects_from_known_validation_only() -> None:
    path = Path(__file__).resolve().parents[1] / "configs/experiments/p2/b5_main_v2.yaml"
    config = load_openmax_config(path)
    activations, labels, ids = _seven_class_selection_fixture()
    validation = activations + 0.02
    fitted, probabilities, actual_labels, audit = _select_openmax_model(
        activations,
        labels,
        labels.copy(),
        ids,
        config,
        lambda candidate: (openmax_probabilities(validation, candidate), labels),
    )
    assert fitted.class_count == 7
    assert probabilities.shape == (28, 8)
    np.testing.assert_array_equal(actual_labels, labels)
    assert audit["candidate_count"] == 27
    assert audit["known_validation_only"] is True
    assert audit["unknown_validation_entity_count"] == 0
    assert len([candidate for candidate in audit["candidates"] if candidate.get("selected")]) == 1


def test_openmax_v2_rejects_unknown_validation_labels() -> None:
    path = Path(__file__).resolve().parents[1] / "configs/experiments/p2/b5_main_v2.yaml"
    config = load_openmax_config(path)
    activations, labels, ids = _seven_class_selection_fixture()
    forbidden_labels = labels.copy()
    forbidden_labels[0] = 7
    with pytest.raises(DataValidationError, match="non-known validation"):
        _select_openmax_model(
            activations,
            labels,
            labels.copy(),
            ids,
            config,
            lambda candidate: (openmax_probabilities(activations, candidate), forbidden_labels),
        )
