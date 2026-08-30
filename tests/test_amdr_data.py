from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from hrrp_osr.amdr.data import (
    CANONICAL_SLOT_ORDER,
    PEAK_RELATIVE_AMPLITUDE_TRANSFORM_ID,
    PEAK_RELATIVE_POWER_TRANSFORM_ID,
    RANDOMIZED_SLOT_ORDER,
    _select_balanced_pairs,
    assign_odd_angle_folds,
    peak_relative_amplitude_from_power_db,
    peak_relative_power_from_db,
)
from hrrp_osr.data.errors import DataValidationError


def test_odd_angle_folds_are_equal_and_cover_every_frame() -> None:
    assignment = assign_odd_angle_folds(fold_count=5, base_seed=20260830)
    assert set(assignment) == set(range(1, 360, 2))
    assert Counter(assignment.values()) == {0: 36, 1: 36, 2: 36, 3: 36, 4: 36}
    for fold_index in range(5):
        assert {
            angle // 15
            for angle, assigned_fold in assignment.items()
            if assigned_fold == fold_index
        } == set(range(24))
    assert assignment == assign_odd_angle_folds(
        fold_count=5, base_seed=20260830
    )


def _synthetic_rows() -> list[dict[str, object]]:
    rows = []
    for angle in range(1, 60, 2):
        rows.append(
            {
                "sample_id": f"sample-{angle}",
                "class_name": "known-a",
                "class_role": "known",
                "angle_deg": angle,
                "processed_row_index": angle,
            }
        )
    return rows


def test_balanced_pair_sampling_is_reproducible_cross_frame_and_unique() -> None:
    rows = _synthetic_rows()
    kwargs = {
        "count": 40,
        "base_seed": 20260830,
        "protocol_id": "fixture",
        "split": "train",
        "fold_index": 0,
        "class_name": "known-a",
    }
    first = _select_balanced_pairs(rows, **kwargs)
    second = _select_balanced_pairs(rows, **kwargs)
    assert first == second
    assert len(first) == 40
    unordered = {
        frozenset((pair.view1_sample_id, pair.view2_sample_id)) for pair in first
    }
    assert len(unordered) == 40
    assert all(pair.view1_frame_id != pair.view2_frame_id for pair in first)
    usage = Counter(
        sample_id
        for pair in first
        for sample_id in (pair.view1_sample_id, pair.view2_sample_id)
    )
    assert max(usage.values()) - min(usage.values()) <= 1
    assert {pair.view1_frame_id for pair in first}
    assert {pair.view2_frame_id for pair in first}


def test_canonical_slots_preserve_selected_pairs_and_sort_endpoints() -> None:
    rows = _synthetic_rows()
    kwargs = {
        "count": 40,
        "base_seed": 20260830,
        "protocol_id": "fixture",
        "split": "train",
        "fold_index": 0,
        "class_name": "known-a",
    }
    randomized = _select_balanced_pairs(
        rows,
        slot_order=RANDOMIZED_SLOT_ORDER,
        **kwargs,
    )
    canonical = _select_balanced_pairs(
        rows,
        slot_order=CANONICAL_SLOT_ORDER,
        **kwargs,
    )
    randomized_unordered = {
        frozenset((pair.view1_sample_id, pair.view2_sample_id))
        for pair in randomized
    }
    canonical_unordered = {
        frozenset((pair.view1_sample_id, pair.view2_sample_id)) for pair in canonical
    }
    assert canonical_unordered == randomized_unordered
    assert all(pair.view1_angle_deg < pair.view2_angle_deg for pair in canonical)
    assert any(pair.view1_angle_deg > pair.view2_angle_deg for pair in randomized)


def test_peak_relative_power_matches_power_db_definition() -> None:
    profile = np.tile(np.linspace(-20.0, 0.0, 601), (2, 1))
    relative = peak_relative_power_from_db(profile)
    assert relative.shape == (2, 601)
    assert np.max(relative, axis=1) == pytest.approx(np.ones(2))
    assert relative[:, 0] == pytest.approx(np.full(2, 0.01))
    assert np.isfinite(relative).all()


def test_peak_relative_power_is_invariant_to_profile_db_offset() -> None:
    profile = np.tile(np.linspace(-40.0, 10.0, 601), (2, 1))
    transformed = peak_relative_power_from_db(profile)
    shifted = peak_relative_power_from_db(profile + np.asarray([[17.0], [-9.0]]))
    np.testing.assert_allclose(shifted, transformed, rtol=1.0e-14, atol=0.0)
    assert PEAK_RELATIVE_POWER_TRANSFORM_ID == "power_db_to_peak_relative_power_v1"


def test_peak_relative_amplitude_recovers_sqrt_of_relative_power() -> None:
    profile = np.tile(np.linspace(-40.0, 0.0, 601), (2, 1))
    amplitude = peak_relative_amplitude_from_power_db(profile)
    power = peak_relative_power_from_db(profile)
    np.testing.assert_allclose(amplitude * amplitude, power, rtol=1.0e-14, atol=0.0)
    assert amplitude[:, 0] == pytest.approx(np.full(2, 0.01))
    assert amplitude.max(axis=1) == pytest.approx(np.ones(2))
    assert (
        PEAK_RELATIVE_AMPLITUDE_TRANSFORM_ID
        == "power_db_to_peak_relative_amplitude_v1"
    )


def test_peak_relative_power_rejects_wrong_shape_and_nonfinite() -> None:
    with pytest.raises(DataValidationError, match="shape"):
        peak_relative_power_from_db(np.zeros((2, 600)))
    invalid = np.zeros((2, 601))
    invalid[0, 0] = np.nan
    with pytest.raises(DataValidationError, match="NaN or Inf"):
        peak_relative_power_from_db(invalid)
