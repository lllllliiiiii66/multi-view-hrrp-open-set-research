from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from hrrp_osr.amdr.data import (
    _select_balanced_pairs,
    assign_odd_angle_folds,
    relative_power_from_db,
)


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


def test_relative_power_normalization_matches_power_db_definition() -> None:
    profile = np.tile(np.linspace(-20.0, 0.0, 601), (2, 1))
    relative = relative_power_from_db(profile)
    assert relative.shape == (2, 601)
    assert np.max(relative, axis=1) == pytest.approx(np.ones(2))
    assert relative[:, 0] == pytest.approx(np.full(2, 0.01))
    assert np.isfinite(relative).all()
