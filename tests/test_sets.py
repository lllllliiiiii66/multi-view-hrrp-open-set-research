from __future__ import annotations

from hrrp_osr.data.config import DataConfig
from hrrp_osr.data.manifest import build_manifest_rows
from hrrp_osr.data.sets import (
    build_v3_sets,
    build_v3_evaluation_sets,
    build_v5_sets,
    render_set_manifest_csv,
    select_b0_single_view,
)


def test_v3_training_sets_exactly_cover_known_training_pool(
    data_config: DataConfig, synthetic_raw_root
) -> None:
    build = build_manifest_rows(data_config, synthetic_raw_root)
    sets = build_v3_sets(build.rows, split="train", base_seed=20260810)
    assert len(sets) == 7 * 72
    member_ids = [sample_id for item in sets for sample_id in item.member_sample_ids]
    assert len(member_ids) == len(set(member_ids)) == 7 * 216
    assert all(len(set(item.member_domain_ids)) == 3 for item in sets)


def test_v5_leave_one_domain_out_sets_have_balanced_fivefold_coverage(
    data_config: DataConfig, synthetic_raw_root
) -> None:
    build = build_manifest_rows(data_config, synthetic_raw_root)
    sets = build_v5_sets(build.rows, split="test", base_seed=20260810)
    assert len(sets) == 10 * 72
    counts: dict[str, int] = {}
    for item in sets:
        assert len(set(item.member_domain_ids)) == 5
        for sample_id in item.member_sample_ids:
            counts[sample_id] = counts.get(sample_id, 0) + 1
    assert len(counts) == 10 * 72
    assert set(counts.values()) == {5}
    assert render_set_manifest_csv(sets) == render_set_manifest_csv(
        build_v5_sets(build.rows, split="test", base_seed=20260810)
    )


def test_v3_validation_sets_have_exact_coverage_and_distinct_domains(
    data_config: DataConfig, synthetic_raw_root
) -> None:
    build = build_manifest_rows(data_config, synthetic_raw_root)
    sets = build_v3_evaluation_sets(
        build.rows,
        split="validation",
        base_seed=20260810,
    )
    assert len(sets) == 7 * 24
    assert all(item.class_role == "known" for item in sets)
    assert all(len(set(item.member_domain_ids)) == 3 for item in sets)
    member_ids = [sample_id for item in sets for sample_id in item.member_sample_ids]
    assert len(member_ids) == len(set(member_ids)) == 7 * 72


def test_v3_test_sets_include_known_and_unknown_without_base_sample_reuse(
    data_config: DataConfig, synthetic_raw_root
) -> None:
    build = build_manifest_rows(data_config, synthetic_raw_root)
    sets = build_v3_evaluation_sets(
        build.rows,
        split="test",
        base_seed=20260810,
    )
    assert len(sets) == 10 * 24
    assert {item.class_role for item in sets} == {"known", "unknown"}
    member_ids = [sample_id for item in sets for sample_id in item.member_sample_ids]
    assert len(member_ids) == len(set(member_ids)) == 10 * 72


def test_v3_set_generation_is_seeded_and_byte_reproducible(
    data_config: DataConfig, synthetic_raw_root
) -> None:
    build = build_manifest_rows(data_config, synthetic_raw_root)
    first = build_v3_evaluation_sets(
        build.rows, split="test", base_seed=20260810
    )
    repeated = build_v3_evaluation_sets(
        build.rows, split="test", base_seed=20260810
    )
    changed = build_v3_evaluation_sets(
        build.rows, split="test", base_seed=20260811
    )
    assert first == repeated
    assert render_set_manifest_csv(first) == render_set_manifest_csv(repeated)
    assert first != changed


def test_b0_single_view_selection_is_not_array_first_and_is_reproducible(
    data_config: DataConfig, synthetic_raw_root
) -> None:
    build = build_manifest_rows(data_config, synthetic_raw_root)
    sets = build_v3_evaluation_sets(
        build.rows, split="test", base_seed=20260810
    )
    first = [
        select_b0_single_view(item, base_seed=20260820, selection_repeat=0)
        for item in sets
    ]
    repeated = [
        select_b0_single_view(item, base_seed=20260820, selection_repeat=0)
        for item in sets
    ]
    next_repeat = [
        select_b0_single_view(item, base_seed=20260820, selection_repeat=1)
        for item in sets
    ]
    assert first == repeated
    assert any(item.selected_index != 0 for item in first)
    assert {item.selected_index for item in first} == {0, 1, 2}
    assert first != next_repeat
    for view_set, selection in zip(sets, first, strict=True):
        assert selection.selected_sample_id == view_set.member_sample_ids[
            selection.selected_index
        ]
