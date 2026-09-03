from __future__ import annotations

import copy
import csv
import io
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
import yaml

from hrrp_osr.data.errors import DataConfigError
from hrrp_osr.data.processed import ProcessedBundle
from hrrp_osr.training.ms_mean_head_factorial import (
    METHODS,
    _prepare_split,
    build_phase_plan,
    load_ms_mean_head_factorial_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/experiments/arpl/ms_mean_head_factorial_surrogate_v1.yaml"
)

EXPECTED_METHODS = (
    "R0_SHALLOW_MEAN_CE",
    "R1_SHALLOW_MEAN_ARPL",
    "R2_MS_MEAN_CE",
    "R3_MS_MEAN_ARPL",
)
EXPECTED_PAIRS = {
    "N0": ((0, 2), (1, 3, 4, 5, 6)),
    "N1": ((2, 5), (0, 1, 3, 4, 6)),
    "N2": ((3, 5), (0, 1, 2, 4, 6)),
    "N3": ((1, 3), (0, 2, 4, 5, 6)),
    "N4": ((1, 6), (0, 2, 3, 4, 5)),
    "N5": ((4, 6), (0, 1, 2, 3, 5)),
    "N6": ((0, 4), (1, 2, 3, 5, 6)),
}
PROHIBITED_PAIRS = {
    (0, 1),
    (2, 3),
    (4, 5),
    (0, 6),
    (1, 5),
    (2, 4),
    (3, 6),
}
EXPECTED_FOLDS = (0, 4)
EXPECTED_SEEDS = (20260830, 20260831, 20260832)


def _synthetic_bundle() -> ProcessedBundle:
    rows: list[dict[str, object]] = []
    profiles = np.zeros((3600, 601), dtype=np.float64)
    row_index = 0
    bins = np.arange(601, dtype=np.float64) / 100_000.0
    for class_index in range(10):
        class_name = f"class-{class_index}"
        class_role = "known" if class_index < 7 else "unknown"
        for angle in range(360):
            # Poison both forbidden populations.  Preparing a surrogate split
            # remains finite only if final-unknown and even-angle profiles are
            # never indexed or materialized.
            if class_role == "unknown" or angle % 2 == 0:
                profiles[row_index] = np.nan
            else:
                offset = 1_000_000.0 if class_index in {0, 2} else class_index
                profiles[row_index] = offset + angle / 1000.0 + bins
            rows.append(
                {
                    "sample_id": f"{class_name}-{angle}",
                    "class_name": class_name,
                    "class_role": class_role,
                    "angle_deg": angle,
                    "processed_row_index": row_index,
                }
            )
            row_index += 1
    return ProcessedBundle(
        root=Path("/synthetic"),
        profiles=profiles,
        rows=tuple(rows),
        profiles_sha256="a" * 64,
        manifest_sha256="b" * 64,
        bundle_sha256="c" * 64,
    )


def _synthetic_config() -> dict[str, Any]:
    config = copy.deepcopy(load_ms_mean_head_factorial_config(CONFIG_PATH))
    config["classes"]["source_known_order"] = [
        f"class-{index}" for index in range(7)
    ]
    return config


def _manifest_rows(prepared: Any) -> list[dict[str, str]]:
    return list(
        csv.DictReader(
            io.StringIO(prepared.pair_manifest_bytes.decode("utf-8"))
        )
    )


def test_config_freezes_factorial_scope_and_training_contract() -> None:
    config = load_ms_mean_head_factorial_config(CONFIG_PATH)

    assert tuple(METHODS) == EXPECTED_METHODS
    assert tuple(config["model"]["methods"]) == EXPECTED_METHODS
    assert tuple(config["training"]["methods"]) == EXPECTED_METHODS
    assert tuple(config["classes"]["angle_folds"]) == EXPECTED_FOLDS
    assert tuple(config["training"]["confirmation_seeds"]) == EXPECTED_SEEDS
    assert config["sampling"]["pairs_per_class"]["full"] == 500
    assert config["training"]["total_epochs"] == 100
    assert config["training"]["formal_checkpoint_epoch"] == 100
    assert config["training"]["early_stopping"] is False
    assert config["training"]["calibration_checkpoint_selection"] is False
    assert config["training"]["performance_fallback"] is False

    scope = config["evidence_scope"]
    assert scope["source_known_odd_angle_only"] is True
    assert scope["final_unknown_classes_used"] is False
    assert scope["even_angle_test_used"] is False
    assert scope["surrogate_unknown_used_for_training"] is False
    assert scope["surrogate_unknown_used_for_normalization"] is False
    assert scope["surrogate_unknown_used_for_threshold"] is False
    assert config["sampling"]["final_test_pairs_generated"] is False
    assert config["decision"]["final_unknown_test_authorized"] is False


def test_identity_pairs_are_exact_balanced_complements_and_previously_unseen() -> None:
    config = load_ms_mean_head_factorial_config(CONFIG_PATH)
    rows = config["classes"]["identity_pairs"]
    observed = {
        str(row["pair_id"]): (
            tuple(int(value) for value in row["surrogate_unknown_indices"]),
            tuple(int(value) for value in row["train_known_indices"]),
        )
        for row in rows
    }

    assert observed == EXPECTED_PAIRS
    appearances = Counter(
        index for unknown, _ in observed.values() for index in unknown
    )
    assert appearances == Counter({index: 2 for index in range(7)})
    assert {
        tuple(sorted(unknown)) for unknown, _ in observed.values()
    }.isdisjoint(PROHIBITED_PAIRS)
    for unknown, train_known in observed.values():
        assert not set(unknown) & set(train_known)
        assert set(unknown) | set(train_known) == set(range(7))


@pytest.mark.parametrize(
    "mutate",
    (
        lambda config: config["classes"]["identity_pairs"][0].update(
            surrogate_unknown_indices=[0, 1],
            train_known_indices=[2, 3, 4, 5, 6],
        ),
        lambda config: config["classes"].update(angle_folds=[0]),
        lambda config: config["training"].update(
            confirmation_seeds=[20260830, 20260831]
        ),
        lambda config: config["evidence_scope"].update(
            final_unknown_classes_used=True
        ),
        lambda config: config["evidence_scope"].update(even_angle_test_used=True),
        lambda config: config["sampling"].update(final_test_pairs_generated=True),
        lambda config: config["decision"].update(
            final_unknown_test_authorized=True
        ),
    ),
)
def test_config_rejects_factorial_or_final_test_mutation(
    tmp_path: Path, mutate: Callable[[dict[str, Any]], None]
) -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    mutate(config)
    changed = tmp_path / "changed.yaml"
    changed.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(DataConfigError):
        load_ms_mean_head_factorial_config(changed)


def test_confirmation_plan_is_exact_42_units_and_168_training_tasks() -> None:
    config = load_ms_mean_head_factorial_config(CONFIG_PATH)
    first = build_phase_plan(config, "confirmation")
    second = build_phase_plan(copy.deepcopy(config), "confirmation")

    assert first == second
    assert len(first) == 7 * 2 * 3
    assert sum(len(unit["methods"]) for unit in first) == 168
    assert all(tuple(unit["methods"]) == EXPECTED_METHODS for unit in first)
    assert all(unit["mode"] == "full" for unit in first)
    assert {
        (
            str(unit["spec"]["pair_id"]),
            int(unit["spec"]["angle_fold"]),
            int(unit["seed"]),
        )
        for unit in first
    } == {
        (pair_id, angle_fold, seed)
        for pair_id in EXPECTED_PAIRS
        for angle_fold in EXPECTED_FOLDS
        for seed in EXPECTED_SEEDS
    }
    for pair_id in EXPECTED_PAIRS:
        rows = [unit for unit in first if unit["spec"]["pair_id"] == pair_id]
        assert {int(unit["spec"]["angle_fold"]) for unit in rows} == {0, 4}
        assert Counter(int(unit["seed"]) for unit in rows) == Counter(
            {seed: 2 for seed in EXPECTED_SEEDS}
        )


def test_smoke_plan_is_only_n0_fold0_seed20260830_with_all_four_methods() -> None:
    config = load_ms_mean_head_factorial_config(CONFIG_PATH)
    plan = build_phase_plan(config, "smoke")

    assert len(plan) == 1
    unit = plan[0]
    assert unit["spec"]["pair_id"] == "N0"
    assert int(unit["spec"]["angle_fold"]) == 0
    assert int(unit["seed"]) == 20260830
    assert tuple(unit["methods"]) == EXPECTED_METHODS
    assert unit["mode"] == "smoke"


def test_prepare_split_is_deterministic_across_initialization_seeds() -> None:
    bundle = _synthetic_bundle()
    config = _synthetic_config()
    units = [
        unit
        for unit in build_phase_plan(config, "confirmation")
        if unit["spec"]["pair_id"] == "N0"
        and int(unit["spec"]["angle_fold"]) == 0
    ]

    prepared = [
        _prepare_split(bundle, config, unit["spec"], mode="smoke")
        for unit in units
    ]
    assert len({item.pair_manifest_sha256 for item in prepared}) == 1
    assert len({item.pair_manifest_bytes for item in prepared}) == 1


def test_pair_sampling_random_stream_is_shared_by_fold_not_identity_pair() -> None:
    bundle = _synthetic_bundle()
    config = _synthetic_config()
    specifications = {
        unit["pair_id"]: unit["spec"]
        for unit in build_phase_plan(config, "confirmation")
        if int(unit["angle_fold"]) == 0 and int(unit["seed"]) == 20260830
    }
    n0 = _manifest_rows(_prepare_split(bundle, config, specifications["N0"], mode="smoke"))
    n1 = _manifest_rows(_prepare_split(bundle, config, specifications["N1"], mode="smoke"))

    def sampled_pairs(rows: list[dict[str, str]], class_name: str, role: str) -> list[str]:
        return [
            row["pair_id"]
            for row in rows
            if row["class_name"] == class_name and row["experiment_role"] == role
        ]

    # class-1 is train-known in both N0 and N1; class-2 is surrogate in both.
    # Their pair samples must therefore be identical within the shared fold.
    assert sampled_pairs(n0, "class-1", "train_known") == sampled_pairs(
        n1, "class-1", "train_known"
    )
    assert sampled_pairs(n0, "class-1", "known_calibration") == sampled_pairs(
        n1, "class-1", "known_calibration"
    )
    assert sampled_pairs(n0, "class-2", "surrogate_unknown") == sampled_pairs(
        n1, "class-2", "surrogate_unknown"
    )


@pytest.mark.parametrize("angle_fold", EXPECTED_FOLDS)
def test_prepare_split_blocks_final_unknown_even_angles_and_surrogate_leakage(
    angle_fold: int,
) -> None:
    bundle = _synthetic_bundle()
    config = _synthetic_config()
    spec = next(
        unit["spec"]
        for unit in build_phase_plan(config, "confirmation")
        if unit["spec"]["pair_id"] == "N0"
        and int(unit["spec"]["angle_fold"]) == angle_fold
    )
    prepared = _prepare_split(bundle, config, spec, mode="smoke")
    rows = _manifest_rows(prepared)

    assert prepared.split_id == f"N0_F{angle_fold}"
    assert prepared.angle_fold == angle_fold
    assert prepared.train_class_order == (
        "class-1",
        "class-3",
        "class-4",
        "class-5",
        "class-6",
    )
    assert prepared.surrogate_class_order == ("class-0", "class-2")
    assert set(prepared.inputs) == {
        "train",
        "known_calibration",
        "surrogate_unknown",
    }
    assert prepared.inputs["train"].shape == (50, 2, 601)
    assert prepared.inputs["known_calibration"].shape == (50, 2, 601)
    assert prepared.inputs["surrogate_unknown"].shape == (20, 2, 601)
    assert all(np.isfinite(value).all() for value in prepared.inputs.values())
    assert set(prepared.labels["train"].tolist()) == set(range(5))
    assert set(prepared.labels["known_calibration"].tolist()) == set(range(5))
    assert set(prepared.labels["surrogate_unknown"].tolist()) == {5}

    assert prepared.pair_audit["train_evaluation_base_overlap"] == 0
    assert prepared.pair_audit["final_unknown_pairs"] == 0
    assert prepared.pair_audit["even_angle_pairs"] == 0
    assert prepared.pair_audit["test_pairs_generated"] is False
    assert prepared.pair_audit["test_features_materialized"] is False
    assert prepared.pair_audit["surrogate_train_pairs_materialized"] is False
    source_audit = prepared.pair_audit["source_pair_audit"]
    assert source_audit["included_splits"] == ["train", "calibration"]
    assert source_audit["fold_assignment"]["fold_index"] == angle_fold
    assert source_audit["fold_assignment"]["odd_angles_per_fold"] == {
        str(fold): 36 for fold in range(5)
    }
    assert source_audit["fold_assignment"]["every_fold_covers_all_frames"] is True

    assert {row["experiment_role"] for row in rows} == {
        "train_known",
        "known_calibration",
        "surrogate_unknown",
    }
    assert not any(
        row["class_name"] in {"class-7", "class-8", "class-9"}
        for row in rows
    )
    assert not any(
        int(row[field]) % 2 == 0
        for row in rows
        for field in ("view1_angle_deg", "view2_angle_deg")
    )
    assert all(row["split"] in {"train", "calibration"} for row in rows)
    assert all(row["view1_frame_id"] != row["view2_frame_id"] for row in rows)
    assert {
        row["class_name"]
        for row in rows
        if row["experiment_role"] == "surrogate_unknown"
    } == {"class-0", "class-2"}
    assert not any(
        row["class_name"] in {"class-0", "class-2"}
        and row["experiment_role"] in {"train_known", "known_calibration"}
        for row in rows
    )

    row_by_sample_id = {str(row["sample_id"]): row for row in bundle.rows}
    train_sample_ids = {
        row[field]
        for row in rows
        if row["experiment_role"] == "train_known"
        for field in ("view1_sample_id", "view2_sample_id")
    }
    evaluation_sample_ids = {
        row[field]
        for row in rows
        if row["experiment_role"] != "train_known"
        for field in ("view1_sample_id", "view2_sample_id")
    }
    assert train_sample_ids.isdisjoint(evaluation_sample_ids)
    assert {
        str(row_by_sample_id[sample_id]["class_name"])
        for sample_id in train_sample_ids
    }.isdisjoint({"class-0", "class-2", "class-7", "class-8", "class-9"})

    train_indices = np.asarray(
        [
            int(row_by_sample_id[sample_id]["processed_row_index"])
            for sample_id in train_sample_ids
        ],
        dtype=np.int64,
    )
    expected_values = np.asarray(bundle.profiles[train_indices])
    assert prepared.normalization.unique_base_sample_count == len(train_sample_ids)
    assert prepared.normalization.mean == pytest.approx(float(expected_values.mean()))
    assert prepared.normalization.std == pytest.approx(float(expected_values.std()))
