from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path

import pytest

from hrrp_osr.data.config import DataConfig
from hrrp_osr.data.errors import DataValidationError
from hrrp_osr.data.manifest import (
    build_manifest_rows,
    render_manifest_csv,
    validate_manifest_rows,
)


@pytest.fixture(scope="module")
def synthetic_build(data_config: DataConfig, synthetic_raw_root: Path):
    return build_manifest_rows(data_config, synthetic_raw_root)


def test_synthetic_mat_files_build_complete_manifest(synthetic_build) -> None:
    assert len(synthetic_build.rows) == 3600
    assert synthetic_build.validation["status"] == "passed"
    assert synthetic_build.validation["row_count"] == 3600
    assert synthetic_build.validation["class_count"] == 10
    assert {
        item["profile_length"] for item in synthetic_build.summary["classes"]
    } == {3, 5, 7}
    for class_name in {row["class_name"] for row in synthetic_build.rows}:
        rows = [row for row in synthetic_build.rows if row["class_name"] == class_name]
        assert Counter(row["split"] for row in rows) == {
            "train": 216,
            "validation": 72,
            "test": 72,
        }


def test_unknown_classes_are_never_train_or_validation_eligible(synthetic_build) -> None:
    unknown_rows = [row for row in synthetic_build.rows if row["class_role"] == "unknown"]
    assert unknown_rows
    assert all(row["eligible_for_training"] == 0 for row in unknown_rows)
    assert all(row["eligible_for_validation"] == 0 for row in unknown_rows)
    assert all(
        row["eligible_for_evaluation"] == int(row["split"] == "test")
        for row in unknown_rows
    )


def test_manifest_bytes_are_reproducible(
    data_config: DataConfig, synthetic_raw_root: Path, synthetic_build
) -> None:
    repeated = build_manifest_rows(data_config, synthetic_raw_root)
    assert render_manifest_csv(synthetic_build.rows) == render_manifest_csv(repeated.rows)


def test_validation_rejects_unknown_training_leakage(
    data_config: DataConfig, synthetic_build
) -> None:
    rows = [dict(row) for row in synthetic_build.rows]
    row = next(
        item for item in rows if item["class_role"] == "unknown" and item["split"] == "train"
    )
    row["eligible_for_training"] = 1
    row["usage"] = "model_train"
    with pytest.raises(DataValidationError, match="unknown class"):
        validate_manifest_rows(rows, data_config)


def test_validation_rejects_cross_split_source_identity_leakage(
    data_config: DataConfig, synthetic_build
) -> None:
    rows = [dict(row) for row in synthetic_build.rows]
    train = next(item for item in rows if item["split"] == "train")
    test = next(
        item
        for item in rows
        if item["class_name"] == train["class_name"] and item["split"] == "test"
    )
    test["source_file"] = train["source_file"]
    test["source_row_index"] = train["source_row_index"]
    with pytest.raises(DataValidationError, match="source file/row identity"):
        validate_manifest_rows(rows, data_config)


def test_validation_rejects_cross_split_derived_profile_copy(
    data_config: DataConfig, synthetic_build
) -> None:
    rows = [dict(row) for row in synthetic_build.rows]
    train = next(item for item in rows if item["split"] == "train")
    test = next(
        item
        for item in rows
        if item["class_name"] == train["class_name"] and item["split"] == "test"
    )
    test["profile_sha256"] = train["profile_sha256"]
    with pytest.raises(DataValidationError, match="profile hashes"):
        validate_manifest_rows(rows, data_config)


def test_validation_rejects_missing_angle(data_config: DataConfig, synthetic_build) -> None:
    rows = list(copy.deepcopy(synthetic_build.rows))
    rows.pop()
    with pytest.raises(DataValidationError, match="row count"):
        validate_manifest_rows(rows, data_config)
