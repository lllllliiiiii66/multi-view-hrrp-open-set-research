from __future__ import annotations

import copy
import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from hrrp_osr.data.config import DataConfig
from hrrp_osr.data.errors import DataConfigError, DataValidationError
from hrrp_osr.data.manifest import ManifestBuild, build_manifest_rows, render_manifest_csv
from hrrp_osr.data.padding import (
    PaddingConfig,
    audit_padding_on_manifest,
    db_to_linear_power,
    linear_power_to_db,
    load_padding_config,
    materialize_padded_dataset,
    pad_profile_db,
)


def _padding_values(result, source_length: int) -> np.ndarray:
    return np.concatenate(
        [
            result.profile_db[: result.left_padding_bins],
            result.profile_db[result.left_padding_bins + source_length :],
        ]
    )


def test_power_db_round_trip() -> None:
    values_db = np.array([-130.0, -60.0, 0.0, 72.0])
    restored = linear_power_to_db(db_to_linear_power(values_db))
    np.testing.assert_allclose(restored, values_db, rtol=0.0, atol=1e-12)
    assert db_to_linear_power(np.array([0.0]))[0] == 1.0


def test_padding_preserves_source_and_uses_fixed_mean_power(
    padding_config: PaddingConfig,
) -> None:
    source = np.array([-40.0, -15.0, 0.0, 10.0, -50.0])
    result = pad_profile_db(source, "sample-a", padding_config)
    assert result.profile_db.shape == (601,)
    assert result.left_padding_bins == result.right_padding_bins == 298
    assert np.array_equal(result.profile_db[298:303], source)
    padding_power = db_to_linear_power(_padding_values(result, source.size))
    assert result.configured_mean_noise_power == pytest.approx(1.0e-14, rel=1e-15)
    assert padding_power.mean() == pytest.approx(
        result.generated_mean_noise_power, rel=2e-15
    )
    assert np.isfinite(result.profile_db).all()


def test_padding_noise_does_not_encode_source_profile_energy(
    padding_config: PaddingConfig,
) -> None:
    source = np.linspace(-100.0, 20.0, 121)
    stronger_source = source + 80.0
    first = pad_profile_db(source, "same-sample-id", padding_config)
    second = pad_profile_db(stronger_source, "same-sample-id", padding_config)
    assert np.array_equal(
        _padding_values(first, source.size),
        _padding_values(second, stronger_source.size),
    )


def test_padding_is_per_sample_deterministic(
    padding_config: PaddingConfig,
) -> None:
    source = np.linspace(-100.0, 20.0, 121)
    first = pad_profile_db(source, "sample-a", padding_config)
    repeated = pad_profile_db(source, "sample-a", padding_config)
    other = pad_profile_db(source, "sample-b", padding_config)
    assert np.array_equal(first.profile_db, repeated.profile_db)
    assert first.derived_seed == repeated.derived_seed
    assert first.derived_seed != other.derived_seed
    assert not np.array_equal(
        _padding_values(first, source.size), _padding_values(other, source.size)
    )


def test_target_length_profile_is_returned_unchanged(
    padding_config: PaddingConfig,
) -> None:
    source = np.linspace(-120.0, 30.0, 601)
    result = pad_profile_db(source, "already-601", padding_config)
    assert np.array_equal(result.profile_db, source)
    assert result.padding_bins == 0
    assert result.generated_mean_noise_power == 0.0


@pytest.mark.parametrize(
    "source,match",
    [
        (np.ones((2, 2)), "one-dimensional"),
        (np.ones(602), "exceeds target"),
        (np.ones(600), "same parity"),
        (np.array([0.0, np.nan, 1.0]), "NaN or Inf"),
    ],
)
def test_padding_rejects_invalid_profiles(
    source: np.ndarray, match: str, padding_config: PaddingConfig
) -> None:
    with pytest.raises(DataValidationError, match=match):
        pad_profile_db(source, "invalid", padding_config)


def test_padding_config_rejects_unfrozen_noise_scale(
    padding_config: PaddingConfig, tmp_path: Path
) -> None:
    changed = copy.deepcopy(dict(padding_config.raw_mapping))
    changed["noise"]["mean_power_db"] = -130.0
    path = tmp_path / "invalid_padding.yaml"
    path.write_text(yaml.safe_dump(changed, allow_unicode=True), encoding="utf-8")
    with pytest.raises(DataConfigError, match="-140 dB"):
        load_padding_config(path)


def test_padding_config_records_explicit_user_risk_acceptance(
    padding_config: PaddingConfig,
) -> None:
    assert padding_config.risk_id == "profile_length_role_shortcut_v1"
    assert padding_config.risk_acceptance_status == "accepted_for_first_round"
    assert padding_config.risk_accepted_by == "user"
    assert padding_config.risk_treatment == "document_and_continue"


def test_synthetic_manifest_padding_audit_passes(
    synthetic_raw_root: Path,
    data_config: DataConfig,
    padding_config: PaddingConfig,
) -> None:
    synthetic_build = build_manifest_rows(data_config, synthetic_raw_root)
    manifest_hash = hashlib.sha256(render_manifest_csv(synthetic_build.rows)).hexdigest()
    synthetic_padding = replace(
        padding_config,
        input_manifest_sha256=manifest_hash,
    )
    report = audit_padding_on_manifest(
        synthetic_build, data_config, synthetic_padding, synthetic_raw_root
    )
    assert report["status"] == "passed"
    assert report["sample_count"] == 3600
    assert report["padded_profile_count"] == 3600
    assert report["preserved_original_profile_count"] == 3600
    assert report["unique_derived_seed_count"] == 3600
    assert report["configured_mean_noise_power_db"] == -140.0
    assert abs(report["observed_mean_padding_power_db"] + 140.0) < 0.02


def test_padding_audit_blocks_disjoint_known_unknown_lengths(
    synthetic_raw_root: Path,
    data_config: DataConfig,
    padding_config: PaddingConfig,
) -> None:
    original = build_manifest_rows(data_config, synthetic_raw_root)
    rows = tuple(
        {
            **row,
            "class_role": "unknown" if int(row["profile_length"]) == 7 else "known",
        }
        for row in original.rows
    )
    build = ManifestBuild(rows=rows, summary=original.summary, validation=original.validation)
    manifest_hash = hashlib.sha256(render_manifest_csv(build.rows)).hexdigest()
    config = replace(
        padding_config,
        input_manifest_sha256=manifest_hash,
        risk_acceptance_status="not_accepted",
    )
    report = audit_padding_on_manifest(build, data_config, config, synthetic_raw_root)
    assert report["status"] == "blocked"
    assert report["experiment_readiness"] == "blocked_by_profile_length_role_shortcut"
    assert report["profile_lengths_by_role"] == {
        "known": [3, 5],
        "unknown": [7],
    }
    assert report["known_unknown_profile_length_overlap"] == []
    assert report["checks"]["known_unknown_profile_length_overlap"] == "failed"


def test_padding_audit_allows_explicitly_accepted_length_risk(
    synthetic_raw_root: Path,
    data_config: DataConfig,
    padding_config: PaddingConfig,
) -> None:
    original = build_manifest_rows(data_config, synthetic_raw_root)
    rows = tuple(
        {
            **row,
            "class_role": "unknown" if int(row["profile_length"]) == 7 else "known",
        }
        for row in original.rows
    )
    build = ManifestBuild(rows=rows, summary=original.summary, validation=original.validation)
    manifest_hash = hashlib.sha256(render_manifest_csv(build.rows)).hexdigest()
    config = replace(padding_config, input_manifest_sha256=manifest_hash)
    report = audit_padding_on_manifest(build, data_config, config, synthetic_raw_root)
    assert report["status"] == "passed"
    assert (
        report["experiment_readiness"]
        == "passed_with_accepted_profile_length_role_risk"
    )
    assert report["checks"]["known_unknown_profile_length_overlap"] == "accepted_risk"
    assert report["accepted_risks"][0]["risk_id"] == config.risk_id


def test_materialized_padding_bundle_is_traceable_and_reproducible(
    synthetic_raw_root: Path,
    data_config: DataConfig,
    padding_config: PaddingConfig,
    tmp_path: Path,
) -> None:
    build = build_manifest_rows(data_config, synthetic_raw_root)
    manifest_hash = hashlib.sha256(render_manifest_csv(build.rows)).hexdigest()
    config = replace(padding_config, input_manifest_sha256=manifest_hash)
    first = materialize_padded_dataset(
        build,
        data_config,
        config,
        synthetic_raw_root,
        tmp_path / "first",
    )
    second = materialize_padded_dataset(
        build,
        data_config,
        config,
        synthetic_raw_root,
        tmp_path / "second",
    )
    assert first["profiles_sha256"] == second["profiles_sha256"]
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["bundle_sha256"] == second["bundle_sha256"]
    profiles = np.load(first["profiles"], allow_pickle=False)
    assert profiles.shape == (3600, 601)
    assert profiles.dtype == np.float64
    assert np.isfinite(profiles).all()
    manifest_lines = Path(first["manifest"]).read_text(encoding="utf-8").splitlines()
    assert len(manifest_lines) == 3601
    assert "processed_profile_sha256" in manifest_lines[0]
    assert "accepted_risk_status" in manifest_lines[0]
