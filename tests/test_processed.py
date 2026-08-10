from __future__ import annotations

import hashlib
from dataclasses import replace

from hrrp_osr.data.config import DataConfig
from hrrp_osr.data.manifest import build_manifest_rows, render_manifest_csv
from hrrp_osr.data.padding import PaddingConfig, materialize_padded_dataset
from hrrp_osr.data.processed import load_processed_bundle


def test_processed_bundle_loader_verifies_hashes_and_isolation(
    data_config: DataConfig,
    padding_config: PaddingConfig,
    synthetic_raw_root,
    tmp_path,
) -> None:
    build = build_manifest_rows(data_config, synthetic_raw_root)
    input_hash = hashlib.sha256(render_manifest_csv(build.rows)).hexdigest()
    config = replace(padding_config, input_manifest_sha256=input_hash)
    result = materialize_padded_dataset(
        build,
        data_config,
        config,
        synthetic_raw_root,
        tmp_path / "processed",
    )
    bundle = load_processed_bundle(
        result["output_dir"],
        expected_profiles_sha256=result["profiles_sha256"],
        expected_manifest_sha256=result["manifest_sha256"],
        expected_bundle_sha256=result["bundle_sha256"],
    )
    assert bundle.profiles.shape == (3600, 601)
    assert len(bundle.rows) == 3600
    assert len(bundle.known_classes) == 7
    assert all(
        not row["eligible_for_training"] and not row["eligible_for_validation"]
        for row in bundle.rows
        if row["class_role"] == "unknown"
    )
