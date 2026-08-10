from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from hrrp_osr.data.config import DataConfig, load_data_config
from hrrp_osr.data.errors import DataConfigError


def test_versioned_config_records_raw_manifest_and_disabled_padding(
    data_config: DataConfig,
) -> None:
    preprocessing = data_config.raw_mapping["preprocessing"]
    assert preprocessing["manifest_uses_raw_profiles"] is True
    assert preprocessing["padding"]["enabled"] is False
    assert preprocessing["padding"]["db_linearization"] == "10_log10_power"
    assert preprocessing["padding"]["mean_padding_power_db"] == -140.0
    assert (
        preprocessing["padding"]["experiment_readiness"]
        == "passed_with_accepted_profile_length_role_risk"
    )
    assert data_config.expected.profile_linearization == "10_log10_power"


def test_frozen_protocol_change_is_rejected(
    data_config: DataConfig, tmp_path: Path
) -> None:
    changed = copy.deepcopy(dict(data_config.raw_mapping))
    changed["protocol"]["boundary_buffer_deg"] = 1
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(changed, allow_unicode=True), encoding="utf-8")
    with pytest.raises(DataConfigError, match="boundary buffers"):
        load_data_config(path)


def test_enabling_padding_in_raw_manifest_config_is_rejected(
    data_config: DataConfig, tmp_path: Path
) -> None:
    changed = copy.deepcopy(dict(data_config.raw_mapping))
    changed["preprocessing"]["padding"]["enabled"] = True
    path = tmp_path / "invalid_padding.yaml"
    path.write_text(yaml.safe_dump(changed, allow_unicode=True), encoding="utf-8")
    with pytest.raises(DataConfigError, match="padding must remain disabled"):
        load_data_config(path)
