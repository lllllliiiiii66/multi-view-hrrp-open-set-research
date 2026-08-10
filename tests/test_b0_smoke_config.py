from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

pytest.importorskip("torch")

from hrrp_osr.data.errors import DataConfigError  # noqa: E402
from hrrp_osr.training.b0_smoke import (  # noqa: E402
    EXPECTED_KNOWN_CLASSES,
    load_b0_smoke_config,
)


def _config_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "configs/experiments/p0/b0_smoke_v1.yaml"
    )


def test_b0_smoke_config_freezes_p0_protocol_and_known_order() -> None:
    config = load_b0_smoke_config(_config_path())
    assert config["stage"] == "P0"
    assert config["baseline"] == "B0"
    assert config["run_kind"] == "diagnostic_smoke"
    assert tuple(config["classes"]["known_order"]) == EXPECTED_KNOWN_CLASSES
    assert config["training"]["unknown_data_used"] is False
    assert config["set_protocol"]["view_count"] == 3
    assert config["b0_view_selection"]["average_metrics_not_predictions"] is True


def test_b0_smoke_config_rejects_unknown_training(
    tmp_path: Path,
) -> None:
    config = load_b0_smoke_config(_config_path())
    changed = copy.deepcopy(config)
    changed["training"]["unknown_data_used"] = True
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(changed, allow_unicode=True), encoding="utf-8")
    with pytest.raises(DataConfigError, match="unknown data"):
        load_b0_smoke_config(path)


def test_b0_smoke_config_rejects_first_array_view_selection(
    tmp_path: Path,
) -> None:
    config = load_b0_smoke_config(_config_path())
    changed = copy.deepcopy(config)
    changed["b0_view_selection"]["algorithm_version"] = "array_first"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(changed, allow_unicode=True), encoding="utf-8")
    with pytest.raises(DataConfigError, match="view selection"):
        load_b0_smoke_config(path)
