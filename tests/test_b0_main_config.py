from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

pytest.importorskip("torch")

from hrrp_osr.data.errors import DataConfigError  # noqa: E402
from hrrp_osr.training.b0_smoke import load_b0_main_config  # noqa: E402


def _config_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "configs/experiments/p0/b0_main_v1_seed20260810.yaml"
    )


def test_b0_main_config_freezes_budget_seed_registry_and_main_scope() -> None:
    config = load_b0_main_config(_config_path())
    assert config["result_scope"] == "main_v3"
    assert config["training"]["budget_id"] == "neural_budget_v1"
    assert config["training"]["epochs"] == 100
    assert config["training"]["early_stopping_patience"] == 15
    assert config["b0_view_selection"]["repeats"] == 30
    assert config["training"]["unknown_data_used"] is False


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("training", "unknown_data_used", True, "unknown data"),
        ("training", "epochs", 99, "100 epochs"),
        ("b0_view_selection", "repeats", 29, "30 view-selection"),
        ("artifact_policy", "fail_if_output_nonempty", False, "refuse"),
    ],
)
def test_b0_main_config_rejects_protocol_drift(
    tmp_path: Path,
    section: str,
    key: str,
    value: object,
    message: str,
) -> None:
    changed = copy.deepcopy(load_b0_main_config(_config_path()))
    changed[section][key] = value
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(changed, allow_unicode=True), encoding="utf-8")
    with pytest.raises(DataConfigError, match=message):
        load_b0_main_config(path)


def test_b0_main_config_rejects_seed_registry_mismatch(tmp_path: Path) -> None:
    changed = copy.deepcopy(load_b0_main_config(_config_path()))
    changed["training"]["active_seed_index"] = 1
    path = tmp_path / "invalid_seed.yaml"
    path.write_text(yaml.safe_dump(changed, allow_unicode=True), encoding="utf-8")
    with pytest.raises(DataConfigError, match="does not match"):
        load_b0_main_config(path)


@pytest.mark.parametrize(
    ("suffix", "seed_index", "initialization_seed", "dataloader_seed"),
    [
        ("20260820", 1, 20260820, 20260821),
        ("20260830", 2, 20260830, 20260831),
        ("20260840", 3, 20260840, 20260841),
        ("20260850", 4, 20260850, 20260851),
    ],
)
def test_b0_main_seed_overlays_only_activate_registered_seed(
    suffix: str,
    seed_index: int,
    initialization_seed: int,
    dataloader_seed: int,
) -> None:
    path = _config_path().with_name(f"b0_main_v1_seed{suffix}.yaml")
    config = load_b0_main_config(path)
    assert config["training"]["active_seed_index"] == seed_index
    assert config["training"]["initialization_seed"] == initialization_seed
    assert config["training"]["dataloader_seed"] == dataloader_seed
    assert config["training"]["budget_id"] == "neural_budget_v1"
    assert config["b0_view_selection"]["repeats"] == 30
