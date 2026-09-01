from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import yaml

from hrrp_osr.data.errors import DataConfigError
from hrrp_osr.models.sets import DeepSetsClassifier, SetTransformerClassifier
from hrrp_osr.training.set_models import load_set_model_config


@pytest.mark.parametrize(("name", "baseline"), [("b2_main_v1.yaml", "B2"), ("b3_main_v1.yaml", "B3")])
def test_set_configs_share_neural_budget(name: str, baseline: str) -> None:
    path = Path(__file__).resolve().parents[1] / "configs/experiments/p1" / name
    config = load_set_model_config(path)
    assert config["baseline"] == baseline
    assert config["training"]["budget_id"] == "neural_budget_v1"
    assert config["training"]["unknown_data_used"] is False
    assert config["model"]["angle_or_position_encoding"] is False


def test_b2_b3_same_seed_initialize_identical_encoders() -> None:
    torch.manual_seed(20260810); b2 = DeepSetsClassifier()
    torch.manual_seed(20260810); b3 = SetTransformerClassifier()
    for left, right in zip(b2.encoder.parameters(), b3.encoder.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_set_config_rejects_unknown_training(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "configs/experiments/p1/b2_main_v1.yaml"
    config = copy.deepcopy(load_set_model_config(source)); config["training"]["unknown_data_used"] = True
    path = tmp_path / "invalid.yaml"; path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    with pytest.raises(DataConfigError, match="budget"):
        load_set_model_config(path)
