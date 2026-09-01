from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from hrrp_osr.data.errors import DataConfigError
from hrrp_osr.training.b1 import _fuse_member_logits, load_b1_config


def _config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs/experiments/p1/b1_main_v1.yaml"


def test_b1_config_freezes_checkpoint_reuse_and_fusion() -> None:
    config = load_b1_config(_config_path())
    assert config["checkpoint_reuse"]["training_allowed"] is False
    assert config["fusion"]["mean_logits_before_energy"] is False
    assert config["set_protocol"]["view_count"] == 3


def test_b1_rejects_mean_logit_energy_drift(tmp_path: Path) -> None:
    config = copy.deepcopy(load_b1_config(_config_path()))
    config["fusion"]["mean_logits_before_energy"] = True
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    with pytest.raises(DataConfigError, match="fusion"):
        load_b1_config(path)


def test_b1_fusion_is_permutation_invariant_and_energy_is_view_score_mean() -> None:
    logits = np.array([[4.0, 0.0], [1.0, 2.0], [0.5, -1.0]])
    reference = _fuse_member_logits(logits, 1.0)
    changed = _fuse_member_logits(logits[[2, 0, 1]], 1.0)
    np.testing.assert_allclose(reference[0], changed[0], atol=1e-15)
    assert reference[1] == pytest.approx(changed[1])
    assert reference[2] == pytest.approx(changed[2])
    mean_logit_energy = -np.log(np.exp(logits.mean(axis=0)).sum())
    assert reference[2] != pytest.approx(mean_logit_energy)
