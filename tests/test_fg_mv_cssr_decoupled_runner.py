from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from hrrp_osr.data.errors import DataConfigError, DataValidationError
from hrrp_osr.models.cssr_decoupled_1d import (
    D1_DECOUPLED_REL_CSSR,
    D2_DECOUPLED_ABSREL_CSSR,
)
from hrrp_osr.training.fg_mv_cssr_decoupled import (
    EXPERIMENT_ID,
    TASK_SOURCE_FILES,
    _learning_rate_factor,
    _u_statistics,
    load_fg_mv_cssr_decoupled_config,
)
from hrrp_osr.training.fg_mv_cssr_decoupled_protocol import build_phase_plan


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/experiments/cssr/fg_mv_cssr_decoupled_audit_v3.yaml"
)


def test_frozen_config_loads_without_final_test_authorization() -> None:
    config = load_fg_mv_cssr_decoupled_config(CONFIG_PATH)
    assert config["experiment_id"] == EXPERIMENT_ID
    assert config["outputs"]["final_unknown_test_authorized"] is False
    assert config["evidence_scope"]["final_unknown_classes_used"] is False
    assert config["evidence_scope"]["even_angle_test_used"] is False


def test_source_hash_scope_includes_direct_and_transitive_r2_implementation() -> None:
    assert "src/hrrp_osr/models/ms_mean_factorial.py" in TASK_SOURCE_FILES
    assert "src/hrrp_osr/models/mv_rpformer.py" in TASK_SOURCE_FILES
    assert "src/hrrp_osr/training/ms_mean_head_factorial.py" in TASK_SOURCE_FILES
    assert "src/hrrp_osr/training/mv_rpformer.py" in TASK_SOURCE_FILES


def test_config_rejects_any_final_test_authorization(tmp_path: Path) -> None:
    import yaml

    config = load_fg_mv_cssr_decoupled_config(CONFIG_PATH)
    candidate = {
        key: copy.deepcopy(value)
        for key, value in config.items()
        if not key.startswith("_")
    }
    candidate["outputs"]["final_unknown_test_authorized"] = True
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(candidate, allow_unicode=True), encoding="utf-8")
    with pytest.raises(DataConfigError, match="final authorization"):
        load_fg_mv_cssr_decoupled_config(path)


def test_phase_plans_are_exactly_frozen() -> None:
    smoke = build_phase_plan("smoke")
    pilot = build_phase_plan("pilot")
    confirmation = build_phase_plan(
        "confirmation", selected_method=D1_DECOUPLED_REL_CSSR
    )
    assert [(row["pair_id"], row["method"]) for row in smoke] == [
        ("N1", D1_DECOUPLED_REL_CSSR),
        ("N1", D2_DECOUPLED_ABSREL_CSSR),
    ]
    assert len(pilot) == 6
    assert [(row["pair_id"], row["method"]) for row in confirmation] == [
        (pair_id, D1_DECOUPLED_REL_CSSR)
        for pair_id in ("N0", "N3", "N5", "N6")
    ]
    assert all(row["final_unknown_test_authorized"] is False for row in pilot)


def test_confirmation_plan_requires_the_selected_candidate() -> None:
    with pytest.raises(DataValidationError, match="selected"):
        build_phase_plan("confirmation")


def test_frozen_learning_rate_is_positive_through_epoch_20() -> None:
    factors = [
        _learning_rate_factor(epoch, warmup_epochs=2, total_epochs=20)
        for epoch in range(1, 21)
    ]
    assert factors[:3] == [0.5, 1.0, 1.0]
    assert all(value > 0.0 for value in factors)
    assert factors[2:] == sorted(factors[2:], reverse=True)
    with pytest.raises(DataValidationError):
        _learning_rate_factor(21, warmup_epochs=2, total_epochs=20)


def test_u_statistics_detect_noncollapsed_and_constant_features() -> None:
    config = load_fg_mv_cssr_decoupled_config(CONFIG_PATH)
    values = np.zeros((720, 128, 76), dtype=np.float32)
    values[:, 0, 0] = np.linspace(0.0, 1.0, 720, dtype=np.float32)
    result = _u_statistics(values, config=config)
    assert result["collapsed"] is False
    assert result["channel_variance_max"] > 1.0e-12
    assert result["effective_rank"] >= 1.0 - 1.0e-9
    with pytest.raises(DataValidationError, match="collapsed"):
        _u_statistics(np.ones_like(values), config=config)
