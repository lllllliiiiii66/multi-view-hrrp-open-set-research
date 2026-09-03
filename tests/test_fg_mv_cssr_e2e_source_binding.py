from __future__ import annotations

from pathlib import Path

from hrrp_osr.training.fg_mv_cssr_e2e_redesign import (
    TASK_SOURCE_FILES,
    task_source_hashes,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CRITICAL_TRANSITIVE_SOURCES = {
    "src/hrrp_osr/amdr/data.py",
    "src/hrrp_osr/amdr/model.py",
    "src/hrrp_osr/amdr/reduction.py",
    "src/hrrp_osr/amdr/smoke.py",
    "src/hrrp_osr/data/config.py",
    "src/hrrp_osr/data/errors.py",
    "src/hrrp_osr/data/protocol.py",
    "src/hrrp_osr/evaluation/ms_mean_factorial.py",
    "src/hrrp_osr/models/arpl.py",
    "src/hrrp_osr/models/cnn1d.py",
    "src/hrrp_osr/models/mv_rpformer.py",
    "src/hrrp_osr/training/arpl_mv_evidence.py",
    "src/hrrp_osr/training/mv_rpformer.py",
}


def test_task_source_binding_covers_r2_training_and_evaluation_dependencies() -> None:
    assert CRITICAL_TRANSITIVE_SOURCES <= set(TASK_SOURCE_FILES)
    hashes = task_source_hashes(PROJECT_ROOT)
    assert set(hashes) == set(TASK_SOURCE_FILES)
    assert all(len(value) == 64 for value in hashes.values())
