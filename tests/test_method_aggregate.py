from __future__ import annotations

import json
from pathlib import Path

from hrrp_osr.evaluation.method_aggregate import aggregate_method_seed_runs


def test_method_aggregate_preserves_seed_hierarchy(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    for index, seed in enumerate([20260810, 20260820, 20260830, 20260840, 20260850]):
        run = run_root / f"seed_{seed}"
        run.mkdir(parents=True)
        (run / "metrics.json").write_text(json.dumps({
            "baseline": "B1", "result_scope": "main_v3", "model_seed": seed,
            "metrics": {"msp": {"known_accuracy": 0.7 + 0.01 * index}},
            "accepted_risks": [],
        }), encoding="utf-8")
        (run / "environment.json").write_text(
            json.dumps({"code_tree_sha256": "same"}), encoding="utf-8"
        )
    output = tmp_path / "summary.json"
    summary = aggregate_method_seed_runs(
        run_root, output, expected_baseline="B1", expected_scope="main_v3"
    )
    assert summary["seed_registry"] == [20260810, 20260820, 20260830, 20260840, 20260850]
    assert summary["aggregate_across_initialization_seeds"]["msp"]["known_accuracy"]["mean"] == 0.72
    assert output.exists()
