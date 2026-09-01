from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hrrp_osr.data.errors import DataValidationError
from hrrp_osr.evaluation.aggregate import summarize_seed_values


def aggregate_method_seed_runs(
    run_root: str | Path,
    output_path: str | Path,
    *,
    expected_baseline: str,
    expected_scope: str,
) -> dict[str, Any]:
    root = Path(run_root).resolve()
    run_dirs = sorted(path for path in root.glob("seed_*") if path.is_dir())
    if len(run_dirs) != 5:
        raise DataValidationError(f"expected five seed runs, got {len(run_dirs)}")
    records = []
    for run_dir in run_dirs:
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
        if metrics.get("baseline") != expected_baseline or metrics.get("result_scope") != expected_scope:
            raise DataValidationError(f"unexpected result entered aggregation: {run_dir}")
        records.append((int(metrics["model_seed"]), metrics, environment))
    seeds = [record[0] for record in records]
    expected_seeds = [20260810, 20260820, 20260830, 20260840, 20260850]
    if seeds != expected_seeds:
        raise DataValidationError(f"seed order mismatch: {seeds}")
    code_hashes = {record[2]["code_tree_sha256"] for record in records}
    if len(code_hashes) != 1:
        raise DataValidationError("seed runs do not share one code tree")
    score_names = sorted(records[0][1]["metrics"])
    aggregate = {
        score: {
            metric: summarize_seed_values(
                record[1]["metrics"][score][metric] for record in records
            )
            for metric in sorted(records[0][1]["metrics"][score])
        }
        for score in score_names
    }
    summary = {
        "baseline": expected_baseline,
        "result_scope": expected_scope,
        "seed_registry": seeds,
        "confidence_interval": "two-sided 95% Student-t interval across five model seeds",
        "shared_code_tree_sha256": next(iter(code_hashes)),
        "per_seed": {
            str(seed): metrics["metrics"] for seed, metrics, _ in records
        },
        "aggregate_across_initialization_seeds": aggregate,
        "accepted_risks": records[0][1].get("accepted_risks", []),
    }
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
