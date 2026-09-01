from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from scipy.stats import t as student_t

from hrrp_osr.data.errors import DataValidationError


def summarize_seed_values(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.isfinite(array).all():
        raise DataValidationError(
            "seed aggregation requires at least two finite scalar values"
        )
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1))
    half_width = float(
        student_t.ppf(0.975, df=array.size - 1) * std / math.sqrt(array.size)
    )
    return {
        "seed_count": int(array.size),
        "mean": mean,
        "sample_std": std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def aggregate_b0_main_runs(run_root: str | Path, output_path: str | Path) -> dict[str, Any]:
    root = Path(run_root).resolve()
    run_dirs = sorted(path for path in root.glob("seed_*") if path.is_dir())
    if len(run_dirs) != 5:
        raise DataValidationError(f"expected exactly five B0 seed directories, got {len(run_dirs)}")

    run_records: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
        environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
        timing = json.loads((run_dir / "inference_timing.json").read_text(encoding="utf-8"))
        if metrics.get("run_kind") != "main" or metrics.get("result_scope") != "main_v3":
            raise DataValidationError(f"non-main result entered seed aggregation: {run_dir}")
        seed = int(config["training"]["initialization_seed"])
        run_records.append(
            {
                "run_dir": str(run_dir),
                "seed": seed,
                "metrics": metrics,
                "config": config,
                "environment": environment,
                "timing": timing,
            }
        )

    seeds = [record["seed"] for record in run_records]
    registered = list(run_records[0]["config"]["training"]["planned_initialization_seeds"])
    if seeds != registered:
        raise DataValidationError(
            f"materialized seeds {seeds} do not match registered seed order {registered}"
        )
    code_hashes = {record["environment"]["code_tree_sha256"] for record in run_records}
    bundle_hashes = {
        record["config"]["data"]["bundle_sha256"] for record in run_records
    }
    set_hashes = {
        record["config"]["_resolved"]["set_manifest_sha256"] for record in run_records
    }
    budget_ids = {record["config"]["training"]["budget_id"] for record in run_records}
    if any(len(values) != 1 for values in (code_hashes, bundle_hashes, set_hashes, budget_ids)):
        raise DataValidationError("B0 seed runs do not share code, data, sets, and budget")

    scores = ("msp", "energy")
    aggregate: dict[str, dict[str, dict[str, float | int]]] = {}
    per_seed: dict[str, Any] = {}
    for score in scores:
        metric_names = sorted(
            run_records[0]["metrics"]["aggregate_across_selection_repeats"][score]
        )
        aggregate[score] = {
            metric_name: summarize_seed_values(
                record["metrics"]["aggregate_across_selection_repeats"][score][
                    metric_name
                ]["mean"]
                for record in run_records
            )
            for metric_name in metric_names
        }
    for record in run_records:
        per_seed[str(record["seed"])] = {
            "best_epoch": record["metrics"]["best_epoch"],
            "best_known_validation_accuracy": record["metrics"][
                "best_known_validation_accuracy"
            ],
            "stopped_epoch": record["metrics"]["stopped_epoch"],
            "metrics_after_selection_repeat_mean": {
                score: {
                    metric: values["mean"]
                    for metric, values in record["metrics"][
                        "aggregate_across_selection_repeats"
                    ][score].items()
                }
                for score in scores
            },
        }

    summary = {
        "stage": "P0",
        "baseline": "B0",
        "result_scope": "main_v3",
        "seed_registry": registered,
        "aggregation_hierarchy": (
            "first mean metrics across 30 fixed-seed B0 view selections within each "
            "model seed, then summarize five model-seed means"
        ),
        "confidence_interval": "two-sided 95% Student-t interval across five model seeds",
        "shared_code_tree_sha256": next(iter(code_hashes)),
        "shared_data_bundle_sha256": next(iter(bundle_hashes)),
        "shared_set_manifest_sha256": next(iter(set_hashes)),
        "shared_training_budget_id": next(iter(budget_ids)),
        "per_seed": per_seed,
        "aggregate_across_initialization_seeds": aggregate,
        "single_profile_forward_latency_seconds": summarize_seed_values(
            record["timing"]["mean_seconds_per_profile"] for record in run_records
        ),
        "accepted_risks": run_records[0]["metrics"]["accepted_risks"],
    }
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
