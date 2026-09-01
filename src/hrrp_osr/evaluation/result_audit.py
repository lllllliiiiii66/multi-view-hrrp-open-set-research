from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hrrp_osr.data.errors import DataValidationError
from hrrp_osr.evaluation.metrics import evaluate_open_set, macro_f1_score


SCORE_COLUMNS = {
    "msp": "msp_unknown_score",
    "energy": "energy_unknown_score",
    "openmax": "openmax_unknown_score",
    "dual_tail_gpd": "dual_tail_unknown_score",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _recompute(rows: Sequence[Mapping[str, str]], score_column: str) -> dict[str, float]:
    validation = [row for row in rows if row["split"] == "validation"]
    test = [row for row in rows if row["split"] == "test"]
    known = [row for row in test if row["class_role"] == "known"]
    unknown = [row for row in test if row["class_role"] == "unknown"]
    if not validation or not known or not unknown:
        raise DataValidationError("result audit needs known validation and known/unknown test rows")
    return evaluate_open_set(
        known_true=np.asarray([int(row["true_label"]) for row in known]),
        known_pred=np.asarray([int(row["predicted_known_label"]) for row in known]),
        known_unknown_scores=np.asarray([float(row[score_column]) for row in known]),
        unknown_pred=np.asarray([int(row["predicted_known_label"]) for row in unknown]),
        unknown_unknown_scores=np.asarray([float(row[score_column]) for row in unknown]),
        known_validation_scores=np.asarray([float(row[score_column]) for row in validation]),
        known_class_count=7,
        known_acceptance_rate=0.95,
    )


def _maximum_metric_difference(actual: Mapping[str, float], expected: Mapping[str, float]) -> float:
    if set(actual) != set(expected):
        raise DataValidationError("recomputed and saved metric keys differ")
    return max(abs(float(actual[key]) - float(expected[key])) for key in actual)


def audit_result_run(run_directory: str | Path, tolerance: float = 1.0e-12) -> dict[str, Any]:
    root = Path(run_directory).resolve()
    metrics_document = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    with (root / "predictions.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    available_methods = [method for method, column in SCORE_COLUMNS.items() if column in rows[0]]
    if not available_methods:
        raise DataValidationError(f"no recognized score column in {root}")
    metric_differences: dict[str, float] = {}
    if "selection_repeat" in rows[0]:
        repeat_ids = sorted({int(row["selection_repeat"]) for row in rows})
        for method in available_methods:
            expected_repeats = metrics_document["per_selection_repeat"][method]
            if len(expected_repeats) != len(repeat_ids):
                raise DataValidationError("B0 saved repeat count differs from predictions")
            for position, repeat_id in enumerate(repeat_ids):
                selected = [row for row in rows if int(row["selection_repeat"]) == repeat_id]
                actual = _recompute(selected, SCORE_COLUMNS[method])
                difference = _maximum_metric_difference(actual, expected_repeats[position])
                metric_differences[f"{method}/repeat_{repeat_id}"] = difference
    else:
        saved = metrics_document["metrics"]
        for method in available_methods:
            actual = _recompute(rows, SCORE_COLUMNS[method])
            difference = _maximum_metric_difference(actual, saved[method])
            metric_differences[method] = difference
    if max(metric_differences.values()) > tolerance:
        raise DataValidationError(
            f"saved metrics cannot be reproduced from predictions in {root}: {max(metric_differences.values())}"
        )
    fixed_delta_difference = None
    if "paper_fixed_delta_operating_point" in metrics_document:
        fixed = metrics_document["paper_fixed_delta_operating_point"]
        threshold = float(fixed["threshold"])
        test = [row for row in rows if row["split"] == "test"]
        known = [row for row in test if row["class_role"] == "known"]
        unknown = [row for row in test if row["class_role"] == "unknown"]
        known_scores = np.asarray([float(row["dual_tail_unknown_score"]) for row in known])
        unknown_scores = np.asarray([float(row["dual_tail_unknown_score"]) for row in unknown])
        known_pred = np.asarray([int(row["predicted_known_label"]) for row in known])
        unknown_pred = np.asarray([int(row["predicted_known_label"]) for row in unknown])
        unknown_label = 7
        true = np.concatenate([
            np.asarray([int(row["true_label"]) for row in known]),
            np.full(len(unknown), unknown_label, dtype=int),
        ])
        predicted = np.concatenate([
            np.where(known_scores > threshold, unknown_label, known_pred),
            np.where(unknown_scores > threshold, unknown_label, unknown_pred),
        ])
        recomputed_fixed = {
            "known_acceptance_rate": float(np.mean(known_scores <= threshold)),
            "unknown_rejection_rate": float(np.mean(unknown_scores > threshold)),
            "k_plus_1_macro_f1": macro_f1_score(true, predicted, labels=range(8)),
        }
        fixed_delta_difference = max(
            abs(recomputed_fixed[key] - float(fixed[key])) for key in recomputed_fixed
        )
        if fixed_delta_difference > tolerance:
            raise DataValidationError("paper fixed-delta metrics cannot be reproduced")
    artifact_document = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    hash_mismatches = []
    for name, expected_hash in artifact_document["artifacts"].items():
        if _sha256(root / name) != expected_hash:
            hash_mismatches.append(name)
    if hash_mismatches:
        raise DataValidationError(f"artifact hash mismatches in {root}: {hash_mismatches}")
    return {
        "status": "passed", "run_directory": str(root),
        "prediction_row_count": len(rows), "score_methods": available_methods,
        "maximum_metric_difference": max(metric_differences.values()),
        "fixed_delta_maximum_difference": fixed_delta_difference,
        "artifact_hash_count": len(artifact_document["artifacts"]),
    }


def audit_result_tree(
    roots: Sequence[str | Path],
    output: str | Path,
    tolerance: float = 1.0e-12,
) -> dict[str, Any]:
    run_directories = sorted({
        metrics_path.parent
        for root in roots
        for metrics_path in Path(root).resolve().rglob("metrics.json")
        if (metrics_path.parent / "predictions.csv").is_file()
        and (metrics_path.parent / "artifact_manifest.json").is_file()
    })
    if not run_directories:
        raise DataValidationError("result audit found no run directories")
    audits = [audit_result_run(path, tolerance=tolerance) for path in run_directories]
    document = {
        "status": "passed", "run_count": len(audits),
        "total_prediction_rows": sum(item["prediction_row_count"] for item in audits),
        "total_hashed_artifacts": sum(item["artifact_hash_count"] for item in audits),
        "maximum_metric_difference": max(item["maximum_metric_difference"] for item in audits),
        "runs": audits,
    }
    destination = Path(output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document
