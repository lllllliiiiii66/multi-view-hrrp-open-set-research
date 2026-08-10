from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from hrrp_osr.evaluation.metrics import evaluate_open_set
from hrrp_osr.evaluation.result_audit import audit_result_run


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_result_audit_recomputes_predictions_and_hashes(tmp_path: Path) -> None:
    rows = [
        {"split": "validation", "class_role": "known", "true_label": "0", "predicted_known_label": "0", "openmax_unknown_score": "0.1"},
        {"split": "validation", "class_role": "known", "true_label": "1", "predicted_known_label": "1", "openmax_unknown_score": "0.2"},
        {"split": "test", "class_role": "known", "true_label": "0", "predicted_known_label": "0", "openmax_unknown_score": "0.1"},
        {"split": "test", "class_role": "known", "true_label": "1", "predicted_known_label": "1", "openmax_unknown_score": "0.2"},
        {"split": "test", "class_role": "unknown", "true_label": "7", "predicted_known_label": "1", "openmax_unknown_score": "0.9"},
    ]
    predictions = tmp_path / "predictions.csv"
    with predictions.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    metrics = evaluate_open_set(
        known_true=np.asarray([0, 1]), known_pred=np.asarray([0, 1]),
        known_unknown_scores=np.asarray([0.1, 0.2]), unknown_pred=np.asarray([1]),
        unknown_unknown_scores=np.asarray([0.9]), known_validation_scores=np.asarray([0.1, 0.2]),
        known_class_count=7, known_acceptance_rate=0.95,
    )
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"metrics": {"openmax": metrics}}), encoding="utf-8")
    manifest_path = tmp_path / "artifact_manifest.json"
    manifest_path.write_text(json.dumps({"artifacts": {
        "predictions.csv": _hash(predictions), "metrics.json": _hash(metrics_path),
    }}), encoding="utf-8")
    audit = audit_result_run(tmp_path)
    assert audit["status"] == "passed"
    assert audit["prediction_row_count"] == 5
    assert audit["maximum_metric_difference"] == 0.0
