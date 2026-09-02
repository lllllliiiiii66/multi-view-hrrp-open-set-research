from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
import pytest

from hrrp_osr.data.processed import ProcessedBundle
from hrrp_osr.training.arpl_pilot import (
    SOURCE_KNOWN_ORDER,
    load_arpl_pilot_config,
    prepare_surrogate_split,
    recompute_metrics_from_prediction_rows,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _synthetic_bundle() -> ProcessedBundle:
    rows = []
    profiles = np.zeros((3600, 601), dtype=np.float64)
    row_index = 0
    for class_index in range(10):
        class_name = f"class-{class_index}"
        class_role = "known" if class_index < 7 else "unknown"
        for angle in range(360):
            profiles[row_index] = (
                class_index
                + angle / 1000.0
                + np.arange(601, dtype=np.float64) / 100000.0
            )
            rows.append(
                {
                    "sample_id": f"{class_name}-{angle}",
                    "class_name": class_name,
                    "class_role": class_role,
                    "angle_deg": angle,
                    "processed_row_index": row_index,
                }
            )
            row_index += 1
    return ProcessedBundle(
        root=Path("/synthetic"),
        profiles=profiles,
        rows=tuple(rows),
        profiles_sha256="a" * 64,
        manifest_sha256="b" * 64,
        bundle_sha256="c" * 64,
    )


def test_arpl_config_freezes_surrogate_only_scope() -> None:
    config = load_arpl_pilot_config(
        PROJECT_ROOT / "configs/experiments/arpl/arpl_lite_surrogate_osr_v1.yaml"
    )
    assert config["classes"]["source_known_order"] == list(SOURCE_KNOWN_ORDER)
    assert config["evidence_scope"]["final_unknown_classes_used"] is False
    assert config["evidence_scope"]["even_angle_test_used"] is False
    assert config["model"]["methods"] == ["CE_MLS", "ARPL_LITE"]


def test_surrogate_protocol_is_deterministic_isolated_and_known_source_only() -> None:
    bundle = _synthetic_bundle()
    kwargs = {
        "source_known_order": [f"class-{index}" for index in range(7)],
        "split_id": "S0",
        "angle_fold": 1,
        "train_known_indices": [2, 3, 4, 5, 6],
        "surrogate_unknown_indices": [0, 1],
        "pairs_per_class": 5,
        "base_seed": 20260830,
    }
    first = prepare_surrogate_split(bundle, **kwargs)
    repeated = prepare_surrogate_split(bundle, **kwargs)
    assert first.pair_manifest_bytes == repeated.pair_manifest_bytes
    assert first.pair_manifest_sha256 == repeated.pair_manifest_sha256
    assert first.train_class_order == tuple(f"class-{index}" for index in range(2, 7))
    assert first.surrogate_class_order == ("class-0", "class-1")
    assert first.inputs["train"].shape == (25, 2, 601)
    assert first.inputs["known_calibration"].shape == (25, 2, 601)
    assert first.inputs["surrogate_unknown"].shape == (10, 2, 601)
    assert first.pair_audit["train_evaluation_base_overlap"] == 0
    assert first.pair_audit["final_unknown_pairs"] == 0
    assert first.pair_audit["even_angle_pairs"] == 0
    assert first.pair_audit["test_pairs_generated"] is False
    rows = list(
        csv.DictReader(io.StringIO(first.pair_manifest_bytes.decode("utf-8")))
    )
    assert not any(row["class_name"] in {"class-7", "class-8", "class-9"} for row in rows)
    assert not any(
        int(row[field]) % 2 == 0
        for row in rows
        for field in ("view1_angle_deg", "view2_angle_deg")
    )
    assert not any(
        row["experiment_role"] == "train_known"
        and row["class_name"] in {"class-0", "class-1"}
        for row in rows
    )


def test_prediction_rows_recompute_all_aggregate_metrics() -> None:
    rows = [
        {
            "evaluation_role": "known_calibration",
            "true_label": label,
            "predicted_known_label": prediction,
            "unknown_score": score,
        }
        for label, prediction, score in (
            (0, 0, 0.1),
            (1, 1, 0.2),
            (0, 1, 0.8),
            (1, 1, 0.3),
        )
    ] + [
        {
            "evaluation_role": "surrogate_unknown",
            "true_label": 2,
            "predicted_known_label": prediction,
            "unknown_score": score,
        }
        for prediction, score in ((0, 0.9), (1, 0.7), (0, 0.6), (1, 0.95))
    ]
    metrics = recompute_metrics_from_prediction_rows(
        rows, known_class_count=2, known_acceptance_rate=0.75
    )
    assert metrics["known_accuracy"] == pytest.approx(0.75)
    assert metrics["threshold"] == pytest.approx(0.3)
    assert metrics["unknown_rejection_rate"] == pytest.approx(1.0)

