from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hrrp_osr.data.errors import DataValidationError
from hrrp_osr.data.manifest import file_sha256
from hrrp_osr.models.cssr_decoupled_1d import D1_DECOUPLED_REL_CSSR
from hrrp_osr.training.fg_mv_cssr_decoupled import (
    D0_R2_CLASS_CONDITIONAL_MLS,
    _audit_frozen_r2_evaluation_binding,
    _rebuild_evaluation_manifest,
)
from hrrp_osr.training.fg_mv_cssr_pilot import _read_csv, _render_csv


PAIR_ID = "N1"
RELATIVE_R2_UNIT = Path(PAIR_ID) / "fold_0" / "seed_20260830" / "R2_MS_MEAN_CE"
ARRAY_NAMES = (
    "per_view_features",
    "fused_features",
    "per_view_logits",
    "global_logits",
    "unknown_score",
    "labels",
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _source_row(
    *, role: str, class_name: str, model_label: int, row_index: int
) -> dict[str, Any]:
    angle1 = 2 * row_index + 1
    angle2 = 2 * row_index + 3
    return {
        "pair_id": f"pair-{role}-{class_name}",
        "split": "train" if role == "train_known" else "calibration",
        "fold_index": 0,
        "class_name": class_name,
        "class_role": "known",
        "view1_sample_id": f"sample-{role}-{class_name}-1",
        "view2_sample_id": f"sample-{role}-{class_name}-2",
        "view1_row_index": row_index * 2,
        "view2_row_index": row_index * 2 + 1,
        "view1_frame_id": angle1 // 15,
        "view2_frame_id": angle2 // 15,
        "view1_angle_deg": angle1,
        "view2_angle_deg": angle2,
        "algorithm_version": "fixture",
        "experiment_role": role,
        "surrogate_split_id": "N1_F0",
        "model_label": model_label,
    }


def _role_arrays(count: int, labels: np.ndarray, offset: float) -> dict[str, np.ndarray]:
    logits = (
        np.arange(count * 5, dtype=np.float64).reshape(count, 5) / 10.0 + offset
    )
    features = (
        np.arange(count * 2 * 3, dtype=np.float32).reshape(count, 2, 3) + offset
    )
    return {
        "per_view_features": features,
        "fused_features": features.mean(axis=1).astype(np.float32),
        "per_view_logits": np.repeat(logits[:, None, :], 2, axis=1),
        "global_logits": logits,
        "unknown_score": -logits.max(axis=1),
        "labels": np.asarray(labels, dtype=np.int64),
    }


def _fixture(tmp_path: Path) -> dict[str, Any]:
    source_order = tuple(f"C{index}" for index in range(7))
    train_order = tuple(source_order[index] for index in (0, 1, 3, 4, 6))
    surrogate_order = tuple(source_order[index] for index in (2, 5))
    source_rows = [
        *[
            _source_row(
                role="train_known",
                class_name=class_name,
                model_label=label,
                row_index=label,
            )
            for label, class_name in enumerate(train_order)
        ],
        *[
            _source_row(
                role="known_calibration",
                class_name=class_name,
                model_label=label,
                row_index=10 + label,
            )
            for label, class_name in enumerate(train_order)
        ],
        *[
            _source_row(
                role="surrogate_unknown",
                class_name=class_name,
                model_label=5,
                row_index=20 + label,
            )
            for label, class_name in enumerate(surrogate_order)
        ],
    ]

    results_root = tmp_path / "prior-r2"
    prior_unit = results_root / RELATIVE_R2_UNIT
    prior_unit.mkdir(parents=True)
    prior_manifest = prior_unit / "pair_manifest.csv"
    prior_manifest.write_bytes(_render_csv(source_rows))
    checkpoint = prior_unit / "checkpoint.pt"
    checkpoint.write_bytes(b"fixed-prior-r2-checkpoint")
    arrays_by_role = {
        "train": _role_arrays(5, np.arange(5), 0.0),
        "known_calibration": _role_arrays(5, np.arange(5), 10.0),
        "surrogate_unknown": _role_arrays(2, np.full(2, 5), 20.0),
    }
    output_path = prior_unit / "features_logits_scores.npz"
    np.savez_compressed(
        output_path,
        **{
            f"{role}_{name}": value
            for role, values in arrays_by_role.items()
            for name, value in values.items()
        },
    )
    expected_hashes = {
        "checkpoint.pt": file_sha256(checkpoint),
        "pair_manifest.csv": file_sha256(prior_manifest),
        "features_logits_scores.npz": file_sha256(output_path),
    }
    unit_hashes = dict(expected_hashes)
    _write_json(prior_unit / "artifact_hashes.json", unit_hashes)
    root_hashes = {
        str(RELATIVE_R2_UNIT / name): digest
        for name, digest in expected_hashes.items()
    }
    _write_json(results_root / "artifact_hashes.json", root_hashes)

    config = {
        "prior_r2": {
            "formal_code_commit": "frozen-r2-commit",
            "unit_relative_template": "{pair_id}/fold_0/seed_20260830/R2_MS_MEAN_CE",
            "root_artifact_hash_manifest_sha256": file_sha256(
                results_root / "artifact_hashes.json"
            ),
            "unit_artifact_hashes": {PAIR_ID: expected_hashes},
        },
        "classes": {
            "source_known_order": list(source_order),
            "identity_pairs": [
                {
                    "pair_id": PAIR_ID,
                    "surrogate_unknown_indices": [2, 5],
                    "train_known_indices": [0, 1, 3, 4, 6],
                }
            ],
        },
        "data": {
            "evaluation_pairs_per_class": 1,
            "smoke": {"evaluation_pairs_per_class": 1},
        },
    }
    exact_keys = {
        f"{role}_{name}"
        for role in ("train", "known_calibration", "surrogate_unknown")
        for name in ARRAY_NAMES
    }
    r2_audit = {
        "status": "passed",
        "pair_id": PAIR_ID,
        "unit_root": str(prior_unit),
        "prior_formal_code_commit": "frozen-r2-commit",
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": expected_hashes["checkpoint.pt"],
        "pair_manifest_sha256": expected_hashes["pair_manifest.csv"],
        "artifact_hash_manifest_sha256": file_sha256(
            prior_unit / "artifact_hashes.json"
        ),
        "root_artifact_hash_manifest_sha256": config["prior_r2"][
            "root_artifact_hash_manifest_sha256"
        ],
        "expected_unit_artifact_hashes": expected_hashes,
        "artifact_count": len(unit_hashes),
        "strict_load": True,
        "all_parameters_frozen": True,
        "arpl_module_instantiated": False,
        "old_outputs_exact": True,
        "old_output_exact_checks": {key: True for key in exact_keys},
        "old_output_maximum_absolute_errors": {key: 0.0 for key in exact_keys},
        "train_class_order": list(train_order),
        "surrogate_class_order": list(surrogate_order),
        "final_unknown_used": False,
        "even_angle_test_used": False,
    }

    unit_root = tmp_path / "decoupled-unit"
    unit_root.mkdir()
    (unit_root / "source_pair_manifest.csv").write_bytes(prior_manifest.read_bytes())
    parsed_source = _read_csv(prior_manifest)
    evaluation_rows, indices = _rebuild_evaluation_manifest(
        parsed_source,
        config=config,
        phase="smoke",
        pair_id=PAIR_ID,
    )
    (unit_root / "evaluation_pair_manifest.csv").write_bytes(
        _render_csv(evaluation_rows)
    )
    _write_json(unit_root / "r2_reference_audit.json", r2_audit)

    method_rows = []
    d0_rows = []
    for evaluation_row in evaluation_rows:
        role = str(evaluation_row["evaluation_subset_role"])
        index = int(evaluation_row["evaluation_subset_index"])
        source_index = int(indices[role][index])
        logits = arrays_by_role[role]["global_logits"][source_index]
        common = {
            "pair_id": evaluation_row["pair_id"],
            "evaluation_role": role,
            "class_name": evaluation_row["class_name"],
            "true_label": int(evaluation_row["model_label"]),
            "predicted_known_label": int(logits.argmax()),
            "fused_logits": json.dumps(logits.tolist(), separators=(",", ":")),
            "view1_sample_id": evaluation_row["view1_sample_id"],
            "view2_sample_id": evaluation_row["view2_sample_id"],
            "view1_angle_deg": int(evaluation_row["view1_angle_deg"]),
            "view2_angle_deg": int(evaluation_row["view2_angle_deg"]),
            "view1_frame_id": int(evaluation_row["view1_frame_id"]),
            "view2_frame_id": int(evaluation_row["view2_frame_id"]),
        }
        method_rows.append({"method": D1_DECOUPLED_REL_CSSR, **common})
        d0_rows.append({"method": D0_R2_CLASS_CONDITIONAL_MLS, **common})
    reference_arrays = {
        "full_calibration_logits": arrays_by_role["known_calibration"][
            "global_logits"
        ],
        "full_calibration_labels": arrays_by_role["known_calibration"]["labels"],
        "full_calibration_pair_ids": np.asarray(
            [
                row["pair_id"]
                for row in parsed_source
                if row["experiment_role"] == "known_calibration"
            ],
            dtype=np.str_,
        ),
    }
    return {
        "root": unit_root,
        "config": config,
        "method_rows": method_rows,
        "d0_rows": d0_rows,
        "reference_arrays": reference_arrays,
        "prior_manifest": prior_manifest,
        "prior_outputs": output_path,
    }


def _audit(fixture: dict[str, Any]) -> dict[str, Any]:
    return _audit_frozen_r2_evaluation_binding(
        fixture["root"],
        config=fixture["config"],
        phase="smoke",
        pair_id=PAIR_ID,
        method=D1_DECOUPLED_REL_CSSR,
        method_rows=fixture["method_rows"],
        d0_rows=fixture["d0_rows"],
        reference_arrays=fixture["reference_arrays"],
    )


def test_clean_artifacts_bind_to_frozen_r2_outputs(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result = _audit(fixture)
    assert result["audit"]["status"] == "passed"
    assert result["audit"]["prediction_rows_bound_to_prior_r2"] is True


def test_tampered_evaluation_manifest_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    path = fixture["root"] / "evaluation_pair_manifest.csv"
    rows = _read_csv(path)
    rows[0]["view1_sample_id"] = "tampered-sample"
    path.write_bytes(_render_csv(rows))
    with pytest.raises(DataValidationError, match="evaluation pair manifest"):
        _audit(fixture)


def test_tampered_r2_reference_audit_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    path = fixture["root"] / "r2_reference_audit.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["old_outputs_exact"] = False
    _write_json(path, value)
    with pytest.raises(DataValidationError, match="reference audit"):
        _audit(fixture)


def test_tampered_prior_pair_manifest_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["prior_manifest"].write_bytes(
        fixture["prior_manifest"].read_bytes() + b"tampered\n"
    )
    with pytest.raises(DataValidationError, match="pair_manifest.csv"):
        _audit(fixture)


def test_tampered_prior_features_logits_scores_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with np.load(fixture["prior_outputs"], allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    arrays["known_calibration_global_logits"] = arrays[
        "known_calibration_global_logits"
    ].copy()
    arrays["known_calibration_global_logits"][0, 0] += 1.0
    np.savez_compressed(fixture["prior_outputs"], **arrays)
    with pytest.raises(DataValidationError, match="features_logits_scores.npz"):
        _audit(fixture)


def test_self_consistent_prediction_logits_cannot_replace_frozen_r2(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    replacement = json.dumps([99.0, 0.0, 0.0, 0.0, 0.0], separators=(",", ":"))
    for rows in (fixture["method_rows"], fixture["d0_rows"]):
        rows[0]["fused_logits"] = replacement
        rows[0]["predicted_known_label"] = 0
    with pytest.raises(DataValidationError, match="not the frozen R2 outputs"):
        _audit(fixture)
