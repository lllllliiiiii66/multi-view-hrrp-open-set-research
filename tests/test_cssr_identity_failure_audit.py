from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from hrrp_osr.data.errors import DataValidationError  # noqa: E402
from hrrp_osr.data.manifest import file_sha256  # noqa: E402
from hrrp_osr.models.cssr_decoupled_1d import (  # noqa: E402
    D1_DECOUPLED_REL_CSSR,
    FGMVCSSRDecoupled1D,
)
from hrrp_osr.training import cssr_identity_failure_audit as audit  # noqa: E402


def _base_row(
    role: str,
    sample_id: str,
    class_name: str,
    model_label: int,
    index: int,
) -> dict[str, Any]:
    return {
        "experiment_role": role,
        "sample_id": sample_id,
        "processed_row_index": index,
        "class_name": class_name,
        "model_label": model_label,
        "angle_deg": 2 * index + 1,
        "frame_id": index % 24,
        "source_class_role": "known" if model_label >= 0 else "surrogate_unknown",
    }


def _small_unique_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in range(5):
        rows.append(
            _base_row(
                "train_known",
                f"train-{label}",
                f"known-{label}",
                label,
                len(rows),
            )
        )
    for label in range(5):
        for local_index in range(2):
            rows.append(
                _base_row(
                    "known_calibration",
                    f"cal-{label}-{local_index}",
                    f"known-{label}",
                    label,
                    len(rows),
                )
            )
    for identity in ("surrogate-a", "surrogate-b"):
        for local_index in range(2):
            rows.append(
                _base_row(
                    "surrogate_unknown",
                    f"{identity}-{local_index}",
                    identity,
                    5,
                    len(rows),
                )
            )
    return rows


def _array_decomposition_fixture() -> tuple[list[dict[str, Any]], audit.ScoreDecomposition]:
    rows = _small_unique_rows()
    count, classes, channels, length = len(rows), 5, 2, 3
    adapted = np.linspace(0.5, 2.5, count * channels * length).reshape(
        count, channels, length
    )
    errors = (
        np.arange(count * classes, dtype=np.float64).reshape(count, classes) / 50.0
        + 0.1
    )
    reconstructions = adapted[:, None, :, :] + errors[:, :, None, None]
    logits = -errors[:, :, None] * np.ones((1, 1, length), dtype=np.float64)
    probabilities = np.exp(logits.mean(axis=2))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    decomposition = audit.decompose_scores_from_arrays(
        adapted,
        reconstructions,
        logits,
        probabilities,
        rows,
    )
    return rows, decomposition


def test_score_decomposition_matches_frozen_e_m_r_definitions() -> None:
    rows, decomposition = _array_decomposition_fixture()
    assert decomposition.raw_error.shape == (len(rows), 5)
    expected_m = np.abs(decomposition.adapted_features).mean(axis=(1, 2))
    expected_r = decomposition.raw_error / (expected_m[:, None] + 1.0e-8)
    np.testing.assert_allclose(decomposition.activation_magnitude, expected_m)
    np.testing.assert_allclose(decomposition.normalized_error, expected_r)
    np.testing.assert_allclose(
        decomposition.anomaly,
        -np.log(decomposition.p_value + 1.0e-8),
    )


def test_conformal_uses_true_class_loo_and_greater_equal_ties() -> None:
    rows: list[dict[str, Any]] = []
    values: list[np.ndarray] = []
    for label in range(5):
        for local_index, diagonal in enumerate((1.0, 2.0)):
            rows.append(
                {
                    "experiment_role": "known_calibration",
                    "sample_id": f"cal-{label}-{local_index}",
                    "model_label": label,
                }
            )
            row = np.full(5, 4.0, dtype=np.float64)
            row[label] = diagonal
            values.append(row)
    rows.append(
        {
            "experiment_role": "train_known",
            "sample_id": "train-query",
            "model_label": 0,
        }
    )
    values.append(np.full(5, 2.0, dtype=np.float64))
    p_value, _, references, reference_ids = audit.conformal_p_and_anomaly(
        np.asarray(values), rows
    )

    # The second class-0 calibration value is removed from its own true-class
    # reference: only [1] remains, so (1 + 0) / (1 + 1) = 0.5.
    assert p_value[1, 0] == pytest.approx(0.5)
    # The train query is not LOO and the >= tie at 2 is counted: [1, 2].
    assert p_value[-1, 0] == pytest.approx((1.0 + 1.0) / 3.0)
    assert [len(value) for value in references] == [2] * 5
    assert reference_ids[0] == ("cal-0-0", "cal-0-1")


def test_conformal_rejects_repeated_unique_base_id() -> None:
    rows, decomposition = _array_decomposition_fixture()
    rows[-1]["sample_id"] = rows[-2]["sample_id"]
    with pytest.raises(DataValidationError, match="repeats a sample ID"):
        audit.conformal_p_and_anomaly(decomposition.normalized_error, rows)


def test_score_normalization_augmentation_is_deterministic_and_input_level() -> None:
    normalized = np.stack(
        (np.linspace(-1.0, 1.0, 601), np.linspace(1.0, -1.0, 601))
    )
    first, first_audit = audit.build_score_normalization_augmentations(
        normalized, ("sample-a", "sample-b"), pair_id="N1"
    )
    repeated, repeated_audit = audit.build_score_normalization_augmentations(
        normalized, ("sample-a", "sample-b"), pair_id="N1"
    )
    other_pair, _ = audit.build_score_normalization_augmentations(
        normalized, ("sample-a", "sample-b"), pair_id="N4"
    )
    assert first.shape == (8, 601)
    assert first.dtype == np.float32
    assert np.array_equal(first, repeated)
    assert first_audit == repeated_audit
    assert not np.array_equal(first, other_pair)
    assert first_audit["method_id_in_seed_material"] is False
    assert first_audit["final_unknown_used"] is False


def test_output_path_must_be_disjoint_from_every_immutable_input(tmp_path: Path) -> None:
    immutable = tmp_path / "sealed"
    immutable.mkdir()
    with pytest.raises(DataValidationError, match="must be disjoint"):
        audit.require_disjoint_output((immutable,), immutable / "new-results")
    with pytest.raises(DataValidationError, match="must be disjoint"):
        audit.require_disjoint_output((immutable,), tmp_path)
    assert audit.require_disjoint_output(
        (immutable,), tmp_path / "independent" / "stage_a"
    ) == (tmp_path / "independent" / "stage_a").resolve()


def _build_synthetic_phase(root: Path) -> dict[str, str]:
    root.mkdir(parents=True)
    (root / "payload.bin").write_bytes(b"sealed-payload")
    gate = {
        "signal": "decoupled_cssr_failed",
        "selected_method": None,
        "confirmation_allowed": False,
    }
    summary = {
        "status": "complete",
        "experiment_id": audit.LEGACY_EXPERIMENT_ID,
        "phase": "pilot",
        "pair_ids": list(audit.PAIR_IDS),
        "unit_count": 6,
        "gate": gate,
        "decision": "decoupled_cssr_failed",
        "config_sha256": audit.LEGACY_CONFIG_SHA256,
        "final_unknown_used": False,
        "even_angle_test_used": False,
        "final_unknown_test_authorized": False,
        "automatic_followon_authorized": False,
    }
    audit._write_json(root / "pilot_gate.json", gate)
    audit._write_json(root / "phase_summary.json", summary)
    audit._write_json(root / "artifact_hashes.json", audit._tree_artifact_hashes(root))
    audit._write_json(
        root / "_PHASE_SUCCESS.json",
        {
            "status": "complete",
            "phase_summary_sha256": file_sha256(root / "phase_summary.json"),
            "artifact_hashes_sha256": file_sha256(root / "artifact_hashes.json"),
        },
    )
    return {
        "manifest": file_sha256(root / "artifact_hashes.json"),
        "summary": file_sha256(root / "phase_summary.json"),
        "gate": file_sha256(root / "pilot_gate.json"),
        "success": file_sha256(root / "_PHASE_SUCCESS.json"),
    }


def _verify_synthetic_phase(root: Path, hashes: dict[str, str]) -> dict[str, Any]:
    return audit.verify_sealed_pilot_root(
        root,
        expected_manifest_sha256=hashes["manifest"],
        expected_summary_sha256=hashes["summary"],
        expected_gate_sha256=hashes["gate"],
        expected_success_sha256=hashes["success"],
    )


def test_phase_seal_verifies_every_recorded_file_and_detects_tampering(
    tmp_path: Path,
) -> None:
    phase = tmp_path / "sealed-pilot"
    hashes = _build_synthetic_phase(phase)
    result = _verify_synthetic_phase(phase, hashes)
    assert result["status"] == "passed"
    assert result["decision"] == "decoupled_cssr_failed"

    (phase / "payload.bin").write_bytes(b"tampered")
    with pytest.raises(DataValidationError, match="manifest does not reproduce"):
        _verify_synthetic_phase(phase, hashes)


def _evaluation_fixture(
    rows: list[dict[str, Any]],
    decomposition: audit.ScoreDecomposition,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_id = {str(row["sample_id"]): index for index, row in enumerate(rows)}
    class_names = {label: f"known-{label}" for label in range(5)}
    identities = [f"known-{label}" for label in range(5)] + [
        "surrogate-a",
        "surrogate-b",
    ]
    evaluation: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    d0_rows: list[dict[str, Any]] = []
    for ordinal, identity in enumerate(identities):
        known = identity.startswith("known-")
        label = int(identity.split("-")[1]) if known else -1
        source_ids = (
            (f"cal-{label}-0", f"cal-{label}-1")
            if known
            else (f"{identity}-0", f"{identity}-1")
        )
        predicted = label if known else ordinal % 5
        fused_logits = np.full(5, -2.0, dtype=np.float64)
        fused_logits[predicted] = 2.0
        manifest = {
            "pair_id": f"evaluation-{ordinal}",
            "class_name": identity,
            "view1_sample_id": source_ids[0],
            "view2_sample_id": source_ids[1],
            "view1_angle_deg": rows[by_id[source_ids[0]]]["angle_deg"],
            "view2_angle_deg": rows[by_id[source_ids[1]]]["angle_deg"],
            "view1_frame_id": rows[by_id[source_ids[0]]]["frame_id"],
            "view2_frame_id": rows[by_id[source_ids[1]]]["frame_id"],
            "evaluation_subset_role": (
                "known_calibration" if known else "surrogate_unknown"
            ),
            "evaluation_subset_index": ordinal,
        }
        anomalies = [
            decomposition.anomaly[by_id[source_ids[0]]],
            decomposition.anomaly[by_id[source_ids[1]]],
        ]
        prediction: dict[str, Any] = {
            **{key: value for key, value in manifest.items() if not key.startswith("evaluation_")},
            "evaluation_role": manifest["evaluation_subset_role"],
            "surrogate_identity": "" if known else identity,
            "true_label": label,
            "predicted_known_label": predicted,
            "predicted_known_class_name": class_names[predicted],
            "fused_logits": audit._json_vector(fused_logits),
            "unknown_score": 0.5
            * (float(anomalies[0][predicted]) + float(anomalies[1][predicted])),
        }
        for view, sample_id in enumerate(source_ids, start=1):
            index = by_id[sample_id]
            prediction[f"view{view}_r"] = audit._json_vector(
                decomposition.normalized_error[index]
            )
            prediction[f"view{view}_p_value"] = audit._json_vector(
                decomposition.p_value[index]
            )
            prediction[f"view{view}_a"] = audit._json_vector(
                decomposition.anomaly[index]
            )
        evaluation.append(manifest)
        predictions.append(prediction)
        d0_rows.append(dict(prediction))
    return evaluation, predictions, d0_rows


def _write_synthetic_unit(
    pilot_root: Path,
) -> tuple[Path, str, audit.ScoreDecomposition]:
    torch.manual_seed(7)
    model = FGMVCSSRDecoupled1D().eval()
    rows = _small_unique_rows()
    generator = np.random.default_rng(17)
    z = generator.normal(size=(len(rows), 128, 4)).astype(np.float32)
    decomposition, replay_r = audit._infer_and_decompose(
        model,
        z,
        rows,
        device=torch.device("cpu"),
        batch_size=128,
    )
    unit_root = (
        pilot_root
        / "N1"
        / "fold_0"
        / f"seed_{audit.LEGACY_SEED}"
        / D1_DECOUPLED_REL_CSSR
    )
    unit_root.mkdir(parents=True)
    audit._write_json(
        unit_root / "unit_contract.json",
        {
            "experiment_id": audit.LEGACY_EXPERIMENT_ID,
            "phase": "pilot",
            "mode": "full",
            "pair_id": "N1",
            "method": D1_DECOUPLED_REL_CSSR,
            "config_sha256": audit.LEGACY_CONFIG_SHA256,
            "source_hashes": audit.LEGACY_SOURCE_SHA256,
            "r2_retrained_or_finetuned": False,
            "final_unknown_used": False,
            "even_angle_test_used": False,
            "test_features_materialized": False,
        },
    )
    audit._write_csv(unit_root / "unique_base_sample_manifest.csv", rows)
    (unit_root / "source_pair_manifest.csv").write_bytes(b"synthetic-source-pairs\n")
    evaluation, prediction_rows, d0_rows = _evaluation_fixture(rows, decomposition)
    audit._write_csv(unit_root / "evaluation_pair_manifest.csv", evaluation)
    audit._write_csv(unit_root / "predictions_and_scores.csv", prediction_rows)
    audit._write_csv(unit_root / "d0_predictions_and_scores.csv", d0_rows)
    calibration = np.asarray(
        [row["experiment_role"] == "known_calibration" for row in rows]
    )
    surrogate = np.asarray(
        [row["experiment_role"] == "surrogate_unknown" for row in rows]
    )
    reference_arrays = {
        "r": decomposition.normalized_error,
        "known_calibration_p": decomposition.p_value[calibration],
        "known_calibration_a": decomposition.anomaly[calibration],
        "surrogate_unknown_p": decomposition.p_value[surrogate],
        "surrogate_unknown_a": decomposition.anomaly[surrogate],
        **{
            f"class_{index}_reference_r": values
            for index, values in enumerate(decomposition.references)
        },
    }
    audit._write_npz(unit_root / "reference_scores.npz", reference_arrays)
    audit._write_npz(
        unit_root / "checkpoint_replay.npz",
        {
            "unique_features": z,
            "expected_u": decomposition.adapted_features.astype(np.float32),
            "expected_r": replay_r.astype(np.float32),
            "expected_probabilities": decomposition.probabilities.astype(np.float32),
        },
    )
    checkpoint = {
        "experiment_id": audit.LEGACY_EXPERIMENT_ID,
        "phase": "pilot",
        "mode": "full",
        "pair_id": "N1",
        "method": D1_DECOUPLED_REL_CSSR,
        "architecture": "fg_mv_cssr_decoupled_1d_v1",
        "model_state_dict": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        "checkpoint_epoch": 20,
        "formal_checkpoint": True,
        "checkpoint_selection": "fixed_final_epoch",
        "cssr_seed": audit.LEGACY_SEED,
        "config_sha256": audit.LEGACY_CONFIG_SHA256,
        "unique_base_manifest_sha256": hashlib.sha256(
            (unit_root / "unique_base_sample_manifest.csv").read_bytes()
        ).hexdigest(),
        "source_pair_manifest_sha256": file_sha256(
            unit_root / "source_pair_manifest.csv"
        ),
        "unique_feature_map_sha256": audit.array_sha256(z),
        "final_unknown_used": False,
        "even_angle_test_used": False,
    }
    torch.save(checkpoint, unit_root / "checkpoint.pt")
    audit._write_json(unit_root / "unit_summary.json", {"status": "complete"})
    audit._write_json(
        unit_root / "artifact_hashes.json", audit._tree_artifact_hashes(unit_root)
    )
    audit._write_json(
        unit_root / "_SUCCESS.json",
        {
            "status": "complete",
            "unit_summary_sha256": file_sha256(unit_root / "unit_summary.json"),
            "artifact_hashes_sha256": file_sha256(
                unit_root / "artifact_hashes.json"
            ),
        },
    )
    return unit_root, file_sha256(unit_root / "checkpoint.pt"), decomposition


def test_strict_checkpoint_replay_and_saved_predictions_reproduce(
    tmp_path: Path,
) -> None:
    pilot = tmp_path / "pilot"
    _, checkpoint_hash, expected = _write_synthetic_unit(pilot)
    unit = audit.load_verified_legacy_unit(
        pilot,
        pair_id="N1",
        method=D1_DECOUPLED_REL_CSSR,
        device=torch.device("cpu"),
        expected_shape=(19, 128, 4),
        expected_role_counts={
            "train_known": 5,
            "known_calibration": 10,
            "surrogate_unknown": 4,
        },
        expected_checkpoint_sha256=checkpoint_hash,
    )
    assert np.array_equal(
        unit.decomposition.normalized_error,
        expected.normalized_error,
    )
    assert unit.prediction_rows[0]["predicted_known_label"] == "0"
    assert unit.model.training is False
    assert not any(parameter.requires_grad for parameter in unit.model.parameters())


def test_checkpoint_replay_rejects_one_changed_saved_value(tmp_path: Path) -> None:
    pilot = tmp_path / "pilot"
    unit_root, checkpoint_hash, _ = _write_synthetic_unit(pilot)
    with np.load(unit_root / "checkpoint_replay.npz", allow_pickle=False) as saved:
        arrays = {name: np.asarray(saved[name]) for name in saved.files}
    arrays["expected_r"] = arrays["expected_r"].copy()
    arrays["expected_r"][0, 0] += np.float32(1.0e-4)
    audit._write_npz(unit_root / "checkpoint_replay.npz", arrays)
    audit._write_json(
        unit_root / "artifact_hashes.json", audit._tree_artifact_hashes(unit_root)
    )
    audit._write_json(
        unit_root / "_SUCCESS.json",
        {
            "status": "complete",
            "unit_summary_sha256": file_sha256(unit_root / "unit_summary.json"),
            "artifact_hashes_sha256": file_sha256(
                unit_root / "artifact_hashes.json"
            ),
        },
    )
    with pytest.raises(DataValidationError, match="checkpoint replay changed"):
        audit.load_verified_legacy_unit(
            pilot,
            pair_id="N1",
            method=D1_DECOUPLED_REL_CSSR,
            device=torch.device("cpu"),
            expected_shape=(19, 128, 4),
            expected_role_counts={
                "train_known": 5,
                "known_calibration": 10,
                "surrogate_unknown": 4,
            },
            expected_checkpoint_sha256=checkpoint_hash,
        )


def test_pair_view_decomposition_uses_r2_class_and_linear_reference_p95(
    tmp_path: Path,
) -> None:
    pilot = tmp_path / "pilot"
    _, checkpoint_hash, _ = _write_synthetic_unit(pilot)
    unit = audit.load_verified_legacy_unit(
        pilot,
        pair_id="N1",
        method=D1_DECOUPLED_REL_CSSR,
        device=torch.device("cpu"),
        expected_shape=(19, 128, 4),
        expected_role_counts={
            "train_known": 5,
            "known_calibration": 10,
            "surrogate_unknown": 4,
        },
        expected_checkpoint_sha256=checkpoint_hash,
    )
    rows = audit.build_pair_view_decomposition(unit)
    first = rows[0]
    expected_p95 = np.quantile(
        unit.decomposition.references[0], 0.95, method="linear"
    )
    assert first["r2_predicted_label"] == 0
    assert first["predicted_class_reference_p95_r"] == pytest.approx(expected_p95)
    assert first["pair_a_mean"] == pytest.approx(
        0.5 * (first["view1_a_predicted"] + first["view2_a_predicted"])
    )
    assert first["pair_a_view1_minus_view2"] == pytest.approx(
        first["view1_a_predicted"] - first["view2_a_predicted"]
    )
    assert first["view_acceptance_pattern"] in {
        "both_accept",
        "one_accept",
        "both_reject",
    }
    identity_rows = audit.identity_view_diagnostics(rows)
    assert len(identity_rows) == 2
    identity = identity_rows[0]
    assert set(identity["score_variant_view_metrics"]) == {
        "raw_error_e",
        "normalized_error_r",
        "anomaly_a",
    }
    assert set(identity["a_score_view_metrics"]) == {"view1", "view2", "mean"}
    assert identity["predicted_class_reference_width"]["by_class"]
    assert "population_std" in identity["predicted_class_reference_width"][
        "pair_weighted"
    ]
    assert identity["view_acceptance_pattern_fraction"]
    assert identity["raw_activation_normalization_attribution"]["causal_label"] is False


def _statistical_unit() -> audit.VerifiedLegacyUnit:
    rows: list[dict[str, Any]] = []
    values: list[np.ndarray] = []
    for role, count in (("train_known", 1), ("known_calibration", 36)):
        for label in range(5):
            for local_index in range(count):
                rows.append(
                    {
                        "experiment_role": role,
                        "sample_id": f"{role}-{label}-{local_index}",
                        "class_name": f"known-{label}",
                        "model_label": label,
                    }
                )
                current = 1.0 + 0.2 * np.arange(5, dtype=np.float64)
                current[label] = 0.2 + 0.002 * local_index
                values.append(current)
    for surrogate_index, identity in enumerate(("surrogate-a", "surrogate-b")):
        for local_index in range(36):
            rows.append(
                {
                    "experiment_role": "surrogate_unknown",
                    "sample_id": f"{identity}-{local_index}",
                    "class_name": identity,
                    "model_label": -1,
                }
            )
            current = 0.6 + 0.1 * np.arange(5, dtype=np.float64)
            current[(surrogate_index + 1) % 5] = 0.15 + 0.001 * local_index
            values.append(current)
    normalized = np.asarray(values)
    p_value, anomaly, references, reference_ids = audit.conformal_p_and_anomaly(
        normalized, rows
    )
    count = len(rows)
    decomposition = audit.ScoreDecomposition(
        adapted_features=np.ones((count, 1, 2), dtype=np.float64),
        logits=np.zeros((count, 5, 2), dtype=np.float64),
        probabilities=np.full((count, 5), 0.2, dtype=np.float64),
        raw_error=normalized.copy(),
        activation_magnitude=np.ones(count, dtype=np.float64),
        normalized_error=normalized,
        p_value=p_value,
        anomaly=anomaly,
        references=references,
        reference_ids=reference_ids,
    )
    return audit.VerifiedLegacyUnit(
        pair_id="N2",
        method=D1_DECOUPLED_REL_CSSR,
        root=Path("."),
        model=None,  # type: ignore[arg-type]
        unique_rows=tuple(rows),
        evaluation_rows=(),
        prediction_rows=(),
        d0_prediction_rows=(),
        z=np.ones((count, 1, 2), dtype=np.float64),
        decomposition=decomposition,
        checkpoint_sha256="checkpoint",
        artifact_manifest_sha256="manifest",
    )


def test_ae_cross_matrices_and_specificity_use_identity_equal_weighting() -> None:
    unit = _statistical_unit()
    raw_rows, normalized_rows = audit.ae_cross_reconstruction_rows(unit)
    # Five train identities + five calibration identities + two surrogate
    # identities are kept separate by role.
    assert len(raw_rows) == len(normalized_rows) == 12
    surrogate_a = next(
        row
        for row in normalized_rows
        if row["experiment_role"] == "surrogate_unknown"
        and row["identity"] == "surrogate-a"
    )
    assert surrogate_a["best_ae_by_mean_r"] == 1
    assert surrogate_a["lowest_r_share_ae_1"] == 1.0

    specificity = audit.ae_specificity_summary(unit)
    assert set(specificity["by_ae"]) == {"0", "1", "2", "3", "4"}
    assert specificity["by_ae"]["0"]["identity_equal_weighting"] is True
    assert specificity["by_ae"]["0"]["own_known_reference_role"] == (
        "known_calibration"
    )


def test_reference_summary_uses_population_std_and_numpy_linear_quantiles() -> None:
    unit = _statistical_unit()
    rows = audit.reference_distribution_rows(unit)
    assert len(rows) == 5
    first = rows[0]
    values = unit.decomposition.references[0]
    q25, q75 = np.quantile(values, (0.25, 0.75), method="linear")
    assert first["count"] == 36
    assert first["population_std"] == pytest.approx(values.std(ddof=0))
    assert first["iqr"] == pytest.approx(q75 - q25)
    assert first["p95"] == pytest.approx(
        np.quantile(values, 0.95, method="linear")
    )


def _geometry_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role, identities in (
        ("train_known", (("known-0", 0), ("known-1", 1))),
        ("known_calibration", (("known-0", 0), ("known-1", 1))),
        ("surrogate_unknown", (("surrogate-a", -1), ("surrogate-b", -1))),
    ):
        for identity, label in identities:
            for local_index in range(2):
                rows.append(
                    {
                        "experiment_role": role,
                        "sample_id": f"{role}-{identity}-{local_index}",
                        "class_name": identity,
                        "model_label": label,
                    }
                )
    return rows


def test_representation_geometry_compares_z_and_u_on_four_unique_pools() -> None:
    rows = _geometry_rows()
    generator = np.random.default_rng(31)
    z = generator.normal(size=(len(rows), 4, 3)).astype(np.float32)
    u = (z + 0.1 * np.tanh(z)).astype(np.float32)
    result = audit.representation_geometry(z, u, rows)
    assert set(result["pools"]) == {
        "train_known",
        "known_calibration",
        "surrogate",
        "evaluation",
    }
    assert result["pair_multiplicity_used"] is False
    assert result["pools"]["evaluation"]["Z"]["identity_count"] == 4
    assert result["pools"]["train_known"]["U"]["entropy_effective_rank"] > 0.0
    assert result["adapter_residual_ratio"]["train_known"]["base_mean"] > 0.0


def _output_base_arrays() -> dict[str, np.ndarray]:
    return {
        "pair_ids": np.asarray(audit.PAIR_IDS, dtype=np.str_),
        "methods": np.asarray(audit.METHODS, dtype=np.str_),
        "sample_ids": np.asarray([["sample"]] * 3, dtype=np.str_),
        "experiment_roles": np.asarray([["train_known"]] * 3, dtype=np.str_),
        "class_names": np.asarray([["known"]] * 3, dtype=np.str_),
        "model_labels": np.zeros((3, 1), dtype=np.int64),
        "raw_error_e": np.ones((3, 2, 1, 5), dtype=np.float64),
        "activation_magnitude_m": np.ones((3, 2, 1), dtype=np.float64),
        "normalized_error_r": np.ones((3, 2, 1, 5), dtype=np.float64),
        "conformal_p": np.ones((3, 2, 1, 5), dtype=np.float64),
        "anomaly_a": np.zeros((3, 2, 1, 5), dtype=np.float64),
        "z_sha256": np.asarray(["z"] * 3, dtype=np.str_),
        "u_sha256": np.asarray([["u", "u"]] * 3, dtype=np.str_),
    }


def test_atomic_output_is_independent_hashed_and_non_gating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy = tmp_path / "legacy-pilot"
    phase_hashes = _build_synthetic_phase(legacy)
    phase = _verify_synthetic_phase(legacy, phase_hashes)
    before = audit._tree_snapshot(legacy)
    output = tmp_path / "independent-stage-a"
    row = {"field": "value"}
    mechanism_audit = {
        "status": "complete",
        "experiment_id": audit.EXPERIMENT_ID,
        "source_binding": {"status": "synthetic_test_fixture"},
        "runtime_contract": {
            "device": "cuda:0",
            "cuda_device_name": "NVIDIA GeForce RTX 4090",
            "deterministic_algorithms": True,
            "deterministic_warn_only": False,
            "cudnn_benchmark": False,
            "matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "cublas_workspace_config": ":4096:8",
            "global_seed": audit.LEGACY_SEED,
            "amp": False,
        },
        "sealed_pilot": phase,
        "stage_a_training_performed": False,
        "performance_gate_eligible": False,
        "stage_b_configuration_modified": False,
        "stage_b_selection_used": False,
        "immutable_input_tree_snapshot_unchanged": True,
        "confirmation_allowed": False,
        "automatic_followon_authorized": False,
        "final_unknown_test_authorized": False,
        "final_unknown_used": False,
        "even_angle_test_used": False,
        "augmentation_audits": {
            pair_id: {
                "status": "passed",
                "pair_id": pair_id,
                "normalization": {
                    "method": "reuse_exact_r2_global_scalar_zscore",
                    "mean": 0.0,
                    "std": 1.0,
                    "epsilon": 1.0e-8,
                    "unique_base_sample_count": 720,
                    "fitted_from_train_known_only": True,
                },
                "raw_z_replay_exact": True,
                "raw_z_sha256": "0" * 64,
                "augmented_z_sha256": "1" * 64,
                "augmentation": {
                    "status": "passed",
                    "family": "gain_uniform_0.9_1.1_plus_gaussian_std_0.02",
                    "seed": audit.SCORE_NORM_SEED,
                    "pair_id": pair_id,
                    "variant_count_per_base": 4,
                    "sample_variant_ids_sha256": "2" * 64,
                    "gain_sha256": "3" * 64,
                    "noise_sha256": "4" * 64,
                    "augmented_input_sha256": "5" * 64,
                    "method_id_in_seed_material": False,
                    "final_unknown_used": False,
                    "even_angle_test_used": False,
                },
                "data_access_audit": {
                    "status": "passed",
                    "policy": "enforced_source_known_odd_index_allowlist_v1",
                    "authorized_row_count": 7 * 180,
                    "profile_values_materialized_only_through_allowlist": True,
                    "final_unknown_profile_values_read": False,
                    "even_angle_profile_values_read": False,
                    "final_unknown_pairs_generated": False,
                    "even_angle_test_pairs_generated": False,
                },
                "surrogate_unknown_used": False,
                "known_calibration_used": False,
                "final_unknown_used": False,
                "even_angle_test_used": False,
            }
            for pair_id in audit.PAIR_IDS
        },
    }
    saved = audit.save_mechanism_outputs(
        output,
        input_roots=(legacy,),
        base_arrays=_output_base_arrays(),
        pair_rows=(row,),
        cross_raw_rows=(row,),
        cross_normalized_rows=(row,),
        specificity={"status": "passed"},
        reference_rows=(row,),
        geometry={"status": "passed"},
        official_rows=(
            {field: "" for field in audit.OFFICIAL_SCORE_CSV_FIELDS},
        ),
        mechanism_audit=mechanism_audit,
    )
    assert before == audit._tree_snapshot(legacy)
    assert {path.name for path in saved.iterdir()} == set(audit.OUTPUT_FILENAMES)
    monkeypatch.setattr(audit, "_verify_stage_a_source_record", lambda record: None)
    with pytest.raises(DataValidationError, match="shape or population"):
        audit.audit_mechanism_output(
            saved,
            legacy_pilot_root=legacy,
            expected_phase_hashes=phase_hashes,
        )
    monkeypatch.setattr(
        audit,
        "_audit_mechanism_payload",
        lambda root, payload, legacy_root: None,
    )
    checked = audit.audit_mechanism_output(
        saved,
        legacy_pilot_root=legacy,
        expected_phase_hashes=phase_hashes,
    )
    assert checked["status"] == "passed"
    assert checked["performance_gate_eligible"] is False
    assert checked["confirmation_allowed"] is False


def test_stage_a_runtime_contract_rejects_nondeterministic_or_wrong_gpu() -> None:
    valid = {
        "device": "cuda:0",
        "cuda_device_name": "NVIDIA GeForce RTX 4090",
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "cudnn_benchmark": False,
        "matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "cublas_workspace_config": ":4096:8",
        "global_seed": audit.LEGACY_SEED,
        "amp": False,
    }
    audit._validate_stage_a_runtime_contract(valid)
    for key, value in (
        ("cuda_device_name", "Other GPU"),
        ("deterministic_algorithms", False),
        ("deterministic_warn_only", True),
        ("cudnn_benchmark", True),
        ("matmul_allow_tf32", True),
        ("cudnn_allow_tf32", True),
        ("cublas_workspace_config", ":16:8"),
    ):
        changed = {**valid, key: value}
        with pytest.raises(DataValidationError, match="runtime contract"):
            audit._validate_stage_a_runtime_contract(changed)


def test_official_posthoc_diagnostic_cannot_modify_old_gate(tmp_path: Path) -> None:
    pilot = tmp_path / "pilot"
    _, checkpoint_hash, _ = _write_synthetic_unit(pilot)
    unit = audit.load_verified_legacy_unit(
        pilot,
        pair_id="N1",
        method=D1_DECOUPLED_REL_CSSR,
        device=torch.device("cpu"),
        expected_shape=(19, 128, 4),
        expected_role_counts={
            "train_known": 5,
            "known_calibration": 10,
            "surrogate_unknown": 4,
        },
        expected_checkpoint_sha256=checkpoint_hash,
    )
    # The synthetic random checkpoint need not predict all five classes on its
    # five train bases.  Bind one train base to each class so this test reaches
    # the post-hoc non-gating contract instead of the intended empty-template
    # hard failure.
    probabilities = unit.decomposition.probabilities.copy()
    train_indices = [
        index
        for index, row in enumerate(unit.unique_rows)
        if row["experiment_role"] == "train_known"
    ]
    probabilities[train_indices] = np.eye(5, dtype=probabilities.dtype)
    decomposition = audit.ScoreDecomposition(
        adapted_features=unit.decomposition.adapted_features,
        logits=unit.decomposition.logits,
        probabilities=probabilities,
        raw_error=unit.decomposition.raw_error,
        activation_magnitude=unit.decomposition.activation_magnitude,
        normalized_error=unit.decomposition.normalized_error,
        p_value=unit.decomposition.p_value,
        anomaly=unit.decomposition.anomaly,
        references=unit.decomposition.references,
        reference_ids=unit.decomposition.reference_ids,
    )
    unit = audit.VerifiedLegacyUnit(
        pair_id=unit.pair_id,
        method=unit.method,
        root=unit.root,
        model=unit.model,
        unique_rows=unit.unique_rows,
        evaluation_rows=unit.evaluation_rows,
        prediction_rows=unit.prediction_rows,
        d0_prediction_rows=unit.d0_prediction_rows,
        z=unit.z,
        decomposition=decomposition,
        checkpoint_sha256=unit.checkpoint_sha256,
        artifact_manifest_sha256=unit.artifact_manifest_sha256,
    )
    train_count = sum(
        row["experiment_role"] == "train_known" for row in unit.unique_rows
    )
    generator = np.random.default_rng(41)
    augmented_z = generator.normal(size=(train_count * 4, 128, 4)).astype(np.float32)
    augmented = audit.build_augmented_unit_arrays(
        unit,
        augmented_z,
        device=torch.device("cpu"),
        batch_size=7,
    )
    detailed, metrics, posthoc = audit.official_posthoc_score_rows(
        unit,
        augmented,
        device=torch.device("cpu"),
        batch_size=4,
    )
    assert detailed and metrics
    assert list(detailed[0]) == list(audit.OFFICIAL_SCORE_CSV_FIELDS)
    assert {row["score_rule"] for row in metrics} == {"S1", "S2", "S3", "full"}
    assert {row["scope"] for row in metrics} == {"pair", "identity"}
    assert all(row["performance_gate_eligible"] is False for row in detailed)
    assert all(row["performance_gate_eligible"] is False for row in metrics)
    assert posthoc["stage_b_model_or_gate_modified"] is False
    assert posthoc["stage_b_selection_used"] is False
    assert posthoc["final_unknown_used"] is False


def test_official_posthoc_failure_is_local_and_does_not_block_other_stage_a(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unit = _statistical_unit()

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("predicted template class is empty")

    monkeypatch.setattr(audit, "official_posthoc_score_rows", fail)
    detailed, metrics, status = audit.guarded_official_posthoc_score_rows(
        unit,
        {},
        device=torch.device("cpu"),
    )
    assert detailed == []
    assert metrics == []
    assert status["status"] == "failed"
    assert status["performance_gate_eligible"] is False
    assert status["stage_b_model_or_gate_modified"] is False
    assert status["confirmation_allowed"] is False
