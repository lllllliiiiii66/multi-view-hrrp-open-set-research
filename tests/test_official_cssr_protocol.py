from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hrrp_osr.data.errors import DataConfigError, DataValidationError
from hrrp_osr.training.official_cssr_protocol import (
    METHODS,
    O0_R2_CC_MLS,
    O1_OFFICIAL_LINEAR_FT,
    O2_OFFICIAL_PCSSR_FT,
    O3_OFFICIAL_LINEAR_E2E,
    O4_OFFICIAL_PCSSR_E2E,
    OFFICIAL_CSSR_SEED,
    PILOT_PAIRS,
    TRAINABLE_METHODS,
    build_phase_plan,
    build_score_norm_augmentation,
    build_training_epoch_material,
    derive_seed,
    evaluate_pilot_gate,
    learning_rates_for_update,
    load_official_cssr_config,
    model_initialization_seed,
    validate_official_cssr_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/experiments/cssr/official_cssr_hrrp_pilot_v1.yaml"
)


def _training_population() -> tuple[list[dict[str, Any]], np.ndarray]:
    rows: list[dict[str, Any]] = []
    inputs: list[np.ndarray] = []
    base = np.linspace(-1.0, 1.0, 601, dtype=np.float64)
    for label in range(5):
        for index in range(144):
            rows.append(
                {
                    "experiment_role": "train_known",
                    "model_label": label,
                    "sample_id": f"class-{label}-sample-{143-index:03d}",
                    "angle_deg": 2 * (index % 180) + 1,
                }
            )
            inputs.append(base + label + index / 1000.0)
    return rows, np.asarray(inputs, dtype=np.float64)


def test_config_is_byte_locked_and_freezes_sources_and_scope(tmp_path: Path) -> None:
    config = load_official_cssr_config(CONFIG_PATH)

    assert config["experiment_id"] == "official_cssr_hrrp_pilot_v1"
    assert config["_config_path"] == str(CONFIG_PATH.resolve())
    assert len(config["_config_sha256"]) == 64
    assert config["classes"]["pilot_pairs"] == ["N1", "N4", "N2"]
    assert config["bundle"]["bundle_sha256"] == (
        "79176091b5b745e5df5957f1bb4ade7304c1403a4a32e64622e012db67d5a0c5"
    )
    assert config["prior_r2"]["unit_artifact_hashes"]["N1"]["checkpoint.pt"] == (
        "a4f6fa3235fbb5cf74b712588a0318f614a05287adec4ee881820424cddbcbaa"
    )
    assert config["outputs"]["confirmation_allowed"] is False
    assert config["outputs"]["automatic_followon_authorized"] is False
    assert config["outputs"]["final_unknown_test_authorized"] is False

    changed = tmp_path / "changed.yaml"
    changed.write_bytes(CONFIG_PATH.read_bytes().replace(b"epochs: 40", b"epochs: 41", 1))
    with pytest.raises(DataConfigError, match="config bytes changed"):
        load_official_cssr_config(changed)


def test_in_memory_config_validation_rejects_protocol_mutation() -> None:
    config = copy.deepcopy(load_official_cssr_config(CONFIG_PATH))
    config["training"]["head_base_lr"] = 0.051
    with pytest.raises(DataConfigError, match="training"):
        validate_official_cssr_config(config)


def test_phase_plans_are_exact_and_never_authorize_followon() -> None:
    config = load_official_cssr_config(CONFIG_PATH)
    smoke = build_phase_plan(config, "smoke")
    pilot = build_phase_plan(config, "pilot")

    assert [(row["pair_id"], row["method"]) for row in smoke] == [
        ("N1", method) for method in TRAINABLE_METHODS
    ]
    assert all(row["epochs"] == 1 and row["diagnostic_only"] for row in smoke)
    assert len(pilot) == 12
    assert [(row["pair_id"], row["method"]) for row in pilot] == [
        (pair_id, method)
        for pair_id in PILOT_PAIRS
        for method in TRAINABLE_METHODS
    ]
    assert all(row["epochs"] == 40 for row in pilot)
    assert all(row["reused_baseline"] == O0_R2_CC_MLS for row in pilot)
    for row in smoke + pilot:
        assert row["confirmation_allowed"] is False
        assert row["automatic_followon_authorized"] is False
        assert row["final_unknown_test_authorized"] is False
        assert row["even_angle_test_authorized"] is False
    for forbidden in ("confirmation", "final_test", "even_angle_test"):
        with pytest.raises(DataValidationError, match="not authorized"):
            build_phase_plan(config, forbidden)


def test_seed_and_initialization_streams_are_deterministic_and_shared() -> None:
    config = load_official_cssr_config(CONFIG_PATH)
    material = "official_cssr_hrrp_schedule_v1|20260906|pilot|N1|fold_0|1|gain"
    assert derive_seed(material) == derive_seed(material)
    assert 0 <= derive_seed(material) < 2**64
    linear_seed = model_initialization_seed("linear", "N1", config)
    assert linear_seed == model_initialization_seed("linear", "N1", config)
    assert 0 <= linear_seed < 2**64
    assert linear_seed >= 2**32
    assert model_initialization_seed("linear", "N1", config) != model_initialization_seed(
        "pcssr", "N1", config
    )
    assert OFFICIAL_CSSR_SEED == 20260906


def test_training_material_is_deterministic_sample_bound_and_source_aligned() -> None:
    config = load_official_cssr_config(CONFIG_PATH)
    rows, inputs = _training_population()
    first = build_training_epoch_material(rows, inputs, "pilot", "N1", 1, config)
    repeated = build_training_epoch_material(rows, inputs, "pilot", "N1", 1, config)
    next_epoch = build_training_epoch_material(rows, inputs, "pilot", "N1", 2, config)

    assert np.array_equal(first["indices"], repeated["indices"])
    assert np.array_equal(first["gain"], repeated["gain"])
    assert np.array_equal(first["noise"], repeated["noise"])
    assert np.array_equal(first["augmented_inputs"], repeated["augmented_inputs"])
    assert not np.array_equal(first["indices"], next_epoch["indices"])
    assert first["augmented_inputs"].dtype == np.float32
    assert first["augmented_inputs"].shape == (720, 601)
    assert len(set(first["indices"].tolist())) == 720
    expected = (first["gain"][:, None] * inputs + first["noise"]).astype(np.float32)
    assert np.array_equal(first["augmented_inputs"], expected)
    assert first["audit"]["batch_sizes"] == [128, 128, 128, 128, 128, 80]
    assert first["audit"]["optimizer_updates"] == 6
    assert len(first["audit"]["scheduled_sample_ids"]) == 720
    assert first["audit"]["scheduled_sample_ids"] == [
        rows[int(index)]["sample_id"] for index in first["indices"]
    ]
    assert repeated["audit"]["scheduled_sample_ids"] == first["audit"][
        "scheduled_sample_ids"
    ]
    for name in (
        "schedule_sha256",
        "gain_sha256",
        "noise_sha256",
        "augmented_inputs_sha256",
    ):
        assert first[name] == first["audit"][name]

    permutation = np.arange(720)[::-1]
    reordered_rows = [rows[int(index)] for index in permutation]
    reordered_inputs = inputs[permutation]
    reordered = build_training_epoch_material(
        reordered_rows, reordered_inputs, "pilot", "N1", 1, config
    )
    by_id = {
        row["sample_id"]: (
            first["gain"][index],
            first["noise"][index],
            first["augmented_inputs"][index],
        )
        for index, row in enumerate(rows)
    }
    for index, row in enumerate(reordered_rows):
        gain, noise, augmented = by_id[row["sample_id"]]
        assert reordered["gain"][index] == gain
        assert np.array_equal(reordered["noise"][index], noise)
        assert np.array_equal(reordered["augmented_inputs"][index], augmented)
    assert reordered["schedule_sha256"] == first["schedule_sha256"]
    assert reordered["gain_sha256"] == first["gain_sha256"]
    assert reordered["noise_sha256"] == first["noise_sha256"]


def test_training_material_rejects_leakage_duplicate_and_wrong_smoke_unit() -> None:
    config = load_official_cssr_config(CONFIG_PATH)
    rows, inputs = _training_population()
    leaked = copy.deepcopy(rows)
    leaked[0]["experiment_role"] = "surrogate_unknown"
    with pytest.raises(DataValidationError, match="non-train-known"):
        build_training_epoch_material(leaked, inputs, "pilot", "N1", 1, config)
    duplicated = copy.deepcopy(rows)
    duplicated[1]["sample_id"] = duplicated[0]["sample_id"]
    with pytest.raises(DataValidationError, match="not unique"):
        build_training_epoch_material(duplicated, inputs, "pilot", "N1", 1, config)
    even = copy.deepcopy(rows)
    even[0]["angle_deg"] = 2
    with pytest.raises(DataValidationError, match="even or invalid"):
        build_training_epoch_material(even, inputs, "pilot", "N1", 1, config)
    with pytest.raises(DataValidationError, match="smoke schedule"):
        build_training_epoch_material(rows, inputs, "smoke", "N4", 1, config)


def test_score_normalization_augmentation_is_four_variant_sample_bound() -> None:
    config = load_official_cssr_config(CONFIG_PATH)
    rows, inputs = _training_population()
    first = build_score_norm_augmentation(rows, inputs, "N2", 1, config)
    repeated = build_score_norm_augmentation(rows, inputs, "N2", 1, config)
    second = build_score_norm_augmentation(rows, inputs, "N2", 2, config)

    assert np.array_equal(first["augmented_inputs"], repeated["augmented_inputs"])
    assert first["gain_sha256"] == repeated["gain_sha256"]
    assert first["noise_sha256"] == repeated["noise_sha256"]
    assert not np.array_equal(first["augmented_inputs"], second["augmented_inputs"])
    assert first["augmented_inputs"].dtype == np.float32
    expected = (first["gain"][:, None] * inputs + first["noise"]).astype(np.float32)
    assert np.array_equal(first["augmented_inputs"], expected)
    assert first["audit"]["known_calibration_used"] is False
    assert first["audit"]["surrogate_unknown_used"] is False
    with pytest.raises(DataValidationError, match="namespace"):
        build_score_norm_augmentation(
            rows, inputs, "N2", 1, config, namespace="posthoc"
        )
    with pytest.raises(DataValidationError, match="outside"):
        build_score_norm_augmentation(rows, inputs, "N2", 5, config)


def test_learning_rate_is_exact_per_update_with_frozen_encoder() -> None:
    config = load_official_cssr_config(CONFIG_PATH)
    assert learning_rates_for_update(1, 0, 6) == pytest.approx(
        {"head": 0.05 / 12.0, "encoder": 0.0}
    )
    assert learning_rates_for_update(2, 5, 6) == pytest.approx(
        {"head": 0.05, "encoder": 0.0}
    )
    assert learning_rates_for_update(5, 5, 6) == pytest.approx(
        {"head": 0.05, "encoder": 0.0}
    )
    assert learning_rates_for_update(6, 0, 6) == pytest.approx(
        {"head": 0.05, "encoder": 0.005}
    )
    assert learning_rates_for_update(
        40,
        0,
        6,
        method=O1_OFFICIAL_LINEAR_FT,
        config=config,
    ) == pytest.approx({"head": 0.0005, "encoder": 0.0})
    assert learning_rates_for_update(
        6,
        0,
        6,
        method=O4_OFFICIAL_PCSSR_E2E,
        config=config,
    ) == pytest.approx({"head": 0.05, "encoder": 0.005})
    assert learning_rates_for_update(25, 0, 6) == pytest.approx(
        {"head": 0.005, "encoder": 0.0005}
    )
    assert learning_rates_for_update(35, 0, 6) == pytest.approx(
        {"head": 0.0005, "encoder": 0.00005}
    )
    with pytest.raises(DataValidationError, match="frozen schedule"):
        learning_rates_for_update(41, 0, 6)


_IDENTITIES = {
    "N1": ("DDG-112", "迷你好望角型散货船"),
    "N4": ("DDG-1000", "集装箱船达飞罗尔多夫级"),
    "N2": ("油气轮MARVEL CRANE", "迷你好望角型散货船"),
}


def _gate_rows(
    overrides: dict[str, dict[str, float]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    overrides = overrides or {}
    metrics: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    for pair_id in PILOT_PAIRS:
        for method in METHODS:
            values = {
                "auroc": 0.60,
                "oscr": 0.50,
                "known_correct_acceptance_rate": 0.90,
                "fpr95": 0.20,
            }
            values.update(overrides.get(method, {}))
            metrics.append({"pair_id": pair_id, "method": method, **values})
            for identity in _IDENTITIES[pair_id]:
                identities.append(
                    {
                        "pair_id": pair_id,
                        "method": method,
                        "surrogate_identity": identity,
                        "auroc": values["auroc"],
                    }
                )
    return metrics, identities


def _score_rows(delta: float = 0.02) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair_id in PILOT_PAIRS:
        for method in (O2_OFFICIAL_PCSSR_FT, O4_OFFICIAL_PCSSR_E2E):
            rows.extend(
                [
                    {
                        "pair_id": pair_id,
                        "method": method,
                        "score_variant": "S1",
                        "auroc": 0.60,
                    },
                    {
                        "pair_id": pair_id,
                        "method": method,
                        "score_variant": "full",
                        "auroc": 0.60 + delta,
                    },
                ]
            )
    return rows


def _successful_tasks() -> list[dict[str, Any]]:
    return [
        {
            "pair_id": pair_id,
            "method": method,
            "status": "success",
            "audit_passed": True,
        }
        for pair_id in PILOT_PAIRS
        for method in TRAINABLE_METHODS
    ]


def test_gate_priority_strong_then_method_then_ft() -> None:
    metrics, identities = _gate_rows(
        {
            O3_OFFICIAL_LINEAR_E2E: {"auroc": 0.59},
            O4_OFFICIAL_PCSSR_E2E: {
                "auroc": 0.62,
                "oscr": 0.51,
                "known_correct_acceptance_rate": 0.90,
                "fpr95": 0.19,
            },
            O2_OFFICIAL_PCSSR_FT: {
                "auroc": 0.63,
                "oscr": 0.53,
                "known_correct_acceptance_rate": 0.90,
                "fpr95": 0.17,
            },
        }
    )
    strong = evaluate_pilot_gate(metrics, identities, _score_rows(), task_rows=_successful_tasks())
    assert strong["pilot_status"] == "completed"
    assert strong["pilot_gate"] == "evaluated"
    assert strong["result_label"] == "official_cssr_strong_signal"
    assert strong["selected_method"] == O4_OFFICIAL_PCSSR_E2E

    metrics, identities = _gate_rows(
        {
            O3_OFFICIAL_LINEAR_E2E: {"auroc": 0.57},
            O4_OFFICIAL_PCSSR_E2E: {"auroc": 0.60},
        }
    )
    method_only = evaluate_pilot_gate(metrics, identities)
    assert method_only["result_label"] == "official_cssr_method_signal_only"

    metrics, identities = _gate_rows(
        {
            O2_OFFICIAL_PCSSR_FT: {
                "auroc": 0.63,
                "oscr": 0.53,
                "known_correct_acceptance_rate": 0.90,
                "fpr95": 0.17,
            }
        }
    )
    ft_only = evaluate_pilot_gate(metrics, identities)
    assert ft_only["result_label"] == "official_cssr_ft_signal_only"
    assert ft_only["selected_method"] == O2_OFFICIAL_PCSSR_FT


def test_gate_score_integration_no_signal_and_identity_catastrophe() -> None:
    metrics, identities = _gate_rows()
    integrated = evaluate_pilot_gate(metrics, identities, _score_rows())
    assert integrated["result_label"] == "official_cssr_score_integration_only"
    assert integrated["selected_method"] is None
    assert integrated["score_integration"]["qualifying_variants"] == [
        O2_OFFICIAL_PCSSR_FT,
        O4_OFFICIAL_PCSSR_E2E,
    ]

    no_signal = evaluate_pilot_gate(metrics, identities)
    assert no_signal["result_label"] == "official_cssr_no_signal"

    strong_metrics, strong_identities = _gate_rows(
        {
            O3_OFFICIAL_LINEAR_E2E: {"auroc": 0.59},
            O4_OFFICIAL_PCSSR_E2E: {"auroc": 0.62},
        }
    )
    for row in strong_identities:
        if (
            row["pair_id"] == "N2"
            and row["method"] == O4_OFFICIAL_PCSSR_E2E
            and row["surrogate_identity"] == "油气轮MARVEL CRANE"
        ):
            row["auroc"] = 0.39
    catastrophe = evaluate_pilot_gate(strong_metrics, strong_identities)
    assert catastrophe["result_label"] == "official_cssr_no_signal"
    assert catastrophe["safe_identity"][O4_OFFICIAL_PCSSR_E2E]["passed"] is False


def test_gate_hard_failure_precedes_no_signal_and_all_authorizations_are_false() -> None:
    metrics, identities = _gate_rows()
    incomplete = evaluate_pilot_gate(metrics[:-1], identities)
    assert incomplete["pilot_status"] == "hard_failed_incomplete"
    assert incomplete["pilot_gate"] == "not_evaluated"
    assert incomplete["result_label"] is None
    assert incomplete["selected_method"] is None
    assert incomplete["hard_failure_reasons"]

    tasks = _successful_tasks()
    tasks[-1]["status"] = "failed"
    failed = evaluate_pilot_gate(metrics, identities, task_rows=tasks)
    assert failed["pilot_status"] == "hard_failed_incomplete"
    assert failed["result_label"] is None

    evaluated = evaluate_pilot_gate(metrics, identities)
    for result in (incomplete, failed, evaluated):
        assert result["confirmation_allowed"] is False
        assert result["automatic_followon_authorized"] is False
        assert result["final_unknown_test_authorized"] is False
        assert result["even_angle_test_authorized"] is False


def test_supplied_score_ablation_must_be_complete_and_finite() -> None:
    metrics, identities = _gate_rows()
    score_rows = _score_rows()
    incomplete = evaluate_pilot_gate(metrics, identities, score_rows[:-1])
    assert incomplete["pilot_status"] == "hard_failed_incomplete"
    assert any(
        reason.startswith("missing_score_ablation")
        for reason in incomplete["hard_failure_reasons"]
    )
