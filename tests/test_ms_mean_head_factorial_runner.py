from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from hrrp_osr.data.errors import DataValidationError
from hrrp_osr.models.arpl import ARPLReciprocalHead
from hrrp_osr.training import ms_mean_head_factorial as runner
from hrrp_osr.training.arpl_pilot import (
    PreparedSurrogateSplit,
    ScalarNormalization,
    _state_sha256,
)
from hrrp_osr.training.ms_mean_head_factorial import (
    METRIC_KEYS,
    METHODS,
    IntentionalTrainingInterruption,
    aggregate_phase_root,
    audit_method_result,
    audit_phase_root,
    evaluate_inference_arrays,
    load_ms_mean_head_factorial_config,
    recompute_unit_metrics_from_rows,
    save_method_result,
    task_source_hashes,
    train_one_method,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/experiments/arpl/ms_mean_head_factorial_surrogate_v1.yaml"
)


class _ArtifactModel(nn.Module):
    """Small state container used to exercise persistence without training a CNN."""

    def __init__(self, method: str) -> None:
        super().__init__()
        self.architecture_id = f"test-only-{method}"
        if method in {METHODS[0], METHODS[2]}:
            self.head_type = "ce"
            self.global_head: nn.Module = nn.Linear(1, 5)
        else:
            self.head_type = "arpl"
            self.global_head = ARPLReciprocalHead(
                known_class_count=5,
                feature_dim=1,
            )


class _TinyTrainModel(nn.Module):
    """Fast differentiable stand-in for checkpoint/resume orchestration tests."""

    def __init__(self, method: str) -> None:
        super().__init__()
        self.method = method
        self.known_class_count = 5
        self.global_head = nn.Linear(2, self.known_class_count)
        self.view_head = None

    def forward(
        self, inputs: torch.Tensor, *, compute_rejector: bool = False
    ) -> SimpleNamespace:
        assert compute_rejector is False
        features = inputs.mean(dim=2)
        return SimpleNamespace(global_logits=self.global_head(features))

    def representation_loss(
        self,
        output: SimpleNamespace,
        labels: torch.Tensor,
        *,
        lambda_view: float,
    ) -> dict[str, torch.Tensor]:
        assert lambda_view == 0.0
        classification = F.cross_entropy(output.global_logits, labels)
        zero = classification.new_zeros(())
        return {
            "total": classification,
            "global_classification": classification,
            "global_margin": zero,
        }


def _pair_rows_and_roles() -> tuple[
    tuple[dict[str, Any], ...],
    dict[str, tuple[str, ...]],
    dict[str, tuple[str, ...]],
    dict[str, np.ndarray],
]:
    rows: list[dict[str, Any]] = []
    pair_ids: dict[str, list[str]] = {
        "train": [],
        "known_calibration": [],
        "surrogate_unknown": [],
    }
    class_names: dict[str, list[str]] = {key: [] for key in pair_ids}
    labels: dict[str, list[int]] = {key: [] for key in pair_ids}
    role_specs = (
        ("train", "train_known", 5),
        ("known_calibration", "known_calibration", 5),
        ("surrogate_unknown", "surrogate_unknown", 2),
    )
    row_index = 0
    for prepared_role, manifest_role, class_count in role_specs:
        for class_index in range(class_count):
            class_name = (
                f"known-{class_index}"
                if prepared_role != "surrogate_unknown"
                else f"unknown-{class_index}"
            )
            for repeat in range(10):
                pair_id = f"{prepared_role}-{class_index}-{repeat}"
                base_prefix = "train" if prepared_role == "train" else "eval"
                rows.append(
                    {
                        "pair_id": pair_id,
                        "experiment_role": manifest_role,
                        "class_name": class_name,
                        "view1_frame_id": repeat % 24,
                        "view2_frame_id": (repeat + 1) % 24,
                        "view1_angle_deg": 2 * (repeat % 12) + 1,
                        "view2_angle_deg": 2 * ((repeat + 1) % 12) + 1,
                        "view1_sample_id": f"{base_prefix}-v1-{row_index}",
                        "view2_sample_id": f"{base_prefix}-v2-{row_index}",
                    }
                )
                pair_ids[prepared_role].append(pair_id)
                class_names[prepared_role].append(class_name)
                labels[prepared_role].append(
                    class_index if prepared_role != "surrogate_unknown" else 5
                )
                row_index += 1
    return (
        tuple(rows),
        {key: tuple(value) for key, value in pair_ids.items()},
        {key: tuple(value) for key, value in class_names.items()},
        {key: np.asarray(value, dtype=np.int64) for key, value in labels.items()},
    )


def _prepared_for_artifacts() -> PreparedSurrogateSplit:
    rows, pair_ids, class_names, labels = _pair_rows_and_roles()
    # Use the production CSV serializer so the stored manifest bytes and hash
    # exercise exactly the same persistence boundary as a real prepared split.
    manifest_bytes = runner._render_csv(rows)
    inputs = {
        role: np.zeros((len(role_labels), 2, 601), dtype=np.float32)
        for role, role_labels in labels.items()
    }
    return PreparedSurrogateSplit(
        split_id="N0_F0",
        angle_fold=0,
        train_class_order=tuple(f"known-{index}" for index in range(5)),
        surrogate_class_order=("unknown-0", "unknown-1"),
        pair_manifest_rows=rows,
        pair_manifest_bytes=manifest_bytes,
        pair_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        pair_audit={
            "status": "passed",
            "train_evaluation_base_overlap": 0,
            "final_unknown_pairs": 0,
            "even_angle_pairs": 0,
            "test_pairs_generated": False,
            "test_features_materialized": False,
            "surrogate_train_pairs_materialized": False,
        },
        normalization=ScalarNormalization(0.0, 1.0, 1e-8, 100),
        inputs=inputs,
        labels=labels,
        pair_ids=pair_ids,
        class_names=class_names,
    )


def _prepared_for_training() -> PreparedSurrogateSplit:
    rng = np.random.default_rng(907)
    train_labels = np.arange(10, dtype=np.int64) % 5
    manifest = b"test-only-checkpoint-manifest\n"
    return PreparedSurrogateSplit(
        split_id="N0_F0",
        angle_fold=0,
        train_class_order=tuple(f"known-{index}" for index in range(5)),
        surrogate_class_order=("unknown-0", "unknown-1"),
        pair_manifest_rows=(),
        pair_manifest_bytes=manifest,
        pair_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        pair_audit={},
        normalization=ScalarNormalization(0.0, 1.0, 1e-8, 20),
        inputs={
            "train": rng.normal(size=(10, 2, 601)).astype(np.float32),
            "known_calibration": rng.normal(size=(10, 2, 601)).astype(np.float32),
            "surrogate_unknown": rng.normal(size=(4, 2, 601)).astype(np.float32),
        },
        labels={
            "train": train_labels,
            "known_calibration": train_labels.copy(),
            "surrogate_unknown": np.full(4, 5, dtype=np.int64),
        },
        pair_ids={
            "train": tuple(f"train-{index}" for index in range(10)),
            "known_calibration": tuple(f"cal-{index}" for index in range(10)),
            "surrogate_unknown": tuple(f"unknown-{index}" for index in range(4)),
        },
        class_names={
            "train": tuple(f"known-{value}" for value in train_labels),
            "known_calibration": tuple(f"known-{value}" for value in train_labels),
            "surrogate_unknown": (
                "unknown-0",
                "unknown-0",
                "unknown-1",
                "unknown-1",
            ),
        },
    )


def _role_arrays(labels: np.ndarray) -> dict[str, np.ndarray]:
    count = int(labels.size)
    logits = np.full((count, 5), -4.0, dtype=np.float64)
    if np.all(labels < 5):
        for index, label in enumerate(labels):
            logits[index, int(label)] = 3.0 - index / 1000.0
    else:
        for index in range(count):
            logits[index, index % 5] = -1.0 - index / 1000.0
    per_view_logits = np.repeat(logits[:, None, :], 2, axis=1)
    return {
        "per_view_features": np.zeros((count, 2, 1), dtype=np.float32),
        "fused_features": np.zeros((count, 1), dtype=np.float32),
        "per_view_logits": per_view_logits,
        "global_logits": logits,
        "unknown_score": -logits.max(axis=1),
        "labels": labels.copy(),
    }


def _artifact_inputs(
    prepared: PreparedSurrogateSplit,
    config: Mapping[str, Any],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, float], list[dict[str, Any]]]:
    arrays = {
        role: _role_arrays(prepared.labels[role])
        for role in ("train", "known_calibration", "surrogate_unknown")
    }
    metrics, rows = evaluate_inference_arrays(
        arrays,
        prepared=prepared,
        config=config,
    )
    return arrays, metrics, rows


def _initialization_audit() -> dict[str, Any]:
    return {
        "seed": 20260830,
        "component_hashes": {
            METHODS[0]: {"backbone": "shallow", "head": "ce"},
            METHODS[1]: {"backbone": "shallow", "head": "arpl"},
            METHODS[2]: {"backbone": "multiscale", "head": "ce"},
            METHODS[3]: {"backbone": "multiscale", "head": "arpl"},
        },
        "checks": {
            "shallow_backbone_equal": True,
            "multiscale_backbone_equal": True,
            "ce_head_equal": True,
            "arpl_head_equal": True,
            "independent_model_objects": True,
            "forbidden_components_absent": True,
        },
    }


def _trained_artifact(
    method: str,
    *,
    source_hashes: Mapping[str, str],
    order_hash: str,
) -> dict[str, Any]:
    model = _ArtifactModel(method)
    epoch_order_hash = "same-epoch-order"
    return {
        "model": model,
        "final_state": {
            name: value.detach().clone() for name, value in model.state_dict().items()
        },
        "checkpoint_epoch": 1,
        "formal_checkpoint": False,
        "training_log": [
            {
                "epoch": 1,
                "method": method,
                "pseudo_unknown_generated": False,
                "checkpoint_selected_for_open_set_performance": False,
                "train_order_epoch_sha256": epoch_order_hash,
            }
        ],
        "source_hashes": dict(source_hashes),
        "runtime_contract": {
            "device": "cpu",
            "device_type": "cpu",
            "device_name": None,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "cuda_version": torch.version.cuda,
            "cublas_workspace_config": ":4096:8",
            "torch_intraop_threads": 4,
            "torch_interop_threads": 1,
            "deterministic_algorithms": True,
            "amp": False,
            "tf32": False,
            "torch_compile": False,
        },
        "training_order_sha256": order_hash,
        "epoch_order_hashes": [epoch_order_hash],
        "pseudo_audit": {
            "status": "not_applicable",
            "pseudo_unknown_generated": False,
            "pseudo_count": 0,
            "surrogate_unknown_used": False,
            "final_unknown_used": False,
            "even_angle_test_used": False,
        },
    }


def _save_smoke_matrix(root: Path) -> tuple[dict[str, Any], PreparedSurrogateSplit]:
    config = load_ms_mean_head_factorial_config(CONFIG_PATH)
    prepared = _prepared_for_artifacts()
    arrays, metrics, prediction_rows = _artifact_inputs(prepared, config)
    source_hashes = task_source_hashes(PROJECT_ROOT)
    epoch_order_hash = "same-epoch-order"
    training_order_hash = runner._training_order_sha256([epoch_order_hash])
    for method in METHODS:
        destination = root / "N0" / "fold_0" / "seed_20260830" / method
        save_method_result(
            destination,
            phase="smoke",
            pair_id="N0",
            angle_fold=0,
            method=method,
            seed=20260830,
            prepared=prepared,
            trained=_trained_artifact(
                method,
                source_hashes=source_hashes,
                order_hash=training_order_hash,
            ),
            arrays=arrays,
            metrics=metrics,
            prediction_rows=prediction_rows,
            permutation={"status": "passed", "eval_mode": True},
            length_audit={
                "diagnostic_only": True,
                "used_for_gate": False,
                "fold_length_safe": True,
            },
            length_rows=(
                {
                    "sample_id": "test-only",
                    "experiment_role": "train_known",
                    "profile_length": 601,
                },
            ),
            initialization=_initialization_audit(),
            config=config,
            wall_time_seconds=0.0,
        )
    return config, prepared


def test_prediction_rows_recompute_all_nine_metrics_with_zero_error() -> None:
    config = load_ms_mean_head_factorial_config(CONFIG_PATH)
    prepared = _prepared_for_artifacts()
    _, metrics, rows = _artifact_inputs(prepared, config)

    recomputed = recompute_unit_metrics_from_rows(
        rows,
        known_class_count=5,
        known_acceptance_rate=0.95,
    )

    assert tuple(METRIC_KEYS) == (
        "known_accuracy",
        "known_macro_f1",
        "auroc",
        "oscr",
        "fpr95",
        "known_correct_acceptance_rate",
        "unknown_rejection_rate",
        "open_set_harmonic_score",
        "k_plus_1_macro_f1",
    )
    assert {key: abs(metrics[key] - recomputed[key]) for key in METRIC_KEYS} == {
        key: 0.0 for key in METRIC_KEYS
    }
    assert metrics["threshold"] == recomputed["threshold"]


def test_save_audit_and_phase_fairness_chain_has_no_pseudo_path(
    tmp_path: Path,
) -> None:
    phase_root = tmp_path / "smoke"
    config, _ = _save_smoke_matrix(phase_root)

    for method in METHODS:
        destination = phase_root / "N0" / "fold_0" / "seed_20260830" / method
        audit = audit_method_result(
            destination,
            config=config,
            phase="smoke",
            pair_id="N0",
            angle_fold=0,
            seed=20260830,
            method=method,
            require_formal=False,
        )
        assert audit["all_nine_metrics_recomputed"] is True
        assert audit["npz_predictions_crosschecked"] is True
        assert audit["pseudo_unknown_absent"] is True
        pseudo = json.loads(
            (destination / "pseudo_unknown_audit.json").read_text(encoding="utf-8")
        )
        assert pseudo == {
            "even_angle_test_used": False,
            "final_unknown_used": False,
            "pseudo_count": 0,
            "pseudo_unknown_generated": False,
            "status": "not_applicable",
            "surrogate_unknown_used": False,
        }
        with np.load(destination / "features_logits_scores.npz") as stored:
            assert not any(
                "pseudo" in key or "reject" in key for key in stored.files
            )

    summary = aggregate_phase_root(
        CONFIG_PATH,
        phase_root,
        phase="smoke",
    )
    integrity = audit_phase_root(
        CONFIG_PATH,
        phase_root,
        phase="smoke",
        verify_root_hashes=True,
    )

    assert summary["status"] == "complete"
    assert summary["analysis"]["performance_used_for_decision"] is False
    assert summary["final_unknown_test_authorized"] is False
    assert integrity["training_task_count"] == 4
    assert integrity["all_pair_manifests_shared_within_units"] is True
    assert integrity["all_prediction_orders_shared_within_units"] is True
    assert integrity["all_dataloader_orders_shared_within_units"] is True
    assert integrity["all_pseudo_unknown_paths_absent"] is True
    fairness = json.loads(
        (phase_root / "fairness_by_unit.json").read_text(encoding="utf-8")
    )
    assert len(fairness) == 1
    checks = fairness[0]["checks"]
    assert checks["pair_manifest_sha256"] is True
    assert checks["prediction_order_sha256"] is True
    assert checks["prediction_label_order_sha256"] is True
    assert checks["training_order_sha256"] is True
    # Identity-conditioned outputs are confirmatory diagnostics and must not be
    # generated from the diagnostic smoke phase.
    assert not (phase_root / "surrogate_identity_metrics_by_unit.csv").exists()
    assert not (phase_root / "surrogate_identity_metrics_aggregate.csv").exists()
    assert not (phase_root / "surrogate_identity_metrics_overall.csv").exists()
    assert not (phase_root / "surrogate_absorption.csv").exists()


def test_checkpoint_resume_is_exact_and_never_creates_pseudo_unknown(
    tmp_path: Path,
) -> None:
    config = copy.deepcopy(load_ms_mean_head_factorial_config(CONFIG_PATH))
    config["training"].update(
        {
            "batch_size": 5,
            "total_epochs": 2,
            "smoke_epochs": 2,
            "warmup_epochs": 1,
        }
    )
    prepared = _prepared_for_training()
    method = METHODS[0]

    torch.manual_seed(119)
    continuous_model = _TinyTrainModel(method)
    continuous = train_one_method(
        continuous_model,
        method=method,
        prepared=prepared,
        seed=20260830,
        config=config,
        mode="smoke",
        device=torch.device("cpu"),
    )

    checkpoint = tmp_path / "latest.pt"
    torch.manual_seed(119)
    interrupted_model = _TinyTrainModel(method)
    with pytest.raises(IntentionalTrainingInterruption):
        train_one_method(
            interrupted_model,
            method=method,
            prepared=prepared,
            seed=20260830,
            config=config,
            mode="smoke",
            device=torch.device("cpu"),
            resume_checkpoint=checkpoint,
            _interrupt_after_epoch=1,
        )
    assert checkpoint.is_file()
    checkpoint_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert checkpoint_state["completed_epoch"] == 1
    assert checkpoint_state["pseudo_unknown_generated"] is False
    assert not any("pseudo" in key for key in checkpoint_state["model_state_dict"])

    torch.manual_seed(119)
    resumed_model = _TinyTrainModel(method)
    resumed = train_one_method(
        resumed_model,
        method=method,
        prepared=prepared,
        seed=20260830,
        config=config,
        mode="smoke",
        device=torch.device("cpu"),
        resume_checkpoint=checkpoint,
    )

    assert _state_sha256(continuous["final_state"]) == _state_sha256(
        resumed["final_state"]
    )
    assert continuous["epoch_order_hashes"] == resumed["epoch_order_hashes"]
    assert continuous["training_order_sha256"] == resumed["training_order_sha256"]
    assert continuous["pseudo_audit"] == resumed["pseudo_audit"] == {
        "status": "not_applicable",
        "pseudo_unknown_generated": False,
        "pseudo_count": 0,
        "surrogate_unknown_used": False,
        "final_unknown_used": False,
        "even_angle_test_used": False,
    }
    for uninterrupted_row, resumed_row in zip(
        continuous["training_log"], resumed["training_log"], strict=True
    ):
        assert {
            key: value
            for key, value in uninterrupted_row.items()
            if key != "elapsed_seconds"
        } == {
            key: value
            for key, value in resumed_row.items()
            if key != "elapsed_seconds"
        }


def _write_prediction_unit(
    root: Path,
    *,
    pair_id: str,
    fold: int,
    seed: int,
    method: str,
    rows: list[dict[str, Any]],
) -> None:
    destination = runner._unit_destination(
        root,
        pair_id=pair_id,
        angle_fold=fold,
        seed=seed,
        method=method,
    )
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "predictions.csv").write_bytes(runner._render_csv(rows))


def _identity_prediction_rows() -> list[dict[str, Any]]:
    rows = [
        {
            "evaluation_role": "known_calibration",
            "class_name": f"known-{index}",
            "unknown_score": score,
            "rejected": False,
            "threshold": 0.3,
            "predicted_known_class_name": f"known-{index}",
        }
        for index, score in enumerate((0.0, 0.2))
    ]
    rows.extend(
        {
            "evaluation_role": "surrogate_unknown",
            "class_name": "unknown-A",
            "unknown_score": score,
            "rejected": rejected,
            "threshold": 0.3,
            "predicted_known_class_name": "known-0",
        }
        for score, rejected in ((0.8, True), (0.9, False))
    )
    rows.extend(
        {
            "evaluation_role": "surrogate_unknown",
            "class_name": "unknown-B",
            "unknown_score": score,
            "rejected": False,
            "threshold": 0.3,
            "predicted_known_class_name": "known-1",
        }
        for score in (-0.2, -0.1)
    )
    return rows


def test_confirmation_surrogate_identity_metrics_cover_unit_pair_and_overall(
    tmp_path: Path,
) -> None:
    # Each identity is deliberately present in two pair contexts.  This is the
    # frozen confirmatory multiplicity checked by the overall aggregator.
    expected = []
    method = METHODS[0]
    for pair_id in ("P0", "P1"):
        for fold in (0, 4):
            for seed in (20260830, 20260831, 20260832):
                expected.append((pair_id, fold, seed, method))
                _write_prediction_unit(
                    tmp_path,
                    pair_id=pair_id,
                    fold=fold,
                    seed=seed,
                    method=method,
                    rows=_identity_prediction_rows(),
                )

    unit_rows, pair_context_rows, overall_rows = (
        runner._surrogate_identity_metric_rows(tmp_path, expected)
    )

    assert len(unit_rows) == 2 * 2 * 3 * 2
    for row in unit_rows:
        if row["surrogate_identity"] == "unknown-A":
            assert row["auroc"] == 1.0
            assert row["unknown_rejection_rate"] == 0.5
        else:
            assert row["surrogate_identity"] == "unknown-B"
            assert row["auroc"] == 0.0
            assert row["unknown_rejection_rate"] == 0.0

    assert len(pair_context_rows) == 2 * 2
    for row in pair_context_rows:
        assert row["unit_count"] == 6
        expected_auroc = 1.0 if row["surrogate_identity"] == "unknown-A" else 0.0
        expected_urr = 0.5 if row["surrogate_identity"] == "unknown-A" else 0.0
        assert row["mean_auroc"] == expected_auroc
        assert row["minimum_auroc"] == expected_auroc
        assert row["maximum_auroc"] == expected_auroc
        assert row["mean_unknown_rejection_rate"] == expected_urr

    assert len(overall_rows) == 2
    for row in overall_rows:
        assert row["identity_pair_context_count"] == 2
        assert row["unit_count"] == 12
        expected_auroc = 1.0 if row["surrogate_identity"] == "unknown-A" else 0.0
        expected_urr = 0.5 if row["surrogate_identity"] == "unknown-A" else 0.0
        assert row["mean_auroc"] == expected_auroc
        assert row["minimum_auroc"] == expected_auroc
        assert row["maximum_auroc"] == expected_auroc
        assert row["mean_unknown_rejection_rate"] == expected_urr


def test_absorption_counts_only_false_accepts_and_keeps_fully_rejected_summary(
    tmp_path: Path,
) -> None:
    expected = []
    method = METHODS[0]
    for pair_index, pair_id in enumerate(("P0", "P1")):
        expected.append((pair_id, 0, 20260830, method))
        rows = [
            {
                "evaluation_role": "surrogate_unknown",
                "class_name": "unknown-A",
                "unknown_score": 0.2,
                "rejected": False,
                "threshold": 0.3,
                "predicted_known_class_name": f"known-{pair_index}",
            },
            {
                "evaluation_role": "surrogate_unknown",
                "class_name": "unknown-A",
                "unknown_score": 0.8,
                "rejected": True,
                "threshold": 0.3,
                # A rejected sample must not be counted as absorption.
                "predicted_known_class_name": "known-rejected",
            },
            {
                "evaluation_role": "surrogate_unknown",
                "class_name": "unknown-B",
                "unknown_score": 0.9,
                "rejected": True,
                "threshold": 0.3,
                "predicted_known_class_name": "known-rejected",
            },
            {
                "evaluation_role": "surrogate_unknown",
                "class_name": "unknown-B",
                "unknown_score": 1.0,
                "rejected": True,
                "threshold": 0.3,
                "predicted_known_class_name": "known-rejected",
            },
        ]
        _write_prediction_unit(
            tmp_path,
            pair_id=pair_id,
            fold=0,
            seed=20260830,
            method=method,
            rows=rows,
        )

    detail, summary = runner._absorption_rows(tmp_path, expected)

    assert len(summary) == 4
    unknown_a_summary = [
        row for row in summary if row["surrogate_identity"] == "unknown-A"
    ]
    unknown_b_summary = [
        row for row in summary if row["surrogate_identity"] == "unknown-B"
    ]
    assert all(row["total_surrogate_count"] == 2 for row in summary)
    assert all(row["rejected_count"] == 1 for row in unknown_a_summary)
    assert all(row["false_accept_count"] == 1 for row in unknown_a_summary)
    assert all(row["unknown_rejection_rate"] == 0.5 for row in unknown_a_summary)
    assert all(row["rejected_count"] == 2 for row in unknown_b_summary)
    assert all(row["false_accept_count"] == 0 for row in unknown_b_summary)
    assert all(row["unknown_rejection_rate"] == 1.0 for row in unknown_b_summary)
    assert {row["absorbed_as_known_identity"] for row in detail} == {
        "known-0",
        "known-1",
    }
    assert all(row["surrogate_identity"] == "unknown-A" for row in detail)
    assert all(row["false_accept_count"] == 1 for row in detail)
    assert all(row["composition_within_false_accepts"] == 1.0 for row in detail)

    overall_detail, overall_summary = runner._absorption_overall_rows(
        detail, summary
    )
    assert len(overall_summary) == 2
    overall_by_identity = {
        row["surrogate_identity"]: row for row in overall_summary
    }
    assert overall_by_identity["unknown-A"] == {
        "method": method,
        "surrogate_identity": "unknown-A",
        "identity_pair_context_count": 2,
        "total_surrogate_count": 4,
        "rejected_count": 2,
        "false_accept_count": 2,
        "unknown_rejection_rate": 0.5,
        "false_accept_rate": 0.5,
    }
    assert overall_by_identity["unknown-B"] == {
        "method": method,
        "surrogate_identity": "unknown-B",
        "identity_pair_context_count": 2,
        "total_surrogate_count": 4,
        "rejected_count": 4,
        "false_accept_count": 0,
        "unknown_rejection_rate": 1.0,
        "false_accept_rate": 0.0,
    }
    assert {row["absorbed_as_known_identity"] for row in overall_detail} == {
        "known-0",
        "known-1",
    }
    assert all(row["surrogate_identity"] == "unknown-A" for row in overall_detail)


def _write_confirmation_launcher_records(root: Path) -> dict[str, Any]:
    config = load_ms_mean_head_factorial_config(CONFIG_PATH)
    tasks = runner._plan_payload(config, "confirmation")["tasks"]
    gpu_count = 4
    jobs_per_gpu = 4
    worker_count = gpu_count * jobs_per_gpu
    tokens = [f"GPU-test-{index}" for index in range(gpu_count)]
    assignments = []
    results = []
    launch_root = root / "launcher"
    launch_root.mkdir(parents=True)
    for index, task in enumerate(tasks):
        worker, gpu = runner._gpu_worker_assignment(
            index,
            gpu_count=gpu_count,
            jobs_per_gpu=jobs_per_gpu,
        )
        assignment = {
            "task_index": index,
            "worker_slot": worker,
            "physical_gpu_index": gpu,
            "visible_gpu_token": tokens[gpu],
            **task,
        }
        assignments.append(assignment)
        log_path = launch_root / f"task_{index:03d}.log"
        log_path.write_text("completed\n", encoding="utf-8")
        results.append(
            {
                "worker_slot": worker,
                "physical_gpu_index": gpu,
                "visible_gpu_token": tokens[gpu],
                **task,
                "exit_code": 0,
                "elapsed_seconds": 1.0,
                "log": str(log_path.relative_to(root)),
            }
        )
    runner._write_json(
        launch_root / "launch_manifest.json",
        {
            "experiment_id": runner.EXPERIMENT_ID,
            "phase": "confirmation",
            "config_sha256": config["_config_sha256"],
            "gpu_names": [config["runtime"]["expected_gpu_model"]] * gpu_count,
            "inherited_cuda_visible_devices": None,
            "child_visible_gpu_tokens": tokens,
            "jobs_per_gpu": jobs_per_gpu,
            "worker_count": worker_count,
            "gpu_assignment": "rotating_method_balanced_v1",
            "training_task_count": len(tasks),
            "assignments": assignments,
            "resume": False,
            "final_unknown_test_authorized": False,
        },
    )
    runner._write_json(
        launch_root / "launch_results.json",
        {
            "training_task_count": len(results),
            "successful_task_count": len(results),
            "failed_task_count": 0,
            "results": results,
        },
    )
    return config


def test_confirmation_launcher_audit_accepts_exact_four_by_four_plan(
    tmp_path: Path,
) -> None:
    config = _write_confirmation_launcher_records(tmp_path)

    audit = runner._audit_confirmation_launcher(tmp_path, config)
    expected_method_tasks_per_gpu = {
        METHODS[0]: {"0": 11, "1": 11, "2": 10, "3": 10},
        METHODS[1]: {"0": 10, "1": 11, "2": 11, "3": 10},
        METHODS[2]: {"0": 10, "1": 10, "2": 11, "3": 11},
        METHODS[3]: {"0": 11, "1": 10, "2": 10, "3": 11},
    }

    assert audit == {
        "status": "passed",
        "gpu_count": 4,
        "jobs_per_gpu": 4,
        "worker_count": 16,
        "tasks_per_gpu": {"0": 42, "1": 42, "2": 42, "3": 42},
        "method_tasks_per_gpu": expected_method_tasks_per_gpu,
        "every_unit_covers_four_gpus": True,
        "every_method_balanced_across_gpus": True,
        "training_task_count": 168,
        "successful_task_count": 168,
        "failed_task_count": 0,
        "all_logs_present": True,
        "all_assignments_exact": True,
    }
    assert audit["every_unit_covers_four_gpus"] is True
    assert audit["every_method_balanced_across_gpus"] is True
    assert all(
        sum(per_gpu.values()) == 42
        and max(per_gpu.values()) - min(per_gpu.values()) == 1
        for per_gpu in audit["method_tasks_per_gpu"].values()
    )


@pytest.mark.parametrize("tamper", ("exit_code", "worker_slot", "gpu_mapping"))
def test_confirmation_launcher_audit_rejects_tampered_execution_mapping(
    tmp_path: Path,
    tamper: str,
) -> None:
    config = _write_confirmation_launcher_records(tmp_path)
    manifest_path = tmp_path / "launcher" / "launch_manifest.json"
    results_path = tmp_path / "launcher" / "launch_results.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if tamper == "exit_code":
        results["results"][0]["exit_code"] = 1
        runner._write_json(results_path, results)
    elif tamper == "worker_slot":
        results["results"][0]["worker_slot"] = 1
        runner._write_json(results_path, results)
    else:
        manifest["assignments"][0]["physical_gpu_index"] = 1
        runner._write_json(manifest_path, manifest)

    with pytest.raises(DataValidationError, match="GPU launch"):
        runner._audit_confirmation_launcher(tmp_path, config)
