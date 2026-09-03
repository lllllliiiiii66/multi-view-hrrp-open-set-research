from __future__ import annotations

import copy
import hashlib
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from hrrp_osr.data.errors import DataValidationError  # noqa: E402
from hrrp_osr.data.manifest import file_sha256  # noqa: E402
from hrrp_osr.data.processed import ProcessedBundle  # noqa: E402
from hrrp_osr.evaluation.ms_mean_factorial import REPORT_METRIC_KEYS  # noqa: E402
from hrrp_osr.models.arpl import ARPLReciprocalHead  # noqa: E402
from hrrp_osr.training import fg_mv_cssr_pilot as runner  # noqa: E402
from hrrp_osr.training.arpl_pilot import (  # noqa: E402
    PreparedSurrogateSplit,
    ScalarNormalization,
)
from hrrp_osr.training.fg_mv_cssr_pilot import (  # noqa: E402
    SCORE_RULES,
    _metric_rows_from_audits,
    build_cssr_reference_scores,
    build_unique_base_sample_manifest,
    load_and_audit_frozen_r2,
    load_fg_mv_cssr_config,
    recompute_metrics_from_prediction_rows,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/experiments/cssr/fg_mv_cssr_frozen_r2_v1.yaml"
)


def _prepared_with_rows(
    pair_rows: list[dict[str, Any]],
) -> PreparedSurrogateSplit:
    return PreparedSurrogateSplit(
        split_id="N1",
        angle_fold=0,
        train_class_order=tuple(f"known-{index}" for index in range(5)),
        surrogate_class_order=("surrogate-0", "surrogate-1"),
        pair_manifest_rows=tuple(pair_rows),
        pair_manifest_bytes=b"synthetic-manifest",
        pair_manifest_sha256="a" * 64,
        pair_audit={},
        normalization=ScalarNormalization(
            mean=0.0,
            std=1.0,
            epsilon=1.0e-8,
            unique_base_sample_count=900,
        ),
        inputs={},
        labels={},
        pair_ids={},
        class_names={},
    )


def _formal_unique_base_fixture() -> tuple[PreparedSurrogateSplit, ProcessedBundle]:
    bundle_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []

    def add_population(
        *,
        experiment_role: str,
        class_name: str,
        model_label: int,
        angles: list[int],
    ) -> None:
        samples: list[dict[str, Any]] = []
        for angle in angles:
            row_index = len(bundle_rows)
            sample_id = f"{class_name}-angle-{angle}"
            source = {
                "sample_id": sample_id,
                "processed_row_index": row_index,
                "class_name": class_name,
                "class_role": "known",
                "angle_deg": angle,
            }
            bundle_rows.append(source)
            samples.append(source)
        half = len(samples) // 2
        for pair_index, (left, right) in enumerate(
            zip(samples[:half], samples[half:], strict=True)
        ):
            pair_rows.append(
                {
                    "experiment_role": experiment_role,
                    "pair_id": (
                        f"{experiment_role}-{class_name}-pair-{pair_index}"
                    ),
                    "class_name": class_name,
                    "model_label": model_label,
                    "view1_sample_id": left["sample_id"],
                    "view1_row_index": left["processed_row_index"],
                    "view1_angle_deg": left["angle_deg"],
                    "view1_frame_id": int(left["angle_deg"]) // 15,
                    "view2_sample_id": right["sample_id"],
                    "view2_row_index": right["processed_row_index"],
                    "view2_angle_deg": right["angle_deg"],
                    "view2_frame_id": int(right["angle_deg"]) // 15,
                }
            )

    odd_angles = list(range(1, 360, 2))
    for class_index in range(5):
        add_population(
            experiment_role="train_known",
            class_name=f"known-{class_index}",
            model_label=class_index,
            angles=odd_angles[:144],
        )
        add_population(
            experiment_role="known_calibration",
            class_name=f"known-{class_index}",
            model_label=class_index,
            angles=odd_angles[144:],
        )
    for surrogate_index in range(2):
        add_population(
            experiment_role="surrogate_unknown",
            class_name=f"surrogate-{surrogate_index}",
            model_label=surrogate_index,
            angles=odd_angles[:36],
        )

    bundle = ProcessedBundle(
        root=Path("/synthetic"),
        profiles=np.empty((len(bundle_rows), 0), dtype=np.float64),
        rows=tuple(bundle_rows),
        profiles_sha256="b" * 64,
        manifest_sha256="c" * 64,
        bundle_sha256="d" * 64,
    )
    return _prepared_with_rows(pair_rows), bundle


def test_unique_base_manifest_deduplicates_pair_multiplicity_without_weighting() -> None:
    prepared, bundle = _formal_unique_base_fixture()
    baseline = build_unique_base_sample_manifest(prepared, bundle)

    # Repeat existing pair rows many times.  This changes pair multiplicity but
    # must not change the unique-base training or reference populations.
    repeated = replace(
        prepared,
        pair_manifest_rows=(
            *prepared.pair_manifest_rows,
            *prepared.pair_manifest_rows[:10],
            *prepared.pair_manifest_rows[:10],
        ),
    )
    observed = build_unique_base_sample_manifest(repeated, bundle)

    assert observed == baseline
    assert len(observed) == 720 + 180 + 72
    assert len({row["sample_id"] for row in observed}) == len(observed)
    assert Counter(row["experiment_role"] for row in observed) == Counter(
        {
            "train_known": 720,
            "known_calibration": 180,
            "surrogate_unknown": 72,
        }
    )
    assert Counter(
        int(row["model_label"])
        for row in observed
        if row["experiment_role"] == "train_known"
    ) == Counter({index: 144 for index in range(5)})


def _reference_fixture() -> tuple[list[dict[str, Any]], np.ndarray]:
    rows: list[dict[str, Any]] = []
    rho_rows: list[list[float]] = []
    for class_index in range(5):
        for sample_index in range(36):
            rows.append(
                {
                    "experiment_role": "known_calibration",
                    "sample_id": f"cal-{class_index}-{sample_index}",
                    "model_label": class_index,
                }
            )
            # Every class reference is 1..36.  This makes the denominator
            # difference between own-class LOO and other classes observable.
            rho_rows.append([float(sample_index + 1)] * 5)
    for sample_index in range(2):
        rows.append(
            {
                "experiment_role": "surrogate_unknown",
                "sample_id": f"surrogate-{sample_index}",
                "model_label": sample_index,
            }
        )
        rho_rows.append([10_000.0] * 5)
    return rows, np.asarray(rho_rows, dtype=np.float64)


def test_cssr_references_exclude_surrogate_and_loo_only_the_true_class_base() -> None:
    unique_rows, rho = _reference_fixture()

    arrays, references, reference_ids, metadata = build_cssr_reference_scores(
        unique_rows=unique_rows,
        rho=rho,
        epsilon=1.0e-8,
    )

    assert metadata["surrogate_unknown_in_reference"] is False
    assert metadata["calibration_leave_one_base_sample_out"] is True
    assert metadata["reference_counts"] == [36] * 5
    for class_index in range(5):
        np.testing.assert_array_equal(
            references[class_index], np.arange(1.0, 37.0)
        )
        assert all(
            sample_id.startswith(f"cal-{class_index}-")
            for sample_id in reference_ids[class_index]
        )
        assert not any("surrogate" in value for value in reference_ids[class_index])

    # cal-0-35 has rho=36 for every class.  Its own true-class reference drops
    # that base: (1+0)/(35+1).  Other classes retain all 36 bases: (1+1)/(36+1).
    own_class_p = arrays["known_calibration_p"][35, 0]
    other_class_p = arrays["known_calibration_p"][35, 1]
    assert own_class_p == pytest.approx(1.0 / 36.0)
    assert other_class_p == pytest.approx(2.0 / 37.0)
    np.testing.assert_allclose(
        arrays["surrogate_unknown_p"],
        np.full((2, 5), 1.0 / 37.0),
    )


class _TinyFrozenR2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(2, 2)
        self.global_head = nn.Linear(2, 5)

    @property
    def forbidden_component_status(self) -> Mapping[str, bool]:
        return {
            "sab_created": False,
            "pma_created": False,
            "view_head_created": False,
            "rejector_created": False,
            "pseudo_unknown_supported": False,
        }


def _minimal_frozen_r2_fixture(
    tmp_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    PreparedSurrogateSplit,
    Path,
    dict[str, dict[str, np.ndarray]],
]:
    config = copy.deepcopy(load_fg_mv_cssr_config(CONFIG_PATH))
    manifest_bytes = b"pair_id\nsynthetic\n"
    prepared = PreparedSurrogateSplit(
        split_id="N1",
        angle_fold=0,
        train_class_order=tuple(f"known-{index}" for index in range(5)),
        surrogate_class_order=("surrogate-0", "surrogate-1"),
        pair_manifest_rows=(),
        pair_manifest_bytes=manifest_bytes,
        pair_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        pair_audit={},
        normalization=ScalarNormalization(0.0, 1.0, 1.0e-8, 900),
        inputs={
            role: np.zeros((1, 2, 601), dtype=np.float32)
            for role in ("train", "known_calibration", "surrogate_unknown")
        },
        labels={
            role: np.zeros(1, dtype=np.int64)
            for role in ("train", "known_calibration", "surrogate_unknown")
        },
        pair_ids={
            role: (f"{role}-pair",)
            for role in ("train", "known_calibration", "surrogate_unknown")
        },
        class_names={
            role: ("known-0",)
            for role in ("train", "known_calibration", "surrogate_unknown")
        },
    )
    unit_root = (
        tmp_path
        / "N1"
        / "fold_0"
        / "seed_20260830"
        / "R2_MS_MEAN_CE"
    )
    unit_root.mkdir(parents=True)
    (unit_root / "pair_manifest.csv").write_bytes(manifest_bytes)
    (unit_root / "artifact_hashes.json").write_text("{}", encoding="utf-8")

    model = _TinyFrozenR2()
    checkpoint = {
        "experiment_id": config["prior_r2"]["experiment_id"],
        "phase": "confirmation",
        "pair_id": "N1",
        "angle_fold": 0,
        "method": "R2_MS_MEAN_CE",
        "checkpoint_epoch": 100,
        "formal_checkpoint": True,
        "checkpoint_selection": "fixed_final_epoch",
        "initialization_seed": 20260830,
        "config_sha256": config["prior_r2"]["source_config_sha256"],
        "pair_manifest_sha256": prepared.pair_manifest_sha256,
        "pseudo_unknown_generated": False,
        "train_class_order": prepared.train_class_order,
        "surrogate_class_order": prepared.surrogate_class_order,
        "normalization": asdict(prepared.normalization),
        "model_state_dict": model.state_dict(),
    }
    torch.save(checkpoint, unit_root / "checkpoint.pt")

    inferred = {
        role: {
            "global_logits": np.full((1, 5), index, dtype=np.float32),
        }
        for index, role in enumerate(
            ("train", "known_calibration", "surrogate_unknown")
        )
    }
    np.savez(
        unit_root / "features_logits_scores.npz",
        **{
            f"{role}_{name}": value
            for role, role_arrays in inferred.items()
            for name, value in role_arrays.items()
        },
    )
    expected_unit_hashes = {
        filename: file_sha256(unit_root / filename)
        for filename in (
            "checkpoint.pt",
            "pair_manifest.csv",
            "features_logits_scores.npz",
        )
    }
    runner._write_json(unit_root / "artifact_hashes.json", expected_unit_hashes)
    relative = str(unit_root.relative_to(tmp_path))
    runner._write_json(
        tmp_path / "artifact_hashes.json",
        {
            f"{relative}/{filename}": digest
            for filename, digest in expected_unit_hashes.items()
        },
    )
    config["prior_r2"]["root_artifact_hash_manifest_sha256"] = file_sha256(
        tmp_path / "artifact_hashes.json"
    )
    config["prior_r2"]["unit_artifact_hashes"]["N1"] = expected_unit_hashes
    prior_config = {"training": {"batch_size": 1}}
    return config, prior_config, prepared, tmp_path, inferred


def test_frozen_r2_loader_disables_gradients_and_contains_no_arpl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, prior_config, prepared, results_root, inferred = (
        _minimal_frozen_r2_fixture(tmp_path)
    )
    monkeypatch.setattr(
        runner,
        "_build_model",
        lambda _method, _known_class_count, _config: _TinyFrozenR2(),
    )

    def fake_infer(
        _model: nn.Module,
        inputs: np.ndarray,
        _labels: np.ndarray,
        **_kwargs: Any,
    ) -> dict[str, np.ndarray]:
        role = {
            id(prepared.inputs[name]): name
            for name in ("train", "known_calibration", "surrogate_unknown")
        }[id(inputs)]
        return inferred[role]

    monkeypatch.setattr(runner, "infer_model", fake_infer)

    model, arrays, audit = load_and_audit_frozen_r2(
        project_root=PROJECT_ROOT,
        r2_results_root=results_root,
        pair_id="N1",
        config=config,
        prepared=prepared,
        prior_config=prior_config,
        device=torch.device("cpu"),
    )

    assert arrays == inferred
    assert model.training is False
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert not any(
        isinstance(module, ARPLReciprocalHead) for module in model.modules()
    )
    assert audit["all_parameters_frozen"] is True
    assert audit["arpl_module_instantiated"] is False
    assert audit["old_outputs_exact"] is True


def test_frozen_r2_loader_rejects_changed_root_hash_manifest(
    tmp_path: Path,
) -> None:
    config, prior_config, prepared, results_root, _ = _minimal_frozen_r2_fixture(
        tmp_path
    )
    (results_root / "artifact_hashes.json").write_text(
        '{"tampered": "root binding"}\n', encoding="utf-8"
    )

    with pytest.raises(
        DataValidationError, match="root artifact hash manifest changed"
    ):
        load_and_audit_frozen_r2(
            project_root=PROJECT_ROOT,
            r2_results_root=results_root,
            pair_id="N1",
            config=config,
            prepared=prepared,
            prior_config=prior_config,
            device=torch.device("cpu"),
        )


def test_frozen_r2_loader_rejects_changed_bound_artifact_even_if_legacy_audit_is_stubbed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, prior_config, prepared, results_root, _ = _minimal_frozen_r2_fixture(
        tmp_path
    )
    unit_root = (
        results_root
        / "N1"
        / "fold_0"
        / "seed_20260830"
        / "R2_MS_MEAN_CE"
    )
    (unit_root / "features_logits_scores.npz").write_bytes(b"tampered")
    monkeypatch.setattr(
        runner,
        "_verify_artifact_hash_manifest",
        lambda _root: {"legacy_check": "stubbed"},
    )

    with pytest.raises(
        DataValidationError,
        match="frozen R2 expected artifact changed: N1/features_logits_scores.npz",
    ):
        load_and_audit_frozen_r2(
            project_root=PROJECT_ROOT,
            r2_results_root=results_root,
            pair_id="N1",
            config=config,
            prepared=prepared,
            prior_config=prior_config,
            device=torch.device("cpu"),
        )


def _prediction_rows() -> list[dict[str, Any]]:
    known = (
        (0, 0, 0.1),
        (1, 1, 0.2),
        (2, 2, 0.3),
        (3, 0, 0.4),
        (4, 4, 0.9),
    )
    unknown = (
        (0, 0.5),
        (1, 0.6),
        (2, 0.7),
        (3, 0.8),
        (4, 1.0),
    )
    rows: list[dict[str, Any]] = []
    for true_label, prediction, score in known:
        row: dict[str, Any] = {
            "evaluation_role": "known_calibration",
            "true_label": true_label,
            "predicted_known_label": prediction,
        }
        for rule_index, rule in enumerate(SCORE_RULES):
            row[f"{rule.lower()}_unknown_score"] = score + 10.0 * rule_index
        rows.append(row)
    for prediction, score in unknown:
        row = {
            "evaluation_role": "surrogate_unknown",
            "true_label": 5,
            "predicted_known_label": prediction,
        }
        for rule_index, rule in enumerate(SCORE_RULES):
            row[f"{rule.lower()}_unknown_score"] = score + 10.0 * rule_index
        rows.append(row)
    return rows


def test_prediction_rows_recompute_nine_metric_wide_table() -> None:
    metrics = recompute_metrics_from_prediction_rows(
        _prediction_rows(), known_class_count=5, known_acceptance_rate=0.8
    )

    assert tuple(metrics) == SCORE_RULES
    assert len(REPORT_METRIC_KEYS) == 9
    for rule_index, rule in enumerate(SCORE_RULES):
        values = metrics[rule]
        assert set(REPORT_METRIC_KEYS).issubset(values)
        assert values["known_accuracy"] == pytest.approx(0.8)
        assert values["threshold"] == pytest.approx(0.4 + 10.0 * rule_index)
        assert values["known_correct_acceptance_rate"] == pytest.approx(0.6)
        assert values["unknown_rejection_rate"] == pytest.approx(1.0)
        assert values["open_set_harmonic_score"] == pytest.approx(0.75)

    wide_rows = _metric_rows_from_audits(
        [{"pair_id": "N1", "metrics": metrics}]
    )
    assert len(wide_rows) == 5
    for row, rule in zip(wide_rows, SCORE_RULES, strict=True):
        assert row["pair_id"] == "N1"
        assert row["method"] == rule
        assert all(metric in row for metric in REPORT_METRIC_KEYS)
        assert row["threshold"] == pytest.approx(metrics[rule]["threshold"])


def test_confirmation_is_blocked_before_any_experiment_work_without_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_work(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("confirmation performed experiment work before authorization")

    monkeypatch.setattr(runner, "_load_bundle", forbidden_work)
    with pytest.raises(DataValidationError, match="audited pilot root"):
        runner.run_unit(
            CONFIG_PATH,
            tmp_path / "bundle",
            tmp_path / "r2",
            tmp_path / "confirmation",
            phase="confirmation",
            pair_id="N0",
            pilot_root=None,
        )


def test_no_signal_pilot_gate_cannot_authorize_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runner,
        "audit_phase_root",
        lambda *_args, **_kwargs: {"status": "passed"},
    )
    monkeypatch.setattr(
        runner,
        "_read_json",
        lambda _path: {
            "signal": "no_cssr_signal",
            "selected_rule": None,
            "confirmation_allowed": False,
        },
    )
    with pytest.raises(DataValidationError, match="does not authorize confirmation"):
        runner._read_authorized_pilot(
            tmp_path / "pilot",
            load_fg_mv_cssr_config(CONFIG_PATH),
        )


def test_numerical_runtime_reapplies_frozen_cuda_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_fg_mv_cssr_config(CONFIG_PATH)
    original_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    original_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    original_benchmark = torch.backends.cudnn.benchmark
    original_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)

        observed = runner._configure_numerical_runtime(config)

        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
        assert torch.backends.cudnn.benchmark is False
        assert torch.are_deterministic_algorithms_enabled() is True
        assert observed == {
            "deterministic_algorithms": True,
            "tf32": False,
            "cudnn_benchmark": False,
        }
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_matmul_tf32
        torch.backends.cudnn.allow_tf32 = original_cudnn_tf32
        torch.backends.cudnn.benchmark = original_benchmark
        torch.use_deterministic_algorithms(original_deterministic)


def test_unit_audit_reapplies_numerical_runtime_before_artifact_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_fg_mv_cssr_config(CONFIG_PATH)
    calls: list[Mapping[str, Any]] = []
    monkeypatch.setattr(
        runner,
        "_configure_numerical_runtime",
        lambda observed: calls.append(observed),
    )

    with pytest.raises(DataValidationError, match="missing files"):
        runner.audit_unit_result(
            tmp_path,
            config=config,
            phase="smoke",
            pair_id="N1",
        )

    assert calls == [config]
