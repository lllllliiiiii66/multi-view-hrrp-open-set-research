from __future__ import annotations

import copy
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
import yaml


torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from hrrp_osr.data.errors import DataConfigError, DataValidationError  # noqa: E402
from hrrp_osr.data.manifest import file_sha256  # noqa: E402
from hrrp_osr.models.cssr_e2e_1d import FGMVCSSRE2EModel  # noqa: E402
from hrrp_osr.models.ms_mean_factorial import (  # noqa: E402
    MSMeanHeadFactorialModel,
    clone_state_dict,
)
import hrrp_osr.training.cssr_gradient_pathology_audit as gradient_audit_module  # noqa: E402
from hrrp_osr.training.cssr_gradient_pathology_audit import (  # noqa: E402
    AUDIT_EPOCHS,
    AUDIT_METHOD,
    AUDIT_PAIRS,
    AUDIT_SEED,
    EXPECTED_EPOCH0_R2_STATE_SHA256,
    EXPECTED_Q2_AE_INITIAL_SHA256,
    EXPECTED_R2_CHECKPOINT_SHA256,
    EXPECTED_R2_PAIR_MANIFEST_SHA256,
    LEGACY_CONFIG_SHA256,
    PARAMETER_GROUP_NAMES,
    AtomicGradientAuditSink,
    _audit_artifact_hashes,
    _finalize_marker,
    _parser,
    _parameter_groups,
    _record_batch_diagnostics,
    _render_jsonl,
    _unit_destination,
    aggregate_gradient_audit_phase,
    annotate_original_gate,
    audit_gradient_audit_phase,
    audit_task_source_hashes,
    audit_gradient_audit_unit,
    build_gradient_audit_plan,
    classify_gradient_pathology,
    clip_diagnostics,
    gradient_pair_statistics,
    load_gradient_pathology_config,
    load_legacy_q2_config,
    relative_parameter_updates,
    snapshot_parameter_groups,
    summarize_gradient_epoch,
)
from hrrp_osr.training.fg_mv_cssr_e2e_redesign import (  # noqa: E402
    _build_optimizer,
    _learning_rate_factor,
    _set_optimizer_lrs,
)
from hrrp_osr.training.arpl_pilot import _set_determinism  # noqa: E402
from hrrp_osr.training.fg_mv_cssr_pilot import (  # noqa: E402
    _atomic_write_bytes,
    _read_json,
    _sequence_sha256,
    _write_csv,
    _write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/experiments/cssr/fg_mv_cssr_decoupled_audit_v3.yaml"
)


def _write_changed_config(tmp_path: Path, config: Mapping[str, Any]) -> Path:
    path = tmp_path / "changed.yaml"
    path.write_text(
        yaml.safe_dump(dict(config), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _epoch(
    epoch: int,
    *,
    small_fraction: float = 0.0,
    ratio_b: float = 1.0,
    ratio_c: float = 1.0,
    relative_median: float = 1.0,
    clipping: float = 0.0,
    update: float = 0.001,
    cosine_median: float | None = 0.0,
    cosine_negative: float = 0.0,
    accuracy: float = 0.8,
    nll: float = 0.5,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "batch_count": 10,
        "mean_of_batch_ratios": ratio_b,
        "ratio_of_mean_norms": ratio_b,
        "rms_norm_ratio": ratio_c,
        "weighted_relative_gradient_norm_median": relative_median,
        "classification_gradient_below_1e-04_fraction": small_fraction,
        "gradient_clipping_fraction": clipping,
        "cosine_median": cosine_median,
        "cosine_negative_fraction": cosine_negative,
        "parameter_relative_updates": {
            name: update for name in PARAMETER_GROUP_NAMES
        },
        "calibration": {"accuracy": accuracy, "nll": nll},
    }


def test_config_and_plan_freeze_the_two_five_epoch_audit_units() -> None:
    config = load_gradient_pathology_config(CONFIG_PATH)
    plan = build_gradient_audit_plan(config)

    assert [(row["pair_id"], row["method"]) for row in plan] == [
        ("N1", AUDIT_METHOD),
        ("N4", AUDIT_METHOD),
    ]
    assert all(row["epochs"] == 5 and row["audit_seed"] == AUDIT_SEED for row in plan)
    assert all(row["performance_gate_eligible"] is False for row in plan)
    assert all(row["final_unknown_test_authorized"] is False for row in plan)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda config: config["legacy_e2e_q2"].update(audit_epochs=6),
        lambda config: config["legacy_e2e_q2"].update(audit_pairs=["N4", "N1"]),
        lambda config: config["legacy_e2e_q2"].update(source_config_sha256="changed"),
        lambda config: config["prior_r2"]["unit_artifact_hashes"]["N1"].update(
            {"pair_manifest.csv": "changed"}
        ),
        lambda config: config["gradient_pathology_audit"].update(
            denominator_floor=1.0e-9
        ),
        lambda config: config["gradient_pathology_audit"].update(
            disable_only_original_100x_exception=False
        ),
        lambda config: config["gradient_pathology_audit"]["label_thresholds"].update(
            frequent_clipping_fraction=0.4
        ),
        lambda config: config["evidence_scope"].update(even_angle_test_used=True),
        lambda config: config["outputs"].update(
            final_unknown_test_authorized=True
        ),
    ),
)
def test_config_rejects_stage_a_protocol_mutations(
    tmp_path: Path, mutate: Any
) -> None:
    config = copy.deepcopy(yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")))
    mutate(config)
    with pytest.raises(DataConfigError):
        load_gradient_pathology_config(_write_changed_config(tmp_path, config))


def test_gradient_pair_statistics_match_an_ordered_hand_calculation() -> None:
    actual = gradient_pair_statistics(
        (torch.tensor([1.0, 0.0]), None),
        (torch.tensor([-2.0, 2.0]), None),
        reference=torch.tensor(0.0),
    )

    assert actual["classification_gradient_norm"] == pytest.approx(1.0)
    assert actual["relative_raw_gradient_norm"] == pytest.approx(math.sqrt(8.0))
    assert actual["relative_weighted_gradient_norm"] == pytest.approx(math.sqrt(2.0))
    assert actual["weighted_relative_to_classification_ratio"] == pytest.approx(
        math.sqrt(2.0)
    )
    assert actual["classification_weighted_relative_dot"] == pytest.approx(-1.0)
    assert actual["classification_weighted_relative_cosine"] == pytest.approx(
        -1.0 / math.sqrt(2.0)
    )


def test_gradient_cosine_is_null_at_the_frozen_norm_product_floor() -> None:
    actual = gradient_pair_statistics(
        (torch.zeros(2),),
        (torch.ones(2),),
        reference=torch.tensor(0.0),
    )
    assert actual["classification_weighted_relative_cosine"] is None
    assert actual["classification_weighted_relative_dot"] == 0.0
    assert actual["weighted_relative_to_classification_ratio"] == pytest.approx(
        math.sqrt(0.5) / 1.0e-12
    )


def test_clip_diagnostics_use_the_preregistered_pytorch_formula() -> None:
    assert clip_diagnostics(10.0, 5.0) == {
        "pre_clip_total_gradient_norm": 10.0,
        "gradient_clipping_scale": pytest.approx(5.0 / (10.0 + 1.0e-6)),
        "post_clip_estimated_gradient_norm": pytest.approx(
            10.0 * 5.0 / (10.0 + 1.0e-6)
        ),
        "gradient_clipped": True,
    }
    assert clip_diagnostics(5.0, 5.0)["gradient_clipped"] is False


def test_epoch_summary_uses_all_three_ratios_and_linear_quantiles() -> None:
    rows = [
        {
            "classification_gradient_norm": 1.0,
            "relative_weighted_gradient_norm": 2.0,
            "weighted_relative_to_classification_ratio": 2.0,
            "classification_weighted_relative_cosine": -1.0,
            "gradient_clipped": False,
        },
        {
            "classification_gradient_norm": 2.0,
            "relative_weighted_gradient_norm": 8.0,
            "weighted_relative_to_classification_ratio": 4.0,
            "classification_weighted_relative_cosine": None,
            "gradient_clipped": True,
        },
    ]
    summary = summarize_gradient_epoch(rows)

    assert summary["mean_of_batch_ratios"] == pytest.approx(3.0)
    assert summary["ratio_of_mean_norms"] == pytest.approx(10.0 / 3.0)
    assert summary["rms_norm_ratio"] == pytest.approx(math.sqrt(13.6))
    assert summary["ratio_median"] == pytest.approx(3.0)
    assert summary["ratio_p90"] == pytest.approx(3.8)
    assert summary["ratio_p95"] == pytest.approx(3.9)
    assert summary["cosine_mean"] == pytest.approx(-1.0)
    assert summary["cosine_negative_fraction"] == pytest.approx(1.0)
    assert summary["cosine_undefined_fraction"] == pytest.approx(0.5)
    assert summary["gradient_clipping_fraction"] == pytest.approx(0.5)

    all_undefined = summarize_gradient_epoch(
        [
            {
                **row,
                "classification_weighted_relative_cosine": None,
            }
            for row in rows
        ]
    )
    assert all_undefined["cosine_mean"] is None
    assert all_undefined["cosine_median"] is None
    assert all_undefined["cosine_positive_fraction"] is None
    assert all_undefined["cosine_negative_fraction"] is None
    assert all_undefined["cosine_undefined_fraction"] == 1.0


def test_parameter_update_uses_the_frozen_relative_l2_definition() -> None:
    parameter = nn.Parameter(torch.tensor([3.0, 4.0]))
    groups = {"group": (parameter,)}
    before = snapshot_parameter_groups(groups)
    with torch.no_grad():
        parameter.mul_(2.0)
    assert relative_parameter_updates(before, groups)["group"] == pytest.approx(1.0)


def test_original_gate_is_strictly_greater_than_100_and_consecutive() -> None:
    rows = [
        {"epoch": 1, "mean_of_batch_ratios": 100.0},
        {"epoch": 2, "mean_of_batch_ratios": 101.0},
        {"epoch": 3, "mean_of_batch_ratios": 102.0},
        {"epoch": 4, "mean_of_batch_ratios": 103.0},
        {"epoch": 5, "mean_of_batch_ratios": 1.0},
    ]
    result = annotate_original_gate(rows)

    assert result["would_have_triggered_original_100x_gate"] is True
    assert result["first_original_100x_trigger_epoch"] == 4
    assert result["epochs"][0]["original_100x_epoch_violation"] is False
    assert result["epochs"][3]["would_have_triggered_original_100x_gate"] is True
    assert result["epochs"][4]["original_100x_violation_streak"] == 0


@pytest.mark.parametrize(
    ("rows", "numerical_anomaly", "expected"),
    (
        (
            [_epoch(index, small_fraction=0.6) for index in range(1, 6)],
            False,
            "ratio_denominator_collapse_likely",
        ),
        (
            [
                _epoch(index, small_fraction=0.6, clipping=0.5 if index <= 3 else 0.0)
                for index in range(1, 6)
            ],
            False,
            "mixed_gradient_conflict",
        ),
        (
            [
                _epoch(
                    index,
                    ratio_b=101.0 if index <= 3 else 1.0,
                    relative_median=5.0 if index <= 3 else 1.0,
                )
                for index in range(1, 6)
            ],
            False,
            "true_auxiliary_domination",
        ),
        (
            [_epoch(index) for index in range(1, 6)],
            False,
            "inconclusive",
        ),
        (
            [_epoch(index, small_fraction=0.6) for index in range(1, 6)],
            True,
            "inconclusive",
        ),
    ),
)
def test_pathology_labels_follow_the_frozen_priority(
    rows: list[dict[str, Any]], numerical_anomaly: bool, expected: str
) -> None:
    result = classify_gradient_pathology(
        rows,
        epoch0_calibration={"accuracy": 0.8, "nll": 0.5},
        numerical_anomaly=numerical_anomaly,
    )
    assert result["label"] == expected
    assert result["performance_gate_eligible"] is False
    assert result["stage_b_decision"] == "continue_if_no_code_or_numerical_failure"
    assert result["final_unknown_test_authorized"] is False


class _ToyQ2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stage = nn.Linear(3, 4, bias=False)
        self.projection = nn.Linear(4, 2, bias=False)
        self.head = nn.Linear(2, 2, bias=False)
        self.ae = nn.Linear(4, 4, bias=False)
        self.dropout = nn.Dropout(0.25)

    def components(self, inputs: torch.Tensor, labels: torch.Tensor) -> Any:
        features = self.dropout(self.stage(inputs))
        logits = self.head(self.projection(features))
        classification = torch.nn.functional.cross_entropy(logits, labels)
        relative = (self.ae(features) - features).square().mean()
        return SimpleNamespace(fused_logits=logits), SimpleNamespace(
            classification_loss=classification,
            relative_loss=relative,
            total_loss=classification + 0.5 * relative,
        )

    def groups(self) -> dict[str, tuple[nn.Parameter, ...]]:
        return {
            "last_residual_stage": tuple(self.stage.parameters()),
            "projection": tuple(self.projection.parameters()),
            "ce_head": tuple(self.head.parameters()),
            "cssr_autoencoders": tuple(self.ae.parameters()),
        }


def _assert_nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, Mapping):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def test_extra_gradient_reads_do_not_change_the_original_optimizer_update() -> None:
    torch.manual_seed(71)
    baseline = _ToyQ2()
    audited = copy.deepcopy(baseline)
    baseline_optimizer = torch.optim.AdamW(baseline.parameters(), lr=1.0e-3)
    audited_optimizer = torch.optim.AdamW(audited.parameters(), lr=1.0e-3)
    inputs = torch.randn(4, 3, generator=torch.Generator().manual_seed(19))
    labels = torch.tensor([0, 1, 1, 0])

    torch.manual_seed(983)
    baseline_optimizer.zero_grad(set_to_none=True)
    _, baseline_losses = baseline.components(inputs, labels)
    baseline_losses.total_loss.backward()
    torch.nn.utils.clip_grad_norm_(baseline.parameters(), max_norm=0.5)
    baseline_optimizer.step()
    baseline_rng = torch.random.get_rng_state().clone()

    torch.manual_seed(983)
    audited_optimizer.zero_grad(set_to_none=True)
    audited_output, audited_losses = audited.components(inputs, labels)
    row, _ = _record_batch_diagnostics(
        model=audited,  # type: ignore[arg-type]
        losses=audited_losses,
        output=audited_output,
        labels=labels,
        groups=audited.groups(),
        epoch=1,
        batch_index=1,
        batch_start=0,
        pair_ids=("p0", "p1", "p2", "p3"),
        clip_norm=0.5,
    )
    audited_optimizer.step()
    audited_rng = torch.random.get_rng_state().clone()

    _assert_nested_equal(baseline.state_dict(), audited.state_dict())
    _assert_nested_equal(
        baseline_optimizer.state_dict(), audited_optimizer.state_dict()
    )
    assert torch.equal(baseline_rng, audited_rng)
    assert row["relative_projection_gradient_norm"] <= 1.0e-12
    assert row["relative_ce_head_gradient_norm"] <= 1.0e-12
    assert set(row["total_gradient_norm_by_parameter_group"]) == set(
        PARAMETER_GROUP_NAMES
    )


def _assert_real_q2_single_batch_equivalence(device: torch.device) -> None:
    torch.manual_seed(211)
    source = MSMeanHeadFactorialModel("R2_MS_MEAN_CE", known_class_count=5)
    r2_state = clone_state_dict(source.state_dict())
    baseline = FGMVCSSRE2EModel.from_r2_state_dict(
        r2_state,
        AUDIT_METHOD,
        known_class_count=5,
        autoencoder_seed=AUDIT_SEED,
    ).to(device)
    audited = FGMVCSSRE2EModel.from_r2_state_dict(
        r2_state,
        AUDIT_METHOD,
        known_class_count=5,
        autoencoder_seed=AUDIT_SEED,
    ).to(device)
    legacy_config = load_legacy_q2_config(PROJECT_ROOT)
    baseline_optimizer = _build_optimizer(baseline, legacy_config)
    audited_optimizer = _build_optimizer(audited, legacy_config)
    factor = _learning_rate_factor(
        1,
        warmup_epochs=int(legacy_config["training"]["warmup_epochs"]),
        total_epochs=int(legacy_config["training"]["epochs"]),
    )
    assert _set_optimizer_lrs(baseline_optimizer, factor) == _set_optimizer_lrs(
        audited_optimizer, factor
    )
    inputs = torch.randn(
        2, 2, 601, generator=torch.Generator().manual_seed(37)
    ).to(device)
    labels = torch.tensor([0, 4], device=device)
    baseline.train()
    audited.train()

    _set_determinism(AUDIT_SEED, True)
    baseline_optimizer.zero_grad(set_to_none=True)
    baseline_output = baseline(inputs)
    baseline_losses = baseline.loss(baseline_output, labels)
    baseline_losses.total_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in baseline.parameters() if parameter.requires_grad],
        max_norm=float(legacy_config["training"]["gradient_clip_norm"]),
    )
    baseline_optimizer.step()
    baseline_cpu_rng = torch.random.get_rng_state().clone()
    baseline_cuda_rng = (
        torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    )

    _set_determinism(AUDIT_SEED, True)
    audited_optimizer.zero_grad(set_to_none=True)
    audited_output = audited(inputs)
    audited_losses = audited.loss(audited_output, labels)
    row, _ = _record_batch_diagnostics(
        model=audited,
        losses=audited_losses,
        output=audited_output,
        labels=labels,
        groups=_parameter_groups(audited),
        epoch=1,
        batch_index=1,
        batch_start=0,
        pair_ids=("pair-a", "pair-b"),
        clip_norm=float(legacy_config["training"]["gradient_clip_norm"]),
    )
    audited_optimizer.step()
    audited_cpu_rng = torch.random.get_rng_state().clone()
    audited_cuda_rng = (
        torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    )

    _assert_nested_equal(baseline.state_dict(), audited.state_dict())
    _assert_nested_equal(
        baseline_optimizer.state_dict(), audited_optimizer.state_dict()
    )
    assert torch.equal(baseline_cpu_rng, audited_cpu_rng)
    if device.type == "cuda":
        assert torch.equal(baseline_cuda_rng, audited_cuda_rng)
    assert row["relative_projection_gradient_norm"] <= 1.0e-12
    assert row["relative_ce_head_gradient_norm"] <= 1.0e-12


def test_real_q2_cpu_diagnostics_preserve_state_optimizer_buffers_and_rng() -> None:
    _assert_real_q2_single_batch_equivalence(torch.device("cpu"))


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_name(torch.device("cuda")) != "NVIDIA GeForce RTX 4090",
    reason="the preregistered CUDA differential requires the formal RTX 4090",
)
def test_real_q2_4090_diagnostics_preserve_state_optimizer_and_rng() -> None:
    _assert_real_q2_single_batch_equivalence(torch.device("cuda"))


def test_atomic_sink_persists_latest_batches_and_failure_context(
    tmp_path: Path,
) -> None:
    sink = AtomicGradientAuditSink(tmp_path / "staging")
    sink.append_batch({"epoch": 1, "batch_index": 1, "value": 1.0})
    sink.append_batch({"epoch": 1, "batch_index": 2, "value": 2.0})
    pending = {"epoch": 1, "batch_index": 3, "value": float("inf")}
    first = KeyboardInterrupt("interrupted during optimizer update")
    sink.save_failure(first, pending_batch=pending)
    # The outer unit handler may save the same failure again; the pending batch
    # must survive that second atomic state write.
    sink.save_failure(first, wall_time_seconds=1.5)

    rows = [
        json.loads(line)
        for line in (sink.staging / "batch_diagnostics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["batch_index"] for row in rows] == [1, 2]
    assert _read_json(sink.staging / "failure_batch_diagnostic.json") == {
        "epoch": 1,
        "batch_index": 3,
        "value": None,
    }
    failure = _read_json(sink.staging / "failure_state.json")
    assert failure["status"] == "failed"
    assert failure["exception_type"] == "KeyboardInterrupt"
    assert failure["completed_batch_count"] == 2
    assert failure["pending_batch"]["batch_index"] == 3


def _synthetic_success_unit(
    root: Path, *, config: Mapping[str, Any], pair_id: str = "N1"
) -> tuple[str, ...]:
    root.mkdir(parents=True)
    source_hashes = audit_task_source_hashes(PROJECT_ROOT)
    _atomic_write_bytes(
        root / "resolved_config.yaml",
        yaml.safe_dump(dict(config), allow_unicode=True, sort_keys=False).encode("utf-8"),
    )
    _write_csv(root / "source_pair_manifest.csv", [{"pair_id": "source-pair"}])
    _write_json(
        root / "r2_reference_audit.json",
        {
            "status": "passed",
            "pair_id": pair_id,
            "checkpoint_sha256": EXPECTED_R2_CHECKPOINT_SHA256[pair_id],
            "pair_manifest_sha256": file_sha256(root / "source_pair_manifest.csv"),
            "strict_load": True,
            "old_outputs_exact": True,
            "all_parameters_frozen": True,
            "final_unknown_used": False,
            "even_angle_test_used": False,
        },
    )
    unique_rows: list[dict[str, Any]] = []
    for label in range(5):
        for index in range(144):
            frame = index % 24
            angle = 15 * frame + (1 if frame % 2 == 0 else 0)
            unique_rows.append(
                {
                    "experiment_role": "train_known",
                    "sample_id": f"train-{label}-{index:03d}",
                    "processed_row_index": len(unique_rows),
                    "class_name": f"known-{label}",
                    "model_label": label,
                    "angle_deg": angle,
                    "frame_id": frame,
                    "source_class_role": "known",
                }
            )
        for index in range(36):
            unique_rows.append(
                {
                    "experiment_role": "known_calibration",
                    "sample_id": f"calibration-{label}-{index:03d}",
                    "processed_row_index": len(unique_rows),
                    "class_name": f"known-{label}",
                    "model_label": label,
                    "angle_deg": 1,
                    "frame_id": 0,
                    "source_class_role": "known",
                }
            )
    for index in range(72):
        unique_rows.append(
            {
                "experiment_role": "surrogate_unknown",
                "sample_id": f"surrogate-{index:03d}",
                "processed_row_index": len(unique_rows),
                "class_name": "surrogate",
                "model_label": -1,
                "angle_deg": 1,
                "frame_id": 0,
                "source_class_role": "known",
            }
        )
    _write_csv(root / "unique_base_sample_manifest.csv", unique_rows)
    batch_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    schedule_hashes: list[str] = []
    gradient_groups = {
        "last_residual_stage": 1.25,
        "projection": 0.4,
        "ce_head": 0.3,
        "cssr_autoencoders": 0.5,
    }
    pre_clip = math.sqrt(sum(value**2 for value in gradient_groups.values()))
    for epoch in range(1, AUDIT_EPOCHS + 1):
        schedule: list[dict[str, Any]] = []
        for label in range(5):
            for index in range(144):
                right = (index + 1) % 144
                left_frame = index % 24
                right_frame = right % 24
                left_angle = 15 * left_frame + (1 if left_frame % 2 == 0 else 0)
                right_angle = 15 * right_frame + (1 if right_frame % 2 == 0 else 0)
                schedule.append(
                    {
                        "epoch": epoch,
                        "identity_pair_id": pair_id,
                        "pair_id": f"{pair_id}-e{epoch}-c{label}-p{index:03d}",
                        "model_label": label,
                        "view1_sample_id": f"train-{label}-{index:03d}",
                        "view2_sample_id": f"train-{label}-{right:03d}",
                        "view1_frame_id": left_frame,
                        "view2_frame_id": right_frame,
                        "view1_angle_deg": left_angle,
                        "view2_angle_deg": right_angle,
                    }
                )
        schedule_path = root / "pair_schedules" / f"epoch_{epoch:03d}.csv"
        _write_csv(schedule_path, schedule)
        schedule_hash = file_sha256(schedule_path)
        schedule_hashes.append(schedule_hash)
        _write_json(
            root / "pair_schedule_audits" / f"epoch_{epoch:03d}.json",
            {
                "status": "passed",
                "pair_id": pair_id,
                "epoch": epoch,
                "pair_count": 720,
                "all_constraints_passed": True,
                "epoch_manifest_sha256": schedule_hash,
            },
        )
        selected: list[dict[str, Any]] = []
        for batch_index, start in enumerate(range(0, 720, 64), start=1):
            size = min(64, 720 - start)
            selected.append(
                {
                "epoch": epoch,
                "batch_index": batch_index,
                "batch_start": start,
                "batch_size": size,
                "batch_pair_id_sequence_sha256": _sequence_sha256(
                    row["pair_id"] for row in schedule[start : start + size]
                ),
                "classification_gradient_norm": 1.0,
                "relative_raw_gradient_norm": 2.0,
                "relative_weighted_gradient_norm": 1.0,
                "weighted_relative_to_classification_ratio": 1.0,
                "classification_weighted_relative_dot": 0.25,
                "classification_weighted_relative_cosine": 0.25,
                "total_gradient_norm": gradient_groups["last_residual_stage"],
                "total_last_residual_stage_gradient_norm": gradient_groups[
                    "last_residual_stage"
                ],
                "total_gradient_norm_by_parameter_group": gradient_groups,
                "relative_projection_gradient_norm": 0.0,
                "relative_ce_head_gradient_norm": 0.0,
                "pre_clip_total_gradient_norm": pre_clip,
                "gradient_clipping_scale": 1.0,
                "post_clip_estimated_gradient_norm": pre_clip,
                "gradient_clipped": False,
                "post_clip_observed_gradient_norm": pre_clip,
                "clip_grad_norm_returned_pre_clip_norm": pre_clip,
                "classification_loss": 1.0,
                "relative_loss": 2.0,
                "total_loss": 2.0,
                "train_accuracy": 0.5,
                "ce_mean_max_confidence": 0.6,
                "original_100x_exception_suppressed": True,
                "final_unknown_used": False,
                "even_angle_test_used": False,
            }
            )
        batch_rows.extend(selected)
        epoch_rows.append(
            {
                "epoch": epoch,
                "method": AUDIT_METHOD,
                **summarize_gradient_epoch(selected),
                "parameter_relative_updates": {
                    name: 0.001 for name in PARAMETER_GROUP_NAMES
                },
                "calibration": {
                    "accuracy": 0.8,
                    "nll": 0.5,
                    "ece": 0.1,
                    "mean_max_logit": 1.0,
                    "mean_single_view_feature_norm": 2.0,
                    "mean_fused_feature_norm": 1.5,
                },
                "pair_schedule_sha256": schedule_hash,
                "performance_gate_eligible": False,
                "surrogate_unknown_used": False,
                "final_unknown_used": False,
                "even_angle_test_used": False,
            }
        )
    gate = annotate_original_gate(epoch_rows)
    epoch_rows = gate["epochs"]
    epoch0 = {
        "accuracy": 0.8,
        "nll": 0.5,
        "ece": 0.1,
        "mean_max_logit": 1.0,
        "mean_single_view_feature_norm": 2.0,
        "mean_fused_feature_norm": 1.5,
    }
    classification = classify_gradient_pathology(
        epoch_rows, epoch0_calibration=epoch0
    )
    _write_json(
        root / "unit_contract.json",
        {
            "experiment_id": "fg_mv_cssr_decoupled_audit_v3",
            "phase": "gradient_pathology_audit",
            "pair_id": pair_id,
            "method": AUDIT_METHOD,
            "angle_fold": 0,
            "model_seed": 20260830,
            "audit_seed": AUDIT_SEED,
            "epochs": AUDIT_EPOCHS,
            "config_sha256": config["_config_sha256"],
            "legacy_config_sha256": LEGACY_CONFIG_SHA256,
            "source_hashes": source_hashes,
            "performance_gate_eligible": False,
            "surrogate_unknown_used": False,
            "final_unknown_used": False,
            "even_angle_test_used": False,
        },
    )
    _atomic_write_bytes(root / "batch_diagnostics.jsonl", _render_jsonl(batch_rows))
    _atomic_write_bytes(root / "epoch_diagnostics.jsonl", _render_jsonl(epoch_rows))
    _write_json(root / "epoch0_calibration.json", epoch0)
    _write_json(
        root / "original_100x_gate.json",
        {key: value for key, value in gate.items() if key != "epochs"},
    )
    _write_json(root / "pathology_classification.json", classification)
    combined_schedule_hash = _sequence_sha256(schedule_hashes)
    _write_json(
        root / "training_audit.json",
        {
            "status": "passed",
            "pair_id": pair_id,
            "method": AUDIT_METHOD,
            "epochs": AUDIT_EPOCHS,
            "batch_count": 60,
            "schedule_sha256": combined_schedule_hash,
            "epoch_schedule_sha256": schedule_hashes,
            "epoch0_common_r2_state_sha256": EXPECTED_EPOCH0_R2_STATE_SHA256[
                pair_id
            ],
            "ae_initial_state_sha256": EXPECTED_Q2_AE_INITIAL_SHA256,
            "frozen_prefix_unchanged": True,
            "all_parameters_finite": True,
            "original_100x_exception_suppressed_only": True,
            "performance_gate_eligible": False,
            "surrogate_unknown_used": False,
            "final_unknown_used": False,
            "even_angle_test_used": False,
        },
    )
    torch.manual_seed(101)
    checkpoint_model = FGMVCSSRE2EModel(
        MSMeanHeadFactorialModel("R2_MS_MEAN_CE", known_class_count=5),
        AUDIT_METHOD,
        autoencoder_seed=AUDIT_SEED,
    )
    checkpoint = {
        "experiment_id": "fg_mv_cssr_decoupled_audit_v3",
        "phase": "gradient_pathology_audit",
        "pair_id": pair_id,
        "method": AUDIT_METHOD,
        "checkpoint_epoch": AUDIT_EPOCHS,
        "diagnostic_only": True,
        "model_state_dict": clone_state_dict(checkpoint_model.state_dict()),
        "config_sha256": config["_config_sha256"],
        "legacy_config_sha256": LEGACY_CONFIG_SHA256,
        "schedule_sha256": combined_schedule_hash,
        "final_unknown_used": False,
        "even_angle_test_used": False,
    }
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    _atomic_write_bytes(root / "checkpoint.pt", buffer.getvalue())
    _write_json(root / "environment.json", {"task_source_hashes": source_hashes})
    _write_json(
        root / "latest_state.json",
        {
            "status": "complete",
            "completed_epoch_count": AUDIT_EPOCHS,
            "completed_batch_count": 60,
        },
    )
    _write_json(
        root / "unit_summary.json",
        {
            "status": "complete",
            "experiment_id": "fg_mv_cssr_decoupled_audit_v3",
            "phase": "gradient_pathology_audit",
            "pair_id": pair_id,
            "method": AUDIT_METHOD,
            "epochs": AUDIT_EPOCHS,
            "diagnosis": classification,
            "original_gate": {
                key: value for key, value in gate.items() if key != "epochs"
            },
            "diagnostic_only": True,
            "performance_gate_eligible": False,
            "final_unknown_used": False,
            "even_angle_test_used": False,
        },
    )
    _finalize_marker(root, success=True, summary_name="unit_summary.json")
    return tuple(schedule_hashes)


def test_unit_audit_recomputes_batch_to_epoch_evidence_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_gradient_pathology_config(CONFIG_PATH)
    root = tmp_path / "unit"
    hashes = _synthetic_success_unit(root, config=config)
    monkeypatch.setitem(
        gradient_audit_module.EXPECTED_SCHEDULE_SHA256, "N1", hashes
    )
    monkeypatch.setitem(
        gradient_audit_module.EXPECTED_R2_PAIR_MANIFEST_SHA256,
        "N1",
        file_sha256(root / "source_pair_manifest.csv"),
    )

    audit = audit_gradient_audit_unit(root, config=config, pair_id="N1")
    assert audit["batch_count"] == 60
    assert audit["epoch_count"] == 5
    assert audit["batch_to_epoch_recomputation"] == "exact"
    assert audit["final_unknown_used"] is False
    assert audit["even_angle_test_used"] is False

    epoch_rows = [
        json.loads(line)
        for line in (root / "epoch_diagnostics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    epoch_rows[0]["ratio_p95"] = 999.0
    _atomic_write_bytes(root / "epoch_diagnostics.jsonl", _render_jsonl(epoch_rows))
    _finalize_marker(root, success=True, summary_name="unit_summary.json")
    with pytest.raises(DataValidationError, match="statistic"):
        audit_gradient_audit_unit(root, config=config, pair_id="N1")


def test_unit_audit_rejects_a_summary_only_false_success(tmp_path: Path) -> None:
    config = load_gradient_pathology_config(CONFIG_PATH)
    root = tmp_path / "incomplete"
    root.mkdir()
    _write_json(root / "unit_summary.json", {"status": "complete"})
    _finalize_marker(root, success=True, summary_name="unit_summary.json")

    with pytest.raises(DataValidationError, match="required artifacts"):
        audit_gradient_audit_unit(root, config=config, pair_id="N1")


def test_unit_audit_rejects_a_self_consistent_but_wrong_r2_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_gradient_pathology_config(CONFIG_PATH)
    root = tmp_path / "wrong-manifest"
    hashes = _synthetic_success_unit(root, config=config)
    original_manifest_hash = file_sha256(root / "source_pair_manifest.csv")
    monkeypatch.setitem(
        gradient_audit_module.EXPECTED_SCHEDULE_SHA256, "N1", hashes
    )
    monkeypatch.setitem(
        gradient_audit_module.EXPECTED_R2_PAIR_MANIFEST_SHA256,
        "N1",
        original_manifest_hash,
    )
    _write_csv(root / "source_pair_manifest.csv", [{"pair_id": "wrong-source"}])
    r2_audit = _read_json(root / "r2_reference_audit.json")
    r2_audit["pair_manifest_sha256"] = file_sha256(
        root / "source_pair_manifest.csv"
    )
    _write_json(root / "r2_reference_audit.json", r2_audit)
    _finalize_marker(root, success=True, summary_name="unit_summary.json")

    with pytest.raises(DataValidationError, match="R2 reference"):
        audit_gradient_audit_unit(root, config=config, pair_id="N1")


def test_unit_audit_rejects_arbitrary_finite_checkpoint_tensor_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_gradient_pathology_config(CONFIG_PATH)
    root = tmp_path / "wrong-checkpoint"
    hashes = _synthetic_success_unit(root, config=config)
    monkeypatch.setitem(
        gradient_audit_module.EXPECTED_SCHEDULE_SHA256, "N1", hashes
    )
    monkeypatch.setitem(
        gradient_audit_module.EXPECTED_R2_PAIR_MANIFEST_SHA256,
        "N1",
        file_sha256(root / "source_pair_manifest.csv"),
    )
    checkpoint = torch.load(root / "checkpoint.pt", map_location="cpu", weights_only=False)
    checkpoint["model_state_dict"] = {"weight": torch.ones(1)}
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    _atomic_write_bytes(root / "checkpoint.pt", buffer.getvalue())
    _finalize_marker(root, success=True, summary_name="unit_summary.json")

    with pytest.raises(DataValidationError, match="checkpoint"):
        audit_gradient_audit_unit(root, config=config, pair_id="N1")


def test_phase_aggregate_is_hash_sealed_and_reauditable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_gradient_pathology_config(CONFIG_PATH)
    hashes_by_pair: dict[str, tuple[str, ...]] = {}
    for pair_id in AUDIT_PAIRS:
        hashes_by_pair[pair_id] = _synthetic_success_unit(
            _unit_destination(tmp_path, pair_id),
            config=config,
            pair_id=pair_id,
        )
        monkeypatch.setitem(
            gradient_audit_module.EXPECTED_SCHEDULE_SHA256,
            pair_id,
            hashes_by_pair[pair_id],
        )
        monkeypatch.setitem(
            gradient_audit_module.EXPECTED_R2_PAIR_MANIFEST_SHA256,
            pair_id,
            file_sha256(
                _unit_destination(tmp_path, pair_id) / "source_pair_manifest.csv"
            ),
        )

    summary = aggregate_gradient_audit_phase(tmp_path, config=config)
    assert summary["stage_b_allowed"] is True
    assert summary["unit_count"] == 2
    assert (tmp_path / "_PHASE_SUCCESS.json").is_file()
    assert _read_json(tmp_path / "artifact_hashes.json") == _audit_artifact_hashes(
        tmp_path
    )
    assert audit_gradient_audit_phase(tmp_path, config=config) == summary
    with pytest.raises(DataValidationError, match="already aggregated"):
        aggregate_gradient_audit_phase(tmp_path, config=config)


def test_cli_and_output_namespace_cannot_request_final_test(tmp_path: Path) -> None:
    parser = _parser()
    command = next(action for action in parser._actions if action.dest == "command")
    assert set(command.choices) == {
        "load-config",
        "plan",
        "run-unit",
        "audit-unit",
        "aggregate",
        "audit-phase",
    }
    assert _unit_destination(tmp_path, "N4") == (
        tmp_path
        / "gradient_pathology_audit"
        / "N4"
        / "fold_0"
        / f"seed_{AUDIT_SEED}"
        / AUDIT_METHOD
    )
    assert AUDIT_PAIRS == ("N1", "N4")
