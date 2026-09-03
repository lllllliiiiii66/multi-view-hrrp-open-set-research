from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from hrrp_osr.data.errors import DataValidationError
from hrrp_osr.evaluation.ms_mean_factorial import REPORT_METRIC_KEYS
from hrrp_osr.training.fg_mv_cssr_e2e_redesign import (
    CSSR_METHODS,
    FINETUNE_SEED,
    Q0_FROZEN_R2_CC_MLS,
    Q1_CE_FINETUNE_CONTROL,
    Q2_E2E_REL_CSSR_1X1,
    Q3_E2E_ABSREL_CSSR_1X1,
    Q4_E2E_ABSREL_CSSR_LOCAL3,
    _learning_rate_factor,
    _group_gradient_norms,
    _parser,
    _phase_rows_from_audits,
    _unit_destination,
    recompute_method_metrics_from_prediction_rows,
)


def _metrics(value: float) -> dict[str, float]:
    return {
        **{key: float(value) for key in REPORT_METRIC_KEYS},
        "threshold_known_acceptance_target": 0.95,
        "threshold": 0.5,
        "known_acceptance_rate": 0.95,
    }


def _identity_rows(pair_id: str, method: str, value: float) -> list[dict[str, object]]:
    return [
        {
            "pair_id": pair_id,
            "method": method,
            "surrogate_identity": identity,
            **{key: value for key in REPORT_METRIC_KEYS},
            "threshold": 0.5,
        }
        for identity in ("unknown_a", "unknown_b")
    ]


def _audit(pair_id: str, method: str, value: float) -> dict[str, object]:
    return {
        "pair_id": pair_id,
        "method": method,
        "metrics": _metrics(value),
        "q0_metrics": _metrics(0.25),
        "identity_metrics": [
            *_identity_rows(pair_id, method, value),
            *_identity_rows(pair_id, Q0_FROZEN_R2_CC_MLS, 0.25),
        ],
        "schedule_sha256": "shared-schedule",
        "epoch0_common_r2_state_sha256": "shared-r2",
        "ae_initial_state_sha256": (
            "shared-1x1-ae"
            if method in (Q2_E2E_REL_CSSR_1X1, Q3_E2E_ABSREL_CSSR_1X1)
            else ("local3-ae" if method == Q4_E2E_ABSREL_CSSR_LOCAL3 else None)
        ),
    }


def test_learning_rate_keeps_all_twenty_optimizer_updates_positive() -> None:
    assert _learning_rate_factor(1, warmup_epochs=2, total_epochs=20) == 0.5
    assert _learning_rate_factor(2, warmup_epochs=2, total_epochs=20) == 1.0
    assert _learning_rate_factor(3, warmup_epochs=2, total_epochs=20) == 1.0
    assert _learning_rate_factor(20, warmup_epochs=2, total_epochs=20) == pytest.approx(
        0.5 * (1.0 + np.cos(17.0 * np.pi / 18.0))
    )
    assert _learning_rate_factor(20, warmup_epochs=2, total_epochs=20) > 0.0
    values = [
        _learning_rate_factor(epoch, warmup_epochs=2, total_epochs=20)
        for epoch in range(2, 21)
    ]
    assert values == sorted(values, reverse=True)
    with pytest.raises(DataValidationError):
        _learning_rate_factor(0, warmup_epochs=2, total_epochs=20)


def test_gradient_audit_reads_groups_without_populating_parameter_gradients() -> None:
    first = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    second = torch.nn.Parameter(torch.tensor([2.0]))
    loss = first.square().sum() + 2.0 * second.square().sum()
    norms = _group_gradient_norms(
        loss,
        {"last_stage": (first,), "autoencoders": (second,)},
        retain_graph=True,
    )
    assert norms["last_stage"] == pytest.approx(10.0)
    assert norms["autoencoders"] == pytest.approx(8.0)
    assert first.grad is None and second.grad is None
    loss.backward()
    assert first.grad is not None and second.grad is not None


def test_prediction_rows_recompute_all_nine_metrics_and_threshold() -> None:
    known = [
        {
            "evaluation_role": "known_calibration",
            "true_label": index,
            "predicted_known_label": index,
            "unknown_score": score,
        }
        for index, score in enumerate((0.05, 0.10, 0.15, 0.20, 0.25))
    ]
    unknown = [
        {
            "evaluation_role": "surrogate_unknown",
            "true_label": -1,
            "predicted_known_label": 0,
            "unknown_score": score,
        }
        for score in (0.7, 0.8, 0.9)
    ]
    metrics = recompute_method_metrics_from_prediction_rows([*known, *unknown])
    assert all(key in metrics for key in REPORT_METRIC_KEYS)
    assert metrics["threshold"] == pytest.approx(0.25)
    assert metrics["known_accuracy"] == pytest.approx(1.0)
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["unknown_rejection_rate"] == pytest.approx(1.0)


def test_phase_collection_is_plan_ordered_and_audits_common_initial_state() -> None:
    plan = [
        {"pair_id": "N1", "method": method}
        for method in (
            Q1_CE_FINETUNE_CONTROL,
            Q2_E2E_REL_CSSR_1X1,
            Q3_E2E_ABSREL_CSSR_1X1,
            Q4_E2E_ABSREL_CSSR_LOCAL3,
        )
    ]
    audits = [
        _audit("N1", method, 0.4 + index / 10.0)
        for index, method in enumerate(
            (
                Q1_CE_FINETUNE_CONTROL,
                Q2_E2E_REL_CSSR_1X1,
                Q3_E2E_ABSREL_CSSR_1X1,
                Q4_E2E_ABSREL_CSSR_LOCAL3,
            )
        )
    ]
    metrics, rows, identity_rows, integrity = _phase_rows_from_audits(
        list(reversed(audits)), plan=plan
    )
    assert tuple(metrics["N1"]) == (
        Q0_FROZEN_R2_CC_MLS,
        Q1_CE_FINETUNE_CONTROL,
        Q2_E2E_REL_CSSR_1X1,
        Q3_E2E_ABSREL_CSSR_1X1,
        Q4_E2E_ABSREL_CSSR_LOCAL3,
    )
    assert [(row["pair_id"], row["method"]) for row in rows] == [
        ("N1", method) for method in metrics["N1"]
    ]
    assert len(identity_rows) == 10
    assert integrity["shared_schedule_across_methods"] is True
    assert integrity["q2_q3_same_ae_initialization"] is True


@pytest.mark.parametrize(
    ("field", "changed"),
    (
        ("schedule_sha256", "different-schedule"),
        ("epoch0_common_r2_state_sha256", "different-r2"),
        ("ae_initial_state_sha256", "different-ae"),
    ),
)
def test_phase_collection_fails_closed_on_cross_method_drift(
    field: str, changed: str
) -> None:
    methods = (
        Q1_CE_FINETUNE_CONTROL,
        Q2_E2E_REL_CSSR_1X1,
        Q3_E2E_ABSREL_CSSR_1X1,
    )
    plan = [{"pair_id": "N1", "method": method} for method in methods]
    audits = [_audit("N1", method, 0.5) for method in methods]
    target = 2 if field == "ae_initial_state_sha256" else 1
    audits[target] = copy.deepcopy(audits[target])
    audits[target][field] = changed
    with pytest.raises(DataValidationError):
        _phase_rows_from_audits(audits, plan=plan)


def test_cli_exposes_only_the_six_audited_operations() -> None:
    parser = _parser()
    action = next(action for action in parser._actions if action.dest == "command")
    assert set(action.choices) == {
        "load-config",
        "plan",
        "run-unit",
        "audit-unit",
        "aggregate",
        "audit-phase",
    }


def test_unit_destination_names_pair_fold_seed_and_method(tmp_path: Path) -> None:
    path = _unit_destination(tmp_path, "N1", Q2_E2E_REL_CSSR_1X1)
    assert path == (
        tmp_path
        / "N1"
        / "fold_0"
        / f"seed_{FINETUNE_SEED}"
        / Q2_E2E_REL_CSSR_1X1
    )
    assert tuple(CSSR_METHODS) == (
        Q2_E2E_REL_CSSR_1X1,
        Q3_E2E_ABSREL_CSSR_1X1,
        Q4_E2E_ABSREL_CSSR_LOCAL3,
    )
