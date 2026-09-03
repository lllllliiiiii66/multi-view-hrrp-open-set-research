from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from hrrp_osr.data.errors import DataValidationError
from hrrp_osr.data.manifest import file_sha256
from hrrp_osr.evaluation.ms_mean_factorial import REPORT_METRIC_KEYS
import hrrp_osr.training.fg_mv_cssr_decoupled as runner
from hrrp_osr.training.fg_mv_cssr_decoupled_protocol import (
    CONFIRMATION_PAIRS,
    D0_R2_CLASS_CONDITIONAL_MLS,
    D1_DECOUPLED_REL_CSSR,
    D2_DECOUPLED_ABSREL_CSSR,
    PILOT_PAIRS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/experiments/cssr/fg_mv_cssr_decoupled_audit_v3.yaml"
)

_IDENTITIES = {
    "N0": ("CVN77", "DDG-112"),
    "N1": ("DDG-112", "迷你好望角型散货船"),
    "N2": ("油气轮MARVEL CRANE", "迷你好望角型散货船"),
    "N3": ("DDG-1000", "油气轮MARVEL CRANE"),
    "N4": ("DDG-1000", "集装箱船达飞罗尔多夫级"),
    "N5": ("爱达魔都号", "集装箱船达飞罗尔多夫级"),
    "N6": ("CVN77", "爱达魔都号"),
}
_DESTINATIONS = ("DDG-1000", "DDG-112", "known-2", "known-3", "known-4")
_GRADIENT_AUTHORIZATION = {
    "gradient_audit_root": "/frozen/gradient-audit",
    "phase_success_sha256": "gradient-phase-seal",
    "summary_sha256": "gradient-summary",
    "stage_b_allowed": True,
    "final_unknown_test_authorized": False,
}


def _metrics(delta: float) -> dict[str, float]:
    values = {
        "known_accuracy": 0.80,
        "known_macro_f1": 0.79,
        "auroc": 0.60 + delta,
        "oscr": 0.50 + delta,
        "fpr95": 0.20 - delta,
        "known_correct_acceptance_rate": 0.90,
        "unknown_rejection_rate": 0.65 + delta,
        "open_set_harmonic_score": 0.75 + delta,
        "k_plus_1_macro_f1": 0.70 + delta,
        "threshold": 0.42,
    }
    assert set(REPORT_METRIC_KEYS) <= set(values)
    return values


def _identity_rows(
    pair_id: str,
    method: str,
    *,
    delta: float,
) -> list[dict[str, Any]]:
    return [
        {
            "pair_id": pair_id,
            "method": method,
            "surrogate_identity": identity,
            **_metrics(delta),
        }
        for identity in _IDENTITIES[pair_id]
    ]


def _absorption_rows(
    pair_id: str,
    method: str,
    *,
    target_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for identity in _IDENTITIES[pair_id]:
        for destination in _DESTINATIONS:
            is_ddg_direction = (
                pair_id == "N1"
                and identity == "DDG-112"
                and destination == "DDG-1000"
            ) or (
                pair_id == "N4"
                and identity == "DDG-1000"
                and destination == "DDG-112"
            )
            count = target_count if is_ddg_direction else 0
            rows.append(
                {
                    "pair_id": pair_id,
                    "method": method,
                    "surrogate_identity": identity,
                    "absorbed_as_known_identity": destination,
                    "false_accept_count": count,
                    "total_surrogate_count": 36,
                    "total_false_accept_count": count,
                    "rate_over_all_surrogate": count / 36.0,
                    "composition_within_false_accepts": 1.0 if count else 0.0,
                }
            )
    return rows


def _unit_audit(
    *,
    phase: str,
    pair_id: str,
    method: str,
    method_deltas: Mapping[str, float],
    smoke_authorization: Mapping[str, Any] | None,
    confirmation_authorization: Mapping[str, Any] | None,
) -> dict[str, Any]:
    delta = float(method_deltas.get(method, 0.0))
    d0_metrics = _metrics(0.0)
    method_metrics = _metrics(delta)
    d0_identity = _identity_rows(
        pair_id, D0_R2_CLASS_CONDITIONAL_MLS, delta=0.0
    )
    method_identity = _identity_rows(pair_id, method, delta=delta)
    d0_absorption = _absorption_rows(
        pair_id, D0_R2_CLASS_CONDITIONAL_MLS, target_count=5
    )
    method_absorption = _absorption_rows(pair_id, method, target_count=4)
    return {
        "status": "passed",
        "phase": phase,
        "pair_id": pair_id,
        "method": method,
        "destination": f"/synthetic/{phase}/{pair_id}/{method}",
        "metrics": method_metrics,
        "d0_metrics": d0_metrics,
        "global_mls_background_metrics": _metrics(-0.02),
        "identity_metrics": [*method_identity, *d0_identity],
        "absorption_rows": [*method_absorption, *d0_absorption],
        "schedule_sha256": f"schedule-{pair_id}",
        "initial_state_sha256": f"initial-{pair_id}",
        "shared_r2_prediction_audit": {
            "known_logits_sha256": f"known-logits-{pair_id}",
            "known_prediction_sha256": f"known-prediction-{pair_id}",
            "surrogate_logits_sha256": f"surrogate-logits-{pair_id}",
            "surrogate_prediction_sha256": f"surrogate-prediction-{pair_id}",
        },
        "source_hashes": {"runner.py": "frozen-source"},
        "gradient_audit_authorization": dict(_GRADIENT_AUTHORIZATION),
        "smoke_authorization": (
            None if smoke_authorization is None else dict(smoke_authorization)
        ),
        "confirmation_authorization": (
            None
            if confirmation_authorization is None
            else dict(confirmation_authorization)
        ),
        "checkpoint_strict_load": True,
        "checkpoint_replay": "bitwise_exact",
        "metric_recomputation": "exact",
        "final_unknown_used": False,
        "even_angle_test_used": False,
    }


def _install_synthetic_unit_audits(
    monkeypatch: pytest.MonkeyPatch,
    *,
    method_deltas: Mapping[str, float] | None = None,
    smoke_authorization: Mapping[str, Any] | None = None,
    confirmation_authorization: Mapping[str, Any] | None = None,
    smoke_authorization_by_method: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    deltas = dict(method_deltas or {})

    def fake_audit_unit_result(
        unit_root: str | Path,
        *,
        config: Mapping[str, Any],
        phase: str,
        pair_id: str,
        method: str,
        device_request: str = "auto",
    ) -> dict[str, Any]:
        del unit_root, config, device_request
        unit_smoke_authorization = smoke_authorization
        if smoke_authorization_by_method is not None:
            unit_smoke_authorization = smoke_authorization_by_method[method]
        return _unit_audit(
            phase=phase,
            pair_id=pair_id,
            method=method,
            method_deltas=deltas,
            smoke_authorization=unit_smoke_authorization,
            confirmation_authorization=confirmation_authorization,
        )

    monkeypatch.setattr(runner, "audit_unit_result", fake_audit_unit_result)


def _aggregate_and_audit(
    root: Path,
    *,
    config: Mapping[str, Any],
    phase: str,
    pilot_root: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = runner.aggregate_phase_root(
        root,
        config=config,
        phase=phase,
        pilot_root=pilot_root,
        device_request="cpu",
    )
    audit = runner.audit_phase_root(
        root,
        config=config,
        phase=phase,
        pilot_root=pilot_root,
        device_request="cpu",
    )
    return summary, audit


def _reseal(root: Path) -> None:
    runner._write_json(root / "artifact_hashes.json", runner._artifact_hashes(root))
    runner._write_json(
        root / "_PHASE_SUCCESS.json",
        {
            "status": "complete",
            "phase_summary_sha256": file_sha256(root / "phase_summary.json"),
            "artifact_hashes_sha256": file_sha256(root / "artifact_hashes.json"),
        },
    )


def test_smoke_phase_save_audit_round_trip_and_resealed_tamper_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runner.load_fg_mv_cssr_decoupled_config(CONFIG_PATH)
    root = tmp_path / "smoke"
    _install_synthetic_unit_audits(monkeypatch)

    summary, audit = _aggregate_and_audit(root, config=config, phase="smoke")

    assert summary["decision"] == "diagnostic_smoke_only"
    assert summary["unit_count"] == 2
    assert summary["diagnostic_only"] is True
    assert summary["automatic_followon_authorized"] is False
    assert summary["final_unknown_test_authorized"] is False
    assert audit["status"] == "passed"
    assert audit["metric_recomputation"] == "exact"

    tampered = json.loads((root / "phase_summary.json").read_text(encoding="utf-8"))
    tampered["automatic_followon_authorized"] = True
    runner._write_json(root / "phase_summary.json", tampered)
    _reseal(root)
    with pytest.raises(DataValidationError, match="phase summary contract changed"):
        runner.audit_phase_root(
            root,
            config=config,
            phase="smoke",
            pilot_root=None,
            device_request="cpu",
        )


def test_smoke_pilot_confirmation_authorization_chain_is_sealed_and_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runner.load_fg_mv_cssr_decoupled_config(CONFIG_PATH)

    smoke_root = tmp_path / "smoke"
    _install_synthetic_unit_audits(monkeypatch)
    _aggregate_and_audit(smoke_root, config=config, phase="smoke")
    smoke_authorization = runner._read_authorized_smoke(smoke_root, config)

    pilot_root = tmp_path / "pilot"
    _install_synthetic_unit_audits(
        monkeypatch,
        method_deltas={
            D1_DECOUPLED_REL_CSSR: 0.03,
            D2_DECOUPLED_ABSREL_CSSR: 0.01,
        },
        smoke_authorization=smoke_authorization,
    )
    pilot_summary, pilot_audit = _aggregate_and_audit(
        pilot_root, config=config, phase="pilot"
    )
    assert pilot_summary["pair_ids"] == list(PILOT_PAIRS)
    assert pilot_summary["gate"]["selected_method"] == D1_DECOUPLED_REL_CSSR
    assert pilot_summary["automatic_followon_authorized"] is True
    assert pilot_audit["status"] == "passed"
    pilot_authorization = runner._read_authorized_pilot(pilot_root, config)
    assert pilot_authorization["selected_method"] == D1_DECOUPLED_REL_CSSR
    assert pilot_authorization["smoke_authorization"] == smoke_authorization

    confirmation_root = tmp_path / "confirmation"
    _install_synthetic_unit_audits(
        monkeypatch,
        method_deltas={D1_DECOUPLED_REL_CSSR: 0.03},
        confirmation_authorization=pilot_authorization,
    )
    confirmation_summary, confirmation_audit = _aggregate_and_audit(
        confirmation_root,
        config=config,
        phase="confirmation",
        pilot_root=pilot_root,
    )

    assert confirmation_summary["pair_ids"] == list(CONFIRMATION_PAIRS)
    assert confirmation_summary["unit_count"] == 4
    assert confirmation_summary["decision"] == "decoupled_cssr_worth_full_validation"
    assert confirmation_summary["automatic_followon_authorized"] is False
    assert confirmation_summary["final_unknown_test_authorized"] is False
    assert confirmation_audit["authorization"] == pilot_authorization
    task_plan = json.loads(
        (confirmation_root / "task_plan.json").read_text(encoding="utf-8")
    )
    assert {(row["pair_id"], row["method"]) for row in task_plan["units"]} == {
        (pair_id, D1_DECOUPLED_REL_CSSR) for pair_id in CONFIRMATION_PAIRS
    }

    smoke_summary = json.loads(
        (smoke_root / "phase_summary.json").read_text(encoding="utf-8")
    )
    smoke_summary["decision"] = "tampered"
    runner._write_json(smoke_root / "phase_summary.json", smoke_summary)
    with pytest.raises(DataValidationError, match="smoke phase success seal is invalid"):
        runner._read_authorized_pilot(pilot_root, config)


def test_failed_pilot_gate_cannot_authorize_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runner.load_fg_mv_cssr_decoupled_config(CONFIG_PATH)

    smoke_root = tmp_path / "smoke"
    _install_synthetic_unit_audits(monkeypatch)
    runner.aggregate_phase_root(
        smoke_root, config=config, phase="smoke", device_request="cpu"
    )
    smoke_authorization = runner._read_authorized_smoke(smoke_root, config)

    pilot_root = tmp_path / "pilot"
    _install_synthetic_unit_audits(
        monkeypatch,
        method_deltas={
            D1_DECOUPLED_REL_CSSR: 0.0,
            D2_DECOUPLED_ABSREL_CSSR: 0.0,
        },
        smoke_authorization=smoke_authorization,
    )
    pilot_summary, pilot_audit = _aggregate_and_audit(
        pilot_root, config=config, phase="pilot"
    )
    assert pilot_summary["decision"] == "decoupled_cssr_failed"
    assert pilot_summary["automatic_followon_authorized"] is False
    assert pilot_audit["status"] == "passed"

    with pytest.raises(
        DataValidationError, match="does not authorize confirmation"
    ):
        runner._read_authorized_pilot(pilot_root, config)
    with pytest.raises(
        DataValidationError, match="does not authorize confirmation"
    ):
        runner.aggregate_phase_root(
            tmp_path / "confirmation",
            config=config,
            phase="confirmation",
            pilot_root=pilot_root,
            device_request="cpu",
        )


def test_pilot_aggregation_rejects_mixed_smoke_authorizations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runner.load_fg_mv_cssr_decoupled_config(CONFIG_PATH)
    first = {
        "smoke_root": "/smoke/a",
        "phase_success_sha256": "a",
        "phase_summary_sha256": "a",
        "status": "passed",
        "final_unknown_test_authorized": False,
    }
    second = {**first, "smoke_root": "/smoke/b"}
    _install_synthetic_unit_audits(
        monkeypatch,
        method_deltas={
            D1_DECOUPLED_REL_CSSR: 0.03,
            D2_DECOUPLED_ABSREL_CSSR: 0.01,
        },
        smoke_authorization_by_method={
            D1_DECOUPLED_REL_CSSR: first,
            D2_DECOUPLED_ABSREL_CSSR: second,
        },
    )

    with pytest.raises(DataValidationError, match="one audited smoke authorization"):
        runner.aggregate_phase_root(
            tmp_path / "pilot",
            config=config,
            phase="pilot",
            device_request="cpu",
        )


def test_confirmation_rejects_unit_authorization_from_a_different_pilot_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runner.load_fg_mv_cssr_decoupled_config(CONFIG_PATH)
    expected_authorization = {
        "pilot_root": str((tmp_path / "selected-pilot").resolve()),
        "pilot_gate_sha256": "selected-gate",
        "selected_method": D1_DECOUPLED_REL_CSSR,
        "decision": "decoupled_relative_signal",
        "smoke_authorization": {
            "smoke_root": "/smoke/selected",
            "phase_success_sha256": "smoke-seal",
            "phase_summary_sha256": "smoke-summary",
            "status": "passed",
            "final_unknown_test_authorized": False,
        },
    }
    wrong_authorization = {
        **expected_authorization,
        "pilot_root": str((tmp_path / "different-pilot").resolve()),
        "pilot_gate_sha256": "different-gate",
    }

    def fake_read_authorized_pilot(
        pilot_root: str | Path, config_value: Mapping[str, Any]
    ) -> dict[str, Any]:
        del config_value
        assert Path(pilot_root).resolve() == Path(
            expected_authorization["pilot_root"]
        )
        return dict(expected_authorization)

    monkeypatch.setattr(runner, "_read_authorized_pilot", fake_read_authorized_pilot)
    _install_synthetic_unit_audits(
        monkeypatch,
        method_deltas={D1_DECOUPLED_REL_CSSR: 0.03},
        confirmation_authorization=wrong_authorization,
    )

    with pytest.raises(DataValidationError, match="confirmation.*authorization"):
        runner.aggregate_phase_root(
            tmp_path / "confirmation",
            config=config,
            phase="confirmation",
            pilot_root=expected_authorization["pilot_root"],
            device_request="cpu",
        )
