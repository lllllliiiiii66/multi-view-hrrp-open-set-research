from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hrrp_osr.data.errors import DataValidationError  # noqa: E402
from hrrp_osr.data.processed import ProcessedBundle  # noqa: E402
from hrrp_osr.models.arpl import (  # noqa: E402
    TwoViewARPLClassifier,
    TwoViewCEClassifier,
)
from hrrp_osr.training.arpl_mv_evidence import (  # noqa: E402
    KnownCalibrationECDF,
    fit_and_apply_evidence_ecdfs,
    load_mv_evidence_config,
    raw_view_evidence,
    require_confirmation_gate,
    select_development_fusion,
    view_auxiliary_loss,
)
from hrrp_osr.training.arpl_pilot import SOURCE_KNOWN_ORDER, prepare_surrogate_split  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_mv_evidence_config_freezes_scope_and_confirmation() -> None:
    config = load_mv_evidence_config(
        PROJECT_ROOT
        / "configs/experiments/arpl/arpl_mv_evidence_surrogate_v1.yaml"
    )
    assert config["model"]["lambda_view"] == 0.5
    assert config["evidence_scope"]["final_unknown_classes_used"] is False
    assert config["evidence_scope"]["even_angle_test_used"] is False
    assert [row["split_id"] for row in config["classes"]["confirmation_splits"]] == [
        "C0",
        "C1",
        "C2",
        "C3",
    ]


@pytest.mark.parametrize("model_type", [TwoViewCEClassifier, TwoViewARPLClassifier])
def test_all_view_outputs_swap_equivariantly_and_fusion_is_invariant(model_type) -> None:
    torch.manual_seed(3)
    model = model_type(5).eval()
    inputs = torch.randn(4, 2, 601)
    with torch.no_grad():
        original = model.forward_all_views(inputs)
        swapped = model.forward_all_views(inputs[:, [1, 0]])
    assert original.per_view_logits.shape == (4, 2, 5)
    torch.testing.assert_close(
        original.per_view_features[:, [1, 0]], swapped.per_view_features
    )
    torch.testing.assert_close(
        original.per_view_logits[:, [1, 0]], swapped.per_view_logits
    )
    torch.testing.assert_close(original.fused_features, swapped.fused_features)
    torch.testing.assert_close(original.fused_logits, swapped.fused_logits)
    raw_original = raw_view_evidence(
        original.per_view_features.numpy(),
        original.per_view_logits.numpy(),
        original.fused_logits.numpy(),
    )
    raw_swapped = raw_view_evidence(
        swapped.per_view_features.numpy(),
        swapped.per_view_logits.numpy(),
        swapped.fused_logits.numpy(),
    )
    for name in ("u_worst", "u_mean", "u_gap", "js", "feature_l2_mean"):
        np.testing.assert_allclose(raw_original[name], raw_swapped[name], atol=1e-7)


def test_ce_fused_logits_equal_mean_per_view_logits() -> None:
    model = TwoViewCEClassifier(5).eval()
    with torch.no_grad():
        output = model.forward_all_views(torch.randn(5, 2, 601))
    torch.testing.assert_close(output.fused_logits, output.per_view_logits.mean(dim=1))


def test_arpl_mean_view_logit_identity_has_class_independent_correction() -> None:
    model = TwoViewARPLClassifier(5).eval()
    with torch.no_grad():
        output = model.forward_all_views(torch.randn(5, 2, 601))
    correction = (
        (output.per_view_features[:, 0] - output.per_view_features[:, 1])
        .pow(2)
        .mean(dim=1)
        / 4.0
    )
    torch.testing.assert_close(
        output.per_view_logits.mean(dim=1),
        output.fused_logits + correction[:, None],
        rtol=1e-5,
        atol=1e-6,
    )


def test_arpl_view_aux_margin_is_fused_only_and_gradients_are_finite() -> None:
    torch.manual_seed(9)
    baseline = TwoViewARPLClassifier(5)
    auxiliary = TwoViewARPLClassifier(5)
    auxiliary.load_state_dict(baseline.state_dict())
    inputs = torch.randn(6, 2, 601)
    labels = torch.tensor([0, 1, 2, 3, 4, 0])
    _, base_loss = view_auxiliary_loss(
        baseline, "ARPL_LITE", inputs, labels, lambda_view=0.5
    )
    _, aux_loss = view_auxiliary_loss(
        auxiliary, "ARPL_VIEW_AUX", inputs, labels, lambda_view=0.5
    )
    base_loss["total_loss"].backward()
    aux_loss["total_loss"].backward()
    torch.testing.assert_close(baseline.head.radius.grad, auxiliary.head.radius.grad)
    assert auxiliary.head.reciprocal_points.grad is not None
    assert torch.isfinite(auxiliary.head.reciprocal_points.grad).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in auxiliary.backbone.parameters()
    )


def test_ce_view_aux_produces_encoder_and_shared_head_gradients() -> None:
    model = TwoViewCEClassifier(5)
    inputs = torch.randn(5, 2, 601)
    labels = torch.tensor([0, 1, 2, 3, 4])
    _, losses = view_auxiliary_loss(
        model, "CE_VIEW_AUX", inputs, labels, lambda_view=0.5
    )
    losses["total_loss"].backward()
    assert torch.isfinite(model.classifier.weight.grad).all()
    assert any(parameter.grad is not None for parameter in model.backbone.parameters())


def test_ecdf_rejects_non_known_fit_and_fixed_rules_match_hand_example() -> None:
    with pytest.raises(DataValidationError, match="known_calibration"):
        KnownCalibrationECDF.fit(np.array([0.1, 0.2]), role="surrogate_unknown")
    known = {
        "u_f": np.array([0.0, 1.0]),
        "u_1": np.array([0.0, 2.0]),
        "u_2": np.array([1.0, 3.0]),
        "js": np.array([0.1, 0.3]),
    }
    unknown = {
        "u_f": np.array([0.5]),
        "u_1": np.array([2.5]),
        "u_2": np.array([0.5]),
        "js": np.array([0.4]),
    }
    transformed, parameters = fit_and_apply_evidence_ecdfs(known, unknown)
    row = transformed["surrogate_unknown"]
    assert row["F0_FUSED"].item() == pytest.approx(0.5)
    assert row["F1_WORST_VIEW"].item() == pytest.approx(0.75)
    assert row["F2_EVIDENCE_UNION"].item() == pytest.approx(0.75)
    assert row["F3_DISAGREEMENT_AWARE"].item() == pytest.approx(1.0)
    assert parameters["surrogate_unknown_used_for_fit"] is False


def _selection_rows(deltas_by_rule):
    rows = []
    for method in ("CE_VIEW_AUX", "ARPL_VIEW_AUX"):
        for split_index, split in enumerate(("S0", "S1", "S2")):
            rows.append(
                {
                    "split_id": split,
                    "seed": 20260830,
                    "method": method,
                    "rule": "F0_FUSED",
                    "auroc": 0.5,
                    "oscr": 0.5,
                }
            )
            unit = split_index + (0 if method == "CE_VIEW_AUX" else 3)
            for rule, deltas in deltas_by_rule.items():
                rows.append(
                    {
                        "split_id": split,
                        "seed": 20260830,
                        "method": method,
                        "rule": rule,
                        "auroc": 0.5 + deltas[unit],
                        "oscr": 0.5 + deltas[unit],
                    }
                )
    return rows


def test_development_selection_is_deterministic_and_gate_blocks_failure() -> None:
    config = load_mv_evidence_config(
        PROJECT_ROOT
        / "configs/experiments/arpl/arpl_mv_evidence_surrogate_v1.yaml"
    )
    passing = _selection_rows(
        {
            "F1_WORST_VIEW": [0.03] * 6,
            "F2_EVIDENCE_UNION": [0.04] * 6,
            "F3_DISAGREEMENT_AWARE": [0.04] * 6,
        }
    )
    first = select_development_fusion(passing, config)
    second = select_development_fusion(list(reversed(passing)), config)
    assert first == second
    assert first["selected_rule"] == "F2_EVIDENCE_UNION"
    failing = _selection_rows(
        {
            "F1_WORST_VIEW": [0.0] * 6,
            "F2_EVIDENCE_UNION": [0.01] * 6,
            "F3_DISAGREEMENT_AWARE": [-0.01] * 6,
        }
    )
    result = select_development_fusion(failing, config)
    assert result["gate_passed"] is False
    with pytest.raises(DataValidationError, match="confirmation is forbidden"):
        require_confirmation_gate(result)


def _synthetic_bundle() -> ProcessedBundle:
    rows = []
    profiles = np.zeros((3600, 601), dtype=np.float64)
    names = [*SOURCE_KNOWN_ORDER, "unknown-0", "unknown-1", "unknown-2"]
    row_index = 0
    for class_index, class_name in enumerate(names):
        for angle in range(360):
            profiles[row_index] = class_index + angle / 1000.0
            rows.append(
                {
                    "sample_id": f"{class_name}-{angle}",
                    "class_name": class_name,
                    "class_role": "known" if class_index < 7 else "unknown",
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


def test_confirmation_splits_exclude_final_unknown_and_even_angle_test() -> None:
    config = load_mv_evidence_config(
        PROJECT_ROOT
        / "configs/experiments/arpl/arpl_mv_evidence_surrogate_v1.yaml"
    )
    bundle = _synthetic_bundle()
    for spec in config["classes"]["confirmation_splits"]:
        prepared = prepare_surrogate_split(
            bundle,
            source_known_order=SOURCE_KNOWN_ORDER,
            split_id=spec["split_id"],
            angle_fold=spec["angle_fold"],
            train_known_indices=spec["train_known_indices"],
            surrogate_unknown_indices=spec["surrogate_unknown_indices"],
            pairs_per_class=4,
            base_seed=20260830,
        )
        assert prepared.pair_audit["final_unknown_pairs"] == 0
        assert prepared.pair_audit["even_angle_pairs"] == 0
        assert prepared.pair_audit["test_pairs_generated"] is False
        assert prepared.pair_audit["surrogate_train_pairs_materialized"] is False
        assert len(prepared.pair_ids["train"]) == 20
        assert len(prepared.pair_ids["known_calibration"]) == 20
        assert len(prepared.pair_ids["surrogate_unknown"]) == 8
