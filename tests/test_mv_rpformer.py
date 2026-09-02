from __future__ import annotations

import copy
import fcntl
from pathlib import Path

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from hrrp_osr.data.errors import DataConfigError, DataValidationError  # noqa: E402
from hrrp_osr.models.arpl import ARPLReciprocalHead  # noqa: E402
from hrrp_osr.models.hrrp_ms_resnet import HRRPMultiScaleResNet1D  # noqa: E402
from hrrp_osr.models.mv_rpformer import (  # noqa: E402
    METHODS,
    MVRPFormer,
    PoolingByMultiheadAttention,
    PreNormSAB,
    build_rejector_evidence,
)
from hrrp_osr.training.mv_rpformer import (  # noqa: E402
    IntentionalTrainingInterruption,
    PseudoAuditAccumulator,
    _assert_exact_method_directory_matrix,
    build_initialized_method_group,
    build_phase_plan,
    coherent_feature_mixup,
    learning_rate_for_epoch,
    load_mv_rpformer_config,
    loss_weights_for_epoch,
    recompute_unit_metrics_from_rows,
    require_train_known_pseudo_source,
    sample_cross_class_indices,
    sample_cross_class_mismatch,
    run_single_method,
    summarize_paired_comparisons,
    train_one_method,
    truncated_beta_lambdas,
    uniform_kl_loss,
)
from hrrp_osr.training.arpl_pilot import (  # noqa: E402
    PreparedSurrogateSplit,
    ScalarNormalization,
    _state_sha256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/experiments/arpl/mv_rpformer_surrogate_v1.yaml"


def test_config_freezes_scope_matrix_and_confirmation() -> None:
    config = load_mv_rpformer_config(CONFIG_PATH)
    assert config["model"]["methods"] == list(METHODS)
    assert config["training"]["confirmation_methods"] == list(METHODS)
    assert config["evidence_scope"]["final_unknown_classes_used"] is False
    assert config["evidence_scope"]["even_angle_test_used"] is False
    assert len(build_phase_plan(config, "confirmation")) == 12


def test_config_rejects_protocol_mutation(tmp_path: Path) -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    changed = copy.deepcopy(config)
    changed["training"]["formal_checkpoint_epoch"] = 99
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(changed, allow_unicode=True), encoding="utf-8")
    with pytest.raises(DataConfigError, match="training contract changed"):
        load_mv_rpformer_config(path)


def test_multi_scale_encoder_shape_scales_pooling_and_parameter_budget() -> None:
    encoder = HRRPMultiScaleResNet1D().eval()
    with torch.no_grad():
        output = encoder(torch.randn(4, 601))
    assert output.shape == (4, 128)
    assert encoder.parameter_count < 500_000
    for stage in encoder.stages:
        assert [branch[0].kernel_size[0] for branch in stage.branches] == [3, 7, 15]
        assert all(branch[0].groups == 1 for branch in stage.branches)


def test_sab_tokens_are_swap_equivariant() -> None:
    torch.manual_seed(1)
    sab = PreNormSAB(dropout=0.1).eval()
    tokens = torch.randn(3, 2, 128)
    with torch.no_grad():
        original, _ = sab(tokens)
        swapped, _ = sab(tokens[:, [1, 0]])
    torch.testing.assert_close(original[:, [1, 0]], swapped, rtol=1e-5, atol=1e-6)


def test_two_seed_pma_shape_and_swap_invariance() -> None:
    torch.manual_seed(2)
    pma = PoolingByMultiheadAttention(dropout=0.1).eval()
    tokens = torch.randn(3, 2, 128)
    with torch.no_grad():
        original, attention = pma(tokens)
        swapped, swapped_attention = pma(tokens[:, [1, 0]])
    assert original.shape == (3, 2, 128)
    assert attention.shape == (3, 4, 2, 2)
    torch.testing.assert_close(original, swapped, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        attention[..., [1, 0]], swapped_attention, rtol=1e-5, atol=1e-6
    )


def test_attention_weights_remain_probabilities_in_training_mode() -> None:
    torch.manual_seed(20)
    tokens = torch.randn(4, 2, 128)
    sab = PreNormSAB(dropout=0.1).train()
    pma = PoolingByMultiheadAttention(dropout=0.1).train()
    contextual, sab_attention = sab(tokens)
    _, pma_attention = pma(contextual)
    torch.testing.assert_close(
        sab_attention.sum(dim=-1), torch.ones_like(sab_attention.sum(dim=-1))
    )
    torch.testing.assert_close(
        pma_attention.sum(dim=-1), torch.ones_like(pma_attention.sum(dim=-1))
    )


def test_global_and_view_arpl_parameters_are_independent_but_views_share_one_head() -> None:
    model = MVRPFormer("M4_MS_SET_HIER_ARPL", 5).eval()
    assert isinstance(model.global_head, ARPLReciprocalHead)
    assert isinstance(model.view_head, ARPLReciprocalHead)
    assert model.global_head is not model.view_head
    assert (
        model.global_head.reciprocal_points.data_ptr()
        != model.view_head.reciprocal_points.data_ptr()
    )
    inputs = torch.randn(3, 2, 601)
    with torch.no_grad():
        output = model(inputs)
        direct = model.view_head.logits(output.contextual_view_tokens.reshape(-1, 128))
    torch.testing.assert_close(direct.reshape(3, 2, 5), output.per_view_logits)


def test_hierarchical_arpl_loss_matches_manual_formula() -> None:
    torch.manual_seed(3)
    model = MVRPFormer("M4_MS_SET_HIER_ARPL", 5).eval()
    labels = torch.tensor([0, 1, 2, 3])
    output = model(torch.randn(4, 2, 601))
    actual = model.representation_loss(output, labels, lambda_view=0.5)
    global_loss = model.global_head.loss(output.global_class_token, labels)
    repeated = labels[:, None].expand(-1, 2).reshape(-1)
    view_loss = model.view_head.loss(
        output.contextual_view_tokens.reshape(-1, 128), repeated
    )
    torch.testing.assert_close(
        actual["total"], global_loss.total_loss + 0.5 * view_loss.total_loss
    )


def test_m4_m6_common_trunk_starts_equal_and_rejector_is_not_sampled_in_warmup() -> None:
    config = load_mv_rpformer_config(CONFIG_PATH)
    models, audit = build_initialized_method_group(5, seed=20260830, config=config)
    assert all(audit["shared_initialization_checks"].values())
    inputs = torch.randn(2, 2, 601)
    models["M4_MS_SET_HIER_ARPL"].train()
    models["M6_MV_RPFORMER_FULL"].train()
    torch.manual_seed(123)
    m4 = models["M4_MS_SET_HIER_ARPL"](inputs, compute_rejector=False)
    torch.manual_seed(123)
    m6 = models["M6_MV_RPFORMER_FULL"](inputs, compute_rejector=False)
    torch.testing.assert_close(m4.global_logits, m6.global_logits)
    torch.testing.assert_close(m4.per_view_logits, m6.per_view_logits)
    assert m6.unknown_probability is None


def test_mismatch_sampler_guarantees_different_classes_and_frames() -> None:
    anchor_labels = torch.tensor([0, 1, 2, 3])
    pool_labels = torch.tensor([0, 0, 1, 2, 3, 4])
    anchor_frames = torch.tensor([0, 1, 2, 3])
    pool_frames = torch.tensor([0, 2, 1, 3, 4, 5])
    indices = sample_cross_class_indices(
        anchor_labels,
        pool_labels,
        generator=torch.Generator().manual_seed(4),
        anchor_frames=anchor_frames,
        pool_partner_frames=pool_frames,
    )
    assert torch.all(pool_labels[indices] != anchor_labels)
    assert torch.all(pool_frames[indices] != anchor_frames)
    anchors = torch.randn(4, 2, 601)
    partners = torch.randn(4, 2, 601)
    mismatch = sample_cross_class_mismatch(
        anchors, anchor_labels, partners, pool_labels[indices]
    )
    torch.testing.assert_close(mismatch[:, 0], anchors[:, 0])
    torch.testing.assert_close(mismatch[:, 1], partners[:, 1])


def test_coherent_mixup_reuses_one_lambda_for_both_views() -> None:
    first = torch.zeros(5, 2, 4)
    second = torch.ones(5, 2, 4)
    values = truncated_beta_lambdas(5, rng=np.random.default_rng(5))
    mixed = coherent_feature_mixup(first, second, torch.from_numpy(values))
    assert np.all((values >= 0.3) & (values <= 0.7))
    torch.testing.assert_close(mixed[:, 0], mixed[:, 1])
    torch.testing.assert_close(mixed[:, 0, 0], 1.0 - torch.from_numpy(values))


@pytest.mark.parametrize("role", ["surrogate_unknown", "final_unknown", "known_calibration"])
def test_pseudo_generation_rejects_non_train_roles(role: str) -> None:
    with pytest.raises(DataValidationError, match="train_known"):
        require_train_known_pseudo_source(role)
    require_train_known_pseudo_source("train_known")


def test_pseudo_audit_independently_enforces_composition_and_lambda_range() -> None:
    valid = PseudoAuditAccumulator(method="M6_MV_RPFORMER_FULL", seed=1)
    valid.real_count = 4
    valid.mismatch_count = 2
    valid.mixup_count = 2
    valid.lambda_min = 0.3
    valid.lambda_max = 0.7
    assert valid.to_json()["status"] == "passed"

    wrong_composition = PseudoAuditAccumulator(
        method="M6_MV_RPFORMER_FULL", seed=1
    )
    wrong_composition.real_count = 4
    wrong_composition.mismatch_count = 4
    assert wrong_composition.to_json()["status"] == "failed"

    wrong_lambda = PseudoAuditAccumulator(method="M6_MV_RPFORMER_FULL", seed=1)
    wrong_lambda.real_count = 4
    wrong_lambda.mismatch_count = 2
    wrong_lambda.mixup_count = 2
    wrong_lambda.lambda_min = 0.2
    wrong_lambda.lambda_max = 0.8
    assert wrong_lambda.to_json()["status"] == "failed"


def test_rejector_uses_global_predicted_class_support() -> None:
    global_logits = torch.tensor([[0.0, 3.0, 1.0]])
    view_logits = torch.tensor([[[100.0, 20.0, 30.0], [101.0, 21.0, 31.0]]])
    evidence = build_rejector_evidence(
        global_reject_token=torch.zeros(1, 4),
        global_logits=global_logits,
        per_view_logits=view_logits,
        contextual_view_tokens=torch.zeros(1, 2, 4),
        pma_attention=torch.full((1, 2, 2, 2), 0.5),
    )
    assert evidence.shape == (1, 14)
    # After the four-dimensional token and global unknown scalar, support is
    # sorted as 21,20 from global class 1, not the per-view argmax class 0.
    torch.testing.assert_close(evidence[0, 5:7], torch.tensor([21.0, 20.0]))


@pytest.mark.parametrize(
    "method", ["M3_MS_SET_GLOBAL_ARPL", "M4_MS_SET_HIER_ARPL", "M6_MV_RPFORMER_FULL", "M7_MV_CEFORMER_FULL"]
)
def test_model_outputs_are_strictly_permutation_invariant(method: str) -> None:
    torch.manual_seed(6)
    model = MVRPFormer(method, 5).eval()
    inputs = torch.randn(2, 2, 601)
    with torch.no_grad():
        original = model(inputs)
        swapped = model(inputs[:, [1, 0]])
    torch.testing.assert_close(
        original.contextual_view_tokens[:, [1, 0]],
        swapped.contextual_view_tokens,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(original.global_class_token, swapped.global_class_token)
    torch.testing.assert_close(original.global_reject_token, swapped.global_reject_token)
    torch.testing.assert_close(original.global_logits, swapped.global_logits, rtol=1e-5, atol=1e-6)
    if original.unknown_probability is not None:
        torch.testing.assert_close(original.unknown_probability, swapped.unknown_probability)


def test_warmup_disables_reject_losses_until_epoch_31_and_scheduler_boundaries() -> None:
    assert loss_weights_for_epoch(30) == {
        "representation": 1.0,
        "reject": 0.0,
        "uniform": 0.0,
    }
    assert loss_weights_for_epoch(31) == {
        "representation": 1.0,
        "reject": 1.0,
        "uniform": 0.1,
    }
    assert learning_rate_for_epoch(1) == pytest.approx(6e-5)
    assert learning_rate_for_epoch(5) == pytest.approx(3e-4)
    assert learning_rate_for_epoch(100) == pytest.approx(0.0)


def test_uniform_kl_has_correct_direction() -> None:
    assert uniform_kl_loss(torch.zeros(4, 5)).item() == pytest.approx(0.0, abs=1e-7)
    assert uniform_kl_loss(torch.tensor([[10.0, 0.0, 0.0]] * 4)).item() > 0.0


def test_confirmation_plan_is_config_only_and_full_matrix() -> None:
    config = load_mv_rpformer_config(CONFIG_PATH)
    first = build_phase_plan(config, "confirmation")
    second = build_phase_plan(copy.deepcopy(config), "confirmation")
    assert first == second
    assert all(unit["methods"] == list(METHODS) for unit in first)
    assert {(unit["spec"]["split_id"], unit["seed"]) for unit in first} == {
        (split_id, seed)
        for split_id in ("C0", "C1", "C2", "C3")
        for seed in (20260830, 20260831, 20260832)
    }


def test_phase_directory_matrix_rejects_extra_method(tmp_path: Path) -> None:
    config = load_mv_rpformer_config(CONFIG_PATH)
    plan = build_phase_plan(config, "smoke")
    seed_root = tmp_path / "S0" / "seed_20260830"
    for method in METHODS:
        (seed_root / method).mkdir(parents=True)
    _assert_exact_method_directory_matrix(tmp_path, plan)
    (seed_root / "EXTRA_METHOD").mkdir()
    with pytest.raises(DataValidationError, match="extra method"):
        _assert_exact_method_directory_matrix(tmp_path, plan)


def test_confirmation_summary_rejects_incomplete_or_extra_units() -> None:
    config = load_mv_rpformer_config(CONFIG_PATH)
    rows = [
        {
            "split_id": "C0",
            "seed": 20260830,
            "method": method,
            **{metric: 0.5 for metric in (
                "known_accuracy",
                "known_macro_f1",
                "auroc",
                "oscr",
                "fpr95",
                "unknown_rejection_rate",
                "k_plus_1_macro_f1",
            )},
        }
        for method in METHODS
    ]
    with pytest.raises(DataValidationError, match="exact C0-C3"):
        summarize_paired_comparisons(
            rows, config=config, require_confirmation_units=True
        )


def test_prediction_rows_recompute_every_metric_exactly() -> None:
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
    first = recompute_unit_metrics_from_rows(
        rows, known_class_count=2, known_acceptance_rate=0.75
    )
    second = recompute_unit_metrics_from_rows(
        list(reversed(rows)), known_class_count=2, known_acceptance_rate=0.75
    )
    assert first == second
    assert first["known_accuracy"] == pytest.approx(0.75)
    assert first["unknown_rejection_rate"] == pytest.approx(1.0)


def _tiny_prepared_split() -> PreparedSurrogateSplit:
    rng = np.random.default_rng(300)
    labels = np.arange(10, dtype=np.int64) % 5
    pair_rows = tuple(
        {
            "pair_id": f"train-{index}",
            "experiment_role": "train_known",
            "view1_frame_id": int(index % 4),
            "view2_frame_id": int((index + 1) % 4),
            "view1_sample_id": f"a-{index}",
            "view2_sample_id": f"b-{index}",
        }
        for index in range(10)
    )
    return PreparedSurrogateSplit(
        split_id="T0",
        angle_fold=0,
        train_class_order=tuple(f"known-{index}" for index in range(5)),
        surrogate_class_order=("unknown-a", "unknown-b"),
        pair_manifest_rows=pair_rows,
        pair_manifest_bytes=b"tiny-test-manifest",
        pair_manifest_sha256="tiny-test-manifest-sha",
        pair_audit={
            "final_unknown_pairs": 0,
            "even_angle_pairs": 0,
            "test_pairs_generated": False,
        },
        normalization=ScalarNormalization(0.0, 1.0, 1e-8, 20),
        inputs={
            "train": rng.normal(size=(10, 2, 601)).astype(np.float32),
            "known_calibration": rng.normal(size=(10, 2, 601)).astype(np.float32),
            "surrogate_unknown": rng.normal(size=(4, 2, 601)).astype(np.float32),
        },
        labels={
            "train": labels,
            "known_calibration": labels.copy(),
            "surrogate_unknown": np.full(4, 5, dtype=np.int64),
        },
        pair_ids={
            "train": tuple(f"train-{index}" for index in range(10)),
            "known_calibration": tuple(f"cal-{index}" for index in range(10)),
            "surrogate_unknown": tuple(f"unknown-{index}" for index in range(4)),
        },
        class_names={
            "train": tuple(f"known-{value}" for value in labels),
            "known_calibration": tuple(f"known-{value}" for value in labels),
            "surrogate_unknown": ("unknown-a", "unknown-a", "unknown-b", "unknown-b"),
        },
    )


def test_epoch_checkpoint_resume_is_exact_across_rejector_activation(tmp_path: Path) -> None:
    config = copy.deepcopy(load_mv_rpformer_config(CONFIG_PATH))
    config["training"].update(
        {
            "batch_size": 10,
            "total_epochs": 2,
            "smoke_epochs": 2,
            "representation_only_epochs": 1,
            "warmup_epochs": 1,
        }
    )
    prepared = _tiny_prepared_split()
    method = "M6_MV_RPFORMER_FULL"
    continuous_models, _ = build_initialized_method_group(5, seed=44, config=config)
    continuous = train_one_method(
        continuous_models[method],
        method=method,
        prepared=prepared,
        seed=44,
        config=config,
        mode="smoke",
        device=torch.device("cpu"),
    )
    checkpoint = tmp_path / "latest.pt"
    interrupted_models, _ = build_initialized_method_group(5, seed=44, config=config)
    with pytest.raises(IntentionalTrainingInterruption):
        train_one_method(
            interrupted_models[method],
            method=method,
            prepared=prepared,
            seed=44,
            config=config,
            mode="smoke",
            device=torch.device("cpu"),
            resume_checkpoint=checkpoint,
            _interrupt_after_epoch=1,
        )
    incompatible = tmp_path / "incompatible.pt"
    checkpoint_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_state["runtime_contract"]["device_type"] = "cuda"
    torch.save(checkpoint_state, incompatible)
    incompatible_models, _ = build_initialized_method_group(5, seed=44, config=config)
    with pytest.raises(DataValidationError, match="resume checkpoint contract differs"):
        train_one_method(
            incompatible_models[method],
            method=method,
            prepared=prepared,
            seed=44,
            config=config,
            mode="smoke",
            device=torch.device("cpu"),
            resume_checkpoint=incompatible,
        )
    resumed_models, _ = build_initialized_method_group(5, seed=44, config=config)
    resumed = train_one_method(
        resumed_models[method],
        method=method,
        prepared=prepared,
        seed=44,
        config=config,
        mode="smoke",
        device=torch.device("cpu"),
        resume_checkpoint=checkpoint,
    )
    assert _state_sha256(continuous["final_state"]) == _state_sha256(
        resumed["final_state"]
    )
    assert continuous["pseudo_audit"] == resumed["pseudo_audit"]
    for first, second in zip(
        continuous["training_log"], resumed["training_log"], strict=True
    ):
        assert {k: v for k, v in first.items() if k != "elapsed_seconds"} == {
            k: v for k, v in second.items() if k != "elapsed_seconds"
        }


def test_duplicate_train_unit_lock_fails_before_touching_bundle(tmp_path: Path) -> None:
    phase_root = tmp_path / "phase"
    method = "M0_CURRENT_CE_MEAN"
    lock_path = (
        tmp_path
        / "_locks"
        / phase_root.name
        / "S0"
        / "seed_20260830"
        / f"{method}.lock"
    )
    lock_path.parent.mkdir(parents=True)
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(DataValidationError, match="already running"):
            run_single_method(
                CONFIG_PATH,
                tmp_path / "missing-bundle",
                phase_root,
                phase="smoke",
                split_id="S0",
                seed=20260830,
                method=method,
                device_request="cpu",
                resume=True,
            )
