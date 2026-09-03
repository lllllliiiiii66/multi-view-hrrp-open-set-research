from __future__ import annotations

from collections import OrderedDict

import pytest


torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from hrrp_osr.models.cssr_1d import PCSSRCore1D  # noqa: E402
from hrrp_osr.models.cssr_e2e_1d import (  # noqa: E402
    ABSOLUTE_VARIANTS,
    CSSR_VARIANTS,
    Q1_CE_FINETUNE_CONTROL,
    Q2_E2E_REL_CSSR_1X1,
    Q3_E2E_ABSREL_CSSR_1X1,
    Q4_E2E_ABSREL_CSSR_LOCAL3,
    FGMVCSSRE2EModel,
    LocalPCSSRCore1D,
    absolute_normalized_reconstruction_error,
    fusion_guided_class_score,
)
from hrrp_osr.models.ms_mean_factorial import (  # noqa: E402
    MSMeanHeadFactorialModel,
    clone_state_dict,
)


VARIANTS = (
    Q1_CE_FINETUNE_CONTROL,
    Q2_E2E_REL_CSSR_1X1,
    Q3_E2E_ABSREL_CSSR_1X1,
    Q4_E2E_ABSREL_CSSR_LOCAL3,
)


def _r2(seed: int = 20260904) -> MSMeanHeadFactorialModel:
    torch.manual_seed(seed)
    return MSMeanHeadFactorialModel("R2_MS_MEAN_CE", known_class_count=5)


def _model(variant: str, *, seed: int = 20260904) -> FGMVCSSRE2EModel:
    source = _r2(seed)
    return FGMVCSSRE2EModel.from_r2_state_dict(
        clone_state_dict(source.state_dict()),
        variant,
        known_class_count=5,
        autoencoder_seed=731,
    )


def _assert_same_state(
    left: OrderedDict[str, torch.Tensor] | dict[str, torch.Tensor],
    right: OrderedDict[str, torch.Tensor] | dict[str, torch.Tensor],
) -> None:
    assert left.keys() == right.keys()
    for name in left:
        assert torch.equal(left[name], right[name]), name


def test_strict_r2_wrap_preserves_epoch_zero_state_and_rejects_bad_state() -> None:
    source = _r2(41)
    state = clone_state_dict(source.state_dict())
    wrapped = FGMVCSSRE2EModel.from_r2_state_dict(
        state,
        Q2_E2E_REL_CSSR_1X1,
        known_class_count=5,
    )
    _assert_same_state(state, wrapped.r2_model.state_dict())

    broken = dict(state)
    broken.pop(next(iter(broken)))
    with pytest.raises(RuntimeError):
        FGMVCSSRE2EModel.from_r2_state_dict(
            broken,
            Q1_CE_FINETUNE_CONTROL,
            known_class_count=5,
        )


def test_q1_to_q4_have_byte_identical_epoch_zero_r2_state() -> None:
    state = clone_state_dict(_r2(77).state_dict())
    models = [
        FGMVCSSRE2EModel.from_r2_state_dict(
            state,
            variant,
            known_class_count=5,
            autoencoder_seed=13,
        )
        for variant in VARIANTS
    ]
    for model in models:
        _assert_same_state(state, model.r2_model.state_dict())
    _assert_same_state(
        models[1].cssr_core.state_dict(),
        models[2].cssr_core.state_dict(),
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_frozen_and_trainable_scope_including_bn_eval(variant: str) -> None:
    model = _model(variant).train()

    assert not model.encoder.stem.training
    assert not model.encoder.stages[0].training
    assert not model.encoder.stages[1].training
    assert model.encoder.stages[2].training
    assert model.encoder.projection.training
    assert model.global_head.training
    assert all(not parameter.requires_grad for parameter in model.encoder.stem.parameters())
    assert all(not parameter.requires_grad for parameter in model.encoder.stages[0].parameters())
    assert all(not parameter.requires_grad for parameter in model.encoder.stages[1].parameters())
    assert all(parameter.requires_grad for parameter in model.encoder.stages[2].parameters())
    assert all(parameter.requires_grad for parameter in model.encoder.projection.parameters())
    assert all(parameter.requires_grad for parameter in model.global_head.parameters())
    if variant in CSSR_VARIANTS:
        assert model.cssr_core is not None and model.cssr_core.training
        assert all(parameter.requires_grad for parameter in model.cssr_core.parameters())
    else:
        assert model.cssr_core is None

    groups = model.trainable_parameter_groups()
    assert set(groups) == {"last_stage", "projection_and_ce_head", "autoencoders"}
    flat = [parameter for values in groups.values() for parameter in values]
    assert len(flat) == len({id(parameter) for parameter in flat})

    model.eval()
    assert not any(module.training for module in model.modules())
    model.train()
    for module in (
        model.encoder.stem,
        model.encoder.stages[0],
        model.encoder.stages[1],
    ):
        assert not module.training
        assert all(not child.training for child in module.modules())


def test_frozen_batch_norm_buffers_do_not_change_during_training_step() -> None:
    model = _model(Q3_E2E_ABSREL_CSSR_1X1).train()
    frozen = (model.encoder.stem, model.encoder.stages[0], model.encoder.stages[1])
    before = {
        f"{index}.{name}": value.detach().clone()
        for index, module in enumerate(frozen)
        for name, value in module.state_dict().items()
    }
    inputs = torch.randn(2, 2, 601)
    labels = torch.tensor([0, 3])
    loss = model.loss(model(inputs), labels).total_loss
    loss.backward()
    after = {
        f"{index}.{name}": value.detach().clone()
        for index, module in enumerate(frozen)
        for name, value in module.state_dict().items()
    }
    _assert_same_state(before, after)


@pytest.mark.parametrize(
    ("variant", "kernel"),
    [
        (Q2_E2E_REL_CSSR_1X1, 1),
        (Q3_E2E_ABSREL_CSSR_1X1, 1),
        (Q4_E2E_ABSREL_CSSR_LOCAL3, 3),
    ],
)
def test_cssr_ae_structure_independence_and_forbidden_components(
    variant: str,
    kernel: int,
) -> None:
    model = _model(variant)
    core = model.cssr_core
    assert isinstance(core, LocalPCSSRCore1D if kernel == 3 else PCSSRCore1D)
    assert next(core.parameters()).device == next(model.r2_model.parameters()).device
    assert next(core.parameters()).dtype == next(model.r2_model.parameters()).dtype
    assert len(core.class_autoencoders) == 5
    addresses = []
    for autoencoder in core.class_autoencoders:
        encoder = autoencoder.encoder[0]
        decoder = autoencoder.decoder
        assert encoder.kernel_size == (kernel,)
        assert decoder.kernel_size == (kernel,)
        assert encoder.padding == decoder.padding == ((kernel - 1) // 2,)
        assert encoder.bias is None and decoder.bias is None
        leaves = [
            module
            for module in autoencoder.modules()
            if not tuple(module.children())
        ]
        assert [type(module) for module in leaves] == [nn.Conv1d, nn.Tanh, nn.Conv1d]
        addresses.extend((encoder.weight.data_ptr(), decoder.weight.data_ptr()))
    assert len(addresses) == len(set(addresses))
    forbidden_types = (nn.MultiheadAttention, nn.BatchNorm1d, nn.LayerNorm)
    assert not any(isinstance(module, forbidden_types) for module in core.modules())
    forbidden_names = ("arpl", "reciprocal", "attention", "rejector", "pseudo")
    assert not any(
        token in name.lower()
        for name, _ in model.named_modules()
        for token in forbidden_names
    )


def test_absolute_r_matches_manual_global_activation_formula() -> None:
    features = torch.tensor([[[1.0, -3.0], [2.0, 4.0]]])
    reconstructions = torch.stack(
        [
            torch.zeros_like(features),
            features + 2.0,
        ],
        dim=1,
    )
    actual = absolute_normalized_reconstruction_error(
        features,
        reconstructions,
        epsilon=1.0e-8,
    )
    denominator = features.abs().mean() + 1.0e-8
    expected = torch.tensor([[features.abs().mean(), torch.tensor(2.0)]]) / denominator
    torch.testing.assert_close(actual, expected)


def test_absolute_and_separation_losses_match_hand_calculation() -> None:
    model = _model(Q3_E2E_ABSREL_CSSR_1X1).eval()
    output = model(torch.randn(2, 2, 601))
    labels = torch.tensor([0, 1])
    manual_r = torch.tensor(
        [
            [[0.2, 0.6, 0.7, 0.8, 0.9], [0.4, 0.5, 0.8, 0.9, 1.0]],
            [[0.5, 0.3, 0.6, 0.7, 0.8], [0.4, 0.6, 0.7, 0.8, 0.9]],
        ]
    )
    output = type(output)(
        feature_maps=output.feature_maps,
        per_view_features=output.per_view_features,
        fused_features=output.fused_features,
        fused_logits=output.fused_logits,
        cssr_outputs=output.cssr_outputs,
        normalized_reconstruction_errors=manual_r,
    )
    losses = model.loss(output, labels)
    expected_true = torch.tensor([[0.2, 0.4], [0.3, 0.6]])
    expected_wrong = torch.tensor([[0.6, 0.5], [0.5, 0.4]])
    expected_margin = expected_wrong - expected_true
    torch.testing.assert_close(losses.true_class_r, expected_true)
    torch.testing.assert_close(losses.nearest_wrong_class_r, expected_wrong)
    torch.testing.assert_close(losses.reconstruction_margin, expected_margin)
    torch.testing.assert_close(losses.absolute_loss, expected_true.mean())
    torch.testing.assert_close(
        losses.separation_loss,
        torch.relu(torch.tensor(0.2) - expected_margin).mean(),
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_forward_shapes_finite_loss_and_exact_composition(variant: str) -> None:
    model = _model(variant).eval()
    inputs = torch.randn(2, 2, 601, generator=torch.Generator().manual_seed(5))
    labels = torch.tensor([0, 4])
    output = model(inputs)
    losses = model.loss(output, labels)

    assert output.feature_maps.shape == (2, 2, 128, 76)
    assert output.per_view_features.shape == (2, 2, 128)
    assert output.fused_features.shape == (2, 128)
    assert output.fused_logits.shape == (2, 5)
    assert torch.isfinite(losses.total_loss)
    weights = losses.component_weights
    expected = (
        weights["classification"] * losses.classification_loss
        + weights["relative"] * losses.relative_loss
        + weights["absolute"] * losses.absolute_loss
        + weights["separation"] * losses.separation_loss
    )
    torch.testing.assert_close(losses.total_loss, expected)
    if variant == Q1_CE_FINETUNE_CONTROL:
        assert output.cssr_outputs is None
        assert output.normalized_reconstruction_errors is None
        assert losses.relative_loss.item() == 0.0
    else:
        assert output.cssr_outputs is not None
        assert output.normalized_reconstruction_errors.shape == (2, 2, 5)
        assert torch.isfinite(output.normalized_reconstruction_errors).all()
        assert losses.true_class_r.shape == (2, 2)
    assert (variant in ABSOLUTE_VARIANTS) == (weights["absolute"] == 0.25)
    assert (variant in ABSOLUTE_VARIANTS) == (weights["separation"] == 0.5)


def test_epoch_zero_forward_matches_original_r2_exactly_in_eval_mode() -> None:
    original = _r2(313).eval()
    wrapped = FGMVCSSRE2EModel.from_r2_state_dict(
        clone_state_dict(original.state_dict()),
        Q1_CE_FINETUNE_CONTROL,
        known_class_count=5,
    ).eval()
    inputs = torch.randn(3, 2, 601, generator=torch.Generator().manual_seed(17))
    with torch.no_grad():
        expected = original(inputs, compute_rejector=False)
        actual = wrapped(inputs)
    torch.testing.assert_close(actual.per_view_features, expected.raw_view_tokens, rtol=0, atol=0)
    torch.testing.assert_close(actual.fused_features, expected.global_class_token, rtol=0, atol=0)
    torch.testing.assert_close(actual.fused_logits, expected.global_logits, rtol=0, atol=0)


@pytest.mark.parametrize("variant", VARIANTS)
def test_view_swap_preserves_fused_prediction_loss_and_guided_score(variant: str) -> None:
    model = _model(variant).eval()
    inputs = torch.randn(2, 2, 601, generator=torch.Generator().manual_seed(29))
    labels = torch.tensor([1, 3])
    with torch.no_grad():
        original = model(inputs)
        swapped = model(inputs[:, [1, 0]])
        original_loss = model.loss(original, labels)
        swapped_loss = model.loss(swapped, labels)
    torch.testing.assert_close(original.fused_logits, swapped.fused_logits)
    assert torch.equal(original.fused_logits.argmax(1), swapped.fused_logits.argmax(1))
    torch.testing.assert_close(original_loss.total_loss, swapped_loss.total_loss)
    if variant in CSSR_VARIANTS:
        torch.testing.assert_close(
            original.normalized_reconstruction_errors[:, [1, 0]],
            swapped.normalized_reconstruction_errors,
        )
        anomaly = -torch.log(original.normalized_reconstruction_errors + 0.3)
        swapped_anomaly = anomaly[:, [1, 0]]
        torch.testing.assert_close(
            fusion_guided_class_score(anomaly, original.fused_logits),
            fusion_guided_class_score(swapped_anomaly, swapped.fused_logits),
        )
