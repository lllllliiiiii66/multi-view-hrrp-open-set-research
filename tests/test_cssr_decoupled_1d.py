from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch
from torch import nn

from hrrp_osr.models.cssr_decoupled_1d import (
    D1_DECOUPLED_REL_CSSR,
    D2_DECOUPLED_ABSREL_CSSR,
    DECOUPLED_METHODS,
    DecoupledClassSpecificAutoEncoder1D,
    FGMVCSSRDecoupled1D,
    SharedCSSRSemanticAdapter1D,
    absolute_and_separation_losses,
    compose_decoupled_cssr_loss,
    normalized_absolute_reconstruction_error,
)


def _state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in module.state_dict().items()
    }


def _assert_same_state(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> None:
    assert left.keys() == right.keys()
    for name in left:
        assert torch.equal(left[name], right[name]), name


def _state_changed(before: Mapping[str, torch.Tensor], module: nn.Module) -> bool:
    return any(
        not torch.equal(before[name], value)
        for name, value in module.state_dict().items()
    )


def test_frozen_structure_and_parameter_count() -> None:
    model = FGMVCSSRDecoupled1D()

    assert model.architecture_id == "fg_mv_cssr_decoupled_1d_v1"
    assert DECOUPLED_METHODS == (
        D1_DECOUPLED_REL_CSSR,
        D2_DECOUPLED_ABSREL_CSSR,
    )
    assert isinstance(model.adapter, SharedCSSRSemanticAdapter1D)
    assert model.adapter.residual_scale == 0.1
    assert not any(name.endswith("residual_scale") for name, _ in model.named_parameters())

    adapter_layers = list(model.adapter.delta.children())
    assert [type(layer) for layer in adapter_layers] == [
        nn.Conv1d,
        nn.GroupNorm,
        nn.GELU,
        nn.Conv1d,
    ]
    first_conv, group_norm, _, second_conv = adapter_layers
    assert first_conv.weight.shape == (64, 128, 3)
    assert first_conv.kernel_size == (3,)
    assert first_conv.padding == (1,)
    assert first_conv.bias is None
    assert group_norm.num_groups == 8
    assert group_norm.num_channels == 64
    assert group_norm.eps == 1.0e-5
    assert group_norm.affine
    assert second_conv.weight.shape == (128, 64, 1)
    assert second_conv.kernel_size == (1,)
    assert second_conv.padding == (0,)
    assert second_conv.bias is None

    assert len(model.class_autoencoders) == 5
    parameter_addresses: list[int] = []
    for autoencoder in model.class_autoencoders:
        assert isinstance(autoencoder, DecoupledClassSpecificAutoEncoder1D)
        encoder = autoencoder.encoder[0]
        activation = autoencoder.encoder[1]
        decoder = autoencoder.decoder
        assert isinstance(encoder, nn.Conv1d)
        assert isinstance(activation, nn.Tanh)
        assert isinstance(decoder, nn.Conv1d)
        assert encoder.weight.shape == (32, 128, 3)
        assert decoder.weight.shape == (128, 32, 3)
        assert encoder.kernel_size == decoder.kernel_size == (3,)
        assert encoder.padding == decoder.padding == (1,)
        assert encoder.bias is None and decoder.bias is None
        parameter_addresses.extend((encoder.weight.data_ptr(), decoder.weight.data_ptr()))
    assert len(parameter_addresses) == len(set(parameter_addresses))
    assert sum(parameter.numel() for parameter in model.parameters()) == 155_776

    forbidden = (nn.BatchNorm1d, nn.LayerNorm, nn.MultiheadAttention)
    assert not any(
        isinstance(module, forbidden)
        for module in model.class_autoencoders.modules()
    )


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("num_classes", 4),
        ("input_channels", 64),
        ("latent_channels", 64),
        ("residual_scale", 0.2),
        ("gamma", 0.2),
        ("clip_length", 50.0),
        ("epsilon", 1.0e-6),
        ("margin", 0.1),
    ],
)
def test_frozen_architecture_rejects_configuration_drift(
    argument: str,
    value: int | float,
) -> None:
    with pytest.raises(ValueError, match="architecture changed"):
        FGMVCSSRDecoupled1D(**{argument: value})


def test_adapter_is_exact_fixed_scale_residual() -> None:
    model = FGMVCSSRDecoupled1D().eval()
    features = torch.randn(2, 128, 11, generator=torch.Generator().manual_seed(7))

    with torch.no_grad():
        expected = features + 0.1 * model.adapter.delta(features)
        actual = model.adapter(features)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_class_autoencoders_are_functionally_independent() -> None:
    model = FGMVCSSRDecoupled1D().eval()
    features = torch.randn(2, 128, 9, generator=torch.Generator().manual_seed(11))

    with torch.no_grad():
        before = model(features).reconstructions.clone()
        model.class_autoencoders[0].decoder.weight.add_(0.25)
        after = model(features).reconstructions

    assert not torch.equal(before[:, 0], after[:, 0])
    torch.testing.assert_close(before[:, 1:], after[:, 1:], rtol=0.0, atol=0.0)


def test_forward_matches_frozen_pcssr_formulas() -> None:
    model = FGMVCSSRDecoupled1D().eval()
    features = torch.randn(2, 128, 7, generator=torch.Generator().manual_seed(13))

    with torch.no_grad():
        output = model(features)
        manual_errors = torch.stack(
            [
                torch.norm(reconstruction - output.adapted_features, p=1, dim=1)
                for reconstruction in output.reconstructions.unbind(dim=1)
            ],
            dim=1,
        )
        manual_logits = torch.clamp(-0.1 * manual_errors, -100.0, 100.0)
        manual_probabilities = torch.softmax(manual_logits, dim=1).mean(dim=-1)

    assert output.adapted_features.shape == (2, 128, 7)
    assert output.reconstructions.shape == (2, 5, 128, 7)
    assert output.reconstruction_errors.shape == (2, 5, 7)
    assert output.normalized_reconstruction_errors.shape == (2, 5)
    torch.testing.assert_close(output.reconstruction_errors, manual_errors)
    torch.testing.assert_close(output.logits, manual_logits)
    torch.testing.assert_close(output.probabilities, manual_probabilities)


def test_normalized_absolute_error_matches_hand_calculation() -> None:
    features = torch.tensor([[[1.0, -3.0], [2.0, 4.0]]])
    reconstructions = torch.stack(
        [
            torch.zeros_like(features),
            features + 2.0,
        ],
        dim=1,
    )

    actual = normalized_absolute_reconstruction_error(
        features,
        reconstructions,
        epsilon=1.0e-8,
    )
    denominator = features.abs().mean() + 1.0e-8
    expected = torch.stack(
        [features.abs().mean(), torch.tensor(2.0)],
    ).reshape(1, 2) / denominator

    torch.testing.assert_close(actual, expected)


def test_absolute_and_separation_losses_match_hand_calculation() -> None:
    normalized_errors = torch.tensor(
        [
            [0.2, 0.6, 0.7, 0.8, 0.9],
            [0.5, 0.3, 0.6, 0.7, 0.8],
            [0.4, 0.6, 0.7, 0.8, 0.9],
        ]
    )
    targets = torch.tensor([0, 1, 1])

    absolute, separation, true_r, wrong_r, reconstruction_margin = (
        absolute_and_separation_losses(
            normalized_errors,
            targets,
            separation_margin=0.2,
        )
    )
    expected_true = torch.tensor([0.2, 0.3, 0.6])
    expected_wrong = torch.tensor([0.6, 0.5, 0.4])
    expected_margin = expected_wrong - expected_true

    torch.testing.assert_close(true_r, expected_true)
    torch.testing.assert_close(wrong_r, expected_wrong)
    torch.testing.assert_close(reconstruction_margin, expected_margin)
    torch.testing.assert_close(absolute, expected_true.mean())
    torch.testing.assert_close(
        separation,
        torch.relu(torch.tensor(0.2) - expected_margin).mean(),
    )


def test_d1_and_d2_have_exact_loss_weights_including_zero_terms() -> None:
    relative = torch.tensor(2.0, requires_grad=True)
    absolute = torch.tensor(3.0, requires_grad=True)
    separation = torch.tensor(5.0, requires_grad=True)

    d1 = compose_decoupled_cssr_loss(
        D1_DECOUPLED_REL_CSSR,
        relative_loss=relative,
        absolute_loss=absolute,
        separation_loss=separation,
    )
    d1_gradients = torch.autograd.grad(d1, (relative, absolute, separation))
    torch.testing.assert_close(d1, torch.tensor(2.0))
    assert tuple(gradient.item() for gradient in d1_gradients) == (1.0, 0.0, 0.0)

    d2 = compose_decoupled_cssr_loss(
        D2_DECOUPLED_ABSREL_CSSR,
        relative_loss=relative,
        absolute_loss=absolute,
        separation_loss=separation,
    )
    d2_gradients = torch.autograd.grad(d2, (relative, absolute, separation))
    torch.testing.assert_close(d2, torch.tensor(5.25))
    assert tuple(gradient.item() for gradient in d2_gradients) == (1.0, 0.25, 0.5)


def test_d1_and_d2_share_epoch_zero_state_and_loss_composition() -> None:
    torch.manual_seed(20260905)
    d1_model = FGMVCSSRDecoupled1D()
    torch.manual_seed(20260905)
    d2_model = FGMVCSSRDecoupled1D()
    _assert_same_state(d1_model.state_dict(), d2_model.state_dict())

    features = torch.randn(3, 128, 7, generator=torch.Generator().manual_seed(17))
    targets = torch.tensor([0, 2, 4])
    d1_loss = d1_model.loss(features, targets, D1_DECOUPLED_REL_CSSR)
    d2_loss = d2_model.loss(features, targets, D2_DECOUPLED_ABSREL_CSSR)

    torch.testing.assert_close(d1_loss.total_loss, d1_loss.relative_loss)
    torch.testing.assert_close(
        d2_loss.total_loss,
        d2_loss.relative_loss
        + 0.25 * d2_loss.absolute_loss
        + 0.5 * d2_loss.separation_loss,
    )
    assert d1_loss.component_weights == {
        "relative": 1.0,
        "absolute": 0.0,
        "separation": 0.0,
    }
    assert d2_loss.component_weights == {
        "relative": 1.0,
        "absolute": 0.25,
        "separation": 0.5,
    }
    torch.testing.assert_close(d1_loss.output.logits, d2_loss.output.logits)


@pytest.mark.parametrize("method", DECOUPLED_METHODS)
def test_each_method_has_finite_gradients(method: str) -> None:
    torch.manual_seed(19)
    model = FGMVCSSRDecoupled1D()
    model.configure_for_epoch(6)
    features = torch.randn(3, 128, 9)
    targets = torch.tensor([0, 2, 4])

    loss = model.loss(features, targets, method).total_loss
    loss.backward()

    assert torch.isfinite(loss)
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)


def test_adapter_freeze_then_unfreeze_with_one_optimizer() -> None:
    torch.manual_seed(23)
    model = FGMVCSSRDecoupled1D().train()
    groups = model.parameter_groups()
    optimizer = torch.optim.AdamW(
        [
            {"params": groups["adapter"], "lr": 3.0e-4, "weight_decay": 1.0e-4},
            {
                "params": groups["autoencoders"],
                "lr": 1.0e-3,
                "weight_decay": 1.0e-4,
            },
        ]
    )
    assert {id(parameter) for group in optimizer.param_groups for parameter in group["params"]} == {
        id(parameter) for parameter in model.parameters()
    }
    features = torch.randn(5, 128, 7)
    targets = torch.arange(5)

    model.configure_for_epoch(1)
    adapter_before = _state(model.adapter)
    autoencoders_before = _state(model.class_autoencoders)
    optimizer.zero_grad(set_to_none=True)
    model.loss(features, targets, D2_DECOUPLED_ABSREL_CSSR).total_loss.backward()
    assert all(parameter.grad is None for parameter in model.adapter.parameters())
    optimizer.step()
    _assert_same_state(adapter_before, model.adapter.state_dict())
    assert _state_changed(autoencoders_before, model.class_autoencoders)

    model.configure_for_epoch(6)
    assert all(parameter.requires_grad for parameter in model.adapter.parameters())
    adapter_before = _state(model.adapter)
    optimizer.zero_grad(set_to_none=True)
    model.loss(features, targets, D2_DECOUPLED_ABSREL_CSSR).total_loss.backward()
    assert all(parameter.grad is not None for parameter in model.adapter.parameters())
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.adapter.parameters()
        if parameter.grad is not None
    )
    optimizer.step()
    assert _state_changed(adapter_before, model.adapter)


def test_shared_two_view_path_is_equivariant_to_view_swap() -> None:
    model = FGMVCSSRDecoupled1D().eval()
    feature_maps = torch.randn(
        2,
        2,
        128,
        9,
        generator=torch.Generator().manual_seed(29),
    )

    with torch.no_grad():
        original = model.forward_views(feature_maps)
        swapped = model.forward_views(feature_maps.flip(1))

    for field in (
        "adapted_features",
        "reconstructions",
        "reconstruction_errors",
        "logits",
        "normalized_reconstruction_errors",
    ):
        torch.testing.assert_close(
            getattr(swapped, field),
            getattr(original, field).flip(1),
            rtol=0.0,
            atol=0.0,
        )
    # The batched softmax kernel may differ by one float32 rounding unit after
    # reordering the flattened batch, while retaining the same semantics.
    torch.testing.assert_close(
        swapped.probabilities,
        original.probabilities.flip(1),
        rtol=1.0e-6,
        atol=1.0e-7,
    )


def test_invalid_method_and_epoch_fail_explicitly() -> None:
    model = FGMVCSSRDecoupled1D()
    with pytest.raises(ValueError, match="unknown decoupled CSSR"):
        model.loss(torch.randn(1, 128, 5), torch.tensor([0]), "not_registered")
    with pytest.raises(ValueError, match=r"\[1,20\]"):
        model.configure_for_epoch(0)
    with pytest.raises(ValueError, match=r"\[1,20\]"):
        model.configure_for_epoch(21)
