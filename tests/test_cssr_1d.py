from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from hrrp_osr.models.cssr_1d import (
    PCSSR_CORE_1D,
    PCSSRCore1D,
    pcssr_nll_loss,
    scale_normalized_reconstruction_inconsistency,
    softmax_average,
)


class _OfficialAutoEncoder2D(nn.Module):
    """Minimal oracle from xyzedd/CSSR@d5a99e9 methods/cssr.py:150-185."""

    def __init__(self, channels: int, latent_channels: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, latent_channels, kernel_size=1, bias=False),
            nn.Tanh(),
        )
        self.decoder = nn.Conv2d(
            latent_channels,
            channels,
            kernel_size=1,
            bias=False,
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(inputs)
        return self.decoder(latent), latent


class _OfficialPCSSRCore2D(nn.Module):
    """Device-neutral oracle for official lines 188-224 and 478-500."""

    def __init__(
        self,
        num_classes: int,
        channels: int,
        latent_channels: int,
        gamma: float = 0.1,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.class_autoencoders = nn.ModuleList(
            [
                _OfficialAutoEncoder2D(channels, latent_channels)
                for _ in range(num_classes)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        reconstructions = []
        reconstruction_errors = []
        logits = []
        for autoencoder in self.class_autoencoders:
            reconstruction, _ = autoencoder(inputs)
            reconstructions.append(reconstruction)
            error = torch.norm(
                reconstruction - inputs,
                p=1,
                dim=1,
                keepdim=True,
            )
            reconstruction_errors.append(error)
            logits.append(torch.clamp(-self.gamma * error, -100.0, 100.0))
        reconstruction_tensor = torch.stack(reconstructions, dim=1)
        error_tensor = torch.cat(reconstruction_errors, dim=1)
        logit_tensor = torch.cat(logits, dim=1)
        probabilities = torch.softmax(logit_tensor, dim=1).mean(dim=(2, 3))
        return reconstruction_tensor, error_tensor, logit_tensor, probabilities


def _copy_2d_weights_to_1d(
    reference: _OfficialPCSSRCore2D,
    candidate: PCSSRCore1D,
) -> None:
    with torch.no_grad():
        for reference_ae, candidate_ae in zip(
            reference.class_autoencoders,
            candidate.class_autoencoders,
            strict=True,
        ):
            candidate_ae.encoder[0].weight.copy_(
                reference_ae.encoder[0].weight.squeeze(2)
            )
            candidate_ae.decoder.weight.copy_(reference_ae.decoder.weight.squeeze(2))


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (torch.float32, 1.0e-5, 1.0e-6),
        (torch.float64, 1.0e-9, 1.0e-11),
    ],
)
def test_conv1d_matches_official_conv2d_forward_and_gradients(
    dtype: torch.dtype,
    rtol: float,
    atol: float,
) -> None:
    torch.manual_seed(20260903)
    classes, channels, latent_channels = 3, 128, 64
    reference = _OfficialPCSSRCore2D(
        classes,
        channels,
        latent_channels,
    ).to(dtype=dtype)
    candidate = PCSSRCore1D(
        classes,
        input_channels=channels,
        latent_channels=latent_channels,
    ).to(dtype=dtype)
    _copy_2d_weights_to_1d(reference, candidate)

    one_dimensional = (
        torch.randn(2, channels, 7, dtype=dtype) * 0.2
    ).requires_grad_()
    two_dimensional = one_dimensional.detach().unsqueeze(2).requires_grad_()
    targets = torch.tensor([0, 2], dtype=torch.long)

    (
        reference_reconstruction,
        reference_errors,
        reference_logits,
        reference_probabilities,
    ) = reference(two_dimensional)
    candidate_loss, candidate_output = candidate.loss(one_dimensional, targets)
    one_hot = F.one_hot(targets, num_classes=classes).to(dtype=dtype)
    reference_loss = -(
        one_hot * torch.log(reference_probabilities)
    ).sum(dim=1).mean()

    torch.testing.assert_close(
        candidate_output.reconstructions,
        reference_reconstruction.squeeze(3),
        rtol=rtol,
        atol=atol,
    )
    torch.testing.assert_close(
        candidate_output.reconstruction_errors,
        reference_errors.squeeze(2),
        rtol=rtol,
        atol=atol,
    )
    torch.testing.assert_close(
        candidate_output.logits,
        reference_logits.squeeze(2),
        rtol=rtol,
        atol=atol,
    )
    torch.testing.assert_close(
        candidate_output.probabilities,
        reference_probabilities,
        rtol=rtol,
        atol=atol,
    )
    torch.testing.assert_close(candidate_loss, reference_loss, rtol=rtol, atol=atol)

    candidate_loss.backward()
    reference_loss.backward()
    torch.testing.assert_close(
        one_dimensional.grad,
        two_dimensional.grad.squeeze(2),
        rtol=rtol,
        atol=atol,
    )
    for reference_ae, candidate_ae in zip(
        reference.class_autoencoders,
        candidate.class_autoencoders,
        strict=True,
    ):
        torch.testing.assert_close(
            candidate_ae.encoder[0].weight.grad,
            reference_ae.encoder[0].weight.grad.squeeze(2),
            rtol=rtol,
            atol=atol,
        )
        torch.testing.assert_close(
            candidate_ae.decoder.weight.grad,
            reference_ae.decoder.weight.grad.squeeze(2),
            rtol=rtol,
            atol=atol,
        )


def test_default_structure_is_independent_128_64_128_autoencoders() -> None:
    model = PCSSRCore1D(num_classes=4)

    assert PCSSR_CORE_1D is PCSSRCore1D
    assert model.architecture_id == "PCSSR_CORE_1D"
    assert model.gamma == 0.1
    assert model.clip_length == 100.0
    assert model.epsilon == 1.0e-8
    assert len(model.class_autoencoders) == 4
    parameter_addresses = []
    for autoencoder in model.class_autoencoders:
        assert list(autoencoder.encoder.children())[1].__class__ is nn.Tanh
        encoder = autoencoder.encoder[0]
        decoder = autoencoder.decoder
        assert isinstance(encoder, nn.Conv1d)
        assert isinstance(decoder, nn.Conv1d)
        assert encoder.weight.shape == (64, 128, 1)
        assert decoder.weight.shape == (128, 64, 1)
        assert encoder.bias is None
        assert decoder.bias is None
        assert encoder.kernel_size == decoder.kernel_size == (1,)
        leaf_modules = [
            module
            for module in autoencoder.modules()
            if len(list(module.children())) == 0
        ]
        assert [type(module) for module in leaf_modules] == [
            nn.Conv1d,
            nn.Tanh,
            nn.Conv1d,
        ]
        parameter_addresses.extend(
            [encoder.weight.data_ptr(), decoder.weight.data_ptr()]
        )
    assert len(parameter_addresses) == len(set(parameter_addresses))


def test_l1_logits_are_clipped_after_gamma_scaling() -> None:
    model = PCSSRCore1D(num_classes=3)
    with torch.no_grad():
        for autoencoder in model.class_autoencoders:
            autoencoder.encoder[0].weight.zero_()
            autoencoder.decoder.weight.zero_()

    inputs = torch.full((2, 128, 5), 100.0)
    output = model(inputs)

    torch.testing.assert_close(
        output.reconstruction_errors,
        torch.full((2, 3, 5), 12_800.0),
    )
    torch.testing.assert_close(output.logits, torch.full((2, 3, 5), -100.0))
    torch.testing.assert_close(
        output.probabilities,
        torch.full((2, 3), 1.0 / 3.0),
    )


def test_epsilon_floor_keeps_scale_normalized_rho_finite() -> None:
    model = PCSSRCore1D(num_classes=2, epsilon=1.0e-8)
    with torch.no_grad():
        for autoencoder in model.class_autoencoders:
            autoencoder.encoder[0].weight.zero_()
            autoencoder.decoder.weight.zero_()

    zero_inputs = torch.zeros(1, 128, 3)
    zero_rho = model.reconstruction_inconsistency(zero_inputs)
    assert torch.isfinite(zero_rho).all()
    torch.testing.assert_close(zero_rho, torch.zeros(1, 2))

    inputs = torch.full((1, 128, 3), 1.0e-12)
    output = model(inputs)
    rho = model.reconstruction_inconsistency(inputs, output=output)
    expected = torch.full((1, 2), 0.1 * 128 * 1.0e-12 / (1.0e-8**2))

    assert torch.isfinite(rho).all()
    torch.testing.assert_close(rho, expected)


def test_changing_one_class_autoencoder_does_not_change_other_classes() -> None:
    torch.manual_seed(41)
    model = PCSSRCore1D(num_classes=3)
    inputs = torch.randn(2, 128, 5)
    before = model(inputs).reconstructions.detach().clone()

    with torch.no_grad():
        model.class_autoencoders[0].decoder.weight.add_(0.5)
    after = model(inputs).reconstructions.detach()

    assert not torch.equal(before[:, 0], after[:, 0])
    torch.testing.assert_close(before[:, 1:], after[:, 1:], rtol=0.0, atol=0.0)


def test_softmax_is_applied_before_position_average() -> None:
    logits = torch.tensor([[[10.0, 0.0], [0.0, 0.0]]])

    actual = softmax_average(logits)
    expected = torch.softmax(logits, dim=1).mean(dim=-1)
    forbidden_avg_softmax = torch.softmax(logits.mean(dim=-1), dim=1)

    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, forbidden_avg_softmax)


def test_scale_normalization_is_pointwise_before_position_average() -> None:
    logits = torch.tensor([[[-2.0, -8.0], [-4.0, -1.0]]])
    features = torch.tensor([[[1.0, 2.0], [1.0, 2.0]]])

    actual = scale_normalized_reconstruction_inconsistency(
        logits,
        features,
        epsilon=1.0e-8,
    )
    activation_scale = features.abs().mean(dim=1)
    expected = (-logits / activation_scale.square().unsqueeze(1)).mean(dim=-1)
    forbidden_global_ratio = -logits.mean(dim=-1) / activation_scale.mean(
        dim=-1, keepdim=True
    ).square()

    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, forbidden_global_ratio)


def test_nll_is_computed_after_probability_average() -> None:
    logits = torch.tensor([[[4.0, 0.0], [0.0, 4.0]]])
    targets = torch.tensor([0])
    probabilities = softmax_average(logits)

    actual = pcssr_nll_loss(probabilities, targets)
    expected = -torch.log(probabilities[0, 0])
    forbidden_positionwise_ce = -torch.log_softmax(logits, dim=1)[0, 0].mean()

    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, forbidden_positionwise_ce)
