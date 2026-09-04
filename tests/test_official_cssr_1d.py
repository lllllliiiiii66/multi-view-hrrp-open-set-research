from __future__ import annotations

import copy

import pytest


torch = pytest.importorskip("torch")

from hrrp_osr.models.hrrp_ms_resnet import HRRPMultiScaleResNet1D  # noqa: E402
from hrrp_osr.models.official_cssr_1d import (  # noqa: E402
    MATCHED_LINEAR_CONTROL_1D,
    OFFICIAL_SEMANTICS_PCSSR_1D,
    HRRPFeatureMapEncoder1D,
    MatchedLinearHead1D,
    OfficialCSSRHRRPModel1D,
    OfficialPCSSRHead1D,
    official_pcssr_loss,
)


def test_official_pcssr_head_matches_explicit_formula_and_gradients() -> None:
    generator = torch.Generator().manual_seed(41)
    head = OfficialPCSSRHead1D(
        num_classes=3,
        input_channels=4,
        latent_channels=2,
        gamma=0.1,
    )
    features = torch.randn(5, 4, 7, generator=generator, requires_grad=True)
    output = head(features)

    explicit_reconstructions = []
    explicit_latents = []
    for autoencoder in head.class_autoencoders:
        latent = torch.tanh(autoencoder.encoder[0](features))
        explicit_latents.append(latent)
        explicit_reconstructions.append(autoencoder.decoder(latent))
    reconstructions = torch.stack(explicit_reconstructions, dim=1)
    latents = torch.stack(explicit_latents, dim=1)
    errors = (reconstructions - features[:, None]).abs().sum(dim=2)
    logits = torch.clamp(-0.1 * errors, min=-100.0, max=100.0)
    probabilities = torch.softmax(logits, dim=1).mean(dim=-1)

    torch.testing.assert_close(output.reconstructions, reconstructions)
    torch.testing.assert_close(output.latents, latents)
    torch.testing.assert_close(output.reconstruction_errors, errors)
    torch.testing.assert_close(output.logits, logits)
    torch.testing.assert_close(output.probabilities, probabilities)
    targets = torch.tensor([0, 1, 2, 1, 0])
    loss = official_pcssr_loss(output.probabilities, targets)
    expected = -torch.log(probabilities[torch.arange(5), targets]).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in head.parameters()
    )


def test_matched_linear_control_uses_frozen_gamma_and_softmax_average() -> None:
    head = MatchedLinearHead1D(
        num_classes=3,
        input_channels=2,
        gamma=0.1,
    )
    with torch.no_grad():
        head.classifier.weight.copy_(
            torch.tensor(
                [
                    [[1.0], [0.0]],
                    [[0.0], [1.0]],
                    [[-1.0], [-1.0]],
                ]
            )
        )
    features = torch.tensor([[[2.0, -1.0], [0.5, 3.0]]])
    output = head(features)
    expected_logits = 0.1 * torch.nn.functional.conv1d(
        features,
        head.classifier.weight,
    )
    torch.testing.assert_close(output.logits, expected_logits)
    torch.testing.assert_close(
        output.probabilities,
        torch.softmax(expected_logits, dim=1).mean(dim=-1),
    )
    assert head.architecture_id == MATCHED_LINEAR_CONTROL_1D


def test_feature_map_encoder_registers_only_deep_copied_stem_and_stages() -> None:
    source = HRRPMultiScaleResNet1D(dropout=0.0)
    source.eval()
    encoder = HRRPFeatureMapEncoder1D.from_r2_encoder(source)
    encoder.eval()

    registered_names = {name.split(".", 1)[0] for name, _ in encoder.named_parameters()}
    assert registered_names == {"stem", "stages"}
    assert not hasattr(encoder, "projection")
    assert encoder.stem is not source.stem
    assert encoder.stages is not source.stages
    for left, right in zip(
        encoder.state_dict().values(),
        {
            **{f"stem.{key}": value for key, value in source.stem.state_dict().items()},
            **{
                f"stages.{key}": value
                for key, value in source.stages.state_dict().items()
            },
        }.values(),
        strict=True,
    ):
        assert torch.equal(left, right)

    inputs = torch.randn(2, 601, generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        expected = source.forward_feature_map(inputs)
        actual = encoder(inputs)
    assert actual.shape == (2, 128, 76)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    original = copy.deepcopy(source.stem[0].weight)
    with torch.no_grad():
        encoder.stem[0].weight.add_(1.0)
    assert torch.equal(source.stem[0].weight, original)


@pytest.mark.parametrize(
    "head_kind",
    [OFFICIAL_SEMANTICS_PCSSR_1D, MATCHED_LINEAR_CONTROL_1D],
)
def test_official_hrrp_wrapper_has_expected_feature_and_head_shapes(
    head_kind: str,
) -> None:
    source = HRRPMultiScaleResNet1D(dropout=0.0)
    model = OfficialCSSRHRRPModel1D.from_r2_encoder(
        source,
        head_kind=head_kind,
        num_classes=3,
        latent_channels=4,
    )
    model.eval()
    inputs = torch.randn(2, 601, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        output = model(inputs)
    assert output.feature_maps.shape == (2, 128, 76)
    assert output.head_output.logits.shape == (2, 3, 76)
    assert output.head_output.probabilities.shape == (2, 3)


def test_official_pcssr_loss_rejects_invalid_targets() -> None:
    probabilities = torch.full((2, 3), 1.0 / 3.0)
    with pytest.raises(ValueError, match="outside"):
        official_pcssr_loss(probabilities, torch.tensor([0, 3]))
