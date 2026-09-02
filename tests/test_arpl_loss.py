from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hrrp_osr.models.arpl import (  # noqa: E402
    TwoViewARPLClassifier,
    TwoViewCEClassifier,
)


def test_arpl_lite_cpu_backward_and_parameter_gradients() -> None:
    torch.manual_seed(20260830)
    model = TwoViewARPLClassifier(known_class_count=5)
    inputs = torch.randn(4, 2, 601)
    labels = torch.tensor([0, 1, 2, 3])
    output, loss = model.loss(inputs, labels)
    assert output.per_view_features.shape == (4, 2, 128)
    assert output.fused_features.shape == (4, 128)
    assert output.logits.shape == (4, 5)
    loss.total_loss.backward()
    assert model.head.reciprocal_points.grad is not None
    assert model.head.radius.grad is not None
    assert torch.isfinite(model.head.reciprocal_points.grad).all()
    assert torch.isfinite(model.head.radius.grad).all()
    assert any(
        parameter.grad is not None
        for parameter in model.backbone.encoder.parameters()
    )


@pytest.mark.parametrize("model_type", [TwoViewCEClassifier, TwoViewARPLClassifier])
def test_two_view_mean_models_are_permutation_invariant(model_type) -> None:
    torch.manual_seed(7)
    model = model_type(known_class_count=5).eval()
    inputs = torch.randn(3, 2, 601)
    with torch.no_grad():
        original = model.forward_representation(inputs)
        swapped = model.forward_representation(inputs[:, [1, 0], :])
    assert torch.allclose(
        original.per_view_features[:, [1, 0], :],
        swapped.per_view_features,
        atol=1e-6,
    )
    assert torch.allclose(original.fused_features, swapped.fused_features, atol=1e-6)
    assert torch.allclose(original.logits, swapped.logits, atol=1e-6)


def test_ce_and_arpl_start_from_identical_seeded_backbone() -> None:
    torch.manual_seed(20260830)
    ce = TwoViewCEClassifier(known_class_count=5)
    torch.manual_seed(20260830)
    arpl = TwoViewARPLClassifier(known_class_count=5)
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            ce.backbone.state_dict().values(),
            arpl.backbone.state_dict().values(),
            strict=True,
        )
    )


def test_two_view_models_reject_wrong_shape() -> None:
    model = TwoViewCEClassifier(known_class_count=5)
    with pytest.raises(ValueError, match=r"\[batch, 2, 601\]"):
        model(torch.randn(2, 3, 601))
