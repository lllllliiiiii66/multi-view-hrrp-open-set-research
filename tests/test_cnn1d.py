from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hrrp_osr.models.cnn1d import HRRPClassifier1D  # noqa: E402


def test_cnn_input_output_and_feature_shapes() -> None:
    model = HRRPClassifier1D()
    inputs = torch.randn(4, 601)
    features = model.forward_features(inputs)
    logits = model(inputs)
    assert features.shape == (4, 128)
    assert logits.shape == (4, 7)
    assert model.parameter_count > 0


def test_cnn_accepts_explicit_channel_dimension() -> None:
    model = HRRPClassifier1D()
    logits = model(torch.randn(2, 1, 601))
    assert logits.shape == (2, 7)


def test_cnn_rejects_wrong_profile_length() -> None:
    model = HRRPClassifier1D()
    with pytest.raises(ValueError, match="expected 601"):
        model(torch.randn(2, 600))


def test_seeded_initialization_is_reproducible() -> None:
    torch.manual_seed(20260810)
    first = HRRPClassifier1D()
    torch.manual_seed(20260810)
    second = HRRPClassifier1D()
    assert all(
        torch.equal(first_value, second_value)
        for first_value, second_value in zip(
            first.state_dict().values(), second.state_dict().values(), strict=True
        )
    )
