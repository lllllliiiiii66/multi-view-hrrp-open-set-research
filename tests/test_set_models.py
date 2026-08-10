from __future__ import annotations

import itertools

import pytest

torch = pytest.importorskip("torch")

from hrrp_osr.models.sets import DeepSetsClassifier, SetTransformerClassifier  # noqa: E402


@pytest.mark.parametrize("model_class", [DeepSetsClassifier, SetTransformerClassifier])
def test_set_models_have_expected_shapes_and_no_position_interface(model_class) -> None:
    torch.manual_seed(7)
    model = model_class().eval()
    inputs = torch.randn(2, 3, 601)
    assert model.forward_features(inputs).shape == (2, 128)
    assert model(inputs).shape == (2, 7)


@pytest.mark.parametrize("model_class", [DeepSetsClassifier, SetTransformerClassifier])
def test_set_models_are_invariant_to_all_six_v3_permutations(model_class) -> None:
    torch.manual_seed(11)
    model = model_class().eval()
    inputs = torch.randn(1, 3, 601)
    reference = model(inputs)
    for permutation in itertools.permutations(range(3)):
        actual = model(inputs[:, permutation, :])
        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)


def test_b2_and_b3_use_same_encoder_architecture() -> None:
    b2 = DeepSetsClassifier()
    b3 = SetTransformerClassifier()
    assert b2.encoder.architecture_id == b3.encoder.architecture_id
    assert type(b2.encoder) is type(b3.encoder)
    assert b3.parameter_count > b2.parameter_count
