from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from hrrp_osr.models.arpl import ARPLReciprocalHead  # noqa: E402
from hrrp_osr.models.ms_mean_factorial import (  # noqa: E402
    ARPL_METHODS,
    CE_METHODS,
    METHODS,
    MSMeanHeadFactorialModel,
    clone_state_dict,
)
from hrrp_osr.models.mv_rpformer import (  # noqa: E402
    MVRPFormer,
    PoolingByMultiheadAttention,
    PreNormSAB,
    SmallRejector,
)
from hrrp_osr.training.arpl_pilot import _state_sha256  # noqa: E402


def _seeded_model(method: str, seed: int = 20260830) -> MSMeanHeadFactorialModel:
    torch.manual_seed(seed)
    return MSMeanHeadFactorialModel(method, known_class_count=5)


def _assert_equal_independent_state(left: nn.Module, right: nn.Module) -> None:
    left_state = left.state_dict()
    right_state = right.state_dict()
    assert left_state.keys() == right_state.keys()
    for name in left_state:
        assert torch.equal(left_state[name], right_state[name]), name
        assert left_state[name].data_ptr() != right_state[name].data_ptr(), name


@pytest.mark.parametrize(
    ("left_method", "right_method"),
    [
        ("R0_SHALLOW_MEAN_CE", "R1_SHALLOW_MEAN_ARPL"),
        ("R2_MS_MEAN_CE", "R3_MS_MEAN_ARPL"),
    ],
)
def test_paired_heads_start_from_equal_but_independent_backbones(
    left_method: str, right_method: str
) -> None:
    left = _seeded_model(left_method)
    right = _seeded_model(right_method)

    assert type(left.encoder) is type(right.encoder)
    assert _state_sha256(left.encoder.state_dict()) == _state_sha256(
        right.encoder.state_dict()
    )
    _assert_equal_independent_state(left.encoder, right.encoder)

    snapshot = clone_state_dict(left.encoder.state_dict())
    assert _state_sha256(snapshot) == _state_sha256(left.encoder.state_dict())
    assert all(value.device.type == "cpu" for value in snapshot.values())
    assert all(
        snapshot[name].data_ptr() != left.encoder.state_dict()[name].data_ptr()
        for name in snapshot
    )


def test_r2_and_r3_have_identical_multiscale_structure_except_for_the_head() -> None:
    r2 = _seeded_model("R2_MS_MEAN_CE")
    r3 = _seeded_model("R3_MS_MEAN_ARPL")

    assert set(dict(r2.named_children())) == {"encoder", "global_head"}
    assert set(dict(r3.named_children())) == {"encoder", "global_head"}
    assert type(r2.encoder) is type(r3.encoder)
    assert list(r2.encoder.state_dict()) == list(r3.encoder.state_dict())
    assert isinstance(r2.global_head, nn.Linear)
    assert isinstance(r3.global_head, ARPLReciprocalHead)


def test_r2_feature_map_interface_preserves_legacy_forward_and_checkpoint_state() -> None:
    r2 = _seeded_model("R2_MS_MEAN_CE", seed=20260903).eval()
    encoder = r2.encoder
    inputs = torch.randn(3, 601, generator=torch.Generator().manual_seed(731))

    checkpoint_state = clone_state_dict(r2.state_dict())
    state_keys_before = tuple(r2.state_dict())
    parameter_names_before = tuple(name for name, _ in r2.named_parameters())
    parameter_count_before = sum(parameter.numel() for parameter in r2.parameters())

    with torch.no_grad():
        legacy_feature_map = encoder.stages(encoder.stem(inputs.unsqueeze(1)))
        legacy_pooled = torch.cat(
            [
                encoder.average_pool(legacy_feature_map).flatten(1),
                encoder.maximum_pool(legacy_feature_map).flatten(1),
            ],
            dim=1,
        )
        legacy_forward = encoder.projection(legacy_pooled)
        exposed_feature_map = encoder.forward_feature_map(inputs)
        current_forward = encoder(inputs)

    assert exposed_feature_map.shape == (3, 128, 76)
    assert torch.equal(exposed_feature_map, legacy_feature_map)
    assert torch.equal(current_forward, legacy_forward)
    assert tuple(r2.state_dict()) == state_keys_before
    assert tuple(name for name, _ in r2.named_parameters()) == parameter_names_before
    assert sum(parameter.numel() for parameter in r2.parameters()) == parameter_count_before
    assert not any("feature_map" in name for name in r2.state_dict())

    restored = _seeded_model("R2_MS_MEAN_CE", seed=17)
    incompatible = restored.load_state_dict(checkpoint_state, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert tuple(restored.state_dict()) == state_keys_before
    assert _state_sha256(restored.state_dict()) == _state_sha256(checkpoint_state)


def test_r3_is_forward_loss_and_gradient_equivalent_to_legacy_m2() -> None:
    model_seed = 20260831
    torch.manual_seed(model_seed)
    legacy = MVRPFormer("M2_MS_MEAN_ARPL", known_class_count=5)
    torch.manual_seed(model_seed)
    r3 = MSMeanHeadFactorialModel("R3_MS_MEAN_ARPL", known_class_count=5)

    assert legacy.state_dict().keys() == r3.state_dict().keys()
    assert _state_sha256(legacy.state_dict()) == _state_sha256(r3.state_dict())
    _assert_equal_independent_state(legacy, r3)

    generator = torch.Generator().manual_seed(20260903)
    input_values = torch.randn(2, 2, 601, generator=generator)
    legacy_inputs = input_values.clone().requires_grad_(True)
    r3_inputs = input_values.clone().requires_grad_(True)
    labels = torch.tensor([0, 3], dtype=torch.long)

    legacy.train()
    r3.train()
    forward_seed = 731
    torch.manual_seed(forward_seed)
    legacy_output = legacy(legacy_inputs, compute_rejector=False)
    legacy_loss = legacy.representation_loss(legacy_output, labels)
    legacy_loss["total"].backward()

    torch.manual_seed(forward_seed)
    r3_output = r3(r3_inputs, compute_rejector=False)
    r3_loss = r3.representation_loss(r3_output, labels)
    r3_loss["total"].backward()

    for field in (
        "raw_view_tokens",
        "contextual_view_tokens",
        "global_class_token",
        "global_reject_token",
        "per_view_logits",
        "global_logits",
        "sab_attention",
        "pma_attention",
    ):
        torch.testing.assert_close(
            getattr(r3_output, field),
            getattr(legacy_output, field),
            rtol=0.0,
            atol=0.0,
        )
    assert legacy_output.reject_evidence is None
    assert r3_output.reject_evidence is None
    assert legacy_output.unknown_probability is None
    assert r3_output.unknown_probability is None
    for name in legacy_loss:
        torch.testing.assert_close(r3_loss[name], legacy_loss[name], rtol=0.0, atol=0.0)
    torch.testing.assert_close(r3_inputs.grad, legacy_inputs.grad, rtol=0.0, atol=0.0)

    legacy_parameters = dict(legacy.named_parameters())
    r3_parameters = dict(r3.named_parameters())
    assert legacy_parameters.keys() == r3_parameters.keys()
    for name in legacy_parameters:
        assert legacy_parameters[name].grad is not None, name
        assert r3_parameters[name].grad is not None, name
        torch.testing.assert_close(
            r3_parameters[name].grad,
            legacy_parameters[name].grad,
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.parametrize("method", METHODS)
def test_all_factorial_models_are_two_view_swap_invariant(method: str) -> None:
    model = _seeded_model(method, seed=43).eval()
    inputs = torch.randn(2, 2, 601, generator=torch.Generator().manual_seed(44))

    with torch.no_grad():
        original = model(inputs, compute_rejector=False)
        swapped = model(inputs[:, [1, 0]], compute_rejector=False)

    torch.testing.assert_close(
        original.raw_view_tokens[:, [1, 0]],
        swapped.raw_view_tokens,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        original.contextual_view_tokens[:, [1, 0]],
        swapped.contextual_view_tokens,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        original.per_view_logits[:, [1, 0]],
        swapped.per_view_logits,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        original.global_class_token,
        swapped.global_class_token,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        original.global_logits,
        swapped.global_logits,
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize("method", METHODS)
def test_factorial_models_never_create_forbidden_components(method: str) -> None:
    model = _seeded_model(method, seed=47)

    assert model.forbidden_component_status == {
        "sab_created": False,
        "pma_created": False,
        "view_head_created": False,
        "rejector_created": False,
        "pseudo_unknown_supported": False,
    }
    assert model.sab is None
    assert model.pma is None
    assert model.view_head is None
    assert model.rejector is None
    assert not any(
        isinstance(module, (PreNormSAB, PoolingByMultiheadAttention, SmallRejector))
        for module in model.modules()
    )
    assert not any(
        forbidden in name
        for name in model.state_dict()
        for forbidden in ("sab", "pma", "view_head", "rejector")
    )
    assert model.head_type == ("ce" if method in CE_METHODS else "arpl")
    assert model.uses_arpl is (method in ARPL_METHODS)
    assert not model.uses_set_transformer
    assert not model.uses_hierarchical_head
    assert not model.uses_rejector
