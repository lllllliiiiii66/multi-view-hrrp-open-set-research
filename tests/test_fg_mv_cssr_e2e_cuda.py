from __future__ import annotations

import os
from pathlib import Path

import pytest


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

torch = pytest.importorskip("torch")

from hrrp_osr.models.cssr_e2e_1d import (  # noqa: E402
    CSSR_VARIANTS,
    FGMVCSSRE2EModel,
    Q1_CE_FINETUNE_CONTROL,
    Q2_E2E_REL_CSSR_1X1,
    Q3_E2E_ABSREL_CSSR_1X1,
    Q4_E2E_ABSREL_CSSR_LOCAL3,
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
FROZEN_PREFIXES = (
    "r2_model.encoder.stem.",
    "r2_model.encoder.stages.0.",
    "r2_model.encoder.stages.1.",
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable; the pre-smoke GPU guard requires a CUDA device",
)


def _frozen_state(model: FGMVCSSRE2EModel) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if name.startswith(FROZEN_PREFIXES)
    }


def _assert_state_equal(
    expected: dict[str, torch.Tensor], observed: dict[str, torch.Tensor]
) -> None:
    assert expected.keys() == observed.keys()
    for name, value in expected.items():
        assert torch.equal(value, observed[name]), name


def _cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


@pytest.mark.parametrize("variant", VARIANTS)
def test_q1_q4_cuda_forward_loss_backward_frozen_state_and_checkpoint_roundtrip(
    variant: str,
    tmp_path: Path,
) -> None:
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20260904)
    torch.cuda.manual_seed_all(20260904)
    source = MSMeanHeadFactorialModel("R2_MS_MEAN_CE", known_class_count=5)
    source_state = clone_state_dict(source.state_dict())
    model = FGMVCSSRE2EModel.from_r2_state_dict(
        source_state,
        variant,
        known_class_count=5,
        autoencoder_seed=20260904,
    ).to(device)
    model.train()

    frozen_before = _frozen_state(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1.0e-4,
    )
    inputs = torch.randn(2, 2, 601, device=device)
    labels = torch.tensor([0, 4], dtype=torch.long, device=device)

    optimizer.zero_grad(set_to_none=True)
    output = model(inputs)
    losses = model.loss(output, labels)
    assert output.feature_maps.shape == (2, 2, 128, 76)
    assert output.fused_logits.shape == (2, 5)
    assert output.fused_logits.is_cuda
    assert losses.total_loss.is_cuda
    assert torch.isfinite(losses.total_loss)
    losses.total_loss.backward()

    frozen_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith(FROZEN_PREFIXES)
    ]
    assert frozen_parameters
    assert all(parameter.grad is None for parameter in frozen_parameters)
    for group_name, parameters in model.trainable_parameter_groups().items():
        if group_name == "autoencoders" and variant not in CSSR_VARIANTS:
            assert parameters == ()
            continue
        assert parameters
        gradients = [parameter.grad for parameter in parameters]
        assert any(
            gradient is not None
            and torch.isfinite(gradient).all()
            and bool(torch.count_nonzero(gradient))
            for gradient in gradients
        ), group_name

    optimizer.step()
    torch.cuda.synchronize(device)
    _assert_state_equal(frozen_before, _frozen_state(model))

    checkpoint_path = tmp_path / f"{variant}.pt"
    torch.save(
        {
            "variant": variant,
            "r2_model_state_dict": _cpu_state(model.r2_model),
            "model_state_dict": _cpu_state(model),
        },
        checkpoint_path,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    restored = FGMVCSSRE2EModel.from_r2_state_dict(
        checkpoint["r2_model_state_dict"],
        checkpoint["variant"],
        known_class_count=5,
        autoencoder_seed=20260904,
    )
    incompatible = restored.load_state_dict(
        checkpoint["model_state_dict"], strict=True
    )
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    _assert_state_equal(checkpoint["model_state_dict"], _cpu_state(restored))

    model.eval()
    restored.to(device).eval()
    replay_inputs = torch.randn(2, 2, 601, device=device)
    with torch.no_grad():
        expected = model(replay_inputs)
        actual = restored(replay_inputs)
    assert torch.equal(expected.feature_maps, actual.feature_maps)
    assert torch.equal(expected.per_view_features, actual.per_view_features)
    assert torch.equal(expected.fused_features, actual.fused_features)
    assert torch.equal(expected.fused_logits, actual.fused_logits)
    if variant in CSSR_VARIANTS:
        assert torch.equal(
            expected.normalized_reconstruction_errors,
            actual.normalized_reconstruction_errors,
        )
    else:
        assert expected.normalized_reconstruction_errors is None
        assert actual.normalized_reconstruction_errors is None
