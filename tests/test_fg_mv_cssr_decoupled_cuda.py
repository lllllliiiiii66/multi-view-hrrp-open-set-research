from __future__ import annotations

import io

import numpy as np
import pytest
import torch

from hrrp_osr.models.cssr_decoupled_1d import (
    D1_DECOUPLED_REL_CSSR,
    D2_DECOUPLED_ABSREL_CSSR,
    FGMVCSSRDecoupled1D,
)
from hrrp_osr.training.fg_mv_cssr_decoupled import _infer_decoupled


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _model() -> FGMVCSSRDecoupled1D:
    torch.manual_seed(20260905)
    return FGMVCSSRDecoupled1D().cuda()


@pytest.mark.parametrize("method", [D1_DECOUPLED_REL_CSSR, D2_DECOUPLED_ABSREL_CSSR])
def test_cuda_forward_loss_backward_is_finite(method: str) -> None:
    model = _model()
    model.configure_for_epoch(6)
    features = torch.randn(7, 128, 76, device="cuda")
    labels = torch.arange(7, device="cuda") % 5
    result = model.loss(features, labels, method)
    result.total_loss.backward()
    assert torch.isfinite(result.total_loss)
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


def test_cuda_checkpoint_replay_is_bitwise_exact() -> None:
    model = _model().eval()
    rng = np.random.default_rng(20260905)
    features = rng.standard_normal((9, 128, 76), dtype=np.float32)
    expected = _infer_decoupled(model, features, device=torch.device("cuda"), batch_size=4)
    payload = io.BytesIO()
    torch.save(model.state_dict(), payload)
    payload.seek(0)
    restored = _model().eval()
    restored.load_state_dict(torch.load(payload, map_location="cuda", weights_only=True), strict=True)
    observed = _infer_decoupled(restored, features, device=torch.device("cuda"), batch_size=4)
    assert all(np.array_equal(observed[name], expected[name]) for name in expected)


def test_cuda_adapter_is_frozen_then_unfrozen() -> None:
    model = _model()
    adapter = tuple(model.adapter.parameters())
    model.configure_for_epoch(5)
    assert not any(parameter.requires_grad for parameter in adapter)
    model.configure_for_epoch(6)
    assert all(parameter.requires_grad for parameter in adapter)
