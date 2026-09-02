from __future__ import annotations

import hashlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from hrrp_osr.models.arpl import (  # noqa: E402
    ARPLReciprocalHead,
    reciprocal_distances,
)


FIXTURE = Path(__file__).parent / "fixtures/arpl_official_3ede8b38"


def _load_official_arploss():
    loss_root = FIXTURE / "loss"
    package = types.ModuleType("loss")
    package.__path__ = [str(loss_root)]
    sys.modules["loss"] = package
    dist_spec = importlib.util.spec_from_file_location("loss.Dist", loss_root / "Dist.py")
    dist_module = importlib.util.module_from_spec(dist_spec)
    sys.modules["loss.Dist"] = dist_module
    assert dist_spec.loader is not None
    dist_spec.loader.exec_module(dist_module)
    loss_spec = importlib.util.spec_from_file_location(
        "loss.ARPLoss", loss_root / "ARPLoss.py"
    )
    loss_module = importlib.util.module_from_spec(loss_spec)
    sys.modules["loss.ARPLoss"] = loss_module
    assert loss_spec.loader is not None
    loss_spec.loader.exec_module(loss_module)
    return loss_module.ARPLoss


def test_vendored_official_snapshot_hash_and_commit() -> None:
    assert (FIXTURE / "COMMIT").read_text().strip() == (
        "3ede8b38e1cfb9d70e106cc19d563453110c36ab"
    )
    expected = {
        "Dist.py": "a05fc01c9051d8cb8d87cc7183e0a3d9fd1a11ca9de38d58a4870cb70ad4dc62",
        "ARPLoss.py": "6dec41f0265b6665e8c66a27f506f176a0a7b0b2e4426760c09c203ab0c327ec",
    }
    for name, digest in expected.items():
        assert hashlib.sha256((FIXTURE / "loss" / name).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [(torch.float32, 1e-5, 1e-6), (torch.float64, 1e-9, 1e-10)],
)
def test_official_and_project_arpl_forward_and_gradients_are_equivalent(
    monkeypatch, dtype, rtol, atol
) -> None:
    official_type = _load_official_arploss()
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    generator = torch.Generator().manual_seed(20260903)
    features_data = torch.randn(7, 6, generator=generator, dtype=dtype)
    points_data = torch.randn(4, 6, generator=generator, dtype=dtype) * 0.1
    labels = torch.tensor([0, 1, 2, 3, 0, 2, 1], dtype=torch.long)
    radius_value = 0.37
    temperature = 0.83
    weight_pl = 0.17

    official = official_type(
        use_gpu=False,
        weight_pl=weight_pl,
        temp=temperature,
        num_classes=4,
        feat_dim=6,
    ).to(dtype=dtype)
    with torch.no_grad():
        official.points.copy_(points_data)
        official.radius.fill_(radius_value)
    official_features = features_data.clone().requires_grad_(True)
    official_l2 = official.Dist(official_features, center=official.points)
    official_dot = official.Dist(
        official_features, center=official.points, metric="dot"
    )
    official_logits, official_total = official(
        official_features, labels, labels=labels
    )
    official_classification = torch.nn.functional.cross_entropy(
        official_logits / temperature, labels
    )
    official_distance = (
        official_features - official.points[labels]
    ).pow(2).mean(dim=1)
    official_margin = official.margin_loss(
        official.radius, official_distance, torch.ones_like(official_distance)
    )
    official_grads = torch.autograd.grad(
        official_total,
        (official_features, official.points, official.radius),
    )

    project = ARPLReciprocalHead(
        known_class_count=4,
        feature_dim=6,
        temperature=temperature,
        weight_pl=weight_pl,
        margin=1.0,
    ).to(dtype=dtype)
    with torch.no_grad():
        project.reciprocal_points[:, 0].copy_(points_data)
        project.radius.fill_(radius_value)
    project_features = features_data.clone().requires_grad_(True)
    project_output = project.loss(project_features, labels)
    project_l2, project_dot = reciprocal_distances(
        project_features, project.reciprocal_points
    )
    project_grads = torch.autograd.grad(
        project_output.total_loss,
        (project_features, project.reciprocal_points, project.radius),
    )

    torch.testing.assert_close(project_l2, official_l2, rtol=rtol, atol=atol)
    torch.testing.assert_close(project_dot, official_dot, rtol=rtol, atol=atol)
    torch.testing.assert_close(
        project_output.logits, official_logits, rtol=rtol, atol=atol
    )
    torch.testing.assert_close(
        project_output.classification_loss,
        official_classification,
        rtol=rtol,
        atol=atol,
    )
    torch.testing.assert_close(
        project_output.margin_loss, official_margin, rtol=rtol, atol=atol
    )
    torch.testing.assert_close(
        project_output.total_loss, official_total, rtol=rtol, atol=atol
    )
    torch.testing.assert_close(project_grads[0], official_grads[0], rtol=rtol, atol=atol)
    torch.testing.assert_close(
        project_grads[1][:, 0], official_grads[1], rtol=rtol, atol=atol
    )
    torch.testing.assert_close(project_grads[2], official_grads[2], rtol=rtol, atol=atol)
