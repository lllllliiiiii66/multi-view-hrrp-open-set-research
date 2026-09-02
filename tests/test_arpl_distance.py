from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hrrp_osr.models.arpl import (  # noqa: E402
    ARPLReciprocalHead,
    maximum_logit_unknown_score,
    reciprocal_distances,
)


def test_official_arpl_distance_and_logit_hand_calculation() -> None:
    features = torch.tensor([[1.0, 2.0]])
    points = torch.tensor([[[0.0, 0.0]], [[1.0, -1.0]]])
    squared_l2, dot = reciprocal_distances(features, points)
    torch.testing.assert_close(squared_l2, torch.tensor([[2.5, 4.5]]))
    torch.testing.assert_close(dot, torch.tensor([[0.0, -1.0]]))
    torch.testing.assert_close(squared_l2 - dot, torch.tensor([[2.5, 5.5]]))


def test_arpl_margin_loss_matches_official_margin_ranking_definition() -> None:
    head = ARPLReciprocalHead(
        known_class_count=2,
        feature_dim=2,
        temperature=1.0,
        weight_pl=0.1,
        margin=1.0,
    )
    with torch.no_grad():
        head.reciprocal_points.copy_(
            torch.tensor([[[0.0, 0.0]], [[1.0, -1.0]]])
        )
        head.radius.fill_(3.0)
    output = head.loss(torch.tensor([[1.0, 2.0]]), torch.tensor([0]))
    assert output.true_class_reciprocal_distance.item() == pytest.approx(2.5)
    assert output.margin_loss.item() == pytest.approx(0.5)
    expected_classification = torch.nn.functional.cross_entropy(
        torch.tensor([[2.5, 5.5]]), torch.tensor([0])
    )
    assert output.classification_loss.item() == pytest.approx(
        expected_classification.item()
    )
    assert output.total_loss.item() == pytest.approx(
        expected_classification.item() + 0.05
    )


def test_maximum_logit_unknown_score_has_required_direction() -> None:
    logits = torch.tensor([[4.0, 1.0], [2.0, 1.0]])
    scores = maximum_logit_unknown_score(logits)
    assert scores.tolist() == pytest.approx([-4.0, -2.0])
    assert scores[0] < scores[1]
