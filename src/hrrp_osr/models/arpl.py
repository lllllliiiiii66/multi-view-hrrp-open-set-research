from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .cnn1d import SharedHRRPEncoder1D


@dataclass(frozen=True)
class TwoViewModelOutput:
    per_view_features: torch.Tensor
    fused_features: torch.Tensor
    logits: torch.Tensor


@dataclass(frozen=True)
class TwoViewDetailedOutput:
    per_view_features: torch.Tensor
    fused_features: torch.Tensor
    per_view_logits: torch.Tensor
    fused_logits: torch.Tensor


@dataclass(frozen=True)
class ARPLLossOutput:
    logits: torch.Tensor
    total_loss: torch.Tensor
    classification_loss: torch.Tensor
    margin_loss: torch.Tensor
    true_class_reciprocal_distance: torch.Tensor


class TwoViewMeanEncoder(nn.Module):
    """Shared HRRP encoder followed by permutation-invariant mean pooling."""

    architecture_id = "shared_hrrp_encoder_1d_two_view_mean_v1"

    def __init__(self) -> None:
        super().__init__()
        self.encoder = SharedHRRPEncoder1D()
        self.feature_dim = self.encoder.feature_dim

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3 or inputs.shape[1:] != (2, self.encoder.input_length):
            raise ValueError("two-view HRRP input must have shape [batch, 2, 601]")
        batch, views, length = inputs.shape
        per_view = self.encoder(inputs.reshape(batch * views, length)).reshape(
            batch, views, self.feature_dim
        )
        return per_view, per_view.mean(dim=1)


def reciprocal_distances(
    features: torch.Tensor,
    reciprocal_points: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Official ARPL L2/m and dot distances, averaged over centers."""

    if features.ndim != 2:
        raise ValueError("features must have shape [batch, feature_dim]")
    if reciprocal_points.ndim != 3:
        raise ValueError(
            "reciprocal_points must have shape [classes, centers, feature_dim]"
        )
    if features.shape[1] != reciprocal_points.shape[2]:
        raise ValueError("feature and reciprocal-point dimensions differ")
    points = reciprocal_points.reshape(-1, reciprocal_points.shape[-1])
    squared_l2 = (
        features.pow(2).sum(dim=1, keepdim=True)
        - 2.0 * features @ points.t()
        + points.pow(2).sum(dim=1).unsqueeze(0)
    ) / float(features.shape[1])
    dot = features @ points.t()
    shape = (features.shape[0], reciprocal_points.shape[0], reciprocal_points.shape[1])
    return squared_l2.reshape(shape).mean(dim=2), dot.reshape(shape).mean(dim=2)


def maximum_logit_unknown_score(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("logits must have shape [batch, classes>=2]")
    return -logits.max(dim=1).values


class TwoViewCEClassifier(nn.Module):
    architecture_id = "shared_hrrp_encoder_1d_two_view_mean_ce_v1"

    def __init__(self, known_class_count: int) -> None:
        super().__init__()
        if known_class_count < 2:
            raise ValueError("known_class_count must be at least two")
        self.known_class_count = int(known_class_count)
        self.backbone = TwoViewMeanEncoder()
        self.classifier = nn.Linear(self.backbone.feature_dim, self.known_class_count)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.classifier.bias)

    def forward_representation(self, inputs: torch.Tensor) -> TwoViewModelOutput:
        per_view, fused = self.backbone(inputs)
        return TwoViewModelOutput(per_view, fused, self.classifier(fused))

    def forward_all_views(self, inputs: torch.Tensor) -> TwoViewDetailedOutput:
        per_view, fused = self.backbone(inputs)
        per_view_logits = self.classifier(per_view)
        return TwoViewDetailedOutput(
            per_view_features=per_view,
            fused_features=fused,
            per_view_logits=per_view_logits,
            fused_logits=self.classifier(fused),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_representation(inputs).logits

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class ARPLReciprocalHead(nn.Module):
    """Device-safe, one-center implementation aligned to official ARPLoss.py."""

    algorithm_id = "arpl_official_loss_one_center_device_safe_v1"

    def __init__(
        self,
        *,
        known_class_count: int,
        feature_dim: int,
        temperature: float = 1.0,
        weight_pl: float = 0.1,
        margin: float = 1.0,
        reciprocal_init_std: float = 0.1,
        initial_radius: float = 0.0,
    ) -> None:
        super().__init__()
        if known_class_count < 2 or feature_dim <= 0:
            raise ValueError("invalid ARPL head dimensions")
        if temperature <= 0.0 or weight_pl < 0.0 or margin < 0.0:
            raise ValueError("invalid ARPL loss hyperparameters")
        self.known_class_count = int(known_class_count)
        self.feature_dim = int(feature_dim)
        self.temperature = float(temperature)
        self.weight_pl = float(weight_pl)
        self.margin = float(margin)
        self.reciprocal_points = nn.Parameter(
            torch.empty(self.known_class_count, 1, self.feature_dim)
        )
        self.radius = nn.Parameter(torch.tensor([float(initial_radius)]))
        nn.init.normal_(self.reciprocal_points, mean=0.0, std=reciprocal_init_std)

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        squared_l2, dot = reciprocal_distances(features, self.reciprocal_points)
        return squared_l2 - dot

    def loss(self, features: torch.Tensor, labels: torch.Tensor) -> ARPLLossOutput:
        if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
            raise ValueError("labels must match the feature batch")
        logits = self.logits(features)
        classification = F.cross_entropy(logits / self.temperature, labels)
        true_points = self.reciprocal_points[labels, 0, :]
        distance = (features - true_points).pow(2).mean(dim=1)
        margin_loss = F.margin_ranking_loss(
            self.radius.expand_as(distance),
            distance,
            torch.ones_like(distance),
            margin=self.margin,
        )
        total = classification + self.weight_pl * margin_loss
        return ARPLLossOutput(
            logits=logits,
            total_loss=total,
            classification_loss=classification,
            margin_loss=margin_loss,
            true_class_reciprocal_distance=distance,
        )


class TwoViewARPLClassifier(nn.Module):
    architecture_id = "shared_hrrp_encoder_1d_two_view_mean_arpl_lite_v1"

    def __init__(
        self,
        known_class_count: int,
        *,
        temperature: float = 1.0,
        weight_pl: float = 0.1,
        margin: float = 1.0,
        reciprocal_init_std: float = 0.1,
        initial_radius: float = 0.0,
    ) -> None:
        super().__init__()
        self.known_class_count = int(known_class_count)
        self.backbone = TwoViewMeanEncoder()
        self.head = ARPLReciprocalHead(
            known_class_count=self.known_class_count,
            feature_dim=self.backbone.feature_dim,
            temperature=temperature,
            weight_pl=weight_pl,
            margin=margin,
            reciprocal_init_std=reciprocal_init_std,
            initial_radius=initial_radius,
        )

    def forward_representation(self, inputs: torch.Tensor) -> TwoViewModelOutput:
        per_view, fused = self.backbone(inputs)
        return TwoViewModelOutput(per_view, fused, self.head.logits(fused))

    def forward_all_views(self, inputs: torch.Tensor) -> TwoViewDetailedOutput:
        per_view, fused = self.backbone(inputs)
        shape = per_view.shape
        per_view_logits = self.head.logits(per_view.reshape(-1, shape[-1])).reshape(
            shape[0], shape[1], self.known_class_count
        )
        return TwoViewDetailedOutput(
            per_view_features=per_view,
            fused_features=fused,
            per_view_logits=per_view_logits,
            fused_logits=self.head.logits(fused),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_representation(inputs).logits

    def loss(self, inputs: torch.Tensor, labels: torch.Tensor) -> tuple[TwoViewModelOutput, ARPLLossOutput]:
        per_view, fused = self.backbone(inputs)
        loss_output = self.head.loss(fused, labels)
        return TwoViewModelOutput(per_view, fused, loss_output.logits), loss_output

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
