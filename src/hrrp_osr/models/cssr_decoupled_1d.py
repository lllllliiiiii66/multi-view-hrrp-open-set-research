from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .cssr_1d import pcssr_nll_loss, softmax_average


D1_DECOUPLED_REL_CSSR = "D1_DECOUPLED_REL_CSSR"
D2_DECOUPLED_ABSREL_CSSR = "D2_DECOUPLED_ABSREL_CSSR"
DECOUPLED_METHODS = (
    D1_DECOUPLED_REL_CSSR,
    D2_DECOUPLED_ABSREL_CSSR,
)
DECOUPLED_CSSR_VARIANTS = DECOUPLED_METHODS

LOSS_WEIGHTS: Mapping[str, Mapping[str, float]] = {
    D1_DECOUPLED_REL_CSSR: {
        "relative": 1.0,
        "absolute": 0.0,
        "separation": 0.0,
    },
    D2_DECOUPLED_ABSREL_CSSR: {
        "relative": 1.0,
        "absolute": 0.25,
        "separation": 0.5,
    },
}


@dataclass(frozen=True)
class DecoupledCSSROutput:
    """Inspectable single-view output of the decoupled CSSR path."""

    adapted_features: torch.Tensor
    reconstructions: torch.Tensor
    reconstruction_errors: torch.Tensor
    logits: torch.Tensor
    probabilities: torch.Tensor
    normalized_reconstruction_errors: torch.Tensor


@dataclass(frozen=True)
class DecoupledCSSRViewOutput:
    """Two-view output produced by one shared adapter and one shared AE bank."""

    adapted_features: torch.Tensor
    reconstructions: torch.Tensor
    reconstruction_errors: torch.Tensor
    logits: torch.Tensor
    probabilities: torch.Tensor
    normalized_reconstruction_errors: torch.Tensor


@dataclass(frozen=True)
class DecoupledCSSRLoss:
    """Frozen D1/D2 loss terms and per-example reconstruction diagnostics."""

    total_loss: torch.Tensor
    relative_loss: torch.Tensor
    absolute_loss: torch.Tensor
    separation_loss: torch.Tensor
    true_class_r: torch.Tensor
    nearest_wrong_class_r: torch.Tensor
    reconstruction_margin: torch.Tensor
    component_weights: Mapping[str, float]
    output: DecoupledCSSROutput


class SharedCSSRSemanticAdapter1D(nn.Module):
    """The single view-shared adapter reserved exclusively for CSSR."""

    architecture_id = "fg_mv_cssr_shared_semantic_adapter_1d_v1"

    def __init__(
        self,
        *,
        input_channels: int = 128,
        hidden_channels: int = 64,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden_channels)
        self.residual_scale = float(residual_scale)
        self.delta = nn.Sequential(
            nn.Conv1d(
                self.input_channels,
                self.hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(8, self.hidden_channels, eps=1.0e-5, affine=True),
            nn.GELU(),
            nn.Conv1d(
                self.hidden_channels,
                self.input_channels,
                kernel_size=1,
                bias=False,
            ),
        )

    def forward(self, feature_maps: torch.Tensor) -> torch.Tensor:
        if feature_maps.ndim != 3 or feature_maps.shape[1] != self.input_channels:
            raise ValueError(
                f"CSSR feature maps must have shape [batch,{self.input_channels},length]"
            )
        return feature_maps + self.residual_scale * self.delta(feature_maps)


class DecoupledClassSpecificAutoEncoder1D(nn.Module):
    """One independent no-skip local autoencoder for a single known class."""

    def __init__(
        self,
        *,
        input_channels: int = 128,
        latent_channels: int = 32,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.latent_channels = int(latent_channels)
        self.encoder = nn.Sequential(
            nn.Conv1d(
                self.input_channels,
                self.latent_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.Tanh(),
        )
        self.decoder = nn.Conv1d(
            self.latent_channels,
            self.input_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(inputs)
        return self.decoder(latent), latent


def normalized_absolute_reconstruction_error(
    features: torch.Tensor,
    reconstructions: torch.Tensor,
    *,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Compute the frozen all-map normalized absolute error ``r_k``."""

    if features.ndim != 3:
        raise ValueError("features must have shape [batch,channels,length]")
    if reconstructions.ndim != 4:
        raise ValueError(
            "reconstructions must have shape [batch,classes,channels,length]"
        )
    if (
        reconstructions.shape[0] != features.shape[0]
        or reconstructions.shape[2:] != features.shape[1:]
    ):
        raise ValueError("feature and reconstruction dimensions differ")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    numerator = (reconstructions - features.unsqueeze(1)).abs().mean(dim=(2, 3))
    denominator = features.abs().mean(dim=(1, 2)).unsqueeze(1) + float(epsilon)
    return numerator / denominator


def absolute_and_separation_losses(
    normalized_errors: torch.Tensor,
    targets: torch.Tensor,
    *,
    separation_margin: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return L_abs, L_sep, true r, nearest-wrong r, and wrong-minus-true margin."""

    if normalized_errors.ndim != 2:
        raise ValueError("normalized reconstruction errors must be [batch,classes]")
    if targets.ndim != 1 or targets.shape[0] != normalized_errors.shape[0]:
        raise ValueError("targets must have shape [batch]")
    if normalized_errors.shape[1] <= 1:
        raise ValueError("at least two reconstruction classes are required")
    if separation_margin < 0.0:
        raise ValueError("separation margin must be non-negative")
    if targets.dtype != torch.long:
        targets = targets.long()
    if bool(((targets < 0) | (targets >= normalized_errors.shape[1])).any()):
        raise ValueError("target index is outside the reconstruction classes")

    true_r = normalized_errors.gather(1, targets[:, None]).squeeze(1)
    wrong_mask = F.one_hot(
        targets,
        num_classes=normalized_errors.shape[1],
    ).bool()
    nearest_wrong_r = normalized_errors.masked_fill(wrong_mask, float("inf")).min(
        dim=1
    ).values
    reconstruction_margin = nearest_wrong_r - true_r
    absolute = true_r.mean()
    separation = F.relu(float(separation_margin) - reconstruction_margin).mean()
    return absolute, separation, true_r, nearest_wrong_r, reconstruction_margin


def compose_decoupled_cssr_loss(
    variant: str,
    *,
    relative_loss: torch.Tensor,
    absolute_loss: torch.Tensor,
    separation_loss: torch.Tensor,
) -> torch.Tensor:
    """Compose only the preregistered D1 or D2 objective."""

    if variant not in DECOUPLED_METHODS:
        raise ValueError(f"unknown decoupled CSSR variant: {variant}")
    weights = LOSS_WEIGHTS[variant]
    return (
        weights["relative"] * relative_loss
        + weights["absolute"] * absolute_loss
        + weights["separation"] * separation_loss
    )


class FGMVCSSRDecoupled1D(nn.Module):
    """Trainable CSSR-only path operating on frozen R2 feature maps.

    R2 is intentionally not registered inside this module.  The runner must
    obtain ``Z`` from a strict-loaded, frozen, eval-mode R2 and may then cache
    those feature maps.  This structural separation prevents CSSR gradients or
    mode changes from reaching the CE classification path.
    """

    architecture_id = "fg_mv_cssr_decoupled_1d_v1"

    def __init__(
        self,
        *,
        num_classes: int = 5,
        input_channels: int = 128,
        latent_channels: int = 32,
        residual_scale: float = 0.1,
        gamma: float = 0.1,
        clip_length: float = 100.0,
        epsilon: float = 1.0e-8,
        margin: float = 0.2,
    ) -> None:
        super().__init__()
        observed = {
            "num_classes": int(num_classes),
            "input_channels": int(input_channels),
            "latent_channels": int(latent_channels),
            "residual_scale": float(residual_scale),
            "gamma": float(gamma),
            "clip_length": float(clip_length),
            "epsilon": float(epsilon),
            "margin": float(margin),
        }
        expected = {
            "num_classes": 5,
            "input_channels": 128,
            "latent_channels": 32,
            "residual_scale": 0.1,
            "gamma": 0.1,
            "clip_length": 100.0,
            "epsilon": 1.0e-8,
            "margin": 0.2,
        }
        if observed != expected:
            raise ValueError(
                f"decoupled CSSR architecture changed: expected {expected}, observed {observed}"
            )
        self.num_classes = int(num_classes)
        self.input_channels = int(input_channels)
        self.latent_channels = int(latent_channels)
        self.gamma = float(gamma)
        self.clip_length = float(clip_length)
        self.epsilon = float(epsilon)
        self.margin = float(margin)
        self.adapter = SharedCSSRSemanticAdapter1D(
            input_channels=self.input_channels,
            hidden_channels=64,
            residual_scale=float(residual_scale),
        )
        self.class_autoencoders = nn.ModuleList(
            [
                DecoupledClassSpecificAutoEncoder1D(
                    input_channels=self.input_channels,
                    latent_channels=self.latent_channels,
                )
                for _ in range(self.num_classes)
            ]
        )

    @staticmethod
    def component_weights(method: str) -> Mapping[str, float]:
        if method not in DECOUPLED_METHODS:
            raise ValueError(f"unknown decoupled CSSR method: {method}")
        return dict(LOSS_WEIGHTS[method])

    def set_adapter_trainable(self, trainable: bool) -> None:
        self.adapter.requires_grad_(bool(trainable))

    def configure_for_epoch(self, epoch: int) -> None:
        """Apply the frozen five-epoch adapter warm start."""

        if not 1 <= int(epoch) <= 20:
            raise ValueError("decoupled CSSR epoch must be in [1,20]")
        self.set_adapter_trainable(int(epoch) >= 6)

    def parameter_groups(self) -> dict[str, tuple[nn.Parameter, ...]]:
        """Return stable optimizer groups, including the initially frozen adapter."""

        groups = {
            "adapter": tuple(self.adapter.parameters()),
            "autoencoders": tuple(self.class_autoencoders.parameters()),
        }
        flattened = [parameter for values in groups.values() for parameter in values]
        if len(flattened) != len({id(parameter) for parameter in flattened}):
            raise RuntimeError("decoupled CSSR parameter groups overlap")
        if set(map(id, flattened)) != {id(parameter) for parameter in self.parameters()}:
            raise RuntimeError("decoupled CSSR parameter groups do not cover the model")
        return groups

    def forward(self, feature_maps: torch.Tensor) -> DecoupledCSSROutput:
        if feature_maps.ndim != 3 or feature_maps.shape[1] != self.input_channels:
            raise ValueError("frozen R2 feature maps must be [batch,128,length]")
        adapted = self.adapter(feature_maps)
        reconstructions = []
        reconstruction_errors = []
        for autoencoder in self.class_autoencoders:
            reconstruction, _ = autoencoder(adapted)
            reconstructions.append(reconstruction)
            reconstruction_errors.append(
                torch.norm(reconstruction - adapted, p=1, dim=1)
            )
        stacked_reconstructions = torch.stack(reconstructions, dim=1)
        stacked_errors = torch.stack(reconstruction_errors, dim=1)
        logits = torch.clamp(
            -self.gamma * stacked_errors,
            min=-self.clip_length,
            max=self.clip_length,
        )
        return DecoupledCSSROutput(
            adapted_features=adapted,
            reconstructions=stacked_reconstructions,
            reconstruction_errors=stacked_errors,
            logits=logits,
            probabilities=softmax_average(logits),
            normalized_reconstruction_errors=normalized_absolute_reconstruction_error(
                adapted,
                stacked_reconstructions,
                epsilon=self.epsilon,
            ),
        )

    def forward_views(self, feature_maps: torch.Tensor) -> DecoupledCSSRViewOutput:
        """Evaluate exactly two views through the same CSSR parameter objects."""

        if feature_maps.ndim != 4 or feature_maps.shape[1:3] != (2, 128):
            raise ValueError("two-view frozen R2 maps must be [batch,2,128,length]")
        batch, views, channels, length = feature_maps.shape
        flat = self(feature_maps.reshape(batch * views, channels, length))
        return DecoupledCSSRViewOutput(
            adapted_features=flat.adapted_features.reshape(batch, views, channels, length),
            reconstructions=flat.reconstructions.reshape(
                batch,
                views,
                self.num_classes,
                channels,
                length,
            ),
            reconstruction_errors=flat.reconstruction_errors.reshape(
                batch,
                views,
                self.num_classes,
                length,
            ),
            logits=flat.logits.reshape(
                batch,
                views,
                self.num_classes,
                length,
            ),
            probabilities=flat.probabilities.reshape(
                batch,
                views,
                self.num_classes,
            ),
            normalized_reconstruction_errors=flat.normalized_reconstruction_errors.reshape(
                batch,
                views,
                self.num_classes,
            ),
        )

    def loss_from_output(
        self,
        output: DecoupledCSSROutput,
        targets: torch.Tensor,
        method: str,
    ) -> DecoupledCSSRLoss:
        relative = pcssr_nll_loss(output.probabilities, targets)
        absolute, separation, true_r, wrong_r, margin = (
            absolute_and_separation_losses(
                output.normalized_reconstruction_errors,
                targets,
                separation_margin=self.margin,
            )
        )
        total = compose_decoupled_cssr_loss(
            method,
            relative_loss=relative,
            absolute_loss=absolute,
            separation_loss=separation,
        )
        return DecoupledCSSRLoss(
            total_loss=total,
            relative_loss=relative,
            absolute_loss=absolute,
            separation_loss=separation,
            true_class_r=true_r,
            nearest_wrong_class_r=wrong_r,
            reconstruction_margin=margin,
            component_weights=self.component_weights(method),
            output=output,
        )

    def loss(
        self,
        feature_maps: torch.Tensor,
        targets: torch.Tensor,
        method: str,
    ) -> DecoupledCSSRLoss:
        """Run one single-view forward pass and compute the selected frozen loss."""

        return self.loss_from_output(self(feature_maps), targets, method)
