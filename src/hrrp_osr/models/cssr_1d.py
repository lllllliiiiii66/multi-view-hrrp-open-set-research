from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class PCSSRCoreOutput:
    """Inspectable outputs of the class-specific reconstruction core."""

    reconstructions: torch.Tensor
    reconstruction_errors: torch.Tensor
    logits: torch.Tensor
    probabilities: torch.Tensor


class ClassSpecificAutoEncoder1D(nn.Module):
    """The no-hidden-layer pCSSR autoencoder adapted from 1x1 Conv2d to Conv1d."""

    def __init__(self, input_channels: int = 128, latent_channels: int = 64) -> None:
        super().__init__()
        if input_channels <= 0 or latent_channels <= 0:
            raise ValueError("autoencoder channel counts must be positive")
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, latent_channels, kernel_size=1, bias=False),
            nn.Tanh(),
        )
        self.decoder = nn.Conv1d(
            latent_channels,
            input_channels,
            kernel_size=1,
            bias=False,
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(inputs)
        return self.decoder(latent), latent


def softmax_average(logits: torch.Tensor) -> torch.Tensor:
    """Official ``softmax_avg``: class softmax per position, then spatial mean."""

    if logits.ndim != 3:
        raise ValueError("pCSSR logits must have shape [batch, classes, length]")
    return torch.softmax(logits, dim=1).mean(dim=-1)


def pcssr_nll_loss(probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Official pCSSR loss applied after ``softmax_avg`` aggregation."""

    if probabilities.ndim != 2:
        raise ValueError("pCSSR probabilities must have shape [batch, classes]")
    if targets.ndim != 1 or targets.shape[0] != probabilities.shape[0]:
        raise ValueError("targets must have shape [batch]")
    if targets.dtype != torch.long:
        targets = targets.long()
    one_hot = F.one_hot(targets, num_classes=probabilities.shape[1]).to(
        dtype=probabilities.dtype,
        device=probabilities.device,
    )
    return -(one_hot * torch.log(probabilities)).sum(dim=1).mean()


def scale_normalized_reconstruction_inconsistency(
    logits: torch.Tensor,
    features: torch.Tensor,
    *,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Convert official pCSSR knownness to a larger-means-unknown score.

    The official score divides each position's selected-class reconstruction
    logit by ``mean_c(abs(feature))`` twice and only then averages positions.
    This project computes that quantity for every class, negates its direction,
    and adds the explicitly preregistered denominator floor.
    """

    if logits.ndim != 3:
        raise ValueError("pCSSR logits must have shape [batch, classes, length]")
    if features.ndim != 3:
        raise ValueError("features must have shape [batch, channels, length]")
    if logits.shape[0] != features.shape[0] or logits.shape[2] != features.shape[2]:
        raise ValueError("logits and features must share batch and length dimensions")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    activation_scale = features.abs().mean(dim=1).clamp_min(float(epsilon))
    return (-logits / activation_scale.square().unsqueeze(1)).mean(dim=-1)


class PCSSRCore1D(nn.Module):
    """Only the official pCSSR class-specific reconstruction core in one dimension.

    This deliberately excludes rCSSR, prototypes, Gram features, score
    integration, augmentation, and any learned score fusion.
    """

    architecture_id = "PCSSR_CORE_1D"

    def __init__(
        self,
        num_classes: int,
        *,
        input_channels: int = 128,
        latent_channels: int = 64,
        gamma: float = 0.1,
        clip_length: float = 100.0,
        epsilon: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if num_classes <= 1:
            raise ValueError("pCSSR requires at least two known classes")
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        if clip_length <= 0:
            raise ValueError("clip length must be positive")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")

        self.num_classes = int(num_classes)
        self.input_channels = int(input_channels)
        self.latent_channels = int(latent_channels)
        self.gamma = float(gamma)
        self.clip_length = float(clip_length)
        self.epsilon = float(epsilon)
        self.class_autoencoders = nn.ModuleList(
            [
                ClassSpecificAutoEncoder1D(
                    input_channels=self.input_channels,
                    latent_channels=self.latent_channels,
                )
                for _ in range(self.num_classes)
            ]
        )

    def _validate_inputs(self, inputs: torch.Tensor) -> None:
        if inputs.ndim != 3:
            raise ValueError("pCSSR features must have shape [batch, channels, length]")
        if inputs.shape[1] != self.input_channels:
            raise ValueError(
                f"pCSSR expected {self.input_channels} feature channels, "
                f"received {inputs.shape[1]}"
            )

    def forward(self, inputs: torch.Tensor) -> PCSSRCoreOutput:
        self._validate_inputs(inputs)
        reconstructions = []
        reconstruction_errors = []
        for autoencoder in self.class_autoencoders:
            reconstruction, _ = autoencoder(inputs)
            reconstructions.append(reconstruction)
            reconstruction_errors.append(
                torch.norm(reconstruction - inputs, p=1, dim=1)
            )

        stacked_reconstructions = torch.stack(reconstructions, dim=1)
        stacked_errors = torch.stack(reconstruction_errors, dim=1)
        logits = torch.clamp(
            -self.gamma * stacked_errors,
            min=-self.clip_length,
            max=self.clip_length,
        )
        probabilities = softmax_average(logits)
        return PCSSRCoreOutput(
            reconstructions=stacked_reconstructions,
            reconstruction_errors=stacked_errors,
            logits=logits,
            probabilities=probabilities,
        )

    def loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, PCSSRCoreOutput]:
        output = self(inputs)
        return pcssr_nll_loss(output.probabilities, targets), output

    def reconstruction_inconsistency(
        self,
        inputs: torch.Tensor,
        *,
        output: PCSSRCoreOutput | None = None,
    ) -> torch.Tensor:
        self._validate_inputs(inputs)
        if output is None:
            output = self(inputs)
        return scale_normalized_reconstruction_inconsistency(
            output.logits,
            inputs,
            epsilon=self.epsilon,
        )


# The exact experiment-facing identifier requested by the preregistration.
PCSSR_CORE_1D = PCSSRCore1D
