from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .hrrp_ms_resnet import HRRPMultiScaleResNet1D


OFFICIAL_SEMANTICS_PCSSR_1D = "OFFICIAL_SEMANTICS_PCSSR_1D"
MATCHED_LINEAR_CONTROL_1D = "MATCHED_LINEAR_CONTROL_1D"
OFFICIAL_CSSR_REFERENCE_COMMIT = "d5a99e91f310ec274c7bfe5796fb270719a07ab3"


@dataclass(frozen=True)
class OfficialPCSSRHeadOutput:
    """Inspectable output of the commit-faithful pCSSR classification head."""

    reconstructions: torch.Tensor
    latents: torch.Tensor
    reconstruction_errors: torch.Tensor
    logits: torch.Tensor
    probabilities: torch.Tensor


@dataclass(frozen=True)
class MatchedLinearHeadOutput:
    """Output of the matched linear control used by this HRRP experiment.

    This control retains the official ``LinearClassifier`` 1x1-convolution
    primitive, but intentionally uses pCSSR's ``gamma=0.1`` and
    ``softmax_avg`` aggregation.  It is therefore not a reproduction of the
    official repository's ``linear.json`` configuration.
    """

    logits: torch.Tensor
    probabilities: torch.Tensor


OfficialHeadOutput = OfficialPCSSRHeadOutput | MatchedLinearHeadOutput


@dataclass(frozen=True)
class OfficialCSSRHRRPModelOutput:
    """Single-view HRRP feature map and its official-semantics head output."""

    feature_maps: torch.Tensor
    head_output: OfficialHeadOutput


class OfficialClassSpecificAutoEncoder1D(nn.Module):
    """Official no-hidden pCSSR autoencoder adapted from Conv2d to Conv1d."""

    def __init__(
        self,
        input_channels: int = 128,
        latent_channels: int = 64,
    ) -> None:
        super().__init__()
        if input_channels <= 0 or latent_channels <= 0:
            raise ValueError("autoencoder channel counts must be positive")
        self.input_channels = int(input_channels)
        self.latent_channels = int(latent_channels)
        self.encoder = nn.Sequential(
            nn.Conv1d(
                self.input_channels,
                self.latent_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.Tanh(),
        )
        self.decoder = nn.Conv1d(
            self.latent_channels,
            self.input_channels,
            kernel_size=1,
            bias=False,
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3 or inputs.shape[1] != self.input_channels:
            raise ValueError(
                "official pCSSR autoencoder inputs must have shape "
                f"[batch,{self.input_channels},length]"
            )
        latent = self.encoder(inputs)
        return self.decoder(latent), latent


def official_softmax_average(logits: torch.Tensor) -> torch.Tensor:
    """Apply class SoftMax per position, then average spatial probabilities."""

    if logits.ndim != 3:
        raise ValueError("official spatial logits must have shape [batch,classes,length]")
    return torch.softmax(logits, dim=1).mean(dim=-1)


def official_pcssr_loss(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Official negative log-probability loss after ``softmax_avg``."""

    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [batch,classes]")
    if targets.ndim != 1 or targets.shape[0] != probabilities.shape[0]:
        raise ValueError("targets must have shape [batch]")
    if probabilities.shape[1] <= 1:
        raise ValueError("classification requires at least two classes")
    targets = targets.to(device=probabilities.device, dtype=torch.long)
    if bool(((targets < 0) | (targets >= probabilities.shape[1])).any()):
        raise ValueError("target index is outside the known classes")
    one_hot = F.one_hot(targets, num_classes=probabilities.shape[1]).to(
        dtype=probabilities.dtype,
        device=probabilities.device,
    )
    return -(one_hot * torch.log(probabilities)).sum(dim=1).mean()


class OfficialPCSSRHead1D(nn.Module):
    """Official pCSSR head semantics for a one-dimensional feature map.

    The corresponding two-dimensional source is
    ``xyzedd/CSSR@d5a99e91:methods/cssr.py``.  Replacing a 1x1 Conv2d on
    ``[B,C,1,L]`` with Conv1d on ``[B,C,L]`` is algebraically exact.
    """

    architecture_id = OFFICIAL_SEMANTICS_PCSSR_1D

    def __init__(
        self,
        num_classes: int = 5,
        *,
        input_channels: int = 128,
        latent_channels: int = 64,
        gamma: float = 0.1,
        clip_length: float = 100.0,
    ) -> None:
        super().__init__()
        if num_classes <= 1:
            raise ValueError("pCSSR requires at least two known classes")
        if input_channels <= 0 or latent_channels <= 0:
            raise ValueError("pCSSR channel counts must be positive")
        if gamma <= 0.0:
            raise ValueError("pCSSR gamma must be positive")
        if clip_length <= 0.0:
            raise ValueError("pCSSR clip length must be positive")

        self.num_classes = int(num_classes)
        self.input_channels = int(input_channels)
        self.latent_channels = int(latent_channels)
        self.gamma = float(gamma)
        self.clip_length = float(clip_length)
        self.class_autoencoders = nn.ModuleList(
            [
                OfficialClassSpecificAutoEncoder1D(
                    input_channels=self.input_channels,
                    latent_channels=self.latent_channels,
                )
                for _ in range(self.num_classes)
            ]
        )

    def _validate_features(self, features: torch.Tensor) -> None:
        if features.ndim != 3 or features.shape[1] != self.input_channels:
            raise ValueError(
                "official pCSSR features must have shape "
                f"[batch,{self.input_channels},length]"
            )

    def forward(self, features: torch.Tensor) -> OfficialPCSSRHeadOutput:
        self._validate_features(features)
        reconstructions: list[torch.Tensor] = []
        latents: list[torch.Tensor] = []
        reconstruction_logits: list[torch.Tensor] = []
        reconstruction_errors: list[torch.Tensor] = []

        for autoencoder in self.class_autoencoders:
            reconstruction, latent = autoencoder(features)
            error = torch.norm(reconstruction - features, p=1, dim=1)
            logit = torch.clamp(
                -self.gamma * error,
                min=-self.clip_length,
                max=self.clip_length,
            )
            reconstructions.append(reconstruction)
            latents.append(latent)
            reconstruction_errors.append(error)
            reconstruction_logits.append(logit)

        reconstruction_tensor = torch.stack(reconstructions, dim=1)
        latent_tensor = torch.stack(latents, dim=1)
        error_tensor = torch.stack(reconstruction_errors, dim=1)
        logits = torch.stack(reconstruction_logits, dim=1)
        probabilities = official_softmax_average(logits)
        return OfficialPCSSRHeadOutput(
            reconstructions=reconstruction_tensor,
            latents=latent_tensor,
            reconstruction_errors=error_tensor,
            logits=logits,
            probabilities=probabilities,
        )

    def loss(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, OfficialPCSSRHeadOutput]:
        output = self(features)
        return official_pcssr_loss(output.probabilities, targets), output


class MatchedLinearHead1D(nn.Module):
    """Matched linear control, not the official ``linear.json`` benchmark."""

    architecture_id = MATCHED_LINEAR_CONTROL_1D

    def __init__(
        self,
        num_classes: int = 5,
        *,
        input_channels: int = 128,
        gamma: float = 0.1,
    ) -> None:
        super().__init__()
        if num_classes <= 1:
            raise ValueError("linear classification requires at least two classes")
        if input_channels <= 0:
            raise ValueError("linear input channels must be positive")
        if gamma <= 0.0:
            raise ValueError("linear gamma must be positive")
        self.num_classes = int(num_classes)
        self.input_channels = int(input_channels)
        self.gamma = float(gamma)
        self.classifier = nn.Conv1d(
            self.input_channels,
            self.num_classes,
            kernel_size=1,
            bias=False,
        )

    def forward(self, features: torch.Tensor) -> MatchedLinearHeadOutput:
        if features.ndim != 3 or features.shape[1] != self.input_channels:
            raise ValueError(
                "matched linear features must have shape "
                f"[batch,{self.input_channels},length]"
            )
        logits = self.gamma * self.classifier(features)
        return MatchedLinearHeadOutput(
            logits=logits,
            probabilities=official_softmax_average(logits),
        )

    def loss(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, MatchedLinearHeadOutput]:
        output = self(features)
        return official_pcssr_loss(output.probabilities, targets), output


class HRRPFeatureMapEncoder1D(nn.Module):
    """Only the pretrained HRRP stem and residual stages.

    The global average/max pooling path, 128-dimensional projection, and the
    historical CE head are intentionally absent from this module.  Source
    modules are deep-copied so independently trained O1--O4 models cannot
    share parameters or BatchNorm buffers by accident.
    """

    architecture_id = "hrrp_ms_resnet_feature_map_encoder_1d_v1"

    def __init__(
        self,
        stem: nn.Module,
        stages: nn.Module,
        *,
        input_length: int = 601,
        output_channels: int = 128,
        output_length: int = 76,
    ) -> None:
        super().__init__()
        if input_length <= 0 or output_channels <= 0 or output_length <= 0:
            raise ValueError("feature-map encoder dimensions must be positive")
        self.input_length = int(input_length)
        self.output_channels = int(output_channels)
        self.output_length = int(output_length)
        self.stem = copy.deepcopy(stem)
        self.stages = copy.deepcopy(stages)

    @classmethod
    def from_r2_encoder(
        cls,
        encoder: HRRPMultiScaleResNet1D,
    ) -> HRRPFeatureMapEncoder1D:
        if not isinstance(encoder, HRRPMultiScaleResNet1D):
            raise TypeError("R2 encoder must be HRRPMultiScaleResNet1D")
        return cls(
            encoder.stem,
            encoder.stages,
            input_length=encoder.input_length,
            output_channels=128,
            output_length=76,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(1)
        if inputs.ndim != 3 or inputs.shape[1:] != (1, self.input_length):
            raise ValueError(
                "HRRP feature-map encoder input must have shape "
                f"[batch,{self.input_length}] or [batch,1,{self.input_length}]"
            )
        features = self.stages(self.stem(inputs))
        expected = (self.output_channels, self.output_length)
        if features.shape[1:] != expected:
            raise RuntimeError(
                f"HRRP feature-map shape changed: expected {expected}, "
                f"observed {tuple(features.shape[1:])}"
            )
        return features


class OfficialCSSRHRRPModel1D(nn.Module):
    """Minimal single-view HRRP encoder plus one frozen experiment head type."""

    def __init__(
        self,
        encoder: HRRPFeatureMapEncoder1D,
        *,
        head_kind: Literal[
            "OFFICIAL_SEMANTICS_PCSSR_1D", "MATCHED_LINEAR_CONTROL_1D"
        ],
        num_classes: int = 5,
        latent_channels: int = 64,
        gamma: float = 0.1,
        clip_length: float = 100.0,
    ) -> None:
        super().__init__()
        if not isinstance(encoder, HRRPFeatureMapEncoder1D):
            raise TypeError("encoder must be HRRPFeatureMapEncoder1D")
        if head_kind not in (
            OFFICIAL_SEMANTICS_PCSSR_1D,
            MATCHED_LINEAR_CONTROL_1D,
        ):
            raise ValueError(f"unknown official CSSR head kind: {head_kind}")
        self.encoder = encoder
        self.head_kind = head_kind
        if head_kind == OFFICIAL_SEMANTICS_PCSSR_1D:
            self.head: OfficialPCSSRHead1D | MatchedLinearHead1D = (
                OfficialPCSSRHead1D(
                    num_classes=num_classes,
                    input_channels=encoder.output_channels,
                    latent_channels=latent_channels,
                    gamma=gamma,
                    clip_length=clip_length,
                )
            )
        else:
            self.head = MatchedLinearHead1D(
                num_classes=num_classes,
                input_channels=encoder.output_channels,
                gamma=gamma,
            )

    @classmethod
    def from_r2_encoder(
        cls,
        r2_encoder: HRRPMultiScaleResNet1D,
        *,
        head_kind: Literal[
            "OFFICIAL_SEMANTICS_PCSSR_1D", "MATCHED_LINEAR_CONTROL_1D"
        ],
        num_classes: int = 5,
        latent_channels: int = 64,
        gamma: float = 0.1,
        clip_length: float = 100.0,
    ) -> OfficialCSSRHRRPModel1D:
        return cls(
            HRRPFeatureMapEncoder1D.from_r2_encoder(r2_encoder),
            head_kind=head_kind,
            num_classes=num_classes,
            latent_channels=latent_channels,
            gamma=gamma,
            clip_length=clip_length,
        )

    def forward(self, inputs: torch.Tensor) -> OfficialCSSRHRRPModelOutput:
        feature_maps = self.encoder(inputs)
        return OfficialCSSRHRRPModelOutput(
            feature_maps=feature_maps,
            head_output=self.head(feature_maps),
        )

    def loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, OfficialCSSRHRRPModelOutput]:
        output = self(inputs)
        return official_pcssr_loss(output.head_output.probabilities, targets), output
