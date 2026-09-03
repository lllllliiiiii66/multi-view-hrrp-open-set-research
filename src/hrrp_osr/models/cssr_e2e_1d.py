from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .cssr_1d import PCSSRCore1D, PCSSRCoreOutput, pcssr_nll_loss, softmax_average
from .hrrp_ms_resnet import HRRPMultiScaleResNet1D
from .ms_mean_factorial import MSMeanHeadFactorialModel


Q1_CE_FINETUNE_CONTROL = "Q1_CE_FINETUNE_CONTROL"
Q2_E2E_REL_CSSR_1X1 = "Q2_E2E_REL_CSSR_1X1"
Q3_E2E_ABSREL_CSSR_1X1 = "Q3_E2E_ABSREL_CSSR_1X1"
Q4_E2E_ABSREL_CSSR_LOCAL3 = "Q4_E2E_ABSREL_CSSR_LOCAL3"

TRAINABLE_VARIANTS = (
    Q1_CE_FINETUNE_CONTROL,
    Q2_E2E_REL_CSSR_1X1,
    Q3_E2E_ABSREL_CSSR_1X1,
    Q4_E2E_ABSREL_CSSR_LOCAL3,
)
CSSR_VARIANTS = frozenset(TRAINABLE_VARIANTS[1:])
ABSOLUTE_VARIANTS = frozenset(TRAINABLE_VARIANTS[2:])

LOSS_WEIGHTS: Mapping[str, Mapping[str, float]] = {
    Q1_CE_FINETUNE_CONTROL: {
        "classification": 1.0,
        "relative": 0.0,
        "absolute": 0.0,
        "separation": 0.0,
    },
    Q2_E2E_REL_CSSR_1X1: {
        "classification": 1.0,
        "relative": 0.5,
        "absolute": 0.0,
        "separation": 0.0,
    },
    Q3_E2E_ABSREL_CSSR_1X1: {
        "classification": 1.0,
        "relative": 0.5,
        "absolute": 0.25,
        "separation": 0.5,
    },
    Q4_E2E_ABSREL_CSSR_LOCAL3: {
        "classification": 1.0,
        "relative": 0.5,
        "absolute": 0.25,
        "separation": 0.5,
    },
}


@dataclass(frozen=True)
class FGMVCSSRE2EOutput:
    """Outputs shared by the CE control and the three CSSR variants."""

    feature_maps: torch.Tensor
    per_view_features: torch.Tensor
    fused_features: torch.Tensor
    fused_logits: torch.Tensor
    cssr_outputs: tuple[PCSSRCoreOutput, PCSSRCoreOutput] | None
    normalized_reconstruction_errors: torch.Tensor | None


@dataclass(frozen=True)
class FGMVCSSRE2ELoss:
    """All objective terms plus per-example reconstruction diagnostics."""

    total_loss: torch.Tensor
    classification_loss: torch.Tensor
    relative_loss: torch.Tensor
    absolute_loss: torch.Tensor
    separation_loss: torch.Tensor
    true_class_r: torch.Tensor | None
    nearest_wrong_class_r: torch.Tensor | None
    reconstruction_margin: torch.Tensor | None
    component_weights: Mapping[str, float]


class LocalClassSpecificAutoEncoder1D(nn.Module):
    """The frozen local-structure alternative used only by Q4."""

    def __init__(self, input_channels: int = 128, latent_channels: int = 64) -> None:
        super().__init__()
        if input_channels <= 0 or latent_channels <= 0:
            raise ValueError("autoencoder channel counts must be positive")
        self.encoder = nn.Sequential(
            nn.Conv1d(
                input_channels,
                latent_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.Tanh(),
        )
        self.decoder = nn.Conv1d(
            latent_channels,
            input_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(inputs)
        return self.decoder(latent), latent


class LocalPCSSRCore1D(nn.Module):
    """pCSSR core semantics with only its class AE kernel changed to local k=3."""

    architecture_id = "PCSSR_CORE_LOCAL3_1D"

    def __init__(
        self,
        num_classes: int,
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
        if gamma <= 0.0 or clip_length <= 0.0:
            raise ValueError("pCSSR gamma and clip length must be positive")
        self.num_classes = int(num_classes)
        self.input_channels = int(input_channels)
        self.latent_channels = int(latent_channels)
        self.gamma = float(gamma)
        self.clip_length = float(clip_length)
        self.class_autoencoders = nn.ModuleList(
            [
                LocalClassSpecificAutoEncoder1D(
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
        return PCSSRCoreOutput(
            reconstructions=stacked_reconstructions,
            reconstruction_errors=stacked_errors,
            logits=logits,
            probabilities=softmax_average(logits),
        )

    def loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, PCSSRCoreOutput]:
        output = self(inputs)
        return pcssr_nll_loss(output.probabilities, targets), output


def absolute_normalized_reconstruction_error(
    features: torch.Tensor,
    reconstructions: torch.Tensor,
    *,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Return the preregistered absolute class reconstruction quantity r."""

    if features.ndim != 3:
        raise ValueError("features must have shape [batch, channels, length]")
    if reconstructions.ndim != 4:
        raise ValueError(
            "reconstructions must have shape [batch, classes, channels, length]"
        )
    if reconstructions.shape[0] != features.shape[0] or reconstructions.shape[2:] != features.shape[1:]:
        raise ValueError("feature and reconstruction dimensions differ")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    numerator = (reconstructions - features.unsqueeze(1)).abs().mean(dim=(2, 3))
    denominator = features.abs().mean(dim=(1, 2)).unsqueeze(1) + float(epsilon)
    return numerator / denominator


def fusion_guided_class_score(
    view_class_scores: torch.Tensor,
    fused_logits: torch.Tensor,
) -> torch.Tensor:
    """Average both views' score for the class predicted by fused CE logits."""

    if view_class_scores.ndim != 3 or view_class_scores.shape[1] != 2:
        raise ValueError("view class scores must have shape [batch, 2, classes]")
    if fused_logits.ndim != 2:
        raise ValueError("fused logits must have shape [batch, classes]")
    if (
        view_class_scores.shape[0] != fused_logits.shape[0]
        or view_class_scores.shape[2] != fused_logits.shape[1]
    ):
        raise ValueError("view scores and fused logits do not align")
    predicted = fused_logits.argmax(dim=1)
    selected = view_class_scores.gather(
        2,
        predicted[:, None, None].expand(-1, 2, 1),
    ).squeeze(2)
    return selected.mean(dim=1)


class FGMVCSSRE2EModel(nn.Module):
    """Finite-scope R2 fine-tuning wrapper used by preregistered Q1--Q4."""

    architecture_id = "fg_mv_cssr_e2e_redesign_v2"

    def __init__(
        self,
        r2_model: MSMeanHeadFactorialModel,
        variant: str,
        *,
        autoencoder_seed: int = 20260904,
        input_channels: int = 128,
        latent_channels: int = 64,
        gamma: float = 0.1,
        clip_length: float = 100.0,
        reconstruction_epsilon: float = 1.0e-8,
        separation_margin: float = 0.2,
    ) -> None:
        super().__init__()
        if variant not in TRAINABLE_VARIANTS:
            raise ValueError(f"unknown FG-MV-CSSR variant: {variant}")
        if not isinstance(r2_model, MSMeanHeadFactorialModel):
            raise TypeError("r2_model must be an MSMeanHeadFactorialModel")
        if r2_model.method != "R2_MS_MEAN_CE":
            raise ValueError("FG-MV-CSSR must start from R2_MS_MEAN_CE")
        if not isinstance(r2_model.encoder, HRRPMultiScaleResNet1D):
            raise ValueError("FG-MV-CSSR requires the frozen multi-scale R2 encoder")
        if not isinstance(r2_model.global_head, nn.Linear):
            raise ValueError("FG-MV-CSSR requires the linear R2 CE head")
        if any(r2_model.forbidden_component_status.values()):
            raise ValueError("R2 contains a component forbidden by this experiment")
        if input_channels != 128 or latent_channels != 64:
            raise ValueError("the preregistered CSSR channel dimensions are 128/64")
        if gamma != 0.1 or clip_length != 100.0:
            raise ValueError("the preregistered pCSSR gamma/clip are 0.1/100")
        if reconstruction_epsilon != 1.0e-8 or separation_margin != 0.2:
            raise ValueError("the preregistered r epsilon/margin are 1e-8/0.2")

        self.variant = variant
        self.known_class_count = int(r2_model.known_class_count)
        self.reconstruction_epsilon = float(reconstruction_epsilon)
        self.separation_margin = float(separation_margin)
        self.r2_model = r2_model
        self.cssr_core: PCSSRCore1D | LocalPCSSRCore1D | None = None
        if variant in CSSR_VARIANTS:
            # Preserve the caller's RNG state while guaranteeing paired Q2/Q3 AE starts.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(autoencoder_seed))
                if variant == Q4_E2E_ABSREL_CSSR_LOCAL3:
                    self.cssr_core = LocalPCSSRCore1D(
                        self.known_class_count,
                        input_channels=input_channels,
                        latent_channels=latent_channels,
                        gamma=gamma,
                        clip_length=clip_length,
                    )
                else:
                    self.cssr_core = PCSSRCore1D(
                        self.known_class_count,
                        input_channels=input_channels,
                        latent_channels=latent_channels,
                        gamma=gamma,
                        clip_length=clip_length,
                        epsilon=reconstruction_epsilon,
                    )
            reference_parameter = next(self.r2_model.parameters())
            self.cssr_core.to(
                device=reference_parameter.device,
                dtype=reference_parameter.dtype,
            )
        self.configure_trainable_scope()

    @classmethod
    def from_r2_state_dict(
        cls,
        r2_state_dict: Mapping[str, torch.Tensor],
        variant: str,
        *,
        known_class_count: int,
        feature_dim: int = 128,
        dropout: float = 0.1,
        ce_weight_init_std: float = 0.01,
        autoencoder_seed: int = 20260904,
    ) -> FGMVCSSRE2EModel:
        """Strict-load an unchanged R2 state before registering any new AE keys."""

        r2_model = MSMeanHeadFactorialModel(
            "R2_MS_MEAN_CE",
            known_class_count,
            feature_dim=feature_dim,
            dropout=dropout,
            ce_weight_init_std=ce_weight_init_std,
        )
        incompatible = r2_model.load_state_dict(r2_state_dict, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError("R2 state dict was not strict-load compatible")
        return cls(
            r2_model,
            variant,
            autoencoder_seed=autoencoder_seed,
        )

    @property
    def encoder(self) -> HRRPMultiScaleResNet1D:
        return self.r2_model.encoder

    @property
    def global_head(self) -> nn.Linear:
        return self.r2_model.global_head

    @property
    def component_weights(self) -> Mapping[str, float]:
        return LOSS_WEIGHTS[self.variant]

    def configure_trainable_scope(self) -> None:
        """Freeze early R2 state, including BN buffers through enforced eval mode."""

        self.r2_model.requires_grad_(False)
        self.encoder.stages[2].requires_grad_(True)
        self.encoder.projection.requires_grad_(True)
        self.global_head.requires_grad_(True)
        if self.cssr_core is not None:
            self.cssr_core.requires_grad_(True)
        self._enforce_frozen_module_eval()

    def _enforce_frozen_module_eval(self) -> None:
        self.encoder.stem.eval()
        self.encoder.stages[0].eval()
        self.encoder.stages[1].eval()

    def train(self, mode: bool = True) -> FGMVCSSRE2EModel:
        super().train(mode)
        self._enforce_frozen_module_eval()
        return self

    def trainable_parameter_groups(self) -> dict[str, tuple[nn.Parameter, ...]]:
        """Return disjoint parameter groups matching the three frozen LR groups."""

        result = {
            "last_stage": tuple(self.encoder.stages[2].parameters()),
            "projection_and_ce_head": tuple(self.encoder.projection.parameters())
            + tuple(self.global_head.parameters()),
            "autoencoders": ()
            if self.cssr_core is None
            else tuple(self.cssr_core.parameters()),
        }
        flattened = [parameter for values in result.values() for parameter in values]
        if len(flattened) != len({id(parameter) for parameter in flattened}):
            raise RuntimeError("trainable parameter groups overlap")
        if set(map(id, flattened)) != {
            id(parameter) for parameter in self.parameters() if parameter.requires_grad
        }:
            raise RuntimeError("trainable parameter groups do not cover the model")
        return result

    def forward(self, inputs: torch.Tensor) -> FGMVCSSRE2EOutput:
        if inputs.ndim != 3 or inputs.shape[1:] != (2, 601):
            raise ValueError("two-view HRRP input must have shape [batch, 2, 601]")
        batch = inputs.shape[0]
        feature_maps_flat = self.encoder.forward_feature_map(inputs.reshape(batch * 2, 601))
        pooled = torch.cat(
            [
                self.encoder.average_pool(feature_maps_flat).flatten(1),
                self.encoder.maximum_pool(feature_maps_flat).flatten(1),
            ],
            dim=1,
        )
        per_view_features = self.encoder.projection(pooled).reshape(batch, 2, -1)
        fused_features = per_view_features.mean(dim=1)
        fused_logits = self.global_head(fused_features)
        feature_maps = feature_maps_flat.reshape(
            batch,
            2,
            feature_maps_flat.shape[1],
            feature_maps_flat.shape[2],
        )

        cssr_outputs: tuple[PCSSRCoreOutput, PCSSRCoreOutput] | None = None
        normalized_errors: torch.Tensor | None = None
        if self.cssr_core is not None:
            view_outputs = (
                self.cssr_core(feature_maps[:, 0]),
                self.cssr_core(feature_maps[:, 1]),
            )
            cssr_outputs = view_outputs
            normalized_errors = torch.stack(
                [
                    absolute_normalized_reconstruction_error(
                        feature_maps[:, view_index],
                        view_outputs[view_index].reconstructions,
                        epsilon=self.reconstruction_epsilon,
                    )
                    for view_index in range(2)
                ],
                dim=1,
            )
        return FGMVCSSRE2EOutput(
            feature_maps=feature_maps,
            per_view_features=per_view_features,
            fused_features=fused_features,
            fused_logits=fused_logits,
            cssr_outputs=cssr_outputs,
            normalized_reconstruction_errors=normalized_errors,
        )

    def loss(
        self,
        output: FGMVCSSRE2EOutput,
        targets: torch.Tensor,
    ) -> FGMVCSSRE2ELoss:
        if targets.ndim != 1 or targets.shape[0] != output.fused_logits.shape[0]:
            raise ValueError("targets must have shape [batch]")
        if targets.dtype != torch.long:
            targets = targets.long()
        classification = F.cross_entropy(output.fused_logits, targets)
        zero = classification.new_zeros(())
        relative = zero
        absolute = zero
        separation = zero
        true_r: torch.Tensor | None = None
        wrong_r: torch.Tensor | None = None
        margin: torch.Tensor | None = None

        if self.cssr_core is not None:
            if output.cssr_outputs is None or output.normalized_reconstruction_errors is None:
                raise ValueError("CSSR output is incomplete")
            relative = torch.stack(
                [
                    pcssr_nll_loss(view_output.probabilities, targets)
                    for view_output in output.cssr_outputs
                ]
            ).mean()
            values = output.normalized_reconstruction_errors
            target_indices = targets[:, None, None].expand(-1, 2, 1)
            true_r = values.gather(2, target_indices).squeeze(2)
            wrong_mask = F.one_hot(
                targets,
                num_classes=self.known_class_count,
            ).bool()[:, None, :]
            wrong_r = values.masked_fill(wrong_mask, float("inf")).min(dim=2).values
            margin = wrong_r - true_r
            absolute = true_r.mean()
            separation = F.relu(self.separation_margin - margin).mean()

        weights = self.component_weights
        total = (
            weights["classification"] * classification
            + weights["relative"] * relative
            + weights["absolute"] * absolute
            + weights["separation"] * separation
        )
        return FGMVCSSRE2ELoss(
            total_loss=total,
            classification_loss=classification,
            relative_loss=relative,
            absolute_loss=absolute,
            separation_loss=separation,
            true_class_r=true_r,
            nearest_wrong_class_r=wrong_r,
            reconstruction_margin=margin,
            component_weights=dict(weights),
        )
