from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


OFFICIAL_SCORE_NAMES = ("S1", "S2", "S3")
OFFICIAL_SCORE_RULES = (
    "S1",
    "S2",
    "S3",
    "S1+S2",
    "S1+S3",
    "S2+S3",
    "full",
    "pcssr_max_pair_probability",
)


@dataclass(frozen=True)
class OfficialScoreTemplates:
    """Predicted-class templates built from raw, unique train-known bases."""

    first_order: torch.Tensor
    gram: torch.Tensor
    counts: torch.Tensor
    num_classes: int
    power: int


@dataclass(frozen=True)
class OfficialRawScores:
    """Commit-faithful S1/S2/S3 knownness scores for single views."""

    s1: torch.Tensor
    s2: torch.Tensor
    s3: torch.Tensor

    @property
    def values(self) -> torch.Tensor:
        return torch.stack((self.s1, self.s2, self.s3), dim=-1)


@dataclass(frozen=True)
class OfficialScoreNormalization:
    """Population statistics from four deterministic augmented train passes."""

    mean: torch.Tensor
    std: torch.Tensor
    epsilon: float
    min_std: float


@dataclass(frozen=True)
class OfficialStandardizedScores:
    standardized: torch.Tensor
    integrated: torch.Tensor


@dataclass(frozen=True)
class MatchedLinearPairOutput:
    pair_probabilities: torch.Tensor
    predicted_class: torch.Tensor
    max_spatial_average_logit: torch.Tensor
    max_pair_probability: torch.Tensor
    unknown_score: torch.Tensor


@dataclass(frozen=True)
class OfficialPairScores:
    pair_probabilities: torch.Tensor
    predicted_class: torch.Tensor
    per_view_raw: torch.Tensor
    per_view_standardized: torch.Tensor
    pair_standardized_components: torch.Tensor
    knownness_by_rule: Mapping[str, torch.Tensor]
    unknown_scores_by_rule: Mapping[str, torch.Tensor]

    @property
    def full_knownness(self) -> torch.Tensor:
        return self.knownness_by_rule["full"]

    @property
    def unknown_score(self) -> torch.Tensor:
        return self.unknown_scores_by_rule["full"]


def _validate_feature_map(features: torch.Tensor, *, name: str = "features") -> None:
    if features.ndim != 3:
        raise ValueError(f"{name} must have shape [batch,channels,length]")
    if features.shape[0] <= 0 or features.shape[1] <= 0 or features.shape[2] <= 0:
        raise ValueError(f"{name} dimensions must be non-empty")
    if not features.is_floating_point():
        raise TypeError(f"{name} must use a floating dtype")


def official_g_p_pro(features: torch.Tensor, power: int = 8) -> torch.Tensor:
    """Exact one-dimensional form of official ``G_p_pro``.

    The fixed source detaches features, raises them elementwise to ``power``,
    forms a channel Gram matrix over all positions, then applies the signed
    elementwise ``power``-root.
    """

    _validate_feature_map(features)
    if not isinstance(power, int) or power <= 0:
        raise ValueError("Gram power must be a positive integer")
    powered = features.detach() ** power
    flattened = powered.reshape(powered.shape[0], powered.shape[1], -1)
    gram = torch.matmul(flattened, flattened.transpose(1, 2))
    return gram.sign() * gram.abs().pow(1.0 / power)


def build_official_score_templates(
    features: torch.Tensor,
    predictions: torch.Tensor,
    *,
    num_classes: int = 5,
    power: int = 8,
) -> OfficialScoreTemplates:
    """Build code-reference S2/S3 templates from raw train-known features."""

    _validate_feature_map(features)
    if num_classes <= 1:
        raise ValueError("template construction requires at least two classes")
    if predictions.ndim != 1 or predictions.shape[0] != features.shape[0]:
        raise ValueError("predictions must have shape [batch]")
    predictions = predictions.to(device=features.device, dtype=torch.long)
    if bool(((predictions < 0) | (predictions >= num_classes)).any()):
        raise ValueError("prediction index is outside the known classes")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("template features contain NaN or Inf")

    absolute = features.detach().abs()
    first_order_rows: list[torch.Tensor] = []
    gram_rows: list[torch.Tensor] = []
    counts: list[int] = []
    for class_index in range(num_classes):
        selected = predictions == class_index
        count = int(selected.sum().item())
        if count == 0:
            raise ValueError(
                f"official score template prediction class {class_index} is empty"
            )
        class_features = absolute[selected]
        first_order_rows.append(class_features.mean(dim=(0, 2)))
        gram_rows.append(official_g_p_pro(class_features, power=power).mean(dim=0))
        counts.append(count)

    first_order_raw = torch.stack(first_order_rows, dim=0)
    cross_class_sum = first_order_raw.sum(dim=0)
    if not bool(torch.isfinite(cross_class_sum).all()) or bool(
        (cross_class_sum == 0).any()
    ):
        raise ValueError("first-order cross-class normalization has a zero divisor")
    first_order = first_order_raw / cross_class_sum
    gram = torch.stack(gram_rows, dim=0)
    if not bool(torch.isfinite(first_order).all()) or not bool(
        torch.isfinite(gram).all()
    ):
        raise ValueError("official score templates contain NaN or Inf")
    return OfficialScoreTemplates(
        first_order=first_order,
        gram=gram,
        counts=torch.tensor(counts, dtype=torch.long, device=features.device),
        num_classes=int(num_classes),
        power=int(power),
    )


def raw_official_scores(
    features: torch.Tensor,
    logits: torch.Tensor,
    predicted_class: torch.Tensor,
    templates: OfficialScoreTemplates,
) -> OfficialRawScores:
    """Compute official code-semantics S1/S2/S3 for a common predicted class.

    S2 deliberately uses the raw signed test feature.  The fixed official
    source takes ``abs`` while building the train template but leaves the
    test-time ``abs`` line commented out.
    """

    _validate_feature_map(features)
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch,classes,length]")
    if logits.shape[0] != features.shape[0] or logits.shape[2] != features.shape[2]:
        raise ValueError("features and logits do not share batch/length dimensions")
    if logits.shape[1] != templates.num_classes:
        raise ValueError("logit class count differs from score templates")
    if predicted_class.ndim != 1 or predicted_class.shape[0] != features.shape[0]:
        raise ValueError("predicted_class must have shape [batch]")
    predicted_class = predicted_class.to(device=features.device, dtype=torch.long)
    if bool(
        ((predicted_class < 0) | (predicted_class >= templates.num_classes)).any()
    ):
        raise ValueError("predicted class is outside the score templates")
    if templates.first_order.shape != (
        templates.num_classes,
        features.shape[1],
    ):
        raise ValueError("first-order template shape differs from feature channels")
    if templates.gram.shape != (
        templates.num_classes,
        features.shape[1],
        features.shape[1],
    ):
        raise ValueError("Gram template shape differs from feature channels")
    if templates.first_order.device != features.device or templates.gram.device != features.device:
        raise ValueError("score templates and features must be on the same device")

    gather_index = predicted_class[:, None, None].expand(-1, 1, logits.shape[2])
    selected_logits = logits.gather(dim=1, index=gather_index).squeeze(1)
    activation = features.abs().mean(dim=1)
    # No denominator epsilon is permitted here: this is the fixed source's
    # literal R[0]/R[1]/R[1] array operation.  Non-finite output is a hard
    # experiment failure instead of a silently changed score.
    s1 = (selected_logits / activation / activation).reshape(features.shape[0], -1).mean(
        dim=1
    )

    selected_first_order = templates.first_order.index_select(0, predicted_class)
    s2 = (features * selected_first_order.unsqueeze(-1)).mean(dim=1).reshape(
        features.shape[0], -1
    ).mean(dim=1)

    gram = official_g_p_pro(features, power=templates.power)
    selected_gram = templates.gram.index_select(0, predicted_class)
    s3 = (gram * selected_gram).sum(dim=(1, 2))
    values = torch.stack((s1, s2, s3), dim=-1)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("official raw scores contain NaN or Inf")
    return OfficialRawScores(s1=s1, s2=s2, s3=s3)


def _score_values(scores: torch.Tensor | OfficialRawScores) -> torch.Tensor:
    values = scores.values if isinstance(scores, OfficialRawScores) else scores
    if values.ndim != 2 or values.shape[1] != len(OFFICIAL_SCORE_NAMES):
        raise ValueError("raw scores must have shape [samples,3]")
    if not values.is_floating_point():
        raise TypeError("raw scores must use a floating dtype")
    return values


def fit_score_normalization(
    raw_scores: torch.Tensor | OfficialRawScores,
    *,
    min_std: float = 1.0e-12,
    epsilon: float = 1.0e-8,
) -> OfficialScoreNormalization:
    """Fit float64 population mean/std using augmented train-known scores."""

    if min_std < 0.0:
        raise ValueError("minimum score standard deviation must be non-negative")
    if epsilon < 0.0:
        raise ValueError("normalization epsilon must be non-negative")
    values = _score_values(raw_scores).to(dtype=torch.float64)
    if values.shape[0] <= 0:
        raise ValueError("score normalization needs at least one sample")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("normalization scores contain NaN or Inf")
    mean = values.mean(dim=0)
    std = values.std(dim=0, correction=0)
    if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()):
        raise ValueError("score normalization statistics contain NaN or Inf")
    if bool((std <= float(min_std)).any()):
        raise ValueError("score normalization std is not greater than the frozen minimum")
    return OfficialScoreNormalization(
        mean=mean,
        std=std,
        epsilon=float(epsilon),
        min_std=float(min_std),
    )


def standardize_and_integrate(
    raw_scores: torch.Tensor | OfficialRawScores,
    normalization: OfficialScoreNormalization,
) -> OfficialStandardizedScores:
    """Standardize S1/S2/S3 and sum them with fixed unit weights."""

    values = _score_values(raw_scores).to(dtype=torch.float64)
    mean = normalization.mean.to(device=values.device, dtype=torch.float64)
    std = normalization.std.to(device=values.device, dtype=torch.float64)
    if mean.shape != (3,) or std.shape != (3,):
        raise ValueError("normalization mean/std must have shape [3]")
    if (
        not bool(torch.isfinite(mean).all())
        or not bool(torch.isfinite(std).all())
        or bool((std <= float(normalization.min_std)).any())
    ):
        raise ValueError(
            "normalization mean/std are non-finite or violate the frozen minimum"
        )
    if normalization.epsilon != 1.0e-8 or normalization.min_std != 1.0e-12:
        raise ValueError("official score-normalization contract changed")
    standardized = (values - mean) / (std + normalization.epsilon)
    if not bool(torch.isfinite(standardized).all()):
        raise ValueError("standardized official scores contain NaN or Inf")
    return OfficialStandardizedScores(
        standardized=standardized,
        integrated=standardized.sum(dim=1),
    )


def mean_view_probabilities(view_probabilities: torch.Tensor) -> torch.Tensor:
    """Symmetric two-view extension: average per-view class probabilities."""

    if view_probabilities.ndim != 3 or view_probabilities.shape[1] != 2:
        raise ValueError("view probabilities must have shape [batch,2,classes]")
    if view_probabilities.shape[2] <= 1:
        raise ValueError("view probabilities require at least two classes")
    if not bool(torch.isfinite(view_probabilities).all()):
        raise ValueError("view probabilities contain NaN or Inf")
    return view_probabilities.mean(dim=1)


def matched_linear_pair_output(
    view_logits: torch.Tensor,
    view_probabilities: torch.Tensor,
) -> MatchedLinearPairOutput:
    """Return the preregistered matched-linear pair prediction and score."""

    if view_logits.ndim != 4 or view_logits.shape[1] != 2:
        raise ValueError("view logits must have shape [batch,2,classes,length]")
    if view_logits.shape[:3] != view_probabilities.shape:
        raise ValueError("linear view logits and probabilities do not align")
    pair_probabilities = mean_view_probabilities(view_probabilities)
    max_pair_probability, predicted_class = pair_probabilities.max(dim=1)
    pair_spatial_average_logits = view_logits.mean(dim=-1).mean(dim=1)
    max_spatial_average_logit = pair_spatial_average_logits.max(dim=1).values
    return MatchedLinearPairOutput(
        pair_probabilities=pair_probabilities,
        predicted_class=predicted_class,
        max_spatial_average_logit=max_spatial_average_logit,
        max_pair_probability=max_pair_probability,
        unknown_score=-max_pair_probability,
    )


def official_pcssr_pair_scores(
    view_features: torch.Tensor,
    view_logits: torch.Tensor,
    view_probabilities: torch.Tensor,
    templates: OfficialScoreTemplates,
    normalization: OfficialScoreNormalization,
) -> OfficialPairScores:
    """Apply one common pair prediction to both views and average knownness."""

    if view_features.ndim != 4 or view_features.shape[1] != 2:
        raise ValueError("view features must have shape [batch,2,channels,length]")
    if view_logits.ndim != 4 or view_logits.shape[1] != 2:
        raise ValueError("view logits must have shape [batch,2,classes,length]")
    if (
        view_features.shape[0] != view_logits.shape[0]
        or view_features.shape[3] != view_logits.shape[3]
    ):
        raise ValueError("pCSSR view features and logits do not align")
    if view_logits.shape[:3] != view_probabilities.shape:
        raise ValueError("pCSSR view logits and probabilities do not align")

    batch, views, channels, length = view_features.shape
    pair_probabilities = mean_view_probabilities(view_probabilities)
    max_pair_probability, predicted_class = pair_probabilities.max(dim=1)
    repeated_prediction = predicted_class[:, None].expand(-1, views).reshape(-1)
    raw = raw_official_scores(
        view_features.reshape(batch * views, channels, length),
        view_logits.reshape(
            batch * views,
            view_logits.shape[2],
            view_logits.shape[3],
        ),
        repeated_prediction,
        templates,
    )
    standardized = standardize_and_integrate(raw, normalization).standardized.reshape(
        batch, views, 3
    )
    raw_values = raw.values.reshape(batch, views, 3)
    pair_components = standardized.mean(dim=1)
    knownness_by_rule = {
        "S1": pair_components[:, 0],
        "S2": pair_components[:, 1],
        "S3": pair_components[:, 2],
        "S1+S2": pair_components[:, 0] + pair_components[:, 1],
        "S1+S3": pair_components[:, 0] + pair_components[:, 2],
        "S2+S3": pair_components[:, 1] + pair_components[:, 2],
        "full": pair_components.sum(dim=1),
        "pcssr_max_pair_probability": max_pair_probability,
    }
    unknown_scores_by_rule = {
        name: -knownness for name, knownness in knownness_by_rule.items()
    }
    return OfficialPairScores(
        pair_probabilities=pair_probabilities,
        predicted_class=predicted_class,
        per_view_raw=raw_values,
        per_view_standardized=standardized,
        pair_standardized_components=pair_components,
        knownness_by_rule=knownness_by_rule,
        unknown_scores_by_rule=unknown_scores_by_rule,
    )
