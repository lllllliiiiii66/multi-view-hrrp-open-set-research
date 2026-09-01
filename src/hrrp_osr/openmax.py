from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import weibull_min

from hrrp_osr.data.errors import DataValidationError


@dataclass(frozen=True)
class WeibullTail:
    shape: float
    location: float
    scale: float
    requested_tail_size: int
    effective_tail_size: int


@dataclass(frozen=True)
class OpenMaxModel:
    mean_activation_vectors: np.ndarray
    tails: tuple[WeibullTail, ...]
    distance_type: str
    eucos_euclidean_scale: float
    alpha_rank: int
    fit_sample_ids_by_class: tuple[tuple[str, ...], ...]

    @property
    def class_count(self) -> int:
        return int(self.mean_activation_vectors.shape[0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_activation_vectors": self.mean_activation_vectors.tolist(),
            "tails": [tail.__dict__ for tail in self.tails],
            "distance_type": self.distance_type,
            "eucos_euclidean_scale": self.eucos_euclidean_scale,
            "alpha_rank": self.alpha_rank,
            "fit_sample_ids_by_class": [list(values) for values in self.fit_sample_ids_by_class],
        }


def activation_distance(
    activation: np.ndarray,
    mean_activation: np.ndarray,
    *,
    distance_type: str,
    eucos_euclidean_scale: float,
) -> float:
    left = np.asarray(activation, dtype=np.float64)
    right = np.asarray(mean_activation, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise DataValidationError("OpenMax activation vectors must be same-shape 1D arrays")
    euclidean = float(np.linalg.norm(left - right))
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        cosine_part = 0.0 if left_norm == right_norm else 1.0
    else:
        cosine_part = float(cosine(left, right))
    if distance_type == "euclidean":
        value = euclidean
    elif distance_type == "cosine":
        value = cosine_part
    elif distance_type == "eucos":
        if eucos_euclidean_scale <= 0:
            raise DataValidationError("eucos Euclidean scale must be positive")
        value = euclidean / eucos_euclidean_scale + cosine_part
    else:
        raise DataValidationError(f"unsupported OpenMax distance type: {distance_type}")
    if not np.isfinite(value):
        raise DataValidationError("OpenMax distance is not finite")
    return max(0.0, value)


def _fit_weibull_tail(distances: np.ndarray, requested_tail_size: int) -> WeibullTail:
    values = np.asarray(distances, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 3:
        raise DataValidationError("OpenMax requires at least three finite distances per class")
    if requested_tail_size < 3:
        raise DataValidationError("OpenMax requested tail size must be at least three")
    effective = min(int(requested_tail_size), int(values.size))
    tail = np.sort(np.maximum(values, np.finfo(np.float64).eps))[-effective:]
    if float(np.ptp(tail)) <= np.finfo(np.float64).eps * max(1.0, float(np.max(tail))):
        shape, location, scale = 100.0, 0.0, float(np.mean(tail))
    else:
        shape, location, scale = weibull_min.fit(tail, floc=0.0)
    parameters = np.asarray([shape, location, scale], dtype=np.float64)
    if not np.all(np.isfinite(parameters)) or shape <= 0 or scale <= 0:
        raise DataValidationError("OpenMax Weibull fit produced invalid parameters")
    return WeibullTail(
        shape=float(shape),
        location=float(location),
        scale=float(scale),
        requested_tail_size=int(requested_tail_size),
        effective_tail_size=effective,
    )


def fit_openmax(
    activations: np.ndarray,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    sample_ids: Sequence[str],
    *,
    class_count: int,
    tail_size: int,
    alpha_rank: int,
    distance_type: str,
    eucos_euclidean_scale: float,
) -> OpenMaxModel:
    values = np.asarray(activations, dtype=np.float64)
    truth = np.asarray(true_labels, dtype=int)
    predictions = np.asarray(predicted_labels, dtype=int)
    if values.ndim != 2 or values.shape[1] != class_count:
        raise DataValidationError("OpenMax uses the K-dimensional pre-softmax activation vector")
    if len(values) != len(truth) or len(values) != len(predictions) or len(values) != len(sample_ids):
        raise DataValidationError("OpenMax fit inputs have inconsistent lengths")
    if not 1 <= alpha_rank <= class_count:
        raise DataValidationError("OpenMax alpha rank must be between one and K")
    means: list[np.ndarray] = []
    tails: list[WeibullTail] = []
    fit_ids: list[tuple[str, ...]] = []
    for class_index in range(class_count):
        selected = np.flatnonzero((truth == class_index) & (predictions == class_index))
        if selected.size == 0:
            raise DataValidationError(f"OpenMax class {class_index} has no correctly classified fit samples")
        class_values = values[selected]
        mean = class_values.mean(axis=0)
        distances = np.asarray([
            activation_distance(
                item,
                mean,
                distance_type=distance_type,
                eucos_euclidean_scale=eucos_euclidean_scale,
            )
            for item in class_values
        ])
        means.append(mean)
        tails.append(_fit_weibull_tail(distances, tail_size))
        fit_ids.append(tuple(str(sample_ids[index]) for index in selected))
    return OpenMaxModel(
        mean_activation_vectors=np.stack(means),
        tails=tuple(tails),
        distance_type=distance_type,
        eucos_euclidean_scale=float(eucos_euclidean_scale),
        alpha_rank=int(alpha_rank),
        fit_sample_ids_by_class=tuple(fit_ids),
    )


def openmax_probabilities(activations: np.ndarray, model: OpenMaxModel) -> np.ndarray:
    values = np.asarray(activations, dtype=np.float64)
    one_dimensional = values.ndim == 1
    if one_dimensional:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != model.class_count:
        raise DataValidationError("OpenMax inference activation shape does not match fitted classes")
    outputs: list[np.ndarray] = []
    for activation in values:
        revised = activation.copy()
        order = np.argsort(-activation, kind="stable")
        for rank, class_index in enumerate(order[: model.alpha_rank]):
            distance = activation_distance(
                activation,
                model.mean_activation_vectors[class_index],
                distance_type=model.distance_type,
                eucos_euclidean_scale=model.eucos_euclidean_scale,
            )
            tail = model.tails[class_index]
            outlier_probability = float(
                weibull_min.cdf(distance, tail.shape, loc=tail.location, scale=tail.scale)
            )
            rank_weight = float(model.alpha_rank - rank) / model.alpha_rank
            revised[class_index] = activation[class_index] * (1.0 - rank_weight * outlier_probability)
        unknown_activation = float(np.sum(activation - revised))
        augmented = np.concatenate([revised, np.asarray([unknown_activation])])
        augmented -= np.max(augmented)
        exponentials = np.exp(augmented)
        probabilities = exponentials / exponentials.sum()
        if not np.all(np.isfinite(probabilities)):
            raise DataValidationError("OpenMax produced non-finite probabilities")
        outputs.append(probabilities)
    result = np.stack(outputs)
    return result[0] if one_dimensional else result
