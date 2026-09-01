from __future__ import annotations

import numpy as np

from hrrp_osr.data.errors import DataValidationError


def _validate_logits(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise DataValidationError("logits must have shape [samples, known_classes>=2]")
    if not np.isfinite(values).all():
        raise DataValidationError("logits contain NaN or Inf")
    return values


def softmax_probabilities(logits: np.ndarray) -> np.ndarray:
    values = _validate_logits(logits)
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def msp_unknown_score(logits: np.ndarray) -> np.ndarray:
    """Return 1-max softmax probability; larger values mean more unknown."""

    probabilities = softmax_probabilities(logits)
    return 1.0 - np.max(probabilities, axis=1)


def energy_unknown_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Return -T*logsumexp(logits/T); larger values mean more unknown."""

    values = _validate_logits(logits)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise DataValidationError("energy temperature must be finite and positive")
    scaled = values / temperature
    maximum = np.max(scaled, axis=1, keepdims=True)
    logsumexp = maximum[:, 0] + np.log(
        np.sum(np.exp(scaled - maximum), axis=1)
    )
    return -temperature * logsumexp
