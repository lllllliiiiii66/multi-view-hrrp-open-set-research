from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from hrrp_osr.data.errors import DataValidationError


SHARED_TRAIN_BASE_PCA = "shared_unique_train_base_pca_v1"


@dataclass(frozen=True)
class SharedPCAModel:
    mean: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray
    fit_sample_ids_sha256: str
    fit_sample_count: int

    @property
    def output_dimension(self) -> int:
        return int(self.components.shape[1])

    @property
    def cumulative_explained_variance(self) -> float:
        return float(np.sum(self.explained_variance_ratio))


def sample_ids_sha256(sample_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for sample_id in sample_ids:
        encoded = str(sample_id).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def fit_shared_pca(
    profiles: np.ndarray,
    *,
    sample_ids: Sequence[str],
    output_dimension: int,
) -> SharedPCAModel:
    matrix = np.asarray(profiles, dtype=np.float64)
    ids = tuple(str(sample_id) for sample_id in sample_ids)
    errors: list[str] = []
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        errors.append("PCA profiles must have shape [n_samples, n_features]")
    if matrix.shape[0] != len(ids):
        errors.append("PCA sample IDs do not match the profile count")
    if len(set(ids)) != len(ids):
        errors.append("PCA fit population contains duplicate sample IDs")
    if not np.isfinite(matrix).all():
        errors.append("PCA profiles contain NaN or Inf")
    maximum_dimension = min(matrix.shape) if matrix.ndim == 2 else 0
    if not 1 <= int(output_dimension) <= maximum_dimension:
        errors.append("PCA output dimension is outside the available rank")
    if errors:
        raise DataValidationError("Invalid shared PCA input:\n- " + "\n- ".join(errors))

    mean = matrix.mean(axis=0)
    centered = matrix - mean
    _, singular_values, right_vectors = np.linalg.svd(
        centered,
        full_matrices=False,
    )
    components = right_vectors[: int(output_dimension)].T.copy()

    # Resolve the otherwise arbitrary SVD sign so the saved transform is stable.
    anchors = np.argmax(np.abs(components), axis=0)
    signs = np.sign(components[anchors, np.arange(components.shape[1])])
    signs[signs == 0.0] = 1.0
    components *= signs

    squared = singular_values * singular_values
    total = float(np.sum(squared))
    if not np.isfinite(total) or total <= 0.0:
        raise DataValidationError("PCA fit population has no finite variance")
    explained = squared[: int(output_dimension)] / total
    return SharedPCAModel(
        mean=mean,
        components=components,
        explained_variance_ratio=explained,
        fit_sample_ids_sha256=sample_ids_sha256(ids),
        fit_sample_count=len(ids),
    )


def apply_shared_pca(profiles: np.ndarray, model: SharedPCAModel) -> np.ndarray:
    matrix = np.asarray(profiles, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != model.mean.size:
        raise DataValidationError("PCA transform feature dimension changed")
    transformed = (matrix - model.mean) @ model.components
    if not np.isfinite(transformed).all():
        raise DataValidationError("PCA transform produced NaN or Inf")
    return transformed
