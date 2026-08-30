from __future__ import annotations

import numpy as np
import pytest

from hrrp_osr.amdr.reduction import apply_shared_pca, fit_shared_pca
from hrrp_osr.data.errors import DataValidationError


def test_shared_pca_is_centered_orthonormal_and_deterministic() -> None:
    rng = np.random.default_rng(17)
    profiles = rng.normal(size=(12, 6))
    sample_ids = tuple(f"sample-{index:02d}" for index in range(len(profiles)))

    first = fit_shared_pca(
        profiles,
        sample_ids=sample_ids,
        output_dimension=3,
    )
    second = fit_shared_pca(
        profiles,
        sample_ids=sample_ids,
        output_dimension=3,
    )

    assert np.array_equal(first.mean, second.mean)
    assert np.array_equal(first.components, second.components)
    assert first.fit_sample_ids_sha256 == second.fit_sample_ids_sha256
    np.testing.assert_allclose(
        first.components.T @ first.components,
        np.eye(3),
        atol=1.0e-14,
    )
    transformed = apply_shared_pca(profiles, first)
    np.testing.assert_allclose(transformed.mean(axis=0), np.zeros(3), atol=1.0e-14)
    assert 0.0 < first.cumulative_explained_variance <= 1.0


def test_shared_pca_rejects_duplicate_fit_sample_ids() -> None:
    with pytest.raises(DataValidationError, match="duplicate sample IDs"):
        fit_shared_pca(
            np.eye(3),
            sample_ids=("same", "same", "other"),
            output_dimension=2,
        )
