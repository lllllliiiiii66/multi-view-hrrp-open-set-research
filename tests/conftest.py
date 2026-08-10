from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from hrrp_osr.data.config import DataConfig, load_data_config
from hrrp_osr.data.padding import PaddingConfig, load_padding_config


@pytest.fixture(scope="session")
def data_config() -> DataConfig:
    root = Path(__file__).resolve().parents[1]
    return load_data_config(root / "configs/data/hrrp_10class_theta83_hh_v1.yaml")


@pytest.fixture(scope="session")
def padding_config() -> PaddingConfig:
    root = Path(__file__).resolve().parents[1]
    return load_padding_config(
        root / "configs/data/hrrp_padding_complex_gaussian_v1.yaml"
    )


@pytest.fixture(scope="session")
def synthetic_raw_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("synthetic_hrrp")
    phi_grid = np.round(np.arange(0.0, 360.0 + 0.1, 0.1), 1)
    phi = np.concatenate([phi_grid, phi_grid])
    theta = np.concatenate(
        [np.full(phi_grid.size, 83.0), np.full(phi_grid.size, 85.0)]
    )

    for class_index in range(10):
        profile_length = 3 + 2 * (class_index % 3)
        range_axis = np.linspace(
            -0.6 * (profile_length - 1) / 2,
            0.6 * (profile_length - 1) / 2,
            profile_length,
        )
        profiles = (
            -90.0
            + class_index * 3.0
            + phi[:, None] * 0.01
            + theta[:, None] * 0.001
            + np.arange(profile_length, dtype=float)[None, :] * 0.1
        )
        ranges = np.broadcast_to(range_axis, profiles.shape).copy()
        nfreq = np.full((phi.size, 1), float(profile_length))
        permutation = np.random.default_rng(class_index).permutation(phi.size)
        merged = {
            "phi": phi[permutation, None],
            "theta": theta[permutation, None],
            "Nfreq": nfreq[permutation],
            "RangeX": ranges[permutation],
            "TrcsHH": profiles[permutation],
        }
        savemat(root / f"class_{class_index:02d}.mat", {"merged": merged})
    return root
