from __future__ import annotations

import numpy as np
import pytest

from hrrp_osr.baselines.b0 import energy_unknown_score, msp_unknown_score
from hrrp_osr.data.errors import DataValidationError


def test_msp_and_energy_scores_are_larger_for_less_confident_logits() -> None:
    logits = np.array(
        [
            [10.0, -5.0, -5.0],
            [0.0, 0.0, 0.0],
        ]
    )
    msp = msp_unknown_score(logits)
    energy = energy_unknown_score(logits)
    assert msp[1] > msp[0]
    assert energy[1] > energy[0]


def test_energy_score_is_stable_for_large_logits() -> None:
    logits = np.array([[10000.0, 9999.0], [-10000.0, -10001.0]])
    scores = energy_unknown_score(logits)
    assert np.isfinite(scores).all()


def test_energy_rejects_invalid_temperature() -> None:
    with pytest.raises(DataValidationError, match="temperature"):
        energy_unknown_score(np.ones((2, 3)), temperature=0.0)
