from __future__ import annotations

import math

import pytest

from hrrp_osr.data.errors import DataValidationError
from hrrp_osr.evaluation.aggregate import summarize_seed_values


def test_summarize_seed_values_uses_sample_std_and_student_t_interval() -> None:
    summary = summarize_seed_values([1.0, 2.0, 3.0, 4.0, 5.0])
    assert summary["seed_count"] == 5
    assert summary["mean"] == pytest.approx(3.0)
    assert summary["sample_std"] == pytest.approx(math.sqrt(2.5))
    assert summary["ci95_low"] == pytest.approx(1.0367568385)
    assert summary["ci95_high"] == pytest.approx(4.9632431615)


def test_summarize_seed_values_rejects_single_or_nonfinite_values() -> None:
    with pytest.raises(DataValidationError, match="at least two"):
        summarize_seed_values([1.0])
    with pytest.raises(DataValidationError, match="finite"):
        summarize_seed_values([1.0, float("nan")])
