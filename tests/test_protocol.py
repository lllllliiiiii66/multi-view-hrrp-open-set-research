from __future__ import annotations

from collections import Counter

import pytest

from hrrp_osr.data.config import DataConfig
from hrrp_osr.data.protocol import angle_domain_and_split, stable_class_partition


@pytest.mark.parametrize(
    ("angle", "expected"),
    [
        (0, ("D0", "train")),
        (35, ("D0", "train")),
        (36, ("D0", "validation")),
        (47, ("D0", "validation")),
        (48, ("D0", "test")),
        (59, ("D0", "test")),
        (60, ("D1", "train")),
        (335, ("D5", "train")),
        (336, ("D5", "validation")),
        (347, ("D5", "validation")),
        (348, ("D5", "test")),
        (359, ("D5", "test")),
    ],
)
def test_frozen_boundaries(data_config: DataConfig, angle: int, expected: tuple[str, str]) -> None:
    assert angle_domain_and_split(angle, data_config.protocol) == expected


def test_full_angle_counts(data_config: DataConfig) -> None:
    assignments = [angle_domain_and_split(angle, data_config.protocol) for angle in range(360)]
    assert Counter(split for _, split in assignments) == {
        "train": 216,
        "validation": 72,
        "test": 72,
    }
    for domain_index in range(6):
        per_domain = Counter(
            split for domain, split in assignments if domain == f"D{domain_index}"
        )
        assert per_domain == {"train": 36, "validation": 12, "test": 12}


def test_angles_outside_full_circle_are_rejected(data_config: DataConfig) -> None:
    with pytest.raises(ValueError):
        angle_domain_and_split(-1, data_config.protocol)
    with pytest.raises(ValueError):
        angle_domain_and_split(360, data_config.protocol)


def test_seeded_partition_is_stable_and_order_independent(data_config: DataConfig) -> None:
    classes = [
        "CVN77",
        "DDG-1000",
        "DDG-112",
        "LRYYC",
        "汽车运输船9000车级",
        "油气轮MARVEL CRANE",
        "海洋调查船向阳红10号",
        "爱达魔都号",
        "迷你好望角型散货船",
        "集装箱船达飞罗尔多夫级",
    ]
    forward = stable_class_partition(classes, data_config.class_partition)
    reverse = stable_class_partition(reversed(classes), data_config.class_partition)
    assert forward == reverse
    assert {name for name, role in forward.items() if role == "unknown"} == {
        "LRYYC",
        "汽车运输船9000车级",
        "海洋调查船向阳红10号",
    }
    assert Counter(forward.values()) == {"known": 7, "unknown": 3}
