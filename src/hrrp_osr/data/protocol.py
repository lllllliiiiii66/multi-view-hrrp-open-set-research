from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Iterable

from .config import ClassPartitionConfig, ProtocolConfig
from .errors import DataValidationError


def normalize_class_name(name: str) -> str:
    return unicodedata.normalize("NFC", name.strip())


def angle_domain_and_split(angle_deg: int, protocol: ProtocolConfig) -> tuple[str, str]:
    full_span = protocol.domain_count * protocol.domain_width_deg
    if angle_deg < 0 or angle_deg >= full_span:
        raise ValueError(f"angle_deg must be in [0, {full_span - 1}], got {angle_deg}")
    domain_index = angle_deg // protocol.domain_width_deg
    within_domain = angle_deg % protocol.domain_width_deg
    if within_domain < protocol.train_width_deg:
        split = "train"
    elif within_domain < protocol.train_width_deg + protocol.validation_width_deg:
        split = "validation"
    else:
        split = "test"
    return f"D{domain_index}", split


def _partition_score(name: str, partition: ClassPartitionConfig) -> str:
    payload = (
        f"{partition.algorithm_version}\0{partition.seed}\0{normalize_class_name(name)}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_class_partition(
    class_names: Iterable[str], partition: ClassPartitionConfig
) -> dict[str, str]:
    normalized = [normalize_class_name(name) for name in class_names]
    if len(normalized) != len(set(normalized)):
        raise DataValidationError("Class names collide after Unicode NFC normalization")
    expected_total = partition.known_count + partition.unknown_count
    if len(normalized) != expected_total:
        raise DataValidationError(
            f"Expected {expected_total} classes, found {len(normalized)}"
        )
    ranked = sorted(normalized, key=lambda name: (_partition_score(name, partition), name))
    unknown = set(ranked[: partition.unknown_count])
    return {name: ("unknown" if name in unknown else "known") for name in sorted(normalized)}
