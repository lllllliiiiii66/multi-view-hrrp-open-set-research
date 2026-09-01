from __future__ import annotations

import csv
import hashlib
import io
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from .errors import DataValidationError


SET_ALGORITHM_VERSION = "v3_disjoint_domain_partition_v1"
V5_SET_ALGORITHM_VERSION = "v5_leave_one_domain_out_v1"
B0_SELECTION_ALGORITHM_VERSION = "sha256_uniform_index_v1"


@dataclass(frozen=True)
class ViewSet:
    set_id: str
    dataset_id: str
    split: str
    class_name: str
    class_role: str
    set_repeat: int
    member_sample_ids: tuple[str, ...]
    member_domain_ids: tuple[str, ...]
    algorithm_version: str = SET_ALGORITHM_VERSION


@dataclass(frozen=True)
class B0ViewSelection:
    set_id: str
    selection_repeat: int
    selected_index: int
    selected_sample_id: str
    algorithm_version: str
    seed: int


def _derived_seed(*parts: str | int) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _set_id(
    dataset_id: str,
    split: str,
    class_name: str,
    set_repeat: int,
    round_index: int,
    slot_index: int,
    member_sample_ids: Iterable[str],
    *,
    algorithm_version: str = SET_ALGORITHM_VERSION,
) -> str:
    payload = "\0".join(
        [
            algorithm_version,
            dataset_id,
            split,
            class_name,
            str(set_repeat),
            str(round_index),
            str(slot_index),
            *sorted(member_sample_ids),
        ]
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _eligible_rows(
    rows: Iterable[Mapping[str, Any]], split: str
) -> list[Mapping[str, Any]]:
    if split == "train":
        return [
            row
            for row in rows
            if str(row["split"]) == split
            and str(row["class_role"]) == "known"
            and int(row["eligible_for_training"]) == 1
        ]
    if split == "validation":
        return [
            row
            for row in rows
            if str(row["split"]) == split
            and str(row["class_role"]) == "known"
            and int(row["eligible_for_validation"]) == 1
        ]
    if split == "test":
        return [
            row
            for row in rows
            if str(row["split"]) == split
            and int(row["eligible_for_evaluation"]) == 1
        ]
    raise DataValidationError("view sets may only use train, validation, or test")


def build_v3_evaluation_sets(
    rows: Iterable[Mapping[str, Any]],
    *,
    split: str,
    base_seed: int,
    set_repeat: int = 0,
) -> tuple[ViewSet, ...]:
    if split not in {"validation", "test"}:
        raise DataValidationError("V=3 evaluation sets may only use validation or test")
    return build_v3_sets(
        rows,
        split=split,
        base_seed=base_seed,
        set_repeat=set_repeat,
    )


def build_v3_sets(
    rows: Iterable[Mapping[str, Any]],
    *,
    split: str,
    base_seed: int,
    set_repeat: int = 0,
) -> tuple[ViewSet, ...]:
    selected = _eligible_rows(rows, split)
    by_class: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        by_class[str(row["class_name"])].append(row)
    expected_class_count = 10 if split == "test" else 7
    if len(by_class) != expected_class_count:
        raise DataValidationError(
            f"{split} set pool has {len(by_class)} classes, expected {expected_class_count}"
        )

    sets: list[ViewSet] = []
    for class_name, class_rows in sorted(by_class.items()):
        roles = {str(row["class_role"]) for row in class_rows}
        dataset_ids = {str(row["dataset_id"]) for row in class_rows}
        if len(roles) != 1 or len(dataset_ids) != 1:
            raise DataValidationError(f"{class_name}: inconsistent role or dataset_id")
        class_role = next(iter(roles))
        dataset_id = next(iter(dataset_ids))
        by_domain: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in class_rows:
            by_domain[str(row["domain_id"])].append(row)
        expected_domains = {f"D{index}" for index in range(6)}
        if set(by_domain) != expected_domains:
            raise DataValidationError(f"{class_name}: evaluation pool does not cover D0..D5")
        samples_per_domain = 36 if split == "train" else 12
        if any(len(domain_rows) != samples_per_domain for domain_rows in by_domain.values()):
            raise DataValidationError(
                f"{class_name}: every {split} domain must contain exactly "
                f"{samples_per_domain} samples"
            )

        shuffled_by_domain: dict[str, list[Mapping[str, Any]]] = {}
        for domain_id, domain_rows in sorted(by_domain.items()):
            ordered = sorted(domain_rows, key=lambda row: str(row["sample_id"]))
            rng = np.random.default_rng(
                _derived_seed(
                    SET_ALGORITHM_VERSION,
                    base_seed,
                    set_repeat,
                    split,
                    class_name,
                    domain_id,
                )
            )
            permutation = rng.permutation(len(ordered))
            shuffled_by_domain[domain_id] = [ordered[int(index)] for index in permutation]

        domains = sorted(expected_domains)
        for round_index in range(samples_per_domain):
            domain_rng = np.random.default_rng(
                _derived_seed(
                    SET_ALGORITHM_VERSION,
                    base_seed,
                    set_repeat,
                    split,
                    class_name,
                    "domain_partition",
                    round_index,
                )
            )
            shuffled_domains = [domains[int(index)] for index in domain_rng.permutation(6)]
            for slot_index, domain_triple in enumerate(
                (shuffled_domains[:3], shuffled_domains[3:])
            ):
                members = [
                    shuffled_by_domain[domain_id][round_index]
                    for domain_id in domain_triple
                ]
                order_rng = np.random.default_rng(
                    _derived_seed(
                        SET_ALGORITHM_VERSION,
                        base_seed,
                        set_repeat,
                        split,
                        class_name,
                        "input_order",
                        round_index,
                        slot_index,
                    )
                )
                input_order = order_rng.permutation(3)
                ordered_members = [members[int(index)] for index in input_order]
                member_ids = tuple(str(row["sample_id"]) for row in ordered_members)
                member_domains = tuple(str(row["domain_id"]) for row in ordered_members)
                sets.append(
                    ViewSet(
                        set_id=_set_id(
                            dataset_id,
                            split,
                            class_name,
                            set_repeat,
                            round_index,
                            slot_index,
                            member_ids,
                        ),
                        dataset_id=dataset_id,
                        split=split,
                        class_name=class_name,
                        class_role=class_role,
                        set_repeat=set_repeat,
                        member_sample_ids=member_ids,  # type: ignore[arg-type]
                        member_domain_ids=member_domains,  # type: ignore[arg-type]
                        algorithm_version=SET_ALGORITHM_VERSION,
                    )
                )
    result = tuple(sorted(sets, key=lambda item: (item.class_name, item.set_id)))
    validate_v3_sets(result, selected, split=split)
    return result


def validate_v3_sets(
    sets: Iterable[ViewSet],
    eligible_rows: Iterable[Mapping[str, Any]],
    *,
    split: str,
) -> dict[str, Any]:
    materialized = tuple(sets)
    errors: list[str] = []
    expected_count = {"train": 504, "validation": 168, "test": 240}[split]
    if len(materialized) != expected_count:
        errors.append(f"set count is {len(materialized)}, expected {expected_count}")
    set_ids = [item.set_id for item in materialized]
    if len(set_ids) != len(set(set_ids)):
        errors.append("set_id is not unique")
    used_sample_ids: list[str] = []
    for item in materialized:
        if item.split != split:
            errors.append(f"set {item.set_id} has wrong split")
        if len(set(item.member_sample_ids)) != 3:
            errors.append(f"set {item.set_id} repeats a base sample")
        if len(set(item.member_domain_ids)) != 3:
            errors.append(f"set {item.set_id} does not use three different domains")
        used_sample_ids.extend(item.member_sample_ids)
    expected_ids = {str(row["sample_id"]) for row in eligible_rows}
    if len(used_sample_ids) != len(set(used_sample_ids)):
        errors.append("a base HRRP is reused within one set repeat")
    if set(used_sample_ids) != expected_ids:
        errors.append("set members do not exactly cover the eligible split pool")
    if errors:
        raise DataValidationError("V=3 set validation failed:\n- " + "\n- ".join(errors))
    return {
        "status": "passed",
        "checks": {
            "set_count": "passed",
            "set_id_uniqueness": "passed",
            "three_distinct_domains": "passed",
            "base_sample_nonreuse_within_repeat": "passed",
            "eligible_pool_exact_coverage": "passed",
            "split_isolation": "passed",
        },
    }


def build_v5_sets(
    rows: Iterable[Mapping[str, Any]],
    *,
    split: str,
    base_seed: int,
    set_repeat: int = 0,
) -> tuple[ViewSet, ...]:
    selected = _eligible_rows(rows, split)
    by_class: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        by_class[str(row["class_name"])].append(row)
    expected_class_count = 10 if split == "test" else 7
    if len(by_class) != expected_class_count:
        raise DataValidationError(
            f"{split} V=5 pool has {len(by_class)} classes, expected {expected_class_count}"
        )
    samples_per_domain = 36 if split == "train" else 12
    result: list[ViewSet] = []
    domains = [f"D{index}" for index in range(6)]
    for class_name, class_rows in sorted(by_class.items()):
        by_domain: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in class_rows:
            by_domain[str(row["domain_id"])].append(row)
        if set(by_domain) != set(domains):
            raise DataValidationError(f"{class_name}: V=5 pool does not cover D0..D5")
        shuffled: dict[str, list[Mapping[str, Any]]] = {}
        for domain_id in domains:
            domain_rows = sorted(by_domain[domain_id], key=lambda row: str(row["sample_id"]))
            if len(domain_rows) != samples_per_domain:
                raise DataValidationError(
                    f"{class_name}/{domain_id}: invalid V=5 split count"
                )
            rng = np.random.default_rng(
                _derived_seed(
                    V5_SET_ALGORITHM_VERSION,
                    base_seed,
                    set_repeat,
                    split,
                    class_name,
                    domain_id,
                )
            )
            shuffled[domain_id] = [
                domain_rows[int(index)] for index in rng.permutation(samples_per_domain)
            ]
        dataset_id = str(class_rows[0]["dataset_id"])
        class_role = str(class_rows[0]["class_role"])
        for round_index in range(samples_per_domain):
            for omitted_domain_index in range(6):
                members = [
                    shuffled[domain_id][round_index]
                    for domain_id in domains
                    if domain_id != domains[omitted_domain_index]
                ]
                order_rng = np.random.default_rng(
                    _derived_seed(
                        V5_SET_ALGORITHM_VERSION,
                        base_seed,
                        set_repeat,
                        split,
                        class_name,
                        "input_order",
                        round_index,
                        omitted_domain_index,
                    )
                )
                ordered_members = [members[int(index)] for index in order_rng.permutation(5)]
                member_ids = tuple(str(row["sample_id"]) for row in ordered_members)
                member_domains = tuple(str(row["domain_id"]) for row in ordered_members)
                result.append(
                    ViewSet(
                        set_id=_set_id(
                            dataset_id,
                            split,
                            class_name,
                            set_repeat,
                            round_index,
                            omitted_domain_index,
                            member_ids,
                            algorithm_version=V5_SET_ALGORITHM_VERSION,
                        ),
                        dataset_id=dataset_id,
                        split=split,
                        class_name=class_name,
                        class_role=class_role,
                        set_repeat=set_repeat,
                        member_sample_ids=member_ids,
                        member_domain_ids=member_domains,
                        algorithm_version=V5_SET_ALGORITHM_VERSION,
                    )
                )
    materialized = tuple(sorted(result, key=lambda item: (item.class_name, item.set_id)))
    expected_count = expected_class_count * samples_per_domain * 6
    if len(materialized) != expected_count:
        raise DataValidationError("V=5 set count is invalid")
    usage: dict[str, int] = defaultdict(int)
    for item in materialized:
        if len(item.member_sample_ids) != 5 or len(set(item.member_domain_ids)) != 5:
            raise DataValidationError("V=5 set must contain five distinct domains")
        for sample_id in item.member_sample_ids:
            usage[sample_id] += 1
    expected_ids = {str(row["sample_id"]) for row in selected}
    if set(usage) != expected_ids or set(usage.values()) != {5}:
        raise DataValidationError("V=5 leave-one-domain-out coverage is invalid")
    return materialized


def select_b0_single_view(
    view_set: ViewSet,
    *,
    base_seed: int,
    selection_repeat: int,
) -> B0ViewSelection:
    seed = _derived_seed(
        B0_SELECTION_ALGORITHM_VERSION,
        base_seed,
        selection_repeat,
        view_set.set_id,
    )
    rng = np.random.default_rng(seed)
    selected_index = int(rng.integers(0, 3))
    return B0ViewSelection(
        set_id=view_set.set_id,
        selection_repeat=selection_repeat,
        selected_index=selected_index,
        selected_sample_id=view_set.member_sample_ids[selected_index],
        algorithm_version=B0_SELECTION_ALGORITHM_VERSION,
        seed=seed,
    )


def render_set_manifest_csv(sets: Iterable[ViewSet]) -> bytes:
    materialized = tuple(sets)
    if not materialized:
        raise DataValidationError("cannot render an empty set manifest")
    view_counts = {len(item.member_sample_ids) for item in materialized}
    if len(view_counts) != 1:
        raise DataValidationError("one set manifest cannot mix view counts")
    view_count = next(iter(view_counts))
    buffer = io.StringIO(newline="")
    fields = [
        "set_id",
        "dataset_id",
        "split",
        "class_name",
        "class_role",
        "set_repeat",
    ]
    for index in range(view_count):
        fields.extend([f"member_{index}_sample_id", f"member_{index}_domain_id"])
    fields.append("algorithm_version")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for item in materialized:
        row = {
                "set_id": item.set_id,
                "dataset_id": item.dataset_id,
                "split": item.split,
                "class_name": item.class_name,
                "class_role": item.class_role,
                "set_repeat": item.set_repeat,
                "algorithm_version": item.algorithm_version,
            }
        for index in range(view_count):
            row[f"member_{index}_sample_id"] = item.member_sample_ids[index]
            row[f"member_{index}_domain_id"] = item.member_domain_ids[index]
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8")
