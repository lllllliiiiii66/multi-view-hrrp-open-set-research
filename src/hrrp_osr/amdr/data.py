from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from hrrp_osr.data.errors import DataValidationError
from hrrp_osr.data.processed import ProcessedBundle


PAIR_ALGORITHM_VERSION = "amdr_ordered_cross_frame_balanced_v1"
FOLD_ALGORITHM_VERSION = "odd_angle_five_fold_frame_covered_v1"


@dataclass(frozen=True)
class TwoViewPair:
    pair_id: str
    split: str
    fold_index: int
    class_name: str
    class_role: str
    view1_sample_id: str
    view2_sample_id: str
    view1_row_index: int
    view2_row_index: int
    view1_frame_id: int
    view2_frame_id: int
    view1_angle_deg: int
    view2_angle_deg: int
    algorithm_version: str = PAIR_ALGORITHM_VERSION


def _derived_seed(*parts: str | int) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _pair_id(
    *,
    protocol_id: str,
    split: str,
    fold_index: int,
    class_name: str,
    view1_sample_id: str,
    view2_sample_id: str,
) -> str:
    payload = "\0".join(
        [
            PAIR_ALGORITHM_VERSION,
            protocol_id,
            split,
            str(fold_index),
            class_name,
            view1_sample_id,
            view2_sample_id,
        ]
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assign_odd_angle_folds(
    *,
    fold_count: int = 5,
    base_seed: int,
) -> dict[int, int]:
    """Assign 180 odd angles to equal folds while covering every frame per fold."""

    if fold_count != 5:
        raise DataValidationError("the frozen cross-fit protocol requires five folds")
    assignments: dict[int, int] = {}
    fold_sizes = [0] * fold_count
    rotation = _derived_seed(FOLD_ALGORITHM_VERSION, base_seed, "rotation") % fold_count
    for frame_id in range(24):
        frame_angles = [
            angle
            for angle in range(frame_id * 15, frame_id * 15 + 15)
            if angle % 2 == 1
        ]
        rng = np.random.default_rng(
            _derived_seed(FOLD_ALGORITHM_VERSION, base_seed, frame_id)
        )
        shuffled = [frame_angles[int(i)] for i in rng.permutation(len(frame_angles))]
        fold_order = [int(i) for i in rng.permutation(fold_count)]
        for angle, fold_index in zip(shuffled[:fold_count], fold_order, strict=True):
            assignments[angle] = fold_index
            fold_sizes[fold_index] += 1
        leftovers = shuffled[fold_count:]
        for extra_index, angle in enumerate(leftovers):
            fold_index = int((2 * frame_id + rotation + extra_index) % fold_count)
            assignments[angle] = fold_index
            fold_sizes[fold_index] += 1

    if set(assignments) != set(range(1, 360, 2)):
        raise DataValidationError("odd-angle fold assignment is incomplete")
    if fold_sizes != [36] * fold_count:
        raise DataValidationError(f"odd-angle fold sizes are {fold_sizes}, expected 36")
    for fold_index in range(fold_count):
        covered_frames = {
            angle // 15 for angle, fold in assignments.items() if fold == fold_index
        }
        if covered_frames != set(range(24)):
            raise DataValidationError(
                f"fold {fold_index} does not cover every 15-degree frame"
            )
    return assignments


def _select_balanced_pairs(
    rows: Sequence[Mapping[str, Any]],
    *,
    count: int,
    base_seed: int,
    protocol_id: str,
    split: str,
    fold_index: int,
    class_name: str,
) -> tuple[TwoViewPair, ...]:
    if count <= 0:
        raise DataValidationError("pair count must be positive")
    ordered = sorted(
        rows, key=lambda row: (int(row["angle_deg"]), str(row["sample_id"]))
    )
    if len({str(row["sample_id"]) for row in ordered}) != len(ordered):
        raise DataValidationError(f"{class_name}/{split}: repeated base sample")
    candidate_left: list[int] = []
    candidate_right: list[int] = []
    for left in range(len(ordered)):
        left_frame = int(ordered[left]["angle_deg"]) // 15
        for right in range(left + 1, len(ordered)):
            right_frame = int(ordered[right]["angle_deg"]) // 15
            if left_frame != right_frame:
                candidate_left.append(left)
                candidate_right.append(right)
    if count > len(candidate_left):
        raise DataValidationError(
            f"{class_name}/{split}: requested {count} pairs but only "
            f"{len(candidate_left)} unique cross-frame pairs exist"
        )

    left_array = np.asarray(candidate_left, dtype=np.int32)
    right_array = np.asarray(candidate_right, dtype=np.int32)
    available = np.ones(left_array.size, dtype=bool)
    usage = np.zeros(len(ordered), dtype=np.int32)
    rng = np.random.default_rng(
        _derived_seed(
            PAIR_ALGORITHM_VERSION,
            base_seed,
            protocol_id,
            fold_index,
            split,
            class_name,
        )
    )
    selected: list[TwoViewPair] = []
    for _ in range(count):
        candidate_indices = np.flatnonzero(available)
        left_usage = usage[left_array[candidate_indices]]
        right_usage = usage[right_array[candidate_indices]]
        max_usage = np.maximum(left_usage, right_usage)
        minimum_max = int(max_usage.min())
        best = candidate_indices[max_usage == minimum_max]
        usage_sum = usage[left_array[best]] + usage[right_array[best]]
        best = best[usage_sum == int(usage_sum.min())]
        candidate_index = int(best[int(rng.integers(0, len(best)))])
        available[candidate_index] = False
        left_index = int(left_array[candidate_index])
        right_index = int(right_array[candidate_index])
        usage[left_index] += 1
        usage[right_index] += 1
        if int(rng.integers(0, 2)) == 0:
            view1_index, view2_index = left_index, right_index
        else:
            view1_index, view2_index = right_index, left_index
        view1 = ordered[view1_index]
        view2 = ordered[view2_index]
        view1_sample_id = str(view1["sample_id"])
        view2_sample_id = str(view2["sample_id"])
        selected.append(
            TwoViewPair(
                pair_id=_pair_id(
                    protocol_id=protocol_id,
                    split=split,
                    fold_index=fold_index,
                    class_name=class_name,
                    view1_sample_id=view1_sample_id,
                    view2_sample_id=view2_sample_id,
                ),
                split=split,
                fold_index=fold_index,
                class_name=class_name,
                class_role=str(view1["class_role"]),
                view1_sample_id=view1_sample_id,
                view2_sample_id=view2_sample_id,
                view1_row_index=int(view1["processed_row_index"]),
                view2_row_index=int(view2["processed_row_index"]),
                view1_frame_id=int(view1["angle_deg"]) // 15,
                view2_frame_id=int(view2["angle_deg"]) // 15,
                view1_angle_deg=int(view1["angle_deg"]),
                view2_angle_deg=int(view2["angle_deg"]),
            )
        )
    return tuple(selected)


def build_fold_pairs(
    bundle: ProcessedBundle,
    *,
    protocol_id: str,
    fold_index: int,
    fold_count: int,
    base_seed: int,
    pairs_per_class: Mapping[str, int],
) -> tuple[tuple[TwoViewPair, ...], dict[str, Any]]:
    if fold_index not in range(fold_count):
        raise DataValidationError("fold_index is outside the configured fold range")
    angle_folds = assign_odd_angle_folds(
        fold_count=fold_count,
        base_seed=base_seed,
    )
    rows_by_split_class: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for row in bundle.rows:
        angle = int(row["angle_deg"])
        role = str(row["class_role"])
        class_name = str(row["class_name"])
        if angle % 2 == 0:
            rows_by_split_class[("test", class_name)].append(row)
        elif role == "known":
            split = "calibration" if angle_folds[angle] == fold_index else "train"
            rows_by_split_class[(split, class_name)].append(row)

    known_classes = tuple(sorted(bundle.known_classes))
    all_classes = tuple(
        sorted({str(row["class_name"]) for row in bundle.rows})
    )
    pairs: list[TwoViewPair] = []
    split_classes = {
        "train": known_classes,
        "calibration": known_classes,
        "test": all_classes,
    }
    expected_base_counts = {"train": 144, "calibration": 36, "test": 180}
    for split, classes in split_classes.items():
        pair_count = int(pairs_per_class[split])
        for class_name in classes:
            class_rows = rows_by_split_class[(split, class_name)]
            if len(class_rows) != expected_base_counts[split]:
                raise DataValidationError(
                    f"{class_name}/{split}: found {len(class_rows)} base rows, "
                    f"expected {expected_base_counts[split]}"
                )
            pairs.extend(
                _select_balanced_pairs(
                    class_rows,
                    count=pair_count,
                    base_seed=base_seed,
                    protocol_id=protocol_id,
                    split=split,
                    fold_index=fold_index,
                    class_name=class_name,
                )
            )
    materialized = tuple(
        sorted(pairs, key=lambda pair: (pair.split, pair.class_name, pair.pair_id))
    )
    audit = validate_fold_pairs(
        materialized,
        bundle=bundle,
        fold_index=fold_index,
        pairs_per_class=pairs_per_class,
    )
    audit["fold_assignment"] = {
        "algorithm_version": FOLD_ALGORITHM_VERSION,
        "fold_count": fold_count,
        "fold_index": fold_index,
        "odd_angles_per_fold": {
            str(fold): sum(value == fold for value in angle_folds.values())
            for fold in range(fold_count)
        },
        "every_fold_covers_all_frames": True,
    }
    return materialized, audit


def validate_fold_pairs(
    pairs: Sequence[TwoViewPair],
    *,
    bundle: ProcessedBundle,
    fold_index: int,
    pairs_per_class: Mapping[str, int],
) -> dict[str, Any]:
    errors: list[str] = []
    rows = {str(row["sample_id"]): row for row in bundle.rows}
    pair_ids = [pair.pair_id for pair in pairs]
    if len(pair_ids) != len(set(pair_ids)):
        errors.append("pair_id is not unique")
    unordered_by_split: set[tuple[str, str, str, str]] = set()
    base_usage: Counter[tuple[str, str, str]] = Counter()
    counts: Counter[tuple[str, str]] = Counter()
    for pair in pairs:
        if pair.fold_index != fold_index:
            errors.append(f"{pair.pair_id}: wrong fold index")
        if pair.view1_frame_id == pair.view2_frame_id:
            errors.append(f"{pair.pair_id}: views come from the same frame")
        if pair.view1_sample_id == pair.view2_sample_id:
            errors.append(f"{pair.pair_id}: repeated base sample")
        key = (
            pair.split,
            pair.class_name,
            *sorted((pair.view1_sample_id, pair.view2_sample_id)),
        )
        if key in unordered_by_split:
            errors.append(f"{pair.pair_id}: repeated unordered angle pair")
        unordered_by_split.add(key)
        for sample_id in (pair.view1_sample_id, pair.view2_sample_id):
            row = rows.get(sample_id)
            if row is None:
                errors.append(f"{pair.pair_id}: missing sample_id {sample_id}")
                continue
            if str(row["class_name"]) != pair.class_name:
                errors.append(f"{pair.pair_id}: cross-class pair")
            angle = int(row["angle_deg"])
            if pair.split == "test" and angle % 2 != 0:
                errors.append(f"{pair.pair_id}: test uses an odd angle")
            if pair.split != "test" and angle % 2 != 1:
                errors.append(f"{pair.pair_id}: development split uses an even angle")
            if pair.split != "test" and pair.class_role != "known":
                errors.append(f"{pair.pair_id}: unknown class entered development data")
            base_usage[(pair.split, pair.class_name, sample_id)] += 1
        counts[(pair.split, pair.class_name)] += 1
    expected_classes = {
        "train": 7,
        "calibration": 7,
        "test": 10,
    }
    for split, class_count in expected_classes.items():
        observed = [value for (item_split, _), value in counts.items() if item_split == split]
        if len(observed) != class_count or set(observed) != {
            int(pairs_per_class[split])
        }:
            errors.append(f"{split}: pair counts are invalid")
    usage_summary: dict[str, dict[str, float | int]] = {}
    for split in ("train", "calibration", "test"):
        usages = [
            value
            for (item_split, _, _), value in base_usage.items()
            if item_split == split
        ]
        usage_summary[split] = {
            "minimum": min(usages),
            "maximum": max(usages),
            "mean": float(np.mean(usages)),
        }
    if errors:
        raise DataValidationError(
            "AMDR two-view pair validation failed:\n- " + "\n- ".join(errors)
        )
    return {
        "status": "passed",
        "checks": {
            "pair_id_uniqueness": "passed",
            "cross_frame_views": "passed",
            "unordered_pair_uniqueness_within_split": "passed",
            "odd_development_even_test": "passed",
            "unknown_development_count_zero": "passed",
            "class_balanced_pair_counts": "passed",
        },
        "pair_count": len(pairs),
        "pair_counts_by_split": dict(Counter(pair.split for pair in pairs)),
        "base_usage_by_split": usage_summary,
    }


def relative_power_from_db(profiles_db: np.ndarray) -> np.ndarray:
    profiles = np.asarray(profiles_db, dtype=np.float64)
    if profiles.ndim != 2 or profiles.shape[1] != 601:
        raise DataValidationError("HRRP profiles must have shape [n, 601]")
    if not np.isfinite(profiles).all():
        raise DataValidationError("HRRP profiles contain NaN or Inf")
    shifted = (profiles - profiles.max(axis=1, keepdims=True)) / 10.0
    relative_power = np.power(10.0, shifted)
    if not np.isfinite(relative_power).all():
        raise DataValidationError("relative-power normalization produced NaN or Inf")
    return relative_power


def materialize_pair_views(
    bundle: ProcessedBundle,
    pairs: Sequence[TwoViewPair],
) -> tuple[np.ndarray, np.ndarray]:
    view1_indices = np.asarray([pair.view1_row_index for pair in pairs], dtype=np.int64)
    view2_indices = np.asarray([pair.view2_row_index for pair in pairs], dtype=np.int64)
    view1 = relative_power_from_db(np.asarray(bundle.profiles[view1_indices]))
    view2 = relative_power_from_db(np.asarray(bundle.profiles[view2_indices]))
    return view1, view2


def write_pair_manifest(path: str | Path, pairs: Iterable[TwoViewPair]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(pair) for pair in pairs]
    if not rows:
        raise DataValidationError("cannot write an empty pair manifest")
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
