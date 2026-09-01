from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml
from scipy.io import loadmat

from .config import DataConfig
from .errors import DataValidationError
from .protocol import angle_domain_and_split, normalize_class_name, stable_class_partition


MANIFEST_FIELDS = [
    "sample_id",
    "dataset_id",
    "class_name",
    "class_role",
    "angle_deg",
    "elevation_deg",
    "domain_id",
    "split",
    "usage",
    "eligible_for_training",
    "eligible_for_validation",
    "eligible_for_evaluation",
    "source_file",
    "source_file_sha256",
    "source_variable",
    "source_row_index",
    "source_matlab_row_index",
    "profile_field",
    "profile_sha256",
    "profile_length",
    "profile_representation",
    "range_min",
    "range_max",
    "range_step",
    "class_partition_seed",
    "class_partition_algorithm",
]


@dataclass(frozen=True)
class ManifestBuild:
    rows: tuple[dict[str, Any], ...]
    summary: dict[str, Any]
    validation: dict[str, Any]


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def profile_sha256(profile: np.ndarray) -> str:
    canonical = np.ascontiguousarray(profile, dtype="<f8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _sample_id(
    config: DataConfig,
    source_hash: str,
    source_row_index: int,
    angle_deg: int,
) -> str:
    payload = "\0".join(
        [
            config.dataset_id,
            source_hash,
            config.source.mat_variable,
            str(source_row_index),
            config.source.profile_field,
            str(angle_deg),
            f"{config.selection.elevation_deg:g}",
        ]
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _usage_and_eligibility(class_role: str, split: str) -> tuple[str, int, int, int]:
    if class_role == "known":
        usage = {
            "train": "model_train",
            "validation": "known_validation",
            "test": "known_test",
        }[split]
        return usage, int(split == "train"), int(split == "validation"), int(split == "test")
    usage = "unknown_test" if split == "test" else "held_out_unknown"
    return usage, 0, 0, int(split == "test")


def _extract_struct_fields(mat_path: Path, config: DataConfig) -> dict[str, np.ndarray]:
    loaded = loadmat(
        mat_path,
        variable_names=[config.source.mat_variable],
        struct_as_record=True,
        squeeze_me=False,
    )
    if config.source.mat_variable not in loaded:
        raise DataValidationError(
            f"{mat_path.name}: missing MAT variable {config.source.mat_variable!r}"
        )
    struct = loaded[config.source.mat_variable]
    if struct.shape != (1, 1) or not struct.dtype.names:
        raise DataValidationError(
            f"{mat_path.name}: {config.source.mat_variable!r} must be a 1x1 MATLAB struct"
        )
    required = {
        config.source.angle_field,
        config.source.elevation_field,
        config.source.profile_field,
        config.source.range_field,
        config.source.nfreq_field,
    }
    missing = sorted(required.difference(struct.dtype.names))
    if missing:
        raise DataValidationError(f"{mat_path.name}: missing struct fields {missing}")
    return {name: struct[name][0, 0] for name in required}


def _selected_source_rows(fields: Mapping[str, np.ndarray], config: DataConfig) -> dict[int, int]:
    angles = np.asarray(fields[config.source.angle_field], dtype=float).reshape(-1)
    elevations = np.asarray(fields[config.source.elevation_field], dtype=float).reshape(-1)
    if angles.shape != elevations.shape:
        raise DataValidationError("Angle and elevation arrays have different row counts")
    rounded = np.rint(angles).astype(int)
    selected = (
        np.isclose(
            elevations,
            config.selection.elevation_deg,
            rtol=0.0,
            atol=config.selection.angle_tolerance,
        )
        & np.isclose(
            angles,
            rounded,
            rtol=0.0,
            atol=config.selection.angle_tolerance,
        )
        & (rounded >= config.selection.angle_start_deg)
        & (rounded <= config.selection.angle_stop_deg)
    )
    result: dict[int, int] = {}
    for angle in range(
        config.selection.angle_start_deg,
        config.selection.angle_stop_deg + 1,
        config.selection.angle_step_deg,
    ):
        indices = np.flatnonzero(selected & (rounded == angle))
        if indices.size != 1:
            raise DataValidationError(
                f"Expected exactly one row for theta={config.selection.elevation_deg:g}, "
                f"phi={angle}, found {indices.size}"
            )
        result[angle] = int(indices[0])
    return result


def _class_rows(
    mat_path: Path,
    class_name: str,
    class_role: str,
    source_hash: str,
    config: DataConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fields = _extract_struct_fields(mat_path, config)
    source_rows = _selected_source_rows(fields, config)
    profiles = np.asarray(fields[config.source.profile_field])
    ranges = np.asarray(fields[config.source.range_field], dtype=float)
    nfreq = np.asarray(fields[config.source.nfreq_field], dtype=float).reshape(-1)
    if profiles.ndim != 2 or ranges.ndim != 2 or profiles.shape != ranges.shape:
        raise DataValidationError(
            f"{mat_path.name}: profile and RangeX must be equal-shape 2-D matrices"
        )
    if profiles.shape[0] != nfreq.size:
        raise DataValidationError(f"{mat_path.name}: Nfreq row count does not match profiles")
    selected_indices = np.array([source_rows[angle] for angle in sorted(source_rows)])
    selected_profiles = np.asarray(profiles[selected_indices], dtype=float)
    selected_ranges = np.asarray(ranges[selected_indices], dtype=float)
    if not np.isfinite(selected_profiles).all():
        raise DataValidationError(f"{mat_path.name}: selected profiles contain NaN or Inf")
    if not np.isfinite(selected_ranges).all():
        raise DataValidationError(f"{mat_path.name}: selected RangeX contains NaN or Inf")
    if not np.allclose(selected_ranges, selected_ranges[0], rtol=0.0, atol=1e-12):
        raise DataValidationError(f"{mat_path.name}: RangeX changes between selected rows")

    range_axis = selected_ranges[0]
    if range_axis.size > 1:
        steps = np.diff(range_axis)
        if not np.allclose(steps, steps[0], rtol=0.0, atol=1e-10):
            raise DataValidationError(f"{mat_path.name}: RangeX is not uniformly spaced")
        range_step = float(steps[0])
    else:
        range_step = 0.0
    if not np.allclose(nfreq[selected_indices], profiles.shape[1], rtol=0.0, atol=0.0):
        raise DataValidationError(f"{mat_path.name}: Nfreq does not match profile length")

    rows: list[dict[str, Any]] = []
    for angle_deg, source_row_index in sorted(source_rows.items()):
        domain_id, split = angle_domain_and_split(angle_deg, config.protocol)
        usage, eligible_train, eligible_validation, eligible_evaluation = (
            _usage_and_eligibility(class_role, split)
        )
        profile = np.asarray(profiles[source_row_index], dtype=float).reshape(-1)
        rows.append(
            {
                "sample_id": _sample_id(config, source_hash, source_row_index, angle_deg),
                "dataset_id": config.dataset_id,
                "class_name": class_name,
                "class_role": class_role,
                "angle_deg": angle_deg,
                "elevation_deg": f"{config.selection.elevation_deg:g}",
                "domain_id": domain_id,
                "split": split,
                "usage": usage,
                "eligible_for_training": eligible_train,
                "eligible_for_validation": eligible_validation,
                "eligible_for_evaluation": eligible_evaluation,
                "source_file": mat_path.name,
                "source_file_sha256": source_hash,
                "source_variable": config.source.mat_variable,
                "source_row_index": source_row_index,
                "source_matlab_row_index": source_row_index + 1,
                "profile_field": config.source.profile_field,
                "profile_sha256": profile_sha256(profile),
                "profile_length": int(profile.size),
                "profile_representation": config.expected.profile_representation,
                "range_min": f"{float(range_axis.min()):.12g}",
                "range_max": f"{float(range_axis.max()):.12g}",
                "range_step": f"{range_step:.12g}",
                "class_partition_seed": config.class_partition.seed,
                "class_partition_algorithm": config.class_partition.algorithm_version,
            }
        )

    flat_profiles = selected_profiles.reshape(-1)
    class_summary = {
        "class_name": class_name,
        "class_role": class_role,
        "source_file": mat_path.name,
        "source_file_sha256": source_hash,
        "selected_samples": len(rows),
        "profile_length": int(profiles.shape[1]),
        "range_min": float(range_axis.min()),
        "range_max": float(range_axis.max()),
        "range_step": range_step,
        "profile_min": float(flat_profiles.min()),
        "profile_max": float(flat_profiles.max()),
        "negative_value_count": int(np.count_nonzero(flat_profiles < 0)),
        "zero_value_count": int(np.count_nonzero(flat_profiles == 0)),
        "positive_value_count": int(np.count_nonzero(flat_profiles > 0)),
        "profile_value_count": int(flat_profiles.size),
    }
    return rows, class_summary


def validate_manifest_rows(
    rows: Iterable[Mapping[str, Any]], config: DataConfig
) -> dict[str, Any]:
    materialized = list(rows)
    errors: list[str] = []
    expected_angles = set(
        range(
            config.selection.angle_start_deg,
            config.selection.angle_stop_deg + 1,
            config.selection.angle_step_deg,
        )
    )
    expected_total = config.expected.class_count * config.expected.samples_per_class
    if len(materialized) != expected_total:
        errors.append(f"manifest row count is {len(materialized)}, expected {expected_total}")

    sample_ids = [str(row["sample_id"]) for row in materialized]
    if len(sample_ids) != len(set(sample_ids)):
        errors.append("sample_id is not unique")

    by_class: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    source_identity_splits: dict[tuple[str, int], set[str]] = defaultdict(set)
    angle_identity_splits: dict[tuple[str, int], set[str]] = defaultdict(set)
    profile_hash_splits: dict[str, set[str]] = defaultdict(set)
    for row in materialized:
        class_name = str(row["class_name"])
        angle = int(row["angle_deg"])
        split = str(row["split"])
        by_class[class_name].append(row)
        source_identity_splits[(str(row["source_file"]), int(row["source_row_index"]))].add(split)
        angle_identity_splits[(class_name, angle)].add(split)
        profile_hash_splits[str(row["profile_sha256"])].add(split)

        expected_domain, expected_split = angle_domain_and_split(angle, config.protocol)
        if str(row["domain_id"]) != expected_domain or split != expected_split:
            errors.append(
                f"{class_name} angle {angle}: got {row['domain_id']}/{split}, "
                f"expected {expected_domain}/{expected_split}"
            )
        role = str(row["class_role"])
        eligible_train = int(row["eligible_for_training"])
        eligible_validation = int(row["eligible_for_validation"])
        eligible_evaluation = int(row["eligible_for_evaluation"])
        if role == "unknown" and (eligible_train or eligible_validation):
            errors.append(f"unknown class {class_name} is eligible for train/validation")
        expected_usage = _usage_and_eligibility(role, split)
        actual_usage = (
            str(row["usage"]),
            eligible_train,
            eligible_validation,
            eligible_evaluation,
        )
        if actual_usage != expected_usage:
            errors.append(f"{class_name} angle {angle}: invalid usage/eligibility flags")

    if len(by_class) != config.expected.class_count:
        errors.append(f"found {len(by_class)} classes, expected {config.expected.class_count}")

    role_by_class: dict[str, str] = {}
    split_counts_by_class: dict[str, dict[str, int]] = {}
    for class_name, class_rows in sorted(by_class.items()):
        roles = {str(row["class_role"]) for row in class_rows}
        if len(roles) != 1:
            errors.append(f"{class_name}: class_role changes within class")
            continue
        role_by_class[class_name] = next(iter(roles))
        angles = [int(row["angle_deg"]) for row in class_rows]
        if len(angles) != len(set(angles)):
            errors.append(f"{class_name}: duplicate selected angles")
        if set(angles) != expected_angles:
            missing = sorted(expected_angles.difference(angles))
            extra = sorted(set(angles).difference(expected_angles))
            errors.append(f"{class_name}: angle coverage mismatch; missing={missing}, extra={extra}")
        split_counts = Counter(str(row["split"]) for row in class_rows)
        expected_split_counts = {"train": 216, "validation": 72, "test": 72}
        if dict(split_counts) != expected_split_counts:
            errors.append(
                f"{class_name}: split counts {dict(split_counts)}, expected {expected_split_counts}"
            )
        split_counts_by_class[class_name] = dict(split_counts)
        domain_split_counts = Counter(
            (str(row["domain_id"]), str(row["split"])) for row in class_rows
        )
        for domain_index in range(config.protocol.domain_count):
            domain = f"D{domain_index}"
            for split, expected_count in (("train", 36), ("validation", 12), ("test", 12)):
                if domain_split_counts[(domain, split)] != expected_count:
                    errors.append(
                        f"{class_name}: {domain}/{split} has "
                        f"{domain_split_counts[(domain, split)]}, expected {expected_count}"
                    )
        lengths = {int(row["profile_length"]) for row in class_rows}
        if len(lengths) != 1:
            errors.append(f"{class_name}: profile length changes within class")

    role_counts = Counter(role_by_class.values())
    expected_roles = {
        "known": config.class_partition.known_count,
        "unknown": config.class_partition.unknown_count,
    }
    if dict(role_counts) != expected_roles:
        errors.append(f"class role counts {dict(role_counts)}, expected {expected_roles}")

    if any(len(splits) > 1 for splits in source_identity_splits.values()):
        errors.append("the same source file/row identity appears across multiple splits")
    if any(len(splits) > 1 for splits in angle_identity_splits.values()):
        errors.append("the same class/angle identity appears across multiple splits")
    cross_split_profile_hashes = sorted(
        digest for digest, splits in profile_hash_splits.items() if len(splits) > 1
    )
    if cross_split_profile_hashes:
        errors.append(
            f"{len(cross_split_profile_hashes)} exact profile hashes appear across multiple splits"
        )

    if errors:
        raise DataValidationError("Manifest validation failed:\n- " + "\n- ".join(errors))

    return {
        "status": "passed",
        "checks": {
            "angle_coverage": "passed",
            "boundary_uniqueness": "passed",
            "split_counts": "passed",
            "split_mutual_exclusion": "passed",
            "continuous_block_rule": "passed",
            "class_partition_7_known_3_unknown": "passed",
            "unknown_class_isolation": "passed",
            "source_identity_uniqueness": "passed",
            "exact_profile_hash_cross_split_leakage": "passed",
        },
        "row_count": len(materialized),
        "class_count": len(by_class),
        "class_roles": role_by_class,
        "split_counts_by_class": split_counts_by_class,
    }


def build_manifest_rows(config: DataConfig, raw_root: str | Path) -> ManifestBuild:
    root = Path(raw_root).expanduser().resolve()
    if not root.is_dir():
        raise DataValidationError(f"Raw data root does not exist or is not a directory: {root}")
    mat_files = sorted(root.glob(config.source.file_glob), key=lambda path: normalize_class_name(path.stem))
    if len(mat_files) != config.expected.class_count:
        raise DataValidationError(
            f"Expected {config.expected.class_count} MAT files, found {len(mat_files)} in {root}"
        )
    class_names = [normalize_class_name(path.stem) for path in mat_files]
    class_roles = stable_class_partition(class_names, config.class_partition)

    all_rows: list[dict[str, Any]] = []
    class_summaries: list[dict[str, Any]] = []
    for mat_path, class_name in zip(mat_files, class_names, strict=True):
        source_hash = file_sha256(mat_path)
        rows, class_summary = _class_rows(
            mat_path,
            class_name,
            class_roles[class_name],
            source_hash,
            config,
        )
        all_rows.extend(rows)
        class_summaries.append(class_summary)

    all_rows.sort(key=lambda row: (str(row["class_name"]), int(row["angle_deg"])))
    validation = validate_manifest_rows(all_rows, config)

    total_values = sum(item["profile_value_count"] for item in class_summaries)
    negative_values = sum(item["negative_value_count"] for item in class_summaries)
    zero_values = sum(item["zero_value_count"] for item in class_summaries)
    positive_values = sum(item["positive_value_count"] for item in class_summaries)
    summary = {
        "dataset_id": config.dataset_id,
        "raw_root": str(root),
        "source_layout": config.source.layout,
        "source_format": "MATLAB v5 MAT, one 1x1 struct variable per class file",
        "selected_elevation_deg": config.selection.elevation_deg,
        "selected_profile_field": config.source.profile_field,
        "selected_angle_range_deg": [
            config.selection.angle_start_deg,
            config.selection.angle_stop_deg,
        ],
        "selected_angle_step_deg": config.selection.angle_step_deg,
        "class_partition": {
            "method": config.class_partition.method,
            "algorithm_version": config.class_partition.algorithm_version,
            "seed": config.class_partition.seed,
            "known_classes": sorted(
                name for name, role in class_roles.items() if role == "known"
            ),
            "unknown_classes": sorted(
                name for name, role in class_roles.items() if role == "unknown"
            ),
        },
        "classes": class_summaries,
        "profile_representation": {
            "value": config.expected.profile_representation,
            "linearization": config.expected.profile_linearization,
            "global_min": min(item["profile_min"] for item in class_summaries),
            "global_max": max(item["profile_max"] for item in class_summaries),
            "negative_value_count": negative_values,
            "zero_value_count": zero_values,
            "positive_value_count": positive_values,
            "total_value_count": total_values,
            "negative_value_rate": negative_values / total_values,
            "warning": (
                "Do not zero-pad or draw additive Gaussian values directly in dB. "
                "TrcsHH is 10*log10(power); construct nonnegative noise power in the "
                "linear domain and convert it back with 10*log10."
            ),
        },
        "padding": copy.deepcopy(config.raw_mapping["preprocessing"]["padding"]),
    }
    return ManifestBuild(rows=tuple(all_rows), summary=summary, validation=validation)


def render_manifest_csv(rows: Iterable[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in MANIFEST_FIELDS})
    return buffer.getvalue().encode("utf-8")


def write_manifest_bundle(
    build: ManifestBuild,
    config: DataConfig,
    output_dir: str | Path,
) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_bytes = render_manifest_csv(build.rows)
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    generated_at = datetime.now(timezone.utc).isoformat()

    manifest_path = destination / "samples.csv"
    manifest_path.write_bytes(manifest_bytes)
    (destination / "samples.sha256").write_text(
        f"{manifest_hash}  samples.csv\n", encoding="utf-8"
    )

    resolved_config = copy.deepcopy(dict(config.raw_mapping))
    resolved_config["_resolved"] = {
        "generated_at_utc": generated_at,
        "raw_root": build.summary["raw_root"],
        "known_classes": build.summary["class_partition"]["known_classes"],
        "unknown_classes": build.summary["class_partition"]["unknown_classes"],
        "manifest_sha256": manifest_hash,
    }
    (destination / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    report = {
        "generated_at_utc": generated_at,
        "manifest_file": "samples.csv",
        "manifest_sha256": manifest_hash,
        "summary": build.summary,
        "validation": build.validation,
    }
    (destination / "audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "output_dir": str(destination.resolve()),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": manifest_hash,
        "row_count": len(build.rows),
        "validation_status": build.validation["status"],
        "known_classes": build.summary["class_partition"]["known_classes"],
        "unknown_classes": build.summary["class_partition"]["unknown_classes"],
    }
