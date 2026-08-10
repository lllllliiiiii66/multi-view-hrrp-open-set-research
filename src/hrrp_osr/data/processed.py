from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .errors import DataValidationError
from .manifest import file_sha256


INTEGER_FIELDS = {
    "angle_deg",
    "eligible_for_evaluation",
    "eligible_for_training",
    "eligible_for_validation",
    "source_row_index",
    "source_matlab_row_index",
    "profile_length",
    "class_partition_seed",
    "processed_row_index",
    "processed_profile_length",
    "left_padding_bins",
    "right_padding_bins",
    "padding_base_seed",
    "derived_padding_seed",
}


@dataclass(frozen=True)
class ProcessedBundle:
    root: Path
    profiles: np.ndarray
    rows: tuple[dict[str, Any], ...]
    profiles_sha256: str
    manifest_sha256: str
    bundle_sha256: str

    @property
    def known_classes(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    str(row["class_name"])
                    for row in self.rows
                    if row["class_role"] == "known"
                }
            )
        )


def _read_single_hash(path: Path, expected_name: str) -> str:
    if not path.is_file():
        raise DataValidationError(f"missing checksum file: {path}")
    parts = path.read_text(encoding="utf-8").strip().split()
    if len(parts) != 2 or parts[1] != expected_name or len(parts[0]) != 64:
        raise DataValidationError(f"invalid checksum file: {path}")
    return parts[0]


def _load_rows(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.is_file():
        raise DataValidationError(f"missing processed manifest: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for raw_row in reader:
            row: dict[str, Any] = dict(raw_row)
            for field in INTEGER_FIELDS:
                if field not in row:
                    raise DataValidationError(
                        f"processed manifest is missing required field {field}"
                    )
                row[field] = int(row[field])
            rows.append(row)
    return tuple(rows)


def validate_processed_bundle(
    bundle: ProcessedBundle,
    expected_profiles_sha256: str | None = None,
    expected_manifest_sha256: str | None = None,
    expected_bundle_sha256: str | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    if bundle.profiles.shape != (3600, 601):
        errors.append(f"profile shape is {bundle.profiles.shape}, expected (3600, 601)")
    if bundle.profiles.dtype != np.dtype("float64"):
        errors.append(f"profile dtype is {bundle.profiles.dtype}, expected float64")
    if len(bundle.rows) != 3600:
        errors.append(f"manifest has {len(bundle.rows)} rows, expected 3600")
    if not np.isfinite(bundle.profiles).all():
        errors.append("processed profiles contain NaN or Inf")
    if expected_profiles_sha256 and bundle.profiles_sha256 != expected_profiles_sha256:
        errors.append("processed profile file hash does not match experiment config")
    if expected_manifest_sha256 and bundle.manifest_sha256 != expected_manifest_sha256:
        errors.append("processed manifest hash does not match experiment config")
    if expected_bundle_sha256 and bundle.bundle_sha256 != expected_bundle_sha256:
        errors.append("processed bundle hash does not match experiment config")

    row_indices = [int(row["processed_row_index"]) for row in bundle.rows]
    if row_indices != list(range(len(bundle.rows))):
        errors.append("processed_row_index is not contiguous manifest order")
    sample_ids = [str(row["sample_id"]) for row in bundle.rows]
    if len(sample_ids) != len(set(sample_ids)):
        errors.append("processed manifest sample_id is not unique")
    derived_seeds = [int(row["derived_padding_seed"]) for row in bundle.rows]
    if len(derived_seeds) != len(set(derived_seeds)):
        errors.append("derived padding seed is not unique")
    if any(int(row["processed_profile_length"]) != 601 for row in bundle.rows):
        errors.append("not every processed profile length is 601")
    if any(
        str(row["accepted_risk_id"]) != "profile_length_role_shortcut_v1"
        or str(row["accepted_risk_status"]) != "accepted_for_first_round"
        for row in bundle.rows
    ):
        errors.append("accepted profile-length risk is not traceable on every row")

    role_counts = Counter(
        {
            str(row["class_name"]): str(row["class_role"])
            for row in bundle.rows
        }.values()
    )
    if role_counts != {"known": 7, "unknown": 3}:
        errors.append(f"class role counts are {dict(role_counts)}, expected 7/3")
    unknown_rows = [row for row in bundle.rows if row["class_role"] == "unknown"]
    if any(
        int(row["eligible_for_training"]) or int(row["eligible_for_validation"])
        for row in unknown_rows
    ):
        errors.append("unknown rows are eligible for training or validation")
    if errors:
        raise DataValidationError(
            "Processed bundle validation failed:\n- " + "\n- ".join(errors)
        )
    return {
        "status": "passed",
        "checks": {
            "shape_3600_by_601": "passed",
            "dtype_float64": "passed",
            "finite_values": "passed",
            "file_hash_binding": "passed",
            "manifest_row_order": "passed",
            "sample_id_uniqueness": "passed",
            "padding_seed_uniqueness": "passed",
            "accepted_risk_traceability": "passed",
            "class_partition_7_known_3_unknown": "passed",
            "unknown_class_isolation": "passed",
        },
    }


def load_processed_bundle(
    root: str | Path,
    *,
    expected_profiles_sha256: str | None = None,
    expected_manifest_sha256: str | None = None,
    expected_bundle_sha256: str | None = None,
) -> ProcessedBundle:
    bundle_root = Path(root).expanduser().resolve()
    profiles_path = bundle_root / "profiles.npy"
    manifest_path = bundle_root / "samples.csv"
    if not profiles_path.is_file():
        raise DataValidationError(f"missing processed profiles: {profiles_path}")
    recorded_profiles_hash = _read_single_hash(
        bundle_root / "profiles.npy.sha256", "profiles.npy"
    )
    recorded_manifest_hash = _read_single_hash(
        bundle_root / "samples.csv.sha256", "samples.csv"
    )
    bundle_hash = _read_single_hash(
        bundle_root / "bundle.sha256", "hrrp_padding_complex_gaussian_v1"
    )
    profiles_hash = file_sha256(profiles_path)
    manifest_hash = file_sha256(manifest_path)
    if profiles_hash != recorded_profiles_hash:
        raise DataValidationError("profiles.npy does not match its checksum sidecar")
    if manifest_hash != recorded_manifest_hash:
        raise DataValidationError("samples.csv does not match its checksum sidecar")
    profiles = np.load(profiles_path, mmap_mode="r", allow_pickle=False)
    rows = _load_rows(manifest_path)
    bundle = ProcessedBundle(
        root=bundle_root,
        profiles=profiles,
        rows=rows,
        profiles_sha256=profiles_hash,
        manifest_sha256=manifest_hash,
        bundle_sha256=bundle_hash,
    )
    validate_processed_bundle(
        bundle,
        expected_profiles_sha256=expected_profiles_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_bundle_sha256=expected_bundle_sha256,
    )
    return bundle


def rows_by_sample_id(rows: tuple[Mapping[str, Any], ...]) -> dict[str, Mapping[str, Any]]:
    result = {str(row["sample_id"]): row for row in rows}
    if len(result) != len(rows):
        raise DataValidationError("sample_id is not unique")
    return result
