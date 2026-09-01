from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from .config import DataConfig
from .errors import DataConfigError, DataValidationError
from .manifest import (
    MANIFEST_FIELDS,
    ManifestBuild,
    _extract_struct_fields,
    file_sha256,
    profile_sha256,
    render_manifest_csv,
)


PROCESSED_MANIFEST_FIELDS = [
    *MANIFEST_FIELDS,
    "processed_row_index",
    "processed_profile_sha256",
    "processed_profile_length",
    "left_padding_bins",
    "right_padding_bins",
    "padding_algorithm_version",
    "padding_base_seed",
    "derived_padding_seed",
    "padding_mean_power_db",
    "accepted_risk_id",
    "accepted_risk_status",
]


@dataclass(frozen=True)
class PaddingConfig:
    schema_version: int
    preprocessing_id: str
    input_dataset_id: str
    input_manifest_sha256: str
    input_profile_field: str
    input_representation: str
    input_db_definition: str
    target_length: int
    target_range_min: float
    target_range_max: float
    target_range_step: float
    placement: str
    noise_method: str
    algorithm_version: str
    base_seed: int
    mean_noise_power_db: float
    mean_noise_power_linear: float
    parameter_selection_source_split: str
    parameter_selection_uses_unknown_validation_or_test: bool
    risk_id: str
    risk_acceptance_status: str
    risk_accepted_by: str
    risk_accepted_on: str
    risk_treatment: str
    output_format: str
    output_dtype: str
    output_profile_file: str
    output_manifest_file: str
    output_row_order: str
    raw_mapping: Mapping[str, Any]


@dataclass(frozen=True)
class PaddingResult:
    profile_db: np.ndarray
    left_padding_bins: int
    right_padding_bins: int
    configured_mean_noise_power: float
    generated_mean_noise_power: float
    derived_seed: int

    @property
    def padding_bins(self) -> int:
        return self.left_padding_bins + self.right_padding_bins


def _section(mapping: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = mapping.get(name)
    if not isinstance(value, Mapping):
        raise DataConfigError(f"Missing or invalid padding configuration section: {name}")
    return value


def load_padding_config(path: str | Path) -> PaddingConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise DataConfigError("The padding configuration must be a YAML mapping")

    input_config = _section(raw, "input")
    target = _section(raw, "target_grid")
    noise = _section(raw, "noise")
    parameter_selection = _section(raw, "parameter_selection")
    risk_acceptance = _section(raw, "risk_acceptance")
    output = _section(raw, "output")
    try:
        config = PaddingConfig(
            schema_version=int(raw["schema_version"]),
            preprocessing_id=str(raw["preprocessing_id"]),
            input_dataset_id=str(input_config["dataset_id"]),
            input_manifest_sha256=str(input_config["manifest_sha256"]),
            input_profile_field=str(input_config["profile_field"]),
            input_representation=str(input_config["representation"]),
            input_db_definition=str(input_config["db_definition"]),
            target_length=int(target["length"]),
            target_range_min=float(target["range_min"]),
            target_range_max=float(target["range_max"]),
            target_range_step=float(target["range_step"]),
            placement=str(target["placement"]),
            noise_method=str(noise["method"]),
            algorithm_version=str(noise["algorithm_version"]),
            base_seed=int(noise["base_seed"]),
            mean_noise_power_db=float(noise["mean_power_db"]),
            mean_noise_power_linear=float(noise["mean_power_linear"]),
            parameter_selection_source_split=str(
                parameter_selection["source_split"]
            ),
            parameter_selection_uses_unknown_validation_or_test=bool(
                parameter_selection["unknown_validation_and_test_used"]
            ),
            risk_id=str(risk_acceptance["risk_id"]),
            risk_acceptance_status=str(risk_acceptance["status"]),
            risk_accepted_by=str(risk_acceptance["accepted_by"]),
            risk_accepted_on=str(risk_acceptance["accepted_on"]),
            risk_treatment=str(risk_acceptance["treatment"]),
            output_format=str(output["format"]),
            output_dtype=str(output["dtype"]),
            output_profile_file=str(output["profile_file"]),
            output_manifest_file=str(output["manifest_file"]),
            output_row_order=str(output["row_order"]),
            raw_mapping=raw,
        )
    except KeyError as exc:
        raise DataConfigError(
            f"Missing required padding configuration key: {exc.args[0]}"
        ) from exc
    validate_padding_config(config)
    return config


def validate_padding_config(config: PaddingConfig) -> None:
    errors: list[str] = []
    if config.schema_version != 1:
        errors.append("schema_version must be 1")
    if config.input_representation != "power_decibel":
        errors.append("input representation must be power_decibel")
    if config.input_db_definition != "10_log10_power":
        errors.append("input dB definition must be 10_log10_power")
    if config.target_length != 601:
        errors.append("the first-round common grid length must be 601")
    if not np.isclose(config.target_range_min, -180.0, rtol=0.0, atol=1e-12):
        errors.append("target range_min must be -180")
    if not np.isclose(config.target_range_max, 180.0, rtol=0.0, atol=1e-12):
        errors.append("target range_max must be 180")
    if not np.isclose(config.target_range_step, 0.6, rtol=0.0, atol=1e-12):
        errors.append("target range_step must be 0.6")
    expected_span = (config.target_length - 1) * config.target_range_step
    if not np.isclose(
        config.target_range_max - config.target_range_min,
        expected_span,
        rtol=0.0,
        atol=1e-10,
    ):
        errors.append("target grid bounds, step, and length are inconsistent")
    if config.placement != "centered_on_RangeX_zero":
        errors.append("padding placement must be centered_on_RangeX_zero")
    if config.noise_method != "complex_gaussian_amplitude_to_power":
        errors.append("noise method must be complex_gaussian_amplitude_to_power")
    if config.algorithm_version != "complex_gaussian_fixed_mean_power_v1":
        errors.append(
            "algorithm_version must be complex_gaussian_fixed_mean_power_v1"
        )
    if not np.isclose(config.mean_noise_power_db, -140.0, rtol=0.0, atol=1e-12):
        errors.append("the frozen mean complex-Gaussian noise power must be -140 dB")
    if not np.isclose(config.mean_noise_power_linear, 1.0e-14, rtol=1e-15, atol=0.0):
        errors.append("the frozen mean complex-Gaussian linear power must be 1e-14")
    expected_linear_power = np.power(10.0, config.mean_noise_power_db / 10.0)
    if not np.isclose(
        config.mean_noise_power_linear,
        expected_linear_power,
        rtol=1e-15,
        atol=0.0,
    ):
        errors.append("mean noise power dB and linear values are inconsistent")
    if config.parameter_selection_source_split != "known_train_only":
        errors.append("padding scale must be selected from known_train_only")
    if config.parameter_selection_uses_unknown_validation_or_test:
        errors.append("unknown, validation, and test data cannot select padding scale")
    if config.risk_id != "profile_length_role_shortcut_v1":
        errors.append("risk_id must be profile_length_role_shortcut_v1")
    if config.risk_acceptance_status != "accepted_for_first_round":
        errors.append("profile-length role risk must record accepted_for_first_round")
    if config.risk_accepted_by != "user":
        errors.append("profile-length role risk must record accepted_by=user")
    if config.risk_accepted_on != "2026-08-10":
        errors.append("profile-length role risk acceptance date must be 2026-08-10")
    if config.risk_treatment != "document_and_continue":
        errors.append("profile-length role risk treatment must be document_and_continue")
    if (
        config.output_format,
        config.output_dtype,
        config.output_profile_file,
        config.output_manifest_file,
        config.output_row_order,
    ) != (
        "numpy_npy",
        "float64",
        "profiles.npy",
        "samples.csv",
        "input_manifest_order",
    ):
        errors.append("the frozen derived-data output contract has changed")
    if errors:
        raise DataConfigError("Invalid padding configuration:\n- " + "\n- ".join(errors))


def db_to_linear_power(values_db: np.ndarray | list[float]) -> np.ndarray:
    values = np.asarray(values_db, dtype=np.float64)
    if not np.isfinite(values).all():
        raise DataValidationError("dB values contain NaN or Inf")
    power = np.power(10.0, values / 10.0)
    if not np.isfinite(power).all() or np.any(power <= 0.0):
        raise DataValidationError("dB-to-power conversion produced invalid values")
    return power


def linear_power_to_db(values_power: np.ndarray | list[float]) -> np.ndarray:
    values = np.asarray(values_power, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise DataValidationError("linear power must be finite and strictly positive")
    return 10.0 * np.log10(values)


def derive_sample_seed(sample_id: str, base_seed: int, algorithm_version: str) -> int:
    if not sample_id:
        raise DataValidationError("sample_id must be non-empty")
    payload = "\0".join(
        [algorithm_version, str(base_seed), sample_id]
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def pad_profile_db(
    profile_db: np.ndarray | list[float],
    sample_id: str,
    config: PaddingConfig,
) -> PaddingResult:
    source = np.asarray(profile_db, dtype=np.float64)
    if source.ndim != 1 or source.size == 0:
        raise DataValidationError("profile must be a non-empty one-dimensional array")
    if not np.isfinite(source).all():
        raise DataValidationError("profile contains NaN or Inf")
    missing = config.target_length - source.size
    if missing < 0:
        raise DataValidationError(
            f"profile length {source.size} exceeds target length {config.target_length}"
        )
    if missing % 2 != 0:
        raise DataValidationError(
            "profile and target lengths must have the same parity for centered padding"
        )

    derived_seed = derive_sample_seed(
        sample_id, config.base_seed, config.algorithm_version
    )
    if missing == 0:
        return PaddingResult(
            profile_db=source.copy(),
            left_padding_bins=0,
            right_padding_bins=0,
            configured_mean_noise_power=0.0,
            generated_mean_noise_power=0.0,
            derived_seed=derived_seed,
        )

    rng = np.random.default_rng(derived_seed)
    gaussian = rng.standard_normal((missing, 2), dtype=np.float64)
    unscaled_noise_power = np.sum(gaussian * gaussian, axis=1) / 2.0
    configured_mean_noise_power = config.mean_noise_power_linear
    noise_power = unscaled_noise_power * configured_mean_noise_power
    noise_db = linear_power_to_db(noise_power)

    left = missing // 2
    right = missing - left
    padded = np.concatenate([noise_db[:left], source, noise_db[left:]])
    generated_mean_noise_power = float(np.mean(noise_power, dtype=np.float64))
    if padded.size != config.target_length or not np.isfinite(padded).all():
        raise DataValidationError("padding produced an invalid output profile")
    if not np.array_equal(padded[left : left + source.size], source):
        raise DataValidationError("padding modified original HRRP bins")

    return PaddingResult(
        profile_db=padded,
        left_padding_bins=left,
        right_padding_bins=right,
        configured_mean_noise_power=configured_mean_noise_power,
        generated_mean_noise_power=generated_mean_noise_power,
        derived_seed=derived_seed,
    )


def _quantiles(values: np.ndarray) -> dict[str, float] | None:
    if values.size == 0:
        return None
    quantiles = np.quantile(values, [0.0, 0.01, 0.5, 0.99, 1.0])
    return {
        "min": float(quantiles[0]),
        "q01": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q99": float(quantiles[3]),
        "max": float(quantiles[4]),
    }


def audit_padding_on_manifest(
    build: ManifestBuild,
    data_config: DataConfig,
    padding_config: PaddingConfig,
    raw_root: str | Path,
) -> dict[str, Any]:
    manifest_hash = hashlib.sha256(render_manifest_csv(build.rows)).hexdigest()
    if padding_config.input_dataset_id != data_config.dataset_id:
        raise DataValidationError("padding input dataset_id does not match data config")
    if padding_config.input_manifest_sha256 != manifest_hash:
        raise DataValidationError(
            "padding input manifest hash does not match the rebuilt raw manifest"
        )
    if padding_config.input_profile_field != data_config.source.profile_field:
        raise DataValidationError("padding input profile_field does not match data config")

    root = Path(raw_root).expanduser().resolve()
    rows_by_file: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in build.rows:
        rows_by_file[str(row["source_file"])].append(row)
    lengths_by_role = {
        role: sorted(
            {
                int(row["profile_length"])
                for row in build.rows
                if str(row["class_role"]) == role
            }
        )
        for role in ("known", "unknown")
    }
    role_length_overlap = sorted(
        set(lengths_by_role["known"]).intersection(lengths_by_role["unknown"])
    )
    length_role_shortcut_detected = not role_length_overlap
    length_role_shortcut_accepted = (
        padding_config.risk_id == "profile_length_role_shortcut_v1"
        and padding_config.risk_acceptance_status == "accepted_for_first_round"
        and padding_config.risk_treatment == "document_and_continue"
    )
    length_role_shortcut_is_blocking = (
        length_role_shortcut_detected and not length_role_shortcut_accepted
    )

    all_padding_db: list[np.ndarray] = []
    junction_gaps_db: list[float] = []
    class_reports: list[dict[str, Any]] = []
    derived_seeds: set[int] = set()
    preserved_profiles = 0
    padded_profiles = 0
    unchanged_target_length_profiles = 0

    for source_file, rows in sorted(rows_by_file.items()):
        fields = _extract_struct_fields(root / source_file, data_config)
        profiles = np.asarray(fields[data_config.source.profile_field], dtype=np.float64)
        ranges = np.asarray(fields[data_config.source.range_field], dtype=np.float64)
        class_padding_db: list[np.ndarray] = []
        class_junction_gaps: list[float] = []
        class_pad_bins: int | None = None

        for row in rows:
            source_index = int(row["source_row_index"])
            source_profile = np.asarray(profiles[source_index], dtype=np.float64).reshape(-1)
            source_range = np.asarray(ranges[source_index], dtype=np.float64).reshape(-1)
            result = pad_profile_db(source_profile, str(row["sample_id"]), padding_config)
            if result.derived_seed in derived_seeds:
                raise DataValidationError("derived per-sample padding seed collision detected")
            derived_seeds.add(result.derived_seed)

            expected_left = int(
                round(
                    (float(source_range.min()) - padding_config.target_range_min)
                    / padding_config.target_range_step
                )
            )
            expected_right = int(
                round(
                    (padding_config.target_range_max - float(source_range.max()))
                    / padding_config.target_range_step
                )
            )
            if (
                expected_left != result.left_padding_bins
                or expected_right != result.right_padding_bins
            ):
                raise DataValidationError(
                    f"{source_file} row {source_index}: RangeX is not centered on target grid"
                )
            center = result.profile_db[
                result.left_padding_bins : result.left_padding_bins + source_profile.size
            ]
            if not np.array_equal(center, source_profile):
                raise DataValidationError("padding audit found modified original bins")
            preserved_profiles += 1

            if result.padding_bins == 0:
                unchanged_target_length_profiles += 1
                continue
            padded_profiles += 1
            class_pad_bins = result.padding_bins
            left_noise = result.profile_db[: result.left_padding_bins]
            right_noise = result.profile_db[-result.right_padding_bins :]
            noise_db = np.concatenate([left_noise, right_noise])
            class_padding_db.append(noise_db)
            all_padding_db.append(noise_db)
            class_junction_gaps.extend(
                [
                    abs(float(left_noise[-1] - source_profile[0])),
                    abs(float(right_noise[0] - source_profile[-1])),
                ]
            )
            junction_gaps_db.extend(class_junction_gaps[-2:])

        merged_padding = (
            np.concatenate(class_padding_db) if class_padding_db else np.array([], dtype=float)
        )
        class_reports.append(
            {
                "class_name": str(rows[0]["class_name"]),
                "class_role": str(rows[0]["class_role"]),
                "source_length": int(rows[0]["profile_length"]),
                "padding_bins_per_profile": int(class_pad_bins or 0),
                "sample_count": len(rows),
                "padding_db": _quantiles(merged_padding),
                "junction_absolute_gap_db": _quantiles(
                    np.asarray(class_junction_gaps, dtype=np.float64)
                ),
                "observed_mean_padding_power_db": (
                    float(linear_power_to_db([np.mean(db_to_linear_power(merged_padding))])[0])
                    if merged_padding.size
                    else None
                ),
            }
        )

    merged_all_padding = (
        np.concatenate(all_padding_db) if all_padding_db else np.array([], dtype=float)
    )
    checks = {
        "input_manifest_hash_match": "passed",
        "target_grid_alignment": "passed",
        "output_length_601": "passed",
        "finite_output": "passed",
        "original_bins_preserved_exactly": "passed",
        "fixed_profile_independent_noise_scale": "passed",
        "per_sample_seed_uniqueness": "passed",
        "known_unknown_profile_length_overlap": (
            "accepted_risk"
            if length_role_shortcut_detected and length_role_shortcut_accepted
            else "failed"
            if length_role_shortcut_is_blocking
            else "passed"
        ),
    }
    return {
        "status": "blocked" if length_role_shortcut_is_blocking else "passed",
        "mechanical_padding_status": "passed",
        "experiment_readiness": (
            "blocked_by_profile_length_role_shortcut"
            if length_role_shortcut_is_blocking
            else "passed_with_accepted_profile_length_role_risk"
            if length_role_shortcut_detected
            else "passed"
        ),
        "blocking_reason": (
            "Known and unknown classes have disjoint raw profile-length sets. "
            "If the padding boundary is detectable, length alone reveals the "
            "open-set role. Formal model training must remain blocked."
            if length_role_shortcut_is_blocking
            else None
        ),
        "profile_lengths_by_role": lengths_by_role,
        "known_unknown_profile_length_overlap": role_length_overlap,
        "accepted_risks": (
            [
                {
                    "risk_id": padding_config.risk_id,
                    "status": padding_config.risk_acceptance_status,
                    "accepted_by": padding_config.risk_accepted_by,
                    "accepted_on": padding_config.risk_accepted_on,
                    "treatment": padding_config.risk_treatment,
                }
            ]
            if length_role_shortcut_detected and length_role_shortcut_accepted
            else []
        ),
        "dataset_id": data_config.dataset_id,
        "preprocessing_id": padding_config.preprocessing_id,
        "input_manifest_sha256": manifest_hash,
        "noise_method": padding_config.noise_method,
        "algorithm_version": padding_config.algorithm_version,
        "base_seed": padding_config.base_seed,
        "configured_mean_noise_power_db": padding_config.mean_noise_power_db,
        "sample_count": len(build.rows),
        "padded_profile_count": padded_profiles,
        "already_target_length_profile_count": unchanged_target_length_profiles,
        "preserved_original_profile_count": preserved_profiles,
        "unique_derived_seed_count": len(derived_seeds),
        "total_generated_padding_bins": int(merged_all_padding.size),
        "padding_db": _quantiles(merged_all_padding),
        "junction_absolute_gap_db": _quantiles(
            np.asarray(junction_gaps_db, dtype=np.float64)
        ),
        "observed_mean_padding_power_db": (
            float(linear_power_to_db([np.mean(db_to_linear_power(merged_all_padding))])[0])
            if merged_all_padding.size
            else None
        ),
        "checks": checks,
        "classes": class_reports,
    }


def render_processed_manifest_csv(rows: list[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=PROCESSED_MANIFEST_FIELDS,
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in PROCESSED_MANIFEST_FIELDS})
    return buffer.getvalue().encode("utf-8")


def materialize_padded_dataset(
    build: ManifestBuild,
    data_config: DataConfig,
    padding_config: PaddingConfig,
    raw_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    audit = audit_padding_on_manifest(build, data_config, padding_config, raw_root)
    if audit["status"] != "passed":
        raise DataValidationError(
            f"padding audit status is {audit['status']}; derived data was not written"
        )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    profile_path = destination / padding_config.output_profile_file
    manifest_path = destination / padding_config.output_manifest_file
    root = Path(raw_root).expanduser().resolve()
    shape = (len(build.rows), padding_config.target_length)
    matrix = np.lib.format.open_memmap(
        profile_path,
        mode="w+",
        dtype=np.dtype("<f8"),
        shape=shape,
    )
    processed_rows: list[dict[str, Any] | None] = [None] * len(build.rows)
    indexed_rows_by_file: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for output_index, row in enumerate(build.rows):
        indexed_rows_by_file[str(row["source_file"])].append((output_index, row))

    for source_file, indexed_rows in sorted(indexed_rows_by_file.items()):
        fields = _extract_struct_fields(root / source_file, data_config)
        source_profiles = np.asarray(
            fields[data_config.source.profile_field], dtype=np.float64
        )
        for output_index, row in indexed_rows:
            source_index = int(row["source_row_index"])
            source_profile = np.asarray(
                source_profiles[source_index], dtype=np.float64
            ).reshape(-1)
            result = pad_profile_db(
                source_profile,
                str(row["sample_id"]),
                padding_config,
            )
            matrix[output_index] = result.profile_db
            processed_rows[output_index] = {
                **row,
                "processed_row_index": output_index,
                "processed_profile_sha256": profile_sha256(result.profile_db),
                "processed_profile_length": padding_config.target_length,
                "left_padding_bins": result.left_padding_bins,
                "right_padding_bins": result.right_padding_bins,
                "padding_algorithm_version": padding_config.algorithm_version,
                "padding_base_seed": padding_config.base_seed,
                "derived_padding_seed": result.derived_seed,
                "padding_mean_power_db": f"{padding_config.mean_noise_power_db:g}",
                "accepted_risk_id": padding_config.risk_id,
                "accepted_risk_status": padding_config.risk_acceptance_status,
            }
    matrix.flush()
    del matrix

    if any(row is None for row in processed_rows):
        raise DataValidationError("not every input manifest row was materialized")
    finalized_rows = [row for row in processed_rows if row is not None]
    manifest_bytes = render_processed_manifest_csv(finalized_rows)
    manifest_path.write_bytes(manifest_bytes)

    loaded = np.load(profile_path, mmap_mode="r", allow_pickle=False)
    if loaded.shape != shape:
        raise DataValidationError(
            f"materialized profile shape is {loaded.shape}, expected {shape}"
        )
    if loaded.dtype != np.dtype("float64"):
        raise DataValidationError(
            f"materialized profile dtype is {loaded.dtype}, expected float64"
        )
    if not np.isfinite(loaded).all():
        raise DataValidationError("materialized profiles contain NaN or Inf")
    processed_hash_splits: dict[str, set[str]] = defaultdict(set)
    derived_seed_values: set[int] = set()
    for output_index, row in enumerate(finalized_rows):
        actual_profile_hash = profile_sha256(np.asarray(loaded[output_index]))
        if actual_profile_hash != row["processed_profile_sha256"]:
            raise DataValidationError(
                f"processed profile hash mismatch at output row {output_index}"
            )
        processed_hash_splits[actual_profile_hash].add(str(row["split"]))
        derived_seed_values.add(int(row["derived_padding_seed"]))
        if (
            row["accepted_risk_id"] != padding_config.risk_id
            or row["accepted_risk_status"] != padding_config.risk_acceptance_status
        ):
            raise DataValidationError(
                f"accepted risk traceability mismatch at output row {output_index}"
            )
    if any(len(splits) > 1 for splits in processed_hash_splits.values()):
        raise DataValidationError("processed profile hashes overlap across splits")
    if len(derived_seed_values) != len(finalized_rows):
        raise DataValidationError("derived padding seeds are not unique per sample")
    del loaded

    profile_file_hash = file_sha256(profile_path)
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    (destination / f"{padding_config.output_profile_file}.sha256").write_text(
        f"{profile_file_hash}  {padding_config.output_profile_file}\n",
        encoding="utf-8",
    )
    (destination / f"{padding_config.output_manifest_file}.sha256").write_text(
        f"{manifest_hash}  {padding_config.output_manifest_file}\n",
        encoding="utf-8",
    )

    resolved_config = copy.deepcopy(dict(padding_config.raw_mapping))
    resolved_config["_resolved"] = {
        "input_manifest_sha256": padding_config.input_manifest_sha256,
        "processed_profile_file_sha256": profile_file_hash,
        "processed_manifest_sha256": manifest_hash,
        "processed_shape": list(shape),
        "processed_dtype": "float64",
        "experiment_readiness": audit["experiment_readiness"],
    }
    resolved_config_bytes = yaml.safe_dump(
        resolved_config,
        allow_unicode=True,
        sort_keys=False,
    ).encode("utf-8")
    resolved_config_path = destination / "resolved_preprocessing.yaml"
    resolved_config_path.write_bytes(resolved_config_bytes)
    resolved_config_hash = hashlib.sha256(resolved_config_bytes).hexdigest()

    bundle_payload = "\0".join(
        [
            padding_config.preprocessing_id,
            padding_config.input_manifest_sha256,
            profile_file_hash,
            manifest_hash,
            resolved_config_hash,
        ]
    ).encode("utf-8")
    bundle_hash = hashlib.sha256(bundle_payload).hexdigest()
    (destination / "bundle.sha256").write_text(
        f"{bundle_hash}  {padding_config.preprocessing_id}\n",
        encoding="utf-8",
    )

    generated_at = datetime.now(timezone.utc).isoformat()
    materialization_report = {
        "generated_at_utc": generated_at,
        "raw_root": str(root),
        "output_dir": str(destination.resolve()),
        "input_manifest_sha256": padding_config.input_manifest_sha256,
        "processed_profile_file": padding_config.output_profile_file,
        "processed_profile_file_sha256": profile_file_hash,
        "processed_manifest_file": padding_config.output_manifest_file,
        "processed_manifest_sha256": manifest_hash,
        "resolved_preprocessing_sha256": resolved_config_hash,
        "bundle_sha256": bundle_hash,
        "shape": list(shape),
        "dtype": "float64",
        "validation": {
            "status": "passed",
            "row_count_matches_input_manifest": "passed",
            "profile_shape_3600_by_601": "passed",
            "profile_dtype_float64": "passed",
            "all_values_finite": "passed",
            "processed_manifest_traceability": "passed",
            "accepted_risk_recorded_per_row": "passed",
            "processed_profile_hash_readback": "passed",
            "processed_profile_hash_cross_split_leakage": "passed",
            "derived_padding_seed_uniqueness": "passed",
        },
        "padding_audit": audit,
    }
    report_path = destination / "materialization_report.json"
    report_path.write_text(
        json.dumps(materialization_report, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return {
        "validation_status": "passed",
        "experiment_readiness": audit["experiment_readiness"],
        "output_dir": str(destination.resolve()),
        "profiles": str(profile_path.resolve()),
        "profiles_sha256": profile_file_hash,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": manifest_hash,
        "bundle_sha256": bundle_hash,
        "shape": list(shape),
        "dtype": "float64",
        "accepted_risks": audit["accepted_risks"],
    }


def write_padding_audit(
    report: Mapping[str, Any],
    padding_config: PaddingConfig,
    output_path: str | Path,
) -> dict[str, Any]:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "resolved_padding_config": dict(padding_config.raw_mapping),
        "audit": dict(report),
    }
    destination.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "output": str(destination.resolve()),
        "validation_status": report["status"],
        "sample_count": report["sample_count"],
        "padded_profile_count": report["padded_profile_count"],
        "configured_mean_noise_power_db": report["configured_mean_noise_power_db"],
        "observed_mean_padding_power_db": report["observed_mean_padding_power_db"],
        "experiment_readiness": report["experiment_readiness"],
    }
