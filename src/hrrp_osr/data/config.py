from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .errors import DataConfigError


@dataclass(frozen=True)
class SourceConfig:
    root_env: str
    layout: str
    file_glob: str
    mat_variable: str
    angle_field: str
    elevation_field: str
    profile_field: str
    range_field: str
    nfreq_field: str


@dataclass(frozen=True)
class SelectionConfig:
    elevation_deg: float
    angle_start_deg: int
    angle_stop_deg: int
    angle_step_deg: int
    angle_tolerance: float


@dataclass(frozen=True)
class ExpectedConfig:
    class_count: int
    samples_per_class: int
    profile_representation: str
    profile_linearization: str


@dataclass(frozen=True)
class ClassPartitionConfig:
    method: str
    algorithm_version: str
    seed: int
    known_count: int
    unknown_count: int


@dataclass(frozen=True)
class ProtocolConfig:
    domain_count: int
    domain_width_deg: int
    train_width_deg: int
    validation_width_deg: int
    test_width_deg: int
    rotate_split_position: bool
    boundary_buffer_deg: int


@dataclass(frozen=True)
class DataConfig:
    schema_version: int
    dataset_id: str
    source: SourceConfig
    selection: SelectionConfig
    expected: ExpectedConfig
    class_partition: ClassPartitionConfig
    protocol: ProtocolConfig
    raw_mapping: Mapping[str, Any]


def _section(mapping: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = mapping.get(name)
    if not isinstance(value, Mapping):
        raise DataConfigError(f"Missing or invalid mapping section: {name}")
    return value


def load_data_config(path: str | Path) -> DataConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise DataConfigError("The data configuration must be a YAML mapping")

    source = _section(raw, "source")
    selection = _section(raw, "selection")
    expected = _section(raw, "expected")
    partition = _section(raw, "class_partition")
    protocol = _section(raw, "protocol")

    try:
        config = DataConfig(
            schema_version=int(raw["schema_version"]),
            dataset_id=str(raw["dataset_id"]),
            source=SourceConfig(
                root_env=str(source["root_env"]),
                layout=str(source["layout"]),
                file_glob=str(source["file_glob"]),
                mat_variable=str(source["mat_variable"]),
                angle_field=str(source["angle_field"]),
                elevation_field=str(source["elevation_field"]),
                profile_field=str(source["profile_field"]),
                range_field=str(source["range_field"]),
                nfreq_field=str(source["nfreq_field"]),
            ),
            selection=SelectionConfig(
                elevation_deg=float(selection["elevation_deg"]),
                angle_start_deg=int(selection["angle_start_deg"]),
                angle_stop_deg=int(selection["angle_stop_deg"]),
                angle_step_deg=int(selection["angle_step_deg"]),
                angle_tolerance=float(selection["angle_tolerance"]),
            ),
            expected=ExpectedConfig(
                class_count=int(expected["class_count"]),
                samples_per_class=int(expected["samples_per_class"]),
                profile_representation=str(expected["profile_representation"]),
                profile_linearization=str(expected["profile_linearization"]),
            ),
            class_partition=ClassPartitionConfig(
                method=str(partition["method"]),
                algorithm_version=str(partition["algorithm_version"]),
                seed=int(partition["seed"]),
                known_count=int(partition["known_count"]),
                unknown_count=int(partition["unknown_count"]),
            ),
            protocol=ProtocolConfig(
                domain_count=int(protocol["domain_count"]),
                domain_width_deg=int(protocol["domain_width_deg"]),
                train_width_deg=int(protocol["train_width_deg"]),
                validation_width_deg=int(protocol["validation_width_deg"]),
                test_width_deg=int(protocol["test_width_deg"]),
                rotate_split_position=bool(protocol["rotate_split_position"]),
                boundary_buffer_deg=int(protocol["boundary_buffer_deg"]),
            ),
            raw_mapping=raw,
        )
    except KeyError as exc:
        raise DataConfigError(f"Missing required configuration key: {exc.args[0]}") from exc

    validate_frozen_config(config)
    return config


def validate_frozen_config(config: DataConfig) -> None:
    errors: list[str] = []
    if config.schema_version != 1:
        errors.append("schema_version must be 1")
    if config.source.layout != "one_mat_file_per_class":
        errors.append("source.layout must be one_mat_file_per_class")
    if config.source.profile_field != "TrcsHH":
        errors.append("the frozen primary profile field must be TrcsHH")
    if config.selection.elevation_deg != 83.0:
        errors.append("the frozen primary elevation must be theta=83 degrees")
    if config.selection.angle_start_deg != 0 or config.selection.angle_stop_deg != 359:
        errors.append("selected angles must be exactly 0..359 degrees")
    if config.selection.angle_step_deg != 1:
        errors.append("selected angle step must be exactly 1 degree")
    if config.expected.class_count != 10:
        errors.append("the frozen first-round dataset must contain 10 classes")
    if config.expected.samples_per_class != 360:
        errors.append("each class must contain 360 selected base HRRPs")
    if config.class_partition.method != "stable_hash_rank":
        errors.append("class partition method must be stable_hash_rank")
    if config.class_partition.algorithm_version != "sha256_rank_v1":
        errors.append("class partition algorithm_version must be sha256_rank_v1")
    if config.class_partition.seed != 20260810:
        errors.append("the frozen class partition seed must be 20260810")
    if (config.class_partition.known_count, config.class_partition.unknown_count) != (7, 3):
        errors.append("the frozen open-set partition must be 7 known / 3 unknown")
    protocol = config.protocol
    if (
        protocol.domain_count,
        protocol.domain_width_deg,
        protocol.train_width_deg,
        protocol.validation_width_deg,
        protocol.test_width_deg,
    ) != (6, 60, 36, 12, 12):
        errors.append("the frozen domain/split widths must be 6 x (36/12/12) degrees")
    if protocol.rotate_split_position:
        errors.append("split position rotation is forbidden in the main protocol")
    if protocol.boundary_buffer_deg != 0:
        errors.append("boundary buffers are forbidden in the main protocol")
    if config.expected.profile_representation != "decibel":
        errors.append("the selected TrcsHH representation must remain recorded as decibel")
    if config.expected.profile_linearization != "10_log10_power":
        errors.append("the selected TrcsHH linearization must be 10_log10_power")
    preprocessing = config.raw_mapping.get("preprocessing")
    if not isinstance(preprocessing, Mapping):
        errors.append("preprocessing must be a mapping")
    else:
        if preprocessing.get("manifest_uses_raw_profiles") is not True:
            errors.append("the P0 manifest must reference raw, unmodified profiles")
        padding = preprocessing.get("padding")
        if not isinstance(padding, Mapping) or padding.get("enabled") is not False:
            errors.append("padding must remain disabled in the raw P0 manifest config")
    if errors:
        raise DataConfigError("Invalid frozen data configuration:\n- " + "\n- ".join(errors))
