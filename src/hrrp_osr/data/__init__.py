"""P0 data ingestion, manifest generation, and validation."""

from .config import DataConfig, load_data_config
from .manifest import build_manifest_rows, write_manifest_bundle
from .padding import (
    PaddingConfig,
    load_padding_config,
    materialize_padded_dataset,
    pad_profile_db,
)
from .protocol import angle_domain_and_split, stable_class_partition
from .processed import ProcessedBundle, load_processed_bundle
from .sets import ViewSet, build_v3_evaluation_sets, select_b0_single_view

__all__ = [
    "DataConfig",
    "PaddingConfig",
    "ProcessedBundle",
    "ViewSet",
    "angle_domain_and_split",
    "build_manifest_rows",
    "build_v3_evaluation_sets",
    "load_data_config",
    "load_padding_config",
    "load_processed_bundle",
    "materialize_padded_dataset",
    "pad_profile_db",
    "select_b0_single_view",
    "stable_class_partition",
    "write_manifest_bundle",
]
