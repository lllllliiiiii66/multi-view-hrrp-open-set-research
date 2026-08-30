"""AMDR reference-aligned research components."""

from .data import (
    PEAK_RELATIVE_POWER_TRANSFORM_ID,
    TwoViewPair,
    assign_odd_angle_folds,
    build_fold_pairs,
    materialize_pair_views,
    peak_relative_power_from_db,
)
from .model import (
    AMDR_ALGORITHM_VERSION,
    AMDRCheckpoint,
    AMDRFitResult,
    AMDRModelConfig,
    fit_amdr,
    knn_predict_and_score,
    load_amdr_checkpoint,
    project_views,
    save_amdr_checkpoint,
)

__all__ = [
    "AMDR_ALGORITHM_VERSION",
    "AMDRCheckpoint",
    "AMDRFitResult",
    "AMDRModelConfig",
    "PEAK_RELATIVE_POWER_TRANSFORM_ID",
    "TwoViewPair",
    "assign_odd_angle_folds",
    "build_fold_pairs",
    "fit_amdr",
    "knn_predict_and_score",
    "load_amdr_checkpoint",
    "materialize_pair_views",
    "peak_relative_power_from_db",
    "project_views",
    "save_amdr_checkpoint",
]
