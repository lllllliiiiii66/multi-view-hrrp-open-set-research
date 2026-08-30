"""AMDR reference-aligned research components."""

from .data import (
    TwoViewPair,
    assign_odd_angle_folds,
    build_fold_pairs,
    materialize_pair_views,
)
from .model import (
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
    "AMDRCheckpoint",
    "AMDRFitResult",
    "AMDRModelConfig",
    "TwoViewPair",
    "assign_odd_angle_folds",
    "build_fold_pairs",
    "fit_amdr",
    "knn_predict_and_score",
    "load_amdr_checkpoint",
    "materialize_pair_views",
    "project_views",
    "save_amdr_checkpoint",
]
