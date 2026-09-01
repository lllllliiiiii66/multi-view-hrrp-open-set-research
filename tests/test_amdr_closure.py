from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hrrp_osr.amdr.closure import (
    AUXILIARY,
    PRIMARY,
    REJECT,
    _prepare_fold,
    decide_backbone_status,
    load_closure_config,
    select_global_candidate,
)
from hrrp_osr.data.errors import DataValidationError
from hrrp_osr.data.processed import ProcessedBundle


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _candidate_rows(
    *, converged: bool = True, alpha: tuple[float, float] = (0.5, 0.5)
) -> list[dict[str, object]]:
    rows = []
    for fold in (1, 2, 3):
        for lambda_manifold, accuracy in ((10.0, 0.7), (100.0, 0.8), (1000.0, 0.8)):
            rows.append(
                {
                    "fold_index": fold,
                    "lambda_manifold": lambda_manifold,
                    "lambda_sparse": 1.0 / 3500.0,
                    "calibration_accuracy": accuracy,
                    "calibration_macro_f1": 0.75,
                    "converged": converged,
                    "alpha": list(alpha),
                }
            )
    return rows


def test_closure_config_freezes_known_only_protocol() -> None:
    config = load_closure_config(
        PROJECT_ROOT / "configs/amdr/p0_closure_known_only_v1.yaml"
    )
    assert config["protocol"]["selection_folds"] == [1, 2, 3]
    assert config["protocol"]["confirmation_fold"] == 4
    assert config["sampling"]["included_splits"] == ["train", "calibration"]
    assert config["selection"]["require_converged"] is True
    assert config["evidence_scope"]["test_features_materialized"] is False


def test_three_fold_aggregate_selection_is_deterministic() -> None:
    selected, aggregate = select_global_candidate(_candidate_rows())
    assert len(aggregate) == 3
    assert selected["lambda_manifold"] == 100.0
    repeated, _ = select_global_candidate(list(reversed(_candidate_rows())))
    assert repeated == selected


def test_three_fold_aggregate_fails_if_every_candidate_is_nonconverged() -> None:
    with pytest.raises(DataValidationError, match="no P0 closure candidate"):
        select_global_candidate(_candidate_rows(converged=False))


def test_three_fold_aggregate_rejects_alpha_collapse() -> None:
    with pytest.raises(DataValidationError, match="no P0 closure candidate"):
        select_global_candidate(_candidate_rows(alpha=(0.049, 0.951)))


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {
                "converged": True,
                "alpha": [0.5, 0.5],
                "amdr_accuracy": 0.82,
                "amdr_macro_f1": 0.80,
                "raw_accuracy": 0.80,
                "raw_macro_f1": 0.80,
            },
            PRIMARY,
        ),
        (
            {
                "converged": True,
                "alpha": [0.5, 0.5],
                "amdr_accuracy": 0.79,
                "amdr_macro_f1": 0.79,
                "raw_accuracy": 0.80,
                "raw_macro_f1": 0.80,
            },
            AUXILIARY,
        ),
        (
            {
                "converged": True,
                "alpha": [0.049, 0.951],
                "amdr_accuracy": 0.9,
                "amdr_macro_f1": 0.9,
                "raw_accuracy": 0.8,
                "raw_macro_f1": 0.8,
            },
            REJECT,
        ),
        (
            {
                "converged": True,
                "alpha": [0.5, 0.5],
                "amdr_accuracy": 0.78,
                "amdr_macro_f1": 0.80,
                "raw_accuracy": 0.80,
                "raw_macro_f1": 0.80,
            },
            REJECT,
        ),
    ],
)
def test_fold4_backbone_decision_rule(
    kwargs: dict[str, object], expected: str
) -> None:
    assert decide_backbone_status(**kwargs)["decision"] == expected


def _synthetic_bundle() -> ProcessedBundle:
    rows = []
    profiles = np.zeros((3600, 601), dtype=np.float64)
    row_index = 0
    for class_index in range(10):
        class_name = f"class-{class_index}"
        class_role = "known" if class_index < 7 else "unknown"
        for angle in range(360):
            profiles[row_index] = class_index + angle / 1000.0
            rows.append(
                {
                    "sample_id": f"{class_name}-{angle}",
                    "class_name": class_name,
                    "class_role": class_role,
                    "angle_deg": angle,
                    "processed_row_index": row_index,
                }
            )
            row_index += 1
    return ProcessedBundle(
        root=Path("/synthetic"),
        profiles=profiles,
        rows=tuple(rows),
        profiles_sha256="a" * 64,
        manifest_sha256="b" * 64,
        bundle_sha256="c" * 64,
    )


def test_closure_fold_uses_one_known_only_manifest_for_raw_and_amdr(
    tmp_path: Path,
) -> None:
    config = load_closure_config(
        PROJECT_ROOT / "configs/amdr/p0_closure_known_only_v1.yaml"
    )
    config["sampling"]["pairs_per_class"] = {"train": 5, "calibration": 5}
    fold = _prepare_fold(_synthetic_bundle(), config, 4, tmp_path / "fold_4")
    assert {pair.split for pair in fold["pairs"]} == {"train", "calibration"}
    assert all(pair.class_role == "known" for pair in fold["pairs"])
    assert fold["split_views"]["train"][0].shape == (35, 601)
    raw_train = np.concatenate(fold["split_views"]["train"], axis=1)
    assert raw_train.shape == (35, 1202)
    assert fold["split_labels"]["train"].shape == (35,)
    assert not any(path.name.startswith("test") for path in tmp_path.rglob("*"))

