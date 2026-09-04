from __future__ import annotations

"""Read-only mechanism audit for the sealed decoupled-CSSR pilot.

This module deliberately does not import or call either legacy training entry
point.  The old pilot is an immutable input: its phase seal and every artifact
hash are verified before a checkpoint is loaded, and all new files are written
to a disjoint staging directory only after the analysis has completed.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from scipy.stats import rankdata

from hrrp_osr.data.errors import DataValidationError
from hrrp_osr.data.manifest import file_sha256
from hrrp_osr.evaluation.metrics import evaluate_open_set
from hrrp_osr.models.cssr_decoupled_1d import (
    D1_DECOUPLED_REL_CSSR,
    D2_DECOUPLED_ABSREL_CSSR,
    FGMVCSSRDecoupled1D,
)
from hrrp_osr.training.fg_mv_cssr_pilot import cssr_conformal_p_values


EXPERIMENT_ID = "cssr_identity_failure_mechanism_v1"
LEGACY_EXPERIMENT_ID = "fg_mv_cssr_decoupled_audit_v3"
LEGACY_CONFIG_SHA256 = (
    "b67f84dda0754b9b628ce046beb1b02bc8d7e15e0764bb03889bc6865ece5f7c"
)
LEGACY_PILOT_ARTIFACT_MANIFEST_SHA256 = (
    "881e8c598c4ab9a6e53016a936009f73a23a969c155b91aa3ac12a88c64c237d"
)
LEGACY_PILOT_PHASE_SUMMARY_SHA256 = (
    "bb35c8a22decd2e68d1601faeee9181583ab053ea9455838d522a11144d99f5f"
)
LEGACY_PILOT_GATE_SHA256 = (
    "b73daaedf419cf8704f3feeef99efe94293a0e4115799d79eed7f83ba4d92cc2"
)
LEGACY_PILOT_SUCCESS_SHA256 = (
    "e34f465a6881de68ae6e7c6c314c86b6b76f49cf8a9cad2ed166b2ac4c170c1c"
)
PAIR_IDS = ("N1", "N4", "N2")
METHODS = (D1_DECOUPLED_REL_CSSR, D2_DECOUPLED_ABSREL_CSSR)
LEGACY_SEED = 20260905
SCORE_NORM_SEED = 20260906
EPSILON = 1.0e-8
GEOMETRY_EPSILON = 1.0e-12
OUTPUT_FILENAMES = (
    "per_base_score_decomposition.npz",
    "per_pair_view_decomposition.csv",
    "ae_cross_reconstruction_raw.csv",
    "ae_cross_reconstruction_normalized.csv",
    "ae_specificity.json",
    "reference_distribution_summary.csv",
    "representation_geometry_z_vs_u.json",
    "official_scores_on_d1_d2.csv",
    "mechanism_audit.json",
    "artifact_hashes.json",
)
OFFICIAL_SCORE_CSV_FIELDS = (
    "pair_id",
    "evaluation_pair_id",
    "method",
    "evaluation_role",
    "class_name",
    "surrogate_identity",
    "true_label",
    "pcssr_pair_predicted_label",
    "view1_sample_id",
    "view2_sample_id",
    "view1_raw_s1",
    "view1_raw_s2",
    "view1_raw_s3",
    "view2_raw_s1",
    "view2_raw_s2",
    "view2_raw_s3",
    "view1_standardized_s1",
    "view1_standardized_s2",
    "view1_standardized_s3",
    "view2_standardized_s1",
    "view2_standardized_s2",
    "view2_standardized_s3",
    "pair_standardized_s1",
    "pair_standardized_s2",
    "pair_standardized_s3",
    "unknown_score_s1",
    "unknown_score_s2",
    "unknown_score_s3",
    "unknown_score_full",
    "performance_gate_eligible",
    "stage_b_selection_used",
    "final_unknown_used",
    "even_angle_test_used",
)

EXPECTED_CHECKPOINT_SHA256 = {
    ("N1", D1_DECOUPLED_REL_CSSR): (
        "69376ce7bbb6b57358d216102cfb3e5533932bea129d760d1838066fec800af1"
    ),
    ("N1", D2_DECOUPLED_ABSREL_CSSR): (
        "a8642edf1de6b2dbf8a53b3c4f16dffe4a97995a7dc22ccae58e191a42ee44cc"
    ),
    ("N4", D1_DECOUPLED_REL_CSSR): (
        "b0a7a779cc007ce7b830de8958b8b100c83f446c184c7355596b7aab79dccac4"
    ),
    ("N4", D2_DECOUPLED_ABSREL_CSSR): (
        "c273c6378ee7ef8c963c734afd5464838fa2fb61bc39760c26acec76e78c2e7b"
    ),
    ("N2", D1_DECOUPLED_REL_CSSR): (
        "c34e125c23bcbc98c0e2768c3f265e29eab829ac49db86dc60a54fab5ca77992"
    ),
    ("N2", D2_DECOUPLED_ABSREL_CSSR): (
        "cbad9d174604c74f205e599cf51bd36a235633fb096d48cb713ade5040b41fb5"
    ),
}
LEGACY_SOURCE_SHA256 = {
    "src/hrrp_osr/models/hrrp_ms_resnet.py": (
        "dd55eac0cbb477765e134a088d9be6ed4dbceaebd60359b3d2c51341bc59e8f0"
    ),
    "src/hrrp_osr/models/cssr_1d.py": (
        "da801f37c4c3c832f6cb2968cfe478711adaff56588c09c497d07b5f41872c8d"
    ),
    "src/hrrp_osr/models/cssr_decoupled_1d.py": (
        "e9f0f4be0210eed8d6477b9527c2314ccae43381613c56ab479ecf921c0a18f8"
    ),
    "src/hrrp_osr/training/fg_mv_cssr_decoupled.py": (
        "8c3fdce2852131bd78502df88cd4bea1b1a7ae6cfd69c13ff8375f154fa28205"
    ),
}


@dataclass(frozen=True)
class ScoreDecomposition:
    adapted_features: np.ndarray
    logits: np.ndarray
    probabilities: np.ndarray
    raw_error: np.ndarray
    activation_magnitude: np.ndarray
    normalized_error: np.ndarray
    p_value: np.ndarray
    anomaly: np.ndarray
    references: tuple[np.ndarray, ...]
    reference_ids: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class VerifiedLegacyUnit:
    pair_id: str
    method: str
    root: Path
    model: FGMVCSSRDecoupled1D
    unique_rows: tuple[dict[str, str], ...]
    evaluation_rows: tuple[dict[str, str], ...]
    prediction_rows: tuple[dict[str, str], ...]
    d0_prediction_rows: tuple[dict[str, str], ...]
    z: np.ndarray
    decomposition: ScoreDecomposition
    checkpoint_sha256: str
    artifact_manifest_sha256: str


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"cannot read JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise DataValidationError(f"JSON artifact is not an object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except OSError as exc:
        raise DataValidationError(f"cannot read CSV artifact: {path}") from exc


def _render_csv(
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> bytes:
    if not rows and fieldnames is None:
        raise DataValidationError("cannot render an empty CSV artifact without a schema")
    fields = list(fieldnames) if fieldnames is not None else list(rows[0])
    if not fields or len(fields) != len(set(fields)):
        raise DataValidationError("CSV schema is empty or repeats a field")
    if any(list(row) != fields for row in rows):
        raise DataValidationError("CSV rows do not share one stable schema")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise DataValidationError(f"stale temporary output exists: {temporary}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    _write_bytes(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> None:
    _write_bytes(path, _render_csv(rows, fieldnames=fieldnames))


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    _write_bytes(path, buffer.getvalue())


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _tree_artifact_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name
        not in {"artifact_hashes.json", "_SUCCESS.json", "_PHASE_SUCCESS.json"}
    }


def _tree_snapshot(root: Path) -> dict[str, tuple[int, int]]:
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def require_disjoint_output(input_roots: Sequence[str | Path], output_root: str | Path) -> Path:
    """Reject any output that could overwrite an immutable input tree."""

    destination = Path(output_root).resolve(strict=False)
    for raw in input_roots:
        source = Path(raw).resolve(strict=True)
        if _is_within(destination, source) or _is_within(source, destination):
            raise DataValidationError(
                f"mechanism output must be disjoint from immutable input: {source}"
            )
    return destination


def _stage_a_source_record(device: torch.device) -> dict[str, Any]:
    """Bind Stage A to the same committed source closure as Stage B."""

    from hrrp_osr.training.official_cssr_hrrp_pilot import (
        _git_commit_source_hashes,
        _git_environment,
        _json_sha256,
        _task_source_hashes,
    )

    project_root = Path(__file__).resolve().parents[3]
    source_hashes = _task_source_hashes(project_root)
    environment = _git_environment(project_root, device)
    if environment["git_status_porcelain"] != "":
        raise DataValidationError("formal Stage-A audit requires a clean checkout")
    commit = str(environment["git_commit"])
    if _git_commit_source_hashes(project_root, commit, source_hashes) != source_hashes:
        raise DataValidationError("Stage-A source bytes differ from the recorded commit")
    return {
        "status": "passed",
        "git_commit": commit,
        "git_branch": str(environment["git_branch"]),
        "task_source_hashes": source_hashes,
        "task_source_hashes_sha256": _json_sha256(source_hashes),
        "environment": environment,
    }


def _verify_stage_a_source_record(record: Any) -> None:
    from hrrp_osr.training.official_cssr_hrrp_pilot import (
        _git_commit_source_hashes,
        _json_sha256,
    )

    if not isinstance(record, Mapping) or record.get("status") != "passed":
        raise DataValidationError("Stage-A source binding is absent")
    commit = str(record.get("git_commit", ""))
    source_hashes = record.get("task_source_hashes")
    environment = record.get("environment")
    if (
        not isinstance(source_hashes, Mapping)
        or not isinstance(environment, Mapping)
        or environment.get("git_commit") != commit
        or environment.get("git_status_porcelain") != ""
        or record.get("task_source_hashes_sha256") != _json_sha256(source_hashes)
    ):
        raise DataValidationError("Stage-A source binding contract changed")
    project_root = Path(__file__).resolve().parents[3]
    if _git_commit_source_hashes(project_root, commit, source_hashes) != dict(
        source_hashes
    ):
        raise DataValidationError("recorded Stage-A commit does not reproduce source")


def _current_stage_a_runtime_contract(device: torch.device) -> dict[str, Any]:
    return {
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "global_seed": LEGACY_SEED,
        "amp": False,
    }


def _validate_stage_a_runtime_contract(record: Mapping[str, Any]) -> None:
    expected = {
        "cuda_device_name": "NVIDIA GeForce RTX 4090",
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "cudnn_benchmark": False,
        "matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "cublas_workspace_config": ":4096:8",
        "global_seed": LEGACY_SEED,
        "amp": False,
    }
    if not str(record.get("device", "")).startswith("cuda") or any(
        record.get(name) != value for name, value in expected.items()
    ):
        raise DataValidationError("formal Stage-A CUDA runtime contract changed")


def verify_sealed_pilot_root(
    pilot_root: str | Path,
    *,
    expected_manifest_sha256: str = LEGACY_PILOT_ARTIFACT_MANIFEST_SHA256,
    expected_summary_sha256: str = LEGACY_PILOT_PHASE_SUMMARY_SHA256,
    expected_gate_sha256: str = LEGACY_PILOT_GATE_SHA256,
    expected_success_sha256: str = LEGACY_PILOT_SUCCESS_SHA256,
) -> dict[str, Any]:
    """Verify the externally anchored phase seal and every legacy artifact."""

    root = Path(pilot_root).resolve(strict=True)
    if not root.is_dir():
        raise DataValidationError("legacy pilot root is not a directory")
    required = {
        "_PHASE_SUCCESS.json": expected_success_sha256,
        "artifact_hashes.json": expected_manifest_sha256,
        "phase_summary.json": expected_summary_sha256,
        "pilot_gate.json": expected_gate_sha256,
    }
    for name, expected in required.items():
        path = root / name
        if not path.is_file() or file_sha256(path) != expected:
            raise DataValidationError(f"sealed pilot anchor changed: {name}")
    marker = _read_json(root / "_PHASE_SUCCESS.json")
    if (
        marker.get("status") != "complete"
        or marker.get("artifact_hashes_sha256") != expected_manifest_sha256
        or marker.get("phase_summary_sha256") != expected_summary_sha256
    ):
        raise DataValidationError("sealed pilot phase marker is invalid")
    recorded = _read_json(root / "artifact_hashes.json")
    observed = _tree_artifact_hashes(root)
    if recorded != observed:
        raise DataValidationError("sealed pilot artifact manifest does not reproduce")
    summary = _read_json(root / "phase_summary.json")
    gate = _read_json(root / "pilot_gate.json")
    if (
        summary.get("experiment_id") != LEGACY_EXPERIMENT_ID
        or summary.get("phase") != "pilot"
        or summary.get("unit_count") != 6
        or summary.get("pair_ids") != list(PAIR_IDS)
        or summary.get("config_sha256") != LEGACY_CONFIG_SHA256
        or summary.get("decision") != "decoupled_cssr_failed"
        or summary.get("final_unknown_used") is not False
        or summary.get("even_angle_test_used") is not False
        or summary.get("final_unknown_test_authorized") is not False
        or summary.get("automatic_followon_authorized") is not False
        or summary.get("gate") != gate
        or gate.get("signal") != "decoupled_cssr_failed"
        or gate.get("selected_method") is not None
        or gate.get("confirmation_allowed") is not False
    ):
        raise DataValidationError("sealed pilot summary or gate changed")
    return {
        "status": "passed",
        "root": str(root),
        "artifact_count": len(recorded),
        "artifact_manifest_sha256": expected_manifest_sha256,
        "phase_summary_sha256": expected_summary_sha256,
        "pilot_gate_sha256": expected_gate_sha256,
        "phase_success_sha256": expected_success_sha256,
        "decision": "decoupled_cssr_failed",
        "selected_method": None,
        "confirmation_allowed": False,
        "final_unknown_test_authorized": False,
    }


def derive_seed(material: str) -> int:
    if not material or "|" not in material:
        raise DataValidationError("seed material must be a nonempty pipe-delimited string")
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def build_score_normalization_augmentations(
    normalized_inputs: np.ndarray,
    sample_ids: Sequence[str],
    *,
    pair_id: str,
    variants: int = 4,
    seed: int = SCORE_NORM_SEED,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Create the preregistered input-level gain/noise augmentation exactly."""

    values = np.asarray(normalized_inputs, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 601 or not np.isfinite(values).all():
        raise DataValidationError("normalized HRRP inputs must be finite [N,601]")
    if len(sample_ids) != values.shape[0] or len(set(map(str, sample_ids))) != len(sample_ids):
        raise DataValidationError("augmentation sample IDs must be unique and aligned")
    if pair_id not in PAIR_IDS or variants != 4 or seed != SCORE_NORM_SEED:
        raise DataValidationError("score-normalization augmentation protocol changed")
    augmented: list[np.ndarray] = []
    gains: list[float] = []
    noises: list[np.ndarray] = []
    output_ids: list[str] = []
    for sample_index, sample_id in enumerate(map(str, sample_ids)):
        for variant in range(1, variants + 1):
            prefix = (
                f"cssr_identity_mechanism_score_norm_v1|{seed}|{pair_id}|"
                f"fold_0|{sample_id}|{variant}"
            )
            gain = float(np.random.Generator(np.random.PCG64(derive_seed(prefix + "|gain"))).uniform(0.9, 1.1))
            noise = np.random.Generator(
                np.random.PCG64(derive_seed(prefix + "|noise"))
            ).normal(0.0, 0.02, size=601)
            gains.append(gain)
            noises.append(noise)
            augmented.append(gain * values[sample_index] + noise)
            output_ids.append(f"{sample_id}::v{variant}")
    gain_array = np.asarray(gains, dtype=np.float64)
    noise_array = np.asarray(noises, dtype=np.float64)
    output = np.asarray(augmented, dtype=np.float64).astype(np.float32)
    return output, {
        "status": "passed",
        "family": "gain_uniform_0.9_1.1_plus_gaussian_std_0.02",
        "seed": seed,
        "pair_id": pair_id,
        "variant_count_per_base": variants,
        "sample_variant_ids_sha256": hashlib.sha256(
            json.dumps(output_ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "gain_sha256": array_sha256(gain_array),
        "noise_sha256": array_sha256(noise_array),
        "augmented_input_sha256": array_sha256(output),
        "method_id_in_seed_material": False,
        "final_unknown_used": False,
        "even_angle_test_used": False,
    }


def _reference_indices(
    rows: Sequence[Mapping[str, Any]], num_classes: int
) -> tuple[list[np.ndarray], tuple[tuple[str, ...], ...]]:
    labels = np.asarray([int(row["model_label"]) for row in rows], dtype=np.int64)
    roles = np.asarray([str(row["experiment_role"]) for row in rows])
    references: list[np.ndarray] = []
    reference_ids: list[tuple[str, ...]] = []
    for class_index in range(num_classes):
        indices = np.flatnonzero((roles == "known_calibration") & (labels == class_index))
        if indices.size == 0:
            raise DataValidationError("a class has no known-calibration reference bases")
        references.append(indices)
        reference_ids.append(tuple(str(rows[int(index)]["sample_id"]) for index in indices))
    return references, tuple(reference_ids)


def conformal_p_and_anomaly(
    normalized_error: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    *,
    epsilon: float = EPSILON,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...], tuple[tuple[str, ...], ...]]:
    values = np.asarray(normalized_error, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(rows) or not np.isfinite(values).all():
        raise DataValidationError("normalized reconstruction errors are invalid")
    sample_ids = tuple(str(row["sample_id"]) for row in rows)
    if len(sample_ids) != len(set(sample_ids)):
        raise DataValidationError("base-level score population repeats a sample ID")
    ref_indices, reference_ids = _reference_indices(rows, values.shape[1])
    references = tuple(
        np.asarray(values[indices, class_index], dtype=np.float64)
        for class_index, indices in enumerate(ref_indices)
    )
    roles = np.asarray([str(row["experiment_role"]) for row in rows])
    labels = np.asarray([int(row["model_label"]) for row in rows], dtype=np.int64)
    p = np.empty_like(values)
    calibration = np.flatnonzero(roles == "known_calibration")
    non_calibration = np.flatnonzero(roles != "known_calibration")
    if calibration.size:
        p[calibration] = cssr_conformal_p_values(
            values[calibration],
            references,
            sample_ids=[sample_ids[int(index)] for index in calibration],
            reference_sample_ids=reference_ids,
            true_labels=labels[calibration],
            leave_one_base_sample_out=True,
        )
    if non_calibration.size:
        p[non_calibration] = cssr_conformal_p_values(
            values[non_calibration], references
        )
    anomaly = -np.log(p + float(epsilon))
    if not np.isfinite(anomaly).all():
        raise DataValidationError("conformal anomaly contains NaN or Inf")
    return p, anomaly, references, reference_ids


def decompose_scores_from_arrays(
    adapted_features: np.ndarray,
    reconstructions: np.ndarray,
    logits: np.ndarray,
    probabilities: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    *,
    epsilon: float = EPSILON,
) -> ScoreDecomposition:
    """Compute the frozen E/M/r/p/a definitions from inspectable tensors."""

    u = np.asarray(adapted_features)
    recon = np.asarray(reconstructions)
    q = np.asarray(logits)
    probs = np.asarray(probabilities)
    if u.ndim != 3 or not np.isfinite(u).all():
        raise DataValidationError("adapted features must be finite [N,C,L]")
    if recon.ndim != 4 or recon.shape[0] != u.shape[0] or recon.shape[2:] != u.shape[1:]:
        raise DataValidationError("reconstructions must be [N,K,C,L]")
    if q.shape != (u.shape[0], recon.shape[1], u.shape[2]):
        raise DataValidationError("CSSR logits do not align with reconstructions")
    if probs.shape != (u.shape[0], recon.shape[1]):
        raise DataValidationError("CSSR probabilities do not align")
    if not np.isfinite(recon).all() or not np.isfinite(q).all() or not np.isfinite(probs).all():
        raise DataValidationError("CSSR decomposition contains NaN or Inf")
    raw_error = np.abs(recon - u[:, None]).mean(axis=(2, 3))
    magnitude = np.abs(u).mean(axis=(1, 2))
    normalized = raw_error / (magnitude[:, None] + float(epsilon))
    p, anomaly, references, reference_ids = conformal_p_and_anomaly(
        normalized, rows, epsilon=epsilon
    )
    return ScoreDecomposition(
        adapted_features=u,
        logits=q,
        probabilities=probs,
        raw_error=raw_error,
        activation_magnitude=magnitude,
        normalized_error=normalized,
        p_value=p,
        anomaly=anomaly,
        references=references,
        reference_ids=reference_ids,
    )


def _infer_and_decompose(
    model: FGMVCSSRDecoupled1D,
    features: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int = 128,
) -> tuple[ScoreDecomposition, np.ndarray]:
    values = torch.from_numpy(np.asarray(features, dtype=np.float32))
    collected: dict[str, list[np.ndarray]] = {
        "u": [],
        "recon": [],
        "logits": [],
        "probabilities": [],
        "model_r": [],
    }
    model.requires_grad_(False).eval()
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise DataValidationError("legacy CSSR checkpoint is not frozen in eval mode")
    with torch.no_grad():
        for start in range(0, values.shape[0], batch_size):
            output = model(values[start : start + batch_size].to(device))
            collected["u"].append(output.adapted_features.detach().cpu().numpy())
            collected["recon"].append(output.reconstructions.detach().cpu().numpy())
            collected["logits"].append(output.logits.detach().cpu().numpy())
            collected["probabilities"].append(output.probabilities.detach().cpu().numpy())
            collected["model_r"].append(
                output.normalized_reconstruction_errors.detach().cpu().numpy()
            )
    arrays = {name: np.concatenate(parts, axis=0) for name, parts in collected.items()}
    decomposition = decompose_scores_from_arrays(
        arrays["u"],
        arrays["recon"],
        arrays["logits"],
        arrays["probabilities"],
        rows,
    )
    if not np.allclose(
        decomposition.normalized_error,
        arrays["model_r"],
        rtol=1.0e-6,
        atol=1.0e-7,
    ):
        raise DataValidationError("E/M reconstruction does not match legacy model r")
    # The sealed pilot persisted the model's float32 ``r`` tensor and then
    # promoted it to float64 for conformal scoring.  Recomputing E/M with
    # NumPy is an independent semantic check, but its reduction order can
    # differ by a few ulps.  Use the replay-exact model tensor for every
    # comparison with legacy artifacts and rebuild p/a from that exact value.
    replay_r = np.asarray(arrays["model_r"], dtype=np.float64)
    p_value, anomaly, references, reference_ids = conformal_p_and_anomaly(
        replay_r, rows
    )
    replay_decomposition = ScoreDecomposition(
        adapted_features=decomposition.adapted_features,
        logits=decomposition.logits,
        probabilities=decomposition.probabilities,
        raw_error=decomposition.raw_error,
        activation_magnitude=decomposition.activation_magnitude,
        normalized_error=replay_r,
        p_value=p_value,
        anomaly=anomaly,
        references=references,
        reference_ids=reference_ids,
    )
    return replay_decomposition, arrays["model_r"]


def validate_unique_base_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_role_counts: Mapping[str, int] | None = None,
) -> None:
    expected = (
        {"train_known": 720, "known_calibration": 180, "surrogate_unknown": 72}
        if expected_role_counts is None
        else {str(key): int(value) for key, value in expected_role_counts.items()}
    )
    required = {
        "experiment_role",
        "sample_id",
        "processed_row_index",
        "class_name",
        "model_label",
        "angle_deg",
        "frame_id",
        "source_class_role",
    }
    if not rows or any(not required.issubset(row) for row in rows):
        raise DataValidationError("unique-base manifest schema changed")
    ids = [str(row["sample_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise DataValidationError("unique-base manifest repeats a sample ID")
    observed: dict[str, int] = {}
    role_ids: dict[str, set[str]] = {}
    processed_indices: list[int] = []
    for row in rows:
        role = str(row["experiment_role"])
        observed[role] = observed.get(role, 0) + 1
        role_ids.setdefault(role, set()).add(str(row["sample_id"]))
        processed_indices.append(int(row["processed_row_index"]))
        if int(row["angle_deg"]) % 2 == 0:
            raise DataValidationError("unique-base manifest contains an even-angle sample")
    if observed != expected:
        raise DataValidationError(
            f"unique-base role counts changed: expected {expected}, observed {observed}"
        )
    roles = tuple(role_ids)
    for left_index, left in enumerate(roles):
        for right in roles[left_index + 1 :]:
            if role_ids[left] & role_ids[right]:
                raise DataValidationError("a base sample leaks across experiment roles")
    if len(processed_indices) != len(set(processed_indices)):
        raise DataValidationError("unique-base manifest repeats a processed source row")
    for role in ("train_known", "known_calibration"):
        labels = Counter(
            int(row["model_label"])
            for row in rows
            if str(row["experiment_role"]) == role
        )
        if set(labels) != set(range(5)) or len(set(labels.values())) != 1:
            raise DataValidationError(f"{role} is not balanced across five known classes")
    surrogate = [
        row for row in rows if str(row["experiment_role"]) == "surrogate_unknown"
    ]
    surrogate_identities = Counter(str(row["class_name"]) for row in surrogate)
    if (
        any(int(row["model_label"]) != 5 for row in surrogate)
        or len(surrogate_identities) != 2
        or len(set(surrogate_identities.values())) != 1
    ):
        raise DataValidationError(
            "surrogate bases are not two balanced identities with sentinel label 5"
        )


def _json_array(row: Mapping[str, Any], name: str, width: int) -> np.ndarray:
    try:
        value = np.asarray(json.loads(str(row[name])), dtype=np.float64)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"invalid prediction array field: {name}") from exc
    if value.shape != (width,) or not np.isfinite(value).all():
        raise DataValidationError(f"prediction array field has wrong shape: {name}")
    return value


def verify_saved_reference_and_predictions(
    unit_root: str | Path,
    unique_rows: Sequence[Mapping[str, Any]],
    decomposition: ScoreDecomposition,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Prove that old reference arrays and row-level scores reproduce exactly."""

    root = Path(unit_root)
    with np.load(root / "reference_scores.npz", allow_pickle=False) as saved:
        arrays = {name: np.asarray(saved[name]) for name in saved.files}
    expected_arrays = {
        "r": decomposition.normalized_error.astype(np.float64),
        "known_calibration_p": decomposition.p_value[
            [str(row["experiment_role"]) == "known_calibration" for row in unique_rows]
        ],
        "known_calibration_a": decomposition.anomaly[
            [str(row["experiment_role"]) == "known_calibration" for row in unique_rows]
        ],
        "surrogate_unknown_p": decomposition.p_value[
            [str(row["experiment_role"]) == "surrogate_unknown" for row in unique_rows]
        ],
        "surrogate_unknown_a": decomposition.anomaly[
            [str(row["experiment_role"]) == "surrogate_unknown" for row in unique_rows]
        ],
    }
    for class_index, values in enumerate(decomposition.references):
        expected_arrays[f"class_{class_index}_reference_r"] = values
    for name, expected in expected_arrays.items():
        if name not in arrays or not np.array_equal(arrays[name], expected):
            raise DataValidationError(f"sealed reference score does not reproduce: {name}")

    prediction_rows = _read_csv(root / "predictions_and_scores.csv")
    d0_rows = _read_csv(root / "d0_predictions_and_scores.csv")
    evaluation_rows = _read_csv(root / "evaluation_pair_manifest.csv")
    if not prediction_rows or not (
        len(prediction_rows) == len(d0_rows) == len(evaluation_rows)
    ):
        raise DataValidationError("legacy prediction populations are empty or misaligned")
    by_id = {str(row["sample_id"]): index for index, row in enumerate(unique_rows)}
    class_count = decomposition.normalized_error.shape[1]
    for manifest, method_row, d0_row in zip(
        evaluation_rows, prediction_rows, d0_rows, strict=True
    ):
        for key in (
            "pair_id",
            "class_name",
            "view1_sample_id",
            "view2_sample_id",
            "view1_angle_deg",
            "view2_angle_deg",
            "view1_frame_id",
            "view2_frame_id",
        ):
            if str(method_row.get(key)) != str(manifest.get(key)) or str(d0_row.get(key)) != str(
                manifest.get(key)
            ):
                raise DataValidationError("legacy prediction row is not bound to its manifest")
        if method_row.get("evaluation_role") != manifest.get("evaluation_subset_role"):
            raise DataValidationError("legacy prediction role changed")
        if any(
            method_row.get(key) != d0_row.get(key)
            for key in (
                "evaluation_role",
                "true_label",
                "predicted_known_label",
                "fused_logits",
                "view1_sample_id",
                "view2_sample_id",
            )
        ):
            raise DataValidationError("D0 and CSSR prediction rows no longer share R2 outputs")
        prediction = int(method_row["predicted_known_label"])
        if prediction != int(np.argmax(_json_array(method_row, "fused_logits", class_count))):
            raise DataValidationError("legacy known prediction is not frozen R2 argmax")
        view_anomaly: list[np.ndarray] = []
        for view in (1, 2):
            sample_id = str(method_row[f"view{view}_sample_id"])
            if sample_id not in by_id:
                raise DataValidationError("prediction references an absent unique base")
            index = by_id[sample_id]
            comparisons = {
                f"view{view}_r": decomposition.normalized_error[index],
                f"view{view}_p_value": decomposition.p_value[index],
                f"view{view}_a": decomposition.anomaly[index],
            }
            for name, expected in comparisons.items():
                if not np.array_equal(_json_array(method_row, name, class_count), expected):
                    raise DataValidationError(f"legacy prediction score changed: {name}")
            view_anomaly.append(decomposition.anomaly[index])
        guided = 0.5 * (
            float(view_anomaly[0][prediction]) + float(view_anomaly[1][prediction])
        )
        if not math.isclose(
            float(method_row["unknown_score"]), guided, rel_tol=0.0, abs_tol=1.0e-15
        ):
            raise DataValidationError("legacy guided unknown score does not reproduce")
    return evaluation_rows, prediction_rows, d0_rows


def load_verified_legacy_unit(
    pilot_root: str | Path,
    *,
    pair_id: str,
    method: str,
    device: torch.device,
    expected_shape: tuple[int, int, int] = (972, 128, 76),
    expected_role_counts: Mapping[str, int] | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> VerifiedLegacyUnit:
    """Strict-load one old unit and verify its replay and saved predictions."""

    if pair_id not in PAIR_IDS or method not in METHODS:
        raise DataValidationError("legacy unit is outside the preregistered Stage-A plan")
    root = (
        Path(pilot_root).resolve(strict=True)
        / pair_id
        / "fold_0"
        / f"seed_{LEGACY_SEED}"
        / method
    )
    if not root.is_dir():
        raise DataValidationError(f"missing sealed legacy unit: {root}")
    marker = _read_json(root / "_SUCCESS.json")
    manifest_path = root / "artifact_hashes.json"
    if (
        marker.get("status") != "complete"
        or marker.get("artifact_hashes_sha256") != file_sha256(manifest_path)
        or marker.get("unit_summary_sha256") != file_sha256(root / "unit_summary.json")
        or _read_json(manifest_path) != _tree_artifact_hashes(root)
    ):
        raise DataValidationError("legacy unit success seal or artifact manifest changed")
    contract = _read_json(root / "unit_contract.json")
    contract_expected = {
        "experiment_id": LEGACY_EXPERIMENT_ID,
        "phase": "pilot",
        "mode": "full",
        "pair_id": pair_id,
        "method": method,
        "config_sha256": LEGACY_CONFIG_SHA256,
        "r2_retrained_or_finetuned": False,
        "final_unknown_used": False,
        "even_angle_test_used": False,
        "test_features_materialized": False,
    }
    for name, expected in contract_expected.items():
        if contract.get(name) != expected:
            raise DataValidationError(f"legacy unit contract changed: {name}")
    source_hashes = contract.get("source_hashes")
    if not isinstance(source_hashes, dict) or any(
        source_hashes.get(path) != expected
        for path, expected in LEGACY_SOURCE_SHA256.items()
    ):
        raise DataValidationError("legacy unit source identity changed")
    project_root = Path(__file__).resolve().parents[3]
    for relative, expected in LEGACY_SOURCE_SHA256.items():
        local_source = project_root / relative
        if not local_source.is_file() or file_sha256(local_source) != expected:
            raise DataValidationError(f"local legacy source changed: {relative}")
    checkpoint_path = root / "checkpoint.pt"
    expected_checkpoint = (
        EXPECTED_CHECKPOINT_SHA256[(pair_id, method)]
        if expected_checkpoint_sha256 is None
        else expected_checkpoint_sha256
    )
    if file_sha256(checkpoint_path) != expected_checkpoint:
        raise DataValidationError("legacy checkpoint bitwise hash changed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_metadata = {
        "experiment_id": LEGACY_EXPERIMENT_ID,
        "phase": "pilot",
        "mode": "full",
        "pair_id": pair_id,
        "method": method,
        "architecture": "fg_mv_cssr_decoupled_1d_v1",
        "checkpoint_epoch": 20,
        "formal_checkpoint": True,
        "checkpoint_selection": "fixed_final_epoch",
        "cssr_seed": LEGACY_SEED,
        "config_sha256": LEGACY_CONFIG_SHA256,
        "final_unknown_used": False,
        "even_angle_test_used": False,
    }
    for name, expected in expected_metadata.items():
        if checkpoint.get(name) != expected:
            raise DataValidationError(f"legacy checkpoint metadata changed: {name}")
    model = FGMVCSSRDecoupled1D().to(device)
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise DataValidationError("legacy checkpoint failed strict load")

    unique_rows = _read_csv(root / "unique_base_sample_manifest.csv")
    validate_unique_base_rows(unique_rows, expected_role_counts=expected_role_counts)
    if hashlib.sha256((root / "unique_base_sample_manifest.csv").read_bytes()).hexdigest() != checkpoint.get(
        "unique_base_manifest_sha256"
    ):
        raise DataValidationError("legacy unique-base manifest binding changed")
    if file_sha256(root / "source_pair_manifest.csv") != checkpoint.get(
        "source_pair_manifest_sha256"
    ):
        raise DataValidationError("legacy source-pair manifest binding changed")

    with np.load(root / "checkpoint_replay.npz", allow_pickle=False) as saved:
        required = {"unique_features", "expected_u", "expected_r", "expected_probabilities"}
        if set(saved.files) != required:
            raise DataValidationError("legacy checkpoint replay schema changed")
        z = np.asarray(saved["unique_features"], dtype=np.float32)
        expected_u = np.asarray(saved["expected_u"], dtype=np.float32)
        expected_r = np.asarray(saved["expected_r"], dtype=np.float32)
        expected_probabilities = np.asarray(saved["expected_probabilities"], dtype=np.float32)
    if z.shape != expected_shape or expected_u.shape != expected_shape:
        raise DataValidationError("legacy feature-map population shape changed")
    if expected_r.shape != (expected_shape[0], 5) or expected_probabilities.shape != (
        expected_shape[0],
        5,
    ):
        raise DataValidationError("legacy replay score shape changed")
    if array_sha256(z) != checkpoint.get("unique_feature_map_sha256"):
        raise DataValidationError("legacy replay feature-map hash changed")
    decomposition, replay_r = _infer_and_decompose(
        model, z, unique_rows, device=device, batch_size=128
    )
    exact = {
        "U": np.array_equal(decomposition.adapted_features, expected_u),
        "r": np.array_equal(replay_r, expected_r),
        "probabilities": np.array_equal(decomposition.probabilities, expected_probabilities),
    }
    if not all(exact.values()):
        raise DataValidationError(f"legacy checkpoint replay changed: {exact}")
    evaluation_rows, prediction_rows, d0_rows = verify_saved_reference_and_predictions(
        root, unique_rows, decomposition
    )
    return VerifiedLegacyUnit(
        pair_id=pair_id,
        method=method,
        root=root,
        model=model,
        unique_rows=tuple(unique_rows),
        evaluation_rows=tuple(evaluation_rows),
        prediction_rows=tuple(prediction_rows),
        d0_prediction_rows=tuple(d0_rows),
        z=z,
        decomposition=decomposition,
        checkpoint_sha256=expected_checkpoint,
        artifact_manifest_sha256=file_sha256(manifest_path),
    )


def verify_d1_d2_pair_alignment(
    units: Mapping[tuple[str, str], VerifiedLegacyUnit],
) -> dict[str, Any]:
    """Verify all method-independent sealed inputs before Stage-A statistics."""

    if set(units) != {
        (pair_id, method) for pair_id in PAIR_IDS for method in METHODS
    }:
        raise DataValidationError("Stage-A unit plan is incomplete")
    by_pair: dict[str, Any] = {}
    for pair_id in PAIR_IDS:
        left = units[(pair_id, METHODS[0])]
        right = units[(pair_id, METHODS[1])]
        checks = {
            "unique_base_manifest_exact": left.unique_rows == right.unique_rows,
            "unique_feature_map_exact": np.array_equal(left.z, right.z),
            "evaluation_pair_manifest_exact": (
                left.evaluation_rows == right.evaluation_rows
            ),
            "d0_prediction_rows_exact": (
                left.d0_prediction_rows == right.d0_prediction_rows
            ),
            "source_pair_manifest_sha256_exact": file_sha256(
                left.root / "source_pair_manifest.csv"
            )
            == file_sha256(right.root / "source_pair_manifest.csv"),
        }
        if not all(checks.values()):
            raise DataValidationError(
                f"D1/D2 method-independent sealed inputs differ for {pair_id}: {checks}"
            )
        by_pair[pair_id] = {
            **checks,
            "unique_base_count": len(left.unique_rows),
            "evaluation_pair_count": len(left.evaluation_rows),
            "unique_feature_map_sha256": array_sha256(left.z),
            "source_pair_manifest_sha256": file_sha256(
                left.root / "source_pair_manifest.csv"
            ),
        }
    return {
        "status": "passed",
        "by_pair": by_pair,
        "method_dependent_scores_compared_here": False,
    }


def _json_vector(values: np.ndarray) -> str:
    return json.dumps(np.asarray(values).tolist(), ensure_ascii=False, separators=(",", ":"))


def _reference_statistics(values: np.ndarray) -> dict[str, float | int]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise DataValidationError("reference population is empty or non-finite")
    q25, q75 = np.quantile(vector, (0.25, 0.75), method="linear")
    return {
        "count": int(vector.size),
        "mean": float(vector.mean()),
        "population_std": float(vector.std(ddof=0)),
        "median": float(np.median(vector)),
        "iqr": float(q75 - q25),
        "p90": float(np.quantile(vector, 0.90, method="linear")),
        "p95": float(np.quantile(vector, 0.95, method="linear")),
        "p99": float(np.quantile(vector, 0.99, method="linear")),
    }


def build_pair_view_decomposition(unit: VerifiedLegacyUnit) -> list[dict[str, Any]]:
    """Join unique-base decomposition to the frozen two-view evaluation order."""

    decomposition = unit.decomposition
    class_count = decomposition.normalized_error.shape[1]
    by_id = {
        str(row["sample_id"]): index for index, row in enumerate(unit.unique_rows)
    }
    reference_statistics = [
        _reference_statistics(values) for values in decomposition.references
    ]
    rows: list[dict[str, Any]] = []
    for manifest, prediction, d0 in zip(
        unit.evaluation_rows,
        unit.prediction_rows,
        unit.d0_prediction_rows,
        strict=True,
    ):
        predicted = int(prediction["predicted_known_label"])
        if not 0 <= predicted < class_count:
            raise DataValidationError("predicted class is outside the AE bank")
        indices = [by_id[str(manifest[f"view{view}_sample_id"])] for view in (1, 2)]
        class_arrays = {
            "e": decomposition.raw_error,
            "r": decomposition.normalized_error,
            "p": decomposition.p_value,
            "a": decomposition.anomaly,
        }
        row: dict[str, Any] = {
            "pair_id": unit.pair_id,
            "evaluation_pair_id": manifest["pair_id"],
            "method": unit.method,
            "evaluation_role": prediction["evaluation_role"],
            "class_name": prediction["class_name"],
            "surrogate_identity": prediction.get("surrogate_identity", ""),
            "true_label": int(prediction["true_label"]),
            "r2_predicted_label": predicted,
            "r2_predicted_class_name": prediction["predicted_known_class_name"],
            "r2_fused_logits": prediction["fused_logits"],
            "d0_class_conditional_mls": float(d0["unknown_score"]),
            "view1_sample_id": manifest["view1_sample_id"],
            "view2_sample_id": manifest["view2_sample_id"],
            "view1_angle_deg": int(manifest["view1_angle_deg"]),
            "view2_angle_deg": int(manifest["view2_angle_deg"]),
            "view1_frame_id": int(manifest["view1_frame_id"]),
            "view2_frame_id": int(manifest["view2_frame_id"]),
            **{
                f"predicted_class_reference_{name}_r": value
                for name, value in reference_statistics[predicted].items()
            },
            "final_unknown_used": False,
            "even_angle_test_used": False,
            "performance_gate_eligible": False,
        }
        for local_view, index in enumerate(indices, start=1):
            row[f"view{local_view}_m"] = float(
                decomposition.activation_magnitude[index]
            )
            for short, array in class_arrays.items():
                row[f"view{local_view}_{short}_all_classes"] = _json_vector(array[index])
                row[f"view{local_view}_{short}_predicted"] = float(
                    array[index, predicted]
                )
            row[f"view{local_view}_lowest_r_ae"] = int(
                np.argmin(decomposition.normalized_error[index])
            )
            row[f"view{local_view}_accepted_at_predicted_class_p95"] = bool(
                decomposition.normalized_error[index, predicted]
                <= float(reference_statistics[predicted]["p95"])
            )
        for short in ("e", "m", "r", "p", "a"):
            first = float(row[f"view1_{short}_predicted"] if short != "m" else row["view1_m"])
            second = float(row[f"view2_{short}_predicted"] if short != "m" else row["view2_m"])
            row[f"pair_{short}_mean"] = 0.5 * (first + second)
            row[f"pair_{short}_view1_minus_view2"] = first - second
            row[f"pair_{short}_min"] = min(first, second)
            row[f"pair_{short}_max"] = max(first, second)
        row["lowest_r_ae_same_across_views"] = (
            row["view1_lowest_r_ae"] == row["view2_lowest_r_ae"]
        )
        accepted = int(row["view1_accepted_at_predicted_class_p95"]) + int(
            row["view2_accepted_at_predicted_class_p95"]
        )
        row["view_acceptance_pattern"] = (
            "both_accept" if accepted == 2 else "one_accept" if accepted == 1 else "both_reject"
        )
        rows.append(row)
    return rows


def reference_distribution_rows(unit: VerifiedLegacyUnit) -> list[dict[str, Any]]:
    class_names: dict[int, str] = {}
    for row in unit.unique_rows:
        if row["experiment_role"] == "known_calibration":
            class_names.setdefault(int(row["model_label"]), str(row["class_name"]))
    rows: list[dict[str, Any]] = []
    for class_index, values in enumerate(unit.decomposition.references):
        vector = np.asarray(values, dtype=np.float64)
        if vector.size != 36 or not np.isfinite(vector).all():
            raise DataValidationError("legacy reference population is not 36 finite bases")
        rows.append(
            {
                "pair_id": unit.pair_id,
                "method": unit.method,
                "ae_class_index": class_index,
                "ae_class_name": class_names[class_index],
                **_reference_statistics(vector),
            }
        )
    return rows


def _identity_groups(
    rows: Sequence[Mapping[str, Any]],
) -> list[tuple[str, str, np.ndarray]]:
    ordered: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(rows):
        key = (str(row["experiment_role"]), str(row["class_name"]))
        ordered.setdefault(key, []).append(index)
    return [
        (role, class_name, np.asarray(indices, dtype=np.int64))
        for (role, class_name), indices in ordered.items()
    ]


def ae_cross_reconstruction_rows(
    unit: VerifiedLegacyUnit,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return identity-by-AE E and combined r/a tables."""

    raw: list[dict[str, Any]] = []
    normalized: list[dict[str, Any]] = []
    for role, identity, indices in _identity_groups(unit.unique_rows):
        e = unit.decomposition.raw_error[indices].astype(np.float64)
        r = unit.decomposition.normalized_error[indices].astype(np.float64)
        a = unit.decomposition.anomaly[indices].astype(np.float64)
        base = {
            "pair_id": unit.pair_id,
            "method": unit.method,
            "experiment_role": role,
            "identity": identity,
            "base_count": int(indices.size),
            "best_ae_by_mean_r": int(np.argmin(r.mean(axis=0))),
        }
        raw.append(
            {
                **base,
                **{f"mean_e_ae_{index}": float(e[:, index].mean()) for index in range(e.shape[1])},
            }
        )
        normalized.append(
            {
                **base,
                **{f"mean_r_ae_{index}": float(r[:, index].mean()) for index in range(r.shape[1])},
                **{f"mean_a_ae_{index}": float(a[:, index].mean()) for index in range(a.shape[1])},
                **{
                    f"lowest_r_share_ae_{index}": float(
                        np.mean(np.argmin(r, axis=1) == index)
                    )
                    for index in range(r.shape[1])
                },
            }
        )
    return raw, normalized


def ae_specificity_summary(unit: VerifiedLegacyUnit) -> dict[str, Any]:
    rows = unit.unique_rows
    r = unit.decomposition.normalized_error.astype(np.float64)
    labels = np.asarray([int(row["model_label"]) for row in rows], dtype=np.int64)
    roles = np.asarray([str(row["experiment_role"]) for row in rows])
    identities = np.asarray([str(row["class_name"]) for row in rows])
    best = np.argmin(r, axis=1)
    result: dict[str, Any] = {}
    for ae in range(r.shape[1]):
        own = np.flatnonzero((roles == "known_calibration") & (labels == ae))
        other_identity_values: list[float] = []
        best_shares: list[float] = []
        for identity in dict.fromkeys(identities[roles == "known_calibration"].tolist()):
            current = np.flatnonzero(
                (roles == "known_calibration") & (identities == identity) & (labels != ae)
            )
            if current.size:
                other_identity_values.append(float(np.median(r[current, ae])))
                best_shares.append(float(np.mean(best[current] == ae)))
        surrogate_identity_values: list[float] = []
        surrogate_acceptance: list[float] = []
        own_p95 = float(np.quantile(r[own, ae], 0.95, method="linear"))
        for identity in dict.fromkeys(identities[roles == "surrogate_unknown"].tolist()):
            current = np.flatnonzero(
                (roles == "surrogate_unknown") & (identities == identity)
            )
            surrogate_identity_values.append(float(np.median(r[current, ae])))
            surrogate_acceptance.append(float(np.mean(r[current, ae] <= own_p95)))
            best_shares.append(float(np.mean(best[current] == ae)))
        if own.size != 36 or not other_identity_values or not surrogate_identity_values:
            raise DataValidationError("AE specificity populations are incomplete")
        own_median = float(np.median(r[own, ae]))
        other_median = float(np.mean(other_identity_values))
        result[str(ae)] = {
            "own_known_median_r": own_median,
            "other_known_median_r": other_median,
            "surrogate_median_r": float(np.mean(surrogate_identity_values)),
            "specificity_ratio": other_median / (own_median + GEOMETRY_EPSILON),
            "open_acceptance_rate": float(np.mean(surrogate_acceptance)),
            "best_ae_share_across_all_nonown_identities": float(np.mean(best_shares)),
            "identity_equal_weighting": True,
            "own_known_reference_role": "known_calibration",
        }
    return {
        "pair_id": unit.pair_id,
        "method": unit.method,
        "by_ae": result,
        "causal_label": False,
    }


def _feature_pool_geometry(
    features: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    indices: np.ndarray,
) -> dict[str, Any]:
    values = np.asarray(features, dtype=np.float64)[indices]
    if values.ndim != 3 or values.shape[0] == 0 or not np.isfinite(values).all():
        raise DataValidationError("feature geometry population is empty or non-finite")
    observations = values.transpose(0, 2, 1).reshape(-1, values.shape[1])
    channel_mean = observations.mean(axis=0)
    centered = observations - channel_mean
    channel_variance = np.square(centered).mean(axis=0)
    eigenvalues = np.linalg.eigvalsh(centered.T @ centered)
    eigenvalues = np.maximum(eigenvalues[::-1], 0.0)
    total = float(eigenvalues.sum())
    if total <= 0.0:
        raise DataValidationError("feature population has zero centered energy")
    proportions = eigenvalues / total
    positive = proportions > 0.0
    effective_rank = float(
        np.exp(-np.sum(proportions[positive] * np.log(proportions[positive])))
    )
    singular_values = np.sqrt(eigenvalues)
    identities = [str(rows[int(index)]["class_name"]) for index in indices]
    identity_order = tuple(dict.fromkeys(identities))
    centers: dict[str, np.ndarray] = {}
    within_by_identity: dict[str, float] = {}
    for identity in identity_order:
        local = values[np.asarray([value == identity for value in identities])]
        center = local.transpose(0, 2, 1).reshape(-1, values.shape[1]).mean(axis=0)
        centers[identity] = center
        within_by_identity[identity] = float(
            np.square(local.transpose(0, 2, 1) - center).sum(axis=2).mean()
        )
    pair_distances = [
        float(np.square(centers[left] - centers[right]).sum())
        for left_index, left in enumerate(identity_order)
        for right in identity_order[left_index + 1 :]
    ]
    within = float(np.mean(list(within_by_identity.values())))
    between = float(np.mean(pair_distances)) if pair_distances else 0.0
    top_count = min(10, singular_values.size)
    return {
        "base_count": int(values.shape[0]),
        "identity_count": len(identity_order),
        "identity_order": list(identity_order),
        "position_observation_count": int(observations.shape[0]),
        "mean_base_frobenius_norm": float(
            np.linalg.norm(values, axis=(1, 2)).mean()
        ),
        "channel_population_mean": channel_mean.tolist(),
        "channel_population_variance": channel_variance.tolist(),
        "top_singular_values": singular_values[:top_count].tolist(),
        "top_singular_cumulative_explained_variance": np.cumsum(proportions)[
            :top_count
        ].tolist(),
        "entropy_effective_rank": effective_rank,
        "within_scatter_identity_equal": within,
        "between_center_squared_distance": between,
        "fisher_between_within_ratio": between / (within + GEOMETRY_EPSILON),
    }


def representation_geometry(
    z: np.ndarray,
    u: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare Z and U on the four frozen unique-base populations."""

    z_values = np.asarray(z)
    u_values = np.asarray(u)
    if z_values.shape != u_values.shape or z_values.ndim != 3:
        raise DataValidationError("Z and U must share [N,C,L]")
    if z_values.shape[0] != len(rows) or not np.isfinite(z_values).all() or not np.isfinite(
        u_values
    ).all():
        raise DataValidationError("Z/U population is not finite or manifest-aligned")
    roles = np.asarray([str(row["experiment_role"]) for row in rows])
    pools = {
        "train_known": np.flatnonzero(roles == "train_known"),
        "known_calibration": np.flatnonzero(roles == "known_calibration"),
        "surrogate": np.flatnonzero(roles == "surrogate_unknown"),
        "evaluation": np.flatnonzero(
            (roles == "known_calibration") | (roles == "surrogate_unknown")
        ),
    }
    geometry = {
        name: {
            "Z": _feature_pool_geometry(z_values, rows, indices),
            "U": _feature_pool_geometry(u_values, rows, indices),
        }
        for name, indices in pools.items()
    }
    residual = np.linalg.norm(u_values - z_values, axis=(1, 2)) / (
        np.linalg.norm(z_values, axis=(1, 2)) + GEOMETRY_EPSILON
    )
    residual_by_pool: dict[str, Any] = {}
    for pool_name, indices in pools.items():
        identity_values: dict[str, list[float]] = {}
        for index in indices:
            identity_values.setdefault(str(rows[int(index)]["class_name"]), []).append(
                float(residual[int(index)])
            )
        residual_by_pool[pool_name] = {
            "base_mean": float(residual[indices].mean()),
            "base_median": float(np.median(residual[indices])),
            "identity_equal_mean": float(
                np.mean([np.mean(values) for values in identity_values.values()])
            ),
            "by_identity": {
                identity: float(np.mean(values))
                for identity, values in identity_values.items()
            },
        }

    train = pools["train_known"]
    calibration = pools["known_calibration"]
    surrogate = pools["surrogate"]
    train_labels = np.asarray([int(rows[int(index)]["model_label"]) for index in train])
    calibration_labels = np.asarray(
        [int(rows[int(index)]["model_label"]) for index in calibration]
    )
    class_order = tuple(dict.fromkeys(train_labels.tolist()))
    if class_order != tuple(range(len(class_order))):
        raise DataValidationError("train-known model labels are not contiguous")
    center_diagnostics: dict[str, Any] = {}
    for name, features in (("Z", z_values), ("U", u_values)):
        centers = np.stack(
            [
                features[train[train_labels == class_index]]
                .transpose(0, 2, 1)
                .reshape(-1, features.shape[1])
                .mean(axis=0)
                for class_index in class_order
            ]
        )

        def base_distances(selected: np.ndarray) -> np.ndarray:
            values = features[selected].transpose(0, 2, 1)
            return np.square(values[:, :, None, :] - centers[None, None, :, :]).sum(
                axis=3
            ).mean(axis=1)

        calibration_distances = base_distances(calibration)
        surrogate_distances = base_distances(surrogate)
        surrogate_identities = np.asarray(
            [str(rows[int(index)]["class_name"]) for index in surrogate]
        )
        center_diagnostics[name] = {
            "train_center_shape": list(centers.shape),
            "known_calibration_nearest_center_accuracy": float(
                np.mean(calibration_distances.argmin(axis=1) == calibration_labels)
            ),
            "surrogate_mean_squared_distance_to_train_centers": {
                identity: surrogate_distances[surrogate_identities == identity]
                .mean(axis=0)
                .tolist()
                for identity in dict.fromkeys(surrogate_identities.tolist())
            },
        }
    return {
        "pools": geometry,
        "adapter_residual_ratio": residual_by_pool,
        "train_center_diagnostics": center_diagnostics,
        "base_weighting": "unique_sample_id_once",
        "pair_multiplicity_used": False,
    }


def _correlation(first: np.ndarray, second: np.ndarray, *, rank: bool) -> float | None:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if rank:
        left = rankdata(left, method="average")
        right = rankdata(right, method="average")
    if left.size < 2 or float(left.std(ddof=0)) == 0.0 or float(right.std(ddof=0)) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _distribution_summary(values: np.ndarray) -> dict[str, float]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise DataValidationError("diagnostic distribution is empty or non-finite")
    q25, q50, q75 = np.quantile(vector, (0.25, 0.5, 0.75), method="linear")
    return {
        "mean": float(vector.mean()),
        "population_std": float(vector.std(ddof=0)),
        "q25": float(q25),
        "median": float(q50),
        "q75": float(q75),
    }


def _three_view_open_metrics(
    known: Sequence[Mapping[str, Any]],
    unknown: Sequence[Mapping[str, Any]],
    *,
    known_true: np.ndarray,
    known_pred: np.ndarray,
    unknown_pred: np.ndarray,
    metric: str,
) -> dict[str, dict[str, float]]:
    fields = {
        "view1": f"view1_{metric}_predicted",
        "view2": f"view2_{metric}_predicted",
        "mean": f"pair_{metric}_mean",
    }
    result: dict[str, dict[str, float]] = {}
    for reduction, field in fields.items():
        known_scores = np.asarray([float(row[field]) for row in known])
        metrics = evaluate_open_set(
            known_true=known_true,
            known_pred=known_pred,
            known_unknown_scores=known_scores,
            unknown_pred=unknown_pred,
            unknown_unknown_scores=np.asarray(
                [float(row[field]) for row in unknown]
            ),
            known_validation_scores=known_scores,
            known_class_count=5,
            known_acceptance_rate=0.95,
        )
        result[reduction] = {
            key: float(metrics[key]) for key in ("auroc", "oscr", "fpr95")
        }
    return result


def identity_view_diagnostics(pair_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    known = [row for row in pair_rows if row["evaluation_role"] == "known_calibration"]
    unknown = [row for row in pair_rows if row["evaluation_role"] == "surrogate_unknown"]
    if not known or not unknown:
        raise DataValidationError("identity view diagnostics require known and surrogate pairs")
    identities = tuple(dict.fromkeys(str(row["class_name"]) for row in unknown))
    result: list[dict[str, Any]] = []
    known_true = np.asarray([int(row["true_label"]) for row in known], dtype=np.int64)
    known_pred = np.asarray([int(row["r2_predicted_label"]) for row in known], dtype=np.int64)
    for identity in identities:
        selected = [row for row in unknown if str(row["class_name"]) == identity]
        unknown_pred = np.asarray(
            [int(row["r2_predicted_label"]) for row in selected], dtype=np.int64
        )
        score_variant_metrics = {
            name: _three_view_open_metrics(
                known,
                selected,
                known_true=known_true,
                known_pred=known_pred,
                unknown_pred=unknown_pred,
                metric=metric,
            )
            for name, metric in (
                ("raw_error_e", "e"),
                ("normalized_error_r", "r"),
                ("anomaly_a", "a"),
            )
        }
        scores = score_variant_metrics["anomaly_a"]
        correlations: dict[str, Any] = {}
        distributions: dict[str, Any] = {}
        known_distributions: dict[str, Any] = {}
        for metric in ("e", "m", "r", "p", "a"):
            first_field = f"view1_{metric}_predicted" if metric != "m" else "view1_m"
            second_field = f"view2_{metric}_predicted" if metric != "m" else "view2_m"
            first = np.asarray([float(row[first_field]) for row in selected])
            second = np.asarray([float(row[second_field]) for row in selected])
            combined = np.concatenate([first, second])
            correlations[metric] = {
                "pearson": _correlation(first, second, rank=False),
                "spearman": _correlation(first, second, rank=True),
            }
            distributions[metric] = _distribution_summary(combined)
            known_distributions[metric] = _distribution_summary(
                np.asarray(
                    [float(row[first_field]) for row in known]
                    + [float(row[second_field]) for row in known]
                )
            )
        patterns = {
            name: float(np.mean([row["view_acceptance_pattern"] == name for row in selected]))
            for name in ("both_accept", "one_accept", "both_reject")
        }
        reference_names = (
            "count",
            "mean",
            "population_std",
            "median",
            "iqr",
            "p90",
            "p95",
            "p99",
        )
        predicted_reference_by_class: dict[str, Any] = {}
        for predicted in dict.fromkeys(
            int(row["r2_predicted_label"]) for row in selected
        ):
            current = [
                row for row in selected if int(row["r2_predicted_label"]) == predicted
            ]
            reference = {
                name: current[0][f"predicted_class_reference_{name}_r"]
                for name in reference_names
            }
            if any(
                any(
                    row[f"predicted_class_reference_{name}_r"] != value
                    for row in current[1:]
                )
                for name, value in reference.items()
            ):
                raise DataValidationError("predicted-class reference summary changed within class")
            predicted_reference_by_class[str(predicted)] = {
                "class_name": str(current[0]["r2_predicted_class_name"]),
                "pair_count": len(current),
                "pair_fraction": len(current) / len(selected),
                **reference,
            }
        weighted_reference = {
            name: float(
                np.mean(
                    [
                        float(row[f"predicted_class_reference_{name}_r"])
                        for row in selected
                    ]
                )
            )
            for name in reference_names
        }
        pair_e = np.asarray([float(row["pair_e_mean"]) for row in selected])
        pair_m = np.asarray([float(row["pair_m_mean"]) for row in selected])
        pair_r = np.asarray([float(row["pair_r_mean"]) for row in selected])
        a_aurocs = {name: scores[name]["auroc"] for name in ("view1", "view2", "mean")}
        result.append(
            {
                "pair_id": str(selected[0]["pair_id"]),
                "method": str(selected[0]["method"]),
                "identity": identity,
                "pair_count": len(selected),
                "a_score_view_metrics": scores,
                "score_variant_view_metrics": score_variant_metrics,
                "view_correlations": correlations,
                "predicted_class_distributions": distributions,
                "known_predicted_class_distributions": known_distributions,
                "predicted_class_reference_width": {
                    "by_class": predicted_reference_by_class,
                    "pair_weighted": weighted_reference,
                    "weighting": "evaluation_pair_prediction_frequency",
                },
                "raw_activation_normalization_attribution": {
                    "pair_mean_e_vs_r_pearson": _correlation(
                        pair_e, pair_r, rank=False
                    ),
                    "pair_mean_e_vs_r_spearman": _correlation(
                        pair_e, pair_r, rank=True
                    ),
                    "pair_mean_m_vs_r_pearson": _correlation(
                        pair_m, pair_r, rank=False
                    ),
                    "pair_mean_m_vs_r_spearman": _correlation(
                        pair_m, pair_r, rank=True
                    ),
                    "mean_view_normalized_minus_raw_auroc": (
                        score_variant_metrics["normalized_error_r"]["mean"]["auroc"]
                        - score_variant_metrics["raw_error_e"]["mean"]["auroc"]
                    ),
                    "causal_label": False,
                },
                "view_aggregation_attribution": {
                    "mean_minus_best_single_view_auroc": (
                        a_aurocs["mean"] - max(a_aurocs["view1"], a_aurocs["view2"])
                    ),
                    "mean_minus_worst_single_view_auroc": (
                        a_aurocs["mean"] - min(a_aurocs["view1"], a_aurocs["view2"])
                    ),
                    "causal_label": False,
                },
                "view_acceptance_pattern_fraction": patterns,
                "unknown_r_below_predicted_reference_p95_fraction": float(
                    np.mean(
                        [
                            float(row["view1_r_predicted"])
                            <= float(row["predicted_class_reference_p95_r"])
                            for row in selected
                        ]
                        + [
                            float(row["view2_r_predicted"])
                            <= float(row["predicted_class_reference_p95_r"])
                            for row in selected
                        ]
                    )
                ),
                "causal_label": False,
            }
        )
    return result


def build_augmented_unit_arrays(
    unit: VerifiedLegacyUnit,
    augmented_z: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 128,
) -> dict[str, np.ndarray]:
    """Forward shared augmented Z through one frozen D1/D2 checkpoint."""

    values = torch.from_numpy(np.asarray(augmented_z, dtype=np.float32))
    if values.ndim != 3 or values.shape[1:] != unit.z.shape[1:]:
        raise DataValidationError("augmented Z does not match the sealed feature-map shape")
    collected = {"u": [], "logits": [], "probabilities": []}
    unit.model.requires_grad_(False).eval()
    with torch.no_grad():
        for start in range(0, values.shape[0], batch_size):
            output = unit.model(values[start : start + batch_size].to(device))
            collected["u"].append(output.adapted_features.detach().cpu().numpy())
            collected["logits"].append(output.logits.detach().cpu().numpy())
            collected["probabilities"].append(output.probabilities.detach().cpu().numpy())
    result = {
        name: np.concatenate(parts, axis=0).astype(np.float32)
        for name, parts in collected.items()
    }
    if any(not np.isfinite(values).all() for values in result.values()):
        raise DataValidationError("augmented CSSR forward contains NaN or Inf")
    return result


def _official_score_api() -> tuple[Any, Any, Any, Any]:
    """Delay the Stage-B score import so Stage-A primitives remain independent."""

    try:
        from hrrp_osr.evaluation.official_cssr_scores import (
            build_official_score_templates,
            fit_score_normalization,
            official_pcssr_pair_scores,
            raw_official_scores,
        )
    except ImportError as exc:  # pragma: no cover - only relevant during partial checkout
        raise DataValidationError(
            "official score module is unavailable; Stage-A post-hoc scoring is blocked"
        ) from exc
    return (
        build_official_score_templates,
        raw_official_scores,
        fit_score_normalization,
        official_pcssr_pair_scores,
    )


def _official_metric_row(
    *,
    unit: VerifiedLegacyUnit,
    pair_rows: Sequence[Mapping[str, Any]],
    predictions: np.ndarray,
    unknown_score: np.ndarray,
    rule: str,
    scope: str,
    identity: str,
) -> dict[str, Any]:
    roles = np.asarray([str(row["evaluation_role"]) for row in pair_rows])
    identities = np.asarray([str(row["class_name"]) for row in pair_rows])
    labels = np.asarray([int(row["true_label"]) for row in pair_rows], dtype=np.int64)
    known = np.flatnonzero(roles == "known_calibration")
    unknown = np.flatnonzero(
        (roles == "surrogate_unknown")
        & ((identities == identity) if identity else np.ones(roles.shape, dtype=bool))
    )
    if known.size == 0 or unknown.size == 0:
        raise DataValidationError("official score metric population is incomplete")
    metrics = evaluate_open_set(
        known_true=labels[known],
        known_pred=predictions[known],
        known_unknown_scores=unknown_score[known],
        unknown_pred=predictions[unknown],
        unknown_unknown_scores=unknown_score[unknown],
        known_validation_scores=unknown_score[known],
        known_class_count=5,
        known_acceptance_rate=0.95,
    )
    return {
        "pair_id": unit.pair_id,
        "method": unit.method,
        "scope": scope,
        "surrogate_identity": identity,
        "score_rule": rule,
        "known_accuracy": float(metrics["known_accuracy"]),
        "known_macro_f1": float(metrics["known_macro_f1"]),
        "auroc": float(metrics["auroc"]),
        "oscr": float(metrics["oscr"]),
        "fpr95": float(metrics["fpr95"]),
        "threshold": float(metrics["threshold"]),
        "performance_gate_eligible": False,
        "stage_b_selection_used": False,
        "final_unknown_used": False,
        "even_angle_test_used": False,
    }


def official_posthoc_score_rows(
    unit: VerifiedLegacyUnit,
    augmented: Mapping[str, np.ndarray],
    *,
    device: torch.device,
    batch_size: int = 128,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Apply official S1/S2/S3 to old U without changing the old method gate."""

    (
        build_templates,
        raw_scores,
        fit_normalization,
        pair_scores,
    ) = _official_score_api()
    train = np.asarray(
        [
            index
            for index, row in enumerate(unit.unique_rows)
            if str(row["experiment_role"]) == "train_known"
        ],
        dtype=np.int64,
    )
    raw_u = torch.from_numpy(
        unit.decomposition.adapted_features[train].astype(np.float32)
    ).to(device)
    raw_predictions = torch.from_numpy(
        unit.decomposition.probabilities[train].argmax(axis=1).astype(np.int64)
    ).to(device)
    templates = build_templates(
        raw_u, raw_predictions, num_classes=5, power=8
    )
    counts = templates.counts.detach().cpu().numpy().astype(np.int64)
    if counts.shape != (5,) or np.any(counts <= 0) or int(counts.sum()) != train.size:
        raise DataValidationError("official post-hoc template grouping is incomplete")

    aug_u = torch.from_numpy(np.asarray(augmented["u"], dtype=np.float32)).to(device)
    aug_logits = torch.from_numpy(
        np.asarray(augmented["logits"], dtype=np.float32)
    ).to(device)
    aug_probabilities = np.asarray(augmented["probabilities"], dtype=np.float32)
    if (
        aug_u.shape[0] != train.size * 4
        or aug_logits.shape != (train.size * 4, 5, raw_u.shape[2])
        or aug_probabilities.shape != (train.size * 4, 5)
    ):
        raise DataValidationError("official score normalization does not contain 4x train bases")
    aug_predictions = torch.from_numpy(
        aug_probabilities.argmax(axis=1).astype(np.int64)
    ).to(device)
    augmented_raw_scores = raw_scores(
        aug_u, aug_logits, aug_predictions, templates
    )
    normalization = fit_normalization(
        augmented_raw_scores,
        min_std=1.0e-12,
        epsilon=1.0e-8,
    )

    by_id = {
        str(row["sample_id"]): index for index, row in enumerate(unit.unique_rows)
    }
    prediction_chunks: list[np.ndarray] = []
    raw_view_chunks: list[np.ndarray] = []
    standardized_view_chunks: list[np.ndarray] = []
    pair_component_chunks: list[np.ndarray] = []
    score_chunks: dict[str, list[np.ndarray]] = {
        name: [] for name in ("S1", "S2", "S3", "full")
    }
    for start in range(0, len(unit.evaluation_rows), batch_size):
        current = unit.evaluation_rows[start : start + batch_size]
        indices = np.asarray(
            [
                [by_id[str(row["view1_sample_id"])], by_id[str(row["view2_sample_id"])]]
                for row in current
            ],
            dtype=np.int64,
        )
        view_features = torch.from_numpy(
            unit.decomposition.adapted_features[indices].astype(np.float32)
        ).to(device)
        view_logits = torch.from_numpy(
            unit.decomposition.logits[indices].astype(np.float32)
        ).to(device)
        view_probabilities = torch.from_numpy(
            unit.decomposition.probabilities[indices].astype(np.float32)
        ).to(device)
        output = pair_scores(
            view_features,
            view_logits,
            view_probabilities,
            templates,
            normalization,
        )
        prediction_chunks.append(output.predicted_class.detach().cpu().numpy())
        raw_view_chunks.append(output.per_view_raw.detach().cpu().numpy())
        standardized_view_chunks.append(
            output.per_view_standardized.detach().cpu().numpy()
        )
        pair_component_chunks.append(
            output.pair_standardized_components.detach().cpu().numpy()
        )
        for name in score_chunks:
            score_chunks[name].append(
                output.unknown_scores_by_rule[name].detach().cpu().numpy()
            )
    predictions = np.concatenate(prediction_chunks).astype(np.int64)
    raw_views = np.concatenate(raw_view_chunks).astype(np.float64)
    standardized_views = np.concatenate(standardized_view_chunks).astype(np.float64)
    pair_components = np.concatenate(pair_component_chunks).astype(np.float64)
    unknown_scores = {
        name: np.concatenate(parts).astype(np.float64)
        for name, parts in score_chunks.items()
    }
    rows = [dict(row) for row in unit.prediction_rows]
    if len(rows) != predictions.size:
        raise DataValidationError("official post-hoc pair order is not manifest-aligned")
    detailed_rows: list[dict[str, Any]] = []
    for index, source in enumerate(rows):
        detailed_rows.append(
            {
                "pair_id": unit.pair_id,
                "evaluation_pair_id": source["pair_id"],
                "method": unit.method,
                "evaluation_role": source["evaluation_role"],
                "class_name": source["class_name"],
                "surrogate_identity": source.get("surrogate_identity", ""),
                "true_label": int(source["true_label"]),
                "pcssr_pair_predicted_label": int(predictions[index]),
                "view1_sample_id": source["view1_sample_id"],
                "view2_sample_id": source["view2_sample_id"],
                "view1_raw_s1": float(raw_views[index, 0, 0]),
                "view1_raw_s2": float(raw_views[index, 0, 1]),
                "view1_raw_s3": float(raw_views[index, 0, 2]),
                "view2_raw_s1": float(raw_views[index, 1, 0]),
                "view2_raw_s2": float(raw_views[index, 1, 1]),
                "view2_raw_s3": float(raw_views[index, 1, 2]),
                "view1_standardized_s1": float(standardized_views[index, 0, 0]),
                "view1_standardized_s2": float(standardized_views[index, 0, 1]),
                "view1_standardized_s3": float(standardized_views[index, 0, 2]),
                "view2_standardized_s1": float(standardized_views[index, 1, 0]),
                "view2_standardized_s2": float(standardized_views[index, 1, 1]),
                "view2_standardized_s3": float(standardized_views[index, 1, 2]),
                "pair_standardized_s1": float(pair_components[index, 0]),
                "pair_standardized_s2": float(pair_components[index, 1]),
                "pair_standardized_s3": float(pair_components[index, 2]),
                "unknown_score_s1": float(unknown_scores["S1"][index]),
                "unknown_score_s2": float(unknown_scores["S2"][index]),
                "unknown_score_s3": float(unknown_scores["S3"][index]),
                "unknown_score_full": float(unknown_scores["full"][index]),
                "performance_gate_eligible": False,
                "stage_b_selection_used": False,
                "final_unknown_used": False,
                "even_angle_test_used": False,
            }
        )
    metric_rows: list[dict[str, Any]] = []
    identity_order = tuple(
        dict.fromkeys(
            str(row["class_name"])
            for row in rows
            if row["evaluation_role"] == "surrogate_unknown"
        )
    )
    for rule, values in unknown_scores.items():
        metric_rows.append(
            _official_metric_row(
                unit=unit,
                pair_rows=rows,
                predictions=predictions,
                unknown_score=values,
                rule=rule,
                scope="pair",
                identity="",
            )
        )
        for identity in identity_order:
            metric_rows.append(
                _official_metric_row(
                    unit=unit,
                    pair_rows=rows,
                    predictions=predictions,
                    unknown_score=values,
                    rule=rule,
                    scope="identity",
                    identity=identity,
                )
            )
    audit = {
        "status": "passed",
        "performance_gate_eligible": False,
        "pair_id": unit.pair_id,
        "method": unit.method,
        "template_prediction_counts": counts.tolist(),
        "template_first_order_sha256": array_sha256(
            templates.first_order.detach().cpu().numpy()
        ),
        "template_gram_sha256": array_sha256(templates.gram.detach().cpu().numpy()),
        "normalization": {
            "mean": normalization.mean.detach().cpu().numpy().tolist(),
            "std": normalization.std.detach().cpu().numpy().tolist(),
            "population_ddof": 0,
            "epsilon": normalization.epsilon,
            "minimum_std": normalization.min_std,
            "augmented_train_base_count": int(train.size * 4),
            "raw_score_sha256": array_sha256(
                augmented_raw_scores.values.detach().cpu().numpy()
            ),
        },
        "augmented_u_sha256": array_sha256(
            np.asarray(augmented["u"], dtype=np.float32)
        ),
        "augmented_logits_sha256": array_sha256(
            np.asarray(augmented["logits"], dtype=np.float32)
        ),
        "augmented_probabilities_sha256": array_sha256(
            np.asarray(augmented["probabilities"], dtype=np.float32)
        ),
        "score_rules": ["S1", "S2", "S3", "full"],
        "metric_rows": metric_rows,
        "pair_probability_prediction": "old_D1_or_D2_pcssr_mean_view_argmax",
        "stage_b_model_or_gate_modified": False,
        "stage_b_selection_used": False,
        "final_unknown_used": False,
        "even_angle_test_used": False,
    }
    return detailed_rows, metric_rows, audit


def guarded_official_posthoc_score_rows(
    unit: VerifiedLegacyUnit,
    augmented: Mapping[str, np.ndarray],
    *,
    device: torch.device,
    batch_size: int = 128,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Keep an official-score failure local to its preregistered diagnostic."""

    try:
        return official_posthoc_score_rows(
            unit,
            augmented,
            device=device,
            batch_size=batch_size,
        )
    except (DataValidationError, ValueError) as exc:
        return [], [], {
            "status": "failed",
            "pair_id": unit.pair_id,
            "method": unit.method,
            "failure_type": type(exc).__name__,
            "failure_reason": str(exc),
            "performance_gate_eligible": False,
            "stage_b_model_or_gate_modified": False,
            "stage_b_selection_used": False,
            "confirmation_allowed": False,
            "final_unknown_used": False,
            "even_angle_test_used": False,
        }


def _prepare_augmented_z(
    *,
    pair_id: str,
    representative: VerifiedLegacyUnit,
    legacy_config_path: str | Path,
    bundle_root: str | Path,
    r2_results_root: str | Path,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Recreate input-level augmentations through the strict-frozen R2 encoder."""

    # Delayed imports keep the pure Stage-A statistics usable without loading
    # the old experiment runner, while the real run still reuses its audited
    # data and R2 reconstruction path.
    from hrrp_osr.training.fg_mv_cssr_decoupled import (
        load_fg_mv_cssr_decoupled_config,
    )
    from hrrp_osr.training.fg_mv_cssr_pilot import (
        _load_prior_config,
        _prepare_frozen_split,
        build_unique_base_sample_manifest,
        extract_frozen_feature_maps,
        load_and_audit_frozen_r2,
    )
    from hrrp_osr.training.official_cssr_hrrp_pilot import (
        _load_development_only_bundle,
        _profile_access_audit,
    )

    config = load_fg_mv_cssr_decoupled_config(legacy_config_path)
    project_root = Path(config["_config_path"]).parents[3]
    prior_config = _load_prior_config(project_root, config)
    bundle = _load_development_only_bundle(bundle_root, config)
    prepared = _prepare_frozen_split(bundle, prior_config, config, pair_id)
    r2_model, _, r2_audit = load_and_audit_frozen_r2(
        project_root=project_root,
        r2_results_root=r2_results_root,
        pair_id=pair_id,
        config=config,
        prepared=prepared,
        prior_config=prior_config,
        device=device,
    )
    if r2_model.training or any(parameter.requires_grad for parameter in r2_model.parameters()):
        raise DataValidationError("R2 must remain frozen and eval during Stage-A augmentation")
    rebuilt_rows = build_unique_base_sample_manifest(prepared, bundle)
    if _render_csv(rebuilt_rows) != (
        representative.root / "unique_base_sample_manifest.csv"
    ).read_bytes():
        raise DataValidationError("rebuilt unique-base manifest differs from sealed Stage-A input")
    rebuilt_z, feature_audit = extract_frozen_feature_maps(
        model=r2_model,
        bundle=bundle,
        prepared=prepared,
        rows=rebuilt_rows,
        device=device,
        batch_size=128,
    )
    if not np.array_equal(rebuilt_z, representative.z):
        raise DataValidationError("rebuilt raw Z differs from sealed checkpoint replay")
    expected_normalization = {
        "method": "reuse_exact_r2_global_scalar_zscore",
        "mean": float(prepared.normalization.mean),
        "std": float(prepared.normalization.std),
        "epsilon": float(prepared.normalization.epsilon),
        "unique_base_sample_count": int(
            prepared.normalization.unique_base_sample_count
        ),
    }
    if _read_json(representative.root / "normalization.json") != expected_normalization:
        raise DataValidationError("sealed D1/D2 input normalization does not reproduce")
    train_rows = [row for row in rebuilt_rows if row["experiment_role"] == "train_known"]
    processed_indices = np.asarray(
        [int(row["processed_row_index"]) for row in train_rows], dtype=np.int64
    )
    raw_inputs = np.asarray(bundle.profiles[processed_indices], dtype=np.float64)
    normalized_inputs = (
        raw_inputs - float(prepared.normalization.mean)
    ) / float(prepared.normalization.std)
    augmented_inputs, augmentation_audit = build_score_normalization_augmentations(
        normalized_inputs,
        [str(row["sample_id"]) for row in train_rows],
        pair_id=pair_id,
    )
    tensor = torch.from_numpy(augmented_inputs)
    parts: list[np.ndarray] = []
    r2_model.eval()
    with torch.no_grad():
        for start in range(0, tensor.shape[0], 128):
            parts.append(
                r2_model.encoder.forward_feature_map(tensor[start : start + 128].to(device))
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
    augmented_z = np.concatenate(parts, axis=0)
    if augmented_z.shape != (2880, 128, 76) or not np.isfinite(augmented_z).all():
        raise DataValidationError("augmented R2 feature-map population is invalid")
    return augmented_z, {
        "status": "passed",
        "pair_id": pair_id,
        "normalization": {
            **expected_normalization,
            "fitted_from_train_known_only": True,
        },
        "raw_z_replay_exact": True,
        "raw_z_sha256": array_sha256(rebuilt_z),
        "augmented_z_sha256": array_sha256(augmented_z),
        "feature_audit": feature_audit,
        "r2_audit": r2_audit,
        "augmentation": augmentation_audit,
        "data_access_audit": _profile_access_audit(bundle),
        "surrogate_unknown_used": False,
        "known_calibration_used": False,
        "final_unknown_used": False,
        "even_angle_test_used": False,
    }


def _base_npz(units: Mapping[tuple[str, str], VerifiedLegacyUnit]) -> dict[str, np.ndarray]:
    first_per_pair = [units[(pair_id, METHODS[0])] for pair_id in PAIR_IDS]
    for pair_id in PAIR_IDS:
        left = units[(pair_id, METHODS[0])]
        right = units[(pair_id, METHODS[1])]
        if left.unique_rows != right.unique_rows or not np.array_equal(left.z, right.z):
            raise DataValidationError(f"D1/D2 base population or Z differs for {pair_id}")
    return {
        "pair_ids": np.asarray(PAIR_IDS, dtype=np.str_),
        "methods": np.asarray(METHODS, dtype=np.str_),
        "sample_ids": np.asarray(
            [[str(row["sample_id"]) for row in unit.unique_rows] for unit in first_per_pair],
            dtype=np.str_,
        ),
        "experiment_roles": np.asarray(
            [
                [str(row["experiment_role"]) for row in unit.unique_rows]
                for unit in first_per_pair
            ],
            dtype=np.str_,
        ),
        "class_names": np.asarray(
            [[str(row["class_name"]) for row in unit.unique_rows] for unit in first_per_pair],
            dtype=np.str_,
        ),
        "model_labels": np.asarray(
            [[int(row["model_label"]) for row in unit.unique_rows] for unit in first_per_pair],
            dtype=np.int64,
        ),
        "raw_error_e": np.stack(
            [
                np.stack(
                    [units[(pair_id, method)].decomposition.raw_error for method in METHODS]
                )
                for pair_id in PAIR_IDS
            ]
        ),
        "activation_magnitude_m": np.stack(
            [
                np.stack(
                    [
                        units[(pair_id, method)].decomposition.activation_magnitude
                        for method in METHODS
                    ]
                )
                for pair_id in PAIR_IDS
            ]
        ),
        "normalized_error_r": np.stack(
            [
                np.stack(
                    [
                        units[(pair_id, method)].decomposition.normalized_error
                        for method in METHODS
                    ]
                )
                for pair_id in PAIR_IDS
            ]
        ),
        "conformal_p": np.stack(
            [
                np.stack(
                    [units[(pair_id, method)].decomposition.p_value for method in METHODS]
                )
                for pair_id in PAIR_IDS
            ]
        ),
        "anomaly_a": np.stack(
            [
                np.stack(
                    [units[(pair_id, method)].decomposition.anomaly for method in METHODS]
                )
                for pair_id in PAIR_IDS
            ]
        ),
        "z_sha256": np.asarray(
            [array_sha256(unit.z) for unit in first_per_pair], dtype=np.str_
        ),
        "u_sha256": np.asarray(
            [
                [
                    array_sha256(
                        units[(pair_id, method)].decomposition.adapted_features
                    )
                    for method in METHODS
                ]
                for pair_id in PAIR_IDS
            ],
            dtype=np.str_,
        ),
    }


def save_mechanism_outputs(
    output_root: str | Path,
    *,
    input_roots: Sequence[str | Path],
    base_arrays: Mapping[str, np.ndarray],
    pair_rows: Sequence[Mapping[str, Any]],
    cross_raw_rows: Sequence[Mapping[str, Any]],
    cross_normalized_rows: Sequence[Mapping[str, Any]],
    specificity: Mapping[str, Any],
    reference_rows: Sequence[Mapping[str, Any]],
    geometry: Mapping[str, Any],
    official_rows: Sequence[Mapping[str, Any]],
    mechanism_audit: Mapping[str, Any],
) -> Path:
    """Write a complete result atomically without entering any legacy tree."""

    destination = require_disjoint_output(input_roots, output_root)
    if destination.exists():
        raise DataValidationError(f"mechanism output already exists: {destination}")
    staging = destination.parent / f".{destination.name}.staging"
    if staging.exists():
        raise DataValidationError(f"stale mechanism staging directory exists: {staging}")
    staging.mkdir(parents=True, exist_ok=False)
    _write_npz(staging / "per_base_score_decomposition.npz", base_arrays)
    _write_csv(staging / "per_pair_view_decomposition.csv", pair_rows)
    _write_csv(staging / "ae_cross_reconstruction_raw.csv", cross_raw_rows)
    _write_csv(
        staging / "ae_cross_reconstruction_normalized.csv", cross_normalized_rows
    )
    _write_json(staging / "ae_specificity.json", specificity)
    _write_csv(staging / "reference_distribution_summary.csv", reference_rows)
    _write_json(staging / "representation_geometry_z_vs_u.json", geometry)
    _write_csv(
        staging / "official_scores_on_d1_d2.csv",
        official_rows,
        fieldnames=OFFICIAL_SCORE_CSV_FIELDS,
    )
    _write_json(staging / "mechanism_audit.json", mechanism_audit)
    _write_json(staging / "artifact_hashes.json", _tree_artifact_hashes(staging))
    observed_names = {path.name for path in staging.iterdir() if path.is_file()}
    if observed_names != set(OUTPUT_FILENAMES):
        raise DataValidationError("mechanism output file set is incomplete")
    staging.replace(destination)
    return destination


def _assert_finite_json_tree(value: Any, *, context: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite_json_tree(item, context=f"{context}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _assert_finite_json_tree(item, context=f"{context}[{index}]")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise DataValidationError(f"{context} contains NaN or Inf")


def _csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle).fieldnames or [])


def _close(
    left: Any,
    right: Any,
    *,
    context: str,
    rtol: float = 0.0,
    atol: float = 1.0e-12,
) -> None:
    if not math.isclose(float(left), float(right), rel_tol=rtol, abs_tol=atol):
        raise DataValidationError(f"{context} does not reproduce")


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _typed_pair_row(row: Mapping[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = dict(row)
    integer_fields = {
        "true_label",
        "r2_predicted_label",
        "view1_angle_deg",
        "view2_angle_deg",
        "view1_frame_id",
        "view2_frame_id",
        "predicted_class_reference_count_r",
        "view1_lowest_r_ae",
        "view2_lowest_r_ae",
    }
    boolean_fields = {
        "view1_accepted_at_predicted_class_p95",
        "view2_accepted_at_predicted_class_p95",
        "lowest_r_ae_same_across_views",
        "final_unknown_used",
        "even_angle_test_used",
        "performance_gate_eligible",
    }
    for key, value in tuple(result.items()):
        if key in integer_fields:
            result[key] = int(value)
        elif key in boolean_fields:
            if value not in {"True", "False"}:
                raise DataValidationError(f"invalid Stage-A boolean field {key}")
            result[key] = value == "True"
        elif (
            key.startswith("predicted_class_reference_")
            or key.startswith(
                (
                    "pair_e_",
                    "pair_m_",
                    "pair_r_",
                    "pair_p_",
                    "pair_a_",
                )
            )
            or (
                key.startswith(("view1_", "view2_"))
                and key.endswith("_predicted")
            )
            or key in {"view1_m", "view2_m", "d0_class_conditional_mls"}
        ):
            result[key] = float(value)
    return result


def _audit_mechanism_payload(
    root: Path,
    mechanism: Mapping[str, Any],
    legacy_pilot_root: str | Path,
) -> None:
    """Rebuild Stage-A tables from saved base arrays and reject truncation."""

    unit_keys = {
        f"{pair_id}/{method}" for pair_id in PAIR_IDS for method in METHODS
    }
    with np.load(root / "per_base_score_decomposition.npz", allow_pickle=False) as saved:
        required = {
            "pair_ids",
            "methods",
            "sample_ids",
            "experiment_roles",
            "class_names",
            "model_labels",
            "raw_error_e",
            "activation_magnitude_m",
            "normalized_error_r",
            "conformal_p",
            "anomaly_a",
            "z_sha256",
            "u_sha256",
        }
        if set(saved.files) != required:
            raise DataValidationError("per-base decomposition schema changed")
        arrays = {name: np.asarray(saved[name]).copy() for name in required}
    expected_shapes = {
        "pair_ids": (3,),
        "methods": (2,),
        "sample_ids": (3, 972),
        "experiment_roles": (3, 972),
        "class_names": (3, 972),
        "model_labels": (3, 972),
        "raw_error_e": (3, 2, 972, 5),
        "activation_magnitude_m": (3, 2, 972),
        "normalized_error_r": (3, 2, 972, 5),
        "conformal_p": (3, 2, 972, 5),
        "anomaly_a": (3, 2, 972, 5),
        "z_sha256": (3,),
        "u_sha256": (3, 2),
    }
    if any(arrays[name].shape != shape for name, shape in expected_shapes.items()):
        raise DataValidationError("per-base decomposition shape or population changed")
    if arrays["pair_ids"].tolist() != list(PAIR_IDS) or arrays[
        "methods"
    ].tolist() != list(METHODS):
        raise DataValidationError("per-base decomposition plan changed")
    numeric_names = (
        "raw_error_e",
        "activation_magnitude_m",
        "normalized_error_r",
        "conformal_p",
        "anomaly_a",
    )
    if any(not np.isfinite(arrays[name]).all() for name in numeric_names):
        raise DataValidationError("per-base decomposition contains NaN or Inf")
    if (
        np.any(arrays["raw_error_e"] < 0.0)
        or np.any(arrays["activation_magnitude_m"] < 0.0)
        or np.any(arrays["normalized_error_r"] < 0.0)
        or np.any(arrays["conformal_p"] < 0.0)
        or np.any(arrays["conformal_p"] > 1.0)
        or not np.allclose(
            arrays["normalized_error_r"],
            arrays["raw_error_e"]
            / (arrays["activation_magnitude_m"][..., None] + EPSILON),
            rtol=1.0e-6,
            atol=1.0e-7,
        )
        or not np.array_equal(
            arrays["anomaly_a"], -np.log(arrays["conformal_p"] + EPSILON)
        )
    ):
        raise DataValidationError("saved E/M/r/p/a definitions do not reproduce")
    hashes = arrays["z_sha256"].reshape(-1).tolist() + arrays[
        "u_sha256"
    ].reshape(-1).tolist()
    if any(not _is_sha256(value) for value in hashes):
        raise DataValidationError("saved Z/U hashes are invalid")

    legacy_root = Path(legacy_pilot_root).resolve(strict=True)
    legacy_by_unit: dict[str, dict[str, Any]] = {}
    for pair_id in PAIR_IDS:
        for method in METHODS:
            key = f"{pair_id}/{method}"
            unit_root = (
                legacy_root
                / pair_id
                / "fold_0"
                / f"seed_{LEGACY_SEED}"
                / method
            )
            unique_rows = _read_csv(unit_root / "unique_base_sample_manifest.csv")
            evaluation_rows = _read_csv(unit_root / "evaluation_pair_manifest.csv")
            prediction_rows = _read_csv(unit_root / "predictions_and_scores.csv")
            d0_rows = _read_csv(unit_root / "d0_predictions_and_scores.csv")
            if not (
                len(unique_rows) == 972
                and len(evaluation_rows)
                == len(prediction_rows)
                == len(d0_rows)
                == 3500
            ):
                raise DataValidationError(f"sealed legacy population changed for {key}")
            with np.load(
                unit_root / "checkpoint_replay.npz", allow_pickle=False
            ) as replay:
                required_replay = {
                    "unique_features",
                    "expected_u",
                    "expected_r",
                    "expected_probabilities",
                }
                if set(replay.files) != required_replay:
                    raise DataValidationError(
                        f"sealed checkpoint replay schema changed for {key}"
                    )
                z = np.asarray(replay["unique_features"], dtype=np.float32).copy()
                u = np.asarray(replay["expected_u"], dtype=np.float32).copy()
                r = np.asarray(replay["expected_r"], dtype=np.float32).copy()
                probabilities = np.asarray(
                    replay["expected_probabilities"], dtype=np.float32
                ).copy()
            if (
                z.shape != (972, 128, 76)
                or u.shape != (972, 128, 76)
                or r.shape != (972, 5)
                or probabilities.shape != (972, 5)
                or not all(
                    np.isfinite(value).all()
                    for value in (z, u, r, probabilities)
                )
            ):
                raise DataValidationError(f"sealed checkpoint replay changed for {key}")
            legacy_by_unit[key] = {
                "root": unit_root,
                "unique_rows": unique_rows,
                "evaluation_rows": evaluation_rows,
                "prediction_rows": prediction_rows,
                "d0_rows": d0_rows,
                "z": z,
                "u": u,
                "r": r,
                "probabilities": probabilities,
                "artifact_manifest_sha256": file_sha256(
                    unit_root / "artifact_hashes.json"
                ),
            }

    base_indices: dict[str, dict[str, int]] = {}
    base_rows: dict[str, list[dict[str, Any]]] = {}
    for pair_index, pair_id in enumerate(PAIR_IDS):
        sample_ids = [str(value) for value in arrays["sample_ids"][pair_index]]
        roles = [str(value) for value in arrays["experiment_roles"][pair_index]]
        names = [str(value) for value in arrays["class_names"][pair_index]]
        labels = arrays["model_labels"][pair_index].astype(np.int64)
        if len(set(sample_ids)) != 972 or Counter(roles) != Counter(
            {"train_known": 720, "known_calibration": 180, "surrogate_unknown": 72}
        ):
            raise DataValidationError(f"per-base role population changed for {pair_id}")
        for role, per_class, class_count in (
            ("train_known", 144, 5),
            ("known_calibration", 36, 5),
            ("surrogate_unknown", 36, 2),
        ):
            selected = np.flatnonzero(np.asarray(roles) == role)
            identity_counts = Counter(names[int(index)] for index in selected)
            if (
                len(identity_counts) != class_count
                or set(identity_counts.values()) != {per_class}
            ):
                raise DataValidationError(f"per-base identity population changed for {pair_id}/{role}")
        for role in ("train_known", "known_calibration"):
            selected = np.flatnonzero(np.asarray(roles) == role)
            expected = 144 if role == "train_known" else 36
            if Counter(labels[selected].tolist()) != Counter(
                {index: expected for index in range(5)}
            ):
                raise DataValidationError(f"per-base labels changed for {pair_id}/{role}")
        base_indices[pair_id] = {
            sample_id: index for index, sample_id in enumerate(sample_ids)
        }
        base_rows[pair_id] = [
            {
                "sample_id": sample_ids[index],
                "experiment_role": roles[index],
                "class_name": names[index],
                "model_label": int(labels[index]),
            }
            for index in range(972)
        ]
        for method_index, method in enumerate(METHODS):
            key = f"{pair_id}/{method}"
            legacy = legacy_by_unit[key]
            legacy_rows = legacy["unique_rows"]
            if (
                sample_ids
                != [str(row["sample_id"]) for row in legacy_rows]
                or roles
                != [str(row["experiment_role"]) for row in legacy_rows]
                or names != [str(row["class_name"]) for row in legacy_rows]
                or labels.tolist()
                != [int(row["model_label"]) for row in legacy_rows]
                or not np.array_equal(
                    arrays["normalized_error_r"][pair_index, method_index],
                    np.asarray(legacy["r"], dtype=np.float64),
                )
                or not np.array_equal(
                    arrays["activation_magnitude_m"][pair_index, method_index],
                    np.abs(np.asarray(legacy["u"], dtype=np.float32)).mean(
                        axis=(1, 2)
                    ),
                )
                or str(arrays["z_sha256"][pair_index])
                != array_sha256(np.asarray(legacy["z"]))
                or str(arrays["u_sha256"][pair_index, method_index])
                != array_sha256(np.asarray(legacy["u"]))
            ):
                raise DataValidationError(
                    f"Stage-A base arrays are not bound to sealed input {key}"
                )
            expected_p, expected_a, _, _ = conformal_p_and_anomaly(
                arrays["normalized_error_r"][pair_index, method_index],
                base_rows[pair_id],
            )
            if not np.array_equal(
                arrays["conformal_p"][pair_index, method_index], expected_p
            ) or not np.array_equal(
                arrays["anomaly_a"][pair_index, method_index], expected_a
            ):
                raise DataValidationError(
                    f"Stage-A conformal p/a do not reproduce for {key}"
                )
        augmentation_record = mechanism["augmentation_audits"][pair_id]
        feature_audit = augmentation_record.get("feature_audit")
        if (
            augmentation_record.get("raw_z_sha256")
            != str(arrays["z_sha256"][pair_index])
            or not isinstance(feature_audit, Mapping)
            or feature_audit.get("status") != "passed"
            or feature_audit.get("shape") != [972, 128, 76]
            or feature_audit.get("dtype") != "float32"
            or feature_audit.get("feature_map_sha256")
            != str(arrays["z_sha256"][pair_index])
            or feature_audit.get("input_shape") != [128, 76]
            or feature_audit.get("all_finite") is not True
            or feature_audit.get("r2_eval_mode") is not True
            or feature_audit.get("r2_parameters_frozen") is not True
        ):
            raise DataValidationError(
                f"Stage-A augmented-score R2 replay changed for {pair_id}"
            )

    pair_rows_raw = _read_csv(root / "per_pair_view_decomposition.csv")
    if len(pair_rows_raw) != 3 * 2 * 3500:
        raise DataValidationError("per-pair decomposition row count changed")
    required_pair_fields = {
        "pair_id",
        "evaluation_pair_id",
        "method",
        "evaluation_role",
        "class_name",
        "true_label",
        "r2_predicted_label",
        "view1_sample_id",
        "view2_sample_id",
        "view1_angle_deg",
        "view2_angle_deg",
        "view1_m",
        "view2_m",
        "view1_e_all_classes",
        "view2_e_all_classes",
        "view1_r_all_classes",
        "view2_r_all_classes",
        "view1_p_all_classes",
        "view2_p_all_classes",
        "view1_a_all_classes",
        "view2_a_all_classes",
        "view1_e_predicted",
        "view2_e_predicted",
        "view1_r_predicted",
        "view2_r_predicted",
        "view1_p_predicted",
        "view2_p_predicted",
        "view1_a_predicted",
        "view2_a_predicted",
        "pair_e_mean",
        "pair_m_mean",
        "pair_r_mean",
        "pair_p_mean",
        "pair_a_mean",
        "final_unknown_used",
        "even_angle_test_used",
        "performance_gate_eligible",
    }
    pair_rows_by_unit: dict[str, list[dict[str, Any]]] = {
        key: [] for key in unit_keys
    }
    pair_id_to_index = {name: index for index, name in enumerate(PAIR_IDS)}
    method_to_index = {name: index for index, name in enumerate(METHODS)}
    for raw_row in pair_rows_raw:
        if not required_pair_fields <= set(raw_row):
            raise DataValidationError("per-pair decomposition columns changed")
        row = _typed_pair_row(raw_row)
        pair_id = str(row["pair_id"])
        method = str(row["method"])
        key = f"{pair_id}/{method}"
        if key not in pair_rows_by_unit:
            raise DataValidationError("per-pair decomposition contains an extra unit")
        if (
            row["final_unknown_used"] is not False
            or row["even_angle_test_used"] is not False
            or row["performance_gate_eligible"] is not False
            or int(row["view1_angle_deg"]) % 2 == 0
            or int(row["view2_angle_deg"]) % 2 == 0
        ):
            raise DataValidationError("per-pair decomposition violates evidence scope")
        predicted = int(row["r2_predicted_label"])
        if predicted not in range(5):
            raise DataValidationError("per-pair predicted class is outside 0..4")
        p_index = pair_id_to_index[pair_id]
        m_index = method_to_index[method]
        reference_indices = np.asarray(
            [
                index
                for index, base in enumerate(base_rows[pair_id])
                if base["experiment_role"] == "known_calibration"
                and int(base["model_label"]) == predicted
            ],
            dtype=np.int64,
        )
        reference = _reference_statistics(
            arrays["normalized_error_r"][
                p_index, m_index, reference_indices, predicted
            ]
        )
        for name, expected_value in reference.items():
            _close(
                row[f"predicted_class_reference_{name}_r"],
                expected_value,
                context=f"pair predicted reference {name}",
            )
        try:
            fused_logits = np.asarray(
                json.loads(str(row["r2_fused_logits"])), dtype=np.float64
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise DataValidationError("pair R2 fused logits are invalid") from exc
        if (
            fused_logits.shape != (5,)
            or not np.isfinite(fused_logits).all()
            or int(fused_logits.argmax()) != predicted
        ):
            raise DataValidationError("pair R2 prediction does not reproduce")
        view_values: dict[str, list[float]] = {
            name: [] for name in ("e", "m", "r", "p", "a")
        }
        view_base_rows: list[dict[str, Any]] = []
        lowest_classes: list[int] = []
        accepted_views: list[bool] = []
        for view in (1, 2):
            sample_id = str(row[f"view{view}_sample_id"])
            if sample_id not in base_indices[pair_id]:
                raise DataValidationError("pair row references an absent base sample")
            base_index = base_indices[pair_id][sample_id]
            view_base_rows.append(base_rows[pair_id][base_index])
            expected_m = arrays["activation_magnitude_m"][p_index, m_index, base_index]
            _close(row[f"view{view}_m"], expected_m, context="pair/base M")
            view_values["m"].append(float(expected_m))
            for short, array_name in (
                ("e", "raw_error_e"),
                ("r", "normalized_error_r"),
                ("p", "conformal_p"),
                ("a", "anomaly_a"),
            ):
                vector = arrays[array_name][p_index, m_index, base_index]
                saved_vector = np.asarray(
                    json.loads(str(row[f"view{view}_{short}_all_classes"])),
                    dtype=np.float64,
                )
                if saved_vector.shape != (5,) or not np.array_equal(
                    saved_vector, vector
                ):
                    raise DataValidationError(f"pair/base {short} vector changed")
                _close(
                    row[f"view{view}_{short}_predicted"],
                    vector[predicted],
                    context=f"pair/base predicted {short}",
                )
                view_values[short].append(float(vector[predicted]))
            expected_lowest = int(
                arrays["normalized_error_r"][p_index, m_index, base_index].argmin()
            )
            if int(row[f"view{view}_lowest_r_ae"]) != expected_lowest:
                raise DataValidationError("pair lowest-reconstruction AE changed")
            lowest_classes.append(expected_lowest)
            accepted = bool(
                arrays["normalized_error_r"][p_index, m_index, base_index, predicted]
                <= float(reference["p95"])
            )
            if row[f"view{view}_accepted_at_predicted_class_p95"] is not accepted:
                raise DataValidationError("pair p95 view acceptance changed")
            accepted_views.append(accepted)
        expected_role = str(row["evaluation_role"])
        if (
            expected_role not in {"known_calibration", "surrogate_unknown"}
            or any(base["experiment_role"] != expected_role for base in view_base_rows)
            or any(base["class_name"] != str(row["class_name"]) for base in view_base_rows)
            or str(row["view1_sample_id"]) == str(row["view2_sample_id"])
        ):
            raise DataValidationError("pair row is not aligned to its base population")
        if expected_role == "known_calibration" and (
            int(row["true_label"]) != int(view_base_rows[0]["model_label"])
            or int(view_base_rows[0]["model_label"])
            != int(view_base_rows[1]["model_label"])
        ):
            raise DataValidationError("known pair label does not match its bases")
        expected_pattern = (
            "both_accept"
            if sum(accepted_views) == 2
            else "one_accept"
            if sum(accepted_views) == 1
            else "both_reject"
        )
        if (
            row["lowest_r_ae_same_across_views"]
            is not (lowest_classes[0] == lowest_classes[1])
            or str(row["view_acceptance_pattern"]) != expected_pattern
        ):
            raise DataValidationError("pair cross-view summary changed")
        for short, values in view_values.items():
            first, second = values
            for suffix, expected_value in (
                ("mean", 0.5 * (first + second)),
                ("view1_minus_view2", first - second),
                ("min", min(first, second)),
                ("max", max(first, second)),
            ):
                _close(
                    row[f"pair_{short}_{suffix}"],
                    expected_value,
                    context=f"pair {short} {suffix}",
                )
        pair_rows_by_unit[key].append(row)
    for key, rows in pair_rows_by_unit.items():
        if len(rows) != 3500 or Counter(
            str(row["evaluation_role"]) for row in rows
        ) != Counter({"known_calibration": 2500, "surrogate_unknown": 1000}):
            raise DataValidationError(f"per-pair evaluation population changed for {key}")
        if len({str(row["evaluation_pair_id"]) for row in rows}) != 3500:
            raise DataValidationError(f"per-pair IDs are not unique for {key}")
        legacy = legacy_by_unit[key]
        for row, manifest, prediction, d0 in zip(
            rows,
            legacy["evaluation_rows"],
            legacy["prediction_rows"],
            legacy["d0_rows"],
            strict=True,
        ):
            exact_fields = {
                "evaluation_pair_id": manifest["pair_id"],
                "evaluation_role": prediction["evaluation_role"],
                "class_name": prediction["class_name"],
                "true_label": int(prediction["true_label"]),
                "r2_predicted_label": int(prediction["predicted_known_label"]),
                "r2_predicted_class_name": prediction[
                    "predicted_known_class_name"
                ],
                "r2_fused_logits": prediction["fused_logits"],
                "view1_sample_id": manifest["view1_sample_id"],
                "view2_sample_id": manifest["view2_sample_id"],
                "view1_angle_deg": int(manifest["view1_angle_deg"]),
                "view2_angle_deg": int(manifest["view2_angle_deg"]),
                "view1_frame_id": int(manifest["view1_frame_id"]),
                "view2_frame_id": int(manifest["view2_frame_id"]),
            }
            if any(row.get(name) != value for name, value in exact_fields.items()):
                raise DataValidationError(
                    f"Stage-A pair row is not bound to sealed input {key}"
                )
            _close(
                row["d0_class_conditional_mls"],
                d0["unknown_score"],
                context=f"sealed D0 score {key}",
            )

    raw_cross = _read_csv(root / "ae_cross_reconstruction_raw.csv")
    normalized_cross = _read_csv(root / "ae_cross_reconstruction_normalized.csv")
    if len(raw_cross) != 72 or len(normalized_cross) != 72:
        raise DataValidationError("AE cross-reconstruction row count changed")
    raw_map = {
        (row["pair_id"], row["method"], row["experiment_role"], row["identity"]): row
        for row in raw_cross
    }
    normalized_map = {
        (row["pair_id"], row["method"], row["experiment_role"], row["identity"]): row
        for row in normalized_cross
    }
    if len(raw_map) != 72 or set(raw_map) != set(normalized_map):
        raise DataValidationError("AE cross-reconstruction identities changed")
    for key, raw_row in raw_map.items():
        pair_id, method, role, identity = key
        p_index = pair_id_to_index[pair_id]
        m_index = method_to_index[method]
        selected = np.asarray(
            [
                index
                for index, row in enumerate(base_rows[pair_id])
                if row["experiment_role"] == role and row["class_name"] == identity
            ],
            dtype=np.int64,
        )
        expected_count = 144 if role == "train_known" else 36
        if selected.size != expected_count or int(raw_row["base_count"]) != expected_count:
            raise DataValidationError("AE cross-reconstruction base count changed")
        e = arrays["raw_error_e"][p_index, m_index, selected].astype(np.float64)
        r = arrays["normalized_error_r"][p_index, m_index, selected].astype(
            np.float64
        )
        a = arrays["anomaly_a"][p_index, m_index, selected].astype(np.float64)
        normalized_row = normalized_map[key]
        if int(normalized_row["base_count"]) != expected_count:
            raise DataValidationError("normalized AE cross-reconstruction count changed")
        best = int(np.argmin(r.mean(axis=0)))
        if int(raw_row["best_ae_by_mean_r"]) != best or int(
            normalized_row["best_ae_by_mean_r"]
        ) != best:
            raise DataValidationError("AE cross-reconstruction best class changed")
        for ae in range(5):
            _close(raw_row[f"mean_e_ae_{ae}"], e[:, ae].mean(), context="cross E")
            _close(
                normalized_row[f"mean_r_ae_{ae}"], r[:, ae].mean(), context="cross r"
            )
            _close(
                normalized_row[f"mean_a_ae_{ae}"], a[:, ae].mean(), context="cross a"
            )
            _close(
                normalized_row[f"lowest_r_share_ae_{ae}"],
                np.mean(np.argmin(r, axis=1) == ae),
                context="cross best-AE share",
            )

    reference_rows = _read_csv(root / "reference_distribution_summary.csv")
    if len(reference_rows) != 30:
        raise DataValidationError("reference distribution row count changed")
    reference_map = {
        (row["pair_id"], row["method"], int(row["ae_class_index"])): row
        for row in reference_rows
    }
    expected_reference_keys = {
        (pair_id, method, ae)
        for pair_id in PAIR_IDS
        for method in METHODS
        for ae in range(5)
    }
    if set(reference_map) != expected_reference_keys or len(reference_map) != 30:
        raise DataValidationError("reference distribution unit/class keys changed")
    for row in reference_map.values():
        pair_id = str(row["pair_id"])
        method = str(row["method"])
        ae = int(row["ae_class_index"])
        p_index = pair_id_to_index[pair_id]
        m_index = method_to_index[method]
        selected = np.asarray(
            [
                index
                for index, base in enumerate(base_rows[pair_id])
                if base["experiment_role"] == "known_calibration"
                and int(base["model_label"]) == ae
            ],
            dtype=np.int64,
        )
        stats = _reference_statistics(
            arrays["normalized_error_r"][p_index, m_index, selected, ae]
        )
        if selected.size != 36 or str(row["ae_class_name"]) != str(
            base_rows[pair_id][int(selected[0])]["class_name"]
        ):
            raise DataValidationError("reference distribution identity changed")
        for name, value in stats.items():
            _close(row[name], value, context=f"reference {name}")

    specificity = _read_json(root / "ae_specificity.json")
    geometry = _read_json(root / "representation_geometry_z_vs_u.json")
    if set(specificity) != unit_keys or set(geometry) != unit_keys:
        raise DataValidationError("specificity or geometry unit population changed")
    for key in sorted(unit_keys):
        pair_id, method = key.split("/", 1)
        p_index = pair_id_to_index[pair_id]
        m_index = method_to_index[method]
        synthetic_unit = type("StageAUnit", (), {})()
        synthetic_unit.pair_id = pair_id
        synthetic_unit.method = method
        synthetic_unit.unique_rows = base_rows[pair_id]
        synthetic_unit.decomposition = type("StageADecomposition", (), {})()
        synthetic_unit.decomposition.normalized_error = arrays[
            "normalized_error_r"
        ][p_index, m_index]
        if specificity[key] != ae_specificity_summary(synthetic_unit):
            raise DataValidationError(f"AE specificity does not reproduce for {key}")
        current_geometry = geometry[key]
        if (
            not isinstance(current_geometry, Mapping)
            or set(current_geometry.get("pools", {}))
            != {"train_known", "known_calibration", "surrogate", "evaluation"}
            or set(current_geometry.get("train_center_diagnostics", {})) != {"Z", "U"}
            or current_geometry.get("base_weighting") != "unique_sample_id_once"
            or current_geometry.get("pair_multiplicity_used") is not False
        ):
            raise DataValidationError(f"representation geometry schema changed for {key}")
        expected_pool_counts = {
            "train_known": (720, 5),
            "known_calibration": (180, 5),
            "surrogate": (72, 2),
            "evaluation": (252, 7),
        }
        for pool, (base_count, identity_count) in expected_pool_counts.items():
            for representation in ("Z", "U"):
                record = current_geometry["pools"][pool][representation]
                if (
                    int(record.get("base_count", -1)) != base_count
                    or int(record.get("identity_count", -1)) != identity_count
                    or len(record.get("channel_population_mean", [])) != 128
                    or len(record.get("channel_population_variance", [])) != 128
                    or len(record.get("top_singular_values", [])) != 10
                    or len(record.get("top_singular_cumulative_explained_variance", []))
                    != 10
                ):
                    raise DataValidationError(
                        f"representation geometry population changed for {key}/{pool}"
                    )
        legacy = legacy_by_unit[key]
        expected_geometry = representation_geometry(
            np.asarray(legacy["z"]),
            np.asarray(legacy["u"]),
            legacy["unique_rows"],
        )
        if current_geometry != expected_geometry:
            raise DataValidationError(
                f"representation geometry does not reproduce for {key}"
            )
        _assert_finite_json_tree(current_geometry, context=f"geometry.{key}")

    expected_identity_diagnostics: list[dict[str, Any]] = []
    for pair_id in PAIR_IDS:
        for method in METHODS:
            expected_identity_diagnostics.extend(
                identity_view_diagnostics(pair_rows_by_unit[f"{pair_id}/{method}"])
            )
    if mechanism.get("identity_view_diagnostics") != expected_identity_diagnostics:
        raise DataValidationError("identity diagnostics do not reproduce")

    if _csv_header(root / "official_scores_on_d1_d2.csv") != list(
        OFFICIAL_SCORE_CSV_FIELDS
    ):
        raise DataValidationError("official post-hoc score columns changed")
    official_rows = _read_csv(root / "official_scores_on_d1_d2.csv")
    official_audits = mechanism.get("official_posthoc_audits")
    if not isinstance(official_audits, Mapping) or set(official_audits) != unit_keys:
        raise DataValidationError("official post-hoc audit population changed")
    official_by_unit: dict[str, list[dict[str, str]]] = {key: [] for key in unit_keys}
    for row in official_rows:
        key = f"{row.get('pair_id')}/{row.get('method')}"
        if key not in official_by_unit or any(
            row.get(name) != "False"
            for name in (
                "performance_gate_eligible",
                "stage_b_selection_used",
                "final_unknown_used",
                "even_angle_test_used",
            )
        ):
            raise DataValidationError("official post-hoc row violates evidence scope")
        official_by_unit[key].append(row)
    expected_metric_rows: list[dict[str, Any]] = []
    for pair_id in PAIR_IDS:
        for method in METHODS:
            key = f"{pair_id}/{method}"
            audit = official_audits[key]
            if not isinstance(audit, Mapping) or audit.get("status") not in {
                "passed",
                "failed",
            }:
                raise DataValidationError(f"official post-hoc status changed for {key}")
            if (
                audit.get("pair_id") != pair_id
                or audit.get("method") != method
                or audit.get("performance_gate_eligible") is not False
                or audit.get("stage_b_model_or_gate_modified") is not False
                or audit.get("stage_b_selection_used") is not False
                or audit.get("final_unknown_used") is not False
                or audit.get("even_angle_test_used") is not False
            ):
                raise DataValidationError(
                    f"official post-hoc evidence contract changed for {key}"
                )
            rows = official_by_unit[key]
            if audit["status"] == "failed":
                if rows or "failure_reason" not in audit:
                    raise DataValidationError(f"failed post-hoc unit has rows for {key}")
                continue
            train_indices = np.asarray(
                [
                    index
                    for index, base in enumerate(base_rows[pair_id])
                    if base["experiment_role"] == "train_known"
                ],
                dtype=np.int64,
            )
            expected_template_counts = Counter(
                np.asarray(legacy_by_unit[key]["probabilities"])[train_indices]
                .argmax(axis=1)
                .tolist()
            )
            if list(audit.get("template_prediction_counts", [])) != [
                expected_template_counts[index] for index in range(5)
            ]:
                raise DataValidationError(
                    f"post-hoc template prediction counts changed for {key}"
                )
            if (
                audit.get("score_rules") != ["S1", "S2", "S3", "full"]
                or audit.get("pair_probability_prediction")
                != "old_D1_or_D2_pcssr_mean_view_argmax"
                or not all(
                    _is_sha256(audit.get(name))
                    for name in (
                        "template_first_order_sha256",
                        "template_gram_sha256",
                        "augmented_u_sha256",
                        "augmented_logits_sha256",
                        "augmented_probabilities_sha256",
                    )
                )
            ):
                raise DataValidationError(
                    f"post-hoc template or score identity changed for {key}"
                )
            if len(rows) != 3500 or Counter(
                row["evaluation_role"] for row in rows
            ) != Counter({"known_calibration": 2500, "surrogate_unknown": 1000}):
                raise DataValidationError(f"official post-hoc rows are incomplete for {key}")
            normalization = audit.get("normalization")
            if not isinstance(normalization, Mapping):
                raise DataValidationError(f"post-hoc normalization is absent for {key}")
            mean = np.asarray(normalization.get("mean"), dtype=np.float64)
            std = np.asarray(normalization.get("std"), dtype=np.float64)
            if (
                mean.shape != (3,)
                or std.shape != (3,)
                or not np.isfinite(mean).all()
                or not np.isfinite(std).all()
                or np.any(std <= 1.0e-12)
                or normalization.get("population_ddof") != 0
                or normalization.get("epsilon") != 1.0e-8
                or normalization.get("minimum_std") != 1.0e-12
                or normalization.get("augmented_train_base_count") != 2880
                or not _is_sha256(normalization.get("raw_score_sha256"))
            ):
                raise DataValidationError(
                    f"post-hoc normalization contract changed for {key}"
                )
            expected_pair_rows = pair_rows_by_unit[key]
            sample_index = base_indices[pair_id]
            probabilities = np.asarray(legacy_by_unit[key]["probabilities"])
            for index, row in enumerate(rows):
                source = expected_pair_rows[index]
                if any(
                    str(row[name]) != str(source[name])
                    for name in (
                        "evaluation_pair_id",
                        "evaluation_role",
                        "class_name",
                        "true_label",
                        "view1_sample_id",
                        "view2_sample_id",
                    )
                ):
                    raise DataValidationError(
                        f"official post-hoc row is not pair-aligned for {key}"
                    )
                expected_prediction = int(
                    (
                        probabilities[sample_index[str(row["view1_sample_id"])]]
                        + probabilities[sample_index[str(row["view2_sample_id"])]]
                    ).argmax()
                )
                if int(row["pcssr_pair_predicted_label"]) != expected_prediction:
                    raise DataValidationError(
                        f"official post-hoc prediction changed for {key}"
                    )
                components: list[float] = []
                for component_index, name in enumerate(("s1", "s2", "s3")):
                    raw_values = np.asarray(
                        [float(row[f"view{view}_raw_{name}"]) for view in (1, 2)]
                    )
                    standardized = np.asarray(
                        [
                            float(row[f"view{view}_standardized_{name}"])
                            for view in (1, 2)
                        ]
                    )
                    expected_standardized = (
                        raw_values - mean[component_index]
                    ) / (std[component_index] + 1.0e-8)
                    if not np.allclose(
                        standardized,
                        expected_standardized,
                        rtol=1.0e-9,
                        atol=1.0e-11,
                    ):
                        raise DataValidationError(
                            f"official post-hoc standardization changed for {key}"
                        )
                    component = float(row[f"pair_standardized_{name}"])
                    _close(
                        component,
                        standardized.mean(),
                        context=f"post-hoc pair {name}",
                        rtol=1.0e-9,
                        atol=1.0e-11,
                    )
                    _close(
                        row[f"unknown_score_{name}"],
                        -component,
                        context=f"post-hoc unknown {name}",
                        rtol=1.0e-9,
                        atol=1.0e-11,
                    )
                    components.append(component)
                _close(
                    row["unknown_score_full"],
                    -sum(components),
                    context="post-hoc full unknown score",
                    rtol=1.0e-9,
                    atol=1.0e-11,
                )
            predictions = np.asarray(
                [int(row["pcssr_pair_predicted_label"]) for row in rows],
                dtype=np.int64,
            )
            if np.any(predictions < 0) or np.any(predictions >= 5):
                raise DataValidationError(f"post-hoc prediction is outside 0..4 for {key}")
            identity_order = tuple(
                dict.fromkeys(
                    row["class_name"]
                    for row in rows
                    if row["evaluation_role"] == "surrogate_unknown"
                )
            )
            if len(identity_order) != 2:
                raise DataValidationError(f"post-hoc surrogate identities changed for {key}")
            unit_stub = type("StageAOfficialUnit", (), {})()
            unit_stub.pair_id = pair_id
            unit_stub.method = method
            current_metrics: list[dict[str, Any]] = []
            for rule, field in (
                ("S1", "unknown_score_s1"),
                ("S2", "unknown_score_s2"),
                ("S3", "unknown_score_s3"),
                ("full", "unknown_score_full"),
            ):
                values = np.asarray([float(row[field]) for row in rows])
                current_metrics.append(
                    _official_metric_row(
                        unit=unit_stub,
                        pair_rows=rows,
                        predictions=predictions,
                        unknown_score=values,
                        rule=rule,
                        scope="pair",
                        identity="",
                    )
                )
                for identity in identity_order:
                    current_metrics.append(
                        _official_metric_row(
                            unit=unit_stub,
                            pair_rows=rows,
                            predictions=predictions,
                            unknown_score=values,
                            rule=rule,
                            scope="identity",
                            identity=identity,
                        )
                    )
            if list(audit.get("metric_rows", [])) != current_metrics:
                raise DataValidationError(f"post-hoc metrics do not reproduce for {key}")
            expected_metric_rows.extend(current_metrics)
    if mechanism.get("official_posthoc_metric_rows") != expected_metric_rows:
        raise DataValidationError("aggregated official post-hoc metrics changed")

    unit_artifacts = mechanism.get("unit_artifacts")
    alignment = mechanism.get("d1_d2_method_independent_input_alignment")
    if (
        mechanism.get("unit_count") != 6
        or not isinstance(unit_artifacts, Mapping)
        or set(unit_artifacts) != unit_keys
        or not isinstance(alignment, Mapping)
        or alignment.get("status") != "passed"
        or set(alignment.get("by_pair", {})) != set(PAIR_IDS)
    ):
        raise DataValidationError("Stage-A unit or alignment population changed")
    for pair_id in PAIR_IDS:
        aligned = alignment["by_pair"][pair_id]
        if (
            int(aligned.get("unique_base_count", -1)) != 972
            or int(aligned.get("evaluation_pair_count", -1)) != 3500
            or not all(
                aligned.get(name) is True
                for name in (
                    "unique_base_manifest_exact",
                    "unique_feature_map_exact",
                    "evaluation_pair_manifest_exact",
                    "d0_prediction_rows_exact",
                    "source_pair_manifest_sha256_exact",
                )
            )
        ):
            raise DataValidationError(f"D1/D2 alignment changed for {pair_id}")
    for pair_id in PAIR_IDS:
        for method in METHODS:
            key = f"{pair_id}/{method}"
            record = unit_artifacts[key]
            if (
                not isinstance(record, Mapping)
                or record.get("checkpoint_sha256")
                != EXPECTED_CHECKPOINT_SHA256[(pair_id, method)]
                or record.get("artifact_manifest_sha256")
                != legacy_by_unit[key]["artifact_manifest_sha256"]
                or record.get("checkpoint_strict_load") is not True
                or record.get("checkpoint_replay") != "bitwise_exact"
                or record.get("saved_prediction_recomputation") != "exact"
            ):
                raise DataValidationError(f"Stage-A unit artifact record changed for {key}")
    _assert_finite_json_tree(mechanism, context="mechanism_audit")


def audit_mechanism_output(
    output_root: str | Path,
    *,
    legacy_pilot_root: str | Path,
    expected_phase_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    root = Path(output_root).resolve(strict=True)
    require_disjoint_output((legacy_pilot_root,), root)
    if {path.name for path in root.iterdir() if path.is_file()} != set(OUTPUT_FILENAMES):
        raise DataValidationError("mechanism output file set changed")
    if _read_json(root / "artifact_hashes.json") != _tree_artifact_hashes(root):
        raise DataValidationError("mechanism output artifact hashes changed")
    audit = _read_json(root / "mechanism_audit.json")
    if (
        audit.get("status") != "complete"
        or audit.get("experiment_id") != EXPERIMENT_ID
        or audit.get("stage_a_training_performed") is not False
        or audit.get("performance_gate_eligible") is not False
        or audit.get("stage_b_configuration_modified") is not False
        or audit.get("stage_b_selection_used") is not False
        or audit.get("immutable_input_tree_snapshot_unchanged") is not True
        or audit.get("confirmation_allowed") is not False
        or audit.get("automatic_followon_authorized") is not False
        or audit.get("final_unknown_test_authorized") is not False
        or audit.get("final_unknown_used") is not False
        or audit.get("even_angle_test_used") is not False
    ):
        raise DataValidationError("mechanism audit contract changed")
    _verify_stage_a_source_record(audit.get("source_binding"))
    runtime_contract = audit.get("runtime_contract")
    if not isinstance(runtime_contract, Mapping):
        raise DataValidationError("Stage-A runtime contract is absent")
    _validate_stage_a_runtime_contract(runtime_contract)
    augmentation_audits = audit.get("augmentation_audits")
    if not isinstance(augmentation_audits, Mapping) or set(augmentation_audits) != set(
        PAIR_IDS
    ):
        raise DataValidationError("Stage-A data-access audits are incomplete")
    for pair_id in PAIR_IDS:
        entry = augmentation_audits[pair_id]
        access = entry.get("data_access_audit") if isinstance(entry, Mapping) else None
        normalization = entry.get("normalization") if isinstance(entry, Mapping) else None
        augmentation = entry.get("augmentation") if isinstance(entry, Mapping) else None
        if (
            not isinstance(access, Mapping)
            or access.get("status") != "passed"
            or access.get("policy")
            != "enforced_source_known_odd_index_allowlist_v1"
            or access.get("authorized_row_count") != 7 * 180
            or access.get("profile_values_materialized_only_through_allowlist")
            is not True
            or access.get("final_unknown_profile_values_read") is not False
            or access.get("even_angle_profile_values_read") is not False
            or access.get("final_unknown_pairs_generated") is not False
            or access.get("even_angle_test_pairs_generated") is not False
        ):
            raise DataValidationError(
                f"Stage-A development-only access contract changed for {pair_id}"
            )
        if (
            entry.get("status") != "passed"
            or entry.get("pair_id") != pair_id
            or entry.get("raw_z_replay_exact") is not True
            or entry.get("surrogate_unknown_used") is not False
            or entry.get("known_calibration_used") is not False
            or entry.get("final_unknown_used") is not False
            or entry.get("even_angle_test_used") is not False
            or not _is_sha256(entry.get("raw_z_sha256"))
            or not _is_sha256(entry.get("augmented_z_sha256"))
            or not isinstance(normalization, Mapping)
            or normalization.get("method")
            != "reuse_exact_r2_global_scalar_zscore"
            or normalization.get("fitted_from_train_known_only") is not True
            or normalization.get("unique_base_sample_count") != 720
            or normalization.get("epsilon") != 1.0e-8
            or not math.isfinite(float(normalization.get("mean", math.nan)))
            or not math.isfinite(float(normalization.get("std", math.nan)))
            or float(normalization.get("std", 0.0)) <= 0.0
            or not isinstance(augmentation, Mapping)
            or augmentation.get("status") != "passed"
            or augmentation.get("family")
            != "gain_uniform_0.9_1.1_plus_gaussian_std_0.02"
            or augmentation.get("seed") != SCORE_NORM_SEED
            or augmentation.get("pair_id") != pair_id
            or augmentation.get("variant_count_per_base") != 4
            or augmentation.get("method_id_in_seed_material") is not False
            or augmentation.get("final_unknown_used") is not False
            or augmentation.get("even_angle_test_used") is not False
            or not all(
                _is_sha256(augmentation.get(name))
                for name in (
                    "sample_variant_ids_sha256",
                    "gain_sha256",
                    "noise_sha256",
                    "augmented_input_sha256",
                )
            )
        ):
            raise DataValidationError(
                f"Stage-A score-normalization augmentation changed for {pair_id}"
            )
    phase_kwargs = (
        {}
        if expected_phase_hashes is None
        else {
            "expected_manifest_sha256": expected_phase_hashes["manifest"],
            "expected_summary_sha256": expected_phase_hashes["summary"],
            "expected_gate_sha256": expected_phase_hashes["gate"],
            "expected_success_sha256": expected_phase_hashes["success"],
        }
    )
    phase = verify_sealed_pilot_root(legacy_pilot_root, **phase_kwargs)
    if audit.get("sealed_pilot") != phase:
        raise DataValidationError("mechanism output is not bound to the sealed pilot")
    _audit_mechanism_payload(root, audit, legacy_pilot_root)
    return {
        "status": "passed",
        "root": str(root),
        "artifact_count": len(_read_json(root / "artifact_hashes.json")),
        "performance_gate_eligible": False,
        "confirmation_allowed": False,
        "final_unknown_test_authorized": False,
    }


def run_identity_failure_audit(
    *,
    legacy_pilot_root: str | Path,
    legacy_config_path: str | Path,
    bundle_root: str | Path,
    r2_results_root: str | Path,
    output_root: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    """Run the complete preregistered Stage-A analysis without training."""

    if device.type != "cuda":
        raise DataValidationError("formal Stage-A checkpoint replay requires CUDA")
    project_root = Path(__file__).resolve().parents[3]
    legacy_config = Path(legacy_config_path).resolve(strict=True)
    if not legacy_config.is_file() or file_sha256(legacy_config) != LEGACY_CONFIG_SHA256:
        raise DataValidationError("sealed D1/D2 configuration hash changed")
    from hrrp_osr.training.arpl_pilot import _set_determinism
    from hrrp_osr.training.fg_mv_cssr_decoupled import (
        _configure_runtime as _configure_legacy_runtime,
        load_fg_mv_cssr_decoupled_config,
    )

    legacy_runtime_config = load_fg_mv_cssr_decoupled_config(legacy_config)
    _set_determinism(LEGACY_SEED, True)
    _configure_legacy_runtime(legacy_runtime_config, device)
    runtime_contract = _current_stage_a_runtime_contract(device)
    _validate_stage_a_runtime_contract(runtime_contract)
    source_binding = _stage_a_source_record(device)
    immutable_roots = (legacy_pilot_root, bundle_root, r2_results_root)
    destination = require_disjoint_output(immutable_roots, output_root)
    if destination.exists():
        raise DataValidationError(f"mechanism output already exists: {destination}")
    pilot = Path(legacy_pilot_root).resolve(strict=True)
    before = {str(Path(root).resolve(strict=True)): _tree_snapshot(Path(root).resolve(strict=True)) for root in immutable_roots}
    sealed_pilot = verify_sealed_pilot_root(pilot)
    units = {
        (pair_id, method): load_verified_legacy_unit(
            pilot, pair_id=pair_id, method=method, device=device
        )
        for pair_id in PAIR_IDS
        for method in METHODS
    }
    shared_input_alignment = verify_d1_d2_pair_alignment(units)
    pair_rows: list[dict[str, Any]] = []
    cross_raw: list[dict[str, Any]] = []
    cross_normalized: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    identity_diagnostics: list[dict[str, Any]] = []
    specificity: dict[str, Any] = {}
    geometry: dict[str, Any] = {}
    official_rows: list[dict[str, Any]] = []
    official_metrics: list[dict[str, Any]] = []
    official_audits: dict[str, Any] = {}
    augmentation_audits: dict[str, Any] = {}
    for pair_id in PAIR_IDS:
        representative = units[(pair_id, METHODS[0])]
        augmented_z, augmentation_audits[pair_id] = _prepare_augmented_z(
            pair_id=pair_id,
            representative=representative,
            legacy_config_path=legacy_config,
            bundle_root=bundle_root,
            r2_results_root=r2_results_root,
            device=device,
        )
        for method in METHODS:
            unit = units[(pair_id, method)]
            current_pair_rows = build_pair_view_decomposition(unit)
            pair_rows.extend(current_pair_rows)
            identity_diagnostics.extend(identity_view_diagnostics(current_pair_rows))
            raw_rows, normalized_rows = ae_cross_reconstruction_rows(unit)
            cross_raw.extend(raw_rows)
            cross_normalized.extend(normalized_rows)
            reference_rows.extend(reference_distribution_rows(unit))
            specificity[f"{pair_id}/{method}"] = ae_specificity_summary(unit)
            geometry[f"{pair_id}/{method}"] = representation_geometry(
                unit.z, unit.decomposition.adapted_features, unit.unique_rows
            )
            augmented = build_augmented_unit_arrays(
                unit, augmented_z, device=device, batch_size=128
            )
            detailed, metrics, score_audit = guarded_official_posthoc_score_rows(
                unit, augmented, device=device, batch_size=128
            )
            official_rows.extend(detailed)
            official_metrics.extend(metrics)
            official_audits[f"{pair_id}/{method}"] = score_audit

    # Recheck immutable inputs after every forward and statistic, before the
    # first output byte is created.
    verify_sealed_pilot_root(pilot)
    after = {str(Path(root).resolve(strict=True)): _tree_snapshot(Path(root).resolve(strict=True)) for root in immutable_roots}
    if before != after:
        raise DataValidationError("an immutable Stage-A input tree changed during analysis")
    end_source_binding = _stage_a_source_record(device)
    if end_source_binding != source_binding:
        raise DataValidationError("checkout, environment, or source changed during Stage A")
    end_runtime_contract = _current_stage_a_runtime_contract(device)
    _validate_stage_a_runtime_contract(end_runtime_contract)
    if end_runtime_contract != runtime_contract:
        raise DataValidationError("Stage-A CUDA runtime changed during analysis")
    mechanism_audit = {
        "status": "complete",
        "experiment_id": EXPERIMENT_ID,
        "source_binding": source_binding,
        "runtime_contract": runtime_contract,
        "sealed_pilot": sealed_pilot,
        "unit_count": len(units),
        "d1_d2_method_independent_input_alignment": shared_input_alignment,
        "unit_artifacts": {
            f"{pair_id}/{method}": {
                "checkpoint_sha256": unit.checkpoint_sha256,
                "artifact_manifest_sha256": unit.artifact_manifest_sha256,
                "checkpoint_strict_load": True,
                "checkpoint_replay": "bitwise_exact",
                "saved_prediction_recomputation": "exact",
            }
            for (pair_id, method), unit in units.items()
        },
        "identity_view_diagnostics": identity_diagnostics,
        "augmentation_audits": augmentation_audits,
        "official_posthoc_audits": official_audits,
        "official_posthoc_metric_rows": official_metrics,
        "immutable_input_tree_snapshot_unchanged": True,
        "stage_a_training_performed": False,
        "performance_gate_eligible": False,
        "stage_b_configuration_modified": False,
        "stage_b_selection_used": False,
        "confirmation_allowed": False,
        "automatic_followon_authorized": False,
        "final_unknown_test_authorized": False,
        "final_unknown_used": False,
        "even_angle_test_used": False,
    }
    saved = save_mechanism_outputs(
        destination,
        input_roots=immutable_roots,
        base_arrays=_base_npz(units),
        pair_rows=pair_rows,
        cross_raw_rows=cross_raw,
        cross_normalized_rows=cross_normalized,
        specificity=specificity,
        reference_rows=reference_rows,
        geometry=geometry,
        official_rows=official_rows,
        mechanism_audit=mechanism_audit,
    )
    result = audit_mechanism_output(saved, legacy_pilot_root=pilot)
    if before != {str(Path(root).resolve(strict=True)): _tree_snapshot(Path(root).resolve(strict=True)) for root in immutable_roots}:
        raise DataValidationError("immutable Stage-A inputs changed while saving outputs")
    return result


def _parse_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise DataValidationError("formal Stage-A mechanism audit requires an available CUDA device")
    return device


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only decoupled CSSR identity audit")
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify-input")
    verify.add_argument("--legacy-pilot-root", required=True)
    run = commands.add_parser("run")
    run.add_argument("--legacy-pilot-root", required=True)
    run.add_argument("--legacy-config", required=True)
    run.add_argument("--bundle-root", required=True)
    run.add_argument("--r2-results-root", required=True)
    run.add_argument("--output-root", required=True)
    run.add_argument("--device", default="cuda")
    audit = commands.add_parser("audit")
    audit.add_argument("--legacy-pilot-root", required=True)
    audit.add_argument("--output-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "verify-input":
        result = verify_sealed_pilot_root(arguments.legacy_pilot_root)
    elif arguments.command == "run":
        result = run_identity_failure_audit(
            legacy_pilot_root=arguments.legacy_pilot_root,
            legacy_config_path=arguments.legacy_config,
            bundle_root=arguments.bundle_root,
            r2_results_root=arguments.r2_results_root,
            output_root=arguments.output_root,
            device=_parse_device(arguments.device),
        )
    elif arguments.command == "audit":
        result = audit_mechanism_output(
            arguments.output_root,
            legacy_pilot_root=arguments.legacy_pilot_root,
        )
    else:  # pragma: no cover
        raise AssertionError("unreachable")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
