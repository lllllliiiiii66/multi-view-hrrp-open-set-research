from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import yaml
from torch import nn

from hrrp_osr.data.errors import DataValidationError
from hrrp_osr.data.manifest import file_sha256
from hrrp_osr.data.processed import (
    ProcessedBundle,
    _load_rows as _load_processed_rows,
    _read_single_hash as _read_processed_hash,
)
from hrrp_osr.evaluation.metrics import accuracy_score, macro_f1_score
from hrrp_osr.evaluation.ms_mean_factorial import REPORT_METRIC_KEYS
from hrrp_osr.evaluation.official_cssr_scores import (
    OfficialRawScores,
    OfficialScoreNormalization,
    OfficialScoreTemplates,
    build_official_score_templates,
    fit_score_normalization,
    matched_linear_pair_output,
    official_pcssr_pair_scores,
    raw_official_scores,
)
from hrrp_osr.evaluation.official_cssr_oracle import audit_official_cssr_oracle
from hrrp_osr.models.official_cssr_1d import (
    MATCHED_LINEAR_CONTROL_1D,
    OFFICIAL_SEMANTICS_PCSSR_1D,
    OfficialCSSRHRRPModel1D,
    OfficialPCSSRHeadOutput,
)
from hrrp_osr.training.arpl_pilot import _resolve_device, _set_determinism
from hrrp_osr.training.fg_mv_cssr_e2e_redesign import (
    _build_method_prediction_rows,
    _class_conditional_mls_for_roles,
    _evaluate_score_arrays,
    _metrics_exact,
    _normalization_record,
    _save_npz,
    build_identity_and_absorption_rows,
    recompute_method_metrics_from_prediction_rows,
)
from hrrp_osr.training.fg_mv_cssr_pilot import (
    _artifact_hashes,
    _array_sha256,
    _atomic_write_bytes,
    _load_prior_config,
    _prepare_frozen_split,
    _read_csv,
    _read_json,
    _role_manifest_rows,
    _sequence_sha256,
    _write_csv,
    _write_json,
    build_unique_base_sample_manifest,
    load_and_audit_frozen_r2,
)
from hrrp_osr.training.official_cssr_protocol import (
    O0_R2_CC_MLS,
    O1_OFFICIAL_LINEAR_FT,
    O2_OFFICIAL_PCSSR_FT,
    O3_OFFICIAL_LINEAR_E2E,
    O4_OFFICIAL_PCSSR_E2E,
    PILOT_PAIRS,
    TRAINABLE_METHODS,
    build_phase_plan,
    build_score_norm_augmentation,
    build_training_epoch_material,
    evaluate_pilot_gate,
    learning_rates_for_update,
    load_official_cssr_config,
    model_initialization_seed,
)


EXPERIMENT_ID = "official_cssr_hrrp_pilot_v1"
CONFIG_RELATIVE_PATH = "configs/experiments/cssr/official_cssr_hrrp_pilot_v1.yaml"
ANGLE_FOLD = 0
R2_SEED = 20260830
OFFICIAL_CSSR_SEED = 20260906
CSSR_METHODS = (O2_OFFICIAL_PCSSR_FT, O4_OFFICIAL_PCSSR_E2E)
LINEAR_METHODS = (O1_OFFICIAL_LINEAR_FT, O3_OFFICIAL_LINEAR_E2E)
E2E_METHODS = (O3_OFFICIAL_LINEAR_E2E, O4_OFFICIAL_PCSSR_E2E)
SCORE_VARIANTS = (
    "s1",
    "s2",
    "s3",
    "s1_s2",
    "s1_s3",
    "s2_s3",
    "full",
    "max_pair_probability",
)
TASK_SOURCE_FILES = (
    ".gitignore",
    CONFIG_RELATIVE_PATH,
    "pyproject.toml",
    "src/hrrp_osr/__init__.py",
    "src/hrrp_osr/amdr/__init__.py",
    "src/hrrp_osr/amdr/data.py",
    "src/hrrp_osr/amdr/model.py",
    "src/hrrp_osr/amdr/reduction.py",
    "src/hrrp_osr/amdr/smoke.py",
    "src/hrrp_osr/baselines/__init__.py",
    "src/hrrp_osr/baselines/b0.py",
    "src/hrrp_osr/data/__init__.py",
    "src/hrrp_osr/data/config.py",
    "src/hrrp_osr/data/errors.py",
    "src/hrrp_osr/data/manifest.py",
    "src/hrrp_osr/data/padding.py",
    "src/hrrp_osr/data/processed.py",
    "src/hrrp_osr/data/protocol.py",
    "src/hrrp_osr/data/sets.py",
    "src/hrrp_osr/evaluation/__init__.py",
    "src/hrrp_osr/evaluation/aggregate.py",
    "src/hrrp_osr/evaluation/metrics.py",
    "src/hrrp_osr/evaluation/ms_mean_factorial.py",
    "src/hrrp_osr/evaluation/official_cssr_oracle.py",
    "src/hrrp_osr/evaluation/official_cssr_scores.py",
    "src/hrrp_osr/models/__init__.py",
    "src/hrrp_osr/models/arpl.py",
    "src/hrrp_osr/models/cnn1d.py",
    "src/hrrp_osr/models/cssr_1d.py",
    "src/hrrp_osr/models/cssr_decoupled_1d.py",
    "src/hrrp_osr/models/cssr_e2e_1d.py",
    "src/hrrp_osr/models/hrrp_ms_resnet.py",
    "src/hrrp_osr/models/ms_mean_factorial.py",
    "src/hrrp_osr/models/mv_rpformer.py",
    "src/hrrp_osr/models/official_cssr_1d.py",
    "src/hrrp_osr/models/sets.py",
    "src/hrrp_osr/training/__init__.py",
    "src/hrrp_osr/training/arpl_mv_evidence.py",
    "src/hrrp_osr/training/arpl_pilot.py",
    "src/hrrp_osr/training/b0_smoke.py",
    "src/hrrp_osr/training/cssr_gradient_pathology_audit.py",
    "src/hrrp_osr/training/cssr_identity_failure_audit.py",
    "src/hrrp_osr/training/fg_mv_cssr_decoupled.py",
    "src/hrrp_osr/training/fg_mv_cssr_decoupled_protocol.py",
    "src/hrrp_osr/training/fg_mv_cssr_e2e_redesign.py",
    "src/hrrp_osr/training/fg_mv_cssr_pilot.py",
    "src/hrrp_osr/training/ms_mean_head_factorial.py",
    "src/hrrp_osr/training/mv_rpformer.py",
    "src/hrrp_osr/training/official_cssr_protocol.py",
    "src/hrrp_osr/training/official_cssr_hrrp_pilot.py",
)
PHASE_AGGREGATE_FILES = (
    "task_plan.json",
    "task_audit.csv",
    "metrics_by_pair.csv",
    "identity_metrics.csv",
    "identity_metrics.json",
    "absorption_by_known_class.csv",
    "absorption_by_known_class.json",
    "score_ablation_metrics.csv",
    "score_ablation_metrics.json",
    "phase_integrity_audit.json",
    "pilot_gate_input.json",
    "pilot_gate.json",
    "pre_registered_comparisons.json",
    "phase_summary.json",
    "artifact_hashes.json",
    "_PHASE_SUCCESS.json",
    "_PHASE_INCOMPLETE.json",
)
TASK_AUDIT_FIELDS = (
    "pair_id",
    "method",
    "status",
    "audit_passed",
    "checkpoint_replay",
    "artifact_count",
    "checkpoint_sha256",
    "failure_type",
    "failure_message",
)


class _ProtocolRestrictedProfiles:
    """Expose only preregistered development profile values from a memory map.

    The complete manifest and opaque file hash remain available for provenance,
    but any attempt to index an even-angle or final-unknown profile fails before
    NumPy can materialize that value.
    """

    def __init__(self, values: np.ndarray, allowed_row_indices: Iterable[int]) -> None:
        self._values = values
        self._allowed = frozenset(int(index) for index in allowed_row_indices)
        self.shape = values.shape
        self.dtype = values.dtype
        self.ndim = values.ndim
        if not self._allowed:
            raise DataValidationError("development profile allowlist is empty")
        if min(self._allowed) < 0 or max(self._allowed) >= int(values.shape[0]):
            raise DataValidationError("development profile allowlist is out of bounds")

    @property
    def allowed_row_indices(self) -> tuple[int, ...]:
        return tuple(sorted(self._allowed))

    def _selected_rows(self, key: Any) -> np.ndarray:
        row_key = key[0] if isinstance(key, tuple) else key
        if row_key is Ellipsis or row_key is None:
            raise DataValidationError(
                "whole-array profile access is forbidden by the preregistered protocol"
            )
        try:
            selected = np.arange(int(self.shape[0]), dtype=np.int64)[row_key]
        except (IndexError, TypeError, ValueError) as exc:
            raise DataValidationError("invalid restricted profile row selector") from exc
        return np.asarray(selected, dtype=np.int64).reshape(-1)

    def __getitem__(self, key: Any) -> np.ndarray:
        selected = self._selected_rows(key)
        forbidden = sorted(set(int(index) for index in selected) - self._allowed)
        if forbidden:
            raise DataValidationError(
                "profile-value access outside source-known odd-angle allowlist: "
                f"{forbidden[:5]}"
            )
        return self._values[key]

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        del dtype, copy
        raise DataValidationError(
            "whole-array profile materialization is forbidden by the preregistered protocol"
        )


def _load_development_only_bundle(
    bundle_root: str | Path,
    config: Mapping[str, Any],
) -> ProcessedBundle:
    """Load provenance while exposing only source-known odd-angle profiles."""

    root = Path(bundle_root).expanduser().resolve()
    profiles_path = root / "profiles.npy"
    manifest_path = root / "samples.csv"
    if not profiles_path.is_file() or not manifest_path.is_file():
        raise DataValidationError("processed HRRP bundle is incomplete")
    recorded_profiles_hash = _read_processed_hash(
        root / "profiles.npy.sha256", "profiles.npy"
    )
    recorded_manifest_hash = _read_processed_hash(
        root / "samples.csv.sha256", "samples.csv"
    )
    recorded_bundle_hash = _read_processed_hash(
        root / "bundle.sha256", "hrrp_padding_complex_gaussian_v1"
    )
    profiles_hash = file_sha256(profiles_path)
    manifest_hash = file_sha256(manifest_path)
    expected = config["bundle"]
    if (
        profiles_hash != recorded_profiles_hash
        or profiles_hash != str(expected["profiles_sha256"])
        or manifest_hash != recorded_manifest_hash
        or manifest_hash != str(expected["manifest_sha256"])
        or recorded_bundle_hash != str(expected["bundle_sha256"])
    ):
        raise DataValidationError("processed HRRP bundle hash binding changed")

    values = np.load(profiles_path, mmap_mode="r", allow_pickle=False)
    rows = _load_processed_rows(manifest_path)
    if values.shape != (3600, 601) or values.dtype != np.dtype("float64"):
        raise DataValidationError("processed HRRP profile shape or dtype changed")
    if len(rows) != 3600:
        raise DataValidationError("processed HRRP manifest row count changed")
    row_indices = [int(row["processed_row_index"]) for row in rows]
    if row_indices != list(range(3600)):
        raise DataValidationError("processed HRRP manifest row order changed")
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise DataValidationError("processed HRRP sample ids are not unique")

    source_known = tuple(str(name) for name in config["classes"]["source_known_order"])
    known_names = {
        str(row["class_name"])
        for row in rows
        if str(row["class_role"]) == "known"
    }
    unknown_names = {
        str(row["class_name"])
        for row in rows
        if str(row["class_role"]) == "unknown"
    }
    if known_names != set(source_known) or len(unknown_names) != 3:
        raise DataValidationError("processed HRRP 7-known/3-unknown roles changed")
    if any(
        int(row["eligible_for_training"])
        or int(row["eligible_for_validation"])
        for row in rows
        if str(row["class_role"]) == "unknown"
    ):
        raise DataValidationError("final unknown metadata permits development use")

    development_rows = tuple(
        row
        for row in rows
        if str(row["class_role"]) == "known"
        and str(row["class_name"]) in known_names
        and int(row["angle_deg"]) % 2 == 1
    )
    allowed = tuple(int(row["processed_row_index"]) for row in development_rows)
    if len(development_rows) != 7 * 180 or len(set(allowed)) != len(allowed):
        raise DataValidationError("source-known odd-angle allowlist population changed")
    for class_name in source_known:
        angles = sorted(
            int(row["angle_deg"])
            for row in development_rows
            if str(row["class_name"]) == class_name
        )
        if angles != list(range(1, 360, 2)):
            raise DataValidationError(
                f"source-known odd-angle coverage changed for {class_name}"
            )

    restricted = _ProtocolRestrictedProfiles(values, allowed)
    authorized_values = np.asarray(
        restricted[np.asarray(allowed, dtype=np.int64)], dtype=np.float64
    )
    if not np.isfinite(authorized_values).all():
        raise DataValidationError("authorized development profiles contain NaN or Inf")
    return ProcessedBundle(
        root=root,
        profiles=restricted,  # type: ignore[arg-type]
        rows=development_rows,
        profiles_sha256=profiles_hash,
        manifest_sha256=manifest_hash,
        bundle_sha256=recorded_bundle_hash,
    )


def _profile_access_audit(bundle: ProcessedBundle) -> dict[str, Any]:
    profiles = bundle.profiles
    if not isinstance(profiles, _ProtocolRestrictedProfiles):
        raise DataValidationError("development profile access guard is absent")
    allowed = profiles.allowed_row_indices
    if len(allowed) != 7 * 180:
        raise DataValidationError("development profile access allowlist changed")
    return {
        "status": "passed",
        "policy": "enforced_source_known_odd_index_allowlist_v1",
        "full_manifest_metadata_read_for_integrity": True,
        "full_profile_file_hashed_as_opaque_bytes": True,
        "profile_values_materialized_only_through_allowlist": True,
        "authorized_class_role": "source_known",
        "authorized_angle_parity": "odd",
        "authorized_row_count": len(allowed),
        "authorized_row_indices_sha256": _sequence_sha256(allowed),
        "authorized_values_finite_checked": True,
        "final_unknown_profile_values_read": False,
        "even_angle_profile_values_read": False,
        "final_unknown_pairs_generated": False,
        "even_angle_test_pairs_generated": False,
    }


def _state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _rng_state_record(device: torch.device) -> dict[str, Any]:
    """Serialize the post-initialization RNG state without host-local paths."""

    cpu_state = torch.get_rng_state().detach().cpu().contiguous().numpy().tobytes()
    result: dict[str, Any] = {
        "torch_cpu_state_hex": cpu_state.hex(),
        "torch_cpu_state_sha256": hashlib.sha256(cpu_state).hexdigest(),
    }
    if device.type == "cuda":
        cuda_state = (
            torch.cuda.get_rng_state(device)
            .detach()
            .cpu()
            .contiguous()
            .numpy()
            .tobytes()
        )
        result.update(
            {
                "torch_cuda_state_hex": cuda_state.hex(),
                "torch_cuda_state_sha256": hashlib.sha256(cuda_state).hexdigest(),
            }
        )
    else:
        result.update(
            {
                "torch_cuda_state_hex": None,
                "torch_cuda_state_sha256": None,
            }
        )
    return result


def _parameter_vector(parameters: Iterable[nn.Parameter]) -> torch.Tensor:
    values = [
        parameter.detach().reshape(-1).cpu().to(torch.float64)
        for parameter in parameters
    ]
    return torch.cat(values) if values else torch.zeros(0, dtype=torch.float64)


def _buffer_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.named_buffers()
    }


def _relative_change(before: torch.Tensor, after: torch.Tensor) -> float:
    if before.shape != after.shape:
        raise DataValidationError("parameter vector shape changed")
    denominator = max(float(torch.linalg.vector_norm(before)), 1.0e-12)
    return float(torch.linalg.vector_norm(after - before) / denominator)


def _mapping_to_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _mapping_to_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_mapping_to_json(item) for item in value]
    return value


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        _mapping_to_json(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _official_audit_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the host-path-independent identity of a passing oracle record."""

    dtype_checks = record.get("dtype_checks")
    if not isinstance(dtype_checks, Mapping):
        raise DataValidationError("official differential dtype records are absent")
    portable_checks: dict[str, Any] = {}
    for name in ("float32", "float64"):
        check = dtype_checks.get(name)
        if not isinstance(check, Mapping):
            raise DataValidationError(f"official differential {name} record is absent")
        portable_checks[name] = {
            "passed": check.get("passed"),
            "dtype": check.get("dtype"),
            "rtol": check.get("rtol"),
            "atol": check.get("atol"),
            "clip_boundary_checks": check.get("clip_boundary_checks"),
            "pair_checks": check.get("pair_checks"),
            "deterministic_repeat": check.get("deterministic_repeat"),
        }
    return {
        "passed": record.get("passed"),
        "status": record.get("status"),
        "official_commit": record.get("official_commit"),
        "file_sha256": record.get("file_sha256"),
        "verified_file_sha256": record.get("verified_file_sha256"),
        "source_execution": record.get("source_execution"),
        "method_ids": record.get("method_ids"),
        "oracle_contract": record.get("oracle_contract"),
        "float32": record.get("float32"),
        "float64": record.get("float64"),
        "dtype_checks": portable_checks,
    }


def _r2_audit_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    """Remove runtime locations while retaining every frozen R2 evidence field."""

    return {
        str(key): _mapping_to_json(value)
        for key, value in record.items()
        if key not in {"unit_root", "checkpoint_path"}
    }


def _assert_finite_tree(value: Any, *, context: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite_tree(item, context=f"{context}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _assert_finite_tree(item, context=f"{context}[{index}]")
    elif isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise DataValidationError(f"{context} contains NaN or Inf")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise DataValidationError(f"{context} contains NaN or Inf")


def _task_source_hashes(project_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in TASK_SOURCE_FILES:
        path = project_root / relative
        if not path.is_file():
            raise DataValidationError(f"missing task source: {relative}")
        result[relative] = file_sha256(path)
    return result


def _bound_project_root(config: Mapping[str, Any]) -> Path:
    """Require the loaded runner and frozen config to come from one checkout."""

    module_root = Path(__file__).resolve().parents[3]
    config_path = Path(str(config.get("_config_path", ""))).resolve()
    expected_config_path = (module_root / CONFIG_RELATIVE_PATH).resolve()
    if config_path != expected_config_path:
        raise DataValidationError(
            "official CSSR runner and config must come from the same checkout"
        )
    return module_root


def _git_commit_source_hashes(
    project_root: Path,
    commit: str,
    relative_paths: Iterable[str],
) -> dict[str, str]:
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise DataValidationError("recorded source commit is not a full Git SHA")
    result: dict[str, str] = {}
    for relative in relative_paths:
        completed = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=project_root,
            check=False,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise DataValidationError(
                f"recorded source commit does not contain {relative}"
            )
        result[relative] = hashlib.sha256(completed.stdout).hexdigest()
    return result


def _git_environment(project_root: Path, device: torch.device) -> dict[str, Any]:
    def command(*parts: str) -> str:
        return subprocess.run(
            parts,
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "git_commit": command("git", "rev-parse", "HEAD"),
        "git_branch": command("git", "branch", "--show-current"),
        "git_status_porcelain": command("git", "status", "--porcelain"),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "cuda": torch.version.cuda,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "allow_tf32": False,
        "amp": False,
    }


def _configure_runtime(config: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise DataValidationError(
            "formal official CSSR requires CUBLAS_WORKSPACE_CONFIG=:4096:8"
        )
    runtime = config["runtime"]
    _set_determinism(
        OFFICIAL_CSSR_SEED,
        bool(runtime["deterministic_algorithms"]),
    )
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if device.type != "cuda":
        raise DataValidationError("formal official CSSR units require CUDA")
    expected = str(runtime["expected_gpu_model"])
    observed = torch.cuda.get_device_name(device)
    if observed != expected:
        raise DataValidationError(
            f"formal GPU changed: expected {expected!r}, observed {observed!r}"
        )
    return {
        "device": str(device),
        "device_name": observed,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _assert_runtime_contract_exact(
    recorded: Any,
    current: Mapping[str, Any],
) -> None:
    if not isinstance(recorded, Mapping) or dict(recorded) != dict(current):
        raise DataValidationError("official CSSR runtime contract does not replay exactly")


def _selected_rows(
    rows: Sequence[Mapping[str, Any]], role: str
) -> tuple[list[dict[str, Any]], np.ndarray]:
    indices = np.asarray(
        [index for index, row in enumerate(rows) if row["experiment_role"] == role],
        dtype=np.int64,
    )
    return [dict(rows[int(index)]) for index in indices], indices


def _evaluation_role_indices(
    prepared: Any,
    *,
    smoke: bool,
    config: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    if not smoke:
        return {
            role: np.arange(len(prepared.labels[role]), dtype=np.int64)
            for role in ("known_calibration", "surrogate_unknown")
        }
    per_identity = int(config["data"]["smoke"]["evaluation_pairs_per_class_or_identity"])
    selected_by_role: dict[str, np.ndarray] = {}
    for role in ("known_calibration", "surrogate_unknown"):
        rows = _role_manifest_rows(prepared, role)
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            grouped[str(row["class_name"])].append(index)
        expected_classes = (
            prepared.train_class_order
            if role == "known_calibration"
            else prepared.surrogate_class_order
        )
        if set(grouped) != set(expected_classes):
            raise DataValidationError("smoke pair class population changed")
        selected = [
            index
            for class_name in expected_classes
            for index in sorted(
                grouped[class_name], key=lambda value: str(rows[value]["pair_id"])
            )[:per_identity]
        ]
        if len(selected) != len(expected_classes) * per_identity:
            raise DataValidationError("insufficient pairs for smoke subset")
        selected_by_role[role] = np.asarray(selected, dtype=np.int64)
    return selected_by_role


def _fit_input_normalization(
    *,
    bundle: Any,
    unique_rows: Sequence[Mapping[str, Any]],
    prepared: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_rows, train_indices = _selected_rows(unique_rows, "train_known")
    if len(train_rows) != 720:
        raise DataValidationError("normalization population is not 720 train-known bases")
    raw_all = np.asarray(
        bundle.profiles[
            np.asarray(
                [int(row["processed_row_index"]) for row in unique_rows],
                dtype=np.int64,
            )
        ],
        dtype=np.float64,
    )
    raw_train = raw_all[train_indices]
    mean = float(np.mean(raw_train, dtype=np.float64))
    std = float(np.std(raw_train, ddof=0, dtype=np.float64))
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 1.0e-12:
        raise DataValidationError("train-known global scalar normalization is invalid")
    normalized = np.asarray((raw_all - mean) / (std + 1.0e-12), dtype=np.float32)
    if normalized.shape != (len(unique_rows), 601) or not np.isfinite(normalized).all():
        raise DataValidationError("normalized unique-base inputs are invalid")
    # This task refits the same train-only statistic independently.  Requiring
    # numerical identity proves that the old pair tensors can be reused without
    # silently changing the model input.
    if not (
        math.isclose(mean, float(prepared.normalization.mean), rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(std, float(prepared.normalization.std), rel_tol=0.0, abs_tol=1e-12)
    ):
        raise DataValidationError("new train-known normalization differs from frozen R2")
    return normalized, {
        "method": "global_scalar_zscore_train_known_population_v1",
        "fit_role": "train_known",
        "unique_base_sample_count": 720,
        "element_count": int(raw_train.size),
        "mean": mean,
        "std": std,
        "ddof": 0,
        "epsilon": 1.0e-12,
        "sample_id_order_sha256": _sequence_sha256(
            row["sample_id"] for row in train_rows
        ),
        "raw_train_sha256": _array_sha256(raw_train),
        "normalized_all_sha256": _array_sha256(normalized),
        "matches_frozen_r2": True,
        "known_calibration_used": False,
        "surrogate_unknown_used": False,
    }


def _materialize_pair_inputs(
    *,
    bundle: Any,
    rows: Sequence[Mapping[str, Any]],
    mean: float,
    std: float,
) -> np.ndarray:
    view1 = np.asarray(
        bundle.profiles[
            np.asarray([int(row["view1_row_index"]) for row in rows], dtype=np.int64)
        ],
        dtype=np.float64,
    )
    view2 = np.asarray(
        bundle.profiles[
            np.asarray([int(row["view2_row_index"]) for row in rows], dtype=np.int64)
        ],
        dtype=np.float64,
    )
    result = np.asarray(
        (np.stack((view1, view2), axis=1) - float(mean)) / (float(std) + 1.0e-12),
        dtype=np.float32,
    )
    if result.shape != (len(rows), 2, 601) or not np.isfinite(result).all():
        raise DataValidationError("normalized pair inputs are invalid")
    return result


def _model_head_kind(method: str) -> str:
    if method in CSSR_METHODS:
        return OFFICIAL_SEMANTICS_PCSSR_1D
    if method in LINEAR_METHODS:
        return MATCHED_LINEAR_CONTROL_1D
    raise DataValidationError(f"unsupported trainable method: {method}")


def _model_init_seed(config: Mapping[str, Any], pair_id: str, method: str) -> int:
    purpose = "pcssr" if method in CSSR_METHODS else "linear"
    return model_initialization_seed(purpose, pair_id, config)


def _build_model(
    *,
    method: str,
    pair_id: str,
    r2_model: Any,
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[OfficialCSSRHRRPModel1D, dict[str, Any]]:
    seed = _model_init_seed(config, pair_id, method)
    # The preregistered first-eight-byte seed is a full unsigned 64-bit PyTorch
    # seed.  Do not pass it through legacy np.random.seed (which only accepts
    # uint32); all NumPy schedules use independent PCG64 streams.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(
        bool(config["runtime"]["deterministic_algorithms"])
    )
    model = OfficialCSSRHRRPModel1D.from_r2_encoder(
        r2_model.encoder,
        head_kind=_model_head_kind(method),
        num_classes=5,
        latent_channels=64,
        gamma=0.1,
        clip_length=100.0,
    ).to(device)
    return model, {
        "seed": seed,
        "head_kind": _model_head_kind(method),
        "encoder_initial_state_sha256": _state_sha256(model.encoder.state_dict()),
        "head_initial_state_sha256": _state_sha256(model.head.state_dict()),
        "model_initial_state_sha256": _state_sha256(model.state_dict()),
        "rng_state_after_initialization": _rng_state_record(device),
        "r2_projection_registered": False,
        "r2_ce_head_registered": False,
    }


def _set_encoder_mode(
    model: OfficialCSSRHRRPModel1D,
    *,
    method: str,
    epoch: int,
) -> bool:
    trainable = method in E2E_METHODS and epoch >= 6
    model.encoder.requires_grad_(trainable)
    if trainable:
        model.encoder.train()
    else:
        model.encoder.eval()
    model.head.requires_grad_(True).train()
    return trainable


def _model_norm(parameters: Iterable[nn.Parameter]) -> float:
    squared = 0.0
    for parameter in parameters:
        squared += float(parameter.detach().square().sum())
    return math.sqrt(squared)


def _gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().square().sum())
    return math.sqrt(squared)


def _infer_single(
    model: OfficialCSSRHRRPModel1D,
    inputs: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    values = np.asarray(inputs, dtype=np.float32)
    collected: dict[str, list[np.ndarray]] = {
        "features": [],
        "logits": [],
        "probabilities": [],
    }
    is_cssr = model.head_kind == OFFICIAL_SEMANTICS_PCSSR_1D
    if is_cssr:
        collected["reconstruction_errors"] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, values.shape[0], batch_size):
            output = model(torch.from_numpy(values[start : start + batch_size]).to(device))
            collected["features"].append(output.feature_maps.detach().cpu().numpy())
            collected["logits"].append(output.head_output.logits.detach().cpu().numpy())
            collected["probabilities"].append(
                output.head_output.probabilities.detach().cpu().numpy()
            )
            if is_cssr:
                if not isinstance(output.head_output, OfficialPCSSRHeadOutput):
                    raise DataValidationError("pCSSR model returned a linear output")
                collected["reconstruction_errors"].append(
                    output.head_output.reconstruction_errors.detach().cpu().numpy()
                )
    arrays = {
        name: np.asarray(np.concatenate(parts, axis=0), dtype=np.float32)
        for name, parts in collected.items()
    }
    expected = {
        "features": (values.shape[0], 128, 76),
        "logits": (values.shape[0], 5, 76),
        "probabilities": (values.shape[0], 5),
    }
    if is_cssr:
        expected["reconstruction_errors"] = (values.shape[0], 5, 76)
    for name, shape in expected.items():
        if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
            raise DataValidationError(f"invalid inference array {name}: {arrays[name].shape}")
    return arrays


def _infer_pairs(
    model: OfficialCSSRHRRPModel1D,
    pair_inputs: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    values = np.asarray(pair_inputs, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (2, 601):
        raise DataValidationError("pair inputs must have shape [n,2,601]")
    flattened = values.reshape(-1, 601)
    single = _infer_single(model, flattened, device=device, batch_size=batch_size)
    result: dict[str, np.ndarray] = {}
    for name, array in single.items():
        result[name] = array.reshape(values.shape[0], 2, *array.shape[1:])
    return result


def _calibration_summary(
    model: OfficialCSSRHRRPModel1D,
    *,
    single_inputs: np.ndarray,
    single_labels: np.ndarray,
    pair_inputs: np.ndarray,
    pair_labels: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    single = _infer_single(model, single_inputs, device=device, batch_size=batch_size)
    pair = _infer_pairs(model, pair_inputs, device=device, batch_size=batch_size)
    single_probabilities = np.asarray(single["probabilities"], dtype=np.float64)
    pair_probabilities = np.asarray(pair["probabilities"], dtype=np.float64).mean(axis=1)
    labels = np.asarray(single_labels, dtype=np.int64)
    pair_y = np.asarray(pair_labels, dtype=np.int64)

    def diagnostics(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, float]:
        if (
            not np.isfinite(probabilities).all()
            or np.any(probabilities <= 0.0)
            or not np.allclose(
                probabilities.sum(axis=1), 1.0, rtol=1.0e-6, atol=1.0e-7
            )
        ):
            raise DataValidationError(
                "calibration probabilities are non-finite, zero, or unnormalized"
            )
        indices = np.arange(targets.size)
        one_hot = np.eye(5, dtype=np.float64)[targets]
        confidence = probabilities.max(axis=1)
        prediction = probabilities.argmax(axis=1)
        bins = np.linspace(0.0, 1.0, 16)
        bin_indices = np.digitize(confidence, bins[1:-1], right=False)
        if bin_indices.shape != confidence.shape or np.any(
            (bin_indices < 0) | (bin_indices >= 15)
        ):
            raise DataValidationError("calibration confidence bin assignment failed")
        ece = 0.0
        covered = 0
        for bin_index in range(15):
            selected = bin_indices == bin_index
            covered += int(selected.sum())
            if selected.any():
                ece += float(selected.mean()) * abs(
                    float(np.mean(prediction[selected] == targets[selected]))
                    - float(np.mean(confidence[selected]))
                )
        if covered != targets.size:
            raise DataValidationError("calibration ECE omitted one or more samples")
        result = {
            "accuracy": accuracy_score(targets, prediction),
            "macro_f1": macro_f1_score(targets, prediction, labels=range(5)),
            "nll": float(-np.log(probabilities[indices, targets]).mean()),
            "brier": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
            "ece_15_bin": ece,
        }
        _assert_finite_tree(result, context="calibration diagnostics")
        return result

    return {
        **{f"single_{key}": value for key, value in diagnostics(single_probabilities, labels).items()},
        **{f"pair_{key}": value for key, value in diagnostics(pair_probabilities, pair_y).items()},
    }


def _build_optimizer(
    model: OfficialCSSRHRRPModel1D,
    *,
    method: str,
    config: Mapping[str, Any],
) -> torch.optim.Optimizer:
    training = config["training"]
    groups: list[dict[str, Any]] = [
        {
            "params": tuple(model.head.parameters()),
            "lr": float(training["head_base_lr"]),
            "weight_decay": float(training["head_weight_decay"]),
            "group_name": "head",
            "base_lr": float(training["head_base_lr"]),
        }
    ]
    if method in E2E_METHODS:
        groups.append(
            {
                "params": tuple(model.encoder.parameters()),
                "lr": 0.0,
                "weight_decay": float(training["encoder_weight_decay"]),
                "group_name": "encoder",
                "base_lr": float(training["encoder_base_lr"]),
            }
        )
    return torch.optim.SGD(
        groups,
        momentum=float(training["momentum"]),
        nesterov=bool(training["nesterov"]),
    )


def _set_optimizer_lrs(
    optimizer: torch.optim.Optimizer,
    *,
    head_lr: float,
    encoder_lr: float,
) -> None:
    for group in optimizer.param_groups:
        if group["group_name"] == "head":
            group["lr"] = float(head_lr)
        elif group["group_name"] == "encoder":
            group["lr"] = float(encoder_lr)
        else:
            raise DataValidationError("unknown optimizer group")


def _train_model(
    *,
    model: OfficialCSSRHRRPModel1D,
    method: str,
    pair_id: str,
    train_rows: Sequence[Mapping[str, Any]],
    train_inputs: np.ndarray,
    train_labels: np.ndarray,
    calibration_inputs: np.ndarray,
    calibration_labels: np.ndarray,
    calibration_pair_inputs: np.ndarray,
    calibration_pair_labels: np.ndarray,
    config: Mapping[str, Any],
    phase: str,
    device: torch.device,
) -> dict[str, Any]:
    if len(train_rows) != 720 or np.asarray(train_inputs).shape != (720, 601):
        raise DataValidationError("official CSSR training must use 720 unique bases")
    labels = np.asarray(train_labels, dtype=np.int64)
    if Counter(labels.tolist()) != Counter({index: 144 for index in range(5)}):
        raise DataValidationError("official CSSR training classes are not 5 x 144")
    if len({str(row["sample_id"]) for row in train_rows}) != 720:
        raise DataValidationError("official CSSR training duplicated a base")
    if any(str(row["experiment_role"]) != "train_known" for row in train_rows):
        raise DataValidationError("non-train evidence entered official CSSR training")

    training = config["training"]
    epochs = 1 if phase == "smoke" else int(training["epochs"])
    if epochs != (1 if phase == "smoke" else 40):
        raise DataValidationError("official CSSR epoch count changed")
    batch_size = int(training["batch_size"])
    steps_per_epoch = math.ceil(720 / batch_size)
    if steps_per_epoch != 6:
        raise DataValidationError("official CSSR must use six updates per epoch")
    optimizer = _build_optimizer(model, method=method, config=config)
    head_parameters = tuple(model.head.parameters())
    encoder_parameters = tuple(model.encoder.parameters())
    initial_encoder_vector = _parameter_vector(encoder_parameters)
    initial_encoder_buffers = _buffer_state(model.encoder)
    initial_encoder_state_sha256 = _state_sha256(model.encoder.state_dict())
    logs: list[dict[str, Any]] = []
    schedules: list[dict[str, Any]] = []
    y_all = torch.from_numpy(labels)

    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        encoder_trainable = _set_encoder_mode(model, method=method, epoch=epoch)
        if encoder_trainable != (method in E2E_METHODS and epoch >= 6):
            raise DataValidationError("encoder freeze schedule changed")
        encoder_before = _parameter_vector(encoder_parameters)
        head_before = _parameter_vector(head_parameters)
        buffers_before = _buffer_state(model.encoder)
        material = build_training_epoch_material(
            train_rows,
            np.asarray(train_inputs, dtype=np.float32),
            phase=phase,
            pair_id=pair_id,
            epoch=epoch,
            config=config,
        )
        indices = np.asarray(material["indices"], dtype=np.int64)
        augmented = np.asarray(material["augmented_inputs"], dtype=np.float32)
        if sorted(indices.tolist()) != list(range(720)):
            raise DataValidationError("one epoch does not use every base exactly once")
        if augmented.shape != (720, 601) or not np.isfinite(augmented).all():
            raise DataValidationError("training augmentation is invalid")
        schedules.append(
            {
                key: _mapping_to_json(value)
                for key, value in material.items()
                if key not in {"indices", "gain", "noise", "augmented_inputs"}
            }
        )
        totals = {
            "loss": 0.0,
            "correct": 0,
            "count": 0,
            "head_grad": 0.0,
            "encoder_grad": 0.0,
            "feature_norm": 0.0,
            "true_error": 0.0,
            "wrong_error": 0.0,
            "gap": 0.0,
        }
        batch_count = 0
        for batch_index, start in enumerate(range(0, 720, batch_size)):
            selected = indices[start : start + batch_size]
            # Augmented rows are stored in source-row order, while indices freeze
            # the optimizer visitation order.
            x = torch.from_numpy(augmented[selected]).to(device)
            y = y_all[selected].to(device)
            lrs = learning_rates_for_update(
                epoch=epoch,
                batch_index=batch_index,
                steps_per_epoch=steps_per_epoch,
                method=method,
                config=config,
            )
            _set_optimizer_lrs(
                optimizer,
                head_lr=float(lrs["head"]),
                encoder_lr=float(lrs["encoder"]),
            )
            optimizer.zero_grad(set_to_none=True)
            loss, output = model.loss(x, y)
            if not bool(torch.isfinite(loss)):
                raise DataValidationError("official CSSR loss became NaN or Inf")
            loss.backward()
            head_grad = _gradient_norm(head_parameters)
            encoder_grad = _gradient_norm(encoder_parameters)
            if not math.isfinite(head_grad) or not math.isfinite(encoder_grad):
                raise DataValidationError("official CSSR gradient became NaN or Inf")
            if not encoder_trainable and encoder_grad != 0.0:
                raise DataValidationError("frozen encoder received a gradient")
            optimizer.step()
            if not all(
                bool(torch.isfinite(parameter).all()) for parameter in model.parameters()
            ):
                raise DataValidationError("official CSSR parameter became NaN or Inf")
            count = int(y.numel())
            probabilities = output.head_output.probabilities
            totals["loss"] += float(loss.detach()) * count
            totals["correct"] += int((probabilities.argmax(dim=1) == y).sum())
            totals["count"] += count
            totals["head_grad"] += head_grad
            totals["encoder_grad"] += encoder_grad
            totals["feature_norm"] += float(
                torch.linalg.vector_norm(output.feature_maps.detach(), dim=(1, 2)).sum()
            )
            if isinstance(output.head_output, OfficialPCSSRHeadOutput):
                errors = output.head_output.reconstruction_errors.mean(dim=-1)
                true_error = errors.gather(1, y[:, None]).squeeze(1)
                wrong_error = errors.masked_fill(
                    torch.nn.functional.one_hot(y, num_classes=5).bool(),
                    float("inf"),
                ).min(dim=1).values
                totals["true_error"] += float(true_error.detach().sum())
                totals["wrong_error"] += float(wrong_error.detach().sum())
                totals["gap"] += float((wrong_error - true_error).detach().sum())
            batch_count += 1
        if totals["count"] != 720 or batch_count != 6:
            raise DataValidationError("official CSSR epoch coverage changed")
        encoder_after = _parameter_vector(encoder_parameters)
        buffers_after = _buffer_state(model.encoder)
        parameters_changed = not torch.equal(encoder_before, encoder_after)
        buffers_changed = any(
            not torch.equal(buffers_before[name], buffers_after[name])
            for name in buffers_before
        )
        if not encoder_trainable and (parameters_changed or buffers_changed):
            raise DataValidationError("frozen encoder parameter or BN buffer changed")
        diagnostics = _calibration_summary(
            model,
            single_inputs=calibration_inputs,
            single_labels=calibration_labels,
            pair_inputs=calibration_pair_inputs,
            pair_labels=calibration_pair_labels,
            device=device,
            batch_size=batch_size,
        )
        record = {
            "epoch": epoch,
            "method": method,
            "encoder_trainable": encoder_trainable,
            "sample_count": 720,
            "optimizer_updates": 6,
            "train_loss": totals["loss"] / 720.0,
            "train_accuracy": totals["correct"] / 720.0,
            "mean_head_gradient_norm": totals["head_grad"] / batch_count,
            "mean_encoder_gradient_norm": totals["encoder_grad"] / batch_count,
            "mean_feature_frobenius_norm": totals["feature_norm"] / 720.0,
            "head_parameter_norm": _model_norm(head_parameters),
            "encoder_parameter_relative_update": _relative_change(
                encoder_before, encoder_after
            ),
            "encoder_parameter_changed": parameters_changed,
            "encoder_buffer_changed": buffers_changed,
            "head_parameter_relative_update": _relative_change(
                head_before, _parameter_vector(head_parameters)
            ),
            "learning_rate_first_update": learning_rates_for_update(
                epoch=epoch,
                batch_index=0,
                steps_per_epoch=steps_per_epoch,
                method=method,
                config=config,
            ),
            "learning_rate_last_update": learning_rates_for_update(
                epoch=epoch,
                batch_index=steps_per_epoch - 1,
                steps_per_epoch=steps_per_epoch,
                method=method,
                config=config,
            ),
            "schedule_sha256": material["schedule_sha256"],
            "gain_sha256": material["gain_sha256"],
            "noise_sha256": material["noise_sha256"],
            "augmented_inputs_sha256": material["augmented_inputs_sha256"],
            "known_calibration_used_for_training": False,
            "surrogate_unknown_used_for_training": False,
            "final_unknown_used": False,
            "elapsed_seconds": time.perf_counter() - epoch_started,
            **diagnostics,
        }
        if method in CSSR_METHODS:
            record.update(
                {
                    "train_true_class_reconstruction_error": totals["true_error"] / 720.0,
                    "train_nearest_wrong_reconstruction_error": totals["wrong_error"] / 720.0,
                    "train_reconstruction_gap": totals["gap"] / 720.0,
                }
            )
        _assert_finite_tree(record, context=f"training log epoch {epoch}")
        logs.append(record)
    model.eval()
    final_encoder_vector = _parameter_vector(encoder_parameters)
    final_encoder_buffers = _buffer_state(model.encoder)
    if method not in E2E_METHODS:
        if not torch.equal(initial_encoder_vector, final_encoder_vector):
            raise DataValidationError("FT encoder parameters changed")
        if any(
            not torch.equal(initial_encoder_buffers[name], final_encoder_buffers[name])
            for name in initial_encoder_buffers
        ):
            raise DataValidationError("FT encoder buffers changed")
    return {
        "model": model,
        "training_log": logs,
        "schedule_audits": schedules,
        "audit": {
            "status": "passed",
            "phase": phase,
            "epochs": epochs,
            "formal_checkpoint_epoch": 40 if phase == "pilot" else None,
            "checkpoint_selection": "fixed_final_epoch",
            "train_unique_base_count": 720,
            "train_class_counts": [144] * 5,
            "diagnostic_single_sample_count": int(len(calibration_labels)),
            "diagnostic_pair_count": int(len(calibration_pair_labels)),
            "diagnostic_evaluation_population": (
                "stable_first_two_pairs_per_known_class_views"
                if phase == "smoke"
                else "full_known_calibration"
            ),
            "optimizer_updates": epochs * 6,
            "encoder_frozen_epochs": (
                list(range(1, epochs + 1))
                if method not in E2E_METHODS
                else list(range(1, min(5, epochs) + 1))
            ),
            "encoder_trainable_epochs": (
                [] if method not in E2E_METHODS else list(range(6, epochs + 1))
            ),
            "initial_encoder_state_sha256": initial_encoder_state_sha256,
            "final_encoder_state_sha256": _state_sha256(model.encoder.state_dict()),
            "final_head_state_sha256": _state_sha256(model.head.state_dict()),
            "final_model_state_sha256": _state_sha256(model.state_dict()),
            "encoder_total_relative_drift": _relative_change(
                initial_encoder_vector, final_encoder_vector
            ),
            "pair_multiplicity_weight": False,
            "known_calibration_used_for_training": False,
            "surrogate_unknown_used_for_training": False,
            "final_unknown_used": False,
            "even_angle_test_used": False,
        },
    }


def _raw_scores_to_cpu(scores: OfficialRawScores) -> dict[str, np.ndarray]:
    return {
        "s1": scores.s1.detach().cpu().numpy().astype(np.float64),
        "s2": scores.s2.detach().cpu().numpy().astype(np.float64),
        "s3": scores.s3.detach().cpu().numpy().astype(np.float64),
    }


def _fit_official_statistics(
    *,
    model: OfficialCSSRHRRPModel1D,
    train_rows: Sequence[Mapping[str, Any]],
    train_inputs: np.ndarray,
    pair_id: str,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    if model.head_kind != OFFICIAL_SEMANTICS_PCSSR_1D:
        raise DataValidationError("official score statistics require a pCSSR model")
    batch_size = int(config["training"]["batch_size"])
    raw = _infer_single(model, train_inputs, device=device, batch_size=batch_size)
    features = torch.from_numpy(raw["features"]).to(device)
    logits = torch.from_numpy(raw["logits"]).to(device)
    probabilities = torch.from_numpy(raw["probabilities"]).to(device)
    predictions = probabilities.argmax(dim=1)
    templates = build_official_score_templates(
        features,
        predictions,
        num_classes=5,
        power=8,
    )
    counts = templates.counts.detach().cpu().numpy().astype(np.int64)
    if counts.shape != (5,) or np.any(counts <= 0) or int(counts.sum()) != 720:
        raise DataValidationError("an official template prediction class is empty")
    score_parts: list[torch.Tensor] = []
    augmentation_audits: list[dict[str, Any]] = []
    for variant in range(1, 5):
        material = build_score_norm_augmentation(
            train_rows,
            np.asarray(train_inputs, dtype=np.float32),
            pair_id=pair_id,
            variant=variant,
            config=config,
        )
        augmented = np.asarray(material["augmented_inputs"], dtype=np.float32)
        if augmented.shape != (720, 601) or not np.isfinite(augmented).all():
            raise DataValidationError("score-normalization augmentation is invalid")
        inference = _infer_single(
            model,
            augmented,
            device=device,
            batch_size=batch_size,
        )
        augmented_features = torch.from_numpy(inference["features"]).to(device)
        augmented_logits = torch.from_numpy(inference["logits"]).to(device)
        augmented_prediction = torch.from_numpy(
            inference["probabilities"].argmax(axis=1).astype(np.int64)
        ).to(device)
        raw_scores = raw_official_scores(
            augmented_features,
            augmented_logits,
            augmented_prediction,
            templates,
        )
        score_parts.append(raw_scores.values)
        augmentation_audits.append(
            {
                key: _mapping_to_json(value)
                for key, value in material.items()
                if key not in {"gain", "noise", "augmented_inputs"}
            }
        )
    normalization = fit_score_normalization(
        torch.cat(score_parts, dim=0),
        min_std=1.0e-12,
        epsilon=1.0e-8,
    )
    return {
        "templates": templates,
        "normalization": normalization,
        "raw_train_inference": raw,
        "audit": {
            "status": "passed",
            "template_population": "raw_unique_train_known",
            "template_count": 720,
            "prediction_class_counts": counts.tolist(),
            "prediction_sha256": _array_sha256(
                predictions.detach().cpu().numpy().astype(np.int64)
            ),
            "first_order_sha256": _array_sha256(
                templates.first_order.detach().cpu().numpy()
            ),
            "gram_sha256": _array_sha256(templates.gram.detach().cpu().numpy()),
            "normalization_population": "four_deterministic_augmented_train_known",
            "normalization_count": 2880,
            "normalization_mean": normalization.mean.detach().cpu().tolist(),
            "normalization_std": normalization.std.detach().cpu().tolist(),
            "normalization_epsilon": normalization.epsilon,
            "normalization_min_std": normalization.min_std,
            "augmentation_audits": augmentation_audits,
            "known_calibration_used": False,
            "surrogate_unknown_used": False,
            "final_unknown_used": False,
        },
    }


def _templates_to_device(
    templates: OfficialScoreTemplates,
    device: torch.device,
) -> OfficialScoreTemplates:
    return OfficialScoreTemplates(
        first_order=templates.first_order.to(device),
        gram=templates.gram.to(device),
        counts=templates.counts.to(device),
        num_classes=templates.num_classes,
        power=templates.power,
    )


def _normalization_to_device(
    normalization: OfficialScoreNormalization,
    device: torch.device,
) -> OfficialScoreNormalization:
    return OfficialScoreNormalization(
        mean=normalization.mean.to(device),
        std=normalization.std.to(device),
        epsilon=normalization.epsilon,
        min_std=normalization.min_std,
    )


def _score_rule_to_internal(name: str) -> str:
    return {
        "S1": "s1",
        "S2": "s2",
        "S3": "s3",
        "S1+S2": "s1_s2",
        "S1+S3": "s1_s3",
        "S2+S3": "s2_s3",
        "full": "full",
        "pcssr_max_pair_probability": "max_pair_probability",
    }[name]


def _evaluate_model(
    *,
    model: OfficialCSSRHRRPModel1D,
    method: str,
    pair_id: str,
    prepared: Any,
    pair_inputs_by_role: Mapping[str, np.ndarray],
    frozen_r2_arrays: Mapping[str, Mapping[str, np.ndarray]],
    config: Mapping[str, Any],
    device: torch.device,
    smoke: bool,
    templates: OfficialScoreTemplates | None,
    normalization: OfficialScoreNormalization | None,
) -> dict[str, Any]:
    role_indices = _evaluation_role_indices(prepared, smoke=smoke, config=config)
    role_pair_rows = {
        role: [
            _role_manifest_rows(prepared, role)[int(index)]
            for index in role_indices[role]
        ]
        for role in ("known_calibration", "surrogate_unknown")
    }
    batch_size = int(config["training"]["batch_size"])
    pair_outputs = {
        role: _infer_pairs(
            model,
            np.asarray(pair_inputs_by_role[role], dtype=np.float32)[role_indices[role]],
            device=device,
            batch_size=batch_size,
        )
        for role in role_indices
    }
    model_role: dict[str, dict[str, Any]] = {}
    score_by_rule: dict[str, dict[str, dict[str, Any]]] = {}
    extra_by_role: dict[str, dict[str, np.ndarray]] = {}
    if method in LINEAR_METHODS:
        for role, output in pair_outputs.items():
            pair = matched_linear_pair_output(
                torch.from_numpy(output["logits"]).to(device),
                torch.from_numpy(output["probabilities"]).to(device),
            )
            probabilities = pair.pair_probabilities.detach().cpu().numpy().astype(np.float64)
            prediction = pair.predicted_class.detach().cpu().numpy().astype(np.int64)
            unknown_score = pair.unknown_score.detach().cpu().numpy().astype(np.float64)
            model_role[role] = {
                "known_prediction": prediction,
                "main_unknown_score": unknown_score,
                "main_score_name": "negative_max_pair_probability",
                "diagnostic_class_conditional_mls": None,
            }
            extra_by_role[role] = {
                "pair_probabilities": probabilities,
                "max_pair_probability": pair.max_pair_probability.detach().cpu().numpy(),
                "max_spatial_average_logit": pair.max_spatial_average_logit.detach().cpu().numpy(),
            }
    elif method in CSSR_METHODS:
        if templates is None or normalization is None:
            raise DataValidationError("pCSSR evaluation lacks frozen score statistics")
        templates = _templates_to_device(templates, device)
        normalization = _normalization_to_device(normalization, device)
        for role, output in pair_outputs.items():
            pair = official_pcssr_pair_scores(
                torch.from_numpy(output["features"]).to(device),
                torch.from_numpy(output["logits"]).to(device),
                torch.from_numpy(output["probabilities"]).to(device),
                templates,
                normalization,
            )
            probabilities = pair.pair_probabilities.detach().cpu().numpy().astype(np.float64)
            prediction = pair.predicted_class.detach().cpu().numpy().astype(np.int64)
            main = pair.unknown_score.detach().cpu().numpy().astype(np.float64)
            model_role[role] = {
                "known_prediction": prediction,
                "main_unknown_score": main,
                "main_score_name": "negative_standardized_S1_S2_S3",
                "diagnostic_class_conditional_mls": None,
            }
            for official_name, tensor in pair.unknown_scores_by_rule.items():
                internal = _score_rule_to_internal(official_name)
                score_by_rule.setdefault(internal, {})[role] = {
                    "known_prediction": prediction,
                    "main_unknown_score": tensor.detach().cpu().numpy().astype(np.float64),
                    "main_score_name": f"official_{official_name}",
                    "diagnostic_class_conditional_mls": None,
                }
            extra_by_role[role] = {
                "pair_probabilities": probabilities,
                "per_view_raw_scores": pair.per_view_raw.detach().cpu().numpy(),
                "per_view_standardized_scores": pair.per_view_standardized.detach().cpu().numpy(),
                "pair_standardized_components": pair.pair_standardized_components.detach().cpu().numpy(),
            }
    else:
        raise DataValidationError(f"unsupported official method: {method}")

    view_swap_audit: dict[str, Any] = {
        "status": "passed",
        "real_pair_reinference": True,
        "rtol": 1.0e-5,
        "atol": 1.0e-6,
        "roles": {},
    }
    for role, indices in role_indices.items():
        swapped_output = _infer_pairs(
            model,
            np.asarray(pair_inputs_by_role[role], dtype=np.float32)[indices][
                :, ::-1
            ].copy(),
            device=device,
            batch_size=batch_size,
        )
        if method in LINEAR_METHODS:
            swapped_pair = matched_linear_pair_output(
                torch.from_numpy(swapped_output["logits"]).to(device),
                torch.from_numpy(swapped_output["probabilities"]).to(device),
            )
            swapped_probabilities = (
                swapped_pair.pair_probabilities.detach().cpu().numpy().astype(np.float64)
            )
            swapped_prediction = (
                swapped_pair.predicted_class.detach().cpu().numpy().astype(np.int64)
            )
            swapped_scores = {
                "main": swapped_pair.unknown_score.detach().cpu().numpy().astype(np.float64)
            }
            original_scores = {"main": model_role[role]["main_unknown_score"]}
        else:
            if templates is None or normalization is None:
                raise DataValidationError("pCSSR view-swap audit lacks score statistics")
            swapped_pair = official_pcssr_pair_scores(
                torch.from_numpy(swapped_output["features"]).to(device),
                torch.from_numpy(swapped_output["logits"]).to(device),
                torch.from_numpy(swapped_output["probabilities"]).to(device),
                templates,
                normalization,
            )
            swapped_probabilities = (
                swapped_pair.pair_probabilities.detach().cpu().numpy().astype(np.float64)
            )
            swapped_prediction = (
                swapped_pair.predicted_class.detach().cpu().numpy().astype(np.int64)
            )
            swapped_scores = {
                _score_rule_to_internal(name): tensor.detach()
                .cpu()
                .numpy()
                .astype(np.float64)
                for name, tensor in swapped_pair.unknown_scores_by_rule.items()
            }
            original_scores = {
                rule: score_by_rule[rule][role]["main_unknown_score"]
                for rule in SCORE_VARIANTS
            }
        original_probabilities = np.asarray(
            extra_by_role[role]["pair_probabilities"], dtype=np.float64
        )
        original_prediction = np.asarray(
            model_role[role]["known_prediction"], dtype=np.int64
        )
        if not np.array_equal(original_prediction, swapped_prediction):
            raise DataValidationError(f"real view swap changed {role} known prediction")
        if not np.allclose(
            original_probabilities,
            swapped_probabilities,
            rtol=1.0e-5,
            atol=1.0e-6,
        ):
            raise DataValidationError(f"real view swap changed {role} pair probability")
        score_deltas: dict[str, float] = {}
        for rule, original_score in original_scores.items():
            swapped_score = np.asarray(swapped_scores[rule], dtype=np.float64)
            original_score = np.asarray(original_score, dtype=np.float64)
            if not np.allclose(
                original_score, swapped_score, rtol=1.0e-5, atol=1.0e-6
            ):
                raise DataValidationError(
                    f"real view swap changed {role} unknown score {rule}"
                )
            score_deltas[rule] = float(np.max(np.abs(original_score - swapped_score)))
        view_swap_audit["roles"][role] = {
            "pair_count": int(original_prediction.size),
            "prediction_exact": True,
            "maximum_probability_absolute_delta": float(
                np.max(np.abs(original_probabilities - swapped_probabilities))
            ),
            "maximum_score_absolute_delta": score_deltas,
        }
    _assert_finite_tree(view_swap_audit, context="view swap audit")

    acceptance = float(config["evaluation"]["threshold_known_acceptance_rate"])
    metrics = _evaluate_score_arrays(
        prepared=prepared,
        role_indices=role_indices,
        score_arrays=model_role,
        acceptance_rate=acceptance,
    )
    # The historical row builder stores a vector called fused_logits.  Here the
    # mathematically relevant target-level vector is pair probability, so log
    # probability is used solely as an invertible serialization carrier.
    role_logits = {
        role: np.log(np.maximum(extra_by_role[role]["pair_probabilities"], 1.0e-300))
        for role in role_indices
    }
    prediction_rows = _build_method_prediction_rows(
        method=method,
        prepared=prepared,
        role_indices=role_indices,
        role_pair_rows=role_pair_rows,
        role_logits=role_logits,
        role_scores=model_role,
        metrics=metrics,
        reference_metadata=None,
    )
    offset = 0
    for role in ("known_calibration", "surrogate_unknown"):
        extra = extra_by_role[role]
        count = len(role_pair_rows[role])
        for local_index in range(count):
            row = prediction_rows[offset + local_index]
            row["pair_probabilities"] = json.dumps(
                extra["pair_probabilities"][local_index].tolist(),
                separators=(",", ":"),
            )
            row["fused_logits_semantics"] = "log_pair_probability_serialization_only"
            if method in LINEAR_METHODS:
                row["max_pair_probability"] = float(extra["max_pair_probability"][local_index])
                row["max_spatial_average_logit"] = float(
                    extra["max_spatial_average_logit"][local_index]
                )
                for key in ("s1", "s2", "s3", "s1_s2", "s1_s3", "s2_s3", "full"):
                    row[key] = ""
            else:
                raw = extra["per_view_raw_scores"][local_index]
                standardized = extra["per_view_standardized_scores"][local_index]
                components = extra["pair_standardized_components"][local_index]
                row["per_view_raw_scores"] = json.dumps(raw.tolist(), separators=(",", ":"))
                row["per_view_standardized_scores"] = json.dumps(
                    standardized.tolist(), separators=(",", ":")
                )
                row["s1"] = float(components[0])
                row["s2"] = float(components[1])
                row["s3"] = float(components[2])
                row["s1_s2"] = float(components[0] + components[1])
                row["s1_s3"] = float(components[0] + components[2])
                row["s2_s3"] = float(components[1] + components[2])
                row["full"] = float(components.sum())
                row["max_pair_probability"] = float(
                    extra["pair_probabilities"][local_index].max()
                )
                row["max_spatial_average_logit"] = ""
        offset += count
    _metrics_exact(
        metrics,
        recompute_method_metrics_from_prediction_rows(
            prediction_rows,
            known_acceptance_rate=acceptance,
        ),
        context=method,
    )
    identity_rows, absorption_rows, error_analysis = build_identity_and_absorption_rows(
        prediction_rows,
        method=method,
        pair_id=pair_id,
        train_class_order=prepared.train_class_order,
        acceptance_rate=acceptance,
    )
    ablation_rows: list[dict[str, Any]] = []
    if method in CSSR_METHODS:
        for rule in SCORE_VARIANTS:
            rule_metrics = _evaluate_score_arrays(
                prepared=prepared,
                role_indices=role_indices,
                score_arrays=score_by_rule[rule],
                acceptance_rate=acceptance,
            )
            ablation_rows.append(
                {
                    "method": method,
                    "score_rule": rule,
                    **{key: float(rule_metrics[key]) for key in REPORT_METRIC_KEYS},
                    "threshold": float(rule_metrics["threshold"]),
                }
            )
    return {
        "metrics": metrics,
        "prediction_rows": prediction_rows,
        "identity_rows": identity_rows,
        "absorption_rows": absorption_rows,
        "error_analysis": error_analysis,
        "score_ablation_rows": ablation_rows,
        "role_indices": role_indices,
        "role_pair_rows": role_pair_rows,
        "pair_outputs": pair_outputs,
        "extra_by_role": extra_by_role,
        "view_swap_audit": view_swap_audit,
    }


def _evaluate_o0(
    *,
    prepared: Any,
    frozen_r2_arrays: Mapping[str, Mapping[str, np.ndarray]],
    pair_id: str,
    config: Mapping[str, Any],
    smoke: bool,
) -> dict[str, Any]:
    role_indices = _evaluation_role_indices(prepared, smoke=smoke, config=config)
    role_pair_rows = {
        role: [
            _role_manifest_rows(prepared, role)[int(index)]
            for index in role_indices[role]
        ]
        for role in ("known_calibration", "surrogate_unknown")
    }
    role_logits = {
        role: np.asarray(frozen_r2_arrays[role]["global_logits"], dtype=np.float64)[
            role_indices[role]
        ]
        for role in role_indices
    }
    scores = _class_conditional_mls_for_roles(
        full_calibration_logits=np.asarray(
            frozen_r2_arrays["known_calibration"]["global_logits"], dtype=np.float64
        ),
        full_calibration_labels=np.asarray(
            prepared.labels["known_calibration"], dtype=np.int64
        ),
        full_calibration_pair_ids=prepared.pair_ids["known_calibration"],
        role_logits=role_logits,
        role_pair_rows=role_pair_rows,
    )
    score_arrays = {
        role: {
            "known_prediction": role_logits[role].argmax(axis=1),
            "main_unknown_score": scores[role],
            "main_score_name": "r2_class_conditional_mls",
            "diagnostic_class_conditional_mls": None,
        }
        for role in role_indices
    }
    acceptance = float(config["evaluation"]["threshold_known_acceptance_rate"])
    metrics = _evaluate_score_arrays(
        prepared=prepared,
        role_indices=role_indices,
        score_arrays=score_arrays,
        acceptance_rate=acceptance,
    )
    rows = _build_method_prediction_rows(
        method=O0_R2_CC_MLS,
        prepared=prepared,
        role_indices=role_indices,
        role_pair_rows=role_pair_rows,
        role_logits=role_logits,
        role_scores=score_arrays,
        metrics=metrics,
        reference_metadata=None,
    )
    _metrics_exact(
        metrics,
        recompute_method_metrics_from_prediction_rows(
            rows,
            known_acceptance_rate=acceptance,
        ),
        context=O0_R2_CC_MLS,
    )
    identity, absorption, errors = build_identity_and_absorption_rows(
        rows,
        method=O0_R2_CC_MLS,
        pair_id=pair_id,
        train_class_order=prepared.train_class_order,
        acceptance_rate=acceptance,
    )
    return {
        "metrics": metrics,
        "prediction_rows": rows,
        "identity_rows": identity,
        "absorption_rows": absorption,
        "error_analysis": errors,
    }


def _same_prediction_rows(
    expected: Sequence[Mapping[str, Any]],
    observed: Sequence[Mapping[str, Any]],
) -> None:
    if len(expected) != len(observed):
        raise DataValidationError("checkpoint replay row count changed")
    for index, (left, right) in enumerate(zip(expected, observed, strict=True)):
        if dict(left) != dict(right):
            differing = sorted(
                key
                for key in set(left) | set(right)
                if left.get(key) != right.get(key)
            )
            raise DataValidationError(
                f"checkpoint replay changed row {index}: {differing[:8]}"
            )


def _serialize_templates(
    templates: OfficialScoreTemplates | None,
    normalization: OfficialScoreNormalization | None,
) -> dict[str, np.ndarray]:
    if templates is None or normalization is None:
        return {
            "not_applicable": np.asarray([1], dtype=np.int8),
        }
    return {
        "first_order": templates.first_order.detach().cpu().numpy(),
        "gram": templates.gram.detach().cpu().numpy(),
        "counts": templates.counts.detach().cpu().numpy(),
        "power": np.asarray([templates.power], dtype=np.int64),
        "normalization_mean": normalization.mean.detach().cpu().numpy(),
        "normalization_std": normalization.std.detach().cpu().numpy(),
        "normalization_epsilon": np.asarray([normalization.epsilon], dtype=np.float64),
        "normalization_min_std": np.asarray([normalization.min_std], dtype=np.float64),
    }


def _checkpoint_templates(
    checkpoint: Mapping[str, Any],
    *,
    device: torch.device,
) -> tuple[OfficialScoreTemplates | None, OfficialScoreNormalization | None]:
    payload = checkpoint.get("score_statistics")
    if payload is None:
        return None, None
    templates = OfficialScoreTemplates(
        first_order=payload["first_order"].to(device),
        gram=payload["gram"].to(device),
        counts=payload["counts"].to(device),
        num_classes=5,
        power=int(payload["power"]),
    )
    normalization = OfficialScoreNormalization(
        mean=payload["normalization_mean"].to(device),
        std=payload["normalization_std"].to(device),
        epsilon=float(payload["normalization_epsilon"]),
        min_std=float(payload["normalization_min_std"]),
    )
    return templates, normalization


def _official_audit_record(path: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    record = _read_json(Path(path).resolve())
    expected_commit = str(config["official_reference"]["commit"])
    expected_tolerances = config["official_reference"]["differential"]
    if (
        record.get("passed") is not True
        or record.get("status") != "passed"
        or record.get("official_commit") != expected_commit
        or record.get("float32") != "passed"
        or record.get("float64") != "passed"
    ):
        raise DataValidationError("official differential audit is absent or failed")
    expected_hashes = dict(config["official_reference"]["files"])
    if (
        record.get("file_sha256") != expected_hashes
        or record.get("verified_file_sha256") != expected_hashes
        or record.get("source_execution")
        != "selected_ast_definitions_from_hash_verified_methods/cssr.py"
        or record.get("method_ids")
        != {
            "official": OFFICIAL_SEMANTICS_PCSSR_1D,
            "matched_linear_control": MATCHED_LINEAR_CONTROL_1D,
        }
    ):
        raise DataValidationError("official differential audit source identity changed")
    device = torch.device(str(record.get("device", "")))
    if device.type != "cuda":
        raise DataValidationError("formal official differential must run on CUDA")
    runtime = record.get("runtime_contract")
    expected_gpu = str(config["runtime"]["expected_gpu_model"])
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("device_type") != "cuda"
        or runtime.get("cuda_device_name") != expected_gpu
        or runtime.get("expected_cuda_device_name") != expected_gpu
        or runtime.get("formal_cuda_device_match") is not True
        or runtime.get("cublas_workspace_config") != ":4096:8"
        or runtime.get("deterministic_algorithms") is not True
        or runtime.get("deterministic_algorithms_warn_only") is not False
        or runtime.get("cudnn_benchmark") is not False
        or runtime.get("cuda_matmul_allow_tf32") is not False
        or runtime.get("cudnn_allow_tf32") is not False
        or record.get("cuda_device_name") != expected_gpu
    ):
        raise DataValidationError("formal official differential runtime changed")
    expected_oracle_contract = {
        "seed": 20260904,
        "required_dtypes": ["float32", "float64"],
        "tolerances": {
            name: {
                "rtol": float(expected_tolerances[name]["rtol"]),
                "atol": float(expected_tolerances[name]["atol"]),
            }
            for name in ("float32", "float64")
        },
        "clip_bounds": [-100.0, 100.0],
        "pair_score_rules": [
            "S1",
            "S2",
            "S3",
            "S1+S2",
            "S1+S3",
            "S2+S3",
            "full",
            "pcssr_max_pair_probability",
        ],
        "pair_probability_rule": "arithmetic_mean_of_two_view_probabilities",
        "pair_prediction_rule": "argmax_pair_probability",
        "unknown_score_direction": "negative_knownness",
        "deterministic_repeats": 2,
    }
    if record.get("oracle_contract") != expected_oracle_contract:
        raise DataValidationError("official differential oracle contract changed")
    dtype_checks = record.get("dtype_checks")
    if not isinstance(dtype_checks, Mapping) or set(dtype_checks) != {
        "float32",
        "float64",
    }:
        raise DataValidationError("official differential dtype records are incomplete")
    for name in ("float32", "float64"):
        check = dtype_checks[name]
        expected = expected_tolerances[name]
        if (
            not isinstance(check, Mapping)
            or check.get("passed") is not True
            or float(check.get("rtol", -1.0)) != float(expected["rtol"])
            or float(check.get("atol", -1.0)) != float(expected["atol"])
        ):
            raise DataValidationError(
                f"official differential {name} tolerance or status changed"
            )
        clip = check.get("clip_boundary_checks")
        pair = check.get("pair_checks")
        repeat = check.get("deterministic_repeat")
        differences = check.get("max_absolute_differences")
        required_differences = {
            "clip_lower_candidate_vs_official",
            "clip_lower_official_vs_literal",
            "clip_upper_official_vs_literal",
            "criterion_probability_literal",
            "pcssr_loss",
            "pcssr_input_gradient",
            "linear_loss",
            "linear_input_gradient",
            "raw_s1",
            "raw_s2",
            "raw_s3",
            "normalization_mean",
            "normalization_std",
            "standardized_scores",
            "integrated_score",
            "pair_view_logits",
            "pair_view_probabilities",
            "pair_probabilities",
            "pair_per_view_raw_scores",
            "pair_per_view_standardized_scores",
            "pair_standardized_components",
            "pair_full_integration",
        }
        required_differences.update(
            f"pair_knownness::{rule}"
            for rule in expected_oracle_contract["pair_score_rules"]
        )
        required_differences.update(
            f"pair_unknown_score::{rule}"
            for rule in expected_oracle_contract["pair_score_rules"]
        )
        if (
            not isinstance(clip, Mapping)
            or clip.get("passed") is not True
            or clip.get("bounds") != [-100.0, 100.0]
            or clip.get("lower_interior_exercised") is not True
            or clip.get("lower_exact_boundary_exercised") is not True
            or clip.get("lower_saturation_exercised") is not True
            or clip.get("upper_interior_exercised") is not True
            or clip.get("upper_exact_boundary_exercised") is not True
            or clip.get("upper_saturation_exercised") is not True
            or not isinstance(clip.get("upper_boundary_reference_only"), str)
            or not isinstance(pair, Mapping)
            or pair.get("passed") is not True
            or pair.get("probability") != "passed"
            or pair.get("argmax") != "passed"
            or pair.get("knownness_rules") != expected_oracle_contract["pair_score_rules"]
            or pair.get("unknown_score_direction") != "negative_knownness"
            or not isinstance(repeat, Mapping)
            or repeat.get("passed") is not True
            or int(repeat.get("repeats", -1)) != 2
            or repeat.get("record_equality") != "exact"
            or not isinstance(differences, Mapping)
            or not required_differences.issubset(differences)
        ):
            raise DataValidationError(
                f"official differential {name} coverage contract changed"
            )
    try:
        replay = audit_official_cssr_oracle(
            str(record["official_root"]),
            device=device,
        )
    except Exception as error:
        raise DataValidationError("official differential oracle replay failed") from error
    if _official_audit_identity(replay) != _official_audit_identity(record):
        raise DataValidationError("official differential oracle identity does not replay")
    return dict(record)


def _unit_destination(root: Path, pair_id: str, method: str) -> Path:
    return root / pair_id / "fold_0" / f"seed_{OFFICIAL_CSSR_SEED}" / method


def _phase_artifact_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and str(path.relative_to(root))
        not in {
            "artifact_hashes.json",
            "_PHASE_SUCCESS.json",
            "_PHASE_INCOMPLETE.json",
        }
    }


def _read_smoke_authorization(
    smoke_root: str | Path | None,
    *,
    config: Mapping[str, Any],
    oracle_audit: Mapping[str, Any],
) -> dict[str, Any]:
    if smoke_root is None:
        raise DataValidationError("pilot requires one completed, audited smoke root")
    root = Path(smoke_root).resolve()
    marker = _read_json(root / "_PHASE_SUCCESS.json")
    hashes = _read_json(root / "artifact_hashes.json")
    summary = _read_json(root / "phase_summary.json")
    oracle_identity_sha256 = _json_sha256(_official_audit_identity(oracle_audit))
    if (
        (root / "_PHASE_INCOMPLETE.json").exists()
        or
        marker.get("status") != "complete"
        or marker.get("phase_summary_sha256") != file_sha256(root / "phase_summary.json")
        or marker.get("artifact_hashes_sha256") != file_sha256(root / "artifact_hashes.json")
        or hashes != _phase_artifact_hashes(root)
        or summary.get("status") != "complete"
        or summary.get("phase") != "smoke"
        or summary.get("decision") != "diagnostic_smoke_only"
        or int(summary.get("unit_count", -1)) != 4
        or summary.get("config_sha256") != config["_config_sha256"]
        or summary.get("official_oracle_identity_sha256")
        != oracle_identity_sha256
        or summary.get("diagnostic_only") is not True
        or summary.get("confirmation_allowed") is not False
        or summary.get("automatic_followon_authorized") is not False
        or summary.get("smoke_authorization") is not None
        or summary.get("final_unknown_used") is not False
        or summary.get("even_angle_test_used") is not False
        or summary.get("final_unknown_test_authorized") is not False
    ):
        raise DataValidationError("smoke phase is not a valid pilot authorization")
    project_root = _bound_project_root(config)
    source_hashes = _task_source_hashes(project_root)
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    unit_hashes: dict[str, str] = {}
    code_commits: set[str] = set()
    for method in TRAINABLE_METHODS:
        unit = _unit_destination(root, "N1", method)
        unit_marker = _read_json(unit / "_SUCCESS.json")
        unit_manifest = _read_json(unit / "artifact_hashes.json")
        contract = _read_json(unit / "unit_contract.json")
        replay = _read_json(unit / "checkpoint_replay_audit.json")
        saved_oracle = _read_json(unit / "official_oracle_audit.json")
        if (
            unit_marker.get("status") != "success"
            or int(unit_marker.get("artifact_count", -1)) != len(unit_manifest)
            or unit_marker.get("artifact_hashes_sha256")
            != file_sha256(unit / "artifact_hashes.json")
            or unit_manifest != _artifact_hashes(unit)
            or contract.get("phase") != "smoke"
            or contract.get("pair_id") != "N1"
            or contract.get("method") != method
            or contract.get("config_sha256") != config["_config_sha256"]
            or contract.get("source_hashes") != source_hashes
            or contract.get("smoke_authorization") is not None
            or _json_sha256(_official_audit_identity(saved_oracle))
            != oracle_identity_sha256
            or replay.get("status") != "passed"
            or replay.get("state_dict_strict_load") is not True
            or replay.get("prediction_rows_exact") is not True
            or replay.get("metrics_exact") is not True
            or replay.get("all_prediction_score_fields_exact") is not True
            or replay.get("evaluation_logits_probabilities_exact") is not True
            or replay.get("real_pair_view_swap_reinference") is not True
        ):
            raise DataValidationError(f"smoke authorization unit failed: {method}")
        code_commits.add(str(contract.get("code_commit", "")))
        unit_hashes[method] = file_sha256(unit / "artifact_hashes.json")
    if code_commits != {current_commit} or summary.get("code_commit") != current_commit:
        raise DataValidationError("smoke and pilot do not share one code commit")
    if summary.get("source_hashes_sha256") != _json_sha256(source_hashes):
        raise DataValidationError("smoke source identity changed before pilot")
    return {
        "status": "authorized_by_completed_audited_smoke",
        "phase_summary_sha256": file_sha256(root / "phase_summary.json"),
        "artifact_hashes_sha256": file_sha256(root / "artifact_hashes.json"),
        "phase_success_sha256": file_sha256(root / "_PHASE_SUCCESS.json"),
        "unit_artifact_manifest_sha256": unit_hashes,
        "config_sha256": config["_config_sha256"],
        "official_oracle_identity_sha256": oracle_identity_sha256,
        "code_commit": current_commit,
        "source_hashes_sha256": _json_sha256(source_hashes),
        "pilot_authorized": True,
        "confirmation_allowed": False,
        "final_unknown_test_authorized": False,
    }


def _save_training_log(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = b"".join(
        (json.dumps(_mapping_to_json(row), ensure_ascii=False, sort_keys=True) + "\n").encode(
            "utf-8"
        )
        for row in rows
    )
    _atomic_write_bytes(path, payload)


def _checkpoint_payload(
    *,
    model: OfficialCSSRHRRPModel1D,
    method: str,
    pair_id: str,
    phase: str,
    config: Mapping[str, Any],
    prepared: Any,
    unique_rows: Sequence[Mapping[str, Any]],
    initialization: Mapping[str, Any],
    training_result: Mapping[str, Any],
    templates: OfficialScoreTemplates | None,
    normalization: OfficialScoreNormalization | None,
    r2_audit: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    smoke_authorization: Mapping[str, Any] | None,
    code_commit: str,
) -> dict[str, Any]:
    statistics = None
    if templates is not None and normalization is not None:
        statistics = {
            "first_order": templates.first_order.detach().cpu(),
            "gram": templates.gram.detach().cpu(),
            "counts": templates.counts.detach().cpu(),
            "power": templates.power,
            "normalization_mean": normalization.mean.detach().cpu(),
            "normalization_std": normalization.std.detach().cpu(),
            "normalization_epsilon": normalization.epsilon,
            "normalization_min_std": normalization.min_std,
        }
    return {
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "pair_id": pair_id,
        "angle_fold": ANGLE_FOLD,
        "method": method,
        "official_cssr_seed": OFFICIAL_CSSR_SEED,
        "checkpoint_epoch": 1 if phase == "smoke" else 40,
        "formal_checkpoint": phase == "pilot",
        "checkpoint_selection": "fixed_final_epoch",
        "config_sha256": config["_config_sha256"],
        "code_commit": code_commit,
        "source_hashes": dict(source_hashes),
        "smoke_authorization": (
            None if smoke_authorization is None else dict(smoke_authorization)
        ),
        "pair_manifest_sha256": prepared.pair_manifest_sha256,
        "unique_base_id_order_sha256": _sequence_sha256(
            row["sample_id"] for row in unique_rows
        ),
        "train_class_order": list(prepared.train_class_order),
        "surrogate_class_order": list(prepared.surrogate_class_order),
        "r2_checkpoint_sha256": r2_audit["checkpoint_sha256"],
        "initialization": dict(initialization),
        "training_audit": dict(training_result["audit"]),
        "model_state_dict": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        "score_statistics": statistics,
        "known_calibration_used_for_training": False,
        "surrogate_unknown_used_for_training": False,
        "final_unknown_used": False,
        "even_angle_test_used": False,
    }


def _atomic_torch_save(path: Path, value: Any) -> None:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    _atomic_write_bytes(path, buffer.getvalue())


def _resolved_config_bytes(config: Mapping[str, Any]) -> bytes:
    # ``_config_path`` is loader metadata, not protocol identity.  Omitting it
    # keeps unit artifacts replayable from a different checkout path/host.
    portable = {
        str(key): value for key, value in config.items() if key != "_config_path"
    }
    return yaml.safe_dump(
        _mapping_to_json(portable),
        allow_unicode=True,
        sort_keys=True,
    ).encode("utf-8")


def _metric_row(pair_id: str, method: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "angle_fold": ANGLE_FOLD,
        "seed": OFFICIAL_CSSR_SEED,
        "method": method,
        **{key: float(metrics[key]) for key in REPORT_METRIC_KEYS},
        "threshold": float(metrics["threshold"]),
    }


def _replay_arrays(rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "evaluation_role": np.asarray(
            [str(row["evaluation_role"]) for row in rows], dtype=np.str_
        ),
        "true_label": np.asarray([int(row["true_label"]) for row in rows], dtype=np.int64),
        "predicted_known_label": np.asarray(
            [int(row["predicted_known_label"]) for row in rows], dtype=np.int64
        ),
        "unknown_score": np.asarray(
            [float(row["unknown_score"]) for row in rows], dtype=np.float64
        ),
        "threshold": np.asarray(
            [float(row["threshold"]) for row in rows], dtype=np.float64
        ),
        "rejected": np.asarray(
            [str(row["rejected"]).lower() == "true" if not isinstance(row["rejected"], bool) else row["rejected"] for row in rows],
            dtype=np.bool_,
        ),
        "view1_sample_id": np.asarray(
            [str(row["view1_sample_id"]) for row in rows], dtype=np.str_
        ),
        "view2_sample_id": np.asarray(
            [str(row["view2_sample_id"]) for row in rows], dtype=np.str_
        ),
    }


def _train_inference_arrays(
    inference: Mapping[str, np.ndarray],
    train_rows: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    probabilities = np.asarray(inference["probabilities"], dtype=np.float32)
    logits = np.asarray(inference["logits"], dtype=np.float32)
    if probabilities.shape != (720, 5) or logits.shape != (720, 5, 76):
        raise DataValidationError("raw train inference shape changed")
    return {
        "sample_ids": np.asarray(
            [str(row["sample_id"]) for row in train_rows], dtype=np.str_
        ),
        "model_labels": np.asarray(
            [int(row["model_label"]) for row in train_rows], dtype=np.int64
        ),
        "predicted_classes": probabilities.argmax(axis=1).astype(np.int64),
        "probabilities": probabilities,
        "logits": logits,
    }


def _evaluation_inference_arrays(
    evaluation: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}
    for role in ("known_calibration", "surrogate_unknown"):
        output = evaluation["pair_outputs"][role]
        rows = evaluation["role_pair_rows"][role]
        payload[f"{role}_pair_ids"] = np.asarray(
            [str(row["pair_id"]) for row in rows], dtype=np.str_
        )
        payload[f"{role}_role_indices"] = np.asarray(
            evaluation["role_indices"][role], dtype=np.int64
        )
        payload[f"{role}_logits"] = np.asarray(output["logits"], dtype=np.float32)
        payload[f"{role}_probabilities"] = np.asarray(
            output["probabilities"], dtype=np.float32
        )
    return payload


def run_unit(
    config_path: str | Path,
    bundle_root: str | Path,
    r2_results_root: str | Path,
    phase_root: str | Path,
    oracle_audit_path: str | Path,
    *,
    phase: str,
    pair_id: str,
    method: str,
    device_request: str = "auto",
    smoke_root: str | Path | None = None,
) -> dict[str, Any]:
    config = load_official_cssr_config(config_path)
    project_root = _bound_project_root(config)
    oracle_audit = _official_audit_record(oracle_audit_path, config)
    plan = build_phase_plan(config, phase)
    planned = {(str(row["pair_id"]), str(row["method"])) for row in plan}
    if (pair_id, method) not in planned:
        raise DataValidationError("unit is outside the frozen official CSSR plan")
    if phase == "pilot":
        smoke_authorization = _read_smoke_authorization(
            smoke_root,
            config=config,
            oracle_audit=oracle_audit,
        )
    else:
        if smoke_root is not None:
            raise DataValidationError("smoke units cannot consume a smoke authorization")
        smoke_authorization = None
    root = Path(phase_root).resolve()
    destination = _unit_destination(root, pair_id, method)
    if destination.exists():
        raise DataValidationError(f"official CSSR output already exists: {destination}")
    staging = destination.parent / f".{method}.staging"
    if staging.exists():
        raise DataValidationError(f"stale official CSSR staging output exists: {staging}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_hashes = _task_source_hashes(project_root)
    device = _resolve_device(device_request)
    runtime = _configure_runtime(config, device)
    environment = _git_environment(project_root, device)
    if environment["git_status_porcelain"] != "":
        raise DataValidationError("formal official CSSR run requires a clean checkout")
    if _git_commit_source_hashes(
        project_root,
        str(environment["git_commit"]),
        source_hashes,
    ) != source_hashes:
        raise DataValidationError("formal source bytes differ from the recorded commit")
    started = time.perf_counter()

    prior_config = _load_prior_config(project_root, config)
    bundle = _load_development_only_bundle(bundle_root, config)
    prepared = _prepare_frozen_split(bundle, prior_config, config, pair_id)
    r2_model, frozen_r2_arrays, r2_audit = load_and_audit_frozen_r2(
        project_root=project_root,
        r2_results_root=r2_results_root,
        pair_id=pair_id,
        config=config,
        prepared=prepared,
        prior_config=prior_config,
        device=device,
    )
    if r2_model.training or any(parameter.requires_grad for parameter in r2_model.parameters()):
        raise DataValidationError("frozen R2 is trainable or in train mode")
    r2_state_before = _state_sha256(r2_model.state_dict())
    unique_rows = build_unique_base_sample_manifest(prepared, bundle)
    normalized_inputs, normalization_audit = _fit_input_normalization(
        bundle=bundle,
        unique_rows=unique_rows,
        prepared=prepared,
    )
    train_rows, train_indices = _selected_rows(unique_rows, "train_known")
    calibration_rows, calibration_indices = _selected_rows(
        unique_rows, "known_calibration"
    )
    train_labels = np.asarray(
        [int(row["model_label"]) for row in train_rows], dtype=np.int64
    )
    calibration_labels = np.asarray(
        [int(row["model_label"]) for row in calibration_rows], dtype=np.int64
    )
    pair_inputs_by_role = {
        role: _materialize_pair_inputs(
            bundle=bundle,
            rows=_role_manifest_rows(prepared, role),
            mean=float(normalization_audit["mean"]),
            std=float(normalization_audit["std"]),
        )
        for role in ("known_calibration", "surrogate_unknown")
    }
    model, initialization = _build_model(
        method=method,
        pair_id=pair_id,
        r2_model=r2_model,
        config=config,
        device=device,
    )
    diagnostic_role_indices = _evaluation_role_indices(
        prepared,
        smoke=phase == "smoke",
        config=config,
    )
    diagnostic_calibration_indices = diagnostic_role_indices["known_calibration"]
    diagnostic_pair_inputs = pair_inputs_by_role["known_calibration"][
        diagnostic_calibration_indices
    ]
    diagnostic_pair_labels = np.asarray(
        prepared.labels["known_calibration"], dtype=np.int64
    )[diagnostic_calibration_indices]
    if phase == "smoke":
        diagnostic_single_inputs = diagnostic_pair_inputs.reshape(-1, 601)
        diagnostic_single_labels = np.repeat(diagnostic_pair_labels, 2)
    else:
        diagnostic_single_inputs = normalized_inputs[calibration_indices]
        diagnostic_single_labels = calibration_labels
    training_result = _train_model(
        model=model,
        method=method,
        pair_id=pair_id,
        train_rows=train_rows,
        train_inputs=normalized_inputs[train_indices],
        train_labels=train_labels,
        calibration_inputs=diagnostic_single_inputs,
        calibration_labels=diagnostic_single_labels,
        calibration_pair_inputs=diagnostic_pair_inputs,
        calibration_pair_labels=diagnostic_pair_labels,
        config=config,
        phase=phase,
        device=device,
    )
    model = training_result["model"]
    statistics = None
    templates = None
    score_normalization = None
    if method in CSSR_METHODS:
        statistics = _fit_official_statistics(
            model=model,
            train_rows=train_rows,
            train_inputs=normalized_inputs[train_indices],
            pair_id=pair_id,
            config=config,
            device=device,
        )
        templates = statistics["templates"]
        score_normalization = statistics["normalization"]
        train_inference = statistics["raw_train_inference"]
    else:
        train_inference = _infer_single(
            model,
            normalized_inputs[train_indices],
            device=device,
            batch_size=int(config["training"]["batch_size"]),
        )
    evaluation = _evaluate_model(
        model=model,
        method=method,
        pair_id=pair_id,
        prepared=prepared,
        pair_inputs_by_role=pair_inputs_by_role,
        frozen_r2_arrays=frozen_r2_arrays,
        config=config,
        device=device,
        smoke=phase == "smoke",
        templates=templates,
        normalization=score_normalization,
    )
    o0 = _evaluate_o0(
        prepared=prepared,
        frozen_r2_arrays=frozen_r2_arrays,
        pair_id=pair_id,
        config=config,
        smoke=phase == "smoke",
    )
    r2_state_after = _state_sha256(r2_model.state_dict())
    if r2_state_before != r2_state_after:
        raise DataValidationError("frozen R2 changed during official CSSR training")
    if _task_source_hashes(project_root) != source_hashes:
        raise DataValidationError("official CSSR source changed during unit execution")

    checkpoint = _checkpoint_payload(
        model=model,
        method=method,
        pair_id=pair_id,
        phase=phase,
        config=config,
        prepared=prepared,
        unique_rows=unique_rows,
        initialization=initialization,
        training_result=training_result,
        templates=templates,
        normalization=score_normalization,
        r2_audit=r2_audit,
        source_hashes=source_hashes,
        smoke_authorization=smoke_authorization,
        code_commit=str(environment["git_commit"]),
    )
    replay_model, replay_initialization = _build_model(
        method=method,
        pair_id=pair_id,
        r2_model=r2_model,
        config=config,
        device=device,
    )
    if replay_initialization != initialization:
        raise DataValidationError("deterministic model initialization did not replay")
    incompatible = replay_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise DataValidationError("formal checkpoint was not strict-load compatible")
    replay_model.eval()
    replay_templates, replay_normalization = _checkpoint_templates(
        checkpoint,
        device=device,
    )
    replay = _evaluate_model(
        model=replay_model,
        method=method,
        pair_id=pair_id,
        prepared=prepared,
        pair_inputs_by_role=pair_inputs_by_role,
        frozen_r2_arrays=frozen_r2_arrays,
        config=config,
        device=device,
        smoke=phase == "smoke",
        templates=replay_templates,
        normalization=replay_normalization,
    )
    _same_prediction_rows(evaluation["prediction_rows"], replay["prediction_rows"])
    _metrics_exact(evaluation["metrics"], replay["metrics"], context="checkpoint replay")
    original_inference = _evaluation_inference_arrays(evaluation)
    replay_inference = _evaluation_inference_arrays(replay)
    if set(original_inference) != set(replay_inference) or any(
        not np.array_equal(original_inference[name], replay_inference[name])
        for name in original_inference
    ):
        raise DataValidationError(
            "checkpoint replay changed per-view evaluation logits/probabilities"
        )
    if evaluation["view_swap_audit"] != replay["view_swap_audit"]:
        raise DataValidationError("checkpoint replay changed the view-swap audit")

    data_access_audit = _profile_access_audit(bundle)

    end_environment = _git_environment(project_root, device)
    if (
        end_environment["git_commit"] != environment["git_commit"]
        or end_environment["git_status_porcelain"] != ""
        or _task_source_hashes(project_root) != source_hashes
    ):
        raise DataValidationError("checkout or source changed during official CSSR run")
    environment = end_environment
    environment["runtime_contract"] = runtime
    environment["task_source_hashes"] = source_hashes
    unit_summary = {
        "status": "complete",
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "pair_id": pair_id,
        "angle_fold": ANGLE_FOLD,
        "seed": OFFICIAL_CSSR_SEED,
        "method": method,
        "code_commit": environment["git_commit"],
        "metrics": {key: float(value) for key, value in evaluation["metrics"].items()},
        "o0_metrics": {key: float(value) for key, value in o0["metrics"].items()},
        "training_audit": training_result["audit"],
        "official_statistics": None if statistics is None else statistics["audit"],
        "checkpoint_replay": "exact",
        "r2_unchanged": True,
        "data_access_audit": data_access_audit,
        "smoke_authorization": smoke_authorization,
        "wall_time_seconds": time.perf_counter() - started,
        "diagnostic_only": phase == "smoke",
        "known_calibration_used_for_training": False,
        "surrogate_unknown_used_for_training": False,
        "final_unknown_used": False,
        "even_angle_test_used": False,
        "confirmation_allowed": False,
        "final_unknown_test_authorized": False,
    }
    for name, value in (
        ("unit summary", unit_summary),
        ("training audit", training_result["audit"]),
        ("schedule audits", training_result["schedule_audits"]),
        ("training log", training_result["training_log"]),
        ("method metrics", evaluation["metrics"]),
        ("O0 metrics", o0["metrics"]),
        ("identity metrics", evaluation["identity_rows"]),
        ("absorption metrics", evaluation["absorption_rows"]),
        ("score ablations", evaluation["score_ablation_rows"]),
        ("error analysis", evaluation["error_analysis"]),
    ):
        _assert_finite_tree(value, context=name)

    staging.mkdir(parents=True, exist_ok=False)
    _atomic_write_bytes(staging / "resolved_config.yaml", _resolved_config_bytes(config))
    _atomic_write_bytes(staging / "pair_manifest.csv", prepared.pair_manifest_bytes)
    _write_csv(staging / "unique_base_manifest.csv", unique_rows)
    _write_json(
        staging / "label_order.json",
        {
            "train_class_order": list(prepared.train_class_order),
            "surrogate_class_order": list(prepared.surrogate_class_order),
            "train_label_order_sha256": _array_sha256(train_labels),
            "known_calibration_label_order_sha256": _array_sha256(
                np.asarray(prepared.labels["known_calibration"], dtype=np.int64)
            ),
            "surrogate_label_order_sha256": _array_sha256(
                np.asarray(prepared.labels["surrogate_unknown"], dtype=np.int64)
            ),
            "final_unknown_used": False,
            "even_angle_test_used": False,
        },
    )
    _write_json(staging / "normalization.json", normalization_audit)
    _write_json(staging / "data_access_audit.json", data_access_audit)
    _write_json(staging / "official_oracle_audit.json", oracle_audit)
    _write_json(staging / "r2_audit.json", r2_audit)
    _write_json(staging / "initialization_audit.json", initialization)
    _write_json(staging / "training_audit.json", training_result["audit"])
    _write_json(staging / "schedule_audits.json", training_result["schedule_audits"])
    _save_training_log(staging / "training_log.jsonl", training_result["training_log"])
    _atomic_torch_save(staging / "checkpoint.pt", checkpoint)
    _save_npz(
        staging / "score_statistics.npz",
        _serialize_templates(templates, score_normalization),
    )
    _save_npz(
        staging / "raw_train_inference.npz",
        _train_inference_arrays(train_inference, train_rows),
    )
    _save_npz(
        staging / "evaluation_inference.npz",
        _evaluation_inference_arrays(evaluation),
    )
    _write_json(
        staging / "score_statistics_audit.json",
        {"status": "not_applicable_linear"}
        if statistics is None
        else statistics["audit"],
    )
    _write_csv(staging / "predictions.csv", evaluation["prediction_rows"])
    _write_csv(staging / "o0_predictions.csv", o0["prediction_rows"])
    _write_csv(staging / "identity_metrics.csv", evaluation["identity_rows"])
    _write_csv(staging / "o0_identity_metrics.csv", o0["identity_rows"])
    _write_csv(staging / "absorption_by_known_class.csv", evaluation["absorption_rows"])
    _write_csv(staging / "o0_absorption_by_known_class.csv", o0["absorption_rows"])
    _write_json(staging / "error_analysis.json", evaluation["error_analysis"])
    _write_json(staging / "o0_error_analysis.json", o0["error_analysis"])
    _write_json(staging / "metrics.json", evaluation["metrics"])
    _write_json(staging / "o0_metrics.json", o0["metrics"])
    if evaluation["score_ablation_rows"]:
        _write_csv(staging / "score_ablation_metrics.csv", evaluation["score_ablation_rows"])
    else:
        _write_json(staging / "score_ablation_metrics.json", {"status": "not_applicable_linear"})
    _save_npz(staging / "checkpoint_replay.npz", _replay_arrays(evaluation["prediction_rows"]))
    _write_json(
        staging / "checkpoint_replay_audit.json",
        {
            "status": "passed",
            "state_dict_strict_load": True,
            "prediction_rows_exact": True,
            "metrics_exact": True,
            "all_prediction_score_fields_exact": True,
            "evaluation_logits_probabilities_exact": True,
            "view_swap_checked_by_score_module": True,
            "real_pair_view_swap_reinference": True,
        },
    )
    _write_json(staging / "view_swap_audit.json", evaluation["view_swap_audit"])
    _write_json(staging / "environment.json", environment)
    _write_json(
        staging / "unit_contract.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "phase": phase,
            "pair_id": pair_id,
            "method": method,
            "code_commit": environment["git_commit"],
            "config_sha256": config["_config_sha256"],
            "source_hashes": source_hashes,
            "data_access_policy": data_access_audit["policy"],
            "authorized_row_indices_sha256": data_access_audit[
                "authorized_row_indices_sha256"
            ],
            "smoke_authorization": smoke_authorization,
            "official_commit": config["official_reference"]["commit"],
            "pair_manifest_sha256": prepared.pair_manifest_sha256,
            "unique_base_manifest_sha256": file_sha256(
                staging / "unique_base_manifest.csv"
            ),
            "checkpoint_sha256": file_sha256(staging / "checkpoint.pt"),
            "final_unknown_used": False,
            "even_angle_test_used": False,
            "confirmation_allowed": False,
            "final_unknown_test_authorized": False,
        },
    )
    _write_json(staging / "unit_summary.json", unit_summary)
    hashes = _artifact_hashes(staging)
    _write_json(staging / "artifact_hashes.json", hashes)
    _write_json(
        staging / "_SUCCESS.json",
        {
            "status": "success",
            "artifact_count": len(hashes),
            "artifact_hashes_sha256": file_sha256(staging / "artifact_hashes.json"),
            "checkpoint_replay": "exact",
            "final_unknown_used": False,
            "even_angle_test_used": False,
        },
    )
    staging.replace(destination)
    return {**unit_summary, "destination": str(destination)}


def _csv_rows_exact(
    observed: Sequence[Mapping[str, Any]],
    expected: Sequence[Mapping[str, Any]],
    *,
    context: str,
) -> None:
    """Compare rows after applying the same scalar conversion as csv.DictWriter."""

    if len(observed) != len(expected):
        raise DataValidationError(
            f"{context} row count changed: {len(observed)} != {len(expected)}"
        )
    for index, (left, right) in enumerate(zip(observed, expected, strict=True)):
        if list(left) != list(right):
            raise DataValidationError(f"{context} columns changed at row {index}")
        converted = {
            str(key): "" if value is None else str(value)
            for key, value in right.items()
        }
        if dict(left) != converted:
            differing = [key for key in left if left[key] != converted[key]]
            raise DataValidationError(
                f"{context} changed at row {index}: {differing[:5]}"
            )


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise DataValidationError(
                    f"blank training-log row at line {line_number}"
                )
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise DataValidationError(
                    f"invalid training-log row at line {line_number}"
                )
            rows.append(dict(value))
    return rows


def _require_unit_files(root: Path, *, method: str) -> None:
    required = {
        "resolved_config.yaml",
        "pair_manifest.csv",
        "unique_base_manifest.csv",
        "label_order.json",
        "normalization.json",
        "data_access_audit.json",
        "official_oracle_audit.json",
        "r2_audit.json",
        "initialization_audit.json",
        "training_audit.json",
        "schedule_audits.json",
        "training_log.jsonl",
        "checkpoint.pt",
        "score_statistics.npz",
        "raw_train_inference.npz",
        "evaluation_inference.npz",
        "score_statistics_audit.json",
        "predictions.csv",
        "o0_predictions.csv",
        "identity_metrics.csv",
        "o0_identity_metrics.csv",
        "absorption_by_known_class.csv",
        "o0_absorption_by_known_class.csv",
        "error_analysis.json",
        "o0_error_analysis.json",
        "metrics.json",
        "o0_metrics.json",
        "checkpoint_replay.npz",
        "checkpoint_replay_audit.json",
        "view_swap_audit.json",
        "environment.json",
        "unit_contract.json",
        "unit_summary.json",
        "artifact_hashes.json",
        "_SUCCESS.json",
    }
    required.add(
        "score_ablation_metrics.csv"
        if method in CSSR_METHODS
        else "score_ablation_metrics.json"
    )
    observed = {path.name for path in root.iterdir() if path.is_file()}
    if observed != required:
        raise DataValidationError(
            "official CSSR unit file set changed: "
            f"missing={sorted(required - observed)}, extra={sorted(observed - required)}"
        )


def audit_unit_result(
    destination: str | Path,
    *,
    config: Mapping[str, Any],
    bundle_root: str | Path,
    r2_results_root: str | Path,
    oracle_audit_path: str | Path,
    phase: str,
    pair_id: str,
    method: str,
    device_request: str = "auto",
    smoke_root: str | Path | None = None,
) -> dict[str, Any]:
    """Rebuild every decision-bearing unit output from frozen inputs/checkpoint."""

    project_root = _bound_project_root(config)
    planned = {
        (str(row["pair_id"]), str(row["method"]))
        for row in build_phase_plan(config, phase)
    }
    if (pair_id, method) not in planned:
        raise DataValidationError("audit unit is outside the frozen plan")
    if phase == "pilot":
        current_oracle_audit = _official_audit_record(oracle_audit_path, config)
        smoke_authorization = _read_smoke_authorization(
            smoke_root,
            config=config,
            oracle_audit=current_oracle_audit,
        )
    else:
        if smoke_root is not None:
            raise DataValidationError("smoke audit cannot consume a smoke authorization")
        smoke_authorization = None
    root = Path(destination).resolve()
    if not root.is_dir():
        raise DataValidationError(f"official CSSR unit does not exist: {root}")
    _require_unit_files(root, method=method)

    success = _read_json(root / "_SUCCESS.json")
    recorded_hashes = _read_json(root / "artifact_hashes.json")
    if (
        success.get("status") != "success"
        or int(success.get("artifact_count", -1)) != len(recorded_hashes)
        or success.get("artifact_hashes_sha256")
        != file_sha256(root / "artifact_hashes.json")
        or recorded_hashes != _artifact_hashes(root)
        or success.get("checkpoint_replay") != "exact"
        or success.get("final_unknown_used") is not False
        or success.get("even_angle_test_used") is not False
    ):
        raise DataValidationError("official CSSR unit artifact hash audit failed")
    if (root / "resolved_config.yaml").read_bytes() != _resolved_config_bytes(config):
        raise DataValidationError("resolved official CSSR config changed")

    source_hashes = _task_source_hashes(project_root)
    environment = _read_json(root / "environment.json")
    recorded_commit = str(environment.get("git_commit", ""))
    if _git_commit_source_hashes(
        project_root, recorded_commit, source_hashes
    ) != source_hashes:
        raise DataValidationError("recorded Git commit does not reproduce source hashes")
    contract = _read_json(root / "unit_contract.json")
    expected_contract = {
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "pair_id": pair_id,
        "method": method,
        "code_commit": recorded_commit,
        "config_sha256": config["_config_sha256"],
        "source_hashes": source_hashes,
        "smoke_authorization": smoke_authorization,
        "official_commit": config["official_reference"]["commit"],
        "final_unknown_used": False,
        "even_angle_test_used": False,
        "confirmation_allowed": False,
        "final_unknown_test_authorized": False,
    }
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            raise DataValidationError(f"official CSSR unit contract changed at {key}")

    official_record = (
        current_oracle_audit
        if phase == "pilot"
        else _official_audit_record(oracle_audit_path, config)
    )
    if _official_audit_identity(
        _read_json(root / "official_oracle_audit.json")
    ) != _official_audit_identity(official_record):
        raise DataValidationError("saved official oracle audit changed")
    device = _resolve_device(device_request)
    current_runtime = _configure_runtime(config, device)
    prior_config = _load_prior_config(project_root, config)
    bundle = _load_development_only_bundle(bundle_root, config)
    prepared = _prepare_frozen_split(bundle, prior_config, config, pair_id)
    if (root / "pair_manifest.csv").read_bytes() != prepared.pair_manifest_bytes:
        raise DataValidationError("saved pair manifest changed")
    if contract.get("pair_manifest_sha256") != prepared.pair_manifest_sha256:
        raise DataValidationError("pair manifest contract hash changed")
    expected_access_audit = _profile_access_audit(bundle)
    if _read_json(root / "data_access_audit.json") != expected_access_audit:
        raise DataValidationError("development-only profile access audit changed")
    if (
        contract.get("data_access_policy") != expected_access_audit["policy"]
        or contract.get("authorized_row_indices_sha256")
        != expected_access_audit["authorized_row_indices_sha256"]
    ):
        raise DataValidationError("development-only profile access contract changed")

    unique_rows = build_unique_base_sample_manifest(prepared, bundle)
    saved_unique_rows = _read_csv(root / "unique_base_manifest.csv")
    _csv_rows_exact(saved_unique_rows, unique_rows, context="unique base manifest")
    role_counts = Counter(str(row["experiment_role"]) for row in unique_rows)
    if role_counts != Counter(
        {"train_known": 720, "known_calibration": 180, "surrogate_unknown": 72}
    ):
        raise DataValidationError("official CSSR unique-base roles changed")
    if (
        len({str(row["sample_id"]) for row in unique_rows}) != 972
        or any(int(row["angle_deg"]) % 2 == 0 for row in unique_rows)
    ):
        raise DataValidationError("duplicate or even-angle base entered official CSSR")
    if contract.get("unique_base_manifest_sha256") != file_sha256(
        root / "unique_base_manifest.csv"
    ):
        raise DataValidationError("unique-base manifest contract hash changed")

    normalized_inputs, normalization_audit = _fit_input_normalization(
        bundle=bundle,
        unique_rows=unique_rows,
        prepared=prepared,
    )
    if _read_json(root / "normalization.json") != normalization_audit:
        raise DataValidationError("normalization audit does not reproduce")
    train_rows, train_indices = _selected_rows(unique_rows, "train_known")
    train_labels = np.asarray(
        [int(row["model_label"]) for row in train_rows], dtype=np.int64
    )
    expected_label_order = {
        "train_class_order": list(prepared.train_class_order),
        "surrogate_class_order": list(prepared.surrogate_class_order),
        "train_label_order_sha256": _array_sha256(train_labels),
        "known_calibration_label_order_sha256": _array_sha256(
            np.asarray(prepared.labels["known_calibration"], dtype=np.int64)
        ),
        "surrogate_label_order_sha256": _array_sha256(
            np.asarray(prepared.labels["surrogate_unknown"], dtype=np.int64)
        ),
        "final_unknown_used": False,
        "even_angle_test_used": False,
    }
    if _read_json(root / "label_order.json") != expected_label_order:
        raise DataValidationError("label order audit does not reproduce")
    pair_inputs_by_role = {
        role: _materialize_pair_inputs(
            bundle=bundle,
            rows=_role_manifest_rows(prepared, role),
            mean=float(normalization_audit["mean"]),
            std=float(normalization_audit["std"]),
        )
        for role in ("known_calibration", "surrogate_unknown")
    }

    r2_model, frozen_r2_arrays, r2_audit = load_and_audit_frozen_r2(
        project_root=project_root,
        r2_results_root=r2_results_root,
        pair_id=pair_id,
        config=config,
        prepared=prepared,
        prior_config=prior_config,
        device=device,
    )
    if _r2_audit_identity(
        _read_json(root / "r2_audit.json")
    ) != _r2_audit_identity(r2_audit):
        raise DataValidationError("frozen R2 audit does not reproduce")
    checkpoint = torch.load(
        root / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    expected_epoch = 1 if phase == "smoke" else 40
    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "pair_id": pair_id,
        "angle_fold": ANGLE_FOLD,
        "method": method,
        "official_cssr_seed": OFFICIAL_CSSR_SEED,
        "checkpoint_epoch": expected_epoch,
        "formal_checkpoint": phase == "pilot",
        "checkpoint_selection": "fixed_final_epoch",
        "config_sha256": config["_config_sha256"],
        "code_commit": recorded_commit,
        "source_hashes": source_hashes,
        "smoke_authorization": smoke_authorization,
        "pair_manifest_sha256": prepared.pair_manifest_sha256,
        "r2_checkpoint_sha256": r2_audit["checkpoint_sha256"],
        "known_calibration_used_for_training": False,
        "surrogate_unknown_used_for_training": False,
        "final_unknown_used": False,
        "even_angle_test_used": False,
    }
    for key, expected in metadata.items():
        if checkpoint.get(key) != expected:
            raise DataValidationError(f"official CSSR checkpoint changed at {key}")
    if contract.get("checkpoint_sha256") != file_sha256(root / "checkpoint.pt"):
        raise DataValidationError("checkpoint contract hash changed")

    model, initialization = _build_model(
        method=method,
        pair_id=pair_id,
        r2_model=r2_model,
        config=config,
        device=device,
    )
    if checkpoint.get("initialization") != initialization:
        raise DataValidationError("checkpoint initialization record changed")
    if _read_json(root / "initialization_audit.json") != initialization:
        raise DataValidationError("saved initialization audit changed")
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise DataValidationError("official CSSR checkpoint strict-load failed")
    if not all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
        raise DataValidationError("official CSSR checkpoint contains NaN or Inf")
    model.eval()
    templates, score_normalization = _checkpoint_templates(checkpoint, device=device)
    serialized_statistics = _serialize_templates(templates, score_normalization)
    with np.load(root / "score_statistics.npz", allow_pickle=False) as saved:
        if set(saved.files) != set(serialized_statistics):
            raise DataValidationError("score-statistics keys changed")
        for name, expected in serialized_statistics.items():
            if not np.array_equal(saved[name], expected):
                raise DataValidationError(f"score-statistics changed at {name}")
    if method in CSSR_METHODS:
        recomputed_statistics = _fit_official_statistics(
            model=model,
            train_rows=train_rows,
            train_inputs=normalized_inputs[train_indices],
            pair_id=pair_id,
            config=config,
            device=device,
        )
        recomputed_serialized = _serialize_templates(
            recomputed_statistics["templates"],
            recomputed_statistics["normalization"],
        )
        for name, expected in serialized_statistics.items():
            if not np.array_equal(recomputed_serialized[name], expected):
                raise DataValidationError(
                    f"raw-train score-statistic recomputation changed at {name}"
                )
        if _read_json(root / "score_statistics_audit.json") != recomputed_statistics[
            "audit"
        ]:
            raise DataValidationError("score-statistics audit does not reproduce")
        recomputed_train_inference = recomputed_statistics["raw_train_inference"]
    elif _read_json(root / "score_statistics_audit.json") != {
        "status": "not_applicable_linear"
    }:
        raise DataValidationError("linear score-statistics marker changed")
    else:
        recomputed_train_inference = _infer_single(
            model,
            normalized_inputs[train_indices],
            device=device,
            batch_size=int(config["training"]["batch_size"]),
        )

    evaluation = _evaluate_model(
        model=model,
        method=method,
        pair_id=pair_id,
        prepared=prepared,
        pair_inputs_by_role=pair_inputs_by_role,
        frozen_r2_arrays=frozen_r2_arrays,
        config=config,
        device=device,
        smoke=phase == "smoke",
        templates=templates,
        normalization=score_normalization,
    )
    for filename, expected_arrays in (
        (
            "raw_train_inference.npz",
            _train_inference_arrays(recomputed_train_inference, train_rows),
        ),
        ("evaluation_inference.npz", _evaluation_inference_arrays(evaluation)),
    ):
        with np.load(root / filename, allow_pickle=False) as saved:
            if set(saved.files) != set(expected_arrays):
                raise DataValidationError(f"{filename} keys changed")
            for name, expected in expected_arrays.items():
                if not np.array_equal(saved[name], expected):
                    raise DataValidationError(f"{filename} changed at {name}")
    o0 = _evaluate_o0(
        prepared=prepared,
        frozen_r2_arrays=frozen_r2_arrays,
        pair_id=pair_id,
        config=config,
        smoke=phase == "smoke",
    )
    saved_predictions = _read_csv(root / "predictions.csv")
    saved_o0_predictions = _read_csv(root / "o0_predictions.csv")
    _csv_rows_exact(
        saved_predictions, evaluation["prediction_rows"], context="official predictions"
    )
    _csv_rows_exact(saved_o0_predictions, o0["prediction_rows"], context="O0 predictions")
    saved_metrics = _read_json(root / "metrics.json")
    saved_o0_metrics = _read_json(root / "o0_metrics.json")
    _metrics_exact(saved_metrics, evaluation["metrics"], context="saved official metrics")
    _metrics_exact(saved_o0_metrics, o0["metrics"], context="saved O0 metrics")
    _metrics_exact(
        saved_metrics,
        recompute_method_metrics_from_prediction_rows(
            saved_predictions,
            known_acceptance_rate=float(
                config["evaluation"]["threshold_known_acceptance_rate"]
            ),
        ),
        context="prediction-row metric recomputation",
    )
    _metrics_exact(
        saved_o0_metrics,
        recompute_method_metrics_from_prediction_rows(
            saved_o0_predictions,
            known_acceptance_rate=float(
                config["evaluation"]["threshold_known_acceptance_rate"]
            ),
        ),
        context="O0 prediction-row metric recomputation",
    )
    for filename, expected_rows in (
        ("identity_metrics.csv", evaluation["identity_rows"]),
        ("o0_identity_metrics.csv", o0["identity_rows"]),
        ("absorption_by_known_class.csv", evaluation["absorption_rows"]),
        ("o0_absorption_by_known_class.csv", o0["absorption_rows"]),
    ):
        _csv_rows_exact(_read_csv(root / filename), expected_rows, context=filename)
    if _read_json(root / "error_analysis.json") != evaluation["error_analysis"]:
        raise DataValidationError("official error analysis does not reproduce")
    if _read_json(root / "o0_error_analysis.json") != o0["error_analysis"]:
        raise DataValidationError("O0 error analysis does not reproduce")
    if method in CSSR_METHODS:
        _csv_rows_exact(
            _read_csv(root / "score_ablation_metrics.csv"),
            evaluation["score_ablation_rows"],
            context="score ablation metrics",
        )
    elif _read_json(root / "score_ablation_metrics.json") != {
        "status": "not_applicable_linear"
    }:
        raise DataValidationError("linear score-ablation marker changed")

    training_audit = _read_json(root / "training_audit.json")
    if training_audit != checkpoint["training_audit"]:
        raise DataValidationError("training audit differs from checkpoint")
    if (
        training_audit.get("status") != "passed"
        or int(training_audit.get("epochs", -1)) != expected_epoch
        or int(training_audit.get("train_unique_base_count", -1)) != 720
        or int(training_audit.get("optimizer_updates", -1)) != expected_epoch * 6
        or int(training_audit.get("diagnostic_single_sample_count", -1))
        != (20 if phase == "smoke" else 180)
        or int(training_audit.get("diagnostic_pair_count", -1))
        != (10 if phase == "smoke" else 2500)
        or training_audit.get("diagnostic_evaluation_population")
        != (
            "stable_first_two_pairs_per_known_class_views"
            if phase == "smoke"
            else "full_known_calibration"
        )
        or training_audit.get("known_calibration_used_for_training") is not False
        or training_audit.get("surrogate_unknown_used_for_training") is not False
        or training_audit.get("final_unknown_used") is not False
        or training_audit.get("even_angle_test_used") is not False
    ):
        raise DataValidationError("official CSSR training audit failed")
    log_rows = _jsonl_rows(root / "training_log.jsonl")
    schedules = _read_json(root / "schedule_audits.json")
    if len(log_rows) != expected_epoch or len(schedules) != expected_epoch:
        raise DataValidationError("training log or schedule length changed")
    for epoch, (log_row, schedule) in enumerate(
        zip(log_rows, schedules, strict=True), start=1
    ):
        material = build_training_epoch_material(
            train_rows,
            normalized_inputs[train_indices],
            phase=phase,
            pair_id=pair_id,
            epoch=epoch,
            config=config,
        )
        expected_schedule = {
            key: _mapping_to_json(value)
            for key, value in material.items()
            if key not in {"indices", "gain", "noise", "augmented_inputs"}
        }
        if schedule != expected_schedule:
            raise DataValidationError(f"training schedule {epoch} does not reproduce")
        if log_row.get("schedule_sha256") != material["schedule_sha256"]:
            raise DataValidationError(f"training log schedule {epoch} changed")

    replay = _read_json(root / "checkpoint_replay_audit.json")
    if replay != {
        "status": "passed",
        "state_dict_strict_load": True,
        "prediction_rows_exact": True,
        "metrics_exact": True,
        "all_prediction_score_fields_exact": True,
        "evaluation_logits_probabilities_exact": True,
        "view_swap_checked_by_score_module": True,
        "real_pair_view_swap_reinference": True,
    }:
        raise DataValidationError("checkpoint replay audit marker changed")
    if _read_json(root / "view_swap_audit.json") != evaluation["view_swap_audit"]:
        raise DataValidationError("real pair view-swap audit does not reproduce")
    with np.load(root / "checkpoint_replay.npz", allow_pickle=False) as data:
        expected_replay = _replay_arrays(evaluation["prediction_rows"])
        if set(data.files) != set(expected_replay):
            raise DataValidationError("checkpoint replay keys changed")
        for name, expected in expected_replay.items():
            if not np.array_equal(data[name], expected):
                raise DataValidationError(f"checkpoint replay changed at {name}")

    if (
        not isinstance(environment.get("git_commit"), str)
        or len(str(environment.get("git_commit"))) != 40
        or environment.get("git_status_porcelain") != ""
        or environment.get("task_source_hashes") != source_hashes
    ):
        raise DataValidationError("official CSSR runtime environment changed or was dirty")
    _assert_runtime_contract_exact(
        environment.get("runtime_contract"), current_runtime
    )
    summary = _read_json(root / "unit_summary.json")
    if (
        summary.get("status") != "complete"
        or summary.get("phase") != phase
        or summary.get("pair_id") != pair_id
        or summary.get("method") != method
        or summary.get("code_commit") != recorded_commit
        or summary.get("smoke_authorization") != smoke_authorization
        or summary.get("checkpoint_replay") != "exact"
        or summary.get("r2_unchanged") is not True
        or summary.get("data_access_audit") != expected_access_audit
        or summary.get("metrics") != saved_metrics
        or summary.get("o0_metrics") != saved_o0_metrics
        or summary.get("final_unknown_used") is not False
        or summary.get("even_angle_test_used") is not False
    ):
        raise DataValidationError("official CSSR unit summary changed")
    return {
        "status": "success",
        "audit_passed": True,
        "phase": phase,
        "pair_id": pair_id,
        "method": method,
        "destination": str(root),
        "artifact_count": len(recorded_hashes),
        "source_hashes": source_hashes,
        "code_commit": recorded_commit,
        "smoke_authorization": smoke_authorization,
        "pair_manifest_sha256": prepared.pair_manifest_sha256,
        "unique_base_manifest_sha256": file_sha256(root / "unique_base_manifest.csv"),
        "checkpoint_sha256": file_sha256(root / "checkpoint.pt"),
        "checkpoint_replay": "exact",
        "metrics": saved_metrics,
        "o0_metrics": saved_o0_metrics,
        "identity_rows": evaluation["identity_rows"],
        "o0_identity_rows": o0["identity_rows"],
        "absorption_rows": evaluation["absorption_rows"],
        "o0_absorption_rows": o0["absorption_rows"],
        "score_ablation_rows": evaluation["score_ablation_rows"],
        "initialization": initialization,
        "training_audit": training_audit,
        "final_unknown_used": False,
        "even_angle_test_used": False,
    }


def _aggregate_rows(
    audits: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    audit_map = {
        (str(row["pair_id"]), str(row["method"])): row for row in audits
    }
    if len(audit_map) != len(audits):
        raise DataValidationError("pilot audit population contains a duplicate task")
    expected = {
        (pair_id, method) for pair_id in PILOT_PAIRS for method in TRAINABLE_METHODS
    }
    observed = set(audit_map)
    if observed != expected:
        raise DataValidationError(
            f"pilot audit population changed: missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}"
        )
    metrics: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    absorption: list[dict[str, Any]] = []
    ablations: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    integrity: list[dict[str, Any]] = []
    for pair_id in PILOT_PAIRS:
        pair_audits = [audit_map[(pair_id, method)] for method in TRAINABLE_METHODS]
        first = pair_audits[0]
        for current in pair_audits[1:]:
            _metrics_exact(
                first["o0_metrics"], current["o0_metrics"], context=f"{pair_id} O0"
            )
            if (
                current["o0_identity_rows"] != first["o0_identity_rows"]
                or current["o0_absorption_rows"] != first["o0_absorption_rows"]
                or current["pair_manifest_sha256"] != first["pair_manifest_sha256"]
                or current["unique_base_manifest_sha256"]
                != first["unique_base_manifest_sha256"]
            ):
                raise DataValidationError(f"shared O0/data evidence differs for {pair_id}")
        metrics.append(_metric_row(pair_id, O0_R2_CC_MLS, first["o0_metrics"]))
        identities.extend(first["o0_identity_rows"])
        absorption.extend(first["o0_absorption_rows"])
        for current in pair_audits:
            metrics.append(_metric_row(pair_id, current["method"], current["metrics"]))
            identities.extend(current["identity_rows"])
            absorption.extend(current["absorption_rows"])
            tasks.append(
                {
                    "pair_id": pair_id,
                    "method": current["method"],
                    "status": current["status"],
                    "audit_passed": current["audit_passed"],
                    "checkpoint_replay": current["checkpoint_replay"],
                    "artifact_count": current["artifact_count"],
                    "checkpoint_sha256": current["checkpoint_sha256"],
                    "failure_type": "",
                    "failure_message": "",
                }
            )
            for row in current["score_ablation_rows"]:
                rule = str(row["score_rule"])
                ablations.append(
                    {
                        "pair_id": pair_id,
                        "method": current["method"],
                        "score_variant": {
                            "s1": "S1",
                            "s2": "S2",
                            "s3": "S3",
                            "s1_s2": "S1+S2",
                            "s1_s3": "S1+S3",
                            "s2_s3": "S2+S3",
                            "full": "full",
                            "max_pair_probability": "pCSSR max pair probability",
                        }[rule],
                        **{
                            key: float(row[key])
                            for key in REPORT_METRIC_KEYS
                        },
                        "threshold": float(row["threshold"]),
                    }
                )
        integrity.append(
            {
                "pair_id": pair_id,
                "pair_manifest_sha256": first["pair_manifest_sha256"],
                "unique_base_manifest_sha256": first["unique_base_manifest_sha256"],
                "o0_reused_without_training": True,
                "o0_identical_across_o1_o4": True,
                "all_checkpoints_replayed": True,
            }
        )
    return {
        "metrics": metrics,
        "identities": identities,
        "absorption": absorption,
        "ablations": ablations,
        "tasks": tasks,
        "integrity": integrity,
    }


def _phase_audits(
    root: Path,
    *,
    config: Mapping[str, Any],
    bundle_root: str | Path,
    r2_results_root: str | Path,
    oracle_audit_path: str | Path,
    phase: str,
    device_request: str,
    smoke_root: str | Path | None,
) -> list[dict[str, Any]]:
    return [
        audit_unit_result(
            _unit_destination(root, str(unit["pair_id"]), str(unit["method"])),
            config=config,
            bundle_root=bundle_root,
            r2_results_root=r2_results_root,
            oracle_audit_path=oracle_audit_path,
            phase=phase,
            pair_id=str(unit["pair_id"]),
            method=str(unit["method"]),
            device_request=device_request,
            smoke_root=smoke_root,
        )
        for unit in build_phase_plan(config, phase)
    ]


def _task_audit_row(
    unit: Mapping[str, Any],
    *,
    audit: Mapping[str, Any] | None = None,
    status: str | None = None,
    failure_type: str = "",
    failure_message: str = "",
) -> dict[str, Any]:
    if audit is not None:
        return {
            "pair_id": str(unit["pair_id"]),
            "method": str(unit["method"]),
            "status": str(audit["status"]),
            "audit_passed": bool(audit["audit_passed"]),
            "checkpoint_replay": str(audit["checkpoint_replay"]),
            "artifact_count": int(audit["artifact_count"]),
            "checkpoint_sha256": str(audit["checkpoint_sha256"]),
            "failure_type": "",
            "failure_message": "",
        }
    return {
        "pair_id": str(unit["pair_id"]),
        "method": str(unit["method"]),
        "status": str(status or "failed"),
        "audit_passed": False,
        "checkpoint_replay": "not_available",
        "artifact_count": 0,
        "checkpoint_sha256": "",
        "failure_type": str(failure_type),
        "failure_message": str(failure_message),
    }


def _collect_phase_audits(
    root: Path,
    *,
    config: Mapping[str, Any],
    bundle_root: str | Path,
    r2_results_root: str | Path,
    oracle_audit_path: str | Path,
    phase: str,
    device_request: str,
    smoke_root: str | Path | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Audit every planned unit while preserving a complete failure ledger."""

    audits: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    for unit in build_phase_plan(config, phase):
        destination = _unit_destination(
            root, str(unit["pair_id"]), str(unit["method"])
        )
        if not destination.is_dir():
            task_rows.append(
                _task_audit_row(
                    unit,
                    status="missing",
                    failure_type="missing_unit",
                    failure_message="planned unit directory is absent",
                )
            )
            continue
        try:
            audit = audit_unit_result(
                destination,
                config=config,
                bundle_root=bundle_root,
                r2_results_root=r2_results_root,
                oracle_audit_path=oracle_audit_path,
                phase=phase,
                pair_id=str(unit["pair_id"]),
                method=str(unit["method"]),
                device_request=device_request,
                smoke_root=smoke_root,
            )
        except Exception as error:
            task_rows.append(
                _task_audit_row(
                    unit,
                    status="failed",
                    failure_type=type(error).__name__,
                    failure_message=str(error),
                )
            )
        else:
            audits.append(audit)
            task_rows.append(_task_audit_row(unit, audit=audit))
    return audits, task_rows


def _assert_common_unit_contract(audits: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not audits:
        raise DataValidationError("phase contains no successful unit audits")
    code_commits = {str(row["code_commit"]) for row in audits}
    source_hashes = {_json_sha256(row["source_hashes"]) for row in audits}
    smoke_authorizations = {
        _json_sha256(row.get("smoke_authorization")) for row in audits
    }
    if len(code_commits) != 1 or len(source_hashes) != 1 or len(smoke_authorizations) != 1:
        raise DataValidationError("phase units do not share one code/source/smoke contract")
    first = audits[0]
    return {
        "code_commit": next(iter(code_commits)),
        "source_hashes": dict(first["source_hashes"]),
        "smoke_authorization": first.get("smoke_authorization"),
    }


def _phase_oracle_identity_sha256(
    oracle_audit_path: str | Path,
    config: Mapping[str, Any],
) -> str:
    return _json_sha256(
        _official_audit_identity(_official_audit_record(oracle_audit_path, config))
    )


def _ensure_unaggregated_phase_root(root: Path) -> None:
    existing = [name for name in PHASE_AGGREGATE_FILES if (root / name).exists()]
    if existing:
        raise DataValidationError(
            "official CSSR phase already has aggregate output; use a fresh root: "
            + ", ".join(existing)
        )


def aggregate_phase_root(
    phase_root: str | Path,
    *,
    config: Mapping[str, Any],
    bundle_root: str | Path,
    r2_results_root: str | Path,
    oracle_audit_path: str | Path,
    phase: str,
    device_request: str = "auto",
    smoke_root: str | Path | None = None,
) -> dict[str, Any]:
    _bound_project_root(config)
    root = Path(phase_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _ensure_unaggregated_phase_root(root)
    plan_payload = {"phase": phase, "units": build_phase_plan(config, phase)}
    audits, task_rows = _collect_phase_audits(
        root,
        config=config,
        bundle_root=bundle_root,
        r2_results_root=r2_results_root,
        oracle_audit_path=oracle_audit_path,
        phase=phase,
        device_request=device_request,
        smoke_root=smoke_root,
    )
    complete_population = len(audits) == len(plan_payload["units"]) and all(
        row["audit_passed"] is True and row["status"] == "success"
        for row in task_rows
    )
    aggregate_failure: dict[str, str] | None = None
    rows: dict[str, list[dict[str, Any]]] | None = None
    contract: dict[str, Any] | None = None
    oracle_identity_sha256: str | None = None
    gate_input: dict[str, Any] | None = None
    gate: dict[str, Any] | None = None
    decision: str | None = None
    if complete_population:
        try:
            contract = _assert_common_unit_contract(audits)
            oracle_identity_sha256 = _phase_oracle_identity_sha256(
                oracle_audit_path, config
            )
            if phase == "smoke":
                rows = {
                    "metrics": [],
                    "identities": [],
                    "absorption": [],
                    "ablations": [],
                    "tasks": task_rows,
                    "integrity": [],
                }
                decision = "diagnostic_smoke_only"
            else:
                rows = _aggregate_rows(audits)
                if rows["tasks"] != task_rows:
                    raise DataValidationError("pilot task audit ledger changed")
                gate_ablation_rows = [
                    row
                    for row in rows["ablations"]
                    if row["score_variant"] in {"S1", "full"}
                ]
                gate_input = {
                    "metric_rows": rows["metrics"],
                    "identity_rows": rows["identities"],
                    "score_ablation_rows": gate_ablation_rows,
                    "task_rows": rows["tasks"],
                    "audit_passed": True,
                    "aggregate_failure": None,
                }
                gate = evaluate_pilot_gate(
                    rows["metrics"],
                    rows["identities"],
                    gate_ablation_rows,
                    task_rows=rows["tasks"],
                    audit_passed=True,
                )
                if gate.get("pilot_status") != "completed":
                    raise DataValidationError(
                        "complete pilot population did not produce a gate"
                    )
                decision = str(gate["result_label"])
                if not isinstance(gate.get("comparisons"), Mapping):
                    raise DataValidationError(
                        "complete pilot gate lacks preregistered comparisons"
                    )
        except Exception as error:
            aggregate_failure = {
                "failure_type": type(error).__name__,
                "failure_message": str(error),
            }
            complete_population = False

    if not complete_population:
        gate_input = None
        gate = None
        if phase == "pilot":
            gate_input = {
                "metric_rows": [],
                "identity_rows": [],
                "score_ablation_rows": [],
                "task_rows": task_rows,
                "audit_passed": False,
                "aggregate_failure": aggregate_failure,
            }
            gate = evaluate_pilot_gate(
                [],
                [],
                [],
                task_rows=task_rows,
                audit_passed=False,
            )
        summary = {
            "status": "hard_failed_incomplete",
            "experiment_id": EXPERIMENT_ID,
            "phase": phase,
            "pair_ids": list(PILOT_PAIRS) if phase == "pilot" else ["N1"],
            "planned_unit_count": len(plan_payload["units"]),
            "successful_unit_count": len(audits),
            "decision": None if phase == "pilot" else "diagnostic_smoke_failed",
            "gate": gate,
            "aggregate_failure": aggregate_failure,
            "config_sha256": config["_config_sha256"],
            "diagnostic_only": phase == "smoke",
            "confirmation_allowed": False,
            "automatic_followon_authorized": False,
            "final_unknown_used": False,
            "even_angle_test_used": False,
            "final_unknown_test_authorized": False,
        }
        _assert_finite_tree(summary, context="incomplete phase summary")
        _write_json(root / "task_plan.json", plan_payload)
        _write_csv(root / "task_audit.csv", task_rows)
        if phase == "pilot":
            _write_json(root / "pilot_gate_input.json", gate_input)
            _write_json(root / "pilot_gate.json", gate)
        _write_json(root / "phase_summary.json", summary)
        _write_json(root / "artifact_hashes.json", _phase_artifact_hashes(root))
        _write_json(
            root / "_PHASE_INCOMPLETE.json",
            {
                "status": "hard_failed_incomplete",
                "phase_summary_sha256": file_sha256(root / "phase_summary.json"),
                "artifact_hashes_sha256": file_sha256(root / "artifact_hashes.json"),
                "pilot_gate": "not_evaluated" if phase == "pilot" else None,
                "selected_method": None,
                "final_unknown_used": False,
                "even_angle_test_used": False,
            },
        )
        return summary

    assert (
        rows is not None
        and contract is not None
        and oracle_identity_sha256 is not None
        and decision is not None
    )

    _write_json(root / "task_plan.json", plan_payload)
    _write_csv(root / "task_audit.csv", rows["tasks"])
    if phase == "pilot":
        _write_csv(root / "metrics_by_pair.csv", rows["metrics"])
        _write_csv(root / "identity_metrics.csv", rows["identities"])
        _write_csv(root / "absorption_by_known_class.csv", rows["absorption"])
        _write_csv(root / "score_ablation_metrics.csv", rows["ablations"])
        _write_json(root / "phase_integrity_audit.json", {"status": "passed", "pairs": rows["integrity"]})
    else:
        _write_json(root / "identity_metrics.json", {"status": "diagnostic_smoke_not_aggregated"})
        _write_json(root / "absorption_by_known_class.json", {"status": "diagnostic_smoke_not_aggregated"})
        _write_json(root / "score_ablation_metrics.json", {"status": "diagnostic_smoke_not_aggregated"})
        _write_json(root / "phase_integrity_audit.json", {"status": "passed", "diagnostic_only": True})
    if gate is not None:
        _write_json(root / "pilot_gate_input.json", gate_input)
        _write_json(root / "pilot_gate.json", gate)
        _write_json(root / "pre_registered_comparisons.json", gate["comparisons"])
    summary = {
        "status": "complete",
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "pair_ids": list(PILOT_PAIRS) if phase == "pilot" else ["N1"],
        "unit_count": len(audits),
        "decision": decision,
        "gate": gate,
        "config_sha256": config["_config_sha256"],
        "official_oracle_identity_sha256": oracle_identity_sha256,
        "code_commit": contract["code_commit"],
        "source_hashes_sha256": _json_sha256(contract["source_hashes"]),
        "smoke_authorization": contract["smoke_authorization"],
        "diagnostic_only": phase == "smoke",
        "confirmation_allowed": False,
        "automatic_followon_authorized": False,
        "final_unknown_used": False,
        "even_angle_test_used": False,
        "final_unknown_test_authorized": False,
    }
    _assert_finite_tree(summary, context="complete phase summary")
    _write_json(root / "phase_summary.json", summary)
    _write_json(root / "artifact_hashes.json", _phase_artifact_hashes(root))
    _write_json(
        root / "_PHASE_SUCCESS.json",
        {
            "status": "complete",
            "phase_summary_sha256": file_sha256(root / "phase_summary.json"),
            "artifact_hashes_sha256": file_sha256(root / "artifact_hashes.json"),
            "final_unknown_used": False,
            "even_angle_test_used": False,
        },
    )
    return summary


def audit_phase_root(
    phase_root: str | Path,
    *,
    config: Mapping[str, Any],
    bundle_root: str | Path,
    r2_results_root: str | Path,
    oracle_audit_path: str | Path,
    phase: str,
    device_request: str = "auto",
    smoke_root: str | Path | None = None,
) -> dict[str, Any]:
    _bound_project_root(config)
    root = Path(phase_root).resolve()
    success_path = root / "_PHASE_SUCCESS.json"
    incomplete_path = root / "_PHASE_INCOMPLETE.json"
    if success_path.exists() == incomplete_path.exists():
        raise DataValidationError(
            "phase must contain exactly one success or incomplete marker"
        )
    marker_path = success_path if success_path.exists() else incomplete_path
    marker = _read_json(marker_path)
    if (
        marker.get("phase_summary_sha256") != file_sha256(root / "phase_summary.json")
        or marker.get("artifact_hashes_sha256") != file_sha256(root / "artifact_hashes.json")
        or marker.get("final_unknown_used") is not False
        or marker.get("even_angle_test_used") is not False
        or _read_json(root / "artifact_hashes.json") != _phase_artifact_hashes(root)
    ):
        raise DataValidationError("official CSSR phase artifact hash audit failed")
    expected_plan = {"phase": phase, "units": build_phase_plan(config, phase)}
    if _read_json(root / "task_plan.json") != expected_plan:
        raise DataValidationError("official CSSR phase plan changed")

    summary = _read_json(root / "phase_summary.json")
    if incomplete_path.exists():
        if (
            marker.get("status") != "hard_failed_incomplete"
            or summary.get("status") != "hard_failed_incomplete"
            or summary.get("experiment_id") != EXPERIMENT_ID
            or summary.get("phase") != phase
            or summary.get("pair_ids")
            != (list(PILOT_PAIRS) if phase == "pilot" else ["N1"])
            or summary.get("config_sha256") != config["_config_sha256"]
            or summary.get("diagnostic_only") is not (phase == "smoke")
            or summary.get("confirmation_allowed") is not False
            or summary.get("automatic_followon_authorized") is not False
            or summary.get("final_unknown_used") is not False
            or summary.get("even_angle_test_used") is not False
            or summary.get("final_unknown_test_authorized") is not False
            or (root / "metrics_by_pair.csv").exists()
            or (root / "pre_registered_comparisons.json").exists()
        ):
            raise DataValidationError("incomplete phase semantics changed")
        task_rows_csv = _read_csv(root / "task_audit.csv")
        if len(task_rows_csv) != len(expected_plan["units"]):
            raise DataValidationError("incomplete phase task ledger changed")
        expected_keys = {
            (str(unit["pair_id"]), str(unit["method"]))
            for unit in expected_plan["units"]
        }
        observed_keys = {
            (str(row["pair_id"]), str(row["method"])) for row in task_rows_csv
        }
        if observed_keys != expected_keys:
            raise DataValidationError("incomplete phase task identities changed")
        successful_rows = sum(
            row.get("status") == "success" and row.get("audit_passed") == "True"
            for row in task_rows_csv
        )
        if (
            int(summary.get("planned_unit_count", -1)) != len(expected_plan["units"])
            or int(summary.get("successful_unit_count", -1)) != successful_rows
            or (
                successful_rows == len(expected_plan["units"])
                and summary.get("aggregate_failure") is None
            )
        ):
            raise DataValidationError("incomplete phase counts or failure evidence changed")
        if phase == "pilot":
            gate_input = _read_json(root / "pilot_gate_input.json")
            if (
                gate_input.get("audit_passed") is not False
                or gate_input.get("metric_rows") != []
                or gate_input.get("identity_rows") != []
                or gate_input.get("score_ablation_rows") != []
            ):
                raise DataValidationError("incomplete pilot gate input changed")
            task_rows = gate_input.get("task_rows")
            if not isinstance(task_rows, list):
                raise DataValidationError("incomplete pilot task rows are absent")
            canonical_task_rows = [
                {key: row[key] for key in TASK_AUDIT_FIELDS} for row in task_rows
            ]
            _csv_rows_exact(
                task_rows_csv,
                canonical_task_rows,
                context="incomplete task audit",
            )
            gate = evaluate_pilot_gate(
                [], [], [], task_rows=canonical_task_rows, audit_passed=False
            )
            if (
                _read_json(root / "pilot_gate.json") != gate
                or summary.get("gate") != gate
                or gate.get("pilot_status") != "hard_failed_incomplete"
                or gate.get("pilot_gate") != "not_evaluated"
                or gate.get("selected_method") is not None
                or marker.get("pilot_gate") != "not_evaluated"
                or marker.get("selected_method") is not None
            ):
                raise DataValidationError("incomplete pilot gate changed")
        else:
            gate = None
            if (
                (root / "pilot_gate_input.json").exists()
                or (root / "pilot_gate.json").exists()
                or summary.get("gate") is not None
                or summary.get("decision") != "diagnostic_smoke_failed"
            ):
                raise DataValidationError("failed smoke incorrectly contains a gate")
        return {
            "status": "passed",
            "phase_status": "hard_failed_incomplete",
            "experiment_id": EXPERIMENT_ID,
            "phase": phase,
            "planned_unit_count": len(expected_plan["units"]),
            "successful_unit_count": int(summary.get("successful_unit_count", -1)),
            "decision": summary.get("decision"),
            "gate": gate,
            "artifact_count": len(_read_json(root / "artifact_hashes.json")),
            "all_checkpoints_replayed": False,
            "all_metrics_recomputed_from_predictions": False,
            "final_unknown_used": False,
            "even_angle_test_used": False,
            "confirmation_allowed": False,
            "final_unknown_test_authorized": False,
        }

    if marker.get("status") != "complete":
        raise DataValidationError("successful phase marker changed")
    audits = _phase_audits(
        root,
        config=config,
        bundle_root=bundle_root,
        r2_results_root=r2_results_root,
        oracle_audit_path=oracle_audit_path,
        phase=phase,
        device_request=device_request,
        smoke_root=smoke_root,
    )
    contract = _assert_common_unit_contract(audits)
    oracle_identity_sha256 = _phase_oracle_identity_sha256(
        oracle_audit_path, config
    )
    if phase == "pilot":
        rows = _aggregate_rows(audits)
        for filename, expected in (
            ("task_audit.csv", rows["tasks"]),
            ("metrics_by_pair.csv", rows["metrics"]),
            ("identity_metrics.csv", rows["identities"]),
            ("absorption_by_known_class.csv", rows["absorption"]),
            ("score_ablation_metrics.csv", rows["ablations"]),
        ):
            _csv_rows_exact(_read_csv(root / filename), expected, context=filename)
        gate_input = {
            "metric_rows": rows["metrics"],
            "identity_rows": rows["identities"],
            "score_ablation_rows": [
                row for row in rows["ablations"]
                if row["score_variant"] in {"S1", "full"}
            ],
            "task_rows": rows["tasks"],
            "audit_passed": True,
            "aggregate_failure": None,
        }
        if _read_json(root / "pilot_gate_input.json") != gate_input:
            raise DataValidationError("official CSSR pilot gate input changed")
        gate = evaluate_pilot_gate(
            gate_input["metric_rows"],
            gate_input["identity_rows"],
            gate_input["score_ablation_rows"],
            task_rows=gate_input["task_rows"],
            audit_passed=bool(gate_input["audit_passed"]),
        )
        if _read_json(root / "pilot_gate.json") != gate:
            raise DataValidationError("official CSSR pilot gate does not reproduce")
        if _read_json(root / "pre_registered_comparisons.json") != gate["comparisons"]:
            raise DataValidationError("official CSSR comparisons do not reproduce")
        if _read_json(root / "phase_integrity_audit.json") != {
            "status": "passed",
            "pairs": rows["integrity"],
        }:
            raise DataValidationError("official CSSR phase integrity changed")
        decision = str(gate["result_label"])
    else:
        expected_tasks = [
            _task_audit_row(
                {"pair_id": row["pair_id"], "method": row["method"]},
                audit=row,
            ) for row in audits
        ]
        _csv_rows_exact(_read_csv(root / "task_audit.csv"), expected_tasks, context="smoke task audit")
        if (root / "metrics_by_pair.csv").exists():
            raise DataValidationError("diagnostic smoke exposed an aggregate performance table")
        for filename, expected in (
            ("identity_metrics.json", {"status": "diagnostic_smoke_not_aggregated"}),
            ("absorption_by_known_class.json", {"status": "diagnostic_smoke_not_aggregated"}),
            ("score_ablation_metrics.json", {"status": "diagnostic_smoke_not_aggregated"}),
            ("phase_integrity_audit.json", {"status": "passed", "diagnostic_only": True}),
        ):
            if _read_json(root / filename) != expected:
                raise DataValidationError(f"smoke aggregate changed at {filename}")
        gate = None
        decision = "diagnostic_smoke_only"
    if (
        summary.get("status") != "complete"
        or summary.get("experiment_id") != EXPERIMENT_ID
        or summary.get("phase") != phase
        or summary.get("pair_ids")
        != (list(PILOT_PAIRS) if phase == "pilot" else ["N1"])
        or int(summary.get("unit_count", -1)) != len(audits)
        or summary.get("decision") != decision
        or summary.get("gate") != gate
        or summary.get("config_sha256") != config["_config_sha256"]
        or summary.get("official_oracle_identity_sha256")
        != oracle_identity_sha256
        or summary.get("code_commit") != contract["code_commit"]
        or summary.get("source_hashes_sha256")
        != _json_sha256(contract["source_hashes"])
        or summary.get("smoke_authorization") != contract["smoke_authorization"]
        or summary.get("diagnostic_only") is not (phase == "smoke")
        or summary.get("confirmation_allowed") is not False
        or summary.get("automatic_followon_authorized") is not False
        or summary.get("final_unknown_used") is not False
        or summary.get("even_angle_test_used") is not False
        or summary.get("final_unknown_test_authorized") is not False
    ):
        raise DataValidationError("official CSSR phase summary changed")
    return {
        "status": "passed",
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "unit_count": len(audits),
        "decision": decision,
        "gate": gate,
        "artifact_count": len(_read_json(root / "artifact_hashes.json")),
        "all_checkpoints_replayed": True,
        "all_metrics_recomputed_from_predictions": True,
        "final_unknown_used": False,
        "even_angle_test_used": False,
        "confirmation_allowed": False,
        "final_unknown_test_authorized": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Official-semantics CSSR HRRP pilot")
    commands = parser.add_subparsers(dest="command", required=True)

    load = commands.add_parser("load-config")
    load.add_argument("--config", default=CONFIG_RELATIVE_PATH)

    oracle = commands.add_parser("oracle-audit")
    oracle.add_argument("--config", default=CONFIG_RELATIVE_PATH)
    oracle.add_argument("--official-root", required=True)
    oracle.add_argument("--output", required=True)
    oracle.add_argument("--device", default="cuda")

    plan = commands.add_parser("plan")
    plan.add_argument("--config", default=CONFIG_RELATIVE_PATH)
    plan.add_argument("--phase", choices=("smoke", "pilot"), required=True)

    run = commands.add_parser("run-unit")
    run.add_argument("--config", default=CONFIG_RELATIVE_PATH)
    run.add_argument("--bundle-root", required=True)
    run.add_argument("--r2-results-root", required=True)
    run.add_argument("--phase-root", required=True)
    run.add_argument("--oracle-audit", required=True)
    run.add_argument("--phase", choices=("smoke", "pilot"), required=True)
    run.add_argument("--pair-id", required=True)
    run.add_argument("--method", choices=TRAINABLE_METHODS, required=True)
    run.add_argument("--device", default="auto")
    run.add_argument("--smoke-root")

    audit_unit_parser = commands.add_parser("audit-unit")
    audit_unit_parser.add_argument("--config", default=CONFIG_RELATIVE_PATH)
    audit_unit_parser.add_argument("--bundle-root", required=True)
    audit_unit_parser.add_argument("--r2-results-root", required=True)
    audit_unit_parser.add_argument("--unit-root", required=True)
    audit_unit_parser.add_argument("--oracle-audit", required=True)
    audit_unit_parser.add_argument("--phase", choices=("smoke", "pilot"), required=True)
    audit_unit_parser.add_argument("--pair-id", required=True)
    audit_unit_parser.add_argument("--method", choices=TRAINABLE_METHODS, required=True)
    audit_unit_parser.add_argument("--device", default="auto")
    audit_unit_parser.add_argument("--smoke-root")

    for name in ("aggregate", "audit-phase"):
        command = commands.add_parser(name)
        command.add_argument("--config", default=CONFIG_RELATIVE_PATH)
        command.add_argument("--bundle-root", required=True)
        command.add_argument("--r2-results-root", required=True)
        command.add_argument("--phase-root", required=True)
        command.add_argument("--oracle-audit", required=True)
        command.add_argument("--phase", choices=("smoke", "pilot"), required=True)
        command.add_argument("--device", default="auto")
        command.add_argument("--smoke-root")
    return parser


def _cli_safe_result(
    result: Mapping[str, Any],
    *,
    command: str,
    phase: str | None,
) -> dict[str, Any]:
    """Prevent diagnostic smoke commands from displaying performance results."""

    if phase != "smoke" or command not in {"run-unit", "audit-unit"}:
        return dict(result)
    allowed = (
        "status",
        "phase",
        "pair_id",
        "method",
        "destination",
        "audit_passed",
        "artifact_count",
        "checkpoint_replay",
        "r2_unchanged",
        "diagnostic_only",
        "final_unknown_used",
        "even_angle_test_used",
        "confirmation_allowed",
        "final_unknown_test_authorized",
    )
    return {
        **{key: result[key] for key in allowed if key in result},
        "performance_metrics_displayed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = load_official_cssr_config(arguments.config)
    _bound_project_root(config)
    if arguments.command == "load-config":
        result = {
            "status": "passed",
            "experiment_id": config["experiment_id"],
            "config_path": config["_config_path"],
            "config_sha256": config["_config_sha256"],
            "final_unknown_test_authorized": False,
        }
    elif arguments.command == "oracle-audit":
        output = Path(arguments.output).resolve()
        if output.exists():
            raise DataValidationError(f"oracle audit output already exists: {output}")
        result = audit_official_cssr_oracle(
            arguments.official_root,
            device=arguments.device,
        )
        _write_json(output, result)
        result = {**result, "output": str(output), "output_sha256": file_sha256(output)}
    elif arguments.command == "plan":
        result = {
            "status": "planned",
            "phase": arguments.phase,
            "units": build_phase_plan(config, arguments.phase),
            "confirmation_allowed": False,
            "final_unknown_test_authorized": False,
        }
    elif arguments.command == "run-unit":
        result = run_unit(
            arguments.config,
            arguments.bundle_root,
            arguments.r2_results_root,
            arguments.phase_root,
            arguments.oracle_audit,
            phase=arguments.phase,
            pair_id=arguments.pair_id,
            method=arguments.method,
            device_request=arguments.device,
            smoke_root=arguments.smoke_root,
        )
    elif arguments.command == "audit-unit":
        result = audit_unit_result(
            arguments.unit_root,
            config=config,
            bundle_root=arguments.bundle_root,
            r2_results_root=arguments.r2_results_root,
            oracle_audit_path=arguments.oracle_audit,
            phase=arguments.phase,
            pair_id=arguments.pair_id,
            method=arguments.method,
            device_request=arguments.device,
            smoke_root=arguments.smoke_root,
        )
    elif arguments.command == "aggregate":
        result = aggregate_phase_root(
            arguments.phase_root,
            config=config,
            bundle_root=arguments.bundle_root,
            r2_results_root=arguments.r2_results_root,
            oracle_audit_path=arguments.oracle_audit,
            phase=arguments.phase,
            device_request=arguments.device,
            smoke_root=arguments.smoke_root,
        )
    elif arguments.command == "audit-phase":
        result = audit_phase_root(
            arguments.phase_root,
            config=config,
            bundle_root=arguments.bundle_root,
            r2_results_root=arguments.r2_results_root,
            oracle_audit_path=arguments.oracle_audit,
            phase=arguments.phase,
            device_request=arguments.device,
            smoke_root=arguments.smoke_root,
        )
    else:
        raise AssertionError("unreachable command")
    printable = _cli_safe_result(
        result,
        command=str(arguments.command),
        phase=getattr(arguments, "phase", None),
    )
    print(json.dumps(_mapping_to_json(printable), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
