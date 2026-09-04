from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from hrrp_osr.evaluation.official_cssr_oracle import (  # noqa: E402
    EXPECTED_FORMAL_CUDA_DEVICE_NAME,
    FORMAL_CUBLAS_WORKSPACE_CONFIG,
    OFFICIAL_CSSR_REFERENCE_HASHES,
    ORACLE_DETERMINISTIC_REPEATS,
    OfficialCSSROracleError,
    _validate_oracle_device,
    audit_official_cssr_oracle,
)
from hrrp_osr.evaluation.official_cssr_scores import (  # noqa: E402
    OFFICIAL_SCORE_RULES,
)
from hrrp_osr.models.official_cssr_1d import (  # noqa: E402
    MATCHED_LINEAR_CONTROL_1D,
    OFFICIAL_CSSR_REFERENCE_COMMIT,
    OFFICIAL_SEMANTICS_PCSSR_1D,
)


def _official_root() -> Path:
    configured = os.environ.get("OFFICIAL_CSSR_ROOT")
    if configured:
        return Path(configured)
    return Path("/private/tmp/cssr-official-d5a99e91")


def test_oracle_rejects_missing_official_checkout(tmp_path: Path) -> None:
    with pytest.raises(OfficialCSSROracleError, match="commit"):
        audit_official_cssr_oracle(tmp_path)


def test_hash_verified_official_oracle_passes_float32_and_float64() -> None:
    root = _official_root()
    if not root.exists():
        pytest.skip("fixed official CSSR checkout is not available")
    result = audit_official_cssr_oracle(root, device="cpu")
    assert result["passed"] is True
    assert result["status"] == "passed"
    assert result["official_commit"] == OFFICIAL_CSSR_REFERENCE_COMMIT
    assert result["file_sha256"] == dict(OFFICIAL_CSSR_REFERENCE_HASHES)
    assert result["verified_file_sha256"] == dict(OFFICIAL_CSSR_REFERENCE_HASHES)
    assert result["float32"] == "passed"
    assert result["float64"] == "passed"
    assert set(result["dtype_checks"]) == {"float32", "float64"}
    assert all(record["passed"] for record in result["dtype_checks"].values())
    assert result["dtype_checks"]["float32"]["rtol"] == 1.0e-5
    assert result["dtype_checks"]["float32"]["atol"] == 1.0e-6
    assert result["dtype_checks"]["float64"]["rtol"] == 1.0e-9
    assert result["dtype_checks"]["float64"]["atol"] == 1.0e-11
    assert result["runtime_contract"] == {
        "device": "cpu",
        "device_type": "cpu",
        "cuda_device_name": None,
        "expected_cuda_device_name": None,
        "formal_cuda_device_match": None,
        "cublas_workspace_config": None,
        "deterministic_algorithms": True,
        "deterministic_algorithms_warn_only": False,
        "cudnn_benchmark": False,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
    }
    assert result["oracle_contract"] == {
        "seed": 20260904,
        "required_dtypes": ["float32", "float64"],
        "tolerances": {
            "float32": {"rtol": 1.0e-5, "atol": 1.0e-6},
            "float64": {"rtol": 1.0e-9, "atol": 1.0e-11},
        },
        "clip_bounds": [-100.0, 100.0],
        "pair_score_rules": list(OFFICIAL_SCORE_RULES),
        "pair_probability_rule": "arithmetic_mean_of_two_view_probabilities",
        "pair_prediction_rule": "argmax_pair_probability",
        "unknown_score_direction": "negative_knownness",
        "deterministic_repeats": ORACLE_DETERMINISTIC_REPEATS,
    }
    for dtype_name in ("float32", "float64"):
        dtype_record = result["dtype_checks"][dtype_name]
        assert dtype_record["deterministic_repeat"] == {
            "passed": True,
            "repeats": ORACLE_DETERMINISTIC_REPEATS,
            "record_equality": "exact",
        }
        assert dtype_record["clip_boundary_checks"] == {
            "passed": True,
            "bounds": [-100.0, 100.0],
            "requested_preclip_magnitudes": [0.0, 99.0, 100.0, 101.0, 137.5],
            "lower_interior_exercised": True,
            "lower_exact_boundary_exercised": True,
            "lower_saturation_exercised": True,
            "upper_interior_exercised": True,
            "upper_exact_boundary_exercised": True,
            "upper_saturation_exercised": True,
            "upper_boundary_reference_only": (
                "official RCSSR positive-sign path; the candidate implements pCSSR only"
            ),
        }
        assert dtype_record["pair_checks"]["probability"] == "passed"
        assert dtype_record["pair_checks"]["argmax"] == "passed"
        assert dtype_record["pair_checks"]["knownness_rules"] == list(
            OFFICIAL_SCORE_RULES
        )
        differences = dtype_record["max_absolute_differences"]
        assert {
            "clip_lower_candidate_vs_official",
            "clip_lower_official_vs_literal",
            "clip_upper_official_vs_literal",
            "raw_s1",
            "raw_s2",
            "raw_s3",
            "pair_view_logits",
            "pair_view_probabilities",
            "pair_probabilities",
        }.issubset(differences)
        for rule in OFFICIAL_SCORE_RULES:
            assert f"pair_knownness::{rule}" in differences
            assert f"pair_unknown_score::{rule}" in differences
    assert result["source_execution"].startswith("selected_ast_definitions")
    assert result["method_ids"] == {
        "official": OFFICIAL_SEMANTICS_PCSSR_1D,
        "matched_linear_control": MATCHED_LINEAR_CONTROL_1D,
    }
    json.dumps(result, sort_keys=True)


def test_cuda_oracle_rejects_non_frozen_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "NVIDIA A100")
    with pytest.raises(OfficialCSSROracleError, match="GPU mismatch"):
        _validate_oracle_device(torch.device("cuda"))


def test_cuda_oracle_requires_frozen_cublas_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_name",
        lambda _device: EXPECTED_FORMAL_CUDA_DEVICE_NAME,
    )
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(OfficialCSSROracleError, match="CUBLAS_WORKSPACE_CONFIG"):
        _validate_oracle_device(torch.device("cuda"))
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", FORMAL_CUBLAS_WORKSPACE_CONFIG)
    assert _validate_oracle_device(torch.device("cuda")) == {
        "device": "cuda",
        "device_type": "cuda",
        "cuda_device_name": EXPECTED_FORMAL_CUDA_DEVICE_NAME,
        "expected_cuda_device_name": EXPECTED_FORMAL_CUDA_DEVICE_NAME,
        "formal_cuda_device_match": True,
        "cublas_workspace_config": FORMAL_CUBLAS_WORKSPACE_CONFIG,
    }
