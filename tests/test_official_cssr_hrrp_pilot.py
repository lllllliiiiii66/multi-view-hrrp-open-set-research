from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402
import hrrp_osr.training.official_cssr_hrrp_pilot as official_runner  # noqa: E402

from hrrp_osr.data.errors import DataValidationError  # noqa: E402
from hrrp_osr.evaluation.official_cssr_scores import (  # noqa: E402
    OfficialScoreNormalization,
    OfficialScoreTemplates,
    matched_linear_pair_output,
    official_pcssr_pair_scores,
)
from hrrp_osr.evaluation.ms_mean_factorial import REPORT_METRIC_KEYS  # noqa: E402
from hrrp_osr.models.hrrp_ms_resnet import HRRPMultiScaleResNet1D  # noqa: E402
from hrrp_osr.models.official_cssr_1d import (  # noqa: E402
    MATCHED_LINEAR_CONTROL_1D,
    OFFICIAL_SEMANTICS_PCSSR_1D,
    MatchedLinearHeadOutput,
    OfficialCSSRHRRPModelOutput,
    OfficialPCSSRHeadOutput,
    official_softmax_average,
)
from hrrp_osr.training.official_cssr_hrrp_pilot import (  # noqa: E402
    TASK_SOURCE_FILES,
    _artifact_hashes,
    _aggregate_rows,
    _build_model,
    _evaluation_role_indices,
    _infer_pairs,
    _json_sha256,
    _materialize_pair_inputs,
    _official_audit_identity,
    _official_audit_record,
    _phase_artifact_hashes,
    _load_development_only_bundle,
    _profile_access_audit,
    _read_smoke_authorization,
    _resolved_config_bytes,
    _set_encoder_mode,
    _task_source_hashes,
    _write_json,
    aggregate_phase_root,
    audit_phase_root,
)
from hrrp_osr.data.manifest import file_sha256  # noqa: E402
from hrrp_osr.evaluation.official_cssr_oracle import (  # noqa: E402
    audit_official_cssr_oracle,
)
from hrrp_osr.training.official_cssr_protocol import (  # noqa: E402
    O0_R2_CC_MLS,
    O1_OFFICIAL_LINEAR_FT,
    O2_OFFICIAL_PCSSR_FT,
    O3_OFFICIAL_LINEAR_E2E,
    O4_OFFICIAL_PCSSR_E2E,
    PILOT_PAIRS,
    TRAINABLE_METHODS,
    build_phase_plan,
    load_official_cssr_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/experiments/cssr/official_cssr_hrrp_pilot_v1.yaml"
)


def _clone_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in module.state_dict().items()
    }


def _assert_state_equal(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> None:
    assert left.keys() == right.keys()
    for name in left:
        assert torch.equal(left[name], right[name]), name


def _assert_state_changed(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> None:
    assert left.keys() == right.keys()
    assert any(not torch.equal(left[name], right[name]) for name in left)


def test_ft_and_e2e_counterparts_use_exactly_the_same_head_initialization() -> None:
    config = load_official_cssr_config(CONFIG_PATH)
    r2 = SimpleNamespace(encoder=HRRPMultiScaleResNet1D(dropout=0.0))

    linear_ft, linear_ft_audit = _build_model(
        method=O1_OFFICIAL_LINEAR_FT,
        pair_id="N1",
        r2_model=r2,
        config=config,
        device=torch.device("cpu"),
    )
    linear_e2e, linear_e2e_audit = _build_model(
        method=O3_OFFICIAL_LINEAR_E2E,
        pair_id="N1",
        r2_model=r2,
        config=config,
        device=torch.device("cpu"),
    )
    pcssr_ft, pcssr_ft_audit = _build_model(
        method=O2_OFFICIAL_PCSSR_FT,
        pair_id="N1",
        r2_model=r2,
        config=config,
        device=torch.device("cpu"),
    )
    pcssr_e2e, pcssr_e2e_audit = _build_model(
        method=O4_OFFICIAL_PCSSR_E2E,
        pair_id="N1",
        r2_model=r2,
        config=config,
        device=torch.device("cpu"),
    )

    assert linear_ft_audit["seed"] == linear_e2e_audit["seed"]
    assert pcssr_ft_audit["seed"] == pcssr_e2e_audit["seed"]
    assert linear_ft_audit["head_initial_state_sha256"] == linear_e2e_audit[
        "head_initial_state_sha256"
    ]
    assert pcssr_ft_audit["head_initial_state_sha256"] == pcssr_e2e_audit[
        "head_initial_state_sha256"
    ]
    _assert_state_equal(_clone_state(linear_ft.head), _clone_state(linear_e2e.head))
    _assert_state_equal(_clone_state(pcssr_ft.head), _clone_state(pcssr_e2e.head))
    assert {
        audit["encoder_initial_state_sha256"]
        for audit in (
            linear_ft_audit,
            linear_e2e_audit,
            pcssr_ft_audit,
            pcssr_e2e_audit,
        )
    } == {linear_ft_audit["encoder_initial_state_sha256"]}
    for left, right in (
        (linear_ft_audit, linear_e2e_audit),
        (pcssr_ft_audit, pcssr_e2e_audit),
    ):
        assert left["rng_state_after_initialization"] == right[
            "rng_state_after_initialization"
        ]
        record = left["rng_state_after_initialization"]
        state = bytes.fromhex(record["torch_cpu_state_hex"])
        assert hashlib.sha256(state).hexdigest() == record[
            "torch_cpu_state_sha256"
        ]


class _TinyTrainableModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4, 4, bias=False),
            nn.BatchNorm1d(4),
            nn.Tanh(),
        )
        self.head = nn.Linear(4, 2, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(inputs))


@pytest.mark.parametrize(
    "method",
    [O1_OFFICIAL_LINEAR_FT, O2_OFFICIAL_PCSSR_FT],
)
def test_ft_mode_freezes_encoder_parameters_and_batchnorm_buffers(method: str) -> None:
    torch.manual_seed(8)
    model = _TinyTrainableModel()
    before = _clone_state(model.encoder)

    assert _set_encoder_mode(model, method=method, epoch=40) is False
    assert model.encoder.training is False
    assert all(module.training is False for module in model.encoder.modules())
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert model.head.training is True
    assert all(parameter.requires_grad for parameter in model.head.parameters())

    optimizer = torch.optim.SGD(model.head.parameters(), lr=0.1)
    inputs = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10.0 + 1.0
    loss = model(inputs).square().mean()
    loss.backward()
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    optimizer.step()
    _assert_state_equal(before, _clone_state(model.encoder))


@pytest.mark.parametrize(
    "method",
    [O3_OFFICIAL_LINEAR_E2E, O4_OFFICIAL_PCSSR_E2E],
)
def test_e2e_mode_is_frozen_through_epoch_five_and_unfreezes_at_six(
    method: str,
) -> None:
    torch.manual_seed(9)
    model = _TinyTrainableModel()
    before = _clone_state(model.encoder)

    assert _set_encoder_mode(model, method=method, epoch=5) is False
    assert model.encoder.training is False
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    _assert_state_equal(before, _clone_state(model.encoder))

    assert _set_encoder_mode(model, method=method, epoch=6) is True
    assert model.encoder.training is True
    assert all(parameter.requires_grad for parameter in model.encoder.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    inputs = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10.0 + 1.0
    optimizer.zero_grad(set_to_none=True)
    model(inputs).square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.encoder.parameters())
    optimizer.step()
    _assert_state_changed(before, _clone_state(model.encoder))


class _DeterministicPairModel(nn.Module):
    def __init__(self, head_kind: str) -> None:
        super().__init__()
        self.head_kind = head_kind

    def forward(self, inputs: torch.Tensor) -> OfficialCSSRHRRPModelOutput:
        if inputs.ndim == 3:
            inputs = inputs.squeeze(1)
        base = inputs[:, :76]
        channel_scale = torch.linspace(
            1.0,
            2.0,
            128,
            dtype=base.dtype,
            device=base.device,
        )[None, :, None]
        features = (base[:, None, :].abs() + 0.25) * channel_scale
        logits = torch.stack(
            (
                base,
                -base,
                0.5 * base,
                -0.5 * base,
                torch.zeros_like(base),
            ),
            dim=1,
        )
        probabilities = official_softmax_average(logits)
        if self.head_kind == MATCHED_LINEAR_CONTROL_1D:
            head_output = MatchedLinearHeadOutput(
                logits=logits,
                probabilities=probabilities,
            )
        else:
            errors = -logits / 0.1
            head_output = OfficialPCSSRHeadOutput(
                reconstructions=torch.zeros(
                    inputs.shape[0], 5, 128, 76, dtype=inputs.dtype
                ),
                latents=torch.zeros(inputs.shape[0], 5, 64, 76, dtype=inputs.dtype),
                reconstruction_errors=errors,
                logits=logits,
                probabilities=probabilities,
            )
        return OfficialCSSRHRRPModelOutput(
            feature_maps=features,
            head_output=head_output,
        )


def _pair_inputs_and_swapped() -> tuple[np.ndarray, np.ndarray]:
    profiles = np.stack(
        [
            np.linspace(-0.9, 1.1, 601, dtype=np.float64),
            np.linspace(0.2, 1.7, 601, dtype=np.float64),
            np.linspace(-1.5, -0.1, 601, dtype=np.float64),
            np.linspace(2.0, 0.4, 601, dtype=np.float64),
        ]
    )
    bundle = SimpleNamespace(profiles=profiles)
    rows = [
        {"view1_row_index": 0, "view2_row_index": 1},
        {"view1_row_index": 2, "view2_row_index": 3},
    ]
    swapped_rows = [
        {
            "view1_row_index": row["view2_row_index"],
            "view2_row_index": row["view1_row_index"],
        }
        for row in rows
    ]
    original = _materialize_pair_inputs(
        bundle=bundle,
        rows=rows,
        mean=0.3,
        std=1.7,
    )
    swapped = _materialize_pair_inputs(
        bundle=bundle,
        rows=swapped_rows,
        mean=0.3,
        std=1.7,
    )
    return original, swapped


def test_runner_pair_materialization_and_linear_score_are_view_swap_invariant() -> None:
    original_inputs, swapped_inputs = _pair_inputs_and_swapped()
    np.testing.assert_array_equal(swapped_inputs, original_inputs[:, ::-1])

    model = _DeterministicPairModel(MATCHED_LINEAR_CONTROL_1D)
    original = _infer_pairs(
        model,
        original_inputs,
        device=torch.device("cpu"),
        batch_size=3,
    )
    swapped = _infer_pairs(
        model,
        swapped_inputs,
        device=torch.device("cpu"),
        batch_size=3,
    )
    for name in ("features", "logits"):
        np.testing.assert_array_equal(swapped[name], original[name][:, ::-1])
    np.testing.assert_allclose(
        swapped["probabilities"],
        original["probabilities"][:, ::-1],
        rtol=2.0e-7,
        atol=2.0e-8,
    )

    original_score = matched_linear_pair_output(
        torch.from_numpy(original["logits"]),
        torch.from_numpy(original["probabilities"]),
    )
    swapped_score = matched_linear_pair_output(
        torch.from_numpy(swapped["logits"]),
        torch.from_numpy(swapped["probabilities"]),
    )
    torch.testing.assert_close(
        original_score.pair_probabilities, swapped_score.pair_probabilities
    )
    assert torch.equal(original_score.predicted_class, swapped_score.predicted_class)
    torch.testing.assert_close(original_score.unknown_score, swapped_score.unknown_score)


def test_runner_pair_inference_integrates_with_pcssr_swap_invariant_scores() -> None:
    original_inputs, swapped_inputs = _pair_inputs_and_swapped()
    model = _DeterministicPairModel(OFFICIAL_SEMANTICS_PCSSR_1D)
    original = _infer_pairs(
        model,
        original_inputs,
        device=torch.device("cpu"),
        batch_size=3,
    )
    swapped = _infer_pairs(
        model,
        swapped_inputs,
        device=torch.device("cpu"),
        batch_size=3,
    )
    templates = OfficialScoreTemplates(
        first_order=torch.full((5, 128), 0.2, dtype=torch.float32),
        gram=torch.eye(128, dtype=torch.float32).repeat(5, 1, 1),
        counts=torch.ones(5, dtype=torch.long),
        num_classes=5,
        power=8,
    )
    normalization = OfficialScoreNormalization(
        mean=torch.zeros(3, dtype=torch.float64),
        std=torch.ones(3, dtype=torch.float64),
        epsilon=1.0e-8,
        min_std=1.0e-12,
    )

    def score(arrays: dict[str, np.ndarray]):
        return official_pcssr_pair_scores(
            torch.from_numpy(arrays["features"]),
            torch.from_numpy(arrays["logits"]),
            torch.from_numpy(arrays["probabilities"]),
            templates,
            normalization,
        )

    original_score = score(original)
    swapped_score = score(swapped)
    torch.testing.assert_close(
        original_score.pair_probabilities, swapped_score.pair_probabilities
    )
    assert torch.equal(original_score.predicted_class, swapped_score.predicted_class)
    for rule in original_score.unknown_scores_by_rule:
        torch.testing.assert_close(
            original_score.unknown_scores_by_rule[rule],
            swapped_score.unknown_scores_by_rule[rule],
        )


def test_runner_scope_exposes_only_development_roles_and_rejects_final_phases() -> None:
    config = load_official_cssr_config(CONFIG_PATH)
    prepared = SimpleNamespace(
        labels={
            "known_calibration": np.arange(5),
            "surrogate_unknown": np.arange(2),
            "final_unknown": np.arange(11),
            "even_angle_test": np.arange(13),
        }
    )
    indices = _evaluation_role_indices(prepared, smoke=False, config=config)
    assert set(indices) == {"known_calibration", "surrogate_unknown"}
    assert config["data"]["final_test_pairs_generated"] is False
    assert config["evidence_scope"]["final_unknown_classes_used"] is False
    assert config["evidence_scope"]["even_angle_test_used"] is False
    for phase in ("confirmation", "final", "final_test", "even_angle_test"):
        with pytest.raises(DataValidationError, match="not authorized"):
            build_phase_plan(config, phase)


def _metric_fixture(offset: float = 0.0) -> dict[str, float]:
    return {
        key: 0.5 + offset + index / 1000.0
        for index, key in enumerate(REPORT_METRIC_KEYS)
    } | {"threshold": 0.25 + offset}


def _pilot_audit_fixture() -> list[dict[str, object]]:
    audits: list[dict[str, object]] = []
    for pair_index, pair_id in enumerate(PILOT_PAIRS):
        o0_metrics = _metric_fixture(pair_index / 100.0)
        o0_identity_rows = [
            {
                "pair_id": pair_id,
                "method": O0_R2_CC_MLS,
                "surrogate_identity": f"{pair_id}-identity",
                "auroc": 0.5,
            }
        ]
        o0_absorption_rows = [
            {
                "pair_id": pair_id,
                "method": O0_R2_CC_MLS,
                "known_class": "class-0",
                "false_accept_count": 1,
            }
        ]
        for method_index, method in enumerate(TRAINABLE_METHODS):
            ablations = []
            if method in (O2_OFFICIAL_PCSSR_FT, O4_OFFICIAL_PCSSR_E2E):
                for rule_index, rule in enumerate(
                    (
                        "s1",
                        "s2",
                        "s3",
                        "s1_s2",
                        "s1_s3",
                        "s2_s3",
                        "full",
                        "max_pair_probability",
                    )
                ):
                    ablations.append(
                        {
                            "score_rule": rule,
                            **_metric_fixture(rule_index / 100.0),
                        }
                    )
            audits.append(
                {
                    "status": "success",
                    "audit_passed": True,
                    "pair_id": pair_id,
                    "method": method,
                    "checkpoint_replay": "exact",
                    "artifact_count": 28,
                    "checkpoint_sha256": f"checkpoint-{pair_id}-{method}",
                    "pair_manifest_sha256": f"pairs-{pair_id}",
                    "unique_base_manifest_sha256": f"bases-{pair_id}",
                    "metrics": _metric_fixture(
                        pair_index / 100.0 + method_index / 1000.0
                    ),
                    "o0_metrics": o0_metrics,
                    "identity_rows": [
                        {
                            "pair_id": pair_id,
                            "method": method,
                            "surrogate_identity": f"{pair_id}-identity",
                            "auroc": 0.5,
                        }
                    ],
                    "o0_identity_rows": o0_identity_rows,
                    "absorption_rows": [
                        {
                            "pair_id": pair_id,
                            "method": method,
                            "known_class": "class-0",
                            "false_accept_count": 1,
                        }
                    ],
                    "o0_absorption_rows": o0_absorption_rows,
                    "score_ablation_rows": ablations,
                }
            )
    return audits


def test_pilot_aggregation_is_deterministic_and_deduplicates_shared_o0() -> None:
    audits = _pilot_audit_fixture()
    aggregated = _aggregate_rows(list(reversed(audits)))

    assert len(aggregated["metrics"]) == len(PILOT_PAIRS) * 5
    assert len(aggregated["tasks"]) == len(PILOT_PAIRS) * len(TRAINABLE_METHODS)
    assert len(aggregated["integrity"]) == len(PILOT_PAIRS)
    assert len(aggregated["ablations"]) == len(PILOT_PAIRS) * 2 * 8
    assert [
        (row["pair_id"], row["method"])
        for row in aggregated["metrics"]
    ] == [
        (pair_id, method)
        for pair_id in PILOT_PAIRS
        for method in (O0_R2_CC_MLS, *TRAINABLE_METHODS)
    ]
    assert {
        row["score_variant"] for row in aggregated["ablations"]
    } == {
        "S1",
        "S2",
        "S3",
        "S1+S2",
        "S1+S3",
        "S2+S3",
        "full",
        "pCSSR max pair probability",
    }
    assert all(row["o0_identical_across_o1_o4"] for row in aggregated["integrity"])


def test_pilot_aggregation_rejects_incomplete_duplicate_or_mismatched_shared_data() -> None:
    audits = _pilot_audit_fixture()
    with pytest.raises(DataValidationError, match="population changed"):
        _aggregate_rows(audits[:-1])

    with pytest.raises(DataValidationError, match="population changed|duplicate"):
        _aggregate_rows([*audits, dict(audits[0])])

    mismatched = _pilot_audit_fixture()
    mismatched[1] = {
        **mismatched[1],
        "pair_manifest_sha256": "different-pair-manifest",
    }
    with pytest.raises(DataValidationError, match="shared O0/data evidence differs"):
        _aggregate_rows(mismatched)


def test_resolved_config_and_oracle_identity_ignore_host_local_paths() -> None:
    config = load_official_cssr_config(CONFIG_PATH)
    moved = dict(config)
    moved["_config_path"] = "/another/host/checkout/config.yaml"
    assert _resolved_config_bytes(config) == _resolved_config_bytes(moved)

    dtype_record = {
        "passed": True,
        "dtype": "float32",
        "rtol": 1.0e-5,
        "atol": 1.0e-6,
        "clip_boundary_checks": {"passed": True},
        "pair_checks": {"passed": True},
        "deterministic_repeat": {"passed": True},
        "max_absolute_differences": {"diagnostic": 0.0},
    }
    oracle = {
        "passed": True,
        "status": "passed",
        "official_root": "/host/a/cssr",
        "official_commit": "commit",
        "file_sha256": {"file": "hash"},
        "verified_file_sha256": {"file": "hash"},
        "source_execution": "ast",
        "method_ids": {"official": "id"},
        "oracle_contract": {"contract": "fixed"},
        "float32": "passed",
        "float64": "passed",
        "dtype_checks": {
            "float32": dtype_record,
            "float64": {**dtype_record, "dtype": "float64"},
        },
        "runtime_contract": {"device": "cuda:0"},
        "torch_version": "host-a",
    }
    relocated = copy.deepcopy(oracle)
    relocated["official_root"] = "/host/b/official"
    relocated["runtime_contract"] = {"device": "cuda:3"}
    relocated["torch_version"] = "host-b"
    assert _official_audit_identity(oracle) == _official_audit_identity(relocated)


def test_formal_runner_rejects_config_from_another_checkout(tmp_path: Path) -> None:
    config = load_official_cssr_config(CONFIG_PATH)
    moved = dict(config)
    moved["_config_path"] = str(
        tmp_path / "configs/experiments/cssr/official_cssr_hrrp_pilot_v1.yaml"
    )
    with pytest.raises(DataValidationError, match="same checkout"):
        official_runner._bound_project_root(moved)


def test_runtime_rejects_wrong_cublas_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_official_cssr_config(CONFIG_PATH)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(DataValidationError, match="CUBLAS_WORKSPACE_CONFIG"):
        official_runner._configure_runtime(config, torch.device("cpu"))


def test_saved_runtime_contract_must_replay_exactly() -> None:
    current = {
        "device": "cuda",
        "device_name": "NVIDIA GeForce RTX 4090",
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "cublas_workspace_config": ":4096:8",
    }
    official_runner._assert_runtime_contract_exact(current, current)
    for key, changed_value in (
        ("deterministic_algorithms", False),
        ("cudnn_benchmark", True),
        ("cuda_matmul_allow_tf32", True),
        ("cudnn_allow_tf32", True),
        ("cublas_workspace_config", ":16:8"),
    ):
        tampered = {**current, key: changed_value}
        with pytest.raises(DataValidationError, match="runtime contract"):
            official_runner._assert_runtime_contract_exact(tampered, current)


def test_missing_pilot_population_is_sealed_and_reaudits_as_incomplete(
    tmp_path: Path,
) -> None:
    config = load_official_cssr_config(CONFIG_PATH)
    root = tmp_path / "pilot"
    summary = aggregate_phase_root(
        root,
        config=config,
        bundle_root=tmp_path / "missing-bundle",
        r2_results_root=tmp_path / "missing-r2",
        oracle_audit_path=tmp_path / "missing-oracle.json",
        phase="pilot",
        device_request="cpu",
        smoke_root=tmp_path / "missing-smoke",
    )
    assert summary["status"] == "hard_failed_incomplete"
    assert summary["gate"]["pilot_gate"] == "not_evaluated"
    assert summary["gate"]["selected_method"] is None
    assert (root / "_PHASE_INCOMPLETE.json").is_file()
    assert not (root / "_PHASE_SUCCESS.json").exists()
    assert not (root / "metrics_by_pair.csv").exists()
    audited = audit_phase_root(
        root,
        config=config,
        bundle_root=tmp_path / "missing-bundle",
        r2_results_root=tmp_path / "missing-r2",
        oracle_audit_path=tmp_path / "missing-oracle.json",
        phase="pilot",
        device_request="cpu",
        smoke_root=tmp_path / "missing-smoke",
    )
    assert audited["status"] == "passed"
    assert audited["phase_status"] == "hard_failed_incomplete"
    with pytest.raises(DataValidationError, match="fresh root"):
        aggregate_phase_root(
            root,
            config=config,
            bundle_root=tmp_path / "missing-bundle",
            r2_results_root=tmp_path / "missing-r2",
            oracle_audit_path=tmp_path / "missing-oracle.json",
            phase="pilot",
            device_request="cpu",
            smoke_root=tmp_path / "missing-smoke",
        )


def test_task_source_manifest_covers_imported_stage_a_and_stage_b_modules() -> None:
    script = f"""
import json
import pathlib
import sys
sys.path.insert(0, {str(PROJECT_ROOT / 'src')!r})
import hrrp_osr.training.official_cssr_hrrp_pilot
import hrrp_osr.training.cssr_identity_failure_audit
import hrrp_osr.training.fg_mv_cssr_decoupled
root = pathlib.Path({str(PROJECT_ROOT)!r}).resolve()
loaded = []
for module in tuple(sys.modules.values()):
    path = getattr(module, '__file__', None)
    if not path:
        continue
    candidate = pathlib.Path(path).resolve()
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        continue
    if str(relative).startswith('src/hrrp_osr/') and relative.suffix == '.py':
        loaded.append(str(relative))
print(json.dumps(sorted(set(loaded))))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    loaded = set(json.loads(completed.stdout))
    assert loaded <= set(TASK_SOURCE_FILES), sorted(loaded - set(TASK_SOURCE_FILES))


def test_completed_smoke_authorizes_pilot_without_binding_host_paths(
    tmp_path: Path,
) -> None:
    config = load_official_cssr_config(CONFIG_PATH)
    source_hashes = _task_source_hashes(PROJECT_ROOT)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dtype_record = {
        "passed": True,
        "dtype": "float32",
        "rtol": 1.0e-5,
        "atol": 1.0e-6,
        "clip_boundary_checks": {"passed": True},
        "pair_checks": {"passed": True},
        "deterministic_repeat": {"passed": True},
        "max_absolute_differences": {"diagnostic": 0.0},
    }
    oracle = {
        "passed": True,
        "status": "passed",
        "official_root": "/runtime/host-a/cssr",
        "official_commit": "fixed",
        "file_sha256": {"file": "hash"},
        "verified_file_sha256": {"file": "hash"},
        "source_execution": "ast",
        "method_ids": {"official": "id"},
        "oracle_contract": {"contract": "fixed"},
        "float32": "passed",
        "float64": "passed",
        "dtype_checks": {
            "float32": dtype_record,
            "float64": {**dtype_record, "dtype": "float64"},
        },
    }
    oracle_identity_sha256 = _json_sha256(_official_audit_identity(oracle))
    root = tmp_path / "smoke"
    unit_hashes: dict[str, str] = {}
    for method in TRAINABLE_METHODS:
        unit = root / "N1" / "fold_0" / "seed_20260906" / method
        unit.mkdir(parents=True)
        saved_oracle = copy.deepcopy(oracle)
        saved_oracle["official_root"] = "/original/gpu/checkout/cssr"
        _write_json(unit / "official_oracle_audit.json", saved_oracle)
        _write_json(
            unit / "checkpoint_replay_audit.json",
            {
                "status": "passed",
                "state_dict_strict_load": True,
                "prediction_rows_exact": True,
                "metrics_exact": True,
                "all_prediction_score_fields_exact": True,
                "evaluation_logits_probabilities_exact": True,
                "real_pair_view_swap_reinference": True,
            },
        )
        _write_json(
            unit / "unit_contract.json",
            {
                "phase": "smoke",
                "pair_id": "N1",
                "method": method,
                "code_commit": commit,
                "config_sha256": config["_config_sha256"],
                "source_hashes": source_hashes,
                "smoke_authorization": None,
            },
        )
        manifest = _artifact_hashes(unit)
        _write_json(unit / "artifact_hashes.json", manifest)
        _write_json(
            unit / "_SUCCESS.json",
            {
                "status": "success",
                "artifact_count": len(manifest),
                "artifact_hashes_sha256": file_sha256(
                    unit / "artifact_hashes.json"
                ),
            },
        )
        unit_hashes[method] = file_sha256(unit / "artifact_hashes.json")
    _write_json(
        root / "phase_summary.json",
        {
            "status": "complete",
            "phase": "smoke",
            "decision": "diagnostic_smoke_only",
            "unit_count": 4,
            "config_sha256": config["_config_sha256"],
            "official_oracle_identity_sha256": oracle_identity_sha256,
            "code_commit": commit,
            "source_hashes_sha256": _json_sha256(source_hashes),
            "smoke_authorization": None,
            "diagnostic_only": True,
            "confirmation_allowed": False,
            "automatic_followon_authorized": False,
            "final_unknown_used": False,
            "even_angle_test_used": False,
            "final_unknown_test_authorized": False,
        },
    )
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
    authorization = _read_smoke_authorization(
        root,
        config=config,
        oracle_audit=oracle,
    )
    assert authorization["pilot_authorized"] is True
    assert authorization["official_oracle_identity_sha256"] == oracle_identity_sha256
    assert "smoke_root" not in authorization
    assert authorization["unit_artifact_manifest_sha256"] == unit_hashes


def test_formal_oracle_consumer_accepts_the_complete_frozen_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    official_root = Path("/private/tmp/cssr-official-d5a99e91")
    if not official_root.exists():
        pytest.skip("fixed official CSSR checkout is unavailable")
    record = audit_official_cssr_oracle(official_root, device="cpu")
    record["device"] = "cuda"
    record["cuda_device_name"] = "NVIDIA GeForce RTX 4090"
    record["runtime_contract"] = {
        **record["runtime_contract"],
        "device": "cuda",
        "device_type": "cuda",
        "cuda_device_name": "NVIDIA GeForce RTX 4090",
        "expected_cuda_device_name": "NVIDIA GeForce RTX 4090",
        "formal_cuda_device_match": True,
        "cublas_workspace_config": ":4096:8",
    }
    path = tmp_path / "oracle.json"
    _write_json(path, record)
    monkeypatch.setattr(
        official_runner,
        "audit_official_cssr_oracle",
        lambda _root, device: copy.deepcopy(record),
    )
    config = load_official_cssr_config(CONFIG_PATH)
    assert _official_audit_record(path, config) == record


def test_development_bundle_blocks_final_unknown_and_even_angle_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_official_cssr_config(CONFIG_PATH)
    (tmp_path / "profiles.npy").touch()
    (tmp_path / "samples.csv").touch()
    names = list(config["classes"]["source_known_order"]) + ["U1", "U2", "U3"]
    rows = []
    for class_index, class_name in enumerate(names):
        role = "known" if class_index < 7 else "unknown"
        for angle in range(360):
            index = class_index * 360 + angle
            rows.append(
                {
                    "processed_row_index": index,
                    "sample_id": f"sample-{index}",
                    "class_name": class_name,
                    "class_role": role,
                    "angle_deg": angle,
                    "eligible_for_training": int(role == "known" and angle % 2 == 1),
                    "eligible_for_validation": int(role == "known" and angle % 2 == 1),
                }
            )
    values = np.zeros((3600, 601), dtype=np.float64)
    monkeypatch.setattr(official_runner.np, "load", lambda *args, **kwargs: values)
    monkeypatch.setattr(official_runner, "_load_processed_rows", lambda path: tuple(rows))

    def fake_sidecar(path: Path, expected_name: str) -> str:
        del expected_name
        if path.name == "profiles.npy.sha256":
            return str(config["bundle"]["profiles_sha256"])
        if path.name == "samples.csv.sha256":
            return str(config["bundle"]["manifest_sha256"])
        return str(config["bundle"]["bundle_sha256"])

    monkeypatch.setattr(official_runner, "_read_processed_hash", fake_sidecar)
    monkeypatch.setattr(
        official_runner,
        "file_sha256",
        lambda path: (
            str(config["bundle"]["profiles_sha256"])
            if Path(path).name == "profiles.npy"
            else str(config["bundle"]["manifest_sha256"])
        ),
    )
    bundle = _load_development_only_bundle(tmp_path, config)
    audit = _profile_access_audit(bundle)
    assert len(bundle.rows) == 7 * 180
    assert audit["authorized_row_count"] == 7 * 180
    assert audit["final_unknown_profile_values_read"] is False
    assert audit["even_angle_profile_values_read"] is False
    assert bundle.profiles[np.asarray([1], dtype=np.int64)].shape == (1, 601)
    with pytest.raises(DataValidationError, match="outside source-known odd-angle"):
        bundle.profiles[np.asarray([0], dtype=np.int64)]
    with pytest.raises(DataValidationError, match="outside source-known odd-angle"):
        bundle.profiles[np.asarray([7 * 360 + 1], dtype=np.int64)]
    with pytest.raises(DataValidationError, match="whole-array"):
        np.asarray(bundle.profiles)


def test_calibration_ece_includes_exact_internal_bin_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probabilities = np.full((1, 5), 0.2, dtype=np.float32)
    monkeypatch.setattr(
        official_runner,
        "_infer_single",
        lambda *args, **kwargs: {"probabilities": probabilities},
    )
    monkeypatch.setattr(
        official_runner,
        "_infer_pairs",
        lambda *args, **kwargs: {
            "probabilities": np.stack((probabilities, probabilities), axis=1)
        },
    )
    result = official_runner._calibration_summary(
        object(),
        single_inputs=np.zeros((1, 601), dtype=np.float32),
        single_labels=np.asarray([0], dtype=np.int64),
        pair_inputs=np.zeros((1, 2, 601), dtype=np.float32),
        pair_labels=np.asarray([0], dtype=np.int64),
        device=torch.device("cpu"),
        batch_size=1,
    )
    assert result["single_ece_15_bin"] == pytest.approx(0.8)
    assert result["pair_ece_15_bin"] == pytest.approx(0.8)


def test_successful_smoke_aggregate_never_writes_performance_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_official_cssr_config(CONFIG_PATH)
    audits = []
    task_rows = []
    for method in TRAINABLE_METHODS:
        current = {
            "status": "success",
            "audit_passed": True,
            "phase": "smoke",
            "pair_id": "N1",
            "method": method,
            "artifact_count": 1,
            "source_hashes": {"source.py": "hash"},
            "code_commit": "a" * 40,
            "smoke_authorization": None,
            "checkpoint_sha256": f"checkpoint-{method}",
            "checkpoint_replay": "exact",
            "metrics": {"auroc": 0.5},
        }
        audits.append(current)
        task_rows.append(
            official_runner._task_audit_row(
                {"pair_id": "N1", "method": method}, audit=current
            )
        )
    monkeypatch.setattr(
        official_runner,
        "_collect_phase_audits",
        lambda *args, **kwargs: (audits, task_rows),
    )
    monkeypatch.setattr(
        official_runner,
        "_phase_oracle_identity_sha256",
        lambda *args, **kwargs: "oracle-identity",
    )
    root = tmp_path / "smoke"
    summary = aggregate_phase_root(
        root,
        config=config,
        bundle_root=tmp_path / "unused-bundle",
        r2_results_root=tmp_path / "unused-r2",
        oracle_audit_path=tmp_path / "unused-oracle",
        phase="smoke",
        device_request="cpu",
    )
    assert summary["status"] == "complete"
    assert summary["decision"] == "diagnostic_smoke_only"
    assert not (root / "metrics_by_pair.csv").exists()
    assert not (root / "pilot_gate.json").exists()
    assert (root / "task_audit.csv").is_file()


def test_pilot_gate_failure_is_sealed_as_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_official_cssr_config(CONFIG_PATH)
    plan = build_phase_plan(config, "pilot")
    audits = [
        {
            "status": "success",
            "audit_passed": True,
            "phase": "pilot",
            "pair_id": row["pair_id"],
            "method": row["method"],
            "artifact_count": 1,
            "checkpoint_sha256": "checkpoint",
            "checkpoint_replay": "exact",
        }
        for row in plan
    ]
    task_rows = [
        official_runner._task_audit_row(row, audit=audit)
        for row, audit in zip(plan, audits, strict=True)
    ]
    aggregate_rows = {
        "metrics": [],
        "identities": [],
        "absorption": [],
        "ablations": [],
        "tasks": task_rows,
        "integrity": [],
    }
    monkeypatch.setattr(
        official_runner,
        "_collect_phase_audits",
        lambda *args, **kwargs: (audits, task_rows),
    )
    monkeypatch.setattr(
        official_runner,
        "_assert_common_unit_contract",
        lambda values: {"status": "passed"},
    )
    monkeypatch.setattr(
        official_runner,
        "_phase_oracle_identity_sha256",
        lambda *args, **kwargs: "oracle-identity",
    )
    monkeypatch.setattr(
        official_runner,
        "_aggregate_rows",
        lambda values: aggregate_rows,
    )
    monkeypatch.setattr(
        official_runner,
        "evaluate_pilot_gate",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            DataValidationError("synthetic gate failure")
        )
        if kwargs.get("audit_passed") is True
        else {
            "pilot_gate": "not_evaluated",
            "selected_method": None,
        },
    )
    root = tmp_path / "pilot"
    summary = aggregate_phase_root(
        root,
        config=config,
        bundle_root=tmp_path / "unused-bundle",
        r2_results_root=tmp_path / "unused-r2",
        oracle_audit_path=tmp_path / "unused-oracle",
        phase="pilot",
        device_request="cpu",
        smoke_root=tmp_path / "unused-smoke",
    )
    assert summary["status"] == "hard_failed_incomplete"
    assert summary["aggregate_failure"] == {
        "failure_type": "DataValidationError",
        "failure_message": "synthetic gate failure",
    }
    assert (root / "_PHASE_INCOMPLETE.json").is_file()
    assert not (root / "_PHASE_SUCCESS.json").exists()
    assert not (root / "metrics_by_pair.csv").exists()


def test_smoke_cli_never_displays_performance_fields() -> None:
    raw = {
        "status": "complete",
        "phase": "smoke",
        "pair_id": "N1",
        "method": O2_OFFICIAL_PCSSR_FT,
        "destination": "/external/smoke/unit",
        "metrics": {"auroc": 0.99},
        "o0_metrics": {"auroc": 0.50},
        "identity_rows": [{"auroc": 1.0}],
        "training_audit": {"loss": 0.1},
    }
    printable = official_runner._cli_safe_result(
        raw, command="run-unit", phase="smoke"
    )
    assert printable["performance_metrics_displayed"] is False
    assert printable["pair_id"] == "N1"
    assert not ({"metrics", "o0_metrics", "identity_rows", "training_audit"} & set(printable))
    assert official_runner._cli_safe_result(
        raw, command="run-unit", phase="pilot"
    ) == raw
