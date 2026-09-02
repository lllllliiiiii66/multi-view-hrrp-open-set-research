from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from hrrp_osr.data.errors import DataConfigError, DataValidationError
from hrrp_osr.data.manifest import file_sha256
from hrrp_osr.data.processed import load_processed_bundle
from hrrp_osr.evaluation.metrics import evaluate_open_set
from hrrp_osr.models.arpl import ARPLReciprocalHead
from hrrp_osr.models.mv_rpformer import METHODS, MVRPFormer
from hrrp_osr.training.arpl_pilot import (
    SOURCE_KNOWN_ORDER,
    PreparedSurrogateSplit,
    _artifact_hashes,
    _environment,
    _is_finite_model,
    _resolve_device,
    _set_determinism,
    _state_sha256,
    prepare_surrogate_split,
)
from hrrp_osr.training.arpl_mv_evidence import _length_padding_audit


EXPERIMENT_ID = "mv_rpformer_surrogate_v1"
DEVELOPMENT_SPLITS = (
    ("S0", 1, [2, 3, 4, 5, 6], [0, 1]),
    ("S1", 2, [0, 1, 4, 5, 6], [2, 3]),
    ("S2", 3, [0, 1, 2, 3, 6], [4, 5]),
)
CONFIRMATION_SPLITS = (
    ("C0", 0, [1, 2, 3, 4, 5], [0, 6]),
    ("C1", 4, [0, 2, 3, 4, 6], [1, 5]),
    ("C2", 0, [0, 1, 3, 5, 6], [2, 4]),
    ("C3", 4, [0, 1, 2, 4, 5], [3, 6]),
)
REJECTOR_METHODS = frozenset(METHODS[5:])
ARPL_METHODS = frozenset(METHODS[1:7])
METRIC_KEYS = (
    "known_accuracy",
    "known_macro_f1",
    "auroc",
    "oscr",
    "fpr95",
    "unknown_rejection_rate",
    "k_plus_1_macro_f1",
)
COMPARISONS = {
    "backbone": ("M2_MS_MEAN_ARPL", "M1_CURRENT_ARPL_MEAN"),
    "transformer_fusion": ("M3_MS_SET_GLOBAL_ARPL", "M2_MS_MEAN_ARPL"),
    "hierarchical_arpl": ("M4_MS_SET_HIER_ARPL", "M3_MS_SET_GLOBAL_ARPL"),
    "mismatch_rejector": ("M5_MV_RPFORMER_MISMATCH", "M4_MS_SET_HIER_ARPL"),
    "mixup_increment": ("M6_MV_RPFORMER_FULL", "M5_MV_RPFORMER_MISMATCH"),
    "learned_rejector": ("M6_MV_RPFORMER_FULL", "M4_MS_SET_HIER_ARPL"),
    "arpl_specific": ("M6_MV_RPFORMER_FULL", "M7_MV_CEFORMER_FULL"),
    "complete_vs_arpl_baseline": (
        "M6_MV_RPFORMER_FULL",
        "M1_CURRENT_ARPL_MEAN",
    ),
    "complete_vs_ce_baseline": (
        "M6_MV_RPFORMER_FULL",
        "M0_CURRENT_CE_MEAN",
    ),
}
TASK_SOURCE_FILES = (
    "configs/experiments/arpl/mv_rpformer_surrogate_v1.yaml",
    "src/hrrp_osr/models/arpl.py",
    "src/hrrp_osr/models/hrrp_ms_resnet.py",
    "src/hrrp_osr/models/mv_rpformer.py",
    "src/hrrp_osr/training/arpl_pilot.py",
    "src/hrrp_osr/training/mv_rpformer.py",
    "src/hrrp_osr/evaluation/metrics.py",
)


class NumericalInstabilityError(RuntimeError):
    pass


class IntentionalTrainingInterruption(RuntimeError):
    """Test-only fault injection raised after a recoverable epoch checkpoint."""


class IndexedPairDataset(Dataset):
    def __init__(self, inputs: np.ndarray, labels: np.ndarray) -> None:
        if inputs.ndim != 3 or inputs.shape[1:] != (2, 601):
            raise DataValidationError("pair inputs must have shape [n, 2, 601]")
        if labels.ndim != 1 or labels.shape[0] != inputs.shape[0]:
            raise DataValidationError("pair labels do not match inputs")
        self.inputs = torch.from_numpy(np.asarray(inputs, dtype=np.float32))
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.int64))

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.labels[index], torch.tensor(index, dtype=torch.long)


@dataclass(frozen=True)
class TrainKnownPseudoPool:
    inputs: torch.Tensor
    labels: torch.Tensor
    pair_ids: tuple[str, ...]
    view1_frames: torch.Tensor
    view2_frames: torch.Tensor
    source_role: str = "train_known"


def build_train_known_pseudo_pool(prepared: PreparedSurrogateSplit) -> TrainKnownPseudoPool:
    if prepared.pair_audit.get("final_unknown_pairs") != 0 or prepared.pair_audit.get(
        "even_angle_pairs"
    ) != 0:
        raise DataValidationError("pseudo source preparation is not isolated")
    view1_frames, view2_frames = _train_frame_arrays(prepared)
    return TrainKnownPseudoPool(
        inputs=torch.from_numpy(np.asarray(prepared.inputs["train"], dtype=np.float32)),
        labels=torch.from_numpy(np.asarray(prepared.labels["train"], dtype=np.int64)),
        pair_ids=prepared.pair_ids["train"],
        view1_frames=view1_frames,
        view2_frames=view2_frames,
    )


def require_train_known_pseudo_source(role: str) -> None:
    if role != "train_known":
        raise DataValidationError("pseudo unknown may only use the train_known role")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DataConfigError(f"{name} must be a mapping")
    return value


def _check_split_rows(
    value: Any,
    expected: Sequence[tuple[str, int, list[int], list[int]]],
    name: str,
    errors: list[str],
) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        errors.append(f"{name} must be a sequence")
        return
    observed = tuple(
        (
            row.get("split_id"),
            row.get("angle_fold"),
            list(row.get("train_known_indices", [])),
            list(row.get("surrogate_unknown_indices", [])),
        )
        for row in value
        if isinstance(row, Mapping)
    )
    if observed != tuple(expected):
        errors.append(f"{name} changed")


def load_mv_rpformer_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = dict(_mapping(yaml.safe_load(handle), "MV-RPFormer config"))
    errors: list[str] = []
    if (
        config.get("schema_version") != 1
        or config.get("stage") != "P3_surrogate_mv_rpformer"
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("result_scope")
        != "development_then_mandatory_confirmation_known_source_only"
    ):
        errors.append("experiment identity changed")
    scope = _mapping(config.get("evidence_scope"), "evidence_scope")
    if scope.get("source_known_odd_angle_only") is not True or any(
        scope.get(key) is not False
        for key in (
            "final_unknown_classes_used",
            "even_angle_test_used",
            "surrogate_unknown_used_for_training",
            "surrogate_unknown_used_for_checkpoint_selection",
            "angle_or_view_metadata_used_by_model",
        )
    ):
        errors.append("evidence isolation changed")
    reference = _mapping(config.get("official_reference"), "official_reference")
    if (
        reference.get("commit")
        != "3ede8b38e1cfb9d70e106cc19d563453110c36ab"
        or reference.get("repository") != "https://github.com/gary23ai/ARPL"
        or reference.get("dist_sha256")
        != "a05fc01c9051d8cb8d87cc7183e0a3d9fd1a11ca9de38d58a4870cb70ad4dc62"
        or reference.get("arploss_sha256")
        != "6dec41f0265b6665e8c66a27f506f176a0a7b0b2e4426760c09c203ab0c327ec"
    ):
        errors.append("official ARPL reference changed")
    bundle = _mapping(config.get("bundle"), "bundle")
    if (
        bundle.get("dataset_id") != "hrrp_10class_theta83_hh_v1"
        or bundle.get("preprocessing_id")
        != "hrrp_padding_complex_gaussian_v1"
        or bundle.get("profiles_sha256")
        != "2dd92282c125f0f677cf1f2dfce828781c8ba4385cf9ae552c4a2c56033c3f5b"
        or bundle.get("manifest_sha256")
        != "748b9f30629c3b3cbe66c6a1dac30863fdab2d81a214e46d8bc3ef7c6022a08a"
        or bundle.get("bundle_sha256")
        != "79176091b5b745e5df5957f1bb4ade7304c1403a4a32e64622e012db67d5a0c5"
    ):
        errors.append("processed bundle contract changed")
    classes = _mapping(config.get("classes"), "classes")
    if list(classes.get("source_known_order", [])) != list(SOURCE_KNOWN_ORDER):
        errors.append("source known class order changed")
    if classes.get("final_unknown_count") != 3:
        errors.append("final unknown class count changed")
    _check_split_rows(
        classes.get("development_splits"), DEVELOPMENT_SPLITS, "development_splits", errors
    )
    _check_split_rows(
        classes.get("confirmation_splits"), CONFIRMATION_SPLITS, "confirmation_splits", errors
    )
    sampling = _mapping(config.get("sampling"), "sampling")
    if (
        sampling.get("fold_count") != 5
        or sampling.get("pair_protocol_source") != "arpl_lite_surrogate_osr_v1"
        or sampling.get("frame_width_deg") != 15
        or sampling.get("development_angle_parity") != "odd"
        or sampling.get("base_seed") != 20260830
        or sampling.get("slot_order") != "randomized_seeded"
        or sampling.get("distinct_frames") is not True
        or dict(_mapping(sampling.get("pairs_per_class"), "pairs_per_class"))
        != {"smoke": 10, "full": 500}
        or sampling.get("final_test_pairs_generated") is not False
    ):
        errors.append("sampling protocol changed")
    normalization = _mapping(config.get("normalization"), "normalization")
    if (
        normalization.get("method") != "global_scalar_zscore"
        or normalization.get("fit_population")
        != "unique_train_known_base_samples_only"
        or float(normalization.get("epsilon", -1)) != 1e-8
    ):
        errors.append("normalization protocol changed")
    model = _mapping(config.get("model"), "model")
    encoder = _mapping(model.get("encoder"), "model.encoder")
    transformer = _mapping(model.get("set_transformer"), "model.set_transformer")
    rejector = _mapping(model.get("rejector"), "model.rejector")
    arpl = _mapping(model.get("arpl"), "model.arpl")
    if (
        list(model.get("methods", [])) != list(METHODS)
        or list(model.get("input_shape", [])) != [2, 601]
        or model.get("feature_dim") != 128
        or encoder.get("architecture") != "hrrp_ms_resnet_v1"
        or encoder.get("input_length") != 601
        or encoder.get("stem_channels") != 32
        or encoder.get("stem_kernel_size") != 31
        or list(encoder.get("stage_channels", [])) != [32, 64, 128]
        or list(encoder.get("branch_kernel_sizes", [])) != [3, 7, 15]
        or encoder.get("branch_convolution") != "standard_conv1d"
        or encoder.get("activation") != "GELU"
        or encoder.get("shared_between_views") is not True
        or float(encoder.get("dropout", -1)) != 0.1
        or list(encoder.get("pooling", [])) != ["global_average", "global_max"]
        or encoder.get("projection_dim") != 128
        or transformer.get("sab_layers") != 1
        or transformer.get("embedding_dim") != 128
        or transformer.get("num_heads") != 4
        or transformer.get("ffn_hidden_dim") != 256
        or float(transformer.get("dropout", -1)) != 0.1
        or transformer.get("pre_layer_norm") is not True
        or transformer.get("position_encoding") is not False
        or transformer.get("pma_seed_count") != 2
        or transformer.get("output_sab_layers") != 0
        or list(rejector.get("hidden_dims", [])) != [64, 32]
        or float(rejector.get("dropout", -1)) != 0.1
        or rejector.get("global_prediction_conditions_view_support") is not True
        or rejector.get("symmetric_support_order") != "sorted_descending"
        or rejector.get("output") != "sigmoid_probability_unknown"
        or arpl.get("num_centers_per_class") != 1
        or float(arpl.get("reciprocal_init_std", -1)) != 0.1
        or float(arpl.get("initial_radius", -1)) != 0.0
        or float(arpl.get("temperature", -1)) != 1.0
        or float(arpl.get("weight_pl", -1)) != 0.1
        or float(arpl.get("margin", -1)) != 1.0
        or float(arpl.get("lambda_view", -1)) != 0.5
        or arpl.get("global_view_heads_independent") is not True
    ):
        errors.append("model contract changed")
    pseudo = _mapping(config.get("pseudo_unknown"), "pseudo_unknown")
    mismatch = _mapping(pseudo.get("mismatch"), "pseudo_unknown.mismatch")
    mixup = _mapping(pseudo.get("coherent_mixup"), "pseudo_unknown.coherent_mixup")
    composition = _mapping(pseudo.get("composition"), "pseudo_unknown.composition")
    expected_composition = {
        "M5_MV_RPFORMER_MISMATCH": {"mismatch": 1.0, "coherent_mixup": 0.0},
        "M6_MV_RPFORMER_FULL": {"mismatch": 0.5, "coherent_mixup": 0.5},
        "M7_MV_CEFORMER_FULL": {"mismatch": 0.5, "coherent_mixup": 0.5},
    }
    if (
        pseudo.get("train_known_only") is not True
        or float(pseudo.get("real_to_pseudo_ratio", -1)) != 1.0
        or mismatch.get("require_different_classes") is not True
        or mismatch.get("use_view1_from_anchor_view2_from_partner") is not True
        or mixup.get("require_different_classes") is not True
        or float(mixup.get("beta_alpha", -1)) != 2.0
        or float(mixup.get("beta_beta", -1)) != 2.0
        or float(mixup.get("lambda_min", -1)) != 0.3
        or float(mixup.get("lambda_max", -1)) != 0.7
        or mixup.get("same_lambda_for_both_views") is not True
        or {name: dict(_mapping(value, name)) for name, value in composition.items()}
        != expected_composition
    ):
        errors.append("pseudo-unknown protocol changed")
    loss = _mapping(config.get("loss"), "loss")
    if (
        float(loss.get("lambda_reject", -1)) != 1.0
        or float(loss.get("lambda_uniform", -1)) != 0.1
        or loss.get("uniform_direction") != "KL_uniform_to_softmax"
    ):
        errors.append("loss weights changed")
    training = _mapping(config.get("training"), "training")
    if (
        list(training.get("development_seeds", [])) != [20260830]
        or list(training.get("confirmation_seeds", []))
        != [20260830, 20260831, 20260832]
        or list(training.get("development_methods", [])) != list(METHODS)
        or list(training.get("confirmation_methods", [])) != list(METHODS)
        or training.get("optimizer") != "AdamW"
        or float(training.get("learning_rate", -1)) != 3e-4
        or float(training.get("weight_decay", -1)) != 1e-4
        or int(training.get("batch_size", 0)) != 64
        or training.get("batch_size_semantics")
        != "64_real_known_plus_64_generated_pseudo_when_active"
        or int(training.get("total_epochs", 0)) != 100
        or int(training.get("smoke_epochs", 0)) != 31
        or int(training.get("representation_only_epochs", 0)) != 30
        or training.get("scheduler") != "warmup_cosine"
        or int(training.get("warmup_epochs", 0)) != 5
        or int(training.get("formal_checkpoint_epoch", 0)) != 100
        or training.get("early_stopping") is not False
        or training.get("calibration_checkpoint_selection") is not False
        or int(training.get("dataloader_seed_offset", -1)) != 1
        or int(training.get("pseudo_seed_offset", -1)) != 2
        or training.get("deterministic_algorithms") is not True
        or int(training.get("num_workers", -1)) != 0
    ):
        errors.append("training contract changed")
    evaluation = _mapping(config.get("evaluation"), "evaluation")
    if (
        evaluation.get("base_unknown_score")
        != "negative_maximum_global_raw_logit"
        or evaluation.get("learned_unknown_score")
        != "rejector_probability_unknown"
        or evaluation.get("unknown_score_direction") != "larger_is_more_unknown"
        or evaluation.get("threshold_source") != "known_calibration_only"
        or float(evaluation.get("threshold_known_acceptance_rate", -1)) != 0.95
        or list(evaluation.get("report_metrics", [])) != list(METRIC_KEYS)
    ):
        errors.append("evaluation contract changed")
    runtime = _mapping(config.get("runtime"), "runtime")
    if (
        runtime.get("torch_intraop_threads") != 4
        or runtime.get("torch_interop_threads") != 1
        or runtime.get("recommended_parallel_jobs") != 2
    ):
        errors.append("runtime resource contract changed")
    decision = _mapping(config.get("confirmation_decision"), "confirmation_decision")
    main_gate = _mapping(decision.get("m6_vs_m4"), "confirmation_decision.m6_vs_m4")
    arpl_gate = _mapping(decision.get("m6_vs_m7"), "confirmation_decision.m6_vs_m7")
    if (
        decision.get("unit_count") != 12
        or float(main_gate.get("minimum_mean_auroc_delta", -1)) != 0.02
        or int(main_gate.get("minimum_positive_auroc_units", 0)) != 8
        or float(main_gate.get("minimum_mean_oscr_delta", -1)) != 0.0
        or float(main_gate.get("maximum_mean_known_accuracy_drop", -1)) != 0.01
        or float(main_gate.get("maximum_mean_fpr95_increase", -1)) != 0.02
        or float(arpl_gate.get("minimum_mean_auroc_delta", -1)) != 0.01
        or int(arpl_gate.get("minimum_positive_auroc_units", 0)) != 7
    ):
        errors.append("confirmation decision rule changed")
    outputs = _mapping(config.get("outputs"), "outputs")
    if (
        outputs.get("namespace") != "artifacts/arpl/mv_rpformer_surrogate_v1"
        or outputs.get("fail_if_output_nonempty") is not True
        or any(
            outputs.get(key) is not True
            for key in (
                "save_resolved_config",
                "save_pair_manifest_and_hash",
                "save_real_pseudo_pair_audit",
                "save_epoch_100_checkpoint",
                "save_attention_weights",
                "save_tokens_logits_scores",
                "save_reciprocal_points_and_radius_trajectories",
                "save_predictions",
                "save_training_logs",
                "save_environment_and_hashes",
            )
        )
    ):
        errors.append("output isolation changed")
    if errors:
        raise DataConfigError("Invalid MV-RPFormer config:\n- " + "\n- ".join(errors))
    config["_config_path"] = str(config_path)
    config["_config_sha256"] = file_sha256(config_path)
    return config


def build_phase_plan(config: Mapping[str, Any], phase: str) -> list[dict[str, Any]]:
    if phase == "smoke":
        specs = [config["classes"]["development_splits"][0]]
        seeds = [int(config["training"]["development_seeds"][0])]
        methods = list(METHODS)
        mode = "smoke"
    elif phase == "development":
        specs = config["classes"]["development_splits"]
        seeds = [int(value) for value in config["training"]["development_seeds"]]
        methods = list(config["training"]["development_methods"])
        mode = "full"
    elif phase == "confirmation":
        specs = config["classes"]["confirmation_splits"]
        seeds = [int(value) for value in config["training"]["confirmation_seeds"]]
        methods = list(config["training"]["confirmation_methods"])
        mode = "full"
    else:
        raise DataConfigError("phase must be smoke, development, or confirmation")
    return [
        {"phase": phase, "mode": mode, "spec": spec, "seed": seed, "methods": methods}
        for spec in specs
        for seed in seeds
    ]


def loss_weights_for_epoch(epoch: int, representation_only_epochs: int = 30) -> dict[str, float]:
    if epoch < 1:
        raise ValueError("epoch is one-based")
    active = epoch > representation_only_epochs
    return {"representation": 1.0, "reject": float(active), "uniform": 0.1 * float(active)}


def learning_rate_for_epoch(
    epoch: int,
    *,
    base_learning_rate: float = 3e-4,
    warmup_epochs: int = 5,
    total_epochs: int = 100,
) -> float:
    if not 1 <= epoch <= total_epochs:
        raise ValueError("epoch is outside the training schedule")
    if epoch <= warmup_epochs:
        return base_learning_rate * epoch / warmup_epochs
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return base_learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


def uniform_kl_loss(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("pseudo logits must have shape [batch, classes>=2]")
    uniform = torch.full_like(logits, 1.0 / logits.shape[1])
    return F.kl_div(F.log_softmax(logits, dim=1), uniform, reduction="batchmean")


def sample_cross_class_indices(
    anchor_labels: torch.Tensor,
    pool_labels: torch.Tensor,
    *,
    generator: torch.Generator,
    anchor_frames: torch.Tensor | None = None,
    pool_partner_frames: torch.Tensor | None = None,
) -> torch.Tensor:
    anchor = anchor_labels.detach().cpu().long().reshape(-1)
    pool = pool_labels.detach().cpu().long().reshape(-1)
    choices: list[int] = []
    frames = None if anchor_frames is None else anchor_frames.detach().cpu().long().reshape(-1)
    pool_frames = (
        None
        if pool_partner_frames is None
        else pool_partner_frames.detach().cpu().long().reshape(-1)
    )
    if (frames is None) != (pool_frames is None) or (
        frames is not None and frames.shape != anchor.shape
    ):
        raise ValueError("frame constraints must be supplied together and match anchors")
    for index, label in enumerate(anchor.tolist()):
        mask = pool != label
        if frames is not None:
            mask &= pool_frames != frames[index]
        eligible = torch.nonzero(mask, as_tuple=False).flatten()
        if eligible.numel() == 0:
            raise DataValidationError("cross-class pseudo sampling has no eligible partner")
        selected = torch.randint(eligible.numel(), (1,), generator=generator).item()
        choices.append(int(eligible[selected]))
    result = torch.tensor(choices, dtype=torch.long)
    if torch.any(pool[result] == anchor):
        raise DataValidationError("pseudo partner has the same class as its anchor")
    return result


def sample_cross_class_mismatch(
    anchor_inputs: torch.Tensor,
    anchor_labels: torch.Tensor,
    partner_inputs: torch.Tensor,
    partner_labels: torch.Tensor,
) -> torch.Tensor:
    if anchor_inputs.ndim != 3 or partner_inputs.ndim != 3:
        raise ValueError("mismatch inputs must be two-view tensors")
    if anchor_inputs.shape != partner_inputs.shape or anchor_inputs.shape[1:] != (2, 601):
        raise ValueError("mismatch input shapes differ")
    if torch.any(anchor_labels.detach().cpu() == partner_labels.detach().cpu()):
        raise DataValidationError("mismatch source classes must differ")
    return torch.stack([anchor_inputs[:, 0], partner_inputs[:, 1]], dim=1)


def truncated_beta_lambdas(
    count: int,
    *,
    rng: np.random.Generator,
    alpha: float = 2.0,
    beta: float = 2.0,
    minimum: float = 0.3,
    maximum: float = 0.7,
) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=np.float32)
    accepted: list[float] = []
    while len(accepted) < count:
        candidates = rng.beta(alpha, beta, size=max(8, 2 * (count - len(accepted))))
        accepted.extend(float(value) for value in candidates if minimum <= value <= maximum)
    return np.asarray(accepted[:count], dtype=np.float32)


def coherent_feature_mixup(
    anchor_features: torch.Tensor,
    partner_features: torch.Tensor,
    lambdas: torch.Tensor,
) -> torch.Tensor:
    if anchor_features.shape != partner_features.shape or anchor_features.ndim != 3:
        raise ValueError("mixup features must share shape [batch, views, dim]")
    if lambdas.ndim != 1 or lambdas.shape[0] != anchor_features.shape[0]:
        raise ValueError("mixup lambdas must have shape [batch]")
    weights = lambdas.to(anchor_features).reshape(-1, 1, 1)
    return weights * anchor_features + (1.0 - weights) * partner_features


class PseudoAuditAccumulator:
    def __init__(
        self,
        *,
        method: str,
        seed: int,
        train_pair_rows: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        self.method = method
        self.seed = int(seed)
        self.schedule_hash = hashlib.sha256(b"").hexdigest()
        self.real_count = 0
        self.mismatch_count = 0
        self.mixup_count = 0
        self.same_class_violations = 0
        self.same_frame_violations = 0
        self.lambda_min = np.inf
        self.lambda_max = -np.inf
        self.anchor_class_counts: dict[str, int] = {}
        self.partner_class_counts: dict[str, int] = {}
        self.mismatch_anchor_frame_counts: dict[str, int] = {}
        self.mismatch_partner_frame_counts: dict[str, int] = {}
        self.examples: list[dict[str, Any]] = []
        self.train_pair_rows = tuple(dict(row) for row in train_pair_rows)
        self.train_pair_order_sha256 = hashlib.sha256(
            "\0".join(str(row.get("pair_id")) for row in self.train_pair_rows).encode(
                "utf-8"
            )
        ).hexdigest()

    @staticmethod
    def _increment(counter: dict[str, int], values: np.ndarray) -> None:
        unique, counts = np.unique(values, return_counts=True)
        for value, count in zip(unique, counts, strict=True):
            key = str(int(value))
            counter[key] = counter.get(key, 0) + int(count)

    def update(
        self,
        *,
        kind: str,
        anchor_indices: torch.Tensor,
        partner_indices: torch.Tensor,
        anchor_labels: torch.Tensor,
        partner_labels: torch.Tensor,
        lambdas: np.ndarray | None = None,
        anchor_frames: torch.Tensor | None = None,
        partner_frames: torch.Tensor | None = None,
    ) -> None:
        arrays = [
            anchor_indices.detach().cpu().numpy().astype(np.int64),
            partner_indices.detach().cpu().numpy().astype(np.int64),
            anchor_labels.detach().cpu().numpy().astype(np.int64),
            partner_labels.detach().cpu().numpy().astype(np.int64),
        ]
        event = hashlib.sha256()
        event.update(kind.encode("ascii"))
        for array in arrays:
            event.update(array.tobytes())
        self._increment(self.anchor_class_counts, arrays[2])
        self._increment(self.partner_class_counts, arrays[3])
        violations = int(np.count_nonzero(arrays[2] == arrays[3]))
        self.same_class_violations += violations
        if anchor_frames is not None and partner_frames is not None:
            anchor_frame_values = anchor_frames.detach().cpu().numpy().astype(np.int64)
            partner_frame_values = partner_frames.detach().cpu().numpy().astype(np.int64)
            event.update(anchor_frame_values.tobytes())
            event.update(partner_frame_values.tobytes())
            self._increment(self.mismatch_anchor_frame_counts, anchor_frame_values)
            self._increment(self.mismatch_partner_frame_counts, partner_frame_values)
            self.same_frame_violations += int(
                np.count_nonzero(anchor_frame_values == partner_frame_values)
            )
        count = int(arrays[0].size)
        if kind == "mismatch":
            self.mismatch_count += count
        elif kind == "mixup":
            self.mixup_count += count
        else:
            raise ValueError("unknown pseudo kind")
        if lambdas is not None:
            values = np.asarray(lambdas, dtype=np.float32)
            event.update(values.tobytes())
            self.lambda_min = min(self.lambda_min, float(values.min()))
            self.lambda_max = max(self.lambda_max, float(values.max()))
        self.schedule_hash = hashlib.sha256(
            bytes.fromhex(self.schedule_hash) + event.digest()
        ).hexdigest()
        remaining = max(0, 20 - len(self.examples))
        for index in range(min(remaining, count)):
            example = {
                "kind": kind,
                "anchor_train_index": int(arrays[0][index]),
                "partner_train_index": int(arrays[1][index]),
                "anchor_class": int(arrays[2][index]),
                "partner_class": int(arrays[3][index]),
            }
            if self.train_pair_rows:
                anchor_row = self.train_pair_rows[example["anchor_train_index"]]
                partner_row = self.train_pair_rows[example["partner_train_index"]]
                example.update(
                    {
                        "anchor_pair_id": str(anchor_row["pair_id"]),
                        "partner_pair_id": str(partner_row["pair_id"]),
                        "anchor_view1_sample_id": str(anchor_row["view1_sample_id"]),
                        "partner_view2_sample_id": str(partner_row["view2_sample_id"]),
                    }
                )
            if anchor_frames is not None and partner_frames is not None:
                example["anchor_view1_frame"] = int(anchor_frame_values[index])
                example["partner_view2_frame"] = int(partner_frame_values[index])
            if lambdas is not None:
                example["lambda"] = float(values[index])
            self.examples.append(example)

    def state_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "seed": self.seed,
            "schedule_hash": self.schedule_hash,
            "train_pair_order_sha256": self.train_pair_order_sha256,
            "real_count": self.real_count,
            "mismatch_count": self.mismatch_count,
            "mixup_count": self.mixup_count,
            "same_class_violations": self.same_class_violations,
            "same_frame_violations": self.same_frame_violations,
            "lambda_min": self.lambda_min,
            "lambda_max": self.lambda_max,
            "anchor_class_counts": self.anchor_class_counts,
            "partner_class_counts": self.partner_class_counts,
            "mismatch_anchor_frame_counts": self.mismatch_anchor_frame_counts,
            "mismatch_partner_frame_counts": self.mismatch_partner_frame_counts,
            "examples": self.examples,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("method") != self.method or int(state.get("seed", -1)) != self.seed:
            raise DataValidationError("pseudo audit resume identity differs")
        if state.get("train_pair_order_sha256") != self.train_pair_order_sha256:
            raise DataValidationError("pseudo audit train pair order differs")
        self.schedule_hash = str(state["schedule_hash"])
        for name in (
            "real_count",
            "mismatch_count",
            "mixup_count",
            "same_class_violations",
            "same_frame_violations",
        ):
            setattr(self, name, int(state[name]))
        self.lambda_min = float(state["lambda_min"])
        self.lambda_max = float(state["lambda_max"])
        for name in (
            "anchor_class_counts",
            "partner_class_counts",
            "mismatch_anchor_frame_counts",
            "mismatch_partner_frame_counts",
        ):
            setattr(self, name, {str(key): int(value) for key, value in state[name].items()})
        self.examples = [dict(value) for value in state["examples"]]

    def to_json(self) -> dict[str, Any]:
        pseudo_count = self.mismatch_count + self.mixup_count
        return {
            "status": (
                "passed"
                if self.same_class_violations == 0 and self.same_frame_violations == 0
                else "failed"
            ),
            "method": self.method,
            "seed": self.seed,
            "source_role": "train_known_only",
            "train_pair_order_sha256": self.train_pair_order_sha256,
            "replay_contract": {
                "torch_seed": self.seed + 2,
                "numpy_seed": self.seed + 2,
                "partner_sampling": "uniform_eligible_cross_class_v1",
                "mismatch_frame_constraint": "anchor_view1_frame_ne_partner_view2_frame",
                "mixup_distribution": "beta_2_2_rejection_truncated_0.3_0.7",
            },
            "surrogate_unknown_used": False,
            "final_unknown_used": False,
            "even_angle_test_used": False,
            "real_count": self.real_count,
            "pseudo_count": pseudo_count,
            "mismatch_count": self.mismatch_count,
            "mixup_count": self.mixup_count,
            "real_pseudo_balanced": self.real_count == pseudo_count,
            "same_class_violations": self.same_class_violations,
            "mismatch_same_frame_violations": self.same_frame_violations,
            "mixup_lambda_min": None if self.mixup_count == 0 else self.lambda_min,
            "mixup_lambda_max": None if self.mixup_count == 0 else self.lambda_max,
            "anchor_class_counts": self.anchor_class_counts,
            "partner_class_counts": self.partner_class_counts,
            "mismatch_anchor_view1_frame_counts": self.mismatch_anchor_frame_counts,
            "mismatch_partner_view2_frame_counts": self.mismatch_partner_frame_counts,
            "example_records": self.examples,
            "schedule_sha256": self.schedule_hash,
        }


def _write_json(path: Path, value: Any) -> None:
    _atomic_write_bytes(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )


def _atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _render_csv(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise DataValidationError("cannot render an empty CSV")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_write_bytes(path, _render_csv(rows))


def task_source_hashes(project_root: Path) -> dict[str, str]:
    hashes = {}
    for relative in TASK_SOURCE_FILES:
        path = project_root / relative
        if not path.is_file():
            raise DataValidationError(f"task source file is missing: {relative}")
        hashes[relative] = file_sha256(path)
    return hashes


def _prepare_split(
    bundle: Any,
    config: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    mode: str,
) -> PreparedSurrogateSplit:
    return prepare_surrogate_split(
        bundle,
        source_known_order=config["classes"]["source_known_order"],
        split_id=str(spec["split_id"]),
        angle_fold=int(spec["angle_fold"]),
        train_known_indices=spec["train_known_indices"],
        surrogate_unknown_indices=spec["surrogate_unknown_indices"],
        pairs_per_class=int(config["sampling"]["pairs_per_class"][mode]),
        base_seed=int(config["sampling"]["base_seed"]),
        fold_count=int(config["sampling"]["fold_count"]),
        normalization_epsilon=float(config["normalization"]["epsilon"]),
    )


def _build_model(method: str, known_class_count: int, config: Mapping[str, Any]) -> MVRPFormer:
    arpl = config["model"]["arpl"]
    return MVRPFormer(
        method,
        known_class_count,
        feature_dim=int(config["model"]["feature_dim"]),
        dropout=float(config["model"]["encoder"]["dropout"]),
        temperature=float(arpl["temperature"]),
        weight_pl=float(arpl["weight_pl"]),
        margin=float(arpl["margin"]),
        reciprocal_init_std=float(arpl["reciprocal_init_std"]),
        initial_radius=float(arpl["initial_radius"]),
    )


def build_initialized_method_group(
    known_class_count: int,
    *,
    seed: int,
    config: Mapping[str, Any],
) -> tuple[dict[str, MVRPFormer], dict[str, Any]]:
    models: dict[str, MVRPFormer] = {}
    for method in METHODS:
        _set_determinism(seed, bool(config["training"]["deterministic_algorithms"]))
        models[method] = _build_model(method, known_class_count, config)
    arpl_reference = models["M6_MV_RPFORMER_FULL"].global_head.state_dict()
    for method in METHODS[1:7]:
        models[method].global_head.load_state_dict(arpl_reference)
    models["M0_CURRENT_CE_MEAN"].global_head.load_state_dict(
        models["M7_MV_CEFORMER_FULL"].global_head.state_dict()
    )
    view_reference = models["M6_MV_RPFORMER_FULL"].view_head.state_dict()
    for method in METHODS[4:7]:
        models[method].view_head.load_state_dict(view_reference)
    # Linear modules consume RNG during their constructor. Explicit copies make
    # the components shared by the M5/M6/M7 comparison start identically.
    models["M5_MV_RPFORMER_MISMATCH"].rejector.load_state_dict(
        models["M6_MV_RPFORMER_FULL"].rejector.state_dict()
    )
    models["M7_MV_CEFORMER_FULL"].rejector.load_state_dict(
        models["M6_MV_RPFORMER_FULL"].rejector.state_dict()
    )
    hashes = {
        method: {
            "encoder": _state_sha256(model.encoder.state_dict()),
            "sab": None if model.sab is None else _state_sha256(model.sab.state_dict()),
            "pma": None if model.pma is None else _state_sha256(model.pma.state_dict()),
            "rejector": (
                None if model.rejector is None else _state_sha256(model.rejector.state_dict())
            ),
            "global_head": _state_sha256(model.global_head.state_dict()),
            "view_head": (
                None if model.view_head is None else _state_sha256(model.view_head.state_dict())
            ),
        }
        for method, model in models.items()
    }
    if len({hashes[name]["encoder"] for name in METHODS[:2]}) != 1:
        raise DataValidationError("M0/M1 shallow encoder initialization differs")
    if len({hashes[name]["encoder"] for name in METHODS[2:]}) != 1:
        raise DataValidationError("M2-M7 multi-scale encoder initialization differs")
    if len({hashes[name]["sab"] for name in METHODS[3:]}) != 1:
        raise DataValidationError("M3-M7 SAB initialization differs")
    if len({hashes[name]["pma"] for name in METHODS[3:]}) != 1:
        raise DataValidationError("M3-M7 PMA initialization differs")
    if len({hashes[name]["rejector"] for name in METHODS[5:]}) != 1:
        raise DataValidationError("M5-M7 rejector initialization differs")
    if len({hashes[name]["global_head"] for name in METHODS[1:7]}) != 1:
        raise DataValidationError("M1-M6 ARPL global head initialization differs")
    if hashes[METHODS[0]]["global_head"] != hashes[METHODS[7]]["global_head"]:
        raise DataValidationError("M0/M7 CE global head initialization differs")
    if len({hashes[name]["view_head"] for name in METHODS[4:7]}) != 1:
        raise DataValidationError("M4-M6 ARPL view head initialization differs")
    return models, {
        "component_initialization_hashes": hashes,
        "shared_initialization_checks": {
            "M0_M1_encoder": True,
            "M2_M7_encoder": True,
            "M3_M7_sab_pma": True,
            "M5_M7_rejector": True,
            "M1_M6_global_arpl_head": True,
            "M0_M7_global_ce_head": True,
            "M4_M6_view_arpl_head": True,
        },
    }


def _train_frame_arrays(prepared: PreparedSurrogateSplit) -> tuple[torch.Tensor, torch.Tensor]:
    train_rows = [
        row for row in prepared.pair_manifest_rows if row["experiment_role"] == "train_known"
    ]
    if tuple(str(row["pair_id"]) for row in train_rows) != prepared.pair_ids["train"]:
        raise DataValidationError("train manifest and tensor order differ")
    return (
        torch.tensor([int(row["view1_frame_id"]) for row in train_rows], dtype=torch.long),
        torch.tensor([int(row["view2_frame_id"]) for row in train_rows], dtype=torch.long),
    )


def _rejector_training_terms(
    model: MVRPFormer,
    real_output: Any,
    real_inputs: torch.Tensor,
    real_labels: torch.Tensor,
    batch_indices: torch.Tensor,
    *,
    pool_inputs: torch.Tensor,
    pool_labels: torch.Tensor,
    pool_view1_frames: torch.Tensor,
    pool_view2_frames: torch.Tensor,
    torch_generator: torch.Generator,
    numpy_generator: np.random.Generator,
    composition: Mapping[str, float],
    audit: PseudoAuditAccumulator,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if real_output.unknown_probability is None:
        raise DataValidationError("rejector loss requested for a model without rejector")
    count = int(real_labels.numel())
    audit.real_count += count
    mismatch_count = int(round(count * float(composition["mismatch"])))
    mixup_count = count - mismatch_count
    order = torch.randperm(count, generator=torch_generator)
    mismatch_positions = order[:mismatch_count]
    mixup_positions = order[mismatch_count:]
    pseudo_outputs = []
    mismatch_bce = real_inputs.new_zeros(())
    mixup_bce = real_inputs.new_zeros(())
    if mismatch_count:
        anchor_labels_cpu = real_labels.detach().cpu()[mismatch_positions]
        anchor_global_indices = batch_indices[mismatch_positions]
        anchor_frames = pool_view1_frames[anchor_global_indices]
        partner_indices = sample_cross_class_indices(
            anchor_labels_cpu,
            pool_labels,
            generator=torch_generator,
            anchor_frames=anchor_frames,
            pool_partner_frames=pool_view2_frames,
        )
        partner_labels = pool_labels[partner_indices]
        mismatch_inputs = sample_cross_class_mismatch(
            real_inputs[mismatch_positions.to(device)],
            anchor_labels_cpu,
            pool_inputs[partner_indices].to(device),
            partner_labels,
        )
        mismatch_output = model(mismatch_inputs)
        pseudo_outputs.append(mismatch_output)
        mismatch_bce = F.binary_cross_entropy(
            mismatch_output.unknown_probability,
            torch.ones_like(mismatch_output.unknown_probability),
        )
        audit.update(
            kind="mismatch",
            anchor_indices=anchor_global_indices,
            partner_indices=partner_indices,
            anchor_labels=anchor_labels_cpu,
            partner_labels=partner_labels,
            anchor_frames=anchor_frames,
            partner_frames=pool_view2_frames[partner_indices],
        )
    if mixup_count:
        anchor_labels_cpu = real_labels.detach().cpu()[mixup_positions]
        anchor_global_indices = batch_indices[mixup_positions]
        partner_indices = sample_cross_class_indices(
            anchor_labels_cpu,
            pool_labels,
            generator=torch_generator,
        )
        partner_labels = pool_labels[partner_indices]
        partner_features = model.encode_views(pool_inputs[partner_indices].to(device))
        lambda_values = truncated_beta_lambdas(mixup_count, rng=numpy_generator)
        lambdas = torch.from_numpy(lambda_values).to(device)
        mixed_features = coherent_feature_mixup(
            real_output.raw_view_tokens[mixup_positions.to(device)],
            partner_features,
            lambdas,
        )
        mixup_output = model.forward_encoded(mixed_features)
        pseudo_outputs.append(mixup_output)
        mixup_bce = F.binary_cross_entropy(
            mixup_output.unknown_probability,
            torch.ones_like(mixup_output.unknown_probability),
        )
        audit.update(
            kind="mixup",
            anchor_indices=anchor_global_indices,
            partner_indices=partner_indices,
            anchor_labels=anchor_labels_cpu,
            partner_labels=partner_labels,
            lambdas=lambda_values,
        )
    real_bce = F.binary_cross_entropy(
        real_output.unknown_probability,
        torch.zeros_like(real_output.unknown_probability),
    )
    if mismatch_count and mixup_count:
        reject_loss = real_bce + 0.5 * mismatch_bce + 0.5 * mixup_bce
    elif mismatch_count:
        reject_loss = real_bce + mismatch_bce
    else:
        reject_loss = real_bce + mixup_bce
    pseudo_logits = torch.cat([output.global_logits for output in pseudo_outputs], dim=0)
    return {
        "reject": reject_loss,
        "uniform": uniform_kl_loss(pseudo_logits),
        "real_bce": real_bce,
        "mismatch_bce": mismatch_bce,
        "mixup_bce": mixup_bce,
    }


def _calibration_diagnostics(
    model: MVRPFormer,
    dataset: IndexedPairDataset,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[float, float]:
    predictions: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for inputs, batch_labels, _ in DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        ):
            predictions.append(model(inputs.to(device)).global_logits.argmax(dim=1).cpu().numpy())
            labels.append(batch_labels.numpy())
    predicted = np.concatenate(predictions)
    true = np.concatenate(labels)
    accuracy = float(np.mean(predicted == true))
    f1_values = []
    for label in range(model.known_class_count):
        tp = int(np.count_nonzero((true == label) & (predicted == label)))
        fp = int(np.count_nonzero((true != label) & (predicted == label)))
        fn = int(np.count_nonzero((true == label) & (predicted != label)))
        denominator = 2 * tp + fp + fn
        f1_values.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return accuracy, float(np.mean(f1_values))


def _head_diagnostics(model: MVRPFormer) -> dict[str, float]:
    values: dict[str, float] = {}
    for prefix, head in (("global", model.global_head), ("view", model.view_head)):
        if isinstance(head, ARPLReciprocalHead):
            values[f"{prefix}_radius"] = float(head.radius.detach().cpu().item())
            norms = head.reciprocal_points.detach().norm(dim=2).cpu().numpy()
            values[f"{prefix}_reciprocal_norm_mean"] = float(norms.mean())
            values[f"{prefix}_reciprocal_norm_min"] = float(norms.min())
            values[f"{prefix}_reciprocal_norm_max"] = float(norms.max())
    return values


def train_one_method(
    model: MVRPFormer,
    *,
    method: str,
    prepared: PreparedSurrogateSplit,
    seed: int,
    config: Mapping[str, Any],
    mode: str,
    device: torch.device,
    resume_checkpoint: Path | None = None,
    _interrupt_after_epoch: int | None = None,
) -> dict[str, Any]:
    training = config["training"]
    _set_determinism(seed, bool(training["deterministic_algorithms"]))
    model = model.to(device)
    train_dataset = IndexedPairDataset(prepared.inputs["train"], prepared.labels["train"])
    calibration_dataset = IndexedPairDataset(
        prepared.inputs["known_calibration"], prepared.labels["known_calibration"]
    )
    dataloader_generator = torch.Generator().manual_seed(
        seed + int(training["dataloader_seed_offset"])
    )
    loader = DataLoader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        generator=dataloader_generator,
        num_workers=int(training["num_workers"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    pseudo_torch_generator = torch.Generator().manual_seed(
        seed + int(training["pseudo_seed_offset"])
    )
    pseudo_numpy_generator = np.random.default_rng(
        seed + int(training["pseudo_seed_offset"])
    )
    pseudo_pool = build_train_known_pseudo_pool(prepared)
    require_train_known_pseudo_source(pseudo_pool.source_role)
    pool_inputs = pseudo_pool.inputs
    pool_labels = pseudo_pool.labels
    pool_view1_frames = pseudo_pool.view1_frames
    pool_view2_frames = pseudo_pool.view2_frames
    train_pair_rows = [
        row for row in prepared.pair_manifest_rows if row["experiment_role"] == "train_known"
    ]
    pseudo_audit = PseudoAuditAccumulator(
        method=method, seed=seed, train_pair_rows=train_pair_rows
    )
    project_root = Path(config["_config_path"]).parents[3]
    source_hashes = task_source_hashes(project_root)
    epochs = int(training["smoke_epochs"] if mode == "smoke" else training["total_epochs"])
    log: list[dict[str, Any]] = []
    start_epoch = 1
    if resume_checkpoint is not None and resume_checkpoint.exists():
        state = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        expected = (
            method,
            seed,
            prepared.pair_manifest_sha256,
            config["_config_sha256"],
            mode,
            tuple(prepared.train_class_order),
            source_hashes,
        )
        observed = (
            state.get("method"),
            int(state.get("seed", -1)),
            state.get("pair_manifest_sha256"),
            state.get("config_sha256"),
            state.get("mode"),
            tuple(state.get("train_class_order", ())),
            state.get("source_hashes"),
        )
        if observed != expected:
            raise DataValidationError("resume checkpoint contract differs")
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        for optimizer_state in optimizer.state.values():
            for name, value in optimizer_state.items():
                if isinstance(value, torch.Tensor):
                    optimizer_state[name] = value.to(device)
        dataloader_generator.set_state(state["dataloader_generator_state"])
        pseudo_torch_generator.set_state(state["pseudo_torch_generator_state"])
        pseudo_numpy_generator.bit_generator.state = state["pseudo_numpy_generator_state"]
        pseudo_audit.load_state_dict(state["pseudo_audit_state"])
        torch.set_rng_state(state["torch_rng_state"])
        if device.type == "cuda" and state.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
        log = [dict(row) for row in state["training_log"]]
        start_epoch = int(state["completed_epoch"]) + 1
        if not 1 <= start_epoch <= epochs + 1:
            raise DataValidationError("resume epoch is outside the frozen schedule")
    for epoch in range(start_epoch, epochs + 1):
        started = time.perf_counter()
        learning_rate = learning_rate_for_epoch(
            epoch,
            base_learning_rate=float(training["learning_rate"]),
            warmup_epochs=int(training["warmup_epochs"]),
            total_epochs=int(training["total_epochs"]),
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        weights = loss_weights_for_epoch(
            epoch, int(training["representation_only_epochs"])
        )
        model.train()
        totals = {
            "count": 0,
            "total": 0.0,
            "representation": 0.0,
            "global_classification": 0.0,
            "global_margin": 0.0,
            "view_classification": 0.0,
            "view_margin": 0.0,
            "reject": 0.0,
            "uniform": 0.0,
            "correct": 0,
        }
        for inputs, labels, indices in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                inputs,
                compute_rejector=bool(weights["reject"] and method in REJECTOR_METHODS),
            )
            representation = model.representation_loss(
                output,
                labels,
                lambda_view=float(config["model"]["arpl"]["lambda_view"]),
            )
            reject_loss = inputs.new_zeros(())
            uniform_loss = inputs.new_zeros(())
            if weights["reject"] and method in REJECTOR_METHODS:
                pseudo_terms = _rejector_training_terms(
                    model,
                    output,
                    inputs,
                    labels,
                    indices,
                    pool_inputs=pool_inputs,
                    pool_labels=pool_labels,
                    pool_view1_frames=pool_view1_frames,
                    pool_view2_frames=pool_view2_frames,
                    torch_generator=pseudo_torch_generator,
                    numpy_generator=pseudo_numpy_generator,
                    composition=config["pseudo_unknown"]["composition"][method],
                    audit=pseudo_audit,
                    device=device,
                )
                reject_loss = pseudo_terms["reject"]
                uniform_loss = pseudo_terms["uniform"]
            total_loss = (
                representation["total"]
                + weights["reject"]
                * float(config["loss"]["lambda_reject"])
                * reject_loss
                + weights["uniform"] * uniform_loss
            )
            if not torch.isfinite(total_loss):
                raise NumericalInstabilityError(f"{method} loss became NaN or Inf")
            total_loss.backward()
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise NumericalInstabilityError(f"{method} gradient became NaN or Inf")
            optimizer.step()
            if not _is_finite_model(model):
                raise NumericalInstabilityError(f"{method} parameter became NaN or Inf")
            count = int(labels.numel())
            totals["count"] += count
            totals["total"] += float(total_loss.item()) * count
            totals["representation"] += float(representation["total"].item()) * count
            totals["global_classification"] += float(
                representation["global_classification"].item()
            ) * count
            totals["global_margin"] += float(representation["global_margin"].item()) * count
            totals["view_classification"] += float(
                representation["view_classification"].item()
            ) * count
            totals["view_margin"] += float(representation["view_margin"].item()) * count
            totals["reject"] += float(reject_loss.item()) * count
            totals["uniform"] += float(uniform_loss.item()) * count
            totals["correct"] += int(
                (output.global_logits.argmax(dim=1) == labels).sum().item()
            )
        calibration_accuracy, calibration_macro_f1 = _calibration_diagnostics(
            model,
            calibration_dataset,
            device=device,
            batch_size=int(training["batch_size"]),
        )
        count = int(totals["count"])
        row = {
            "epoch": epoch,
            "method": method,
            "learning_rate": learning_rate,
            "reject_loss_active": bool(weights["reject"] and method in REJECTOR_METHODS),
            "checkpoint_selected_for_open_set_performance": False,
            "train_accuracy": totals["correct"] / count,
            "known_calibration_accuracy_diagnostic": calibration_accuracy,
            "known_calibration_macro_f1_diagnostic": calibration_macro_f1,
            "elapsed_seconds": time.perf_counter() - started,
            **{
                f"train_{key}_loss": totals[key] / count
                for key in (
                    "total",
                    "representation",
                    "global_classification",
                    "global_margin",
                    "view_classification",
                    "view_margin",
                    "reject",
                    "uniform",
                )
            },
            **_head_diagnostics(model),
        }
        log.append(row)
        if resume_checkpoint is not None:
            _atomic_torch_save(
                resume_checkpoint,
                {
                    "method": method,
                    "seed": seed,
                    "mode": mode,
                    "pair_manifest_sha256": prepared.pair_manifest_sha256,
                    "config_sha256": config["_config_sha256"],
                    "train_class_order": prepared.train_class_order,
                    "source_hashes": source_hashes,
                    "completed_epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "dataloader_generator_state": dataloader_generator.get_state(),
                    "pseudo_torch_generator_state": pseudo_torch_generator.get_state(),
                    "pseudo_numpy_generator_state": pseudo_numpy_generator.bit_generator.state,
                    "pseudo_audit_state": pseudo_audit.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state_all": (
                        torch.cuda.get_rng_state_all() if device.type == "cuda" else None
                    ),
                    "training_log": log,
                },
            )
        if _interrupt_after_epoch == epoch:
            if resume_checkpoint is None:
                raise ValueError("fault injection requires a resume checkpoint path")
            raise IntentionalTrainingInterruption(
                f"intentional interruption after checkpointed epoch {epoch}"
            )
    audit = pseudo_audit.to_json()
    if method in REJECTOR_METHODS:
        if audit["status"] != "passed" or not audit["real_pseudo_balanced"]:
            raise DataValidationError("pseudo-unknown generation audit failed")
    else:
        audit["status"] = "not_applicable"
    final_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    model.eval()
    return {
        "model": model,
        "final_state": final_state,
        "checkpoint_epoch": epochs,
        "formal_checkpoint": mode == "full" and epochs == int(training["formal_checkpoint_epoch"]),
        "training_log": log,
        "pseudo_audit": audit,
        "source_hashes": source_hashes,
        "final_known_calibration_accuracy": log[-1][
            "known_calibration_accuracy_diagnostic"
        ],
        "final_known_calibration_macro_f1": log[-1][
            "known_calibration_macro_f1_diagnostic"
        ],
    }


def infer_model(
    model: MVRPFormer,
    inputs: np.ndarray,
    labels: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    dataset = IndexedPairDataset(inputs, labels)
    names = (
        "raw_view_tokens",
        "contextual_view_tokens",
        "global_class_token",
        "global_reject_token",
        "per_view_logits",
        "global_logits",
        "sab_attention",
        "pma_attention",
        "reject_evidence",
        "unknown_score",
        "labels",
    )
    collected: dict[str, list[np.ndarray]] = {name: [] for name in names}
    model.eval()
    with torch.no_grad():
        for batch_inputs, batch_labels, _ in DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        ):
            output = model(batch_inputs.to(device))
            unknown_score = (
                -output.global_logits.max(dim=1).values
                if output.unknown_probability is None
                else output.unknown_probability
            )
            reject_evidence = (
                output.global_logits.new_empty((output.global_logits.shape[0], 0))
                if output.reject_evidence is None
                else output.reject_evidence
            )
            tensors = {
                "raw_view_tokens": output.raw_view_tokens,
                "contextual_view_tokens": output.contextual_view_tokens,
                "global_class_token": output.global_class_token,
                "global_reject_token": output.global_reject_token,
                "per_view_logits": output.per_view_logits,
                "global_logits": output.global_logits,
                "sab_attention": output.sab_attention,
                "pma_attention": output.pma_attention,
                "reject_evidence": reject_evidence,
                "unknown_score": unknown_score,
            }
            for name, tensor in tensors.items():
                dtype = np.float64 if name in {"global_logits", "per_view_logits", "unknown_score"} else np.float32
                collected[name].append(tensor.detach().cpu().numpy().astype(dtype))
            collected["labels"].append(batch_labels.numpy().astype(np.int64))
    result = {name: np.concatenate(values, axis=0) for name, values in collected.items()}
    if not all(np.isfinite(value).all() for value in result.values()):
        raise NumericalInstabilityError("inference produced NaN or Inf")
    return result


def permutation_audit(
    model: MVRPFormer,
    inputs: np.ndarray,
    *,
    device: torch.device,
) -> dict[str, Any]:
    tensor = torch.from_numpy(np.asarray(inputs[: min(16, len(inputs))], dtype=np.float32)).to(device)
    model.eval()
    with torch.no_grad():
        original = model(tensor)
        swapped = model(tensor[:, [1, 0]])
    comparisons = {
        "raw_view_tokens_equivariant": (
            original.raw_view_tokens[:, [1, 0]],
            swapped.raw_view_tokens,
        ),
        "contextual_view_tokens_equivariant": (
            original.contextual_view_tokens[:, [1, 0]],
            swapped.contextual_view_tokens,
        ),
        "global_class_token_invariant": (
            original.global_class_token,
            swapped.global_class_token,
        ),
        "global_reject_token_invariant": (
            original.global_reject_token,
            swapped.global_reject_token,
        ),
        "per_view_logits_equivariant": (
            original.per_view_logits[:, [1, 0]],
            swapped.per_view_logits,
        ),
        "global_logits_invariant": (original.global_logits, swapped.global_logits),
    }
    if original.unknown_probability is not None:
        comparisons["unknown_probability_invariant"] = (
            original.unknown_probability,
            swapped.unknown_probability,
        )
        comparisons["reject_evidence_invariant"] = (
            original.reject_evidence,
            swapped.reject_evidence,
        )
    maximum_errors = {
        name: float((left - right).abs().max().cpu().item())
        for name, (left, right) in comparisons.items()
    }
    passed = all(
        torch.allclose(left, right, rtol=1e-5, atol=1e-6)
        for left, right in comparisons.values()
    )
    if not passed:
        raise DataValidationError("model failed the two-view permutation audit")
    return {
        "status": "passed",
        "eval_mode": True,
        "rtol": 1e-5,
        "atol": 1e-6,
        "maximum_absolute_errors": maximum_errors,
    }


def evaluate_inference_arrays(
    arrays: Mapping[str, Mapping[str, np.ndarray]],
    *,
    prepared: PreparedSurrogateSplit,
    config: Mapping[str, Any],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    known = arrays["known_calibration"]
    unknown = arrays["surrogate_unknown"]
    known_pred = known["global_logits"].argmax(axis=1)
    unknown_pred = unknown["global_logits"].argmax(axis=1)
    metrics = evaluate_open_set(
        known_true=known["labels"],
        known_pred=known_pred,
        known_unknown_scores=known["unknown_score"],
        unknown_pred=unknown_pred,
        unknown_unknown_scores=unknown["unknown_score"],
        known_validation_scores=known["unknown_score"],
        known_class_count=len(prepared.train_class_order),
        known_acceptance_rate=float(config["evaluation"]["threshold_known_acceptance_rate"]),
    )
    rows: list[dict[str, Any]] = []
    for role, values, predictions in (
        ("known_calibration", known, known_pred),
        ("surrogate_unknown", unknown, unknown_pred),
    ):
        for index, (pair_id, class_name) in enumerate(
            zip(prepared.pair_ids[role], prepared.class_names[role], strict=True)
        ):
            rows.append(
                {
                    "pair_id": pair_id,
                    "evaluation_role": role,
                    "class_name": class_name,
                    "true_label": int(values["labels"][index]),
                    "predicted_known_label": int(predictions[index]),
                    "unknown_score": float(values["unknown_score"][index]),
                    "score_source": (
                        "rejector_probability_unknown"
                        if values["reject_evidence"].shape[1]
                        else "negative_maximum_global_raw_logit"
                    ),
                    "global_logits": json.dumps(
                        values["global_logits"][index].tolist(), separators=(",", ":")
                    ),
                    "view1_logits": json.dumps(
                        values["per_view_logits"][index, 0].tolist(), separators=(",", ":")
                    ),
                    "view2_logits": json.dumps(
                        values["per_view_logits"][index, 1].tolist(), separators=(",", ":")
                    ),
                    "threshold": float(metrics["threshold"]),
                    "rejected": bool(values["unknown_score"][index] > metrics["threshold"]),
                }
            )
    return metrics, rows


def recompute_unit_metrics_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    known_class_count: int = 5,
    known_acceptance_rate: float = 0.95,
) -> dict[str, float]:
    known = [row for row in rows if row["evaluation_role"] == "known_calibration"]
    unknown = [row for row in rows if row["evaluation_role"] == "surrogate_unknown"]
    if not known or not unknown:
        raise DataValidationError("prediction rows are missing an evaluation role")
    return evaluate_open_set(
        known_true=np.asarray([int(row["true_label"]) for row in known]),
        known_pred=np.asarray([int(row["predicted_known_label"]) for row in known]),
        known_unknown_scores=np.asarray([float(row["unknown_score"]) for row in known]),
        unknown_pred=np.asarray([int(row["predicted_known_label"]) for row in unknown]),
        unknown_unknown_scores=np.asarray([float(row["unknown_score"]) for row in unknown]),
        known_validation_scores=np.asarray([float(row["unknown_score"]) for row in known]),
        known_class_count=known_class_count,
        known_acceptance_rate=known_acceptance_rate,
    )


def _head_arrays(model: MVRPFormer) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for prefix, head in (("global", model.global_head), ("view", model.view_head)):
        if isinstance(head, ARPLReciprocalHead):
            values[f"{prefix}_reciprocal_points"] = (
                head.reciprocal_points.detach().cpu().numpy().astype(np.float32)
            )
            values[f"{prefix}_radius"] = head.radius.detach().cpu().numpy().astype(np.float32)
        elif isinstance(head, nn.Linear):
            values[f"{prefix}_ce_weight"] = head.weight.detach().cpu().numpy().astype(np.float32)
            values[f"{prefix}_ce_bias"] = head.bias.detach().cpu().numpy().astype(np.float32)
    return values


def save_method_result(
    destination: Path,
    *,
    method: str,
    seed: int,
    prepared: PreparedSurrogateSplit,
    trained: Mapping[str, Any],
    arrays: Mapping[str, Mapping[str, np.ndarray]],
    metrics: Mapping[str, float],
    prediction_rows: Sequence[Mapping[str, Any]],
    permutation: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=False)
    _write_csv(destination / "predictions.csv", prediction_rows)
    _write_json(destination / "metrics.json", metrics)
    _write_json(destination / "permutation_audit.json", permutation)
    _write_json(destination / "pseudo_pair_audit.json", trained["pseudo_audit"])
    (destination / "training_log.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in trained["training_log"]
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        destination / "tokens_logits_scores.npz",
        **{
            f"{role}_{name}": value
            for role, role_arrays in arrays.items()
            for name, value in role_arrays.items()
        },
    )
    np.savez_compressed(destination / "head_parameters.npz", **_head_arrays(trained["model"]))
    checkpoint = {
        "experiment_id": EXPERIMENT_ID,
        "method": method,
        "model_state_dict": trained["final_state"],
        "checkpoint_epoch": trained["checkpoint_epoch"],
        "formal_checkpoint": trained["formal_checkpoint"],
        "checkpoint_selection": "fixed_final_epoch",
        "train_class_order": prepared.train_class_order,
        "surrogate_class_order": prepared.surrogate_class_order,
        "normalization": asdict(prepared.normalization),
        "pair_manifest_sha256": prepared.pair_manifest_sha256,
        "initialization_seed": seed,
        "config_sha256": config["_config_sha256"],
    }
    torch.save(checkpoint, destination / "checkpoint.pt")
    resolved = dict(config)
    resolved["_resolved"] = {
        "method": method,
        "seed": seed,
        "split_id": prepared.split_id,
        "pair_manifest_sha256": prepared.pair_manifest_sha256,
        "train_class_order": list(prepared.train_class_order),
        "surrogate_class_order": list(prepared.surrogate_class_order),
        "checkpoint_epoch": trained["checkpoint_epoch"],
        "checkpoint_selection": "fixed_final_epoch",
    }
    (destination / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    _write_json(destination / "artifact_hashes.json", _artifact_hashes(destination))
    return {
        "method": method,
        "seed": seed,
        "metrics": dict(metrics),
        "checkpoint_epoch": trained["checkpoint_epoch"],
        "formal_checkpoint": trained["formal_checkpoint"],
        "checkpoint_sha256": file_sha256(destination / "checkpoint.pt"),
        "prediction_sha256": file_sha256(destination / "predictions.csv"),
    }


def _configure_torch_runtime(config: Mapping[str, Any]) -> dict[str, int]:
    requested_intra = int(config["runtime"]["torch_intraop_threads"])
    requested_inter = int(config["runtime"]["torch_interop_threads"])
    torch.set_num_threads(requested_intra)
    try:
        torch.set_num_interop_threads(requested_inter)
    except RuntimeError:
        if torch.get_num_interop_threads() != requested_inter:
            raise
    return {
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
    }


def _spec_for_single_unit(
    config: Mapping[str, Any], *, phase: str, split_id: str, seed: int, method: str
) -> tuple[Mapping[str, Any], str]:
    matches = [
        unit
        for unit in build_phase_plan(config, phase)
        if str(unit["spec"]["split_id"]) == split_id
        and int(unit["seed"]) == seed
        and method in unit["methods"]
    ]
    if len(matches) != 1:
        raise DataConfigError("requested split/seed/method is outside the frozen phase plan")
    return matches[0]["spec"], str(matches[0]["mode"])


def _quarantine_path(path: Path, *, phase_root: Path, reason: str) -> Path:
    """Move an incomplete/redundant work item outside the auditable phase tree."""

    quarantine = (
        phase_root.parent
        / "_quarantine"
        / phase_root.name
        / f"{reason}_{time.time_ns()}_{path.name}"
    )
    quarantine.parent.mkdir(parents=True, exist_ok=True)
    path.replace(quarantine)
    return quarantine


def run_single_method(
    config_path: str | Path,
    bundle_root: str | Path,
    phase_root: str | Path,
    *,
    phase: str,
    split_id: str,
    seed: int,
    method: str,
    device_request: str = "auto",
    development_root: str | Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    config = load_mv_rpformer_config(config_path)
    spec, mode = _spec_for_single_unit(
        config, phase=phase, split_id=split_id, seed=seed, method=method
    )
    if phase == "confirmation":
        if development_root is None:
            raise DataConfigError("confirmation unit requires development root")
        verify_development_authorization(config_path, development_root)
    root = Path(phase_root).resolve()
    destination = root / split_id / f"seed_{seed}" / method
    if destination.exists():
        if resume and (destination / "_SUCCESS.json").is_file():
            audited = audit_method_result(
                destination,
                config=config,
                phase=phase,
                split_id=split_id,
                seed=seed,
                method=method,
                require_formal=mode == "full",
            )
            work_root = destination.parent / f".{method}.work"
            if work_root.exists():
                _quarantine_path(
                    work_root,
                    phase_root=root,
                    reason="redundant_completed_work",
                )
            return {"status": "already_complete", **audited}
        raise DataValidationError(f"method output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    work_root = destination.parent / f".{method}.work"
    if work_root.exists() and not resume:
        raise DataValidationError(f"resume work exists; pass --resume: {work_root}")
    work_root.mkdir(exist_ok=True)
    resume_checkpoint = work_root / "latest_checkpoint.pt"
    runtime = _configure_torch_runtime(config)
    device = _resolve_device(device_request)
    bundle_config = config["bundle"]
    bundle = load_processed_bundle(
        bundle_root,
        expected_profiles_sha256=bundle_config["profiles_sha256"],
        expected_manifest_sha256=bundle_config["manifest_sha256"],
        expected_bundle_sha256=bundle_config["bundle_sha256"],
    )
    prepared = _prepare_split(bundle, config, spec, mode=mode)
    models, initialization = build_initialized_method_group(
        len(prepared.train_class_order), seed=seed, config=config
    )
    started = time.perf_counter()
    trained = train_one_method(
        models[method],
        method=method,
        prepared=prepared,
        seed=seed,
        config=config,
        mode=mode,
        device=device,
        resume_checkpoint=resume_checkpoint,
    )
    arrays = {
        role: infer_model(
            trained["model"],
            prepared.inputs[role],
            prepared.labels[role],
            device=device,
            batch_size=int(config["training"]["batch_size"]),
        )
        for role in ("train", "known_calibration", "surrogate_unknown")
    }
    metrics, prediction_rows = evaluate_inference_arrays(
        arrays, prepared=prepared, config=config
    )
    recomputed = recompute_unit_metrics_from_rows(
        prediction_rows,
        known_class_count=len(prepared.train_class_order),
        known_acceptance_rate=float(config["evaluation"]["threshold_known_acceptance_rate"]),
    )
    if recomputed != metrics:
        raise DataValidationError("prediction rows do not exactly reproduce method metrics")
    permutation = permutation_audit(
        trained["model"], prepared.inputs["known_calibration"], device=device
    )
    project_root = Path(config["_config_path"]).parents[3]
    source_hashes = task_source_hashes(project_root)
    if source_hashes != trained["source_hashes"]:
        raise DataValidationError("task source changed while the method was training")
    staging = destination.parent / f".{method}.staging"
    if staging.exists():
        if not resume:
            raise DataValidationError(f"stale staging output exists: {staging}")
        _quarantine_path(
            staging, phase_root=root, reason="interrupted_method_staging"
        )
    saved = save_method_result(
        staging,
        method=method,
        seed=seed,
        prepared=prepared,
        trained=trained,
        arrays=arrays,
        metrics=metrics,
        prediction_rows=prediction_rows,
        permutation=permutation,
        config=config,
    )
    (staging / "pair_manifest.csv").write_bytes(prepared.pair_manifest_bytes)
    _write_json(staging / "pair_audit.json", prepared.pair_audit)
    _write_json(staging / "normalization.json", asdict(prepared.normalization))
    shortcut_metrics, shortcut_rows = _length_padding_audit(
        bundle, prepared.pair_manifest_rows
    )
    _write_json(staging / "length_padding_shortcut_diagnostic.json", {
        "diagnostic_only": True,
        "used_for_gate": False,
        "metrics": shortcut_metrics,
    })
    _write_csv(staging / "length_padding_shortcut_rows.csv", shortcut_rows)
    environment = _environment(project_root, device)
    environment["torch_runtime"] = runtime
    environment["task_source_hashes"] = source_hashes
    _write_json(staging / "environment.json", environment)
    contract = {
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "mode": mode,
        "split_id": split_id,
        "seed": seed,
        "method": method,
        "config_sha256": config["_config_sha256"],
        "pair_manifest_sha256": prepared.pair_manifest_sha256,
        "train_class_order": list(prepared.train_class_order),
        "surrogate_class_order": list(prepared.surrogate_class_order),
        "task_source_hashes": source_hashes,
        "runtime": runtime,
        "initialization_audit": initialization,
        "final_unknown_used": False,
        "even_angle_test_used": False,
    }
    _write_json(staging / "unit_contract.json", contract)
    summary = {
        "status": "complete",
        "phase": phase,
        "split_id": split_id,
        "seed": seed,
        "method": method,
        "metrics": metrics,
        "checkpoint_epoch": trained["checkpoint_epoch"],
        "formal_checkpoint": trained["formal_checkpoint"],
        "pair_manifest_sha256": prepared.pair_manifest_sha256,
        "wall_time_seconds": time.perf_counter() - started,
        **saved,
    }
    _write_json(staging / "method_summary.json", summary)
    _write_json(staging / "_SUCCESS.json", {
        "status": "complete",
        "method_summary_sha256": file_sha256(staging / "method_summary.json"),
    })
    _write_json(staging / "artifact_hashes.json", _artifact_hashes(staging))
    staging.replace(destination)
    if resume_checkpoint.exists():
        resume_checkpoint.unlink()
    if work_root.exists() and not any(work_root.iterdir()):
        work_root.rmdir()
    return summary


def audit_method_result(
    destination: Path,
    *,
    config: Mapping[str, Any],
    phase: str,
    split_id: str,
    seed: int,
    method: str,
    require_formal: bool,
) -> dict[str, Any]:
    required = {
        "_SUCCESS.json",
        "artifact_hashes.json",
        "checkpoint.pt",
        "environment.json",
        "head_parameters.npz",
        "length_padding_shortcut_diagnostic.json",
        "length_padding_shortcut_rows.csv",
        "method_summary.json",
        "metrics.json",
        "normalization.json",
        "pair_audit.json",
        "pair_manifest.csv",
        "permutation_audit.json",
        "predictions.csv",
        "pseudo_pair_audit.json",
        "resolved_config.yaml",
        "tokens_logits_scores.npz",
        "training_log.jsonl",
        "unit_contract.json",
    }
    missing = sorted(name for name in required if not (destination / name).is_file())
    if missing:
        raise DataValidationError(f"method artifact is incomplete: {missing}")
    stored_hashes = json.loads((destination / "artifact_hashes.json").read_text())
    actual_hashes = _artifact_hashes(destination)
    if stored_hashes != actual_hashes:
        raise DataValidationError("method artifact hashes do not match")
    success = json.loads((destination / "_SUCCESS.json").read_text())
    if (
        success.get("status") != "complete"
        or success.get("method_summary_sha256")
        != file_sha256(destination / "method_summary.json")
    ):
        raise DataValidationError("method success seal is invalid")
    contract = json.loads((destination / "unit_contract.json").read_text())
    expected_identity = (phase, split_id, seed, method, config["_config_sha256"])
    observed_identity = (
        contract.get("phase"),
        contract.get("split_id"),
        int(contract.get("seed", -1)),
        contract.get("method"),
        contract.get("config_sha256"),
    )
    if observed_identity != expected_identity:
        raise DataValidationError("method unit contract identity differs")
    project_root = Path(config["_config_path"]).parents[3]
    if contract.get("task_source_hashes") != task_source_hashes(project_root):
        raise DataValidationError("method source hashes differ from current frozen code")
    pair_sha = hashlib.sha256((destination / "pair_manifest.csv").read_bytes()).hexdigest()
    if pair_sha != contract.get("pair_manifest_sha256"):
        raise DataValidationError("method pair manifest hash differs")
    pair_audit = json.loads((destination / "pair_audit.json").read_text())
    if (
        pair_audit.get("status") != "passed"
        or pair_audit.get("train_evaluation_base_overlap") != 0
        or pair_audit.get("final_unknown_pairs") != 0
        or pair_audit.get("even_angle_pairs") != 0
        or pair_audit.get("test_pairs_generated") is not False
    ):
        raise DataValidationError("method pair isolation audit failed")
    checkpoint = torch.load(destination / "checkpoint.pt", map_location="cpu", weights_only=False)
    expected_epoch = int(config["training"]["total_epochs"] if require_formal else config["training"]["smoke_epochs"])
    if (
        checkpoint.get("method") != method
        or checkpoint.get("pair_manifest_sha256") != pair_sha
        or checkpoint.get("config_sha256") != config["_config_sha256"]
        or int(checkpoint.get("checkpoint_epoch", -1)) != expected_epoch
        or bool(checkpoint.get("formal_checkpoint")) != require_formal
        or checkpoint.get("checkpoint_selection") != "fixed_final_epoch"
    ):
        raise DataValidationError("method checkpoint contract failed")
    permutation = json.loads((destination / "permutation_audit.json").read_text())
    if permutation.get("status") != "passed":
        raise DataValidationError("method permutation audit failed")
    pseudo = json.loads((destination / "pseudo_pair_audit.json").read_text())
    if method in REJECTOR_METHODS:
        if (
            pseudo.get("status") != "passed"
            or pseudo.get("real_pseudo_balanced") is not True
            or pseudo.get("same_class_violations") != 0
            or pseudo.get("mismatch_same_frame_violations") != 0
            or pseudo.get("source_role") != "train_known_only"
            or pseudo.get("surrogate_unknown_used") is not False
            or pseudo.get("final_unknown_used") is not False
            or pseudo.get("even_angle_test_used") is not False
        ):
            raise DataValidationError("method pseudo-unknown audit failed")
    elif pseudo.get("status") != "not_applicable":
        raise DataValidationError("non-rejector method has unexpected pseudo audit")
    with (destination / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        prediction_rows = list(csv.DictReader(handle))
    metrics = json.loads((destination / "metrics.json").read_text())
    recomputed = recompute_unit_metrics_from_rows(
        prediction_rows,
        known_class_count=5,
        known_acceptance_rate=float(config["evaluation"]["threshold_known_acceptance_rate"]),
    )
    if recomputed != metrics:
        raise DataValidationError("method prediction metrics do not exactly recompute")
    with np.load(destination / "tokens_logits_scores.npz") as arrays:
        if not arrays.files or any(not np.isfinite(arrays[name]).all() for name in arrays.files):
            raise DataValidationError("method saved arrays are missing or non-finite")
        for role in ("train", "known_calibration", "surrogate_unknown"):
            if f"{role}_global_logits" not in arrays or f"{role}_unknown_score" not in arrays:
                raise DataValidationError("method saved arrays lack a required role")
    return {
        "status": "passed",
        "phase": phase,
        "split_id": split_id,
        "seed": seed,
        "method": method,
        "metrics": metrics,
        "pair_manifest_sha256": pair_sha,
        "prediction_pair_order": tuple(row["pair_id"] for row in prediction_rows),
        "prediction_sha256": file_sha256(destination / "predictions.csv"),
        "pseudo_schedule_sha256": pseudo.get("schedule_sha256"),
        "initialization_audit": contract["initialization_audit"],
        "source_hashes": contract["task_source_hashes"],
    }


def _phase_artifact_hashes(root: Path) -> dict[str, str]:
    excluded = {
        "artifact_hashes.json",
        "confirmation_authorization.json",
        "_PHASE_SUCCESS.json",
    }
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in excluded
    }


def _collect_phase_audit(
    config: Mapping[str, Any], root: Path, *, phase: str
) -> dict[str, Any]:
    plan = build_phase_plan(config, phase)
    require_formal = phase != "smoke"
    audited: list[dict[str, Any]] = []
    for unit in plan:
        split_id = str(unit["spec"]["split_id"])
        seed = int(unit["seed"])
        for method in unit["methods"]:
            destination = root / split_id / f"seed_{seed}" / method
            audited.append(
                audit_method_result(
                    destination,
                    config=config,
                    phase=phase,
                    split_id=split_id,
                    seed=seed,
                    method=method,
                    require_formal=require_formal,
                )
            )
    expected_units = {
        (str(unit["spec"]["split_id"]), int(unit["seed"])) for unit in plan
    }
    observed_units = {(row["split_id"], row["seed"]) for row in audited}
    if observed_units != expected_units or len(audited) != len(plan) * len(METHODS):
        raise DataValidationError("phase artifact matrix is not the frozen Cartesian product")
    fairness_rows = []
    for split_id, seed in sorted(expected_units):
        rows = [
            row for row in audited if row["split_id"] == split_id and row["seed"] == seed
        ]
        if len({row["pair_manifest_sha256"] for row in rows}) != 1:
            raise DataValidationError("methods used different real pair manifests")
        if len({row["prediction_pair_order"] for row in rows}) != 1:
            raise DataValidationError("methods used different prediction pair order")
        serialized_initialization = {
            json.dumps(row["initialization_audit"], sort_keys=True) for row in rows
        }
        if len(serialized_initialization) != 1:
            raise DataValidationError("method initialization audit contracts differ")
        if len({json.dumps(row["source_hashes"], sort_keys=True) for row in rows}) != 1:
            raise DataValidationError("method source hash contracts differ")
        pseudo = {row["method"]: row["pseudo_schedule_sha256"] for row in rows}
        if pseudo["M6_MV_RPFORMER_FULL"] != pseudo["M7_MV_CEFORMER_FULL"]:
            raise DataValidationError("M6 and M7 pseudo schedules differ")
        fairness_rows.append(
            {
                "split_id": split_id,
                "seed": seed,
                "pair_manifest_sha256": rows[0]["pair_manifest_sha256"],
                "same_real_pair_manifest_and_prediction_order": True,
                "same_initialization_audit": True,
                "m6_m7_same_pseudo_schedule": True,
            }
        )
    metric_rows = [
        {
            "split_id": row["split_id"],
            "seed": row["seed"],
            "method": row["method"],
            **{key: row["metrics"][key] for key in METRIC_KEYS},
        }
        for row in audited
    ]
    return {
        "status": "passed",
        "phase": phase,
        "unit_count": len(expected_units),
        "method_result_count": len(audited),
        "expected_unit_ids": [f"{split_id}/seed_{seed}" for split_id, seed in sorted(expected_units)],
        "metric_rows": metric_rows,
        "fairness_rows": fairness_rows,
        "source_hashes": audited[0]["source_hashes"],
        "final_unknown_used": False,
        "even_angle_test_used": False,
        "predictions_exactly_recomputed": True,
        "all_method_hashes_verified": True,
        "all_permutation_audits_passed": True,
        "all_pseudo_audits_passed": True,
    }


def aggregate_phase_root(
    config_path: str | Path,
    phase_root: str | Path,
    *,
    phase: str,
    resume: bool = False,
) -> dict[str, Any]:
    config = load_mv_rpformer_config(config_path)
    root = Path(phase_root).resolve()
    aggregate_names = (
        "metrics_by_unit.csv",
        "paired_deltas.csv",
        "comparison_summary.json",
        "summary.json",
        "artifact_hashes.json",
        "environment.json",
        "fairness_by_unit.json",
        "smoke_integrity.json",
        "development_integrity.json",
        "confirmation_integrity.json",
        "confirmation_decision.json",
        "confirmation_authorization.json",
        "_PHASE_SUCCESS.json",
    )
    existing = [root / name for name in aggregate_names if (root / name).exists()]
    if (root / "_PHASE_SUCCESS.json").is_file():
        if not resume:
            raise DataValidationError("phase aggregate is already complete")
        audit_phase_root(config_path, root, phase=phase, verify_root_hashes=True)
        return json.loads((root / "summary.json").read_text())
    if existing and not resume:
        raise DataValidationError(f"incomplete aggregate output exists: {existing[0]}")
    if existing:
        quarantine = (
            root.parent
            / "_quarantine"
            / root.name
            / f"interrupted_aggregate_{time.time_ns()}"
        )
        quarantine.mkdir(parents=True, exist_ok=False)
        for path in existing:
            path.replace(quarantine / path.name)
    if any(path.name.endswith(".work") or ".staging" in path.name for path in root.rglob(".*")):
        raise DataValidationError("phase root contains incomplete work or staging directories")
    audited = _collect_phase_audit(config, root, phase=phase)
    metric_rows = audited.pop("metric_rows")
    fairness_rows = audited.pop("fairness_rows")
    comparison = summarize_paired_comparisons(
        metric_rows,
        config=config,
        require_confirmation_units=phase == "confirmation",
    )
    paired_rows = comparison.pop("paired_delta_rows")
    _write_csv(root / "metrics_by_unit.csv", metric_rows)
    _write_csv(root / "paired_deltas.csv", paired_rows)
    _write_json(root / "comparison_summary.json", comparison)
    _write_json(root / "fairness_by_unit.json", fairness_rows)
    integrity_name = {
        "smoke": "smoke_integrity.json",
        "development": "development_integrity.json",
        "confirmation": "confirmation_integrity.json",
    }[phase]
    integrity = {
        **audited,
        "config_sha256": config["_config_sha256"],
        "development_performance_gate_used": False,
    }
    _write_json(root / integrity_name, integrity)
    if phase == "confirmation":
        _write_json(root / "confirmation_decision.json", comparison["decision"])
    project_root = Path(config["_config_path"]).parents[3]
    environment = _environment(project_root, torch.device("cpu"))
    environment["task_source_hashes"] = task_source_hashes(project_root)
    _write_json(root / "environment.json", environment)
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "integrity": integrity,
        "comparison_summary": comparison,
        "final_unknown_used": False,
        "even_angle_test_used": False,
    }
    _write_json(root / "summary.json", summary)
    _write_json(root / "artifact_hashes.json", _phase_artifact_hashes(root))
    if phase == "development":
        _write_json(root / "confirmation_authorization.json", {
            "status": "authorized_by_integrity_only",
            "config_sha256": config["_config_sha256"],
            "artifact_hash_manifest_sha256": file_sha256(root / "artifact_hashes.json"),
            "development_integrity_sha256": file_sha256(root / "development_integrity.json"),
            "task_source_hashes": task_source_hashes(project_root),
            "performance_gate_used": False,
        })
    integrity_path = root / integrity_name
    success = {
        "status": "complete",
        "phase": phase,
        "config_sha256": config["_config_sha256"],
        "artifact_hash_manifest_sha256": file_sha256(root / "artifact_hashes.json"),
        "summary_sha256": file_sha256(root / "summary.json"),
        "integrity_sha256": file_sha256(integrity_path),
        "confirmation_authorization_sha256": (
            file_sha256(root / "confirmation_authorization.json")
            if phase == "development"
            else None
        ),
    }
    _write_json(root / "_PHASE_SUCCESS.json", success)
    return summary


def verify_development_authorization(
    config_path: str | Path, development_root: str | Path
) -> dict[str, Any]:
    config = load_mv_rpformer_config(config_path)
    root = Path(development_root).resolve()
    authorization = json.loads((root / "confirmation_authorization.json").read_text())
    stored_hashes = json.loads((root / "artifact_hashes.json").read_text())
    current_hashes = _phase_artifact_hashes(root)
    integrity = json.loads((root / "development_integrity.json").read_text())
    success = json.loads((root / "_PHASE_SUCCESS.json").read_text())
    project_root = Path(config["_config_path"]).parents[3]
    if (
        authorization.get("status") != "authorized_by_integrity_only"
        or authorization.get("config_sha256") != config["_config_sha256"]
        or authorization.get("artifact_hash_manifest_sha256")
        != file_sha256(root / "artifact_hashes.json")
        or authorization.get("development_integrity_sha256")
        != file_sha256(root / "development_integrity.json")
        or authorization.get("task_source_hashes") != task_source_hashes(project_root)
        or authorization.get("performance_gate_used") is not False
        or stored_hashes != current_hashes
        or integrity.get("status") != "passed"
        or integrity.get("config_sha256") != config["_config_sha256"]
        or success.get("status") != "complete"
        or success.get("phase") != "development"
        or success.get("artifact_hash_manifest_sha256")
        != file_sha256(root / "artifact_hashes.json")
        or success.get("confirmation_authorization_sha256")
        != file_sha256(root / "confirmation_authorization.json")
    ):
        raise DataValidationError("development confirmation authorization is invalid")
    return authorization


def audit_phase_root(
    config_path: str | Path,
    phase_root: str | Path,
    *,
    phase: str,
    verify_root_hashes: bool = True,
) -> dict[str, Any]:
    config = load_mv_rpformer_config(config_path)
    root = Path(phase_root).resolve()
    audited = _collect_phase_audit(config, root, phase=phase)
    integrity_name = {
        "smoke": "smoke_integrity.json",
        "development": "development_integrity.json",
        "confirmation": "confirmation_integrity.json",
    }[phase]
    stored_integrity = json.loads((root / integrity_name).read_text())
    metric_rows = audited.pop("metric_rows")
    fairness_rows = audited.pop("fairness_rows")
    expected_integrity = {
        **audited,
        "config_sha256": config["_config_sha256"],
        "development_performance_gate_used": False,
    }
    if stored_integrity != expected_integrity:
        raise DataValidationError("stored phase integrity does not match full re-audit")
    stored_metrics = _load_metric_csv(root / "metrics_by_unit.csv")
    if stored_metrics != metric_rows:
        raise DataValidationError("stored phase metric matrix differs from method artifacts")
    stored_fairness = json.loads((root / "fairness_by_unit.json").read_text())
    if stored_fairness != fairness_rows:
        raise DataValidationError("stored phase fairness rows differ")
    if verify_root_hashes:
        stored_hashes = json.loads((root / "artifact_hashes.json").read_text())
        if stored_hashes != _phase_artifact_hashes(root):
            raise DataValidationError("phase root artifact hashes do not match")
    success = json.loads((root / "_PHASE_SUCCESS.json").read_text())
    expected_success = {
        "status": "complete",
        "phase": phase,
        "config_sha256": config["_config_sha256"],
        "artifact_hash_manifest_sha256": file_sha256(root / "artifact_hashes.json"),
        "summary_sha256": file_sha256(root / "summary.json"),
        "integrity_sha256": file_sha256(root / integrity_name),
        "confirmation_authorization_sha256": (
            file_sha256(root / "confirmation_authorization.json")
            if phase == "development"
            else None
        ),
    }
    if success != expected_success:
        raise DataValidationError("phase completion seal is invalid")
    return expected_integrity


def _metric_rows(result_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "split_id": row["split_id"],
            "seed": row["seed"],
            "method": row["method"],
            **{key: row["metrics"][key] for key in METRIC_KEYS},
        }
        for row in result_rows
    ]


def _find_metric(
    rows: Sequence[Mapping[str, Any]], *, split_id: str, seed: int, method: str
) -> Mapping[str, Any]:
    matches = [
        row
        for row in rows
        if str(row["split_id"]) == split_id
        and int(row["seed"]) == seed
        and str(row["method"]) == method
    ]
    if len(matches) != 1:
        raise DataValidationError("metric unit lookup is not unique")
    return matches[0]


def summarize_paired_comparisons(
    metric_rows: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    require_confirmation_units: bool,
) -> dict[str, Any]:
    units = sorted({(str(row["split_id"]), int(row["seed"])) for row in metric_rows})
    if require_confirmation_units:
        expected_units = {
            (split_id, seed)
            for split_id, *_ in CONFIRMATION_SPLITS
            for seed in (20260830, 20260831, 20260832)
        }
        if set(units) != expected_units:
            raise DataValidationError(
                "confirmation summary requires the exact C0-C3 x three-seed matrix"
            )
    summary: dict[str, Any] = {"unit_count": len(units), "comparisons": {}}
    flat_rows: list[dict[str, Any]] = []
    for name, (left_method, right_method) in COMPARISONS.items():
        paired = []
        for split_id, seed in units:
            left = _find_metric(
                metric_rows, split_id=split_id, seed=seed, method=left_method
            )
            right = _find_metric(
                metric_rows, split_id=split_id, seed=seed, method=right_method
            )
            row = {
                "comparison": name,
                "left_method": left_method,
                "right_method": right_method,
                "split_id": split_id,
                "seed": seed,
                **{
                    f"delta_{metric}": float(left[metric]) - float(right[metric])
                    for metric in METRIC_KEYS
                },
            }
            paired.append(row)
            flat_rows.append(row)
        aggregate = {
            metric: {
                "mean_delta": float(np.mean([row[f"delta_{metric}"] for row in paired])),
                "std_delta": float(
                    np.std([row[f"delta_{metric}"] for row in paired], ddof=0)
                ),
                "positive_units": int(
                    sum(row[f"delta_{metric}"] > 0.0 for row in paired)
                ),
            }
            for metric in METRIC_KEYS
        }
        summary["comparisons"][name] = {
            "left_method": left_method,
            "right_method": right_method,
            "paired_deltas": paired,
            "aggregate": aggregate,
        }
    summary["paired_delta_rows"] = flat_rows
    if require_confirmation_units:
        main = summary["comparisons"]["learned_rejector"]["aggregate"]
        arpl = summary["comparisons"]["arpl_specific"]["aggregate"]
        main_gate = config["confirmation_decision"]["m6_vs_m4"]
        arpl_gate = config["confirmation_decision"]["m6_vs_m7"]
        main_success = bool(
            main["auroc"]["mean_delta"]
            >= float(main_gate["minimum_mean_auroc_delta"])
            and main["auroc"]["positive_units"]
            >= int(main_gate["minimum_positive_auroc_units"])
            and main["oscr"]["mean_delta"]
            >= float(main_gate["minimum_mean_oscr_delta"])
            and main["known_accuracy"]["mean_delta"]
            >= -float(main_gate["maximum_mean_known_accuracy_drop"])
            and main["fpr95"]["mean_delta"]
            <= float(main_gate["maximum_mean_fpr95_increase"])
        )
        arpl_success = bool(
            arpl["auroc"]["mean_delta"]
            >= float(arpl_gate["minimum_mean_auroc_delta"])
            and arpl["auroc"]["positive_units"]
            >= int(arpl_gate["minimum_positive_auroc_units"])
        )
        m7_better = arpl["auroc"]["mean_delta"] < 0.0
        m4_better = main["auroc"]["mean_delta"] < 0.0
        set_better = (
            summary["comparisons"]["transformer_fusion"]["aggregate"]["auroc"][
                "mean_delta"
            ]
            > 0.0
        )
        summary["decision"] = {
            "m6_main_method_success": main_success,
            "arpl_specific_success": arpl_success,
            "freeze_m6_for_final_test": main_success and arpl_success,
            "set_transformer_mean_auroc_better_than_mean_pooling": set_better,
            "m7_mean_auroc_better_than_m6": m7_better,
            "m4_mean_auroc_better_than_m6": m4_better,
            "required_interpretations": {
                "dual_path_effective_but_arpl_not_required": m7_better,
                "hierarchical_arpl_possible_but_pseudo_rejector_ineffective": m4_better,
                "set_transformer_better_than_mean": set_better,
            },
            "final_unknown_test_authorized": False,
        }
    return summary


def _load_metric_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["seed"] = int(row["seed"])
        for key in METRIC_KEYS:
            row[key] = float(row[key])
    return rows


def run_phase(
    config_path: str | Path,
    bundle_root: str | Path,
    output_root: str | Path,
    *,
    phase: str,
    device_request: str = "auto",
    development_root: str | Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    config = load_mv_rpformer_config(config_path)
    plan = build_phase_plan(config, phase)
    if phase == "confirmation":
        if development_root is None:
            raise DataConfigError("confirmation requires a completed development root")
        verify_development_authorization(config_path, development_root)
    output = Path(output_root).resolve()
    if (output / "summary.json").is_file() and resume:
        audit_phase_root(config_path, output, phase=phase, verify_root_hashes=True)
        return json.loads((output / "summary.json").read_text())
    if output.exists() and any(output.iterdir()) and not resume:
        raise DataValidationError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for unit in plan:
        for method in unit["methods"]:
            run_single_method(
                config_path,
                bundle_root,
                output,
                phase=phase,
                split_id=str(unit["spec"]["split_id"]),
                seed=int(unit["seed"]),
                method=method,
                device_request=device_request,
                development_root=development_root,
                resume=resume,
            )
    return aggregate_phase_root(config_path, output, phase=phase, resume=resume)


def finalize_results(
    config_path: str | Path,
    development_root: str | Path,
    confirmation_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    development = Path(development_root).resolve()
    confirmation = Path(confirmation_root).resolve()
    config = load_mv_rpformer_config(config_path)
    audit_phase_root(
        config_path, development, phase="development", verify_root_hashes=True
    )
    audit_phase_root(
        config_path, confirmation, phase="confirmation", verify_root_hashes=True
    )
    rows = _load_metric_csv(confirmation / "metrics_by_unit.csv")
    summary = summarize_paired_comparisons(
        rows, config=config, require_confirmation_units=True
    )
    paired = summary.pop("paired_delta_rows")
    destination = Path(output_path).resolve()
    paired_destination = destination.with_suffix(".paired_deltas.csv")
    if destination.exists() or paired_destination.exists():
        raise DataValidationError("final result or paired-delta path already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_json(destination, {
        **summary,
        "development_root": str(development),
        "confirmation_root": str(confirmation),
        "config_sha256": config["_config_sha256"],
        "final_unknown_used": False,
        "even_angle_test_used": False,
    })
    _write_csv(paired_destination, paired)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the preregistered MV-RPFormer surrogate OSR experiment"
    )
    parser.add_argument(
        "phase",
        choices=(
            "validate",
            "smoke",
            "development",
            "confirmation",
            "train-unit",
            "aggregate",
            "audit",
            "finalize",
        ),
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--development-root", type=Path)
    parser.add_argument("--confirmation-root", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--run-phase", choices=("smoke", "development", "confirmation")
    )
    parser.add_argument("--split-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.phase == "validate":
        result: Any = {
            "status": "passed",
            "config_sha256": load_mv_rpformer_config(args.config)["_config_sha256"],
        }
    elif args.phase in {"smoke", "development", "confirmation"}:
        if args.bundle_root is None or args.output is None:
            parser.error("training phases require --bundle-root and --output")
        result = run_phase(
            args.config,
            args.bundle_root,
            args.output,
            phase=args.phase,
            device_request=args.device,
            development_root=args.development_root,
            resume=args.resume,
        )
    elif args.phase == "train-unit":
        if (
            args.bundle_root is None
            or args.output is None
            or args.run_phase is None
            or args.split_id is None
            or args.seed is None
            or args.method is None
        ):
            parser.error(
                "train-unit requires bundle-root, output, run-phase, split-id, seed, and method"
            )
        result = run_single_method(
            args.config,
            args.bundle_root,
            args.output,
            phase=args.run_phase,
            split_id=args.split_id,
            seed=args.seed,
            method=args.method,
            device_request=args.device,
            development_root=args.development_root,
            resume=args.resume,
        )
    elif args.phase in {"aggregate", "audit"}:
        if args.output is None or args.run_phase is None:
            parser.error("aggregate/audit require output and run-phase")
        if args.phase == "aggregate":
            result = aggregate_phase_root(
                args.config, args.output, phase=args.run_phase, resume=args.resume
            )
        else:
            result = audit_phase_root(
                args.config,
                args.output,
                phase=args.run_phase,
                verify_root_hashes=True,
            )
    else:
        if args.development_root is None or args.confirmation_root is None or args.output is None:
            parser.error("finalize requires development-root, confirmation-root, and output")
        result = finalize_results(
            args.config, args.development_root, args.confirmation_root, args.output
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
