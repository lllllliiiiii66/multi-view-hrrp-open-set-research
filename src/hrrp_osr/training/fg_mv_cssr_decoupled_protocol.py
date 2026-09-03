from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from hrrp_osr.data.errors import DataValidationError
from hrrp_osr.evaluation.metrics import (
    accuracy_score,
    evaluate_open_set,
    macro_f1_score,
)
from hrrp_osr.evaluation.ms_mean_factorial import REPORT_METRIC_KEYS
from hrrp_osr.training.fg_mv_cssr_pilot import (
    compute_class_conditional_mls_scores,
    cssr_conformal_p_values,
)


D0_R2_CLASS_CONDITIONAL_MLS = "D0_R2_CLASS_CONDITIONAL_MLS"
D1_DECOUPLED_REL_CSSR = "D1_DECOUPLED_REL_CSSR"
D2_DECOUPLED_ABSREL_CSSR = "D2_DECOUPLED_ABSREL_CSSR"

METHODS = (
    D0_R2_CLASS_CONDITIONAL_MLS,
    D1_DECOUPLED_REL_CSSR,
    D2_DECOUPLED_ABSREL_CSSR,
)
TRAINABLE_METHODS = METHODS[1:]
PILOT_PAIRS = ("N1", "N4", "N2")
CONFIRMATION_PAIRS = ("N0", "N3", "N5", "N6")

ANGLE_FOLD = 0
R2_SEED = 20260830
CSSR_SEED = 20260905
KNOWN_CLASS_COUNT = 5
TRAIN_BASES_PER_CLASS = 144
TRAIN_BATCH_SIZE = 128
SINGLE_VIEW_CLASS_VERSION = "fg_mv_cssr_decoupled_single_view_class_v1"
SINGLE_VIEW_CLASS_ORDER_VERSION = (
    "fg_mv_cssr_decoupled_single_view_class_order_v1"
)
GATE_TOLERANCE = 1.0e-12

_PILOT_GATE = {
    "minimum_mean_auroc_delta": 0.02,
    "minimum_positive_pair_count": 2,
    "minimum_mean_oscr_delta": 0.0,
    "maximum_mean_kccr_drop": 0.01,
    "maximum_mean_fpr95_increase": 0.02,
    "minimum_identity_auroc": 0.40,
    "minimum_identity_auroc_delta": -0.10,
    "replacement_minimum_mean_auroc_gain": 0.02,
}
_CONFIRMATION_GATE = {
    **_PILOT_GATE,
    "minimum_positive_pair_count": 3,
}
_SIGNAL_BY_METHOD = {
    D1_DECOUPLED_REL_CSSR: "decoupled_relative_signal",
    D2_DECOUPLED_ABSREL_CSSR: "decoupled_absolute_alignment_signal",
}
_DDG_DIRECTIONS = (
    ("N1", "DDG-112", "DDG-1000"),
    ("N4", "DDG-1000", "DDG-112"),
)


def _derived_seed(material: str) -> int:
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")


def _sequence_sha256(values: Sequence[Any]) -> str:
    payload = json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise DataValidationError(f"{name} is NaN or Inf")
    return result


def _csv_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise DataValidationError(f"invalid boolean value: {value!r}")


def build_single_view_schedule(
    rows: Sequence[Mapping[str, Any]],
    *,
    pair_id: str,
    angle_fold: int,
    epoch: int,
    cssr_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the frozen class-balanced, unique-base single-view epoch order."""

    if not pair_id or int(angle_fold) != ANGLE_FOLD or int(epoch) < 1:
        raise DataValidationError("single-view schedule metadata is outside the frozen plan")
    if int(cssr_seed) != CSSR_SEED:
        raise DataValidationError("single-view schedule CSSR seed changed")

    grouped: dict[int, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    sample_ids: list[str] = []
    for source_index, row in enumerate(rows):
        if str(row.get("experiment_role")) != "train_known":
            continue
        label = int(row["model_label"])
        sample_id = str(row["sample_id"])
        if not sample_id:
            raise DataValidationError("train-known base has an empty sample ID")
        if int(row["angle_deg"]) % 2 != 1:
            raise DataValidationError("even-angle sample entered decoupled training")
        grouped[label].append((source_index, row))
        sample_ids.append(sample_id)

    if len(sample_ids) != KNOWN_CLASS_COUNT * TRAIN_BASES_PER_CLASS:
        raise DataValidationError("single-view training population is not 5 x 144")
    if len(sample_ids) != len(set(sample_ids)):
        raise DataValidationError("pair multiplicity repeated a train-known base sample")
    if tuple(sorted(grouped)) != tuple(range(KNOWN_CLASS_COUNT)):
        raise DataValidationError("single-view training labels are not 0..4")
    if any(len(grouped[label]) != TRAIN_BASES_PER_CLASS for label in grouped):
        raise DataValidationError("single-view training classes are not balanced at 144")

    permuted_by_class: dict[int, list[int]] = {}
    class_seeds: dict[str, int] = {}
    for label in range(KNOWN_CLASS_COUNT):
        ordered = sorted(grouped[label], key=lambda item: (label, str(item[1]["sample_id"])))
        material = (
            f"{SINGLE_VIEW_CLASS_VERSION}|{int(cssr_seed)}|{pair_id}|"
            f"{int(angle_fold)}|{int(epoch)}|{label}"
        )
        seed = _derived_seed(material)
        class_seeds[str(label)] = seed
        rng = np.random.Generator(np.random.PCG64(seed))
        permutation = rng.permutation(len(ordered))
        permuted_by_class[label] = [ordered[int(index)][0] for index in permutation]

    order_material = (
        f"{SINGLE_VIEW_CLASS_ORDER_VERSION}|{int(cssr_seed)}|{pair_id}|"
        f"{int(angle_fold)}|{int(epoch)}"
    )
    class_order_seed = _derived_seed(order_material)
    class_order_rng = np.random.Generator(np.random.PCG64(class_order_seed))
    class_order = [int(value) for value in class_order_rng.permutation(KNOWN_CLASS_COUNT)]
    schedule = np.asarray(
        [
            permuted_by_class[label][position]
            for position in range(TRAIN_BASES_PER_CLASS)
            for label in class_order
        ],
        dtype=np.int64,
    )

    scheduled_rows = [rows[int(index)] for index in schedule]
    scheduled_ids = [str(row["sample_id"]) for row in scheduled_rows]
    scheduled_labels = [int(row["model_label"]) for row in scheduled_rows]
    usage = Counter(scheduled_ids)
    class_counts = Counter(scheduled_labels)
    batch_balanced = True
    for start in range(0, len(schedule), TRAIN_BATCH_SIZE):
        batch = scheduled_labels[start : start + TRAIN_BATCH_SIZE]
        if len(batch) != TRAIN_BATCH_SIZE:
            continue
        counts = Counter(batch)
        if set(counts) != set(range(KNOWN_CLASS_COUNT)) or max(counts.values()) - min(
            counts.values()
        ) > 1:
            batch_balanced = False
            break
    if (
        len(schedule) != KNOWN_CLASS_COUNT * TRAIN_BASES_PER_CLASS
        or set(usage.values()) != {1}
        or class_counts != Counter({label: TRAIN_BASES_PER_CLASS for label in range(5)})
        or not batch_balanced
    ):
        raise DataValidationError("single-view schedule failed its frozen constraints")

    schedule_records = [
        {
            "schedule_index": index,
            "source_row_index": int(source_index),
            "model_label": int(rows[int(source_index)]["model_label"]),
            "sample_id": str(rows[int(source_index)]["sample_id"]),
        }
        for index, source_index in enumerate(schedule)
    ]
    audit = {
        "status": "passed",
        "pair_id": str(pair_id),
        "angle_fold": int(angle_fold),
        "epoch": int(epoch),
        "cssr_seed": int(cssr_seed),
        "sample_count": int(schedule.size),
        "class_counts": {str(key): int(value) for key, value in sorted(class_counts.items())},
        "class_seeds": class_seeds,
        "class_order_seed": class_order_seed,
        "class_order": class_order,
        "sample_usage_exactly_once": set(usage.values()) == {1},
        "full_batches_class_balanced": batch_balanced,
        "schedule_sha256": _sequence_sha256(schedule_records),
        "sample_id_population_sha256": _sequence_sha256(sorted(sample_ids)),
        "class_schedule_version": SINGLE_VIEW_CLASS_VERSION,
        "class_order_version": SINGLE_VIEW_CLASS_ORDER_VERSION,
        "dataloader_shuffle": False,
        "pair_multiplicity_used": False,
        "known_calibration_used": False,
        "surrogate_unknown_used": False,
        "final_unknown_used": False,
        "even_angle_test_used": False,
    }
    return schedule, audit


def build_guided_reference_scores(
    unique_rows: Sequence[Mapping[str, Any]],
    r_values: np.ndarray,
    *,
    epsilon: float = 1.0e-8,
) -> tuple[
    dict[str, np.ndarray],
    list[np.ndarray],
    list[tuple[str, ...]],
    dict[str, Any],
]:
    """Build one shared unique-base conformal reference for each true class."""

    values = np.asarray(r_values, dtype=np.float64)
    if (
        values.shape != (len(unique_rows), KNOWN_CLASS_COUNT)
        or not np.isfinite(values).all()
        or float(epsilon) != 1.0e-8
    ):
        raise DataValidationError("guided reconstruction values changed or are invalid")
    role_indices = {
        role: np.asarray(
            [
                index
                for index, row in enumerate(unique_rows)
                if str(row.get("experiment_role")) == role
            ],
            dtype=np.int64,
        )
        for role in ("known_calibration", "surrogate_unknown")
    }
    calibration = role_indices["known_calibration"]
    surrogate = role_indices["surrogate_unknown"]
    if calibration.size == 0 or surrogate.size == 0:
        raise DataValidationError("guided scoring requires calibration and surrogate bases")
    calibration_labels = np.asarray(
        [int(unique_rows[int(index)]["model_label"]) for index in calibration],
        dtype=np.int64,
    )
    calibration_ids = tuple(
        str(unique_rows[int(index)]["sample_id"]) for index in calibration
    )
    all_evaluation_ids = [
        str(unique_rows[int(index)]["sample_id"])
        for role in role_indices.values()
        for index in role
    ]
    if len(all_evaluation_ids) != len(set(all_evaluation_ids)):
        raise DataValidationError("evaluation base manifest repeats a sample ID")

    references: list[np.ndarray] = []
    reference_ids: list[tuple[str, ...]] = []
    for class_index in range(KNOWN_CLASS_COUNT):
        selected = calibration[calibration_labels == class_index]
        ids = tuple(str(unique_rows[int(index)]["sample_id"]) for index in selected)
        if len(ids) < 2 or len(ids) != len(set(ids)):
            raise DataValidationError("each guided class reference needs unique LOO bases")
        references.append(np.asarray(values[selected, class_index], dtype=np.float64))
        reference_ids.append(ids)

    calibration_p = cssr_conformal_p_values(
        values[calibration],
        references,
        sample_ids=calibration_ids,
        reference_sample_ids=reference_ids,
        true_labels=calibration_labels,
        leave_one_base_sample_out=True,
    )
    surrogate_p = cssr_conformal_p_values(values[surrogate], references)
    calibration_a = -np.log(calibration_p + float(epsilon))
    surrogate_a = -np.log(surrogate_p + float(epsilon))
    if not np.isfinite(calibration_a).all() or not np.isfinite(surrogate_a).all():
        raise DataValidationError("guided anomaly contains NaN or Inf")

    score_by_sample: dict[str, np.ndarray] = {}
    r_by_sample: dict[str, np.ndarray] = {}
    p_by_sample: dict[str, np.ndarray] = {}
    for role, p_values, anomaly in (
        ("known_calibration", calibration_p, calibration_a),
        ("surrogate_unknown", surrogate_p, surrogate_a),
    ):
        for local_index, source_index in enumerate(role_indices[role]):
            sample_id = str(unique_rows[int(source_index)]["sample_id"])
            score_by_sample[sample_id] = anomaly[local_index]
            r_by_sample[sample_id] = values[int(source_index)]
            p_by_sample[sample_id] = p_values[local_index]
    arrays = {
        "r": values,
        "known_calibration_p": calibration_p,
        "known_calibration_a": calibration_a,
        "surrogate_unknown_p": surrogate_p,
        "surrogate_unknown_a": surrogate_a,
    }
    metadata = {
        "status": "passed",
        "reference_counts": [len(values) for values in references],
        "reference_sample_id_hashes": [_sequence_sha256(ids) for ids in reference_ids],
        "shared_reference_across_slots": True,
        "calibration_leave_one_base_sample_out": True,
        "surrogate_unknown_in_reference": False,
        "pair_multiplicity_used": False,
        "score_by_sample": score_by_sample,
        "r_by_sample": r_by_sample,
        "p_by_sample": p_by_sample,
    }
    return arrays, references, reference_ids, metadata


def guided_scores_from_r2_predictions(
    fused_r2_logits: np.ndarray,
    view_class_anomaly: np.ndarray,
) -> dict[str, np.ndarray]:
    """Use the unchanged fused R2 prediction to query both CSSR views."""

    logits = np.asarray(fused_r2_logits)
    anomaly = np.asarray(view_class_anomaly, dtype=np.float64)
    if (
        logits.ndim != 2
        or logits.shape[1] != KNOWN_CLASS_COUNT
        or anomaly.shape != (logits.shape[0], 2, KNOWN_CLASS_COUNT)
        or not np.isfinite(logits).all()
        or not np.isfinite(anomaly).all()
    ):
        raise DataValidationError("guided score arrays do not match [n,2,5]")
    prediction = logits.argmax(axis=1).astype(np.int64)
    selected = anomaly[
        np.arange(logits.shape[0])[:, None],
        np.arange(2)[None, :],
        prediction[:, None],
    ]
    return {
        "known_prediction": prediction,
        "unknown_score": selected.mean(axis=1),
    }


def class_conditional_mls_score(
    query_logits: np.ndarray,
    *,
    calibration_logits: np.ndarray,
    calibration_true_labels: np.ndarray,
    query_pair_ids: Sequence[str] | None = None,
    calibration_pair_ids: Sequence[str] | None = None,
    leave_one_pair_out: bool = False,
) -> np.ndarray:
    """Return the frozen D0 predicted-class conditional MLS anomaly score."""

    query = np.asarray(query_logits, dtype=np.float64)
    calibration = np.asarray(calibration_logits, dtype=np.float64)
    true_labels = np.asarray(calibration_true_labels, dtype=np.int64)
    if (
        query.ndim != 2
        or calibration.ndim != 2
        or query.shape[1] != KNOWN_CLASS_COUNT
        or calibration.shape[1] != KNOWN_CLASS_COUNT
        or true_labels.shape != (calibration.shape[0],)
        or not np.isfinite(query).all()
        or not np.isfinite(calibration).all()
    ):
        raise DataValidationError("D0 class-conditional MLS arrays are invalid")
    return compute_class_conditional_mls_scores(
        -query.max(axis=1),
        query.argmax(axis=1),
        reference_nonconformity=-calibration.max(axis=1),
        reference_true_labels=true_labels,
        reference_predicted_labels=calibration.argmax(axis=1),
        query_pair_ids=query_pair_ids,
        reference_pair_ids=calibration_pair_ids,
        leave_one_out=leave_one_pair_out,
    )


def class_conditional_mls_for_roles(
    *,
    full_calibration_logits: np.ndarray,
    full_calibration_labels: np.ndarray,
    full_calibration_pair_ids: Sequence[str],
    role_logits: Mapping[str, np.ndarray],
    role_pair_ids: Mapping[str, Sequence[str]],
) -> dict[str, np.ndarray]:
    expected_roles = {"known_calibration", "surrogate_unknown"}
    if set(role_logits) != expected_roles or set(role_pair_ids) != expected_roles:
        raise DataValidationError("D0 scoring roles changed")
    return {
        role: class_conditional_mls_score(
            role_logits[role],
            calibration_logits=full_calibration_logits,
            calibration_true_labels=full_calibration_labels,
            query_pair_ids=role_pair_ids[role] if role == "known_calibration" else None,
            calibration_pair_ids=(
                full_calibration_pair_ids if role == "known_calibration" else None
            ),
            leave_one_pair_out=role == "known_calibration",
        )
        for role in ("known_calibration", "surrogate_unknown")
    }


def audit_shared_r2_predictions(
    logits_by_method: Mapping[str, np.ndarray],
    known_true_labels: np.ndarray,
) -> dict[str, Any]:
    """Require D0/D1/D2 to preserve exact R2 logits and known predictions."""

    if not set(METHODS) <= set(logits_by_method):
        raise DataValidationError("D0/D1/D2 logits are incomplete")
    true_labels = np.asarray(known_true_labels, dtype=np.int64)
    baseline = np.asarray(logits_by_method[D0_R2_CLASS_CONDITIONAL_MLS])
    if (
        baseline.ndim != 2
        or baseline.shape[1] != KNOWN_CLASS_COUNT
        or true_labels.shape != (baseline.shape[0],)
        or not np.isfinite(baseline).all()
    ):
        raise DataValidationError("shared R2 baseline logits are invalid")
    baseline_prediction = baseline.argmax(axis=1).astype(np.int64)
    accuracy = accuracy_score(true_labels, baseline_prediction)
    macro_f1 = macro_f1_score(true_labels, baseline_prediction, range(KNOWN_CLASS_COUNT))
    hashes: dict[str, str] = {}
    for method in METHODS:
        current = np.asarray(logits_by_method[method])
        if current.dtype != baseline.dtype or not np.array_equal(current, baseline):
            raise DataValidationError(f"{method} changed frozen R2 fused logits")
        prediction = current.argmax(axis=1).astype(np.int64)
        if not np.array_equal(prediction, baseline_prediction):
            raise DataValidationError(f"{method} changed frozen R2 known prediction")
        if accuracy_score(true_labels, prediction) != accuracy or macro_f1_score(
            true_labels, prediction, range(KNOWN_CLASS_COUNT)
        ) != macro_f1:
            raise DataValidationError(f"{method} changed frozen known metrics")
        hashes[method] = _array_sha256(current)
    return {
        "status": "passed",
        "methods": list(METHODS),
        "logits_sha256_by_method": hashes,
        "known_prediction_sha256": _array_sha256(baseline_prediction),
        "known_accuracy": accuracy,
        "known_macro_f1": macro_f1,
        "logits_exactly_equal": True,
        "known_predictions_exactly_equal": True,
        "known_metrics_exactly_equal": True,
    }


def recompute_metrics_from_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    known_class_count: int = KNOWN_CLASS_COUNT,
    known_acceptance_rate: float = 0.95,
) -> dict[str, float]:
    """Recompute all nine report metrics from per-pair predictions and scores."""

    known = [row for row in rows if str(row.get("evaluation_role")) == "known_calibration"]
    unknown = [row for row in rows if str(row.get("evaluation_role")) == "surrogate_unknown"]
    if not known or not unknown:
        raise DataValidationError("prediction rows lack known or surrogate samples")
    values = evaluate_open_set(
        known_true=np.asarray([int(row["true_label"]) for row in known], dtype=np.int64),
        known_pred=np.asarray(
            [int(row["predicted_known_label"]) for row in known], dtype=np.int64
        ),
        known_unknown_scores=np.asarray(
            [_finite_float(row["unknown_score"], "known unknown score") for row in known]
        ),
        unknown_pred=np.asarray(
            [int(row["predicted_known_label"]) for row in unknown], dtype=np.int64
        ),
        unknown_unknown_scores=np.asarray(
            [_finite_float(row["unknown_score"], "surrogate unknown score") for row in unknown]
        ),
        known_validation_scores=np.asarray(
            [_finite_float(row["unknown_score"], "known threshold score") for row in known]
        ),
        known_class_count=int(known_class_count),
        known_acceptance_rate=float(known_acceptance_rate),
    )
    if any(key not in values for key in REPORT_METRIC_KEYS):
        raise DataValidationError("open-set evaluator omitted a required report metric")
    for row in rows:
        if "threshold" in row and not math.isclose(
            float(row["threshold"]), float(values["threshold"]), rel_tol=0.0, abs_tol=1e-15
        ):
            raise DataValidationError("saved threshold does not reproduce")
        if "rejected" in row and _csv_bool(row["rejected"]) != (
            float(row["unknown_score"]) > float(values["threshold"])
        ):
            raise DataValidationError("saved rejection decision does not reproduce")
    return {key: float(value) for key, value in values.items()}


def build_identity_and_absorption_rows(
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    pair_id: str,
    train_class_order: Sequence[str],
    known_acceptance_rate: float = 0.95,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build per-identity metrics and every false-accept destination count."""

    if method not in METHODS or len(train_class_order) != KNOWN_CLASS_COUNT:
        raise DataValidationError("identity analysis method or class order changed")
    if not prediction_rows or {
        str(row.get("method")) for row in prediction_rows
    } != {method}:
        raise DataValidationError("identity prediction rows contain a different method")
    known = [
        row
        for row in prediction_rows
        if str(row.get("evaluation_role")) == "known_calibration"
    ]
    unknown = [
        row
        for row in prediction_rows
        if str(row.get("evaluation_role")) == "surrogate_unknown"
    ]
    identities = tuple(sorted({str(row["class_name"]) for row in unknown}))
    if not known or len(identities) != 2:
        raise DataValidationError("identity analysis requires known rows and two identities")
    known_true = np.asarray([int(row["true_label"]) for row in known], dtype=np.int64)
    known_pred = np.asarray(
        [int(row["predicted_known_label"]) for row in known], dtype=np.int64
    )
    known_scores = np.asarray([float(row["unknown_score"]) for row in known])
    identity_rows: list[dict[str, Any]] = []
    absorption_rows: list[dict[str, Any]] = []
    for identity in identities:
        selected = [row for row in unknown if str(row["class_name"]) == identity]
        metrics = evaluate_open_set(
            known_true=known_true,
            known_pred=known_pred,
            known_unknown_scores=known_scores,
            unknown_pred=np.asarray(
                [int(row["predicted_known_label"]) for row in selected], dtype=np.int64
            ),
            unknown_unknown_scores=np.asarray(
                [float(row["unknown_score"]) for row in selected], dtype=np.float64
            ),
            known_validation_scores=known_scores,
            known_class_count=KNOWN_CLASS_COUNT,
            known_acceptance_rate=float(known_acceptance_rate),
        )
        identity_rows.append(
            {
                "pair_id": pair_id,
                "method": method,
                "surrogate_identity": identity,
                **{key: float(metrics[key]) for key in REPORT_METRIC_KEYS},
                "threshold": float(metrics["threshold"]),
            }
        )
        accepted = [
            row
            for row in selected
            if float(row["unknown_score"]) <= float(metrics["threshold"])
        ]
        counts = Counter(int(row["predicted_known_label"]) for row in accepted)
        for label, known_identity in enumerate(train_class_order):
            count = int(counts.get(label, 0))
            absorption_rows.append(
                {
                    "pair_id": pair_id,
                    "method": method,
                    "surrogate_identity": identity,
                    "absorbed_as_known_identity": str(known_identity),
                    "false_accept_count": count,
                    "total_surrogate_count": len(selected),
                    "total_false_accept_count": len(accepted),
                    "rate_over_all_surrogate": count / len(selected),
                    "composition_within_false_accepts": (
                        0.0 if not accepted else count / len(accepted)
                    ),
                }
            )
    return identity_rows, absorption_rows, {
        "status": "passed",
        "pair_id": pair_id,
        "method": method,
        "surrogate_identities": list(identities),
        "ddg_false_accept_counts": ddg_false_accept_counts(
            absorption_rows, method=method, allow_missing=True
        ),
        "all_known_destinations_reported": True,
        "final_unknown_used": False,
        "even_angle_test_used": False,
    }


def ddg_false_accept_counts(
    absorption_rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    allow_missing: bool = False,
) -> dict[str, int | None]:
    """Extract the two preregistered directional DDG false-accept counts."""

    result: dict[str, int | None] = {}
    for pair_id, source, destination in _DDG_DIRECTIONS:
        key = f"{source}_absorbed_as_{destination}"
        matches = [
            row
            for row in absorption_rows
            if str(row.get("pair_id")) == pair_id
            and str(row.get("method")) == method
            and str(row.get("surrogate_identity")) == source
            and str(row.get("absorbed_as_known_identity")) == destination
        ]
        if len(matches) > 1:
            raise DataValidationError("DDG absorption rows contain a duplicate")
        if not matches:
            if allow_missing:
                result[key] = None
                continue
            raise DataValidationError(f"missing DDG absorption direction: {key}")
        count = int(matches[0]["false_accept_count"])
        if count < 0:
            raise DataValidationError("DDG false-accept count is negative")
        result[key] = count
    return result


def _metric_map_or_missing(
    rows: Sequence[Mapping[str, Any]],
    *,
    pair_ids: Sequence[str],
    methods: Sequence[str],
) -> tuple[dict[str, dict[str, Mapping[str, Any]]], list[str]]:
    result: dict[str, dict[str, Mapping[str, Any]]] = {pair_id: {} for pair_id in pair_ids}
    for row in rows:
        pair_id = str(row.get("pair_id"))
        method = str(row.get("method"))
        if pair_id not in result or method not in methods:
            raise DataValidationError("gate metric row is outside the frozen plan")
        if method in result[pair_id]:
            raise DataValidationError("gate metric rows contain a duplicate")
        result[pair_id][method] = row
    missing = [
        f"metric:{pair_id}:{method}"
        for pair_id in pair_ids
        for method in methods
        if method not in result[pair_id]
    ]
    return result, missing


def _identity_map_or_missing(
    rows: Sequence[Mapping[str, Any]],
    *,
    pair_ids: Sequence[str],
    methods: Sequence[str],
) -> tuple[dict[tuple[str, str, str], Mapping[str, Any]], list[str]]:
    result: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    identities: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        pair_id = str(row.get("pair_id"))
        method = str(row.get("method"))
        identity = str(row.get("surrogate_identity", ""))
        if pair_id not in pair_ids or method not in methods or not identity:
            raise DataValidationError("identity row is outside the frozen plan")
        key = (pair_id, method, identity)
        if key in result:
            raise DataValidationError("identity rows contain a duplicate")
        result[key] = row
        identities[pair_id].add(identity)
    missing: list[str] = []
    for pair_id in pair_ids:
        baseline_identities = {
            identity
            for current_pair, method, identity in result
            if current_pair == pair_id and method == D0_R2_CLASS_CONDITIONAL_MLS
        }
        if len(baseline_identities) != 2:
            missing.append(f"identity_population:{pair_id}:D0")
            continue
        if identities[pair_id] != baseline_identities:
            missing.append(f"identity_population:{pair_id}:method_mismatch")
        for method in methods:
            for identity in baseline_identities:
                if (pair_id, method, identity) not in result:
                    missing.append(f"identity:{pair_id}:{method}:{identity}")
    return result, missing


def _not_evaluated(phase: str, missing: Sequence[str]) -> dict[str, Any]:
    return {
        "status": "not_evaluated",
        "phase": phase,
        "reason": "incomplete_required_tasks",
        "missing": sorted(set(str(value) for value in missing)),
        "selected_method": None,
        "confirmation_allowed": False,
        "final_unknown_test_authorized": False,
    }


def _candidate_gate(
    metric_map: Mapping[str, Mapping[str, Mapping[str, Any]]],
    identity_map: Mapping[tuple[str, str, str], Mapping[str, Any]],
    *,
    pair_ids: Sequence[str],
    candidate: str,
    gate: Mapping[str, float | int],
    absorption_rows: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    deltas: list[dict[str, Any]] = []
    for pair_id in pair_ids:
        candidate_row = metric_map[pair_id][candidate]
        baseline_row = metric_map[pair_id][D0_R2_CLASS_CONDITIONAL_MLS]
        deltas.append(
            {
                "pair_id": pair_id,
                "delta_auroc": _finite_float(candidate_row["auroc"], "candidate AUROC")
                - _finite_float(baseline_row["auroc"], "D0 AUROC"),
                "delta_oscr": _finite_float(candidate_row["oscr"], "candidate OSCR")
                - _finite_float(baseline_row["oscr"], "D0 OSCR"),
                "delta_kccr": _finite_float(
                    candidate_row["known_correct_acceptance_rate"], "candidate KCCR"
                )
                - _finite_float(
                    baseline_row["known_correct_acceptance_rate"], "D0 KCCR"
                ),
                "delta_fpr95": _finite_float(candidate_row["fpr95"], "candidate FPR95")
                - _finite_float(baseline_row["fpr95"], "D0 FPR95"),
            }
        )
    mean = {
        name: float(np.mean([row[f"delta_{name}"] for row in deltas]))
        for name in ("auroc", "oscr", "kccr", "fpr95")
    }
    identity_evidence: list[dict[str, Any]] = []
    for key, row in sorted(identity_map.items()):
        pair_id, method, identity = key
        if pair_id not in pair_ids or method != candidate:
            continue
        baseline = identity_map[(pair_id, D0_R2_CLASS_CONDITIONAL_MLS, identity)]
        candidate_auroc = _finite_float(row["auroc"], "candidate identity AUROC")
        d0_auroc = _finite_float(baseline["auroc"], "D0 identity AUROC")
        identity_evidence.append(
            {
                "pair_id": pair_id,
                "surrogate_identity": identity,
                "candidate_auroc": candidate_auroc,
                "d0_auroc": d0_auroc,
                "delta_auroc": candidate_auroc - d0_auroc,
            }
        )
    minimum_identity_auroc = min(row["candidate_auroc"] for row in identity_evidence)
    minimum_identity_delta = min(row["delta_auroc"] for row in identity_evidence)
    checks = {
        "mean_auroc_delta": mean["auroc"] + GATE_TOLERANCE
        >= float(gate["minimum_mean_auroc_delta"]),
        "positive_pair_count": sum(row["delta_auroc"] > 0.0 for row in deltas)
        >= int(gate["minimum_positive_pair_count"]),
        "mean_oscr_delta": mean["oscr"] + GATE_TOLERANCE
        >= float(gate["minimum_mean_oscr_delta"]),
        "mean_kccr_delta": mean["kccr"] + GATE_TOLERANCE
        >= -float(gate["maximum_mean_kccr_drop"]),
        "mean_fpr95_delta": mean["fpr95"]
        <= float(gate["maximum_mean_fpr95_increase"]) + GATE_TOLERANCE,
        "minimum_identity_auroc": minimum_identity_auroc + GATE_TOLERANCE
        >= float(gate["minimum_identity_auroc"]),
        "minimum_identity_auroc_delta": minimum_identity_delta + GATE_TOLERANCE
        >= float(gate["minimum_identity_auroc_delta"]),
    }
    ddg_evidence: list[dict[str, Any]] = []
    if absorption_rows is not None:
        candidate_counts = ddg_false_accept_counts(absorption_rows, method=candidate)
        baseline_counts = ddg_false_accept_counts(
            absorption_rows, method=D0_R2_CLASS_CONDITIONAL_MLS
        )
        for key in candidate_counts:
            candidate_count = int(candidate_counts[key])
            d0_count = int(baseline_counts[key])
            ddg_evidence.append(
                {
                    "direction": key,
                    "candidate_false_accept_count": candidate_count,
                    "d0_false_accept_count": d0_count,
                    "not_worse": candidate_count <= d0_count,
                }
            )
        checks["ddg_112_to_1000_not_worse"] = ddg_evidence[0]["not_worse"]
        checks["ddg_1000_to_112_not_worse"] = ddg_evidence[1]["not_worse"]
    return {
        "candidate": candidate,
        "baseline": D0_R2_CLASS_CONDITIONAL_MLS,
        "pair_deltas": deltas,
        "mean_deltas": mean,
        "positive_auroc_pair_count": sum(row["delta_auroc"] > 0.0 for row in deltas),
        "identity_evidence": identity_evidence,
        "minimum_identity_auroc": minimum_identity_auroc,
        "minimum_identity_auroc_delta": minimum_identity_delta,
        "ddg_evidence": ddg_evidence,
        "checks": checks,
        "passed": all(checks.values()),
        "mean_candidate_auroc": float(
            np.mean([float(metric_map[pair][candidate]["auroc"]) for pair in pair_ids])
        ),
    }


def evaluate_pilot_gate(
    metric_rows: Sequence[Mapping[str, Any]],
    identity_rows: Sequence[Mapping[str, Any]],
    absorption_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    metric_map, metric_missing = _metric_map_or_missing(
        metric_rows, pair_ids=PILOT_PAIRS, methods=METHODS
    )
    identity_map, identity_missing = _identity_map_or_missing(
        identity_rows, pair_ids=PILOT_PAIRS, methods=METHODS
    )
    absorption_missing: list[str] = []
    for method in METHODS:
        counts = ddg_false_accept_counts(absorption_rows, method=method, allow_missing=True)
        absorption_missing.extend(
            f"absorption:{method}:{direction}"
            for direction, value in counts.items()
            if value is None
        )
    missing = [*metric_missing, *identity_missing, *absorption_missing]
    if missing:
        return _not_evaluated("pilot", missing)

    candidates = {
        method: _candidate_gate(
            metric_map,
            identity_map,
            pair_ids=PILOT_PAIRS,
            candidate=method,
            gate=_PILOT_GATE,
            absorption_rows=absorption_rows,
        )
        for method in TRAINABLE_METHODS
    }
    d1 = candidates[D1_DECOUPLED_REL_CSSR]
    d2 = candidates[D2_DECOUPLED_ABSREL_CSSR]
    selected: str | None = None
    if d1["passed"]:
        selected = D1_DECOUPLED_REL_CSSR
        if (
            d2["passed"]
            and d2["mean_candidate_auroc"] + GATE_TOLERANCE
            >= d1["mean_candidate_auroc"]
            + float(_PILOT_GATE["replacement_minimum_mean_auroc_gain"])
        ):
            selected = D2_DECOUPLED_ABSREL_CSSR
    elif d2["passed"]:
        selected = D2_DECOUPLED_ABSREL_CSSR
    return {
        "status": "evaluated",
        "phase": "pilot",
        "signal": (
            "decoupled_cssr_failed" if selected is None else _SIGNAL_BY_METHOD[selected]
        ),
        "selected_method": selected,
        "confirmation_allowed": selected is not None,
        "candidate_gates": candidates,
        "selection_rule": "D1_priority_D2_requires_additional_2pp_mean_AUROC",
        "final_unknown_test_authorized": False,
    }


def evaluate_confirmation_gate(
    metric_rows: Sequence[Mapping[str, Any]],
    identity_rows: Sequence[Mapping[str, Any]],
    selected_method: str,
) -> dict[str, Any]:
    if selected_method not in TRAINABLE_METHODS:
        raise DataValidationError("confirmation requires the selected D1 or D2 method")
    methods = (D0_R2_CLASS_CONDITIONAL_MLS, selected_method)
    metric_map, metric_missing = _metric_map_or_missing(
        metric_rows, pair_ids=CONFIRMATION_PAIRS, methods=methods
    )
    identity_map, identity_missing = _identity_map_or_missing(
        identity_rows, pair_ids=CONFIRMATION_PAIRS, methods=methods
    )
    missing = [*metric_missing, *identity_missing]
    if missing:
        return {
            **_not_evaluated("confirmation", missing),
            "selected_method": selected_method,
        }
    evidence = _candidate_gate(
        metric_map,
        identity_map,
        pair_ids=CONFIRMATION_PAIRS,
        candidate=selected_method,
        gate=_CONFIRMATION_GATE,
        absorption_rows=None,
    )
    return {
        **evidence,
        "status": "evaluated",
        "phase": "confirmation",
        "selected_method": selected_method,
        "decision": (
            "decoupled_cssr_worth_full_validation"
            if evidence["passed"]
            else "decoupled_cssr_rejected"
        ),
        "confirmation_allowed": False,
        "final_unknown_test_authorized": False,
    }


def build_phase_plan(
    phase: str,
    selected_method: str | None = None,
) -> list[dict[str, Any]]:
    """Return only preregistered stage-B trainable units."""

    if phase == "smoke":
        pair_ids = ("N1",)
        methods = TRAINABLE_METHODS
        epochs = 6
        diagnostic_only = True
    elif phase == "pilot":
        pair_ids = PILOT_PAIRS
        methods = TRAINABLE_METHODS
        epochs = 20
        diagnostic_only = False
    elif phase == "confirmation":
        if selected_method not in TRAINABLE_METHODS:
            raise DataValidationError(
                "confirmation plan requires an audited selected D1 or D2 method"
            )
        pair_ids = CONFIRMATION_PAIRS
        methods = (str(selected_method),)
        epochs = 20
        diagnostic_only = False
    else:
        raise DataValidationError("phase must be smoke, pilot, or confirmation")
    return [
        {
            "phase": phase,
            "pair_id": pair_id,
            "method": method,
            "angle_fold": ANGLE_FOLD,
            "r2_seed": R2_SEED,
            "cssr_seed": CSSR_SEED,
            "epochs": epochs,
            "diagnostic_only": diagnostic_only,
            "d0_retraining": False,
            "final_unknown_test_authorized": False,
        }
        for pair_id in pair_ids
        for method in methods
    ]
