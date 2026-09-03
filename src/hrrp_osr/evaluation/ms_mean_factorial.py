from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from hrrp_osr.data.errors import DataValidationError


REPORT_METRIC_KEYS = (
    "known_accuracy",
    "known_macro_f1",
    "auroc",
    "oscr",
    "fpr95",
    "known_correct_acceptance_rate",
    "unknown_rejection_rate",
    "open_set_harmonic_score",
    "k_plus_1_macro_f1",
)
METRIC_KEYS = REPORT_METRIC_KEYS

METRIC_ALIASES = {
    "known_correct_acceptance_rate": "KCCR",
    "unknown_rejection_rate": "URR",
    "open_set_harmonic_score": "KCCR_URR_harmonic_mean",
}

FACTORIAL_METHODS = (
    "R0_SHALLOW_MEAN_CE",
    "R1_SHALLOW_MEAN_ARPL",
    "R2_MS_MEAN_CE",
    "R3_MS_MEAN_ARPL",
)
METHODS = FACTORIAL_METHODS

IDENTITY_PAIR_IDS = tuple(f"N{index}" for index in range(7))
ANGLE_FOLDS = (0, 4)
INITIALIZATION_SEEDS = (20260830, 20260831, 20260832)

# A delta is always the method-coefficient linear combination at the same
# identity-pair/fold/seed unit.  In particular, the interaction is computed
# before any aggregation.
FACTORIAL_COMPARISONS: dict[str, dict[str, Any]] = {
    "A": {
        "name": "backbone_arpl",
        "formula": "R3-R1",
        "coefficients": {
            "R3_MS_MEAN_ARPL": 1.0,
            "R1_SHALLOW_MEAN_ARPL": -1.0,
        },
    },
    "B": {
        "name": "backbone_ce",
        "formula": "R2-R0",
        "coefficients": {
            "R2_MS_MEAN_CE": 1.0,
            "R0_SHALLOW_MEAN_CE": -1.0,
        },
    },
    "C": {
        "name": "head_multiscale",
        "formula": "R3-R2",
        "coefficients": {
            "R3_MS_MEAN_ARPL": 1.0,
            "R2_MS_MEAN_CE": -1.0,
        },
    },
    "D": {
        "name": "head_shallow",
        "formula": "R1-R0",
        "coefficients": {
            "R1_SHALLOW_MEAN_ARPL": 1.0,
            "R0_SHALLOW_MEAN_CE": -1.0,
        },
    },
    "interaction": {
        "name": "backbone_head_interaction",
        "formula": "(R3-R2)-(R1-R0)",
        "coefficients": {
            "R3_MS_MEAN_ARPL": 1.0,
            "R2_MS_MEAN_CE": -1.0,
            "R1_SHALLOW_MEAN_ARPL": -1.0,
            "R0_SHALLOW_MEAN_CE": 1.0,
        },
    },
}

BACKBONE_GATE = {
    "minimum_mean_auroc_delta": 0.03,
    "minimum_positive_pair_count": 6,
    "minimum_mean_oscr_delta": 0.0,
    "maximum_mean_known_accuracy_drop": 0.005,
    "maximum_mean_fpr95_increase": 0.02,
}

HEAD_GATE = {
    "minimum_mean_auroc_delta": 0.01,
    "minimum_positive_pair_count": 5,
    "minimum_mean_oscr_delta": 0.0,
    "maximum_mean_known_accuracy_drop": 0.005,
    "maximum_mean_fpr95_increase": 0.02,
}

GATE_ABSOLUTE_TOLERANCE = 1.0e-12


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise DataValidationError(f"{name} must be a finite scalar") from error
    if not math.isfinite(result):
        raise DataValidationError(f"{name} must be a finite scalar")
    return result


def extract_report_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Select and validate the nine preregistered metrics.

    ``evaluate_open_set`` also returns threshold metadata.  This selector keeps
    that metadata out of factorial deltas while allowing callers to retain it
    in each method's ``metrics.json``.
    """

    missing = [key for key in REPORT_METRIC_KEYS if key not in metrics]
    if missing:
        raise DataValidationError(f"factorial metrics are missing keys: {missing}")
    selected = {
        key: _finite_float(metrics[key], f"metric {key}") for key in REPORT_METRIC_KEYS
    }
    outside_unit_interval = [
        key for key, value in selected.items() if not 0.0 <= value <= 1.0
    ]
    if outside_unit_interval:
        raise DataValidationError(
            f"factorial metrics are outside [0, 1]: {outside_unit_interval}"
        )
    kccr = selected["known_correct_acceptance_rate"]
    urr = selected["unknown_rejection_rate"]
    expected_harmonic = 0.0 if kccr + urr == 0.0 else 2.0 * kccr * urr / (kccr + urr)
    if not math.isclose(
        selected["open_set_harmonic_score"],
        expected_harmonic,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise DataValidationError("KCCR/URR harmonic score is inconsistent")
    if kccr > selected["known_accuracy"] + 1.0e-12:
        raise DataValidationError("KCCR cannot exceed closed-set known accuracy")
    return selected


def _normalize_unit_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, str]] = set()
    for position, row in enumerate(rows):
        try:
            pair_id = str(row["pair_id"])
            angle_fold = int(row["angle_fold"])
            seed = int(row["seed"])
            method = str(row["method"])
        except (KeyError, TypeError, ValueError) as error:
            raise DataValidationError(
                f"factorial unit row {position} has invalid identity fields"
            ) from error
        identity = (pair_id, angle_fold, seed, method)
        if identity in seen:
            raise DataValidationError(f"duplicate factorial unit row: {identity}")
        seen.add(identity)
        normalized.append(
            {
                "pair_id": pair_id,
                "angle_fold": angle_fold,
                "seed": seed,
                "method": method,
                **extract_report_metrics(row),
            }
        )

    expected = {
        (pair_id, angle_fold, seed, method)
        for pair_id in IDENTITY_PAIR_IDS
        for angle_fold in ANGLE_FOLDS
        for seed in INITIALIZATION_SEEDS
        for method in FACTORIAL_METHODS
    }
    observed = {
        (row["pair_id"], row["angle_fold"], row["seed"], row["method"])
        for row in normalized
    }
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise DataValidationError(
            "factorial unit matrix differs from the frozen 7x2x3x4 Cartesian "
            f"product; missing={missing[:3]}, extra={extra[:3]}"
        )
    return sorted(
        normalized,
        key=lambda row: (
            IDENTITY_PAIR_IDS.index(row["pair_id"]),
            ANGLE_FOLDS.index(row["angle_fold"]),
            INITIALIZATION_SEEDS.index(row["seed"]),
            FACTORIAL_METHODS.index(row["method"]),
        ),
    )


def _mean_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not rows:
        raise DataValidationError("cannot aggregate an empty metric group")
    return {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in REPORT_METRIC_KEYS
    }


def _method_aggregates(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    overall = []
    by_pair = []
    by_fold = []
    by_seed = []
    for method in FACTORIAL_METHODS:
        selected = [row for row in rows if row["method"] == method]
        overall.append({"method": method, "unit_count": len(selected), **_mean_metrics(selected)})
        for pair_id in IDENTITY_PAIR_IDS:
            group = [row for row in selected if row["pair_id"] == pair_id]
            by_pair.append(
                {
                    "pair_id": pair_id,
                    "method": method,
                    "unit_count": len(group),
                    **_mean_metrics(group),
                }
            )
        for angle_fold in ANGLE_FOLDS:
            group = [row for row in selected if row["angle_fold"] == angle_fold]
            by_fold.append(
                {
                    "angle_fold": angle_fold,
                    "method": method,
                    "unit_count": len(group),
                    **_mean_metrics(group),
                }
            )
        for seed in INITIALIZATION_SEEDS:
            group = [row for row in selected if row["seed"] == seed]
            by_seed.append(
                {
                    "seed": seed,
                    "method": method,
                    "unit_count": len(group),
                    **_mean_metrics(group),
                }
            )
    return {
        "overall": overall,
        "by_pair": by_pair,
        "by_fold": by_fold,
        "by_seed": by_seed,
    }


def _unit_deltas(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    lookup = {
        (row["pair_id"], row["angle_fold"], row["seed"], row["method"]): row
        for row in rows
    }
    result: dict[str, list[dict[str, Any]]] = {}
    for comparison, specification in FACTORIAL_COMPARISONS.items():
        comparison_rows = []
        coefficients: Mapping[str, float] = specification["coefficients"]
        for pair_id in IDENTITY_PAIR_IDS:
            for angle_fold in ANGLE_FOLDS:
                for seed in INITIALIZATION_SEEDS:
                    unit = {
                        method: lookup[(pair_id, angle_fold, seed, method)]
                        for method in FACTORIAL_METHODS
                    }
                    comparison_rows.append(
                        {
                            "pair_id": pair_id,
                            "angle_fold": angle_fold,
                            "seed": seed,
                            **{
                                f"delta_{metric}": float(
                                    sum(
                                        coefficient * float(unit[method][metric])
                                        for method, coefficient in coefficients.items()
                                    )
                                )
                                for metric in REPORT_METRIC_KEYS
                            },
                        }
                    )
        result[comparison] = comparison_rows
    return result


def _pair_deltas(
    unit_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for pair_id in IDENTITY_PAIR_IDS:
        rows = [row for row in unit_rows if row["pair_id"] == pair_id]
        if len(rows) != len(ANGLE_FOLDS) * len(INITIALIZATION_SEEDS):
            raise DataValidationError("pair delta does not contain two folds x three seeds")
        result.append(
            {
                "pair_id": pair_id,
                "unit_count": len(rows),
                **{
                    f"delta_{metric}": float(
                        np.mean([float(row[f"delta_{metric}"]) for row in rows])
                    )
                    for metric in REPORT_METRIC_KEYS
                },
            }
        )
    return result


def paired_bootstrap_mean_ci(
    pair_values: Sequence[float],
    *,
    resamples: int = 10_000,
    seed: int = 20260903,
    confidence_level: float = 0.95,
    resample_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    """Percentile interval for the mean of seven paired identity deltas."""

    values = np.asarray(pair_values, dtype=np.float64)
    if values.shape != (len(IDENTITY_PAIR_IDS),) or not np.isfinite(values).all():
        raise DataValidationError("bootstrap requires seven finite identity-pair deltas")
    if isinstance(resamples, bool) or int(resamples) != resamples or resamples < 1:
        raise DataValidationError("bootstrap resamples must be a positive integer")
    if not 0.0 < confidence_level < 1.0:
        raise DataValidationError("bootstrap confidence level must be in (0, 1)")
    resamples = int(resamples)
    if resample_indices is None:
        indices = np.random.default_rng(seed).integers(
            0, values.size, size=(resamples, values.size)
        )
    else:
        indices = np.asarray(resample_indices)
        if indices.shape != (resamples, values.size):
            raise DataValidationError("shared bootstrap index matrix has the wrong shape")
        if not np.issubdtype(indices.dtype, np.integer) or np.any(indices < 0) or np.any(
            indices >= values.size
        ):
            raise DataValidationError("shared bootstrap indices are invalid")
    bootstrap_means = values[indices].mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(bootstrap_means, [alpha, 1.0 - alpha])
    return {
        "observed_mean_delta": float(values.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "confidence_level": float(confidence_level),
        "resamples": resamples,
        "seed": int(seed),
        "interval": "percentile",
        "statistical_unit": "identity_pair",
        "pair_count": int(values.size),
        "paired_method_delta": True,
        "descriptive_only": True,
        "used_for_gate": False,
    }


def _comparison_group_means(
    rows: Sequence[Mapping[str, Any]], field: str, values: Sequence[int]
) -> list[dict[str, Any]]:
    grouped = []
    for value in values:
        selected = [row for row in rows if int(row[field]) == value]
        grouped.append(
            {
                field: value,
                "unit_count": len(selected),
                **{
                    f"mean_delta_{metric}": float(
                        np.mean([float(row[f"delta_{metric}"]) for row in selected])
                    )
                    for metric in REPORT_METRIC_KEYS
                },
            }
        )
    return grouped


def _within_pair_stability(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    stability = []
    for pair_id in IDENTITY_PAIR_IDS:
        pair_rows = [row for row in rows if row["pair_id"] == pair_id]
        seed_means: dict[str, dict[str, float]] = {}
        for seed in INITIALIZATION_SEEDS:
            selected = [row for row in pair_rows if row["seed"] == seed]
            seed_means[str(seed)] = {
                metric: float(
                    np.mean([float(row[f"delta_{metric}"]) for row in selected])
                )
                for metric in REPORT_METRIC_KEYS
            }
        fold_means: dict[str, dict[str, float]] = {}
        for fold in ANGLE_FOLDS:
            selected = [row for row in pair_rows if row["angle_fold"] == fold]
            fold_means[str(fold)] = {
                metric: float(
                    np.mean([float(row[f"delta_{metric}"]) for row in selected])
                )
                for metric in REPORT_METRIC_KEYS
            }
        stability.append(
            {
                "pair_id": pair_id,
                "seed_means_after_averaging_folds": seed_means,
                "seed_population_std": {
                    metric: float(
                        np.std(
                            [seed_means[str(seed)][metric] for seed in INITIALIZATION_SEEDS],
                            ddof=0,
                        )
                    )
                    for metric in REPORT_METRIC_KEYS
                },
                "fold_means_after_averaging_seeds": fold_means,
                "fold_4_minus_fold_0": {
                    metric: float(fold_means["4"][metric] - fold_means["0"][metric])
                    for metric in REPORT_METRIC_KEYS
                },
            }
        )
    return stability


def _pair_aggregate(
    pair_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_indices: np.ndarray,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for metric in REPORT_METRIC_KEYS:
        values = np.asarray(
            [float(row[f"delta_{metric}"]) for row in pair_rows], dtype=np.float64
        )
        aggregate[metric] = {
            "mean_delta": float(values.mean()),
            "positive_pair_count": int(np.count_nonzero(values > 0.0)),
            "negative_pair_count": int(np.count_nonzero(values < 0.0)),
            "zero_pair_count": int(np.count_nonzero(values == 0.0)),
            "pair_count": int(values.size),
            "bootstrap": paired_bootstrap_mean_ci(
                values,
                resamples=bootstrap_resamples,
                seed=bootstrap_seed,
                confidence_level=confidence_level,
                resample_indices=bootstrap_indices,
            ),
        }
    return aggregate


def summarize_factorial_results(
    unit_metric_rows: Iterable[Mapping[str, Any]],
    *,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 20260903,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Aggregate the frozen 168 method results without treating runs as identities."""

    rows = _normalize_unit_rows(unit_metric_rows)
    if isinstance(bootstrap_resamples, bool) or int(bootstrap_resamples) != bootstrap_resamples:
        raise DataValidationError("bootstrap resamples must be an integer")
    bootstrap_resamples = int(bootstrap_resamples)
    if bootstrap_resamples < 1:
        raise DataValidationError("bootstrap resamples must be positive")
    bootstrap_indices = np.random.default_rng(bootstrap_seed).integers(
        0,
        len(IDENTITY_PAIR_IDS),
        size=(bootstrap_resamples, len(IDENTITY_PAIR_IDS)),
    )
    unit_deltas = _unit_deltas(rows)
    comparisons: dict[str, Any] = {}
    for comparison, specification in FACTORIAL_COMPARISONS.items():
        comparison_units = unit_deltas[comparison]
        comparison_pairs = _pair_deltas(comparison_units)
        comparisons[comparison] = {
            "name": specification["name"],
            "formula": specification["formula"],
            "coefficients": dict(specification["coefficients"]),
            "unit_deltas": comparison_units,
            "pair_deltas": comparison_pairs,
            "by_fold": _comparison_group_means(
                comparison_units, "angle_fold", ANGLE_FOLDS
            ),
            "by_seed": _comparison_group_means(
                comparison_units, "seed", INITIALIZATION_SEEDS
            ),
            "within_pair_stability": _within_pair_stability(comparison_units),
            "pair_aggregate": _pair_aggregate(
                comparison_pairs,
                bootstrap_indices=bootstrap_indices,
                bootstrap_resamples=bootstrap_resamples,
                bootstrap_seed=bootstrap_seed,
                confidence_level=confidence_level,
            ),
        }
    summary: dict[str, Any] = {
        "report_metric_keys": list(REPORT_METRIC_KEYS),
        "metric_aliases": dict(METRIC_ALIASES),
        "method_result_count": len(rows),
        "experimental_unit_count": (
            len(IDENTITY_PAIR_IDS) * len(ANGLE_FOLDS) * len(INITIALIZATION_SEEDS)
        ),
        "identity_pair_count": len(IDENTITY_PAIR_IDS),
        "primary_statistical_unit": "identity_pair",
        "within_pair_aggregation": "arithmetic_mean_over_2_folds_x_3_seeds",
        "combination_samples_are_statistical_repeats": False,
        "method_aggregates": _method_aggregates(rows),
        "comparisons": comparisons,
        "bootstrap_contract": {
            "statistical_unit": "identity_pair",
            "resamples": bootstrap_resamples,
            "seed": int(bootstrap_seed),
            "confidence_level": float(confidence_level),
            "interval": "percentile",
            "descriptive_only": True,
            "used_for_gate": False,
        },
    }
    summary["decision"] = decide_factorial_candidate(summary)
    return summary


def _gate_evidence(
    pair_rows: Sequence[Mapping[str, Any]], thresholds: Mapping[str, float | int]
) -> dict[str, Any]:
    if {str(row["pair_id"]) for row in pair_rows} != set(IDENTITY_PAIR_IDS) or len(
        pair_rows
    ) != len(IDENTITY_PAIR_IDS):
        raise DataValidationError("decision gate requires exactly seven identity-pair rows")
    auroc = np.asarray([float(row["delta_auroc"]) for row in pair_rows])
    oscr = np.asarray([float(row["delta_oscr"]) for row in pair_rows])
    accuracy = np.asarray([float(row["delta_known_accuracy"]) for row in pair_rows])
    fpr95 = np.asarray([float(row["delta_fpr95"]) for row in pair_rows])
    actual = {
        "mean_auroc_delta": float(auroc.mean()),
        "positive_auroc_pair_count": int(np.count_nonzero(auroc > 0.0)),
        "mean_oscr_delta": float(oscr.mean()),
        "mean_known_accuracy_delta": float(accuracy.mean()),
        "mean_fpr95_delta": float(fpr95.mean()),
    }
    def at_least(value: float, threshold: float) -> bool:
        return value >= threshold or math.isclose(
            value, threshold, rel_tol=0.0, abs_tol=GATE_ABSOLUTE_TOLERANCE
        )

    def at_most(value: float, threshold: float) -> bool:
        return value <= threshold or math.isclose(
            value, threshold, rel_tol=0.0, abs_tol=GATE_ABSOLUTE_TOLERANCE
        )

    conditions = {
        "mean_auroc_delta": {
            "actual": actual["mean_auroc_delta"],
            "operator": ">=",
            "threshold": float(thresholds["minimum_mean_auroc_delta"]),
            "passed": at_least(
                actual["mean_auroc_delta"],
                float(thresholds["minimum_mean_auroc_delta"]),
            ),
        },
        "positive_auroc_pair_count": {
            "actual": actual["positive_auroc_pair_count"],
            "operator": ">=",
            "threshold": int(thresholds["minimum_positive_pair_count"]),
            "passed": actual["positive_auroc_pair_count"]
            >= int(thresholds["minimum_positive_pair_count"]),
        },
        "mean_oscr_delta": {
            "actual": actual["mean_oscr_delta"],
            "operator": ">=",
            "threshold": float(thresholds["minimum_mean_oscr_delta"]),
            "passed": at_least(
                actual["mean_oscr_delta"],
                float(thresholds["minimum_mean_oscr_delta"]),
            ),
        },
        "mean_known_accuracy_delta": {
            "actual": actual["mean_known_accuracy_delta"],
            "operator": ">=",
            "threshold": -float(thresholds["maximum_mean_known_accuracy_drop"]),
            "passed": at_least(
                actual["mean_known_accuracy_delta"],
                -float(thresholds["maximum_mean_known_accuracy_drop"]),
            ),
        },
        "mean_fpr95_delta": {
            "actual": actual["mean_fpr95_delta"],
            "operator": "<=",
            "threshold": float(thresholds["maximum_mean_fpr95_increase"]),
            "passed": at_most(
                actual["mean_fpr95_delta"],
                float(thresholds["maximum_mean_fpr95_increase"]),
            ),
        },
    }
    return {
        "passed": bool(all(item["passed"] for item in conditions.values())),
        "statistical_unit": "identity_pair",
        "pair_count": len(IDENTITY_PAIR_IDS),
        "values_are_unrounded": True,
        "boundary_comparison_absolute_tolerance": GATE_ABSOLUTE_TOLERANCE,
        "conditions": conditions,
    }


def _reverse_pair_deltas(pair_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **{key: value for key, value in row.items() if not key.startswith("delta_")},
            **{
                f"delta_{metric}": -float(row[f"delta_{metric}"])
                for metric in REPORT_METRIC_KEYS
            },
        }
        for row in pair_rows
    ]


def recommended_candidate(backbone_label: str, head_label: str) -> str:
    valid_backbones = {
        "backbone_general_success",
        "backbone_arpl_only",
        "backbone_ce_only",
        "no_backbone_gain",
    }
    valid_heads = {"ARPL_PREFERRED", "CE_PREFERRED", "HEAD_INDETERMINATE"}
    if backbone_label not in valid_backbones or head_label not in valid_heads:
        raise DataValidationError("unknown factorial decision label")
    if backbone_label == "backbone_arpl_only":
        return "R3_MS_MEAN_ARPL"
    if backbone_label == "backbone_ce_only":
        return "R2_MS_MEAN_CE"
    if backbone_label == "no_backbone_gain":
        return "none"
    return "R3_MS_MEAN_ARPL" if head_label == "ARPL_PREFERRED" else "R2_MS_MEAN_CE"


def decide_factorial_candidate(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Apply only the preregistered pair-level gates to a factorial summary."""

    try:
        comparison_a = summary["comparisons"]["A"]["pair_deltas"]
        comparison_b = summary["comparisons"]["B"]["pair_deltas"]
        comparison_c = summary["comparisons"]["C"]["pair_deltas"]
    except (KeyError, TypeError) as error:
        raise DataValidationError("factorial summary lacks A/B/C pair deltas") from error

    a_gate = _gate_evidence(comparison_a, BACKBONE_GATE)
    b_gate = _gate_evidence(comparison_b, BACKBONE_GATE)
    if a_gate["passed"] and b_gate["passed"]:
        backbone_label = "backbone_general_success"
    elif a_gate["passed"]:
        backbone_label = "backbone_arpl_only"
    elif b_gate["passed"]:
        backbone_label = "backbone_ce_only"
    else:
        backbone_label = "no_backbone_gain"

    arpl_gate = _gate_evidence(comparison_c, HEAD_GATE)
    ce_gate = _gate_evidence(_reverse_pair_deltas(comparison_c), HEAD_GATE)
    if arpl_gate["passed"] and ce_gate["passed"]:
        raise DataValidationError("symmetric ARPL and CE gates cannot both pass")
    if arpl_gate["passed"]:
        head_label = "ARPL_PREFERRED"
    elif ce_gate["passed"]:
        head_label = "CE_PREFERRED"
    else:
        head_label = "HEAD_INDETERMINATE"

    candidate = recommended_candidate(backbone_label, head_label)
    return {
        "primary_statistical_unit": "identity_pair",
        "identity_pair_count": len(IDENTITY_PAIR_IDS),
        "bootstrap_used_for_gate": False,
        "backbone": {
            "comparison_A": a_gate,
            "comparison_B": b_gate,
            "label": backbone_label,
        },
        "head": {
            "R3_minus_R2": arpl_gate,
            "R2_minus_R3": ce_gate,
            "label": head_label,
        },
        "recommended_candidate": candidate,
        "separate_final_test_preregistration_recommended": candidate != "none",
        "final_unknown_test_authorized": False,
    }
