from __future__ import annotations

import copy

import numpy as np
import pytest

from hrrp_osr.data.errors import DataValidationError
from hrrp_osr.evaluation.metrics import evaluate_open_set
from hrrp_osr.evaluation.ms_mean_factorial import (
    ANGLE_FOLDS,
    FACTORIAL_METHODS,
    IDENTITY_PAIR_IDS,
    INITIALIZATION_SEEDS,
    METRIC_ALIASES,
    REPORT_METRIC_KEYS,
    extract_report_metrics,
    paired_bootstrap_mean_ci,
    recommended_candidate,
    summarize_factorial_results,
)


def _metric_values(**updates: float) -> dict[str, float]:
    values = {
        "known_accuracy": 0.80,
        "known_macro_f1": 0.79,
        "auroc": 0.50,
        "oscr": 0.48,
        "fpr95": 0.60,
        "known_correct_acceptance_rate": 0.70,
        "unknown_rejection_rate": 0.50,
        "open_set_harmonic_score": 7.0 / 12.0,
        "k_plus_1_macro_f1": 0.72,
    }
    values.update(updates)
    if (
        "known_correct_acceptance_rate" in updates
        or "unknown_rejection_rate" in updates
    ) and "open_set_harmonic_score" not in updates:
        kccr = values["known_correct_acceptance_rate"]
        urr = values["unknown_rejection_rate"]
        values["open_set_harmonic_score"] = (
            0.0 if kccr + urr == 0.0 else 2.0 * kccr * urr / (kccr + urr)
        )
    return values


def _full_rows(
    overrides: dict[tuple[str, int, int, str], dict[str, float]] | None = None,
) -> list[dict[str, object]]:
    overrides = overrides or {}
    return [
        {
            "pair_id": pair_id,
            "angle_fold": fold,
            "seed": seed,
            "method": method,
            **_metric_values(**overrides.get((pair_id, fold, seed, method), {})),
        }
        for pair_id in IDENTITY_PAIR_IDS
        for fold in ANGLE_FOLDS
        for seed in INITIALIZATION_SEEDS
        for method in FACTORIAL_METHODS
    ]


def _method_metric_overrides(
    method_values: dict[str, dict[str, float]],
) -> dict[tuple[str, int, int, str], dict[str, float]]:
    return {
        (pair_id, fold, seed, method): dict(method_values.get(method, {}))
        for pair_id in IDENTITY_PAIR_IDS
        for fold in ANGLE_FOLDS
        for seed in INITIALIZATION_SEEDS
        for method in FACTORIAL_METHODS
    }


def test_nine_metric_schema_and_hand_computed_kccr_urr_harmonic() -> None:
    assert REPORT_METRIC_KEYS == (
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
    assert len(REPORT_METRIC_KEYS) == 9
    assert METRIC_ALIASES["known_correct_acceptance_rate"] == "KCCR"
    assert METRIC_ALIASES["unknown_rejection_rate"] == "URR"

    all_metrics = evaluate_open_set(
        known_true=np.array([0, 1, 0, 1]),
        known_pred=np.array([0, 1, 1, 1]),
        known_unknown_scores=np.array([0.1, 0.2, 0.8, 0.3]),
        unknown_pred=np.array([0, 1, 0, 1]),
        unknown_unknown_scores=np.array([0.9, 0.7, 0.6, 0.95]),
        known_validation_scores=np.array([0.4, 0.1, 0.3, 0.2]),
        known_class_count=2,
        known_acceptance_rate=0.75,
    )
    metrics = extract_report_metrics(all_metrics)
    assert list(metrics) == list(REPORT_METRIC_KEYS)
    assert metrics["known_accuracy"] == pytest.approx(3 / 4)
    assert metrics["known_macro_f1"] == pytest.approx((2 / 3 + 4 / 5) / 2)
    assert metrics["auroc"] == pytest.approx(14 / 16)
    assert metrics["oscr"] == pytest.approx(3 / 4)
    assert metrics["fpr95"] == pytest.approx(1 / 4)
    assert metrics["known_correct_acceptance_rate"] == pytest.approx(3 / 4)
    assert metrics["unknown_rejection_rate"] == pytest.approx(1.0)
    assert metrics["open_set_harmonic_score"] == pytest.approx(6 / 7)
    assert metrics["k_plus_1_macro_f1"] == pytest.approx(
        (2 / 3 + 1.0 + 8 / 9) / 3
    )


def test_metric_selector_rejects_missing_nonfinite_and_inconsistent_harmonic() -> None:
    valid = _metric_values()
    missing = dict(valid)
    missing.pop("known_correct_acceptance_rate")
    with pytest.raises(DataValidationError, match="missing"):
        extract_report_metrics(missing)

    nonfinite = dict(valid, auroc=float("nan"))
    with pytest.raises(DataValidationError, match="finite"):
        extract_report_metrics(nonfinite)

    inconsistent = dict(valid, open_set_harmonic_score=0.1)
    with pytest.raises(DataValidationError, match="harmonic"):
        extract_report_metrics(inconsistent)


def test_summary_builds_unit_pair_fold_seed_hierarchy_and_factorial_identity() -> None:
    rows = _full_rows(
        _method_metric_overrides(
            {
                "R0_SHALLOW_MEAN_CE": {"auroc": 0.50},
                "R1_SHALLOW_MEAN_ARPL": {"auroc": 0.51},
                "R2_MS_MEAN_CE": {"auroc": 0.53},
                "R3_MS_MEAN_ARPL": {"auroc": 0.55},
            }
        )
    )
    summary = summarize_factorial_results(rows, bootstrap_resamples=200)

    assert summary["method_result_count"] == 168
    assert summary["experimental_unit_count"] == 42
    assert summary["identity_pair_count"] == 7
    assert summary["primary_statistical_unit"] == "identity_pair"
    assert len(summary["method_aggregates"]["overall"]) == 4
    assert len(summary["method_aggregates"]["by_pair"]) == 28
    assert len(summary["method_aggregates"]["by_fold"]) == 8
    assert len(summary["method_aggregates"]["by_seed"]) == 12

    expected = {"A": 0.04, "B": 0.03, "C": 0.02, "D": 0.01, "interaction": 0.01}
    for comparison, delta in expected.items():
        result = summary["comparisons"][comparison]
        assert len(result["unit_deltas"]) == 42
        assert len(result["pair_deltas"]) == 7
        assert len(result["by_fold"]) == 2
        assert len(result["by_seed"]) == 3
        assert len(result["within_pair_stability"]) == 7
        assert result["pair_aggregate"]["auroc"]["mean_delta"] == pytest.approx(delta)
        assert all(
            row["delta_auroc"] == pytest.approx(delta)
            for row in result["pair_deltas"]
        )

    for index in range(42):
        a = summary["comparisons"]["A"]["unit_deltas"][index]["delta_auroc"]
        b = summary["comparisons"]["B"]["unit_deltas"][index]["delta_auroc"]
        c = summary["comparisons"]["C"]["unit_deltas"][index]["delta_auroc"]
        d = summary["comparisons"]["D"]["unit_deltas"][index]["delta_auroc"]
        interaction = summary["comparisons"]["interaction"]["unit_deltas"][index][
            "delta_auroc"
        ]
        assert interaction == pytest.approx(c - d)
        assert interaction == pytest.approx(a - b)


def test_gate_counts_seven_pair_means_not_42_runs() -> None:
    overrides: dict[tuple[str, int, int, str], dict[str, float]] = {}
    for pair_id in IDENTITY_PAIR_IDS:
        r3_auroc = 0.54 if pair_id != "N6" else 0.49
        for fold in ANGLE_FOLDS:
            for seed in INITIALIZATION_SEEDS:
                overrides[(pair_id, fold, seed, "R3_MS_MEAN_ARPL")] = {
                    "auroc": r3_auroc
                }
    summary = summarize_factorial_results(
        _full_rows(overrides), bootstrap_resamples=200
    )
    a_auroc = summary["comparisons"]["A"]["pair_aggregate"]["auroc"]
    assert a_auroc["positive_pair_count"] == 6
    assert sum(
        row["delta_auroc"] > 0.0
        for row in summary["comparisons"]["A"]["unit_deltas"]
    ) == 36
    assert summary["decision"]["backbone"]["comparison_A"]["passed"] is True
    assert summary["decision"]["backbone"]["comparison_B"]["passed"] is False
    assert summary["decision"]["backbone"]["label"] == "backbone_arpl_only"
    assert summary["decision"]["recommended_candidate"] == "R3_MS_MEAN_ARPL"
    assert summary["decision"]["final_unknown_test_authorized"] is False


def test_backbone_gate_includes_all_five_exact_boundaries() -> None:
    overrides = _method_metric_overrides(
        {
            "R1_SHALLOW_MEAN_ARPL": {
                "known_accuracy": 0.80,
                "auroc": 0.50,
                "oscr": 0.48,
                "fpr95": 0.60,
            },
            "R3_MS_MEAN_ARPL": {
                "known_accuracy": 0.795,
                "auroc": 0.53,
                "oscr": 0.48,
                "fpr95": 0.62,
            },
        }
    )
    summary = summarize_factorial_results(
        _full_rows(overrides), bootstrap_resamples=50
    )
    gate = summary["decision"]["backbone"]["comparison_A"]
    assert gate["passed"] is True
    assert all(condition["passed"] for condition in gate["conditions"].values())
    assert gate["conditions"]["mean_auroc_delta"]["actual"] == pytest.approx(0.03)
    assert gate["conditions"]["mean_known_accuracy_delta"]["actual"] == pytest.approx(
        -0.005
    )
    assert gate["conditions"]["mean_fpr95_delta"]["actual"] == pytest.approx(0.02)


@pytest.mark.parametrize(
    ("r3_auroc", "expected"),
    [(0.52, "ARPL_PREFERRED"), (0.48, "CE_PREFERRED"), (0.505, "HEAD_INDETERMINATE")],
)
def test_head_gate_is_symmetric(r3_auroc: float, expected: str) -> None:
    overrides = _method_metric_overrides(
        {
            "R2_MS_MEAN_CE": {"auroc": 0.50},
            "R3_MS_MEAN_ARPL": {"auroc": r3_auroc},
        }
    )
    summary = summarize_factorial_results(
        _full_rows(overrides), bootstrap_resamples=100
    )
    assert summary["decision"]["head"]["label"] == expected


@pytest.mark.parametrize(
    ("backbone", "head", "expected"),
    [
        ("backbone_general_success", "ARPL_PREFERRED", "R3_MS_MEAN_ARPL"),
        ("backbone_general_success", "CE_PREFERRED", "R2_MS_MEAN_CE"),
        ("backbone_general_success", "HEAD_INDETERMINATE", "R2_MS_MEAN_CE"),
        ("backbone_arpl_only", "ARPL_PREFERRED", "R3_MS_MEAN_ARPL"),
        ("backbone_arpl_only", "CE_PREFERRED", "R3_MS_MEAN_ARPL"),
        ("backbone_arpl_only", "HEAD_INDETERMINATE", "R3_MS_MEAN_ARPL"),
        ("backbone_ce_only", "ARPL_PREFERRED", "R2_MS_MEAN_CE"),
        ("backbone_ce_only", "CE_PREFERRED", "R2_MS_MEAN_CE"),
        ("backbone_ce_only", "HEAD_INDETERMINATE", "R2_MS_MEAN_CE"),
        ("no_backbone_gain", "ARPL_PREFERRED", "none"),
        ("no_backbone_gain", "CE_PREFERRED", "none"),
        ("no_backbone_gain", "HEAD_INDETERMINATE", "none"),
    ],
)
def test_recommended_candidate_mapping_is_exhaustive(
    backbone: str, head: str, expected: str
) -> None:
    assert recommended_candidate(backbone, head) == expected


def test_pair_bootstrap_is_deterministic_descriptive_and_requires_seven_pairs() -> None:
    values = [-0.03, -0.01, 0.0, 0.01, 0.02, 0.04, 0.08]
    first = paired_bootstrap_mean_ci(values, resamples=500, seed=20260903)
    second = paired_bootstrap_mean_ci(values, resamples=500, seed=20260903)
    assert first == second
    assert first["pair_count"] == 7
    assert first["statistical_unit"] == "identity_pair"
    assert first["paired_method_delta"] is True
    assert first["descriptive_only"] is True
    assert first["used_for_gate"] is False
    assert first["ci95_low"] <= first["observed_mean_delta"] <= first["ci95_high"]

    constant = paired_bootstrap_mean_ci([0.03] * 7, resamples=50)
    assert constant["ci95_low"] == pytest.approx(0.03)
    assert constant["ci95_high"] == pytest.approx(0.03)
    with pytest.raises(DataValidationError, match="seven"):
        paired_bootstrap_mean_ci([0.1] * 42, resamples=50)


def test_full_matrix_validation_rejects_missing_duplicate_and_extra_units() -> None:
    rows = _full_rows()
    with pytest.raises(DataValidationError, match="Cartesian"):
        summarize_factorial_results(rows[:-1], bootstrap_resamples=10)

    duplicate = [*rows, copy.deepcopy(rows[0])]
    with pytest.raises(DataValidationError, match="duplicate"):
        summarize_factorial_results(duplicate, bootstrap_resamples=10)

    extra = copy.deepcopy(rows)
    extra[0]["angle_fold"] = 1
    with pytest.raises(DataValidationError, match="Cartesian"):
        summarize_factorial_results(extra, bootstrap_resamples=10)


def test_pair_stability_averages_folds_before_seed_std_and_seeds_before_fold_gap() -> None:
    overrides: dict[tuple[str, int, int, str], dict[str, float]] = {}
    seed_offsets = {20260830: 0.00, 20260831: 0.02, 20260832: 0.04}
    for fold in ANGLE_FOLDS:
        for seed in INITIALIZATION_SEEDS:
            overrides[("N0", fold, seed, "R3_MS_MEAN_ARPL")] = {
                "auroc": 0.50 + seed_offsets[seed] + (0.10 if fold == 4 else 0.0)
            }
    summary = summarize_factorial_results(
        _full_rows(overrides), bootstrap_resamples=50
    )
    stability = summary["comparisons"]["C"]["within_pair_stability"][0]
    assert stability["seed_means_after_averaging_folds"]["20260830"][
        "auroc"
    ] == pytest.approx(0.05)
    assert stability["seed_means_after_averaging_folds"]["20260832"][
        "auroc"
    ] == pytest.approx(0.09)
    assert stability["seed_population_std"]["auroc"] == pytest.approx(
        np.std([0.05, 0.07, 0.09], ddof=0)
    )
    assert stability["fold_4_minus_fold_0"]["auroc"] == pytest.approx(0.10)
