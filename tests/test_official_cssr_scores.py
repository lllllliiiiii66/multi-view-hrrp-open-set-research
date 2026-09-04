from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from hrrp_osr.evaluation.official_cssr_scores import (  # noqa: E402
    OfficialScoreNormalization,
    build_official_score_templates,
    fit_score_normalization,
    matched_linear_pair_output,
    official_g_p_pro,
    official_pcssr_pair_scores,
    raw_official_scores,
    standardize_and_integrate,
)


def _fixture() -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.tensor(
        [
            [[1.0, -2.0], [3.0, -4.0]],
            [[-2.0, 1.0], [1.0, 2.0]],
            [[0.5, -1.5], [2.5, -0.5]],
            [[-1.0, 3.0], [-2.0, 1.0]],
        ],
        dtype=torch.float64,
    )
    predictions = torch.tensor([0, 1, 0, 1])
    return features, predictions


def test_official_g_p_pro_matches_literal_power_gram() -> None:
    features, _ = _fixture()
    powered = features**8
    expected_raw = powered @ powered.transpose(1, 2)
    expected = expected_raw.sign() * expected_raw.abs().pow(1.0 / 8.0)
    torch.testing.assert_close(official_g_p_pro(features), expected)


def test_template_uses_absolute_train_features_and_classwise_normalization() -> None:
    features, predictions = _fixture()
    templates = build_official_score_templates(
        features,
        predictions,
        num_classes=2,
    )
    raw_class_means = torch.stack(
        [
            features[predictions == class_index].abs().mean(dim=(0, 2))
            for class_index in range(2)
        ]
    )
    torch.testing.assert_close(
        templates.first_order,
        raw_class_means / raw_class_means.sum(dim=0),
    )
    assert templates.counts.tolist() == [2, 2]


def test_raw_scores_use_signed_test_feature_for_s2() -> None:
    train_features, predictions = _fixture()
    templates = build_official_score_templates(
        train_features,
        predictions,
        num_classes=2,
    )
    test_features = train_features[:2] * 0.7 - 0.2
    logits = torch.tensor(
        [
            [[-1.0, -2.0], [-3.0, -1.0]],
            [[-2.5, -1.5], [-1.0, -2.0]],
        ],
        dtype=torch.float64,
    )
    predicted = torch.tensor([0, 1])
    result = raw_official_scores(test_features, logits, predicted, templates)

    selected_logits = logits[torch.arange(2), predicted]
    activation = test_features.abs().mean(dim=1)
    expected_s1 = (selected_logits / activation / activation).mean(dim=1)
    selected_template = templates.first_order[predicted]
    expected_s2 = (test_features * selected_template[:, :, None]).mean(dim=(1, 2))
    expected_s3 = (
        official_g_p_pro(test_features) * templates.gram[predicted]
    ).sum(dim=(1, 2))
    torch.testing.assert_close(result.s1, expected_s1)
    torch.testing.assert_close(result.s2, expected_s2)
    torch.testing.assert_close(result.s3, expected_s3)

    abs_variant = (test_features.abs() * selected_template[:, :, None]).mean(dim=(1, 2))
    assert not torch.allclose(result.s2, abs_variant)


def test_score_normalization_is_float64_population_std_and_unit_sum() -> None:
    raw = torch.tensor(
        [
            [1.0, 2.0, 5.0],
            [2.0, 4.0, 1.0],
            [4.0, 3.0, 2.0],
            [8.0, 9.0, 7.0],
        ],
        dtype=torch.float32,
    )
    normalization = fit_score_normalization(raw)
    assert normalization.mean.dtype == torch.float64
    assert normalization.std.dtype == torch.float64
    torch.testing.assert_close(normalization.mean, raw.double().mean(dim=0))
    torch.testing.assert_close(
        normalization.std,
        raw.double().std(dim=0, correction=0),
    )
    output = standardize_and_integrate(raw[:2], normalization)
    expected = (raw[:2].double() - normalization.mean) / (
        normalization.std + 1.0e-8
    )
    torch.testing.assert_close(output.standardized, expected)
    torch.testing.assert_close(output.integrated, expected.sum(dim=1))


@pytest.mark.parametrize(
    ("mean", "std", "epsilon", "minimum"),
    [
        ([float("nan"), 0.0, 0.0], [1.0, 1.0, 1.0], 1.0e-8, 1.0e-12),
        ([0.0, 0.0, 0.0], [1.0, -1.0, 1.0], 1.0e-8, 1.0e-12),
        ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.0, 1.0e-12),
        ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 1.0e-8, 0.0),
    ],
)
def test_score_normalization_application_rejects_invalid_checkpoint_contract(
    mean: list[float],
    std: list[float],
    epsilon: float,
    minimum: float,
) -> None:
    normalization = OfficialScoreNormalization(
        mean=torch.tensor(mean, dtype=torch.float64),
        std=torch.tensor(std, dtype=torch.float64),
        epsilon=epsilon,
        min_std=minimum,
    )
    with pytest.raises(ValueError, match="normalization|contract"):
        standardize_and_integrate(torch.ones(2, 3), normalization)


def test_pcssr_pair_scores_are_symmetric_under_view_swap() -> None:
    train_features, predictions = _fixture()
    templates = build_official_score_templates(
        train_features,
        predictions,
        num_classes=2,
    )
    train_logits = torch.tensor(
        [
            [[-1.0, -2.0], [-3.0, -1.0]],
            [[-2.5, -1.5], [-1.0, -2.0]],
            [[-1.5, -2.0], [-2.0, -1.0]],
            [[-2.0, -3.0], [-1.0, -1.5]],
        ],
        dtype=torch.float64,
    )
    train_raw = raw_official_scores(
        train_features,
        train_logits,
        predictions,
        templates,
    )
    normalization = fit_score_normalization(train_raw)
    view_features = train_features.reshape(2, 2, 2, 2)
    view_logits = train_logits.reshape(2, 2, 2, 2)
    view_probabilities = torch.softmax(view_logits, dim=2).mean(dim=-1)
    original = official_pcssr_pair_scores(
        view_features,
        view_logits,
        view_probabilities,
        templates,
        normalization,
    )
    swapped = official_pcssr_pair_scores(
        view_features.flip(1),
        view_logits.flip(1),
        view_probabilities.flip(1),
        templates,
        normalization,
    )
    torch.testing.assert_close(original.pair_probabilities, swapped.pair_probabilities)
    assert torch.equal(original.predicted_class, swapped.predicted_class)
    for rule in original.knownness_by_rule:
        torch.testing.assert_close(
            original.knownness_by_rule[rule],
            swapped.knownness_by_rule[rule],
        )
        torch.testing.assert_close(
            original.unknown_scores_by_rule[rule],
            -original.knownness_by_rule[rule],
        )


def test_matched_linear_pair_score_is_negative_max_pair_probability() -> None:
    view_logits = torch.tensor(
        [
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[2.0, 1.0], [0.0, 0.0]],
            ]
        ]
    )
    view_probabilities = torch.softmax(view_logits, dim=2).mean(dim=-1)
    result = matched_linear_pair_output(view_logits, view_probabilities)
    expected_pair = view_probabilities.mean(dim=1)
    torch.testing.assert_close(result.pair_probabilities, expected_pair)
    torch.testing.assert_close(result.max_pair_probability, expected_pair.max(dim=1).values)
    torch.testing.assert_close(result.unknown_score, -result.max_pair_probability)


def test_template_construction_rejects_empty_predicted_class() -> None:
    features, _ = _fixture()
    with pytest.raises(ValueError, match="class 1 is empty"):
        build_official_score_templates(
            features,
            torch.zeros(4, dtype=torch.long),
            num_classes=2,
        )
