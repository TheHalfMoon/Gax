from __future__ import annotations

import math
from pathlib import Path

import pytest

from gaxbench.io import load_items, load_predictions
from gaxbench.metrics import (
    evaluate_abstention,
    evaluate_action_predictions,
    expected_calibration_error,
    risk_at_coverage,
    risk_coverage_curve,
)
from gaxbench.schema import Prediction

FIXTURES = Path(__file__).parent / "fixtures"


def test_fixture_metrics_are_deterministic() -> None:
    items = load_items(FIXTURES / "items.jsonl")
    predictions = load_predictions(FIXTURES / "predictions.jsonl")
    metrics, records = evaluate_action_predictions(items, predictions, ece_bins=10)

    assert metrics.n == 3
    assert metrics.accuracy == pytest.approx(2 / 3)
    assert metrics.actual_coverage == pytest.approx(2 / 3)
    assert metrics.committed_accuracy == pytest.approx(1.0)
    assert metrics.committed_risk == pytest.approx(0.0)
    assert metrics.selection_score_source == "information_sufficiency"
    assert metrics.risk_at_50 == pytest.approx(0.0)
    assert metrics.risk_at_80 == pytest.approx(1 / 3)
    assert metrics.risk_at_90 == pytest.approx(1 / 3)

    abstention = evaluate_abstention(records, ece_bins=10)
    assert abstention.precision == pytest.approx(1.0)
    assert abstention.recall == pytest.approx(1.0)
    assert abstention.unsafe_commit_rate == pytest.approx(0.0)
    assert abstention.over_abstain_rate == pytest.approx(0.0)
    assert abstention.sufficiency_brier is not None


def test_perfect_predictions_have_zero_nll_and_brier_at_probability_one() -> None:
    items = load_items(FIXTURES / "items.jsonl")[:2]
    predictions = [
        Prediction(
            item_id="case-1",
            probabilities={"routine": 0.0, "urgent": 1.0},
            information_sufficiency=1.0,
        ),
        Prediction(
            item_id="case-2",
            probabilities={"routine": 1.0, "urgent": 0.0},
            information_sufficiency=1.0,
        ),
    ]
    metrics, _ = evaluate_action_predictions(items, predictions, ece_bins=10)
    assert metrics.accuracy == 1.0
    assert metrics.nll == pytest.approx(0.0)
    assert metrics.brier == pytest.approx(0.0)
    assert metrics.ece == pytest.approx(0.0)


def test_risk_coverage_uses_stable_score_order() -> None:
    curve = risk_coverage_curve(
        correctness=[True, False, True, False],
        selection_scores=[0.9, 0.1, 0.8, 0.2],
    )
    assert [point.risk for point in curve] == pytest.approx([0.0, 0.0, 1 / 3, 0.5])
    assert risk_at_coverage(curve, 0.5) == pytest.approx(0.0)
    assert sum(point.risk for point in curve) / len(curve) == pytest.approx(5 / 24)


def test_ece_rejects_nonfinite_confidence() -> None:
    with pytest.raises(ValueError):
        expected_calibration_error([math.nan], [True])


def test_missing_prediction_is_not_silently_dropped() -> None:
    items = load_items(FIXTURES / "items.jsonl")
    predictions = load_predictions(FIXTURES / "predictions.jsonl")[:-1]
    with pytest.raises(ValueError, match="missing"):
        evaluate_action_predictions(items, predictions)
