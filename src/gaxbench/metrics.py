from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from gaxbench.schema import (
    BenchmarkItem,
    EvaluationRecord,
    Prediction,
    validate_prediction_against_item,
)

_EPS = 1e-12


@dataclass(frozen=True)
class RiskCoveragePoint:
    coverage: float
    risk: float
    selected: int


@dataclass(frozen=True)
class ActionMetrics:
    n: int
    accuracy: float
    nll: float
    brier: float
    ece: float
    actual_coverage: float
    committed_accuracy: float | None
    committed_risk: float | None
    selection_score_source: str
    aurc: float
    risk_at_50: float
    risk_at_80: float
    risk_at_90: float


@dataclass(frozen=True)
class AbstentionMetrics:
    n: int
    insufficient_n: int
    sufficient_n: int
    precision: float | None
    recall: float | None
    f1: float | None
    unsafe_commit_rate: float | None
    over_abstain_rate: float | None
    sufficiency_nll: float | None
    sufficiency_brier: float | None
    sufficiency_ece: float | None


def evaluate_action_predictions(
    items: Sequence[BenchmarkItem],
    predictions: Sequence[Prediction],
    *,
    ece_bins: int = 15,
) -> tuple[ActionMetrics, list[EvaluationRecord]]:
    if not items:
        raise ValueError("items must not be empty")
    if ece_bins < 2:
        raise ValueError("ece_bins must be >= 2")

    item_ids = [item.id for item in items]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("item ids must be unique")

    prediction_map: dict[str, Prediction] = {}
    for prediction in predictions:
        if prediction.item_id in prediction_map:
            raise ValueError(f"duplicate prediction for item {prediction.item_id!r}")
        prediction_map[prediction.item_id] = prediction

    if set(prediction_map) != set(item_ids):
        missing = sorted(set(item_ids) - set(prediction_map))
        extra = sorted(set(prediction_map) - set(item_ids))
        raise ValueError(f"prediction ids mismatch: missing={missing}, extra={extra}")

    records: list[EvaluationRecord] = []
    nll_terms: list[float] = []
    brier_terms: list[float] = []

    all_have_sufficiency = all(
        prediction_map[item.id].information_sufficiency is not None for item in items
    )
    score_source = "information_sufficiency" if all_have_sufficiency else "max_action_probability"

    for item in items:
        if item.gold is None or item.gold.action is None:
            raise ValueError(f"item {item.id!r} has no gold action; refusing silent exclusion")
        prediction = prediction_map[item.id]
        validate_prediction_against_item(item, prediction)

        predicted_action, confidence = max(
            prediction.probabilities.items(), key=lambda pair: (pair[1], pair[0])
        )
        gold_action = item.gold.action
        correct = predicted_action == gold_action
        gold_probability = max(prediction.probabilities[gold_action], _EPS)
        nll_terms.append(-math.log(gold_probability))

        brier = 0.0
        for action in item.actions:
            target = 1.0 if action.id == gold_action else 0.0
            brier += (prediction.probabilities[action.id] - target) ** 2
        brier_terms.append(brier)

        if all_have_sufficiency:
            assert prediction.information_sufficiency is not None
            selection_score = prediction.information_sufficiency
        else:
            selection_score = confidence

        records.append(
            EvaluationRecord(
                item_id=item.id,
                gold_action=gold_action,
                predicted_action=predicted_action,
                action_confidence=confidence,
                correct=correct,
                abstain=prediction.abstain,
                selection_score=selection_score,
                gold_sufficient=item.gold.sufficient,
                information_sufficiency=prediction.information_sufficiency,
            )
        )

    accuracy = sum(record.correct for record in records) / len(records)
    confidences = [record.action_confidence for record in records]
    correctness = [record.correct for record in records]
    ece = expected_calibration_error(confidences, correctness, bins=ece_bins)

    committed = [record for record in records if not record.abstain]
    actual_coverage = len(committed) / len(records)
    committed_accuracy = (
        sum(record.correct for record in committed) / len(committed) if committed else None
    )
    committed_risk = None if committed_accuracy is None else 1.0 - committed_accuracy

    curve = risk_coverage_curve(
        [record.correct for record in records],
        [record.selection_score for record in records],
    )

    metrics = ActionMetrics(
        n=len(records),
        accuracy=accuracy,
        nll=sum(nll_terms) / len(nll_terms),
        brier=sum(brier_terms) / len(brier_terms),
        ece=ece,
        actual_coverage=actual_coverage,
        committed_accuracy=committed_accuracy,
        committed_risk=committed_risk,
        selection_score_source=score_source,
        aurc=sum(point.risk for point in curve) / len(curve),
        risk_at_50=risk_at_coverage(curve, 0.50),
        risk_at_80=risk_at_coverage(curve, 0.80),
        risk_at_90=risk_at_coverage(curve, 0.90),
    )
    return metrics, records


def evaluate_abstention(
    records: Sequence[EvaluationRecord], *, ece_bins: int = 15
) -> AbstentionMetrics:
    if not records:
        raise ValueError("records must not be empty")
    if any(record.gold_sufficient is None for record in records):
        raise ValueError("all records need gold_sufficient; refusing silent exclusion")

    tp = fp = fn = tn = 0
    sufficiency_scores: list[float] = []
    sufficiency_labels: list[bool] = []
    all_scores = all(record.information_sufficiency is not None for record in records)

    for record in records:
        assert record.gold_sufficient is not None
        should_abstain = not record.gold_sufficient
        if record.abstain and should_abstain:
            tp += 1
        elif record.abstain and not should_abstain:
            fp += 1
        elif not record.abstain and should_abstain:
            fn += 1
        else:
            tn += 1

        if all_scores:
            assert record.information_sufficiency is not None
            sufficiency_scores.append(record.information_sufficiency)
            sufficiency_labels.append(record.gold_sufficient)

    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = (
        None
        if precision is None or recall is None or precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )
    unsafe_commit_rate = _safe_ratio(fn, tp + fn)
    over_abstain_rate = _safe_ratio(fp, fp + tn)

    if all_scores:
        sufficiency_nll = sum(
            _binary_nll(score, label)
            for score, label in zip(sufficiency_scores, sufficiency_labels, strict=True)
        ) / len(records)
        sufficiency_brier = sum(
            (score - float(label)) ** 2
            for score, label in zip(sufficiency_scores, sufficiency_labels, strict=True)
        ) / len(records)
        sufficiency_ece = expected_calibration_error(
            sufficiency_scores, sufficiency_labels, bins=ece_bins
        )
    else:
        sufficiency_nll = None
        sufficiency_brier = None
        sufficiency_ece = None

    return AbstentionMetrics(
        n=len(records),
        insufficient_n=tp + fn,
        sufficient_n=fp + tn,
        precision=precision,
        recall=recall,
        f1=f1,
        unsafe_commit_rate=unsafe_commit_rate,
        over_abstain_rate=over_abstain_rate,
        sufficiency_nll=sufficiency_nll,
        sufficiency_brier=sufficiency_brier,
        sufficiency_ece=sufficiency_ece,
    )


def expected_calibration_error(
    confidences: Sequence[float], correctness: Sequence[bool], *, bins: int = 15
) -> float:
    if len(confidences) != len(correctness) or not confidences:
        raise ValueError("confidences and correctness must be non-empty and equally sized")
    if bins < 2:
        raise ValueError("bins must be >= 2")

    bin_confidence = [0.0] * bins
    bin_correct = [0.0] * bins
    bin_count = [0] * bins

    for confidence, correct in zip(confidences, correctness, strict=True):
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1]")
        index = min(int(confidence * bins), bins - 1)
        bin_confidence[index] += confidence
        bin_correct[index] += float(correct)
        bin_count[index] += 1

    total = len(confidences)
    ece = 0.0
    for conf_sum, correct_sum, count in zip(
        bin_confidence, bin_correct, bin_count, strict=True
    ):
        if count == 0:
            continue
        mean_confidence = conf_sum / count
        mean_accuracy = correct_sum / count
        ece += (count / total) * abs(mean_accuracy - mean_confidence)
    return ece


def risk_coverage_curve(
    correctness: Sequence[bool], selection_scores: Sequence[float]
) -> list[RiskCoveragePoint]:
    if len(correctness) != len(selection_scores) or not correctness:
        raise ValueError("correctness and selection_scores must be non-empty and equally sized")

    indexed = []
    for index, (correct, score) in enumerate(zip(correctness, selection_scores, strict=True)):
        if not math.isfinite(score):
            raise ValueError("selection scores must be finite")
        indexed.append((score, index, correct))
    indexed.sort(key=lambda row: (-row[0], row[1]))

    points: list[RiskCoveragePoint] = []
    errors = 0
    total = len(indexed)
    for rank, (_, _, correct) in enumerate(indexed, start=1):
        errors += int(not correct)
        points.append(RiskCoveragePoint(coverage=rank / total, risk=errors / rank, selected=rank))
    return points


def risk_at_coverage(curve: Sequence[RiskCoveragePoint], target: float) -> float:
    if not curve:
        raise ValueError("curve must not be empty")
    if not 0.0 < target <= 1.0:
        raise ValueError("target coverage must be in (0, 1]")
    for point in curve:
        if point.coverage + 1e-12 >= target:
            return point.risk
    return curve[-1].risk


def _binary_nll(probability: float, label: bool) -> float:
    clipped = min(1.0 - _EPS, max(_EPS, probability))
    return -math.log(clipped if label else 1.0 - clipped)


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator
