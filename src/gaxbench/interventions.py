from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from gaxbench.provenance import canonical_json_sha256
from gaxbench.schema import BenchmarkItem, Prediction, StrictModel, validate_prediction_against_item

ExpectedRelation = Literal["same", "flip", "directional-only", "abstain"]
Materiality = Literal["material", "irrelevant"]
Direction = Literal["increase", "decrease", "same"]

_EPS = 1e-12


class InterventionPair(StrictModel):
    id: str = Field(min_length=1)
    base_item_id: str = Field(min_length=1)
    intervention_item_id: str = Field(min_length=1)
    family: str = Field(min_length=1)
    materiality: Materiality
    expected_relation: ExpectedRelation
    changed_path: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    target_action: str | None = None
    target_direction: Direction | None = None
    expected_sufficiency_direction: Direction | None = None
    expected_evidence_support_direction: Direction | None = None

    @model_validator(mode="after")
    def validate_pair_contract(self) -> InterventionPair:
        if self.base_item_id == self.intervention_item_id:
            raise ValueError("base_item_id and intervention_item_id must differ")
        if self.materiality == "irrelevant" and self.expected_relation != "same":
            raise ValueError("irrelevant interventions must use expected_relation='same'")
        if self.expected_relation == "directional-only":
            if self.target_action is None or self.target_direction is None:
                raise ValueError(
                    "directional-only interventions require target_action and target_direction"
                )
        if self.target_direction is not None and self.target_action is None:
            raise ValueError("target_direction requires target_action")
        return self


class InterventionManifest(StrictModel):
    schema_version: Literal["0.1"] = "0.1"
    pairs: list[InterventionPair] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_pair_ids(self) -> InterventionManifest:
        ids = [pair.id for pair in self.pairs]
        if len(ids) != len(set(ids)):
            raise ValueError("intervention pair ids must be unique")
        return self


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    family: str
    materiality: Materiality
    expected_relation: ExpectedRelation
    base_item_id: str
    intervention_item_id: str
    base_correct: bool
    intervention_correct: bool
    base_predicted_action: str
    intervention_predicted_action: str
    top1_agreement: bool
    total_variation: float
    jensen_shannon: float
    max_probability_delta: float
    target_action_delta: float | None
    target_direction_success: bool | None
    abstain_success: bool | None
    sufficiency_delta: float | None
    sufficiency_direction_success: bool | None
    evidence_support_delta: float | None
    evidence_support_direction_success: bool | None


@dataclass(frozen=True)
class InterventionMetrics:
    n_pairs: int
    base_accuracy: float
    intervention_accuracy: float
    paired_robust_accuracy: float
    mean_total_variation: float
    mean_jensen_shannon: float
    max_probability_delta: float
    flip_pairs: int
    base_correct_flip_pairs: int
    bias_trap_rate: float | None
    required_flip_success_rate: float | None
    same_pairs: int
    same_top1_agreement: float | None
    stability_violation_rate: float | None
    directional_pairs: int
    directional_success_rate: float | None
    abstain_pairs: int
    abstain_success_rate: float | None
    sufficiency_direction_pairs: int
    sufficiency_direction_success_rate: float | None
    evidence_support_direction_pairs: int
    evidence_support_direction_success_rate: float | None


@dataclass(frozen=True)
class InterventionEvaluation:
    manifest_sha256: str
    stability_tv_threshold: float
    metrics: InterventionMetrics
    per_family: dict[str, InterventionMetrics]
    records: tuple[PairRecord, ...]


def load_intervention_manifest(path: str | Path) -> InterventionManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return InterventionManifest.model_validate(payload)


def intervention_manifest_sha256(manifest: InterventionManifest) -> str:
    return canonical_json_sha256(manifest.model_dump(mode="json"))


def total_variation_distance(
    first: dict[str, float],
    second: dict[str, float],
) -> float:
    _require_same_action_keys(first, second)
    return 0.5 * math.fsum(abs(first[key] - second[key]) for key in sorted(first))


def jensen_shannon_divergence(
    first: dict[str, float],
    second: dict[str, float],
) -> float:
    """Jensen-Shannon divergence using natural logarithms.

    With normalized probability distributions this lies in [0, ln(2)].
    """
    _require_same_action_keys(first, second)
    divergence = 0.0
    for key in sorted(first):
        p = first[key]
        q = second[key]
        midpoint = 0.5 * (p + q)
        if p > 0.0:
            divergence += 0.5 * p * math.log(p / midpoint)
        if q > 0.0:
            divergence += 0.5 * q * math.log(q / midpoint)
    return max(0.0, divergence)


def evaluate_interventions(
    items: Sequence[BenchmarkItem],
    predictions: Sequence[Prediction],
    manifest: InterventionManifest,
    *,
    stability_tv_threshold: float = 0.05,
) -> InterventionEvaluation:
    if not items:
        raise ValueError("items must not be empty")
    if not predictions:
        raise ValueError("predictions must not be empty")
    if not 0.0 <= stability_tv_threshold <= 1.0:
        raise ValueError("stability_tv_threshold must be in [0, 1]")

    item_map = _unique_item_map(items)
    prediction_map = _unique_prediction_map(predictions)
    referenced_ids = {
        item_id
        for pair in manifest.pairs
        for item_id in (pair.base_item_id, pair.intervention_item_id)
    }

    missing_items = sorted(referenced_ids - set(item_map))
    extra_items = sorted(set(item_map) - referenced_ids)
    if missing_items or extra_items:
        raise ValueError(
            f"manifest/item ids mismatch: missing={missing_items}, extra={extra_items}"
        )

    missing_predictions = sorted(referenced_ids - set(prediction_map))
    extra_predictions = sorted(set(prediction_map) - referenced_ids)
    if missing_predictions or extra_predictions:
        raise ValueError(
            "manifest/prediction ids mismatch: "
            f"missing={missing_predictions}, extra={extra_predictions}"
        )

    records: list[PairRecord] = []
    for pair in manifest.pairs:
        base = item_map[pair.base_item_id]
        intervention = item_map[pair.intervention_item_id]
        base_prediction = prediction_map[base.id]
        intervention_prediction = prediction_map[intervention.id]
        _validate_pair_lineage(pair, base, intervention)
        validate_prediction_against_item(base, base_prediction)
        validate_prediction_against_item(intervention, intervention_prediction)
        records.append(
            _evaluate_pair(pair, base, intervention, base_prediction, intervention_prediction)
        )

    grouped: dict[str, list[PairRecord]] = defaultdict(list)
    for record in records:
        grouped[record.family].append(record)

    return InterventionEvaluation(
        manifest_sha256=intervention_manifest_sha256(manifest),
        stability_tv_threshold=stability_tv_threshold,
        metrics=_aggregate(records, stability_tv_threshold),
        per_family={
            family: _aggregate(family_records, stability_tv_threshold)
            for family, family_records in sorted(grouped.items())
        },
        records=tuple(records),
    )


def _evaluate_pair(
    pair: InterventionPair,
    base: BenchmarkItem,
    intervention: BenchmarkItem,
    base_prediction: Prediction,
    intervention_prediction: Prediction,
) -> PairRecord:
    if base.gold is None or base.gold.action is None:
        raise ValueError(f"base item {base.id!r} must have a gold action")
    if intervention.gold is None or intervention.gold.action is None:
        raise ValueError(f"intervention item {intervention.id!r} must have a gold action")

    base_action = _top_action(base_prediction)
    intervention_action = _top_action(intervention_prediction)
    base_correct = base_action == base.gold.action
    intervention_correct = intervention_action == intervention.gold.action

    tv = total_variation_distance(base_prediction.probabilities, intervention_prediction.probabilities)
    js = jensen_shannon_divergence(
        base_prediction.probabilities,
        intervention_prediction.probabilities,
    )
    max_delta = max(
        abs(base_prediction.probabilities[key] - intervention_prediction.probabilities[key])
        for key in base_prediction.probabilities
    )

    target_delta: float | None = None
    target_success: bool | None = None
    if pair.target_action is not None:
        if pair.target_action not in base_prediction.probabilities:
            raise ValueError(
                f"pair {pair.id!r} target_action {pair.target_action!r} is not in action set"
            )
        target_delta = (
            intervention_prediction.probabilities[pair.target_action]
            - base_prediction.probabilities[pair.target_action]
        )
        if pair.target_direction is not None:
            target_success = _direction_matches(target_delta, pair.target_direction)

    sufficiency_delta, sufficiency_success = _optional_score_delta(
        pair.id,
        "information_sufficiency",
        base_prediction.information_sufficiency,
        intervention_prediction.information_sufficiency,
        pair.expected_sufficiency_direction,
    )
    evidence_delta, evidence_success = _optional_score_delta(
        pair.id,
        "evidence_support",
        base_prediction.evidence_support,
        intervention_prediction.evidence_support,
        pair.expected_evidence_support_direction,
    )

    abstain_success = (
        intervention_prediction.abstain if pair.expected_relation == "abstain" else None
    )

    return PairRecord(
        pair_id=pair.id,
        family=pair.family,
        materiality=pair.materiality,
        expected_relation=pair.expected_relation,
        base_item_id=base.id,
        intervention_item_id=intervention.id,
        base_correct=base_correct,
        intervention_correct=intervention_correct,
        base_predicted_action=base_action,
        intervention_predicted_action=intervention_action,
        top1_agreement=base_action == intervention_action,
        total_variation=tv,
        jensen_shannon=js,
        max_probability_delta=max_delta,
        target_action_delta=target_delta,
        target_direction_success=target_success,
        abstain_success=abstain_success,
        sufficiency_delta=sufficiency_delta,
        sufficiency_direction_success=sufficiency_success,
        evidence_support_delta=evidence_delta,
        evidence_support_direction_success=evidence_success,
    )


def _aggregate(
    records: Sequence[PairRecord],
    stability_tv_threshold: float,
) -> InterventionMetrics:
    if not records:
        raise ValueError("records must not be empty")

    flip = [record for record in records if record.expected_relation == "flip"]
    base_correct_flip = [record for record in flip if record.base_correct]
    same = [record for record in records if record.expected_relation == "same"]
    directional = [
        record for record in records if record.target_direction_success is not None
    ]
    abstain = [record for record in records if record.abstain_success is not None]
    sufficiency = [
        record for record in records if record.sufficiency_direction_success is not None
    ]
    evidence = [
        record for record in records if record.evidence_support_direction_success is not None
    ]

    return InterventionMetrics(
        n_pairs=len(records),
        base_accuracy=_mean_bool(record.base_correct for record in records),
        intervention_accuracy=_mean_bool(record.intervention_correct for record in records),
        paired_robust_accuracy=_mean_bool(
            record.base_correct and record.intervention_correct for record in records
        ),
        mean_total_variation=math.fsum(record.total_variation for record in records)
        / len(records),
        mean_jensen_shannon=math.fsum(record.jensen_shannon for record in records)
        / len(records),
        max_probability_delta=max(record.max_probability_delta for record in records),
        flip_pairs=len(flip),
        base_correct_flip_pairs=len(base_correct_flip),
        bias_trap_rate=(
            None
            if not base_correct_flip
            else _mean_bool(not record.intervention_correct for record in base_correct_flip)
        ),
        required_flip_success_rate=(
            None
            if not flip
            else _mean_bool(
                record.base_correct
                and record.intervention_correct
                and not record.top1_agreement
                for record in flip
            )
        ),
        same_pairs=len(same),
        same_top1_agreement=(
            None if not same else _mean_bool(record.top1_agreement for record in same)
        ),
        stability_violation_rate=(
            None
            if not same
            else _mean_bool(record.total_variation > stability_tv_threshold for record in same)
        ),
        directional_pairs=len(directional),
        directional_success_rate=(
            None
            if not directional
            else _mean_bool(bool(record.target_direction_success) for record in directional)
        ),
        abstain_pairs=len(abstain),
        abstain_success_rate=(
            None
            if not abstain
            else _mean_bool(bool(record.abstain_success) for record in abstain)
        ),
        sufficiency_direction_pairs=len(sufficiency),
        sufficiency_direction_success_rate=(
            None
            if not sufficiency
            else _mean_bool(
                bool(record.sufficiency_direction_success) for record in sufficiency
            )
        ),
        evidence_support_direction_pairs=len(evidence),
        evidence_support_direction_success_rate=(
            None
            if not evidence
            else _mean_bool(
                bool(record.evidence_support_direction_success) for record in evidence
            )
        ),
    )


def _validate_pair_lineage(
    pair: InterventionPair,
    base: BenchmarkItem,
    intervention: BenchmarkItem,
) -> None:
    if base.source_id != intervention.source_id:
        raise ValueError(f"pair {pair.id!r} must preserve source_id lineage")
    if base.split != intervention.split:
        raise ValueError(f"pair {pair.id!r} must remain in one split")
    if (
        base.counterfactual_group is None
        or intervention.counterfactual_group is None
        or base.counterfactual_group != intervention.counterfactual_group
    ):
        raise ValueError(f"pair {pair.id!r} must share a non-null counterfactual_group")
    base_actions = {action.id for action in base.actions}
    intervention_actions = {action.id for action in intervention.actions}
    if base_actions != intervention_actions:
        raise ValueError(
            f"pair {pair.id!r} must use the same action ids in P06 v0.1"
        )
    if pair.expected_relation == "same" and base.gold is not None and intervention.gold is not None:
        if base.gold.action != intervention.gold.action:
            raise ValueError(f"pair {pair.id!r} expected same action but gold actions differ")
    if pair.expected_relation == "flip" and base.gold is not None and intervention.gold is not None:
        if base.gold.action == intervention.gold.action:
            raise ValueError(f"pair {pair.id!r} expected a flip but gold actions are identical")


def _unique_item_map(items: Sequence[BenchmarkItem]) -> dict[str, BenchmarkItem]:
    result: dict[str, BenchmarkItem] = {}
    for item in items:
        if item.id in result:
            raise ValueError(f"duplicate benchmark item id: {item.id!r}")
        result[item.id] = item
    return result


def _unique_prediction_map(predictions: Sequence[Prediction]) -> dict[str, Prediction]:
    result: dict[str, Prediction] = {}
    for prediction in predictions:
        if prediction.item_id in result:
            raise ValueError(f"duplicate prediction id: {prediction.item_id!r}")
        result[prediction.item_id] = prediction
    return result


def _top_action(prediction: Prediction) -> str:
    return max(prediction.probabilities.items(), key=lambda pair: (pair[1], pair[0]))[0]


def _direction_matches(delta: float, direction: Direction) -> bool:
    if direction == "increase":
        return delta > _EPS
    if direction == "decrease":
        return delta < -_EPS
    return abs(delta) <= _EPS


def _optional_score_delta(
    pair_id: str,
    score_name: str,
    base_score: float | None,
    intervention_score: float | None,
    expected_direction: Direction | None,
) -> tuple[float | None, bool | None]:
    if expected_direction is None:
        if base_score is None or intervention_score is None:
            return None, None
        return intervention_score - base_score, None
    if base_score is None or intervention_score is None:
        raise ValueError(
            f"pair {pair_id!r} declares {score_name} direction but a score is missing"
        )
    delta = intervention_score - base_score
    return delta, _direction_matches(delta, expected_direction)


def _require_same_action_keys(first: dict[str, float], second: dict[str, float]) -> None:
    if set(first) != set(second):
        raise ValueError("probability distributions must use identical action keys")


def _mean_bool(values: Sequence[bool] | object) -> float:
    sequence = list(values)  # type: ignore[arg-type]
    if not sequence:
        raise ValueError("boolean sequence must not be empty")
    return sum(bool(value) for value in sequence) / len(sequence)
