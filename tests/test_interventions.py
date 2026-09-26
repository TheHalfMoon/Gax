from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from gaxbench.interventions import (
    InterventionManifest,
    InterventionPair,
    evaluate_interventions,
    jensen_shannon_divergence,
    total_variation_distance,
)
from gaxbench.schema import Action, BenchmarkItem, Gold, Prediction, Provenance


def _item(
    item_id: str,
    source_id: str,
    group: str,
    gold_action: str,
    *,
    split: str = "validation",
    state_value: str = "base",
) -> BenchmarkItem:
    return BenchmarkItem(
        id=item_id,
        source_id=source_id,
        split=split,  # type: ignore[arg-type]
        task_family="p06-synthetic",
        state={"signal": state_value},
        actions=[
            Action(id="left", description="Abstract left route"),
            Action(id="right", description="Abstract right route"),
        ],
        gold=Gold(action=gold_action, sufficient=True),
        counterfactual_group=group,
        provenance=Provenance(
            dataset="p06-synthetic",
            revision="1",
            license="CC0-1.0",
            transform_revision="1",
        ),
    )


def _prediction(
    item_id: str,
    left: float,
    right: float,
    *,
    sufficiency: float | None = 0.9,
    evidence_support: float | None = 0.8,
    abstain: bool = False,
) -> Prediction:
    return Prediction(
        item_id=item_id,
        probabilities={"left": left, "right": right},
        information_sufficiency=sufficiency,
        evidence_support=evidence_support,
        abstain=abstain,
    )


def test_probability_distances_have_frozen_conventions() -> None:
    first = {"left": 1.0, "right": 0.0}
    second = {"left": 0.0, "right": 1.0}
    assert total_variation_distance(first, second) == pytest.approx(1.0)
    assert jensen_shannon_divergence(first, second) == pytest.approx(math.log(2.0))
    assert total_variation_distance(first, first) == pytest.approx(0.0)
    assert jensen_shannon_divergence(first, first) == pytest.approx(0.0)


def test_irrelevant_intervention_must_expect_same_action() -> None:
    with pytest.raises(ValidationError, match="irrelevant"):
        InterventionPair(
            id="bad",
            base_item_id="a",
            intervention_item_id="b",
            family="formatting",
            materiality="irrelevant",
            expected_relation="flip",
            changed_path="/state/format",
            rationale="abstract test",
        )


def test_material_flip_and_bias_trap_rate_are_pair_conditioned() -> None:
    items = [
        _item("a0", "source-a", "group-a", "left"),
        _item("a1", "source-a", "group-a", "right", state_value="changed"),
        _item("b0", "source-b", "group-b", "left"),
        _item("b1", "source-b", "group-b", "right", state_value="changed"),
    ]
    predictions = [
        _prediction("a0", 0.9, 0.1),
        _prediction("a1", 0.8, 0.2),
        _prediction("b0", 0.2, 0.8),
        _prediction("b1", 0.1, 0.9),
    ]
    manifest = InterventionManifest(
        pairs=[
            InterventionPair(
                id="pair-a",
                base_item_id="a0",
                intervention_item_id="a1",
                family="material-state",
                materiality="material",
                expected_relation="flip",
                changed_path="/state/signal",
                rationale="abstract material change",
            ),
            InterventionPair(
                id="pair-b",
                base_item_id="b0",
                intervention_item_id="b1",
                family="material-state",
                materiality="material",
                expected_relation="flip",
                changed_path="/state/signal",
                rationale="abstract material change",
            ),
        ]
    )
    result = evaluate_interventions(items, predictions, manifest)
    assert result.metrics.flip_pairs == 2
    assert result.metrics.base_correct_flip_pairs == 1
    assert result.metrics.bias_trap_rate == pytest.approx(1.0)
    assert result.metrics.required_flip_success_rate == pytest.approx(0.0)


def test_irrelevant_pair_measures_stability_without_forcing_abstention() -> None:
    items = [
        _item("base", "source", "group", "left"),
        _item("edit", "source", "group", "left", state_value="format-only"),
    ]
    predictions = [
        _prediction("base", 0.80, 0.20, sufficiency=0.85),
        _prediction("edit", 0.78, 0.22, sufficiency=0.84),
    ]
    manifest = InterventionManifest(
        pairs=[
            InterventionPair(
                id="irrelevant",
                base_item_id="base",
                intervention_item_id="edit",
                family="irrelevant-edit",
                materiality="irrelevant",
                expected_relation="same",
                changed_path="/state/format",
                rationale="determinable case remains determinable",
            )
        ]
    )
    result = evaluate_interventions(
        items,
        predictions,
        manifest,
        stability_tv_threshold=0.05,
    )
    assert result.metrics.same_top1_agreement == pytest.approx(1.0)
    assert result.metrics.stability_violation_rate == pytest.approx(0.0)
    assert result.records[0].abstain_success is None


def test_directional_evidence_and_sufficiency_changes_are_explicit() -> None:
    items = [
        _item("base", "source", "group", "left"),
        _item("evidence", "source", "group", "left", state_value="evidence-removed"),
    ]
    predictions = [
        _prediction("base", 0.8, 0.2, sufficiency=0.9, evidence_support=0.85),
        _prediction("evidence", 0.7, 0.3, sufficiency=0.4, evidence_support=0.20),
    ]
    manifest = InterventionManifest(
        pairs=[
            InterventionPair(
                id="evidence-removal",
                base_item_id="base",
                intervention_item_id="evidence",
                family="evidence-removal",
                materiality="material",
                expected_relation="directional-only",
                changed_path="/evidence",
                rationale="abstract evidence removal",
                target_action="left",
                target_direction="decrease",
                expected_sufficiency_direction="decrease",
                expected_evidence_support_direction="decrease",
            )
        ]
    )
    result = evaluate_interventions(items, predictions, manifest)
    record = result.records[0]
    assert record.target_direction_success is True
    assert record.sufficiency_direction_success is True
    assert record.evidence_support_direction_success is True


def test_declared_score_direction_requires_both_scores() -> None:
    items = [
        _item("base", "source", "group", "left"),
        _item("edit", "source", "group", "left", state_value="changed"),
    ]
    predictions = [
        _prediction("base", 0.8, 0.2, evidence_support=None),
        _prediction("edit", 0.7, 0.3, evidence_support=None),
    ]
    manifest = InterventionManifest(
        pairs=[
            InterventionPair(
                id="missing-score",
                base_item_id="base",
                intervention_item_id="edit",
                family="evidence-removal",
                materiality="material",
                expected_relation="same",
                changed_path="/evidence",
                rationale="abstract test",
                expected_evidence_support_direction="decrease",
            )
        ]
    )
    with pytest.raises(ValueError, match="evidence_support"):
        evaluate_interventions(items, predictions, manifest)


def test_pair_cannot_cross_source_or_split_boundaries() -> None:
    manifest = InterventionManifest(
        pairs=[
            InterventionPair(
                id="lineage",
                base_item_id="base",
                intervention_item_id="edit",
                family="material-state",
                materiality="material",
                expected_relation="same",
                changed_path="/state/signal",
                rationale="abstract test",
            )
        ]
    )
    base = _item("base", "source-a", "group", "left")
    other_source = _item("edit", "source-b", "group", "left")
    predictions = [_prediction("base", 0.8, 0.2), _prediction("edit", 0.8, 0.2)]
    with pytest.raises(ValueError, match="source_id"):
        evaluate_interventions([base, other_source], predictions, manifest)

    other_split = _item("edit", "source-a", "group", "left", split="calibration")
    with pytest.raises(ValueError, match="one split"):
        evaluate_interventions([base, other_split], predictions, manifest)


def test_manifest_and_predictions_must_match_exact_pair_denominator() -> None:
    items = [
        _item("base", "source", "group", "left"),
        _item("edit", "source", "group", "left"),
    ]
    manifest = InterventionManifest(
        pairs=[
            InterventionPair(
                id="pair",
                base_item_id="base",
                intervention_item_id="edit",
                family="same",
                materiality="irrelevant",
                expected_relation="same",
                changed_path="/state/format",
                rationale="abstract test",
            )
        ]
    )
    with pytest.raises(ValueError, match="prediction ids mismatch"):
        evaluate_interventions(items, [_prediction("base", 0.8, 0.2)], manifest)
