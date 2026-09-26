from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from gaxbench.fhir import (
    FHIR_CANONICAL_VERSION,
    FHIRDecisionCase,
    FHIRReadOnlyAction,
    canonicalize_fhir_resource,
    fhir_case_to_benchmark_item,
    render_fhir_resource,
    validate_fhir_resource,
)
from gaxbench.metrics import evaluate_abstention, evaluate_action_predictions
from gaxbench.schema import Prediction, Provenance


def provenance() -> Provenance:
    return Provenance(
        dataset="synthetic-fhir",
        revision="1",
        license="CC0-1.0",
        transform_revision="p07-test",
    )


def test_canonical_render_sorts_object_keys_but_preserves_array_order() -> None:
    resource = {
        "resourceType": "Patient",
        "name": [{"family": "Second"}, {"family": "First"}],
        "id": "synthetic-1",
    }
    rendered = render_fhir_resource(resource)
    parsed = json.loads(rendered)
    assert list(parsed) == ["id", "name", "resourceType"]
    assert [row["family"] for row in parsed["name"]] == ["Second", "First"]


def test_narrative_is_excluded_by_default_and_opt_in_is_explicit() -> None:
    resource = {
        "resourceType": "Observation",
        "id": "synthetic",
        "text": {"status": "generated", "div": "synthetic narrative"},
        "status": "final",
    }
    without = render_fhir_resource(resource, representation="canonical-structured")
    with_narrative = render_fhir_resource(
        resource,
        representation="canonical-with-narrative",
    )
    assert "synthetic narrative" not in without
    assert "synthetic narrative" in with_narrative


def test_nested_resource_narrative_is_excluded_without_dropping_other_text() -> None:
    bundle = {
        "resourceType": "Bundle",
        "entry": [
            {
                "resource": {
                    "resourceType": "Observation",
                    "text": {"status": "generated", "div": "nested narrative"},
                    "valueCodeableConcept": {"text": "meaningful coded text"},
                }
            }
        ],
    }
    rendered = render_fhir_resource(bundle, representation="canonical-structured")
    assert "nested narrative" not in rendered
    assert "meaningful coded text" in rendered


def test_source_order_is_a_real_representation_control() -> None:
    resource = {"resourceType": "Patient", "id": "synthetic"}
    canonical = render_fhir_resource(resource, representation="canonical-structured")
    source_order = render_fhir_resource(resource, representation="source-order-json")
    assert canonical != source_order
    assert json.loads(canonical) == json.loads(source_order)


def test_flat_text_is_deterministic() -> None:
    resource = {"resourceType": "Patient", "id": "synthetic"}
    first = render_fhir_resource(resource, representation="flat-text")
    second = render_fhir_resource(resource, representation="flat-text")
    assert first == second
    assert "$.resourceType=\"Patient\"" in first


def test_missing_resource_type_fails_closed() -> None:
    with pytest.raises(ValueError, match="resourceType"):
        validate_fhir_resource({"id": "synthetic"})


def test_nonfinite_fhir_value_fails_closed() -> None:
    with pytest.raises(ValueError, match="strict JSON|NaN"):
        canonicalize_fhir_resource(
            {"resourceType": "Observation", "valueDecimal": float("nan")}
        )


def test_bundle_nested_resource_is_validated() -> None:
    bad_bundle = {
        "resourceType": "Bundle",
        "entry": [{"resource": {"id": "missing-type"}}],
    }
    with pytest.raises(ValueError, match="resourceType"):
        validate_fhir_resource(bad_bundle)


def test_contained_resource_is_validated() -> None:
    resource = {
        "resourceType": "Patient",
        "contained": [{"id": "missing-type"}],
    }
    with pytest.raises(ValueError, match="resourceType"):
        validate_fhir_resource(resource)


def test_write_method_is_forbidden() -> None:
    with pytest.raises(ValidationError, match="read-only"):
        FHIRReadOnlyAction(
            id="write",
            kind="search",
            description="forbidden",
            resource_type="Observation",
            query_template="POST /Observation",
        )


def test_case_conversion_preserves_explicit_r4_identity() -> None:
    case = FHIRDecisionCase(
        id="case-r4",
        source_id="source-r4",
        split="validation",
        task_family="resource-routing",
        source_fhir_version="R4",
        resources=[{"resourceType": "Patient", "id": "synthetic"}],
        actions=[
            FHIRReadOnlyAction(
                id="read-patient",
                kind="read",
                description="Read patient",
                resource_type="Patient",
            )
        ],
        gold_action="read-patient",
        sufficient=True,
        provenance=provenance(),
    )
    item = fhir_case_to_benchmark_item(case)
    assert item.state["fhir_source_version"] == "R4"
    assert item.state["fhir_canonical_reference"] == FHIR_CANONICAL_VERSION
    assert item.actions[0].metadata["read_only"] is True


def test_action_requires_resource_type_for_retrieval() -> None:
    with pytest.raises(ValidationError, match="resource_type"):
        FHIRReadOnlyAction(
            id="missing-resource",
            kind="search",
            description="Search without a resource type",
        )


def test_fhir_cases_reuse_action_and_abstention_metrics() -> None:
    actions = [
        FHIRReadOnlyAction(
            id="read-patient",
            kind="read",
            description="Read patient",
            resource_type="Patient",
        ),
        FHIRReadOnlyAction(
            id="need-more",
            kind="need-more-information",
            description="Need more information",
        ),
    ]
    cases = [
        FHIRDecisionCase(
            id="fhir-sufficient",
            source_id="source-1",
            split="validation",
            task_family="routing",
            source_fhir_version="R5",
            resources=[{"resourceType": "Patient", "id": "synthetic-1"}],
            actions=actions,
            gold_action="read-patient",
            sufficient=True,
            provenance=provenance(),
        ),
        FHIRDecisionCase(
            id="fhir-insufficient",
            source_id="source-2",
            split="validation",
            task_family="routing",
            source_fhir_version="R5",
            resources=[{"resourceType": "Patient", "id": "synthetic-2"}],
            actions=actions,
            gold_action="need-more",
            sufficient=False,
            provenance=provenance(),
        ),
    ]
    items = [fhir_case_to_benchmark_item(case) for case in cases]
    predictions = [
        Prediction(
            item_id="fhir-sufficient",
            probabilities={"read-patient": 0.9, "need-more": 0.1},
            information_sufficiency=0.9,
            abstain=False,
        ),
        Prediction(
            item_id="fhir-insufficient",
            probabilities={"read-patient": 0.2, "need-more": 0.8},
            information_sufficiency=0.1,
            abstain=True,
        ),
    ]
    action_metrics, records = evaluate_action_predictions(items, predictions)
    abstention_metrics = evaluate_abstention(records)
    assert action_metrics.accuracy == 1.0
    assert abstention_metrics.unsafe_commit_rate == 0.0
    assert abstention_metrics.over_abstain_rate == 0.0
