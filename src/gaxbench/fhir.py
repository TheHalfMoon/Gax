from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from gaxbench.schema import Action, BenchmarkItem, Gold, Provenance, Split, StrictModel

FHIRVersion = Literal["R4", "R5"]
FHIRRepresentation = Literal[
    "canonical-structured",
    "canonical-with-narrative",
    "source-order-json",
    "flat-text",
]
FHIRActionKind = Literal[
    "read",
    "search",
    "select-resource-type",
    "sufficient",
    "need-more-information",
    "abstain",
    "verify-support",
]

FHIR_CANONICAL_RELEASE = "R5"
FHIR_CANONICAL_VERSION = "5.0.0"
FHIR_REPRESENTATION_REVISION = "gax-fhir-v0.1"


class FHIRReadOnlyAction(StrictModel):
    id: str = Field(min_length=1)
    kind: FHIRActionKind
    description: str = Field(min_length=1)
    resource_type: str | None = None
    query_template: str | None = None

    @model_validator(mode="after")
    def validate_action(self) -> FHIRReadOnlyAction:
        if self.kind in {"read", "search", "select-resource-type"}:
            if not self.resource_type:
                raise ValueError(f"{self.kind} actions require resource_type")
        if self.query_template is not None:
            prefix = self.query_template.lstrip().split(maxsplit=1)[0].upper()
            if prefix in {"POST", "PUT", "PATCH", "DELETE"}:
                raise ValueError("P07 actions are read-only; write HTTP methods are forbidden")
        return self


class FHIRDecisionCase(StrictModel):
    id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    split: Split
    task_family: str = Field(min_length=1)
    source_fhir_version: FHIRVersion
    resources: list[dict[str, JsonValue]] = Field(min_length=1)
    actions: list[FHIRReadOnlyAction] = Field(min_length=1)
    gold_action: str | None = None
    sufficient: bool | None = None
    provenance: Provenance

    @field_validator("resources")
    @classmethod
    def validate_resources(
        cls, value: list[dict[str, JsonValue]]
    ) -> list[dict[str, JsonValue]]:
        for resource in value:
            validate_fhir_resource(resource)
        return value

    @model_validator(mode="after")
    def validate_case(self) -> FHIRDecisionCase:
        action_ids = [action.id for action in self.actions]
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("FHIR action ids must be unique")
        if self.gold_action is not None and self.gold_action not in set(action_ids):
            raise ValueError("gold_action must reference an allowed FHIR action")
        return self


def validate_fhir_resource(resource: Mapping[str, JsonValue]) -> None:
    resource_type = resource.get("resourceType")
    if not isinstance(resource_type, str) or not resource_type.strip():
        raise ValueError("FHIR resource requires a non-empty resourceType")
    _validate_strict_json(resource)

    contained = resource.get("contained")
    if contained is not None:
        if not isinstance(contained, list):
            raise ValueError("FHIR contained must be a list when present")
        for nested in contained:
            if not isinstance(nested, dict):
                raise ValueError("FHIR contained members must be resource objects")
            validate_fhir_resource(nested)

    if resource_type == "Bundle":
        entries = resource.get("entry")
        if entries is not None:
            if not isinstance(entries, list):
                raise ValueError("Bundle.entry must be a list when present")
            for entry in entries:
                if not isinstance(entry, dict):
                    raise ValueError("Bundle.entry members must be objects")
                nested = entry.get("resource")
                if nested is not None:
                    if not isinstance(nested, dict):
                        raise ValueError("Bundle.entry.resource must be an object")
                    validate_fhir_resource(nested)


def canonicalize_fhir_resource(
    resource: Mapping[str, JsonValue],
    *,
    include_narrative: bool = False,
) -> dict[str, JsonValue]:
    validate_fhir_resource(resource)
    canonical = _canonical_json_value(
        dict(resource),
        strip_resource_narrative=not include_narrative,
    )
    if not isinstance(canonical, dict):
        raise AssertionError("FHIR resource canonicalization must produce an object")
    return canonical


def render_fhir_resource(
    resource: Mapping[str, JsonValue],
    *,
    representation: FHIRRepresentation = "canonical-structured",
) -> str:
    validate_fhir_resource(resource)
    if representation == "canonical-structured":
        value: JsonValue = canonicalize_fhir_resource(resource, include_narrative=False)
    elif representation == "canonical-with-narrative":
        value = canonicalize_fhir_resource(resource, include_narrative=True)
    elif representation == "source-order-json":
        value = dict(resource)
    elif representation == "flat-text":
        canonical = canonicalize_fhir_resource(resource, include_narrative=False)
        return "\n".join(_flatten_json(canonical))
    else:
        raise AssertionError(f"unhandled representation: {representation}")
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def fhir_case_to_benchmark_item(
    case: FHIRDecisionCase,
    *,
    representation: FHIRRepresentation = "canonical-structured",
) -> BenchmarkItem:
    rendered = [
        render_fhir_resource(resource, representation=representation)
        for resource in case.resources
    ]
    state: dict[str, JsonValue] = {
        "fhir_source_version": case.source_fhir_version,
        "fhir_canonical_reference": FHIR_CANONICAL_VERSION,
        "fhir_representation": representation,
        "fhir_representation_revision": FHIR_REPRESENTATION_REVISION,
        "rendered_resources": rendered,
    }
    actions = [
        Action(
            id=action.id,
            description=action.description,
            metadata={
                "fhir_action_kind": action.kind,
                "resource_type": action.resource_type,
                "query_template": action.query_template,
                "read_only": True,
            },
        )
        for action in case.actions
    ]
    return BenchmarkItem(
        id=case.id,
        source_id=case.source_id,
        split=case.split,
        task_family=case.task_family,
        state=state,
        actions=actions,
        gold=Gold(action=case.gold_action, sufficient=case.sufficient),
        provenance=case.provenance,
    )


def _canonical_json_value(
    value: JsonValue,
    *,
    strip_resource_narrative: bool,
) -> JsonValue:
    if isinstance(value, dict):
        is_resource = isinstance(value.get("resourceType"), str)
        return {
            key: _canonical_json_value(
                value[key],
                strip_resource_narrative=strip_resource_narrative,
            )
            for key in sorted(value)
            if not (strip_resource_narrative and is_resource and key == "text")
        }
    if isinstance(value, list):
        return [
            _canonical_json_value(
                item,
                strip_resource_narrative=strip_resource_narrative,
            )
            for item in value
        ]
    return value


def _flatten_json(value: JsonValue, path: str = "$") -> list[str]:
    if isinstance(value, dict):
        rows: list[str] = []
        for key in sorted(value):
            rows.extend(_flatten_json(value[key], f"{path}.{key}"))
        return rows
    if isinstance(value, list):
        rows = []
        for index, item in enumerate(value):
            rows.extend(_flatten_json(item, f"{path}[{index}]"))
        return rows
    return [f"{path}={json.dumps(value, ensure_ascii=False, allow_nan=False)}"]


def _validate_strict_json(value: object) -> None:
    try:
        json.dumps(value, allow_nan=False, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("FHIR content must be strict JSON without NaN/Infinity") from exc
    _reject_nonfinite(value)


def _reject_nonfinite(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("FHIR content must not contain NaN/Infinity")
    if isinstance(value, dict):
        for child in value.values():
            _reject_nonfinite(child)
    elif isinstance(value, list):
        for child in value:
            _reject_nonfinite(child)
