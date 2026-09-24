from __future__ import annotations

import json
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

Split = Literal["train", "validation", "calibration", "test"]
EvidenceRelation = Literal["support", "contradict", "irrelevant", "unknown"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Action(StrictModel):
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        _validate_strict_json(value, "action.metadata")
        return value


class Evidence(StrictModel):
    id: str = Field(min_length=1)
    relation: EvidenceRelation = "unknown"
    text: str | None = None
    structured: JsonValue | None = None
    source_ref: str | None = None

    @field_validator("structured")
    @classmethod
    def validate_structured(cls, value: JsonValue | None) -> JsonValue | None:
        if value is not None:
            _validate_strict_json(value, "evidence.structured")
        return value

    @model_validator(mode="after")
    def require_content(self) -> Evidence:
        if self.text is None and self.structured is None:
            raise ValueError("evidence must contain text or structured content")
        return self


class Provenance(StrictModel):
    dataset: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    license: str = Field(min_length=1)
    transform_revision: str = Field(min_length=1)
    source_url: str | None = None
    access_requirements: str | None = None


class Gold(StrictModel):
    action: str | None = None
    sufficient: bool | None = None
    action_probabilities: dict[str, float] | None = None

    @field_validator("action_probabilities")
    @classmethod
    def validate_distribution(cls, value: dict[str, float] | None) -> dict[str, float] | None:
        if value is None:
            return None
        _validate_probability_distribution(value, "gold.action_probabilities")
        return value


class BenchmarkItem(StrictModel):
    id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    split: Split
    task_family: str = Field(min_length=1)
    state: JsonValue
    actions: list[Action] = Field(min_length=1)
    gold: Gold | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    counterfactual_group: str | None = None
    provenance: Provenance

    @model_validator(mode="after")
    def validate_item(self) -> BenchmarkItem:
        action_ids = [action.id for action in self.actions]
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("action ids must be unique within an item")

        evidence_ids = [evidence.id for evidence in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence ids must be unique within an item")

        if self.gold is not None:
            valid_actions = set(action_ids)
            if self.gold.action is not None and self.gold.action not in valid_actions:
                raise ValueError("gold action must reference one of the item actions")
            if self.gold.action_probabilities is not None:
                if set(self.gold.action_probabilities) != valid_actions:
                    raise ValueError(
                        "gold action_probabilities keys must exactly match the item action ids"
                    )

        _validate_strict_json(self.state, "state")
        return self


class Prediction(StrictModel):
    item_id: str = Field(min_length=1)
    probabilities: dict[str, float]
    evidence_support: float | None = None
    information_sufficiency: float | None = None
    abstain: bool = False
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("probabilities")
    @classmethod
    def validate_probabilities(cls, value: dict[str, float]) -> dict[str, float]:
        _validate_probability_distribution(value, "prediction.probabilities")
        return value

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        _validate_strict_json(value, "prediction.metadata")
        return value

    @field_validator("evidence_support", "information_sufficiency")
    @classmethod
    def validate_unit_interval(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("probability-like scores must be finite and in [0, 1]")
        return value


class EvaluationRecord(StrictModel):
    item_id: str
    gold_action: str
    predicted_action: str
    action_confidence: float
    correct: bool
    abstain: bool
    selection_score: float
    gold_sufficient: bool | None = None
    information_sufficiency: float | None = None


def validate_prediction_against_item(item: BenchmarkItem, prediction: Prediction) -> None:
    if item.id != prediction.item_id:
        raise ValueError(f"prediction item_id {prediction.item_id!r} != item id {item.id!r}")
    action_ids = {action.id for action in item.actions}
    if set(prediction.probabilities) != action_ids:
        missing = sorted(action_ids - set(prediction.probabilities))
        extra = sorted(set(prediction.probabilities) - action_ids)
        raise ValueError(f"prediction action keys mismatch: missing={missing}, extra={extra}")


def _validate_strict_json(value: object, name: str) -> None:
    try:
        json.dumps(value, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be strict JSON without NaN/Infinity") from exc


def _validate_probability_distribution(value: dict[str, float], name: str) -> None:
    if not value:
        raise ValueError(f"{name} must not be empty")
    for key, probability in value.items():
        if not key:
            raise ValueError(f"{name} contains an empty key")
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"{name}[{key!r}] must be finite and in [0, 1]")
    total = math.fsum(value.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{name} must sum to 1 within 1e-6; got {total:.12g}")
