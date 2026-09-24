from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

Redistribution = Literal["permitted", "restricted", "prohibited", "unknown"]
ReviewStatus = Literal["pending", "approved", "rejected"]


class DatasetRegistryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    source_url: HttpUrl | None = None
    upstream_revision: str = Field(min_length=1)
    license: str = Field(min_length=1)
    redistribution: Redistribution
    access_requirements: str | None = None
    contains_clinical_text: bool
    contains_phi: bool
    patient_or_entity_key: str | None = None
    intended_uses: list[str] = Field(default_factory=list)
    forbidden_uses: list[str] = Field(default_factory=list)
    split_strategy: str = Field(min_length=1)
    transforms: list[str] = Field(default_factory=list)
    derived_artifacts: list[str] = Field(default_factory=list)
    review_status: ReviewStatus = "pending"
    reviewer: str | None = None
    review_date: str | None = None

    @model_validator(mode="after")
    def approved_entry_has_review_metadata(self) -> DatasetRegistryEntry:
        if self.review_status == "approved" and (self.reviewer is None or self.review_date is None):
            raise ValueError("approved registry entries require reviewer and review_date")
        return self


class DatasetRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.1"] = "0.1"
    entries: list[DatasetRegistryEntry]

    @model_validator(mode="after")
    def unique_ids(self) -> DatasetRegistry:
        ids = [entry.id for entry in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("dataset registry ids must be unique")
        return self


def load_registry(path: str | Path) -> DatasetRegistry:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return DatasetRegistry.model_validate(payload)
