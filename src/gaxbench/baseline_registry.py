from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

AccessStatus = Literal["available", "conditional", "unavailable", "unknown"]
QualificationStatus = Literal[
    "qualified-offline",
    "adapter-qualified",
    "planned",
    "external-pending",
    "blocked",
]


class BaselineRegistryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    baseline_class: str = Field(min_length=1)
    source_url: HttpUrl | None = None
    source_revision: str | None = None
    license: str = Field(min_length=1)
    access_status: AccessStatus
    qualification_status: QualificationStatus
    reason: str | None = None
    intended_role: str = Field(min_length=1)

    @model_validator(mode="after")
    def blocked_or_conditional_needs_reason(self) -> BaselineRegistryEntry:
        if self.access_status in {"conditional", "unavailable"} and not self.reason:
            raise ValueError("conditional/unavailable baseline entries require a reason")
        if self.qualification_status == "blocked" and not self.reason:
            raise ValueError("blocked baseline entries require a reason")
        return self


class BaselineRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.1"] = "0.1"
    entries: list[BaselineRegistryEntry]

    @model_validator(mode="after")
    def unique_ids(self) -> BaselineRegistry:
        ids = [entry.id for entry in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("baseline registry ids must be unique")
        return self


def load_baseline_registry(path: str | Path) -> BaselineRegistry:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return BaselineRegistry.model_validate(payload)
