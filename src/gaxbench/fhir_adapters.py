from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from gaxbench.fhir import FHIRDecisionCase, FHIRReadOnlyAction, FHIRVersion
from gaxbench.schema import Provenance, Split, StrictModel

MEDAGENTBENCH_REPOSITORY = "stanfordmlgroup/MedAgentBench"
MEDAGENTBENCH_REVISION = "99260117137b09f04837a8c18d18a1107efa55ae"
MEDAGENTBENCH_CODE_LICENSE = "MIT"
FHIR_AGENTBENCH_REPOSITORY = "glee4810/FHIR-AgentBench"
FHIR_AGENTBENCH_REVISION = "bbb42909a5a7eb907d1cd91f72a560729e7037ea"
FHIR_AGENTBENCH_REPOSITORY_LICENSE = "CC-BY-4.0"

BenchmarkName = Literal["MedAgentBench", "FHIR-AgentBench"]


class ExternalFHIRTask(StrictModel):
    benchmark: BenchmarkName
    benchmark_revision: str = Field(min_length=40, max_length=40)
    task_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    split: Split
    task_family: str = Field(min_length=1)
    source_fhir_version: FHIRVersion
    resources: list[dict[str, JsonValue]] = Field(min_length=1)
    allowed_actions: list[FHIRReadOnlyAction] = Field(min_length=1)
    gold_action: str | None = None
    sufficient: bool | None = None
    dataset_revision: str = Field(min_length=1)
    dataset_license: str = Field(min_length=1)
    redistribution: Literal["permitted", "restricted", "unknown"]
    access_requirements: str | None = None

    @model_validator(mode="after")
    def freeze_source_identity(self) -> ExternalFHIRTask:
        expected = _expected_revision(self.benchmark)
        if self.benchmark_revision != expected:
            raise ValueError(
                f"{self.benchmark} revision must equal frozen P07 revision {expected}"
            )
        if self.benchmark == "FHIR-AgentBench" and self.source_fhir_version != "R4":
            raise ValueError("FHIR-AgentBench P07 exports must preserve source-native R4 identity")
        return self


def load_external_fhir_export(
    path: str | Path,
    *,
    benchmark: BenchmarkName,
    allow_test: bool = False,
) -> list[ExternalFHIRTask]:
    rows: list[ExternalFHIRTask] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                task = ExternalFHIRTask.model_validate(payload)
            except Exception as exc:  # noqa: BLE001 - preserve source line context
                raise ValueError(f"{path}:{line_number}: invalid external FHIR task: {exc}") from exc
            if task.benchmark != benchmark:
                raise ValueError(
                    f"{path}:{line_number}: expected benchmark {benchmark!r}, "
                    f"got {task.benchmark!r}"
                )
            if task.split == "test" and not allow_test:
                raise ValueError("P07 rejects final-test exports; final evaluation is P08 work")
            rows.append(task)
    if not rows:
        raise ValueError(f"{path}: no external FHIR tasks")
    return rows


def external_task_to_case(task: ExternalFHIRTask) -> FHIRDecisionCase:
    repository, code_license = _source_identity(task.benchmark)
    provenance = Provenance(
        dataset=task.benchmark,
        revision=task.dataset_revision,
        license=task.dataset_license,
        transform_revision="gax-p07-external-export-v0.1",
        source_url=f"https://github.com/{repository}/commit/{task.benchmark_revision}",
        access_requirements=task.access_requirements,
    )
    return FHIRDecisionCase(
        id=task.task_id,
        source_id=task.source_id,
        split=task.split,
        task_family=task.task_family,
        source_fhir_version=task.source_fhir_version,
        resources=task.resources,
        actions=task.allowed_actions,
        gold_action=task.gold_action,
        sufficient=task.sufficient,
        provenance=provenance,
    )


def _expected_revision(benchmark: BenchmarkName) -> str:
    if benchmark == "MedAgentBench":
        return MEDAGENTBENCH_REVISION
    return FHIR_AGENTBENCH_REVISION


def _source_identity(benchmark: BenchmarkName) -> tuple[str, str]:
    if benchmark == "MedAgentBench":
        return MEDAGENTBENCH_REPOSITORY, MEDAGENTBENCH_CODE_LICENSE
    return FHIR_AGENTBENCH_REPOSITORY, FHIR_AGENTBENCH_REPOSITORY_LICENSE
