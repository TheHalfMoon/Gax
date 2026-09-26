from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from gaxbench.fhir_adapters import (
    FHIR_AGENTBENCH_REVISION,
    MEDAGENTBENCH_REVISION,
    ExternalFHIRTask,
    external_task_to_case,
    load_external_fhir_export,
)

FIXTURES = Path(__file__).parent / "fixtures"


def payload(benchmark: str = "FHIR-AgentBench") -> dict[str, object]:
    revision = (
        FHIR_AGENTBENCH_REVISION
        if benchmark == "FHIR-AgentBench"
        else MEDAGENTBENCH_REVISION
    )
    version = "R4" if benchmark == "FHIR-AgentBench" else "R5"
    return {
        "benchmark": benchmark,
        "benchmark_revision": revision,
        "task_id": "synthetic-task",
        "source_id": "synthetic-source",
        "split": "validation",
        "task_family": "resource-routing",
        "source_fhir_version": version,
        "resources": [{"resourceType": "Patient", "id": "synthetic"}],
        "allowed_actions": [
            {
                "id": "read-patient",
                "kind": "read",
                "description": "Read patient",
                "resource_type": "Patient",
                "query_template": "GET /Patient/synthetic",
            }
        ],
        "gold_action": "read-patient",
        "sufficient": True,
        "dataset_revision": "synthetic-v1",
        "dataset_license": "CC0-1.0",
        "redistribution": "permitted",
        "access_requirements": None,
    }


def test_fhir_agentbench_must_preserve_r4_identity() -> None:
    value = payload()
    value["source_fhir_version"] = "R5"
    with pytest.raises(ValidationError, match="R4"):
        ExternalFHIRTask.model_validate(value)


def test_frozen_benchmark_revision_is_enforced() -> None:
    value = payload()
    value["benchmark_revision"] = "0" * 40
    with pytest.raises(ValidationError, match="frozen P07 revision"):
        ExternalFHIRTask.model_validate(value)


def test_benchmark_revision_requires_lowercase_hex_sha() -> None:
    value = payload()
    value["benchmark_revision"] = "Z" * 40
    with pytest.raises(ValidationError, match="lowercase hexadecimal"):
        ExternalFHIRTask.model_validate(value)


def test_local_export_rejects_final_test_by_default(tmp_path: Path) -> None:
    value = payload()
    value["split"] = "test"
    path = tmp_path / "export.jsonl"
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="final-test"):
        load_external_fhir_export(path, benchmark="FHIR-AgentBench")


def test_local_export_rejects_wrong_benchmark(tmp_path: Path) -> None:
    path = tmp_path / "export.jsonl"
    path.write_text(json.dumps(payload("MedAgentBench")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected benchmark"):
        load_external_fhir_export(path, benchmark="FHIR-AgentBench")


def test_external_task_conversion_preserves_provenance() -> None:
    task = ExternalFHIRTask.model_validate(payload())
    case = external_task_to_case(task)
    assert case.source_fhir_version == "R4"
    assert case.provenance.dataset == "FHIR-AgentBench"
    assert case.provenance.revision == "synthetic-v1"
    assert FHIR_AGENTBENCH_REVISION in (case.provenance.source_url or "")


def test_medagentbench_protocol_uses_its_frozen_revision() -> None:
    task = ExternalFHIRTask.model_validate(payload("MedAgentBench"))
    case = external_task_to_case(task)
    assert task.benchmark_revision == MEDAGENTBENCH_REVISION
    assert case.provenance.dataset == "MedAgentBench"


def test_repository_fhir_agentbench_fixture_qualifies_local_export_protocol() -> None:
    tasks = load_external_fhir_export(
        FIXTURES / "p07_fhir_agentbench_export.jsonl",
        benchmark="FHIR-AgentBench",
    )
    assert len(tasks) == 1
    assert tasks[0].source_fhir_version == "R4"
    assert tasks[0].redistribution == "permitted"


def test_repository_medagentbench_fixture_qualifies_local_export_protocol() -> None:
    tasks = load_external_fhir_export(
        FIXTURES / "p07_medagentbench_export.jsonl",
        benchmark="MedAgentBench",
    )
    assert len(tasks) == 1
    assert tasks[0].benchmark_revision == MEDAGENTBENCH_REVISION
