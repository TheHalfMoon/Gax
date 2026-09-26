from __future__ import annotations

import json
from pathlib import Path

from gaxbench.fhir import FHIR_CANONICAL_RELEASE, FHIR_CANONICAL_VERSION
from gaxbench.fhir_adapters import (
    FHIR_AGENTBENCH_REPOSITORY,
    FHIR_AGENTBENCH_REPOSITORY_LICENSE,
    FHIR_AGENTBENCH_REVISION,
    MEDAGENTBENCH_CODE_LICENSE,
    MEDAGENTBENCH_REPOSITORY,
    MEDAGENTBENCH_REVISION,
)

ROOT = Path(__file__).parents[1]


def test_p07_fhir_source_registry_matches_code_freeze() -> None:
    payload = json.loads((ROOT / "registry" / "fhir_sources.json").read_text(encoding="utf-8"))
    canonical = payload["canonical_fhir"]
    assert canonical["release"] == FHIR_CANONICAL_RELEASE
    assert canonical["version"] == FHIR_CANONICAL_VERSION

    benchmarks = {row["id"]: row for row in payload["benchmarks"]}
    medagent = benchmarks["medagentbench"]
    assert medagent["repository"] == MEDAGENTBENCH_REPOSITORY
    assert medagent["revision"] == MEDAGENTBENCH_REVISION
    assert medagent["code_license"] == MEDAGENTBENCH_CODE_LICENSE

    fhir_agent = benchmarks["fhir-agentbench"]
    assert fhir_agent["repository"] == FHIR_AGENTBENCH_REPOSITORY
    assert fhir_agent["revision"] == FHIR_AGENTBENCH_REVISION
    assert fhir_agent["repository_license"] == FHIR_AGENTBENCH_REPOSITORY_LICENSE
    assert fhir_agent["source_fhir_version"] == "R4"
    assert fhir_agent["founder_paid_cloud_required"] is False


def test_r4_compatibility_registry_forbids_silent_conversion() -> None:
    payload = json.loads((ROOT / "registry" / "fhir_sources.json").read_text(encoding="utf-8"))
    r4 = next(row for row in payload["compatibility"] if row["release"] == "R4")
    assert r4["mode"] == "source-native-explicit"
    assert r4["semantic_conversion_to_r5"] is False
