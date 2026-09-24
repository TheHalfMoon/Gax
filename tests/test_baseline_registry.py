from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from gaxbench.baseline_registry import (
    BaselineRegistry,
    BaselineRegistryEntry,
    load_baseline_registry,
)

ROOT = Path(__file__).parents[1]


def test_repository_baseline_registry_is_valid() -> None:
    registry = load_baseline_registry(ROOT / "registry" / "baselines.json")
    assert len(registry.entries) >= 8
    jev = next(entry for entry in registry.entries if entry.id == "jev")
    assert jev.access_status == "conditional"
    assert jev.qualification_status == "blocked"
    assert jev.reason

    qualified = {
        entry.id: entry
        for entry in registry.entries
        if entry.qualification_status == "adapter-qualified"
    }
    assert {"clm", "laya", "decider"} <= set(qualified)
    for entry in qualified.values():
        assert entry.source_revision
        assert entry.license == "Apache-2.0"
        assert entry.reason

    required_external = {
        "clm",
        "laya",
        "decider",
        "restricted-logit",
        "clinical-encoder",
        "structured-output-llm",
    }
    by_id = {entry.id: entry for entry in registry.entries}
    for baseline_id in required_external:
        entry = by_id[baseline_id]
        assert entry.source_url is not None
        assert entry.source_revision
        assert entry.license not in {
            "model-license-review-required",
            "terms-review-required",
        }
        assert entry.access_status != "unknown"
        assert entry.qualification_status != "planned"

    clinical = by_id["clinical-encoder"]
    assert clinical.source_revision == "5e17e2f25260b6993e0fb60485f94678ff29779a"
    assert clinical.license == "MIT"

    structured = by_id["structured-output-llm"]
    assert structured.source_revision == "a7b0d22b993d71000cf2eadfb37222a67cee521e"
    assert structured.license == "Apache-2.0"


def test_conditional_registry_entry_requires_reason() -> None:
    with pytest.raises(ValidationError, match="require a reason"):
        BaselineRegistryEntry(
            id="x",
            name="X",
            baseline_class="external",
            source_url=None,
            source_revision=None,
            license="review-required",
            access_status="conditional",
            qualification_status="blocked",
            reason=None,
            intended_role="test",
        )


def test_baseline_registry_schema_snapshot_matches_model() -> None:
    expected = json.loads(
        (ROOT / "schemas" / "gaxbench-baseline-registry-v0.1.json").read_text(
            encoding="utf-8"
        )
    )
    assert expected == BaselineRegistry.model_json_schema()
