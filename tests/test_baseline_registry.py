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
