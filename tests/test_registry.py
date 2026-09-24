from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from gaxbench.registry import DatasetRegistry, DatasetRegistryEntry, load_registry


def entry(**overrides: object) -> DatasetRegistryEntry:
    payload: dict[str, object] = {
        "id": "synthetic-smoke",
        "name": "Synthetic smoke fixture",
        "source_url": None,
        "upstream_revision": "1",
        "license": "CC0-1.0",
        "redistribution": "permitted",
        "access_requirements": None,
        "contains_clinical_text": False,
        "contains_phi": False,
        "patient_or_entity_key": "source_id",
        "intended_uses": ["unit tests"],
        "forbidden_uses": ["clinical claims"],
        "split_strategy": "synthetic fixture only",
        "transforms": [],
        "derived_artifacts": [],
        "review_status": "pending",
        "reviewer": None,
        "review_date": None,
    }
    payload.update(overrides)
    return DatasetRegistryEntry.model_validate(payload)


def test_registry_rejects_duplicate_ids() -> None:
    with pytest.raises(ValidationError, match="unique"):
        DatasetRegistry(entries=[entry(), entry()])


def test_approved_entry_requires_review_metadata() -> None:
    with pytest.raises(ValidationError, match="reviewer"):
        entry(review_status="approved")


def test_registry_loads_strict_json(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    payload = DatasetRegistry(entries=[entry()]).model_dump(mode="json")
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_registry(path)
    assert loaded.schema_version == "0.1"
    assert loaded.entries[0].id == "synthetic-smoke"


def test_repository_registry_is_valid() -> None:
    registry = load_registry(Path(__file__).parents[1] / "registry" / "datasets.json")
    assert registry.entries[0].review_status == "approved"
    assert registry.entries[0].contains_phi is False
