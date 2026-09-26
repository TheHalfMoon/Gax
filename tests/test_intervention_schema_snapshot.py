from __future__ import annotations

import json
from pathlib import Path

from gaxbench.interventions import InterventionManifest

ROOT = Path(__file__).parents[1]


def test_intervention_manifest_schema_snapshot_matches_model() -> None:
    expected = json.loads(
        (ROOT / "schemas" / "gaxbench-intervention-manifest-v0.1.json").read_text(
            encoding="utf-8"
        )
    )
    assert expected == InterventionManifest.model_json_schema()
