from __future__ import annotations

import json
from pathlib import Path

from gaxbench.schema import BenchmarkItem, Prediction

ROOT = Path(__file__).parents[1]


def test_item_json_schema_snapshot_matches_model() -> None:
    path = ROOT / "schemas" / "gaxbench-item-v0.1.json"
    expected = json.loads(path.read_text(encoding="utf-8"))
    assert expected == BenchmarkItem.model_json_schema()


def test_prediction_json_schema_snapshot_matches_model() -> None:
    expected = json.loads(
        (ROOT / "schemas" / "gaxbench-prediction-v0.1.json").read_text(encoding="utf-8")
    )
    assert expected == Prediction.model_json_schema()


def test_dataset_registry_json_schema_snapshot_matches_model() -> None:
    from gaxbench.registry import DatasetRegistry

    expected = json.loads(
        (ROOT / "schemas" / "gaxbench-dataset-registry-v0.1.json").read_text(encoding="utf-8")
    )
    assert expected == DatasetRegistry.model_json_schema()
