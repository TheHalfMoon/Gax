from __future__ import annotations

from pathlib import Path

import pytest

from gaxbench.baselines import (
    LexicographicBaselineAdapter,
    PredictionFileAdapter,
    UniformBaselineAdapter,
)
from gaxbench.io import load_items

FIXTURES = Path(__file__).parent / "fixtures"


def test_uniform_baseline_is_deterministic_and_normalized() -> None:
    item = load_items(FIXTURES / "items.jsonl")[0]
    adapter = UniformBaselineAdapter()
    first = adapter.predict(item)
    second = adapter.predict(item)
    assert first == second
    assert set(first.probabilities) == {action.id for action in item.actions}
    assert sum(first.probabilities.values()) == pytest.approx(1.0)
    assert len(set(first.probabilities.values())) == 1


def test_lexicographic_baseline_selects_smallest_action_id() -> None:
    item = load_items(FIXTURES / "items.jsonl")[0]
    prediction = LexicographicBaselineAdapter().predict(item)
    selected = min(prediction.probabilities, key=lambda key: (-prediction.probabilities[key], key))
    assert selected == min(action.id for action in item.actions)


def test_prediction_file_adapter_requires_exact_item_set() -> None:
    items = load_items(FIXTURES / "items.jsonl")
    adapter = PredictionFileAdapter(
        FIXTURES / "predictions.jsonl",
        name="fixture",
        adapter_version="1",
    )
    adapter.prepare(items)
    with pytest.raises(ValueError, match="item ids mismatch"):
        adapter.prepare(items[:-1])


def test_prediction_file_adapter_detects_mutation(tmp_path: Path) -> None:
    items = load_items(FIXTURES / "items.jsonl")
    copied = tmp_path / "predictions.jsonl"
    copied.write_text(
        (FIXTURES / "predictions.jsonl").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    adapter = PredictionFileAdapter(copied, name="fixture", adapter_version="1")
    copied.write_text(copied.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after adapter initialization"):
        adapter.prepare(items)
