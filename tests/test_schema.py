from __future__ import annotations

import pytest
from pydantic import ValidationError

from gaxbench.schema import Action, BenchmarkItem, Gold, Prediction, Provenance


def item() -> BenchmarkItem:
    return BenchmarkItem(
        id="x",
        source_id="s",
        split="test",
        task_family="unit",
        state={"x": 1},
        actions=[Action(id="a", description="A"), Action(id="b", description="B")],
        gold=Gold(action="a", sufficient=True),
        provenance=Provenance(
            dataset="synthetic", revision="1", license="CC0-1.0", transform_revision="1"
        ),
    )


def test_prediction_requires_normalized_probabilities() -> None:
    with pytest.raises(ValidationError):
        Prediction(item_id="x", probabilities={"a": 0.8, "b": 0.3})


def test_item_rejects_unknown_gold_action() -> None:
    with pytest.raises(ValidationError):
        BenchmarkItem(
            id="x",
            source_id="s",
            split="test",
            task_family="unit",
            state={},
            actions=[Action(id="a", description="A")],
            gold=Gold(action="b"),
            provenance=Provenance(
                dataset="synthetic", revision="1", license="CC0-1.0", transform_revision="1"
            ),
        )


def test_item_is_immutable() -> None:
    value = item()
    with pytest.raises(ValidationError):
        value.id = "changed"  # type: ignore[misc]
