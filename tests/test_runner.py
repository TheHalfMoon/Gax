from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from gaxbench.baselines import AdapterIdentity
from gaxbench.io import load_items
from gaxbench.runner import run_baseline
from gaxbench.schema import BenchmarkItem, Prediction

FIXTURES = Path(__file__).parent / "fixtures"


class FailingAdapter:
    @property
    def identity(self) -> AdapterIdentity:
        return AdapterIdentity(name="failing", adapter_version="1", deterministic=True)

    def prepare(self, items: Sequence[BenchmarkItem]) -> None:
        if not items:
            raise ValueError("empty")

    def predict(self, item: BenchmarkItem) -> Prediction:
        if item.id == "case-2":
            raise RuntimeError("synthetic inference failure")
        probability = 1.0 / len(item.actions)
        return Prediction(
            item_id=item.id,
            probabilities={action.id: probability for action in item.actions},
        )


class PrepareFailureAdapter(FailingAdapter):
    def prepare(self, items: Sequence[BenchmarkItem]) -> None:
        raise RuntimeError("synthetic preparation failure")


def test_complete_uniform_run_produces_metrics() -> None:
    from gaxbench.baselines import UniformBaselineAdapter

    items = load_items(FIXTURES / "items.jsonl")
    result = run_baseline(items, UniformBaselineAdapter(), ece_bins=10)
    assert result.requested == 3
    assert result.completed == 3
    assert result.failed == 0
    assert result.action_metrics is not None
    assert result.evaluation_error is None
    assert len(result.timings) == 3


def test_failed_item_remains_visible_and_blocks_metrics() -> None:
    items = load_items(FIXTURES / "items.jsonl")
    result = run_baseline(items, FailingAdapter())
    assert result.requested == 3
    assert result.completed == 2
    assert result.failed == 1
    assert result.failures[0].item_id == "case-2"
    assert result.failures[0].stage == "predict"
    assert result.action_metrics is None
    assert len(result.timings) == 3


def test_prepare_failure_marks_every_requested_item_failed() -> None:
    items = load_items(FIXTURES / "items.jsonl")
    result = run_baseline(items, PrepareFailureAdapter())
    assert result.completed == 0
    assert result.failed == len(items)
    assert {failure.stage for failure in result.failures} == {"prepare"}
    assert result.timings == ()
