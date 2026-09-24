from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter_ns

from gaxbench.baselines import AdapterIdentity, BaselineAdapter
from gaxbench.metrics import (
    AbstentionMetrics,
    ActionMetrics,
    evaluate_abstention,
    evaluate_action_predictions,
)
from gaxbench.schema import BenchmarkItem, Prediction, validate_prediction_against_item


@dataclass(frozen=True)
class InferenceFailure:
    item_id: str
    stage: str
    error_type: str
    message: str


@dataclass(frozen=True)
class InferenceTiming:
    item_id: str
    duration_ms: float


@dataclass(frozen=True)
class BaselineRunResult:
    identity: AdapterIdentity
    requested: int
    completed: int
    failed: int
    predictions: tuple[Prediction, ...]
    failures: tuple[InferenceFailure, ...]
    timings: tuple[InferenceTiming, ...]
    action_metrics: ActionMetrics | None
    abstention_metrics: AbstentionMetrics | None
    evaluation_error: str | None


def run_baseline(
    items: Sequence[BenchmarkItem],
    adapter: BaselineAdapter,
    *,
    ece_bins: int = 15,
) -> BaselineRunResult:
    if not items:
        raise ValueError("items must not be empty")

    item_ids = [item.id for item in items]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("item ids must be unique")

    try:
        adapter.prepare(items)
    except Exception as exc:  # noqa: BLE001 - preserve adapter failure evidence
        failures = tuple(
            InferenceFailure(
                item_id=item.id,
                stage="prepare",
                error_type=type(exc).__name__,
                message=str(exc),
            )
            for item in items
        )
        return BaselineRunResult(
            identity=adapter.identity,
            requested=len(items),
            completed=0,
            failed=len(items),
            predictions=(),
            failures=failures,
            timings=(),
            action_metrics=None,
            abstention_metrics=None,
            evaluation_error=None,
        )

    predictions: list[Prediction] = []
    failures_list: list[InferenceFailure] = []
    timings: list[InferenceTiming] = []

    for item in items:
        started = perf_counter_ns()
        try:
            prediction = adapter.predict(item)
            validate_prediction_against_item(item, prediction)
            predictions.append(prediction)
        except Exception as exc:  # noqa: BLE001 - every failed item remains visible
            failures_list.append(
                InferenceFailure(
                    item_id=item.id,
                    stage="predict",
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            )
        finally:
            elapsed_ms = (perf_counter_ns() - started) / 1_000_000.0
            timings.append(InferenceTiming(item_id=item.id, duration_ms=elapsed_ms))

    if failures_list:
        return BaselineRunResult(
            identity=adapter.identity,
            requested=len(items),
            completed=len(predictions),
            failed=len(failures_list),
            predictions=tuple(predictions),
            failures=tuple(failures_list),
            timings=tuple(timings),
            action_metrics=None,
            abstention_metrics=None,
            evaluation_error=None,
        )

    action_metrics: ActionMetrics | None = None
    abstention_metrics: AbstentionMetrics | None = None
    evaluation_error: str | None = None
    try:
        action_metrics, records = evaluate_action_predictions(
            items,
            predictions,
            ece_bins=ece_bins,
        )
        if all(record.gold_sufficient is not None for record in records):
            abstention_metrics = evaluate_abstention(records, ece_bins=ece_bins)
    except Exception as exc:  # noqa: BLE001 - evaluation failure is packet evidence
        evaluation_error = f"{type(exc).__name__}: {exc}"

    return BaselineRunResult(
        identity=adapter.identity,
        requested=len(items),
        completed=len(predictions),
        failed=0,
        predictions=tuple(predictions),
        failures=(),
        timings=tuple(timings),
        action_metrics=action_metrics,
        abstention_metrics=abstention_metrics,
        evaluation_error=evaluation_error,
    )
