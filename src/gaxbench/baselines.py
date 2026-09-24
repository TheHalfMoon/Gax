from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from gaxbench.io import load_predictions
from gaxbench.provenance import sha256_file
from gaxbench.schema import BenchmarkItem, Prediction, validate_prediction_against_item


@dataclass(frozen=True)
class AdapterIdentity:
    name: str
    adapter_version: str
    deterministic: bool
    model_id: str | None = None
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    source_revision: str | None = None
    artifact_sha256: str | None = None


class BaselineAdapter(Protocol):
    @property
    def identity(self) -> AdapterIdentity: ...

    def prepare(self, items: Sequence[BenchmarkItem]) -> None: ...

    def predict(self, item: BenchmarkItem) -> Prediction: ...


class UniformBaselineAdapter:
    @property
    def identity(self) -> AdapterIdentity:
        return AdapterIdentity(
            name="uniform",
            adapter_version="0.1",
            deterministic=True,
            source_revision="gax-p02",
        )

    def prepare(self, items: Sequence[BenchmarkItem]) -> None:
        _require_items(items)

    def predict(self, item: BenchmarkItem) -> Prediction:
        probability = 1.0 / len(item.actions)
        return Prediction(
            item_id=item.id,
            probabilities={action.id: probability for action in item.actions},
        )


class LexicographicBaselineAdapter:
    @property
    def identity(self) -> AdapterIdentity:
        return AdapterIdentity(
            name="lexicographic",
            adapter_version="0.1",
            deterministic=True,
            source_revision="gax-p02",
        )

    def prepare(self, items: Sequence[BenchmarkItem]) -> None:
        _require_items(items)

    def predict(self, item: BenchmarkItem) -> Prediction:
        selected = min(action.id for action in item.actions)
        return Prediction(
            item_id=item.id,
            probabilities={
                action.id: 1.0 if action.id == selected else 0.0 for action in item.actions
            },
        )


class PredictionFileAdapter:
    def __init__(
        self,
        path: str | Path,
        *,
        name: str,
        adapter_version: str,
        model_id: str | None = None,
        model_revision: str | None = None,
        tokenizer_revision: str | None = None,
        source_revision: str | None = None,
    ) -> None:
        self._path = Path(path)
        self._digest = sha256_file(self._path)
        predictions = load_predictions(self._path)
        prediction_map: dict[str, Prediction] = {}
        for prediction in predictions:
            if prediction.item_id in prediction_map:
                raise ValueError(f"duplicate prediction for item {prediction.item_id!r}")
            prediction_map[prediction.item_id] = prediction
        self._predictions = prediction_map
        self._identity = AdapterIdentity(
            name=name,
            adapter_version=adapter_version,
            deterministic=True,
            model_id=model_id,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
            source_revision=source_revision,
            artifact_sha256=self._digest,
        )

    @property
    def identity(self) -> AdapterIdentity:
        return self._identity

    def prepare(self, items: Sequence[BenchmarkItem]) -> None:
        _require_items(items)
        self._verify_immutable()
        item_ids = {item.id for item in items}
        prediction_ids = set(self._predictions)
        if prediction_ids != item_ids:
            missing = sorted(item_ids - prediction_ids)
            extra = sorted(prediction_ids - item_ids)
            raise ValueError(
                f"prediction file item ids mismatch: missing={missing}, extra={extra}"
            )
        for item in items:
            validate_prediction_against_item(item, self._predictions[item.id])

    def predict(self, item: BenchmarkItem) -> Prediction:
        self._verify_immutable()
        try:
            prediction = self._predictions[item.id]
        except KeyError as exc:
            raise KeyError(f"prediction file has no item {item.id!r}") from exc
        validate_prediction_against_item(item, prediction)
        return prediction

    def _verify_immutable(self) -> None:
        current = sha256_file(self._path)
        if current != self._digest:
            raise ValueError(
                "prediction file changed after adapter initialization: "
                f"expected={self._digest}, actual={current}"
            )


def _require_items(items: Sequence[BenchmarkItem]) -> None:
    if not items:
        raise ValueError("items must not be empty")
