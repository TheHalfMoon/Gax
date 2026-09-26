from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from gaxbench.baselines import AdapterIdentity
from gaxbench.external_adapters import render_model_state
from gaxbench.provenance import canonical_json_sha256, sha256_file
from gaxbench.schema import Action, BenchmarkItem, Prediction

_ARCHITECTURE_ID = "gax-bilinear-v0"
_CHECKPOINT_SCHEMA = "0.1"
_FEATURE_REVISION = "sha256-word-v0.2"
_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


@dataclass(frozen=True)
class GaxV0Config:
    feature_dim: int = 32
    learning_rate: float = 0.2
    epochs: int = 40
    l2: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.feature_dim < 4:
            raise ValueError("feature_dim must be >= 4")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and > 0")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if not math.isfinite(self.l2) or self.l2 < 0.0:
            raise ValueError("l2 must be finite and >= 0")


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    mean_nll: float


@dataclass(frozen=True)
class TrainingResult:
    model: GaxV0Model
    history: tuple[EpochRecord, ...]
    training_manifest_sha256: str


class GaxV0Model:
    def __init__(self, config: GaxV0Config, weights: Sequence[Sequence[float]] | None = None):
        self.config = config
        if weights is None:
            rng = random.Random(config.seed)
            self._weights = [
                [rng.uniform(-0.01, 0.01) for _ in range(config.feature_dim)]
                for _ in range(config.feature_dim)
            ]
        else:
            self._weights = _validate_weights(weights, config.feature_dim)

    @property
    def architecture_id(self) -> str:
        return _ARCHITECTURE_ID

    @property
    def feature_revision(self) -> str:
        return _FEATURE_REVISION

    @property
    def model_revision(self) -> str:
        return canonical_json_sha256(self.model_payload())

    def model_payload(self) -> dict[str, object]:
        return {
            "architecture_id": self.architecture_id,
            "feature_revision": self.feature_revision,
            "config": asdict(self.config),
            "weights": [list(row) for row in self._weights],
        }

    def predict(self, item: BenchmarkItem) -> Prediction:
        probabilities = self.probabilities(item)
        return Prediction(
            item_id=item.id,
            probabilities=probabilities,
            metadata={
                "architecture_id": self.architecture_id,
                "feature_revision": self.feature_revision,
                "model_revision": self.model_revision,
            },
        )

    def probabilities(self, item: BenchmarkItem) -> dict[str, float]:
        if not item.actions:
            raise ValueError("item actions must not be empty")
        actions = sorted(item.actions, key=lambda action: action.id)
        state_vector = _state_vector(item, self.config.feature_dim)
        logits = [
            self._score(state_vector, _action_vector(action, self.config.feature_dim))
            for action in actions
        ]
        probabilities = _softmax(logits)
        return {
            action.id: probability
            for action, probability in zip(actions, probabilities, strict=True)
        }

    def _score(self, state_vector: Sequence[float], action_vector: Sequence[float]) -> float:
        total = 0.0
        for row_index, state_value in enumerate(state_vector):
            if state_value == 0.0:
                continue
            row = self._weights[row_index]
            subtotal = math.fsum(
                row[col_index] * action_value
                for col_index, action_value in enumerate(action_vector)
                if action_value != 0.0
            )
            total += state_value * subtotal
        if not math.isfinite(total):
            raise ValueError("model produced a non-finite score")
        return total

    def _apply_gradient(
        self,
        state_vector: Sequence[float],
        action_vectors: Sequence[Sequence[float]],
        factors: Sequence[float],
    ) -> None:
        learning_rate = self.config.learning_rate
        if self.config.l2:
            shrink = 1.0 - learning_rate * self.config.l2
            if shrink < 0.0:
                raise ValueError("learning_rate * l2 must be <= 1")
            for row in self._weights:
                for index in range(len(row)):
                    row[index] *= shrink

        for action_vector, factor in zip(action_vectors, factors, strict=True):
            if factor == 0.0:
                continue
            scale = learning_rate * factor
            for row_index, state_value in enumerate(state_vector):
                if state_value == 0.0:
                    continue
                row = self._weights[row_index]
                for col_index, action_value in enumerate(action_vector):
                    if action_value != 0.0:
                        row[col_index] -= scale * state_value * action_value

        _require_finite_weights(self._weights)


class GaxV0Adapter:
    def __init__(self, model: GaxV0Model, *, artifact_sha256: str | None = None) -> None:
        self._model = model
        self._artifact_sha256 = artifact_sha256

    @property
    def identity(self) -> AdapterIdentity:
        return AdapterIdentity(
            name="gax-v0",
            adapter_version="0.1",
            deterministic=True,
            model_id=self._model.architecture_id,
            model_revision=self._model.model_revision,
            tokenizer_revision=self._model.feature_revision,
            source_revision="gax-p03",
            artifact_sha256=self._artifact_sha256,
        )

    def prepare(self, items: Sequence[BenchmarkItem]) -> None:
        if not items:
            raise ValueError("items must not be empty")

    def predict(self, item: BenchmarkItem) -> Prediction:
        return self._model.predict(item)


def train_gax_v0(
    train_items: Sequence[BenchmarkItem],
    config: GaxV0Config,
    *,
    validation_items: Sequence[BenchmarkItem] = (),
) -> TrainingResult:
    _validate_training_items(train_items, required_split="train")
    if validation_items:
        _validate_training_items(validation_items, required_split="validation")

    model = GaxV0Model(config)
    rng = random.Random(config.seed)
    ordered = list(train_items)
    history: list[EpochRecord] = []

    for epoch in range(1, config.epochs + 1):
        rng.shuffle(ordered)
        losses: list[float] = []
        for item in ordered:
            assert item.gold is not None and item.gold.action is not None
            actions = sorted(item.actions, key=lambda action: action.id)
            state_vector = _state_vector(item, config.feature_dim)
            action_vectors = [_action_vector(action, config.feature_dim) for action in actions]
            logits = [model._score(state_vector, vector) for vector in action_vectors]
            probabilities = _softmax(logits)
            gold_index = next(
                index for index, action in enumerate(actions) if action.id == item.gold.action
            )
            gold_probability = max(probabilities[gold_index], 1e-12)
            losses.append(-math.log(gold_probability))
            factors = [
                probability - (1.0 if index == gold_index else 0.0)
                for index, probability in enumerate(probabilities)
            ]
            model._apply_gradient(state_vector, action_vectors, factors)

        mean_nll = math.fsum(losses) / len(losses)
        if not math.isfinite(mean_nll):
            raise ValueError("training produced a non-finite loss")
        history.append(EpochRecord(epoch=epoch, mean_nll=mean_nll))

    return TrainingResult(
        model=model,
        history=tuple(history),
        training_manifest_sha256=training_manifest_sha256(train_items),
    )


def training_manifest_sha256(items: Sequence[BenchmarkItem]) -> str:
    if not items:
        raise ValueError("training manifest requires at least one item")
    payload = [
        item.model_dump(mode="json")
        for item in sorted(items, key=lambda candidate: candidate.id)
    ]
    return canonical_json_sha256(payload)


def save_gax_v0_checkpoint(
    path: str | Path,
    result: TrainingResult,
) -> str:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "schema_version": _CHECKPOINT_SCHEMA,
        "model": result.model.model_payload(),
        "training": {
            "seed": result.model.config.seed,
            "training_manifest_sha256": result.training_manifest_sha256,
            "epochs_completed": len(result.history),
        },
    }
    envelope = {
        "payload": payload,
        "sha256": canonical_json_sha256(payload),
    }
    checkpoint_path.write_text(
        json.dumps(envelope, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return sha256_file(checkpoint_path)


def load_gax_v0_checkpoint(path: str | Path) -> tuple[GaxV0Model, str]:
    checkpoint_path = Path(path)
    raw = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"payload", "sha256"}:
        raise ValueError("checkpoint envelope must contain exactly payload and sha256")
    payload = raw["payload"]
    expected_digest = raw["sha256"]
    if not isinstance(payload, dict) or not isinstance(expected_digest, str):
        raise ValueError("checkpoint envelope has invalid types")
    actual_digest = canonical_json_sha256(payload)
    if actual_digest != expected_digest:
        raise ValueError("checkpoint integrity mismatch")
    if payload.get("schema_version") != _CHECKPOINT_SCHEMA:
        raise ValueError("unsupported checkpoint schema_version")

    model_payload = payload.get("model")
    training_payload = payload.get("training")
    if not isinstance(model_payload, dict) or not isinstance(training_payload, dict):
        raise ValueError("checkpoint payload is missing model/training objects")
    if model_payload.get("architecture_id") != _ARCHITECTURE_ID:
        raise ValueError("unsupported checkpoint architecture")
    if model_payload.get("feature_revision") != _FEATURE_REVISION:
        raise ValueError("unsupported feature revision")

    config_raw = model_payload.get("config")
    weights_raw = model_payload.get("weights")
    if not isinstance(config_raw, dict) or not isinstance(weights_raw, list):
        raise ValueError("checkpoint model config/weights are malformed")
    config = GaxV0Config(**config_raw)
    weights = _validate_checkpoint_weight_container(weights_raw)
    model = GaxV0Model(config, weights=weights)

    manifest = training_payload.get("training_manifest_sha256")
    seed = training_payload.get("seed")
    if not isinstance(manifest, str) or len(manifest) != 64:
        raise ValueError("checkpoint training manifest is invalid")
    if seed != config.seed:
        raise ValueError("checkpoint training seed disagrees with model config")
    return model, sha256_file(checkpoint_path)


def _validate_training_items(items: Sequence[BenchmarkItem], *, required_split: str) -> None:
    if not items:
        raise ValueError(f"{required_split} items must not be empty")
    item_ids = [item.id for item in items]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("training item ids must be unique")
    for item in items:
        if item.split != required_split:
            raise ValueError(
                f"item {item.id!r} has split {item.split!r}; expected {required_split!r}"
            )
        if item.gold is None or item.gold.action is None:
            raise ValueError(f"item {item.id!r} requires a gold action for training")


def _state_vector(item: BenchmarkItem, feature_dim: int) -> list[float]:
    visible = render_model_state(item)
    serialized = json.dumps(
        visible,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return _hashed_vector([serialized], feature_dim, channel="state")


def _action_vector(action: Action, feature_dim: int) -> list[float]:
    return _hashed_vector([action.description], feature_dim, channel="action")


def _hashed_vector(parts: Sequence[str], feature_dim: int, *, channel: str) -> list[float]:
    vector = [0.0] * feature_dim
    tokens: list[str] = []
    for part in parts:
        tokens.extend(_TOKEN_RE.findall(part.casefold()))
    if not tokens:
        tokens = ["<empty>"]

    for token in tokens:
        digest = hashlib.sha256(f"{_FEATURE_REVISION}|{channel}|{token}".encode()).digest()
        index = int.from_bytes(digest[:4], "big") % feature_dim
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign

    norm = math.sqrt(math.fsum(value * value for value in vector))
    if norm == 0.0 or not math.isfinite(norm):
        raise ValueError("feature vector has invalid norm")
    return [value / norm for value in vector]


def _softmax(logits: Sequence[float]) -> list[float]:
    if not logits or any(not math.isfinite(value) for value in logits):
        raise ValueError("softmax requires finite non-empty logits")
    maximum = max(logits)
    exponentials = [math.exp(value - maximum) for value in logits]
    denominator = math.fsum(exponentials)
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise ValueError("softmax denominator is invalid")
    probabilities = [value / denominator for value in exponentials]
    probabilities[-1] += 1.0 - math.fsum(probabilities)
    return probabilities


def _validate_weights(weights: Sequence[Sequence[float]], feature_dim: int) -> list[list[float]]:
    if len(weights) != feature_dim:
        raise ValueError("weight matrix row count does not match feature_dim")
    copied: list[list[float]] = []
    for row in weights:
        if len(row) != feature_dim:
            raise ValueError("weight matrix column count does not match feature_dim")
        copied_row = [float(value) for value in row]
        copied.append(copied_row)
    _require_finite_weights(copied)
    return copied


def _validate_checkpoint_weight_container(value: list[object]) -> list[list[float]]:
    rows: list[list[float]] = []
    for row in value:
        if not isinstance(row, list):
            raise ValueError("checkpoint weight rows must be lists")
        converted: list[float] = []
        for entry in row:
            if isinstance(entry, bool) or not isinstance(entry, (int, float)):
                raise ValueError("checkpoint weights must be numeric")
            converted.append(float(entry))
        rows.append(converted)
    return rows


def _require_finite_weights(weights: Sequence[Sequence[float]]) -> None:
    for row in weights:
        for value in row:
            if not math.isfinite(value):
                raise ValueError("model weights must be finite")
