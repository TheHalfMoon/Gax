from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from gaxbench.external_adapters import render_model_state
from gaxbench.metrics import evaluate_action_predictions, risk_at_coverage, risk_coverage_curve
from gaxbench.provenance import canonical_json_sha256, sha256_file
from gaxbench.schema import BenchmarkItem, Prediction, validate_prediction_against_item

SelectorKind = Literal[
    "max-probability",
    "entropy",
    "margin",
    "learned-sufficiency",
]

_SUFFICIENCY_ARCHITECTURE = "gax-sufficiency-logistic-v0"
_SUFFICIENCY_FEATURE_REVISION = "sha256-sufficiency-v0.1"
_SUFFICIENCY_CHECKPOINT_SCHEMA = "0.1"
_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
_EPS = 1e-12


@dataclass(frozen=True)
class SufficiencyConfig:
    feature_dim: int = 16
    learning_rate: float = 0.15
    epochs: int = 60
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
        if self.learning_rate * self.l2 > 1.0:
            raise ValueError("learning_rate * l2 must be <= 1")


@dataclass(frozen=True)
class SufficiencyEpochRecord:
    epoch: int
    mean_binary_nll: float


@dataclass(frozen=True)
class SufficiencyTrainingResult:
    model: SufficiencyModel
    history: tuple[SufficiencyEpochRecord, ...]
    training_manifest_sha256: str


@dataclass(frozen=True)
class SelectorRankingReport:
    selector: SelectorKind
    n: int
    aurc: float
    risk_at_50: float
    risk_at_80: float
    risk_at_90: float


@dataclass(frozen=True)
class CoveragePolicy:
    selector: SelectorKind
    target_coverage: float
    threshold: float
    calibration_n: int
    calibration_selected: int
    calibration_actual_coverage: float


@dataclass(frozen=True)
class EmpiricalRiskPolicy:
    selector: SelectorKind
    max_calibration_risk: float
    threshold: float
    calibration_n: int
    calibration_selected: int
    calibration_actual_coverage: float
    calibration_empirical_risk: float


class SufficiencyModel:
    def __init__(
        self,
        config: SufficiencyConfig,
        *,
        weights: Sequence[float] | None = None,
        bias: float = 0.0,
    ) -> None:
        self.config = config
        width = config.feature_dim + 4
        if weights is None:
            rng = random.Random(config.seed)
            self._weights = [rng.uniform(-0.01, 0.01) for _ in range(width)]
        else:
            if len(weights) != width:
                raise ValueError("sufficiency weight count does not match feature width")
            self._weights = [float(value) for value in weights]
        self._bias = float(bias)
        _require_finite_parameters(self._weights, self._bias)

    @property
    def architecture_id(self) -> str:
        return _SUFFICIENCY_ARCHITECTURE

    @property
    def feature_revision(self) -> str:
        return _SUFFICIENCY_FEATURE_REVISION

    @property
    def model_revision(self) -> str:
        return canonical_json_sha256(self.model_payload())

    def model_payload(self) -> dict[str, object]:
        return {
            "architecture_id": self.architecture_id,
            "feature_revision": self.feature_revision,
            "config": asdict(self.config),
            "weights": list(self._weights),
            "bias": self._bias,
        }

    def score(self, item: BenchmarkItem, prediction: Prediction) -> float:
        features = _sufficiency_features(item, prediction, self.config.feature_dim)
        logit = self._bias + math.fsum(
            weight * feature
            for weight, feature in zip(self._weights, features, strict=True)
        )
        return _sigmoid(logit)

    def _update(self, features: Sequence[float], target: float) -> float:
        probability = _sigmoid(
            self._bias
            + math.fsum(
                weight * feature
                for weight, feature in zip(self._weights, features, strict=True)
            )
        )
        error = probability - target
        learning_rate = self.config.learning_rate
        for index, feature in enumerate(features):
            gradient = error * feature + self.config.l2 * self._weights[index]
            self._weights[index] -= learning_rate * gradient
        self._bias -= learning_rate * error
        _require_finite_parameters(self._weights, self._bias)
        return _binary_nll(probability, bool(target))


def max_probability_score(prediction: Prediction) -> float:
    return max(prediction.probabilities.values())


def entropy_confidence_score(prediction: Prediction) -> float:
    probabilities = list(prediction.probabilities.values())
    if len(probabilities) == 1:
        return 1.0
    entropy = -math.fsum(
        probability * math.log(max(probability, _EPS))
        for probability in probabilities
    )
    normalized = entropy / math.log(len(probabilities))
    return min(1.0, max(0.0, 1.0 - normalized))


def margin_score(prediction: Prediction) -> float:
    ordered = sorted(prediction.probabilities.values(), reverse=True)
    if len(ordered) == 1:
        return 1.0
    return ordered[0] - ordered[1]


def selector_scores(
    items: Sequence[BenchmarkItem],
    predictions: Sequence[Prediction],
    selector: SelectorKind,
    *,
    sufficiency_model: SufficiencyModel | None = None,
) -> dict[str, float]:
    prediction_map = _aligned_prediction_map(items, predictions)
    scores: dict[str, float] = {}
    for item in items:
        prediction = prediction_map[item.id]
        if selector == "max-probability":
            score = max_probability_score(prediction)
        elif selector == "entropy":
            score = entropy_confidence_score(prediction)
        elif selector == "margin":
            score = margin_score(prediction)
        elif selector == "learned-sufficiency":
            if sufficiency_model is None:
                raise ValueError("learned-sufficiency selector requires sufficiency_model")
            score = sufficiency_model.score(item, prediction)
        else:
            raise ValueError(f"unsupported selector: {selector}")
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("selector scores must be finite and in [0, 1]")
        scores[item.id] = score
    return scores


def evaluate_selector_ranking(
    items: Sequence[BenchmarkItem],
    predictions: Sequence[Prediction],
    scores: Mapping[str, float],
    *,
    selector: SelectorKind,
) -> SelectorRankingReport:
    _, records = evaluate_action_predictions(items, predictions)
    ordered_scores = _scores_for_items(items, scores)
    curve = risk_coverage_curve(
        [record.correct for record in records],
        ordered_scores,
    )
    return SelectorRankingReport(
        selector=selector,
        n=len(records),
        aurc=math.fsum(point.risk for point in curve) / len(curve),
        risk_at_50=risk_at_coverage(curve, 0.50),
        risk_at_80=risk_at_coverage(curve, 0.80),
        risk_at_90=risk_at_coverage(curve, 0.90),
    )


def fit_coverage_policy(
    scores: Mapping[str, float],
    *,
    selector: SelectorKind,
    target_coverage: float,
) -> CoveragePolicy:
    _validate_target(target_coverage, "target_coverage")
    ordered = _validated_score_values(scores)
    required = max(1, math.ceil(target_coverage * len(ordered)))
    threshold = ordered[required - 1]
    selected = sum(score >= threshold for score in ordered)
    return CoveragePolicy(
        selector=selector,
        target_coverage=target_coverage,
        threshold=threshold,
        calibration_n=len(ordered),
        calibration_selected=selected,
        calibration_actual_coverage=selected / len(ordered),
    )


def fit_empirical_risk_policy(
    items: Sequence[BenchmarkItem],
    predictions: Sequence[Prediction],
    scores: Mapping[str, float],
    *,
    selector: SelectorKind,
    max_calibration_risk: float,
) -> EmpiricalRiskPolicy:
    _validate_target(max_calibration_risk, "max_calibration_risk", allow_zero=True)
    if any(item.split != "calibration" for item in items):
        raise ValueError("empirical risk policy requires calibration split items only")
    _, records = evaluate_action_predictions(items, predictions)
    score_values = _scores_for_items(items, scores)
    candidates = sorted(set(score_values), reverse=True)
    best: tuple[float, int, float] | None = None
    for threshold in candidates:
        selected_indices = [
            index for index, score in enumerate(score_values) if score >= threshold
        ]
        errors = sum(not records[index].correct for index in selected_indices)
        risk = errors / len(selected_indices)
        if risk <= max_calibration_risk:
            if best is None or len(selected_indices) > best[1]:
                best = (threshold, len(selected_indices), risk)
    if best is None:
        raise ValueError("no non-empty calibration selection meets max_calibration_risk")
    threshold, selected, risk = best
    return EmpiricalRiskPolicy(
        selector=selector,
        max_calibration_risk=max_calibration_risk,
        threshold=threshold,
        calibration_n=len(records),
        calibration_selected=selected,
        calibration_actual_coverage=selected / len(records),
        calibration_empirical_risk=risk,
    )


def apply_selection_policy(
    predictions: Sequence[Prediction],
    scores: Mapping[str, float],
    policy: CoveragePolicy | EmpiricalRiskPolicy,
    *,
    expose_as_information_sufficiency: bool = False,
) -> list[Prediction]:
    if {prediction.item_id for prediction in predictions} != set(scores):
        raise ValueError("policy scores must exactly match prediction item IDs")
    output: list[Prediction] = []
    for prediction in predictions:
        score = scores[prediction.item_id]
        metadata = dict(prediction.metadata)
        metadata["selection_score"] = score
        metadata["selection_selector"] = policy.selector
        update: dict[str, object] = {
            "abstain": score < policy.threshold,
            "metadata": metadata,
        }
        if expose_as_information_sufficiency:
            update["information_sufficiency"] = score
        output.append(prediction.model_copy(update=update))
    return output


def train_sufficiency_model(
    items: Sequence[BenchmarkItem],
    predictions: Sequence[Prediction],
    config: SufficiencyConfig,
) -> SufficiencyTrainingResult:
    if not items:
        raise ValueError("sufficiency training items must not be empty")
    prediction_map = _aligned_prediction_map(items, predictions)
    for item in items:
        if item.split != "train":
            raise ValueError("sufficiency training accepts train split items only")
        if item.gold is None or item.gold.sufficient is None:
            raise ValueError("sufficiency training requires gold.sufficient labels")

    model = SufficiencyModel(config)
    rng = random.Random(config.seed)
    ordered = list(items)
    history: list[SufficiencyEpochRecord] = []
    for epoch in range(1, config.epochs + 1):
        rng.shuffle(ordered)
        losses: list[float] = []
        for item in ordered:
            prediction = prediction_map[item.id]
            features = _sufficiency_features(item, prediction, config.feature_dim)
            assert item.gold is not None and item.gold.sufficient is not None
            losses.append(model._update(features, float(item.gold.sufficient)))
        history.append(
            SufficiencyEpochRecord(
                epoch=epoch,
                mean_binary_nll=math.fsum(losses) / len(losses),
            )
        )
    return SufficiencyTrainingResult(
        model=model,
        history=tuple(history),
        training_manifest_sha256=sufficiency_training_manifest_sha256(items, predictions),
    )


def sufficiency_training_manifest_sha256(
    items: Sequence[BenchmarkItem],
    predictions: Sequence[Prediction],
) -> str:
    prediction_map = _aligned_prediction_map(items, predictions)
    payload = [
        {
            "item": item.model_dump(mode="json"),
            "prediction": prediction_map[item.id].model_dump(mode="json"),
        }
        for item in sorted(items, key=lambda candidate: candidate.id)
    ]
    return canonical_json_sha256(payload)


def save_sufficiency_checkpoint(
    path: str | Path,
    result: SufficiencyTrainingResult,
) -> str:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "schema_version": _SUFFICIENCY_CHECKPOINT_SCHEMA,
        "model": result.model.model_payload(),
        "training": {
            "training_manifest_sha256": result.training_manifest_sha256,
            "epochs_completed": len(result.history),
        },
    }
    envelope = {"payload": payload, "sha256": canonical_json_sha256(payload)}
    checkpoint_path.write_text(
        json.dumps(envelope, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return sha256_file(checkpoint_path)


def load_sufficiency_checkpoint(path: str | Path) -> tuple[SufficiencyModel, str]:
    checkpoint_path = Path(path)
    raw = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"payload", "sha256"}:
        raise ValueError("sufficiency checkpoint envelope is malformed")
    payload = raw["payload"]
    expected = raw["sha256"]
    if not isinstance(payload, dict) or not isinstance(expected, str):
        raise ValueError("sufficiency checkpoint envelope has invalid types")
    if canonical_json_sha256(payload) != expected:
        raise ValueError("sufficiency checkpoint integrity mismatch")
    if payload.get("schema_version") != _SUFFICIENCY_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported sufficiency checkpoint schema")
    model_payload = payload.get("model")
    if not isinstance(model_payload, dict):
        raise ValueError("sufficiency checkpoint model payload is malformed")
    if model_payload.get("architecture_id") != _SUFFICIENCY_ARCHITECTURE:
        raise ValueError("unsupported sufficiency architecture")
    if model_payload.get("feature_revision") != _SUFFICIENCY_FEATURE_REVISION:
        raise ValueError("unsupported sufficiency feature revision")
    config_raw = model_payload.get("config")
    weights_raw = model_payload.get("weights")
    bias_raw = model_payload.get("bias")
    if not isinstance(config_raw, dict) or not isinstance(weights_raw, list):
        raise ValueError("sufficiency checkpoint parameters are malformed")
    if isinstance(bias_raw, bool) or not isinstance(bias_raw, (int, float)):
        raise ValueError("sufficiency checkpoint bias must be numeric")
    weights: list[float] = []
    for value in weights_raw:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("sufficiency checkpoint weights must be numeric")
        weights.append(float(value))
    config = SufficiencyConfig(**config_raw)
    model = SufficiencyModel(config, weights=weights, bias=float(bias_raw))
    return model, sha256_file(checkpoint_path)


def _aligned_prediction_map(
    items: Sequence[BenchmarkItem],
    predictions: Sequence[Prediction],
) -> dict[str, Prediction]:
    item_ids = [item.id for item in items]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("item ids must be unique")
    prediction_map: dict[str, Prediction] = {}
    for prediction in predictions:
        if prediction.item_id in prediction_map:
            raise ValueError(f"duplicate prediction for item {prediction.item_id!r}")
        prediction_map[prediction.item_id] = prediction
    if set(prediction_map) != set(item_ids):
        missing = sorted(set(item_ids) - set(prediction_map))
        extra = sorted(set(prediction_map) - set(item_ids))
        raise ValueError(f"prediction ids mismatch: missing={missing}, extra={extra}")
    for item in items:
        validate_prediction_against_item(item, prediction_map[item.id])
    return prediction_map


def _sufficiency_features(
    item: BenchmarkItem,
    prediction: Prediction,
    feature_dim: int,
) -> list[float]:
    validate_prediction_against_item(item, prediction)
    visible_payload = {
        "state": render_model_state(item),
        "actions": [
            action.description
            for action in sorted(item.actions, key=lambda candidate: candidate.id)
        ],
    }
    serialized = json.dumps(
        visible_payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    hashed = _hashed_visible_vector(serialized, feature_dim)
    return [
        *hashed,
        max_probability_score(prediction),
        entropy_confidence_score(prediction),
        margin_score(prediction),
        min(len(item.actions), 16) / 16.0,
    ]


def _hashed_visible_vector(serialized: str, feature_dim: int) -> list[float]:
    vector = [0.0] * feature_dim
    tokens = _TOKEN_RE.findall(serialized.casefold()) or ["<empty>"]
    for token in tokens:
        digest = hashlib.sha256(
            f"{_SUFFICIENCY_FEATURE_REVISION}|visible|{token}".encode()
        ).digest()
        index = int.from_bytes(digest[:4], "big") % feature_dim
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign
    norm = math.sqrt(math.fsum(value * value for value in vector))
    if not math.isfinite(norm):
        raise ValueError("sufficiency feature vector has non-finite norm")
    if norm == 0.0:
        fallback = hashlib.sha256(
            (
                f"{_SUFFICIENCY_FEATURE_REVISION}|visible|fallback|"
                + "\x1f".join(sorted(tokens))
            ).encode()
        ).digest()
        vector[int.from_bytes(fallback[:4], "big") % feature_dim] = 1.0
        norm = 1.0
    return [value / norm for value in vector]


def _scores_for_items(
    items: Sequence[BenchmarkItem],
    scores: Mapping[str, float],
) -> list[float]:
    item_ids = {item.id for item in items}
    if set(scores) != item_ids:
        missing = sorted(item_ids - set(scores))
        extra = sorted(set(scores) - item_ids)
        raise ValueError(f"selector score ids mismatch: missing={missing}, extra={extra}")
    output = [scores[item.id] for item in items]
    for score in output:
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("selector scores must be finite and in [0, 1]")
    return output


def _validated_score_values(scores: Mapping[str, float]) -> list[float]:
    if not scores:
        raise ValueError("selector scores must not be empty")
    values = sorted(scores.values(), reverse=True)
    for score in values:
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("selector scores must be finite and in [0, 1]")
    return values


def _validate_target(value: float, name: str, *, allow_zero: bool = False) -> None:
    lower_ok = value >= 0.0 if allow_zero else value > 0.0
    if not math.isfinite(value) or not lower_ok or value > 1.0:
        interval = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"{name} must be in {interval}")


def _sigmoid(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("sufficiency logit must be finite")
    if value >= 0.0:
        denominator = 1.0 + math.exp(-value)
        return 1.0 / denominator
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _binary_nll(probability: float, label: bool) -> float:
    clipped = min(1.0 - _EPS, max(_EPS, probability))
    return -math.log(clipped if label else 1.0 - clipped)


def _require_finite_parameters(weights: Sequence[float], bias: float) -> None:
    if not math.isfinite(bias) or any(not math.isfinite(value) for value in weights):
        raise ValueError("sufficiency model parameters must be finite")
