from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal

from gaxbench.ecal import ExperimentContext, items_manifest_sha256
from gaxbench.gax_v0 import GaxV0Model, _state_vector
from gaxbench.metrics import (
    AbstentionMetrics,
    evaluate_abstention,
    risk_at_coverage,
    risk_coverage_curve,
)
from gaxbench.provenance import canonical_json_sha256
from gaxbench.schema import BenchmarkItem, EvaluationRecord

SelectorName = Literal[
    "max-probability",
    "entropy-confidence",
    "top1-top2-margin",
    "learned-sufficiency",
]
PaperDecision = Literal["keep", "reject", "defer-real-data"]

_P05_SCHEMA_VERSION = "0.1"
_P05_SOURCE_REVISION = "gax-p05"
_EPS = 1e-12


@dataclass(frozen=True)
class SufficiencyConfig:
    feature_dim: int = 32
    learning_rate: float = 0.1
    epochs: int = 80
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

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(asdict(self))


@dataclass(frozen=True)
class SufficiencyEpochRecord:
    epoch: int
    mean_nll: float


@dataclass(frozen=True)
class SufficiencyTrainingResult:
    model: LearnedSufficiencyModel
    history: tuple[SufficiencyEpochRecord, ...]
    training_manifest_sha256: str
    config_sha256: str


@dataclass(frozen=True)
class CoveragePolicy:
    selector: SelectorName
    target_coverage: float
    threshold: float
    calibration_n: int
    calibration_manifest_sha256: str


@dataclass(frozen=True)
class SelectiveRecord:
    item_id: str
    selection_score: float
    action_correct: bool
    committed: bool
    gold_sufficient: bool


@dataclass(frozen=True)
class SelectorMetrics:
    selector: SelectorName
    requested: int
    completed: int
    failed: int
    target_coverage: float
    actual_coverage: float
    action_accuracy: float
    committed_risk: float | None
    aurc: float
    risk_at_target: float
    risk_at_50: float
    risk_at_80: float
    risk_at_90: float
    abstention: AbstentionMetrics
    records: tuple[SelectiveRecord, ...]


@dataclass(frozen=True)
class SelectorSuiteResult:
    experiment_context: ExperimentContext
    train_manifest_sha256: str
    calibration_manifest_sha256: str
    validation_manifest_sha256: str
    action_model_revision: str
    sufficiency_model_revision: str
    sufficiency_config_sha256: str
    target_coverage: float
    evaluations: tuple[SelectorMetrics, ...]
    strongest_confidence_control: SelectorName
    paper_decision: PaperDecision
    paper_decision_rationale: str


class LearnedSufficiencyModel:
    def __init__(
        self,
        config: SufficiencyConfig,
        *,
        weights: Sequence[float] | None = None,
        bias: float = 0.0,
    ) -> None:
        self.config = config
        expected = config.feature_dim + 3
        if weights is None:
            self._weights = [0.0] * expected
        else:
            if len(weights) != expected:
                raise ValueError("sufficiency weight count does not match feature contract")
            self._weights = [float(value) for value in weights]
        self._bias = float(bias)
        self._require_finite_parameters()

    @property
    def model_revision(self) -> str:
        return canonical_json_sha256(
            {
                "source_revision": _P05_SOURCE_REVISION,
                "config": asdict(self.config),
                "weights": self._weights,
                "bias": self._bias,
            }
        )

    def score(
        self,
        item: BenchmarkItem,
        action_probabilities: dict[str, float],
    ) -> float:
        features = _selector_features(item, action_probabilities, self.config.feature_dim)
        logit = math.fsum(
            weight * feature
            for weight, feature in zip(self._weights, features, strict=True)
        ) + self._bias
        return _sigmoid(logit)

    def _update(self, features: Sequence[float], label: float) -> float:
        probability = _sigmoid(
            math.fsum(
                weight * feature
                for weight, feature in zip(self._weights, features, strict=True)
            )
            + self._bias
        )
        error = probability - label
        if self.config.l2:
            shrink = 1.0 - self.config.learning_rate * self.config.l2
            self._weights = [weight * shrink for weight in self._weights]
        for index, feature in enumerate(features):
            self._weights[index] -= self.config.learning_rate * error * feature
        self._bias -= self.config.learning_rate * error
        self._require_finite_parameters()
        return _binary_nll(probability, bool(label))

    def _require_finite_parameters(self) -> None:
        if not math.isfinite(self._bias) or any(
            not math.isfinite(weight) for weight in self._weights
        ):
            raise ValueError("sufficiency model parameters must be finite")


def train_information_sufficiency(
    train_items: Sequence[BenchmarkItem],
    action_model: GaxV0Model,
    config: SufficiencyConfig,
) -> SufficiencyTrainingResult:
    _validate_sufficiency_items(train_items, required_split="train")
    _validate_config_matches_action_model(config, action_model)

    model = LearnedSufficiencyModel(config)
    rng = random.Random(config.seed)
    ordered = list(train_items)
    history: list[SufficiencyEpochRecord] = []

    for epoch in range(1, config.epochs + 1):
        rng.shuffle(ordered)
        losses: list[float] = []
        for item in ordered:
            assert item.gold is not None and item.gold.sufficient is not None
            probabilities = action_model.probabilities(item)
            features = _selector_features(item, probabilities, config.feature_dim)
            losses.append(model._update(features, float(item.gold.sufficient)))
        history.append(
            SufficiencyEpochRecord(
                epoch=epoch,
                mean_nll=math.fsum(losses) / len(losses),
            )
        )

    return SufficiencyTrainingResult(
        model=model,
        history=tuple(history),
        training_manifest_sha256=items_manifest_sha256(train_items),
        config_sha256=config.sha256,
    )


def max_probability_score(probabilities: dict[str, float]) -> float:
    _validate_probabilities(probabilities)
    return max(probabilities.values())


def entropy_confidence_score(probabilities: dict[str, float]) -> float:
    _validate_probabilities(probabilities)
    if len(probabilities) == 1:
        return 1.0
    entropy = -math.fsum(
        probability * math.log(probability)
        for probability in probabilities.values()
        if probability > 0.0
    )
    maximum_entropy = math.log(len(probabilities))
    score = 1.0 - entropy / maximum_entropy
    return min(1.0, max(0.0, score))


def top1_top2_margin_score(probabilities: dict[str, float]) -> float:
    _validate_probabilities(probabilities)
    ordered = sorted(probabilities.values(), reverse=True)
    if len(ordered) == 1:
        return 1.0
    return ordered[0] - ordered[1]


def selector_score(
    selector: SelectorName,
    item: BenchmarkItem,
    action_model: GaxV0Model,
    *,
    sufficiency_model: LearnedSufficiencyModel | None = None,
) -> float:
    probabilities = action_model.probabilities(item)
    if selector == "max-probability":
        return max_probability_score(probabilities)
    if selector == "entropy-confidence":
        return entropy_confidence_score(probabilities)
    if selector == "top1-top2-margin":
        return top1_top2_margin_score(probabilities)
    if selector == "learned-sufficiency":
        if sufficiency_model is None:
            raise ValueError("learned-sufficiency requires a sufficiency model")
        return sufficiency_model.score(item, probabilities)
    raise ValueError(f"unsupported selector: {selector}")


def fit_coverage_policy(
    calibration_items: Sequence[BenchmarkItem],
    action_model: GaxV0Model,
    selector: SelectorName,
    *,
    target_coverage: float,
    sufficiency_model: LearnedSufficiencyModel | None = None,
) -> CoveragePolicy:
    _validate_split_only(calibration_items, required_split="calibration")
    _validate_coverage(target_coverage)
    scores = sorted(
        (
            selector_score(
                selector,
                item,
                action_model,
                sufficiency_model=sufficiency_model,
            )
            for item in calibration_items
        ),
        reverse=True,
    )
    selected = max(1, math.ceil(target_coverage * len(scores)))
    threshold = scores[selected - 1]
    return CoveragePolicy(
        selector=selector,
        target_coverage=target_coverage,
        threshold=threshold,
        calibration_n=len(scores),
        calibration_manifest_sha256=items_manifest_sha256(calibration_items),
    )


def evaluate_selector(
    validation_items: Sequence[BenchmarkItem],
    action_model: GaxV0Model,
    policy: CoveragePolicy,
    *,
    sufficiency_model: LearnedSufficiencyModel | None = None,
    ece_bins: int = 15,
) -> SelectorMetrics:
    _validate_sufficiency_items(validation_items, required_split="validation")
    if ece_bins < 2:
        raise ValueError("ece_bins must be >= 2")

    evaluation_records: list[EvaluationRecord] = []
    selective_records: list[SelectiveRecord] = []
    correctness: list[bool] = []
    selection_scores: list[float] = []

    for item in validation_items:
        assert item.gold is not None
        assert item.gold.action is not None
        assert item.gold.sufficient is not None
        probabilities = action_model.probabilities(item)
        predicted_action = max(
            probabilities.items(),
            key=lambda pair: (pair[1], pair[0]),
        )[0]
        correct = predicted_action == item.gold.action
        score = selector_score(
            policy.selector,
            item,
            action_model,
            sufficiency_model=sufficiency_model,
        )
        committed = score >= policy.threshold
        information_sufficiency = score if policy.selector == "learned-sufficiency" else None
        evaluation_records.append(
            EvaluationRecord(
                item_id=item.id,
                gold_action=item.gold.action,
                predicted_action=predicted_action,
                action_confidence=max(probabilities.values()),
                correct=correct,
                abstain=not committed,
                selection_score=score,
                gold_sufficient=item.gold.sufficient,
                information_sufficiency=information_sufficiency,
            )
        )
        selective_records.append(
            SelectiveRecord(
                item_id=item.id,
                selection_score=score,
                action_correct=correct,
                committed=committed,
                gold_sufficient=item.gold.sufficient,
            )
        )
        correctness.append(correct)
        selection_scores.append(score)

    committed_records = [record for record in evaluation_records if not record.abstain]
    committed_risk = (
        None
        if not committed_records
        else 1.0
        - sum(record.correct for record in committed_records) / len(committed_records)
    )
    curve = risk_coverage_curve(correctness, selection_scores)
    abstention = evaluate_abstention(evaluation_records, ece_bins=ece_bins)
    return SelectorMetrics(
        selector=policy.selector,
        requested=len(validation_items),
        completed=len(validation_items),
        failed=0,
        target_coverage=policy.target_coverage,
        actual_coverage=len(committed_records) / len(validation_items),
        action_accuracy=sum(correctness) / len(correctness),
        committed_risk=committed_risk,
        aurc=math.fsum(point.risk for point in curve) / len(curve),
        risk_at_target=risk_at_coverage(curve, policy.target_coverage),
        risk_at_50=risk_at_coverage(curve, 0.50),
        risk_at_80=risk_at_coverage(curve, 0.80),
        risk_at_90=risk_at_coverage(curve, 0.90),
        abstention=abstention,
        records=tuple(selective_records),
    )


def run_matched_selector_suite(
    train_items: Sequence[BenchmarkItem],
    calibration_items: Sequence[BenchmarkItem],
    validation_items: Sequence[BenchmarkItem],
    action_model: GaxV0Model,
    *,
    context: ExperimentContext,
    sufficiency_config: SufficiencyConfig | None = None,
    target_coverage: float = 0.80,
    ece_bins: int = 15,
) -> SelectorSuiteResult:
    _validate_sufficiency_items(train_items, required_split="train")
    _validate_split_only(calibration_items, required_split="calibration")
    _validate_sufficiency_items(validation_items, required_split="validation")
    _validate_coverage(target_coverage)

    config = sufficiency_config or SufficiencyConfig(feature_dim=action_model.config.feature_dim)
    _validate_config_matches_action_model(config, action_model)
    training = train_information_sufficiency(train_items, action_model, config)
    selectors: tuple[SelectorName, ...] = (
        "max-probability",
        "entropy-confidence",
        "top1-top2-margin",
        "learned-sufficiency",
    )
    evaluations: list[SelectorMetrics] = []
    for selector in selectors:
        learned = training.model if selector == "learned-sufficiency" else None
        policy = fit_coverage_policy(
            calibration_items,
            action_model,
            selector,
            target_coverage=target_coverage,
            sufficiency_model=learned,
        )
        evaluations.append(
            evaluate_selector(
                validation_items,
                action_model,
                policy,
                sufficiency_model=learned,
                ece_bins=ece_bins,
            )
        )

    controls = [
        metric
        for metric in evaluations
        if metric.selector != "learned-sufficiency"
    ]
    strongest = min(
        controls,
        key=lambda metric: (metric.risk_at_target, metric.selector),
    )
    return SelectorSuiteResult(
        experiment_context=context,
        train_manifest_sha256=items_manifest_sha256(train_items),
        calibration_manifest_sha256=items_manifest_sha256(calibration_items),
        validation_manifest_sha256=items_manifest_sha256(validation_items),
        action_model_revision=action_model.model_revision,
        sufficiency_model_revision=training.model.model_revision,
        sufficiency_config_sha256=training.config_sha256,
        target_coverage=target_coverage,
        evaluations=tuple(evaluations),
        strongest_confidence_control=strongest.selector,
        paper_decision="defer-real-data",
        paper_decision_rationale=(
            "P05 synthetic mechanism fixtures qualify implementation only; learned sufficiency "
            "requires licensed leakage-audited development evidence for a paper keep/reject gate."
        ),
    )


def build_p05_manifest(
    train_items: Sequence[BenchmarkItem],
    calibration_items: Sequence[BenchmarkItem],
    validation_items: Sequence[BenchmarkItem],
    action_model: GaxV0Model,
    *,
    context: ExperimentContext,
    sufficiency_config: SufficiencyConfig | None = None,
    target_coverage: float = 0.80,
) -> dict[str, object]:
    _validate_sufficiency_items(train_items, required_split="train")
    _validate_split_only(calibration_items, required_split="calibration")
    _validate_sufficiency_items(validation_items, required_split="validation")
    _validate_coverage(target_coverage)
    config = sufficiency_config or SufficiencyConfig(feature_dim=action_model.config.feature_dim)
    _validate_config_matches_action_model(config, action_model)
    payload: dict[str, object] = {
        "schema_version": _P05_SCHEMA_VERSION,
        "source_revision": _P05_SOURCE_REVISION,
        "experiment_context": asdict(context),
        "action_model_revision": action_model.model_revision,
        "action_model_feature_revision": action_model.feature_revision,
        "train_manifest_sha256": items_manifest_sha256(train_items),
        "calibration_manifest_sha256": items_manifest_sha256(calibration_items),
        "validation_manifest_sha256": items_manifest_sha256(validation_items),
        "sufficiency_config": asdict(config),
        "sufficiency_config_sha256": config.sha256,
        "selectors": [
            "max-probability",
            "entropy-confidence",
            "top1-top2-margin",
            "learned-sufficiency",
        ],
        "target_coverage": target_coverage,
        "threshold_protocol": (
            "calibration-score quantile only; no final-test labels and no formal conformal guarantee"
        ),
        "paper_decision_default": "defer-real-data",
    }
    return {"payload": payload, "sha256": canonical_json_sha256(payload)}


def _selector_features(
    item: BenchmarkItem,
    probabilities: dict[str, float],
    feature_dim: int,
) -> list[float]:
    return [
        *_state_vector(item, feature_dim),
        max_probability_score(probabilities),
        entropy_confidence_score(probabilities),
        top1_top2_margin_score(probabilities),
    ]


def _validate_sufficiency_items(
    items: Sequence[BenchmarkItem],
    *,
    required_split: str,
) -> None:
    _validate_split_only(items, required_split=required_split)
    for item in items:
        if item.gold is None or item.gold.action is None or item.gold.sufficient is None:
            raise ValueError(
                f"item {item.id!r} requires gold action and sufficiency for P05 evaluation"
            )


def _validate_split_only(items: Sequence[BenchmarkItem], *, required_split: str) -> None:
    if not items:
        raise ValueError(f"{required_split} items must not be empty")
    item_ids = [item.id for item in items]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError(f"{required_split} item ids must be unique")
    for item in items:
        if item.split != required_split:
            raise ValueError(
                f"item {item.id!r} has split {item.split!r}; expected {required_split!r}"
            )


def _validate_config_matches_action_model(
    config: SufficiencyConfig,
    action_model: GaxV0Model,
) -> None:
    if config.feature_dim != action_model.config.feature_dim:
        raise ValueError("sufficiency feature_dim must match the action model feature_dim")


def _validate_coverage(value: float) -> None:
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError("target_coverage must be finite and in (0, 1]")


def _validate_probabilities(probabilities: dict[str, float]) -> None:
    if not probabilities:
        raise ValueError("probabilities must not be empty")
    if any(
        not math.isfinite(probability) or not 0.0 <= probability <= 1.0
        for probability in probabilities.values()
    ):
        raise ValueError("probabilities must be finite and in [0, 1]")
    if not math.isclose(
        math.fsum(probabilities.values()),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError("probabilities must sum to 1 within 1e-6")


def _binary_nll(probability: float, label: bool) -> float:
    clipped = min(1.0 - _EPS, max(_EPS, probability))
    return -math.log(clipped if label else 1.0 - clipped)


def _sigmoid(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("sigmoid input must be finite")
    if value >= 0.0:
        exponent = math.exp(-value)
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)
