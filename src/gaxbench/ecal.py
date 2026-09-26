from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Literal

from pydantic import JsonValue

from gaxbench.baselines import AdapterIdentity
from gaxbench.gax_v0 import (
    GaxV0Config,
    GaxV0Model,
    _action_vector,
    _softmax,
    _state_vector,
    _validate_training_items,
)
from gaxbench.metrics import ActionMetrics
from gaxbench.provenance import canonical_json_sha256
from gaxbench.runner import BaselineRunResult, run_baseline
from gaxbench.schema import BenchmarkItem, Evidence, Prediction

NegativePolicy = Literal["none", "hard", "random"]
PaperDecision = Literal["keep", "reject", "defer-real-data"]

_ECAL_SCHEMA_VERSION = "0.1"
_ECAL_SOURCE_REVISION = "gax-p04"
_NEGATIVE_RNG_XOR = 0xEC4A1


@dataclass(frozen=True)
class ExperimentContext:
    git_sha: str
    compute_provenance: str

    def __post_init__(self) -> None:
        if len(self.git_sha) != 40 or any(
            character not in "0123456789abcdef" for character in self.git_sha
        ):
            raise ValueError("git_sha must be a 40-character lowercase hexadecimal SHA")
        if not self.compute_provenance.strip():
            raise ValueError("compute_provenance must be non-empty")


@dataclass(frozen=True)
class EcalConfig:
    base: GaxV0Config = field(default_factory=GaxV0Config)
    action_weight: float = 1.0
    bidirectional_weight: float = 0.0
    hard_negative_weight: float = 0.0
    evidence_weight: float = 0.0
    proper_weight: float = 0.0
    hard_negative_margin: float = 0.25
    evidence_margin: float = 0.10
    negative_policy: NegativePolicy = "none"
    replay_ratio: float = 0.0

    def __post_init__(self) -> None:
        _require_finite_positive(self.action_weight, "action_weight")
        for name, value in (
            ("bidirectional_weight", self.bidirectional_weight),
            ("hard_negative_weight", self.hard_negative_weight),
            ("evidence_weight", self.evidence_weight),
            ("proper_weight", self.proper_weight),
            ("hard_negative_margin", self.hard_negative_margin),
            ("evidence_margin", self.evidence_margin),
        ):
            _require_finite_nonnegative(value, name)
        if self.negative_policy not in {"none", "hard", "random"}:
            raise ValueError("negative_policy must be one of: none, hard, random")
        if self.hard_negative_weight > 0.0 and self.negative_policy == "none":
            raise ValueError("hard_negative_weight > 0 requires hard or random negative_policy")
        if not math.isfinite(self.replay_ratio) or not 0.0 <= self.replay_ratio < 1.0:
            raise ValueError("replay_ratio must be finite and in [0, 1)")

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(asdict(self))


@dataclass(frozen=True)
class EcalEpochRecord:
    epoch: int
    mean_action_nll: float
    mean_bidirectional_loss: float
    mean_hard_negative_loss: float
    mean_evidence_loss: float
    mean_proper_loss: float
    target_steps: int
    replay_steps: int


@dataclass(frozen=True)
class EcalTrainingResult:
    model: GaxV0Model
    history: tuple[EcalEpochRecord, ...]
    target_manifest_sha256: str
    replay_manifest_sha256: str | None
    config_sha256: str
    optimizer_steps: int
    target_steps: int
    replay_steps: int


@dataclass(frozen=True)
class AblationDecision:
    component: str
    paper_decision: PaperDecision
    rationale: str


@dataclass(frozen=True)
class EcalPredictionRecord:
    item_id: str
    probabilities: dict[str, float]
    evidence_support: float | None
    information_sufficiency: float | None
    abstain: bool
    metadata: dict[str, JsonValue]


@dataclass(frozen=True)
class MatchedAblationResult:
    component: str
    experiment_context: ExperimentContext
    control_config_sha256: str
    treatment_config_sha256: str
    control_model_revision: str
    treatment_model_revision: str
    control_optimizer_steps: int
    treatment_optimizer_steps: int
    control_target_steps: int
    treatment_target_steps: int
    control_replay_steps: int
    treatment_replay_steps: int
    control_development: ActionMetrics
    treatment_development: ActionMetrics
    control_development_predictions: tuple[EcalPredictionRecord, ...]
    treatment_development_predictions: tuple[EcalPredictionRecord, ...]
    control_retention: ActionMetrics | None
    treatment_retention: ActionMetrics | None
    control_retention_predictions: tuple[EcalPredictionRecord, ...]
    treatment_retention_predictions: tuple[EcalPredictionRecord, ...]
    optimizer_steps_equal: bool


class EcalAdapter:
    def __init__(self, result: EcalTrainingResult) -> None:
        self._result = result

    @property
    def identity(self) -> AdapterIdentity:
        model = self._result.model
        return AdapterIdentity(
            name="gax-ecal",
            adapter_version=_ECAL_SCHEMA_VERSION,
            deterministic=True,
            model_id=model.architecture_id,
            model_revision=model.model_revision,
            tokenizer_revision=model.feature_revision,
            source_revision=f"{_ECAL_SOURCE_REVISION}:{self._result.config_sha256[:16]}",
        )

    def prepare(self, items: Sequence[BenchmarkItem]) -> None:
        if not items:
            raise ValueError("items must not be empty")

    def predict(self, item: BenchmarkItem) -> Prediction:
        return self._result.model.predict(item)


def train_ecal(
    train_items: Sequence[BenchmarkItem],
    config: EcalConfig,
    *,
    validation_items: Sequence[BenchmarkItem] = (),
    replay_items: Sequence[BenchmarkItem] = (),
) -> EcalTrainingResult:
    _validate_training_items(train_items, required_split="train")
    if validation_items:
        _validate_training_items(validation_items, required_split="validation")
    if replay_items:
        _validate_training_items(replay_items, required_split="train")
    if config.replay_ratio > 0.0 and not replay_items:
        raise ValueError("replay_ratio > 0 requires replay_items")

    model = GaxV0Model(config.base)
    schedule_rng = random.Random(config.base.seed)
    negative_rng = random.Random(config.base.seed ^ _NEGATIVE_RNG_XOR)
    target_pool = tuple(train_items)
    history: list[EcalEpochRecord] = []
    total_target_steps = 0
    total_replay_steps = 0

    for epoch in range(1, config.base.epochs + 1):
        ordered_targets = list(train_items)
        schedule_rng.shuffle(ordered_targets)
        schedule = _build_replay_schedule(
            ordered_targets,
            replay_items,
            ratio=config.replay_ratio,
            rng=schedule_rng,
        )
        action_losses: list[float] = []
        bidirectional_losses: list[float] = []
        hard_losses: list[float] = []
        evidence_losses: list[float] = []
        proper_losses: list[float] = []
        epoch_target_steps = 0
        epoch_replay_steps = 0

        for item, is_replay in schedule:
            gradient = _zero_matrix(config.base.feature_dim)
            action_loss, base_gradient = _action_loss_and_gradient(model, item)
            _add_matrix(gradient, base_gradient, config.action_weight)
            action_losses.append(action_loss)

            if is_replay:
                epoch_replay_steps += 1
            else:
                epoch_target_steps += 1

                if config.bidirectional_weight > 0.0:
                    loss, component_gradient = _bidirectional_loss_and_gradient(
                        model,
                        item,
                        target_pool,
                    )
                    _add_matrix(gradient, component_gradient, config.bidirectional_weight)
                    bidirectional_losses.append(loss)

                if config.hard_negative_weight > 0.0:
                    loss, component_gradient = _hard_negative_loss_and_gradient(
                        model,
                        item,
                        margin=config.hard_negative_margin,
                        policy=config.negative_policy,
                        rng=negative_rng,
                    )
                    _add_matrix(gradient, component_gradient, config.hard_negative_weight)
                    hard_losses.append(loss)

                if config.evidence_weight > 0.0:
                    loss, component_gradient = _evidence_loss_and_gradient(
                        model,
                        item,
                        margin=config.evidence_margin,
                    )
                    _add_matrix(gradient, component_gradient, config.evidence_weight)
                    evidence_losses.append(loss)

                if config.proper_weight > 0.0:
                    loss, component_gradient = _proper_loss_and_gradient(model, item)
                    _add_matrix(gradient, component_gradient, config.proper_weight)
                    proper_losses.append(loss)

            _apply_gradient_matrix(model, gradient)

        total_target_steps += epoch_target_steps
        total_replay_steps += epoch_replay_steps
        history.append(
            EcalEpochRecord(
                epoch=epoch,
                mean_action_nll=_mean(action_losses),
                mean_bidirectional_loss=_mean(bidirectional_losses),
                mean_hard_negative_loss=_mean(hard_losses),
                mean_evidence_loss=_mean(evidence_losses),
                mean_proper_loss=_mean(proper_losses),
                target_steps=epoch_target_steps,
                replay_steps=epoch_replay_steps,
            )
        )

    return EcalTrainingResult(
        model=model,
        history=tuple(history),
        target_manifest_sha256=items_manifest_sha256(train_items),
        replay_manifest_sha256=items_manifest_sha256(replay_items) if replay_items else None,
        config_sha256=config.sha256,
        optimizer_steps=total_target_steps + total_replay_steps,
        target_steps=total_target_steps,
        replay_steps=total_replay_steps,
    )


def items_manifest_sha256(items: Sequence[BenchmarkItem]) -> str:
    if not items:
        raise ValueError("item manifest requires at least one item")
    payload = [
        item.model_dump(mode="json")
        for item in sorted(items, key=lambda candidate: candidate.id)
    ]
    return canonical_json_sha256(payload)


def build_p04_ablation_manifest(
    train_items: Sequence[BenchmarkItem],
    validation_items: Sequence[BenchmarkItem],
    *,
    context: ExperimentContext,
    replay_items: Sequence[BenchmarkItem] = (),
    retention_items: Sequence[BenchmarkItem] = (),
    base: GaxV0Config | None = None,
) -> dict[str, object]:
    _validate_training_items(train_items, required_split="train")
    _validate_training_items(validation_items, required_split="validation")
    if replay_items:
        _validate_training_items(replay_items, required_split="train")
    if retention_items:
        _validate_training_items(retention_items, required_split="validation")
    base_config = base if base is not None else GaxV0Config()

    reference = EcalConfig(base=base_config)
    arms: list[dict[str, object]] = [
        _ablation_arm(
            "bidirectional",
            reference,
            EcalConfig(base=base_config, bidirectional_weight=0.25),
            "ranking/generalization gain without material calibration regression",
        ),
        _ablation_arm(
            "hard-negative",
            EcalConfig(
                base=base_config,
                hard_negative_weight=0.25,
                negative_policy="random",
            ),
            EcalConfig(
                base=base_config,
                hard_negative_weight=0.25,
                negative_policy="hard",
            ),
            "hard-candidate discrimination beyond equal-budget random negative control",
        ),
        _ablation_arm(
            "evidence",
            reference,
            EcalConfig(base=base_config, evidence_weight=0.25),
            "benefit survives evidence removal/swap and hidden-relation invariance controls",
        ),
        _ablation_arm(
            "proper-scoring",
            reference,
            EcalConfig(base=base_config, proper_weight=0.25),
            "improve NLL/Brier calibration without unacceptable action-quality regression",
        ),
    ]
    if replay_items:
        arms.append(
            _ablation_arm(
                "replay",
                reference,
                EcalConfig(base=base_config, replay_ratio=0.25),
                "improve frozen prior-capability retention at matched optimizer-step budget",
            )
        )

    payload: dict[str, object] = {
        "schema_version": _ECAL_SCHEMA_VERSION,
        "source_revision": _ECAL_SOURCE_REVISION,
        "experiment_context": asdict(context),
        "rng_protocol": {
            "schedule_stream": "base seed",
            "negative_stream": f"base seed xor 0x{_NEGATIVE_RNG_XOR:x}",
        },
        "train_manifest_sha256": items_manifest_sha256(train_items),
        "validation_manifest_sha256": items_manifest_sha256(validation_items),
        "replay_manifest_sha256": (
            items_manifest_sha256(replay_items) if replay_items else None
        ),
        "retention_manifest_sha256": (
            items_manifest_sha256(retention_items) if retention_items else None
        ),
        "base_config": asdict(base_config),
        "ablation_arms": arms,
        "test_label_policy": "final test labels prohibited for P04 selection",
    }
    return {"payload": payload, "sha256": canonical_json_sha256(payload)}


def run_matched_ablation(
    component: str,
    train_items: Sequence[BenchmarkItem],
    validation_items: Sequence[BenchmarkItem],
    *,
    context: ExperimentContext,
    replay_items: Sequence[BenchmarkItem] = (),
    retention_items: Sequence[BenchmarkItem] = (),
    base: GaxV0Config | None = None,
    ece_bins: int = 15,
) -> MatchedAblationResult:
    _validate_training_items(train_items, required_split="train")
    _validate_training_items(validation_items, required_split="validation")
    if replay_items:
        _validate_training_items(replay_items, required_split="train")
    if retention_items:
        _validate_training_items(retention_items, required_split="validation")

    base_config = base if base is not None else GaxV0Config()
    control_config, treatment_config = _configs_for_component(component, base_config)
    control = train_ecal(
        train_items,
        control_config,
        validation_items=validation_items,
        replay_items=replay_items,
    )
    treatment = train_ecal(
        train_items,
        treatment_config,
        validation_items=validation_items,
        replay_items=replay_items,
    )
    control_development_run = run_baseline(
        validation_items,
        EcalAdapter(control),
        ece_bins=ece_bins,
    )
    treatment_development_run = run_baseline(
        validation_items,
        EcalAdapter(treatment),
        ece_bins=ece_bins,
    )
    control_development = _require_action_metrics(control_development_run)
    treatment_development = _require_action_metrics(treatment_development_run)

    control_retention = None
    treatment_retention = None
    control_retention_predictions: tuple[EcalPredictionRecord, ...] = ()
    treatment_retention_predictions: tuple[EcalPredictionRecord, ...] = ()
    if retention_items:
        control_retention_run = run_baseline(
            retention_items,
            EcalAdapter(control),
            ece_bins=ece_bins,
        )
        treatment_retention_run = run_baseline(
            retention_items,
            EcalAdapter(treatment),
            ece_bins=ece_bins,
        )
        control_retention = _require_action_metrics(control_retention_run)
        treatment_retention = _require_action_metrics(treatment_retention_run)
        control_retention_predictions = _prediction_records(control_retention_run)
        treatment_retention_predictions = _prediction_records(treatment_retention_run)

    return MatchedAblationResult(
        component=component,
        experiment_context=context,
        control_config_sha256=control_config.sha256,
        treatment_config_sha256=treatment_config.sha256,
        control_model_revision=control.model.model_revision,
        treatment_model_revision=treatment.model.model_revision,
        control_optimizer_steps=control.optimizer_steps,
        treatment_optimizer_steps=treatment.optimizer_steps,
        control_target_steps=control.target_steps,
        treatment_target_steps=treatment.target_steps,
        control_replay_steps=control.replay_steps,
        treatment_replay_steps=treatment.replay_steps,
        control_development=control_development,
        treatment_development=treatment_development,
        control_development_predictions=_prediction_records(control_development_run),
        treatment_development_predictions=_prediction_records(treatment_development_run),
        control_retention=control_retention,
        treatment_retention=treatment_retention,
        control_retention_predictions=control_retention_predictions,
        treatment_retention_predictions=treatment_retention_predictions,
        optimizer_steps_equal=control.optimizer_steps == treatment.optimizer_steps,
    )


def default_p04_decisions() -> tuple[AblationDecision, ...]:
    common = (
        "Synthetic mechanistic qualification can validate implementation behavior but "
        "cannot establish a paper contribution; defer the keep/reject decision until "
        "licensed development data and matched P08 evaluation are available."
    )
    return tuple(
        AblationDecision(component=component, paper_decision="defer-real-data", rationale=common)
        for component in (
            "bidirectional",
            "hard-negative",
            "evidence",
            "proper-scoring",
            "replay",
        )
    )


def _action_loss_and_gradient(
    model: GaxV0Model,
    item: BenchmarkItem,
) -> tuple[float, list[list[float]]]:
    assert item.gold is not None and item.gold.action is not None
    actions = sorted(item.actions, key=lambda action: action.id)
    state_vector = _state_vector(item, model.config.feature_dim)
    action_vectors = [_action_vector(action, model.config.feature_dim) for action in actions]
    logits = [model._score(state_vector, vector) for vector in action_vectors]
    probabilities = _softmax(logits)
    gold_index = next(
        index for index, action in enumerate(actions) if action.id == item.gold.action
    )
    loss = -math.log(max(probabilities[gold_index], 1e-12))
    factors = [
        probability - (1.0 if index == gold_index else 0.0)
        for index, probability in enumerate(probabilities)
    ]
    gradient = _zero_matrix(model.config.feature_dim)
    for action_vector, factor in zip(action_vectors, factors, strict=True):
        _add_outer(gradient, state_vector, action_vector, factor)
    return loss, gradient


def _proper_loss_and_gradient(
    model: GaxV0Model,
    item: BenchmarkItem,
) -> tuple[float, list[list[float]]]:
    assert item.gold is not None and item.gold.action is not None
    actions = sorted(item.actions, key=lambda action: action.id)
    state_vector = _state_vector(item, model.config.feature_dim)
    action_vectors = [_action_vector(action, model.config.feature_dim) for action in actions]
    logits = [model._score(state_vector, vector) for vector in action_vectors]
    probabilities = _softmax(logits)
    gold_index = next(
        index for index, action in enumerate(actions) if action.id == item.gold.action
    )
    targets = [1.0 if index == gold_index else 0.0 for index in range(len(actions))]
    errors = [
        probability - target
        for probability, target in zip(probabilities, targets, strict=True)
    ]
    loss = math.fsum(error * error for error in errors)
    factors = _brier_factors(probabilities, targets)
    gradient = _zero_matrix(model.config.feature_dim)
    for action_vector, factor in zip(action_vectors, factors, strict=True):
        _add_outer(gradient, state_vector, action_vector, factor)
    return loss, gradient


def _brier_factors(
    probabilities: Sequence[float],
    targets: Sequence[float],
) -> list[float]:
    if len(probabilities) != len(targets) or not probabilities:
        raise ValueError("probabilities and targets must be non-empty and equally sized")
    if any(not math.isfinite(value) for value in probabilities):
        raise ValueError("probabilities must be finite")
    errors = [
        probability - target
        for probability, target in zip(probabilities, targets, strict=True)
    ]
    projection = math.fsum(
        probability * error
        for probability, error in zip(probabilities, errors, strict=True)
    )
    factors = [
        2.0 * probability * (error - projection)
        for probability, error in zip(probabilities, errors, strict=True)
    ]
    if any(not math.isfinite(value) for value in factors):
        raise ValueError("Brier gradient factors must be finite")
    return factors


def _hard_negative_loss_and_gradient(
    model: GaxV0Model,
    item: BenchmarkItem,
    *,
    margin: float,
    policy: NegativePolicy,
    rng: random.Random,
) -> tuple[float, list[list[float]]]:
    assert item.gold is not None and item.gold.action is not None
    actions = sorted(item.actions, key=lambda action: action.id)
    if len(actions) < 2:
        return 0.0, _zero_matrix(model.config.feature_dim)

    state_vector = _state_vector(item, model.config.feature_dim)
    action_vectors = [_action_vector(action, model.config.feature_dim) for action in actions]
    scores = [model._score(state_vector, vector) for vector in action_vectors]
    gold_index = next(
        index for index, action in enumerate(actions) if action.id == item.gold.action
    )
    negative_index = _select_negative_index(
        len(actions),
        scores,
        gold_index=gold_index,
        policy=policy,
        rng=rng,
    )
    violation = margin - scores[gold_index] + scores[negative_index]
    if violation <= 0.0:
        return 0.0, _zero_matrix(model.config.feature_dim)

    gradient = _zero_matrix(model.config.feature_dim)
    _add_outer(gradient, state_vector, action_vectors[gold_index], -1.0)
    _add_outer(gradient, state_vector, action_vectors[negative_index], 1.0)
    return violation, gradient


def _select_negative_index(
    action_count: int,
    scores: Sequence[float],
    *,
    gold_index: int,
    policy: NegativePolicy,
    rng: random.Random,
) -> int:
    if action_count != len(scores) or action_count < 2:
        raise ValueError("negative selection requires matching action/score sequences")
    candidates = [index for index in range(action_count) if index != gold_index]
    if policy == "hard":
        return max(candidates, key=lambda index: (scores[index], -index))
    if policy == "random":
        return candidates[rng.randrange(len(candidates))]
    raise ValueError("negative selection requires hard or random policy")


def _evidence_loss_and_gradient(
    model: GaxV0Model,
    item: BenchmarkItem,
    *,
    margin: float,
) -> tuple[float, list[list[float]]]:
    assert item.gold is not None and item.gold.action is not None
    if not item.evidence:
        return 0.0, _zero_matrix(model.config.feature_dim)

    gold_action = next(action for action in item.actions if action.id == item.gold.action)
    action_vector = _action_vector(gold_action, model.config.feature_dim)
    without_evidence = item.model_copy(update={"evidence": []})
    state_without = _state_vector(without_evidence, model.config.feature_dim)
    score_without = model._score(state_without, action_vector)

    gradient = _zero_matrix(model.config.feature_dim)
    supervised = 0
    losses: list[float] = []

    for evidence in item.evidence:
        relation = evidence.relation
        if relation == "unknown":
            continue
        visible_evidence = Evidence(
            id=evidence.id,
            relation="unknown",
            text=evidence.text,
            structured=evidence.structured,
            source_ref=evidence.source_ref,
        )
        with_evidence = item.model_copy(update={"evidence": [visible_evidence]})
        state_with = _state_vector(with_evidence, model.config.feature_dim)
        score_with = model._score(state_with, action_vector)
        delta = score_with - score_without

        if relation == "irrelevant":
            loss = 0.5 * delta * delta
            factor = delta
        else:
            sign = 1.0 if relation == "support" else -1.0
            violation = margin - sign * delta
            if violation <= 0.0:
                loss = 0.0
                factor = 0.0
            else:
                loss = violation
                factor = -sign

        if factor != 0.0:
            _add_outer(gradient, state_with, action_vector, factor)
            _add_outer(gradient, state_without, action_vector, -factor)
        losses.append(loss)
        supervised += 1

    if supervised == 0:
        return 0.0, gradient
    _scale_matrix(gradient, 1.0 / supervised)
    return math.fsum(losses) / supervised, gradient


def _bidirectional_loss_and_gradient(
    model: GaxV0Model,
    item: BenchmarkItem,
    pool: Sequence[BenchmarkItem],
) -> tuple[float, list[list[float]]]:
    assert item.gold is not None and item.gold.action is not None
    if not pool:
        raise ValueError("bidirectional pool must not be empty")

    current_action = next(action for action in item.actions if action.id == item.gold.action)
    current_key = _semantic_action_key(current_action.description)
    current_state = _state_vector(item, model.config.feature_dim)
    current_action_vector = _action_vector(current_action, model.config.feature_dim)

    pool_actions: list[list[float]] = []
    pool_action_keys: list[str] = []
    pool_states: list[list[float]] = []
    pool_state_keys: list[str] = []
    for candidate in pool:
        assert candidate.gold is not None and candidate.gold.action is not None
        gold_action = next(
            action for action in candidate.actions if action.id == candidate.gold.action
        )
        key = _semantic_action_key(gold_action.description)
        pool_actions.append(_action_vector(gold_action, model.config.feature_dim))
        pool_action_keys.append(key)
        pool_states.append(_state_vector(candidate, model.config.feature_dim))
        pool_state_keys.append(key)

    state_logits = [model._score(current_state, vector) for vector in pool_actions]
    state_positive = [
        index for index, key in enumerate(pool_action_keys) if key == current_key
    ]
    state_loss, state_factors = _multi_positive_loss_and_factors(
        state_logits,
        state_positive,
    )

    action_logits = [model._score(vector, current_action_vector) for vector in pool_states]
    action_positive = [
        index for index, key in enumerate(pool_state_keys) if key == current_key
    ]
    action_loss, action_factors = _multi_positive_loss_and_factors(
        action_logits,
        action_positive,
    )

    gradient = _zero_matrix(model.config.feature_dim)
    for action_vector, factor in zip(pool_actions, state_factors, strict=True):
        _add_outer(gradient, current_state, action_vector, 0.5 * factor)
    for state_vector, factor in zip(pool_states, action_factors, strict=True):
        _add_outer(gradient, state_vector, current_action_vector, 0.5 * factor)
    return 0.5 * (state_loss + action_loss), gradient


def _multi_positive_loss_and_factors(
    logits: Sequence[float],
    positive_indices: Sequence[int],
) -> tuple[float, list[float]]:
    if not logits or any(not math.isfinite(value) for value in logits):
        raise ValueError("multi-positive logits must be finite and non-empty")
    positives = sorted(set(positive_indices))
    if not positives or positives[-1] >= len(logits) or positives[0] < 0:
        raise ValueError("positive_indices must identify at least one valid logit")

    probabilities = _softmax(logits)
    positive_logits = [logits[index] for index in positives]
    positive_probabilities = _softmax(positive_logits)
    positive_mass = math.fsum(probabilities[index] for index in positives)
    loss = -math.log(max(positive_mass, 1e-12))

    conditional = {
        index: probability
        for index, probability in zip(positives, positive_probabilities, strict=True)
    }
    factors = [
        probability - conditional.get(index, 0.0)
        for index, probability in enumerate(probabilities)
    ]
    return loss, factors


def _build_replay_schedule(
    target_items: Sequence[BenchmarkItem],
    replay_items: Sequence[BenchmarkItem],
    *,
    ratio: float,
    rng: random.Random,
) -> list[tuple[BenchmarkItem, bool]]:
    if not target_items:
        raise ValueError("target_items must not be empty")
    if not 0.0 <= ratio < 1.0:
        raise ValueError("ratio must be in [0, 1)")
    schedule = [(item, False) for item in target_items]
    if ratio == 0.0:
        return schedule
    if not replay_items:
        raise ValueError("replay_items are required when ratio > 0")
    if len(target_items) < 2:
        raise ValueError("replay requires at least two target items for equal-step replacement")

    replay_count = min(len(target_items) - 1, int(round(len(target_items) * ratio)))
    if replay_count == 0:
        replay_count = 1
    positions = sorted(rng.sample(range(len(target_items)), replay_count))
    replay_order = list(replay_items)
    rng.shuffle(replay_order)
    for offset, position in enumerate(positions):
        schedule[position] = (replay_order[offset % len(replay_order)], True)
    return schedule


def _configs_for_component(
    component: str,
    base: GaxV0Config,
) -> tuple[EcalConfig, EcalConfig]:
    reference = EcalConfig(base=base)
    if component == "bidirectional":
        return reference, EcalConfig(base=base, bidirectional_weight=0.25)
    if component == "hard-negative":
        return (
            EcalConfig(
                base=base,
                hard_negative_weight=0.25,
                negative_policy="random",
            ),
            EcalConfig(
                base=base,
                hard_negative_weight=0.25,
                negative_policy="hard",
            ),
        )
    if component == "evidence":
        return reference, EcalConfig(base=base, evidence_weight=0.25)
    if component == "proper-scoring":
        return reference, EcalConfig(base=base, proper_weight=0.25)
    if component == "replay":
        return reference, EcalConfig(base=base, replay_ratio=0.25)
    raise ValueError(f"unknown ECAL component: {component}")


def _require_action_metrics(run_result: BaselineRunResult) -> ActionMetrics:
    action_metrics = run_result.action_metrics
    if (
        run_result.failed
        or run_result.evaluation_error is not None
        or action_metrics is None
    ):
        raise ValueError("matched ablation evaluation did not produce complete action metrics")
    return action_metrics


def _prediction_records(run_result: BaselineRunResult) -> tuple[EcalPredictionRecord, ...]:
    if run_result.failed or run_result.evaluation_error is not None:
        raise ValueError("cannot serialize predictions from an incomplete ablation run")
    return tuple(
        EcalPredictionRecord(
            item_id=prediction.item_id,
            probabilities=dict(prediction.probabilities),
            evidence_support=prediction.evidence_support,
            information_sufficiency=prediction.information_sufficiency,
            abstain=prediction.abstain,
            metadata=dict(prediction.metadata),
        )
        for prediction in run_result.predictions
    )


def _ablation_arm(
    component: str,
    control: EcalConfig,
    treatment: EcalConfig,
    gate: str,
) -> dict[str, object]:
    return {
        "component": component,
        "control_config": asdict(control),
        "control_config_sha256": control.sha256,
        "treatment_config": asdict(treatment),
        "treatment_config_sha256": treatment.sha256,
        "matched_budget": "same architecture, seed set, optimizer-step budget, data eligibility",
        "development_gate": gate,
        "paper_decision": "defer-real-data",
    }


def _semantic_action_key(description: str) -> str:
    return " ".join(description.casefold().split())


def _zero_matrix(size: int) -> list[list[float]]:
    return [[0.0 for _ in range(size)] for _ in range(size)]


def _add_outer(
    gradient: list[list[float]],
    state_vector: Sequence[float],
    action_vector: Sequence[float],
    scale: float,
) -> None:
    if scale == 0.0:
        return
    for row_index, state_value in enumerate(state_vector):
        if state_value == 0.0:
            continue
        row = gradient[row_index]
        for column_index, action_value in enumerate(action_vector):
            if action_value != 0.0:
                row[column_index] += scale * state_value * action_value


def _add_matrix(
    target: list[list[float]],
    source: Sequence[Sequence[float]],
    scale: float,
) -> None:
    if scale == 0.0:
        return
    for row_index, row in enumerate(source):
        for column_index, value in enumerate(row):
            target[row_index][column_index] += scale * value


def _scale_matrix(matrix: list[list[float]], scale: float) -> None:
    for row in matrix:
        for index in range(len(row)):
            row[index] *= scale


def _apply_gradient_matrix(model: GaxV0Model, gradient: Sequence[Sequence[float]]) -> None:
    learning_rate = model.config.learning_rate
    if model.config.l2:
        shrink = 1.0 - learning_rate * model.config.l2
        if shrink < 0.0:
            raise ValueError("learning_rate * l2 must be <= 1")
        for weight_row in model._weights:
            for index in range(len(weight_row)):
                weight_row[index] *= shrink

    for row_index, gradient_row in enumerate(gradient):
        weights = model._weights[row_index]
        for column_index, value in enumerate(gradient_row):
            weights[column_index] -= learning_rate * value
            if not math.isfinite(weights[column_index]):
                raise ValueError("ECAL training produced non-finite weights")


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    value = math.fsum(values) / len(values)
    if not math.isfinite(value):
        raise ValueError("ECAL component loss must be finite")
    return value


def _require_finite_positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")


def _require_finite_nonnegative(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
