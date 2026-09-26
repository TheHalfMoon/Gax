from __future__ import annotations

import math
import random

import pytest

from gaxbench.ecal import (
    EcalConfig,
    _brier_factors,
    _build_replay_schedule,
    _multi_positive_loss_and_factors,
    _select_negative_index,
    build_p04_ablation_manifest,
    default_p04_decisions,
    run_matched_ablation,
    train_ecal,
)
from gaxbench.gax_v0 import GaxV0Config, GaxV0Model, _softmax
from gaxbench.schema import Action, BenchmarkItem, Evidence, Gold, Provenance


def make_item(
    item_id: str,
    signal: str,
    gold_description: str,
    *,
    split: str = "train",
    relation: str | None = None,
    evidence_text: str = "abstract evidence token",
) -> BenchmarkItem:
    actions = [
        Action(id=f"{item_id}-x", description="route alpha"),
        Action(id=f"{item_id}-y", description="route beta"),
        Action(id=f"{item_id}-z", description="route gamma"),
    ]
    gold = next(action.id for action in actions if action.description == gold_description)
    evidence = (
        []
        if relation is None
        else [
            Evidence(
                id=f"{item_id}-e",
                relation=relation,  # type: ignore[arg-type]
                text=evidence_text,
            )
        ]
    )
    return BenchmarkItem(
        id=item_id,
        source_id=f"source-{item_id}",
        split=split,  # type: ignore[arg-type]
        task_family="ecal-mechanistic",
        state={"synthetic_signal": signal},
        actions=actions,
        gold=Gold(action=gold, sufficient=True),
        evidence=evidence,
        provenance=Provenance(
            dataset="ecal-synthetic",
            revision="1",
            license="CC0-1.0",
            transform_revision="1",
        ),
    )


def target_items() -> list[BenchmarkItem]:
    return [
        make_item("a1", "alpha one", "route alpha", relation="support"),
        make_item("a2", "alpha two", "route alpha", relation="irrelevant"),
        make_item("b1", "beta one", "route beta", relation="support"),
        make_item("b2", "beta two", "route beta", relation="contradict"),
    ]


def replay_items() -> list[BenchmarkItem]:
    return [
        make_item("r1", "prior gamma one", "route gamma"),
        make_item("r2", "prior gamma two", "route gamma"),
    ]


def validation_items() -> list[BenchmarkItem]:
    return [
        make_item("v1", "alpha validation", "route alpha", split="validation"),
        make_item("v2", "beta validation", "route beta", split="validation"),
    ]


def retention_items() -> list[BenchmarkItem]:
    return [
        make_item("rv1", "prior gamma validation one", "route gamma", split="validation"),
        make_item("rv2", "prior gamma validation two", "route gamma", split="validation"),
    ]


def test_config_requires_explicit_negative_policy() -> None:
    with pytest.raises(ValueError, match="negative_policy"):
        EcalConfig(hard_negative_weight=0.5)


def test_multi_positive_groups_are_not_false_negatives() -> None:
    loss, factors = _multi_positive_loss_and_factors([2.0, 2.0, 0.0], [0, 1])
    assert loss > 0.0
    assert factors[0] < 0.0
    assert factors[1] < 0.0
    assert factors[2] > 0.0
    assert math.fsum(factors) == pytest.approx(0.0)


def test_hard_negative_tie_break_is_deterministic() -> None:
    index = _select_negative_index(
        3,
        [0.1, 0.8, 0.8],
        gold_index=0,
        policy="hard",
        rng=random.Random(0),
    )
    assert index == 1


def test_brier_gradient_matches_finite_difference() -> None:
    logits = [0.2, -0.1, 0.4]
    targets = [0.0, 1.0, 0.0]
    probabilities = _softmax(logits)
    analytic = _brier_factors(probabilities, targets)
    epsilon = 1e-6

    def loss(values: list[float]) -> float:
        probs = _softmax(values)
        return sum(
            (probability - target) ** 2
            for probability, target in zip(probs, targets, strict=True)
        )

    for index in range(len(logits)):
        upper = list(logits)
        lower = list(logits)
        upper[index] += epsilon
        lower[index] -= epsilon
        numeric = (loss(upper) - loss(lower)) / (2.0 * epsilon)
        assert analytic[index] == pytest.approx(numeric, rel=1e-6, abs=1e-7)


def test_evidence_relation_is_not_model_visible() -> None:
    support = make_item("same", "alpha", "route alpha", relation="support")
    contradict = make_item("same", "alpha", "route alpha", relation="contradict")
    model = GaxV0Model(GaxV0Config(feature_dim=8, seed=7))
    assert model.probabilities(support) == model.probabilities(contradict)


def test_replay_schedule_is_deterministic_and_equal_step() -> None:
    targets = target_items()
    replay = replay_items()
    first = _build_replay_schedule(
        targets,
        replay,
        ratio=0.25,
        rng=random.Random(11),
    )
    second = _build_replay_schedule(
        targets,
        replay,
        ratio=0.25,
        rng=random.Random(11),
    )
    assert [(item.id, flag) for item, flag in first] == [
        (item.id, flag) for item, flag in second
    ]
    assert len(first) == len(targets)
    assert sum(flag for _, flag in first) == 1


def test_ablation_manifest_is_deterministic_and_defers_paper_decisions() -> None:
    train = target_items()
    validation = validation_items()
    replay = replay_items()
    retention = retention_items()
    base = GaxV0Config(feature_dim=8, epochs=3, learning_rate=0.05, seed=3)
    first = build_p04_ablation_manifest(
        train,
        validation,
        replay_items=replay,
        retention_items=retention,
        base=base,
    )
    second = build_p04_ablation_manifest(
        train,
        validation,
        replay_items=replay,
        retention_items=retention,
        base=base,
    )
    assert first == second
    payload = first["payload"]
    assert isinstance(payload, dict)
    assert payload["retention_manifest_sha256"] is not None
    arms = payload["ablation_arms"]
    assert isinstance(arms, list)
    assert [arm["component"] for arm in arms] == [
        "bidirectional",
        "hard-negative",
        "evidence",
        "proper-scoring",
        "replay",
    ]
    assert {decision.paper_decision for decision in default_p04_decisions()} == {
        "defer-real-data"
    }


def test_all_components_train_deterministically_without_hidden_label_inference() -> None:
    train = target_items()
    replay = replay_items()
    validation = validation_items()
    config = EcalConfig(
        base=GaxV0Config(feature_dim=8, epochs=3, learning_rate=0.05, seed=5),
        bidirectional_weight=0.1,
        hard_negative_weight=0.1,
        evidence_weight=0.1,
        proper_weight=0.1,
        negative_policy="hard",
        replay_ratio=0.25,
    )
    first = train_ecal(train, config, validation_items=validation, replay_items=replay)
    second = train_ecal(train, config, validation_items=validation, replay_items=replay)
    assert first.model.model_revision == second.model.model_revision
    assert first.history == second.history
    assert first.optimizer_steps == 12
    assert first.replay_steps == 3
    assert first.target_steps == 9
    assert all(math.isfinite(record.mean_action_nll) for record in first.history)

    support = make_item("probe", "alpha", "route alpha", relation="support")
    contradict = make_item("probe", "alpha", "route alpha", relation="contradict")
    assert first.model.probabilities(support) == first.model.probabilities(contradict)


def test_matched_replay_ablation_reports_development_and_retention() -> None:
    result = run_matched_ablation(
        "replay",
        target_items(),
        validation_items(),
        replay_items=replay_items(),
        retention_items=retention_items(),
        base=GaxV0Config(feature_dim=8, epochs=3, learning_rate=0.05, seed=13),
        ece_bins=5,
    )
    assert result.component == "replay"
    assert result.optimizer_steps_equal
    assert result.control_development.n == 2
    assert result.treatment_development.n == 2
    assert result.control_retention is not None
    assert result.treatment_retention is not None
    assert result.control_retention.n == 2
    assert result.treatment_retention.n == 2
