from __future__ import annotations

from pathlib import Path

import pytest

from gaxbench.ecal import ExperimentContext
from gaxbench.gax_v0 import GaxV0Config, GaxV0Model
from gaxbench.io import load_items
from gaxbench.schema import Evidence, Gold, Provenance
from gaxbench.selective import (
    LearnedSufficiencyModel,
    SufficiencyConfig,
    build_p05_manifest,
    entropy_confidence_score,
    evaluate_selector,
    fit_coverage_policy,
    max_probability_score,
    run_matched_selector_suite,
    selector_score,
    top1_top2_margin_score,
    train_information_sufficiency,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _items(name: str):
    return load_items(FIXTURES / name)


def _action_model() -> GaxV0Model:
    return GaxV0Model(GaxV0Config(feature_dim=16, seed=7))


def _context() -> ExperimentContext:
    return ExperimentContext(
        git_sha="a" * 40,
        compute_provenance="pytest-cpu",
    )


def test_confidence_controls_have_frozen_semantics() -> None:
    probabilities = {"a": 0.8, "b": 0.2}
    assert max_probability_score(probabilities) == pytest.approx(0.8)
    assert top1_top2_margin_score(probabilities) == pytest.approx(0.6)
    entropy_score = entropy_confidence_score(probabilities)
    assert 0.0 < entropy_score < 1.0
    assert entropy_confidence_score({"only": 1.0}) == 1.0


def test_sufficiency_training_is_deterministic() -> None:
    action_model = _action_model()
    config = SufficiencyConfig(feature_dim=16, epochs=20, seed=11)
    train_items = _items("p05_train.jsonl")
    first = train_information_sufficiency(train_items, action_model, config)
    second = train_information_sufficiency(train_items, action_model, config)
    assert first.model.model_revision == second.model.model_revision
    assert first.history == second.history
    assert first.training_manifest_sha256 == second.training_manifest_sha256


def test_learned_score_cannot_see_gold_or_provenance() -> None:
    action_model = _action_model()
    training = train_information_sufficiency(
        _items("p05_train.jsonl"),
        action_model,
        SufficiencyConfig(feature_dim=16, epochs=10, seed=3),
    )
    item = _items("p05_validation.jsonl")[0]
    probabilities = action_model.probabilities(item)
    original = training.model.score(item, probabilities)

    altered = item.model_copy(
        update={
            "gold": Gold(action="beta", sufficient=False),
            "provenance": Provenance(
                dataset="hidden-change",
                revision="different",
                license="CC0-1.0",
                transform_revision="different",
            ),
            "source_id": "hidden-source-change",
            "task_family": "hidden-family-change",
        }
    )
    assert training.model.score(altered, action_model.probabilities(altered)) == pytest.approx(
        original
    )


def test_learned_score_cannot_see_evidence_relation_label() -> None:
    action_model = _action_model()
    model = LearnedSufficiencyModel(SufficiencyConfig(feature_dim=16))
    base = _items("p05_validation.jsonl")[0]
    support = base.model_copy(
        update={"evidence": [Evidence(id="e", text="visible abstract evidence", relation="support")]}
    )
    contradict = base.model_copy(
        update={
            "evidence": [
                Evidence(id="e", text="visible abstract evidence", relation="contradict")
            ]
        }
    )
    support_score = selector_score(
        "learned-sufficiency",
        support,
        action_model,
        sufficiency_model=model,
    )
    contradict_score = selector_score(
        "learned-sufficiency",
        contradict,
        action_model,
        sufficiency_model=model,
    )
    assert support_score == pytest.approx(contradict_score)
    assert action_model.probabilities(support) == pytest.approx(
        action_model.probabilities(contradict)
    )


def test_coverage_policy_does_not_read_calibration_labels() -> None:
    action_model = _action_model()
    calibration = _items("p05_calibration.jsonl")
    original = fit_coverage_policy(
        calibration,
        action_model,
        "max-probability",
        target_coverage=0.8,
    )
    relabeled = [
        item.model_copy(
            update={"gold": Gold(action="alpha", sufficient=not bool(item.gold.sufficient))}
        )
        for item in calibration
    ]
    changed = fit_coverage_policy(
        relabeled,
        action_model,
        "max-probability",
        target_coverage=0.8,
    )
    assert original.threshold == pytest.approx(changed.threshold)
    assert original.calibration_manifest_sha256 != changed.calibration_manifest_sha256


def test_validation_evaluation_reports_both_abstention_harms() -> None:
    action_model = _action_model()
    calibration = _items("p05_calibration.jsonl")
    validation = _items("p05_validation.jsonl")
    policy = fit_coverage_policy(
        calibration,
        action_model,
        "entropy-confidence",
        target_coverage=0.5,
    )
    metrics = evaluate_selector(validation, action_model, policy)
    assert metrics.requested == len(validation)
    assert metrics.completed == len(validation)
    assert metrics.failed == 0
    assert 0.0 <= metrics.actual_coverage <= 1.0
    assert metrics.abstention.unsafe_commit_rate is not None
    assert metrics.abstention.over_abstain_rate is not None


def test_final_test_split_is_rejected_by_p05_selector_evaluation() -> None:
    action_model = _action_model()
    calibration = _items("p05_calibration.jsonl")
    validation = _items("p05_validation.jsonl")
    policy = fit_coverage_policy(
        calibration,
        action_model,
        "max-probability",
        target_coverage=0.8,
    )
    test_items = [item.model_copy(update={"split": "test"}) for item in validation]
    with pytest.raises(ValueError, match="expected 'validation'"):
        evaluate_selector(test_items, action_model, policy)


def test_matched_suite_is_deterministic_and_defers_paper_decision() -> None:
    action_model = _action_model()
    train_items = _items("p05_train.jsonl")
    calibration = _items("p05_calibration.jsonl")
    validation = _items("p05_validation.jsonl")
    kwargs = {
        "context": _context(),
        "sufficiency_config": SufficiencyConfig(feature_dim=16, epochs=20, seed=5),
        "target_coverage": 0.8,
        "ece_bins": 10,
    }
    first = run_matched_selector_suite(
        train_items,
        calibration,
        validation,
        action_model,
        **kwargs,
    )
    second = run_matched_selector_suite(
        train_items,
        calibration,
        validation,
        action_model,
        **kwargs,
    )
    assert first == second
    assert len(first.evaluations) == 4
    assert first.paper_decision == "defer-real-data"
    assert first.strongest_confidence_control in {
        "max-probability",
        "entropy-confidence",
        "top1-top2-margin",
    }


def test_p05_manifest_binds_data_model_context_and_no_test_labels() -> None:
    action_model = _action_model()
    manifest = build_p05_manifest(
        _items("p05_train.jsonl"),
        _items("p05_calibration.jsonl"),
        _items("p05_validation.jsonl"),
        action_model,
        context=_context(),
        sufficiency_config=SufficiencyConfig(feature_dim=16, epochs=20, seed=5),
        target_coverage=0.8,
    )
    payload = manifest["payload"]
    assert isinstance(payload, dict)
    assert payload["experiment_context"] == {
        "git_sha": "a" * 40,
        "compute_provenance": "pytest-cpu",
    }
    assert payload["action_model_revision"] == action_model.model_revision
    assert "test" not in str(payload["threshold_protocol"]).casefold().replace("test labels", "")
    assert manifest["sha256"]
