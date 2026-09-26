from __future__ import annotations

import json
from pathlib import Path

import pytest

from gaxbench.gax_v0 import (
    GaxV0Adapter,
    GaxV0Config,
    load_gax_v0_checkpoint,
    save_gax_v0_checkpoint,
    train_gax_v0,
)
from gaxbench.io import load_items
from gaxbench.runner import run_baseline

FIXTURES = Path(__file__).parent / "fixtures"


def config() -> GaxV0Config:
    return GaxV0Config(feature_dim=16, learning_rate=0.4, epochs=60, seed=7)


def train():
    train_items = load_items(FIXTURES / "gax_v0_train.jsonl")
    validation_items = load_items(FIXTURES / "gax_v0_validation.jsonl")
    return train_gax_v0(train_items, config(), validation_items=validation_items)


def test_training_reduces_loss_and_learns_synthetic_mapping() -> None:
    result = train()
    assert result.history[-1].mean_nll < result.history[0].mean_nll

    validation = load_items(FIXTURES / "gax_v0_validation.jsonl")
    run = run_baseline(validation, GaxV0Adapter(result.model))
    assert run.failed == 0
    assert run.evaluation_error is None
    assert run.action_metrics is not None
    assert run.action_metrics.accuracy == pytest.approx(1.0)


def test_fixed_seed_training_is_deterministic() -> None:
    first = train()
    second = train()
    assert first.training_manifest_sha256 == second.training_manifest_sha256
    assert first.model.model_revision == second.model.model_revision
    assert first.history == second.history


def test_checkpoint_round_trip_preserves_predictions(tmp_path: Path) -> None:
    result = train()
    checkpoint = tmp_path / "gax-v0.json"
    saved_sha = save_gax_v0_checkpoint(checkpoint, result)
    loaded, loaded_sha = load_gax_v0_checkpoint(checkpoint)
    assert saved_sha == loaded_sha
    assert loaded.model_revision == result.model.model_revision

    validation = load_items(FIXTURES / "gax_v0_validation.jsonl")
    for item in validation:
        assert loaded.probabilities(item) == result.model.probabilities(item)


def test_action_permutation_preserves_id_probability_alignment() -> None:
    result = train()
    item = load_items(FIXTURES / "gax_v0_validation.jsonl")[0]
    reversed_item = item.model_copy(update={"actions": list(reversed(item.actions))})
    assert result.model.probabilities(item) == result.model.probabilities(reversed_item)


def test_training_rejects_non_training_split() -> None:
    validation = load_items(FIXTURES / "gax_v0_validation.jsonl")
    with pytest.raises(ValueError, match="expected 'train'"):
        train_gax_v0(validation, config())


def test_checkpoint_integrity_failure_is_visible(tmp_path: Path) -> None:
    result = train()
    checkpoint = tmp_path / "gax-v0.json"
    save_gax_v0_checkpoint(checkpoint, result)
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["payload"]["model"]["weights"][0][0] += 1.0
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity mismatch"):
        load_gax_v0_checkpoint(checkpoint)


def test_loaded_model_works_through_existing_gaxbench_runner(tmp_path: Path) -> None:
    result = train()
    checkpoint = tmp_path / "gax-v0.json"
    artifact_sha = save_gax_v0_checkpoint(checkpoint, result)
    model, loaded_sha = load_gax_v0_checkpoint(checkpoint)
    assert artifact_sha == loaded_sha

    validation = load_items(FIXTURES / "gax_v0_validation.jsonl")
    run = run_baseline(validation, GaxV0Adapter(model, artifact_sha256=loaded_sha))
    assert run.identity.model_id == "gax-bilinear-v0"
    assert run.identity.artifact_sha256 == loaded_sha
    assert run.completed == len(validation)
    assert run.failed == 0
