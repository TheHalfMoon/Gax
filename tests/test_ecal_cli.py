from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def base_command() -> list[str]:
    return [
        sys.executable,
        "-m",
        "gaxbench.gax_cli",
    ]


def data_args() -> list[str]:
    return [
        "--train-items",
        str(FIXTURES / "ecal_train.jsonl"),
        "--validation-items",
        str(FIXTURES / "ecal_validation.jsonl"),
        "--replay-items",
        str(FIXTURES / "ecal_replay.jsonl"),
        "--retention-items",
        str(FIXTURES / "ecal_retention.jsonl"),
        "--feature-dim",
        "8",
        "--epochs",
        "3",
        "--learning-rate",
        "0.05",
        "--seed",
        "13",
    ]


def test_ecal_manifest_cli_is_deterministic() -> None:
    command = [*base_command(), "ecal-manifest", *data_args()]
    first = subprocess.run(command, check=True, capture_output=True, text=True)
    second = subprocess.run(command, check=True, capture_output=True, text=True)
    assert first.stdout == second.stdout
    payload = json.loads(first.stdout)
    assert len(payload["sha256"]) == 64
    arms = payload["payload"]["ablation_arms"]
    assert [arm["component"] for arm in arms] == [
        "bidirectional",
        "hard-negative",
        "evidence",
        "proper-scoring",
        "replay",
    ]


def test_ecal_replay_ablation_cli_uses_matched_step_budget() -> None:
    command = [
        *base_command(),
        "ecal-ablate",
        "replay",
        *data_args(),
        "--ece-bins",
        "5",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    assert payload["component"] == "replay"
    assert payload["optimizer_steps_equal"] is True
    assert payload["control_development"]["n"] == 2
    assert payload["treatment_development"]["n"] == 2
    assert payload["control_retention"]["n"] == 2
    assert payload["treatment_retention"]["n"] == 2
