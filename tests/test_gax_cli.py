from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def test_gax_train_and_evaluate_cli(tmp_path: Path) -> None:
    checkpoint = tmp_path / "gax-v0.json"
    train = subprocess.run(
        [
            sys.executable,
            "-m",
            "gaxbench.gax_cli",
            "train",
            "--train-items",
            str(FIXTURES / "gax_v0_train.jsonl"),
            "--validation-items",
            str(FIXTURES / "gax_v0_validation.jsonl"),
            "--checkpoint",
            str(checkpoint),
            "--feature-dim",
            "16",
            "--learning-rate",
            "0.4",
            "--epochs",
            "60",
            "--seed",
            "7",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    train_payload = json.loads(train.stdout)
    assert train_payload["architecture_id"] == "gax-bilinear-v0"
    assert checkpoint.is_file()

    evaluate = subprocess.run(
        [
            sys.executable,
            "-m",
            "gaxbench.gax_cli",
            "evaluate",
            "--items",
            str(FIXTURES / "gax_v0_validation.jsonl"),
            "--checkpoint",
            str(checkpoint),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    evaluate_payload = json.loads(evaluate.stdout)
    assert evaluate_payload["completed"] == 2
    assert evaluate_payload["failed"] == 0
    assert evaluate_payload["action_metrics"]["accuracy"] == 1.0
