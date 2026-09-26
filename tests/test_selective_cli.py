from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def _train_checkpoint(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "gax-v0.json"
    subprocess.run(
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
            "20",
            "--seed",
            "7",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return checkpoint


def _selective_base_args(checkpoint: Path) -> list[str]:
    return [
        "--train-items",
        str(FIXTURES / "p05_train.jsonl"),
        "--calibration-items",
        str(FIXTURES / "p05_calibration.jsonl"),
        "--validation-items",
        str(FIXTURES / "p05_validation.jsonl"),
        "--checkpoint",
        str(checkpoint),
        "--target-coverage",
        "0.75",
        "--sufficiency-epochs",
        "12",
        "--sufficiency-seed",
        "5",
        "--git-sha",
        "a" * 40,
        "--compute-provenance",
        "pytest-cpu",
    ]


def test_selective_manifest_cli_binds_checkpoint_and_context(tmp_path: Path) -> None:
    checkpoint = _train_checkpoint(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "gaxbench.gax_cli",
            "selective-manifest",
            *_selective_base_args(checkpoint),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert len(payload["checkpoint_sha256"]) == 64
    manifest = payload["manifest"]
    assert len(manifest["sha256"]) == 64
    assert manifest["payload"]["experiment_context"]["git_sha"] == "a" * 40
    assert manifest["payload"]["target_coverage"] == 0.75


def test_selective_evaluate_cli_reports_all_matched_selectors(tmp_path: Path) -> None:
    checkpoint = _train_checkpoint(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "gaxbench.gax_cli",
            "selective-evaluate",
            *_selective_base_args(checkpoint),
            "--ece-bins",
            "10",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    result = payload["result"]
    assert result["paper_decision"] == "defer-real-data"
    assert len(result["evaluations"]) == 4
    assert {entry["selector"] for entry in result["evaluations"]} == {
        "max-probability",
        "entropy-confidence",
        "top1-top2-margin",
        "learned-sufficiency",
    }
    assert all(entry["requested"] == 8 for entry in result["evaluations"])
    assert all(entry["failed"] == 0 for entry in result["evaluations"])
