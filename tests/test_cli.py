from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def test_cli_evaluate_smoke() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "gaxbench.cli",
            "evaluate",
            "--items",
            str(FIXTURES / "items.jsonl"),
            "--predictions",
            str(FIXTURES / "predictions.jsonl"),
            "--ece-bins",
            "10",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["action"]["n"] == 3
    assert payload["abstention"]["unsafe_commit_rate"] == 0.0


def test_cli_audit_smoke() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "gaxbench.cli",
            "audit",
            "--items",
            str(FIXTURES / "items.jsonl"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
    assert payload["duplicate_item_ids"] == []
