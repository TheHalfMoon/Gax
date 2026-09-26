from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def test_interventions_cli_smoke() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "gaxbench.cli",
            "interventions-evaluate",
            "--items",
            str(FIXTURES / "p06_items.jsonl"),
            "--predictions",
            str(FIXTURES / "p06_predictions.jsonl"),
            "--manifest",
            str(FIXTURES / "p06_interventions.json"),
            "--stability-tv-threshold",
            "0.05",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["metrics"]["n_pairs"] == 5
    assert payload["metrics"]["bias_trap_rate"] == 0.0
    assert payload["metrics"]["same_top1_agreement"] == 1.0
    assert payload["metrics"]["abstain_success_rate"] == 1.0
    assert payload["metrics"]["directional_success_rate"] == 1.0


def test_interventions_cli_rejects_test_split(tmp_path: Path) -> None:
    source = (FIXTURES / "p06_items.jsonl").read_text(encoding="utf-8")
    test_items = tmp_path / "test-items.jsonl"
    test_source = source.replace('"split":"validation"', '"split":"test"')
    test_items.write_text(test_source, encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "gaxbench.cli",
            "interventions-evaluate",
            "--items",
            str(test_items),
            "--predictions",
            str(FIXTURES / "p06_predictions.jsonl"),
            "--manifest",
            str(FIXTURES / "p06_interventions.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "final-test evaluation is P08 work" in completed.stderr
