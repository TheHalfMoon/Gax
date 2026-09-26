from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from gaxbench.io import load_items

FIXTURES = Path(__file__).parent / "fixtures"


def test_fhir_convert_cli_smoke(tmp_path: Path) -> None:
    output = tmp_path / "items.jsonl"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "gaxbench.cli",
            "fhir-convert",
            "--cases",
            str(FIXTURES / "p07_fhir_cases.jsonl"),
            "--output",
            str(output),
            "--representation",
            "canonical-structured",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)
    items = load_items(output)
    assert summary["converted"] == 2
    assert summary["representation"] == "canonical-structured"
    assert len(items) == 2
    assert items[0].state["fhir_source_version"] == "R5"


def test_fhir_external_convert_cli_preserves_r4(tmp_path: Path) -> None:
    output = tmp_path / "items.jsonl"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "gaxbench.cli",
            "fhir-external-convert",
            "--export",
            str(FIXTURES / "p07_fhir_agentbench_export.jsonl"),
            "--benchmark",
            "FHIR-AgentBench",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)
    items = load_items(output)
    assert summary["benchmark"] == "FHIR-AgentBench"
    assert len(items) == 1
    assert items[0].state["fhir_source_version"] == "R4"


def test_fhir_convert_cli_rejects_test_split(tmp_path: Path) -> None:
    source = (FIXTURES / "p07_fhir_cases.jsonl").read_text(encoding="utf-8")
    test_cases = tmp_path / "test-cases.jsonl"
    test_cases.write_text(
        source.replace('"split":"validation"', '"split":"test"'),
        encoding="utf-8",
    )
    output = tmp_path / "items.jsonl"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "gaxbench.cli",
            "fhir-convert",
            "--cases",
            str(test_cases),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "final-test FHIR cases" in completed.stderr
