from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).parents[1]


def test_cli_baseline_run_writes_packet(tmp_path: Path) -> None:
    output = tmp_path / "packet"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "gaxbench.cli",
            "baseline-run",
            "--items",
            str(FIXTURES / "items.jsonl"),
            "--adapter",
            "uniform",
            "--output-dir",
            str(output),
            "--repo-revision",
            "test-sha",
            "--seed",
            "11",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads(completed.stdout)
    assert manifest["counts"] == {"requested": 3, "completed": 3, "failed": 0}
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["action"]["n"] == 3


def test_cli_baseline_registry_smoke() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "gaxbench.cli",
            "baseline-registry",
            "--registry",
            str(ROOT / "registry" / "baselines.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    ids = {entry["id"] for entry in payload["entries"]}
    assert {"clm", "laya", "decider", "jev"} <= ids
