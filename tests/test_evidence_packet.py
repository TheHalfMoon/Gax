from __future__ import annotations

from pathlib import Path
from typing import cast

from gaxbench.baselines import UniformBaselineAdapter
from gaxbench.evidence_packet import write_evidence_packet
from gaxbench.io import load_items
from gaxbench.provenance import sha256_file, verify_file_manifest
from gaxbench.runner import run_baseline

FIXTURES = Path(__file__).parent / "fixtures"


def test_evidence_packet_binds_inputs_metrics_and_files(tmp_path: Path) -> None:
    items_path = FIXTURES / "items.jsonl"
    result = run_baseline(load_items(items_path), UniformBaselineAdapter(), ece_bins=10)
    manifest = write_evidence_packet(
        tmp_path,
        result,
        items_path=items_path,
        repo_revision="abc123",
        dirty_tree=False,
        command=["gaxbench", "baseline-run"],
        seed=7,
        calibration_revision="none",
        runtime_metadata={
            "python_implementation": "cpython",
            "python_version": "test",
            "platform": "test",
            "machine": "test",
        },
    )

    assert manifest["repo_revision"] == "abc123"
    benchmark = cast(dict[str, str], manifest["benchmark"])
    assert benchmark["sha256"] == sha256_file(items_path)
    assert manifest["counts"] == {"requested": 3, "completed": 3, "failed": 0}
    files = cast(dict[str, str], manifest["files"])
    assert verify_file_manifest(files, root=tmp_path) == []
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "predictions.jsonl").is_file()


def test_packet_id_is_stable_for_same_identity_and_runtime(tmp_path: Path) -> None:
    items_path = FIXTURES / "items.jsonl"
    result = run_baseline(load_items(items_path), UniformBaselineAdapter())
    runtime = {
        "python_implementation": "cpython",
        "python_version": "test",
        "platform": "test",
        "machine": "test",
    }
    first = write_evidence_packet(
        tmp_path / "a",
        result,
        items_path=items_path,
        repo_revision="abc123",
        dirty_tree=False,
        command=["gaxbench", "baseline-run"],
        seed=0,
        calibration_revision="none",
        runtime_metadata=runtime,
    )
    second = write_evidence_packet(
        tmp_path / "b",
        result,
        items_path=items_path,
        repo_revision="abc123",
        dirty_tree=False,
        command=["gaxbench", "baseline-run"],
        seed=0,
        calibration_revision="none",
        runtime_metadata=runtime,
    )
    assert first["packet_id"] == second["packet_id"]
