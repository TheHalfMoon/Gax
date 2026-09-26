from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "gaxbench.p08_cli", *args],
        check=True,
        capture_output=True,
        text=True,
    )


def test_stats_bootstrap_cli_is_deterministic(tmp_path: Path) -> None:
    input_path = tmp_path / "paired.json"
    input_path.write_text(
        json.dumps(
            {
                "values_a": [2.0, 3.0, 4.0],
                "values_b": [1.0, 2.0, 3.0],
            }
        ),
        encoding="utf-8",
    )
    args = (
        "stats-bootstrap",
        "--input",
        str(input_path),
        "--replicates",
        "1000",
        "--seed",
        "17",
        "--ci-level",
        "0.95",
    )
    first = run_cli(*args)
    second = run_cli(*args)
    assert first.stdout == second.stdout
    payload = json.loads(first.stdout)
    assert payload["point_estimate"] == 1.0
    assert payload["confidence_interval"]["lower"] == 1.0
    assert payload["confidence_interval"]["upper"] == 1.0


def test_evidence_rank_and_reliability_cli(tmp_path: Path) -> None:
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "scores": [0.9, 0.8, 0.8, 0.1],
                "labels": [True, True, False, False],
            }
        ),
        encoding="utf-8",
    )
    evidence = json.loads(
        run_cli("evidence-rank", "--input", str(evidence_path)).stdout
    )
    assert evidence["auroc"]["defined"] is True
    assert evidence["auroc"]["value"] == pytest.approx(0.875)
    assert evidence["auprc"]["value"] == pytest.approx(5 / 6)

    reliability_path = tmp_path / "reliability.json"
    reliability_path.write_text(
        json.dumps(
            {
                "confidences": [0.1, 0.2, 0.9],
                "correctness": [True, False, True],
            }
        ),
        encoding="utf-8",
    )
    reliability = json.loads(
        run_cli(
            "reliability",
            "--input",
            str(reliability_path),
            "--bins",
            "4",
        ).stdout
    )
    assert len(reliability) == 4
    assert reliability[1]["count"] == 0
    assert reliability[1]["mean_confidence"] is None


def test_comparison_report_cli_writes_digest_bound_report(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    results_path = tmp_path / "results.json"
    output_path = tmp_path / "report.json"

    registry_path.write_text(
        json.dumps(
            {
                "schema_version": "0.1",
                "experiment_revision": "p08-test",
                "comparisons": [
                    {
                        "id": "cmp-action",
                        "metric": "accuracy",
                        "system_a": "gax",
                        "system_b": "baseline",
                        "direction": "higher",
                        "paired": True,
                        "multiplicity_family": "primary",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    results_path.write_text(
        json.dumps(
            {
                "schema_version": "0.1",
                "experiment_revision": "p08-test",
                "results": [
                    {
                        "comparison_id": "cmp-action",
                        "status": "complete",
                        "requested": 20,
                        "completed": 20,
                        "failed": 0,
                        "estimate_a": 0.8,
                        "estimate_b": 0.7,
                        "difference_a_minus_b": 0.1,
                        "ci_level": 0.95,
                        "ci_lower": 0.02,
                        "ci_upper": 0.18,
                        "evidence_packet_ids": ["packet-a"],
                        "reason": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    completed = run_cli(
        "comparison-report",
        "--registry",
        str(registry_path),
        "--results",
        str(results_path),
        "--repo-revision",
        "a" * 40,
        "--output",
        str(output_path),
    )
    summary = json.loads(completed.stdout)
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert summary["comparison_count"] == 1
    assert len(summary["report_digest"]) == 64
    assert summary["report_digest"] == report["report_digest"]
    assert report["repo_revision"] == "a" * 40


def test_comparison_report_cli_does_not_overwrite_inputs(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    results_path = tmp_path / "results.json"
    registry_path.write_text("{}\n", encoding="utf-8")
    results_path.write_text("{}\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "gaxbench.p08_cli",
            "comparison-report",
            "--registry",
            str(registry_path),
            "--results",
            str(results_path),
            "--repo-revision",
            "a" * 40,
            "--output",
            str(registry_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "--output must differ" in completed.stderr
