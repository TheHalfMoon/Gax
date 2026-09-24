from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from gaxbench.provenance import build_file_manifest, canonical_json_sha256, sha256_file
from gaxbench.runner import BaselineRunResult


def write_evidence_packet(
    output_dir: str | Path,
    result: BaselineRunResult,
    *,
    items_path: str | Path,
    repo_revision: str,
    dirty_tree: bool,
    command: list[str],
    seed: int,
    calibration_revision: str,
    runtime_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    benchmark_path = Path(items_path)
    runtime = runtime_metadata or _runtime_metadata()

    predictions_path = output / "predictions.jsonl"
    failures_path = output / "failures.json"
    timings_path = output / "timings.json"
    metrics_path = output / "metrics.json"

    _write_predictions(predictions_path, result)
    _write_json(failures_path, [asdict(failure) for failure in result.failures])
    _write_json(timings_path, [asdict(timing) for timing in result.timings])
    _write_json(
        metrics_path,
        {
            "action": None if result.action_metrics is None else asdict(result.action_metrics),
            "abstention": (
                None if result.abstention_metrics is None else asdict(result.abstention_metrics)
            ),
            "evaluation_error": result.evaluation_error,
        },
    )

    identity = asdict(result.identity)
    stable_identity = {
        "repo_revision": repo_revision,
        "dirty_tree": dirty_tree,
        "adapter": identity,
        "benchmark_sha256": sha256_file(benchmark_path),
        "command": command,
        "seed": seed,
        "calibration_revision": calibration_revision,
        "runtime": runtime,
        "counts": {
            "requested": result.requested,
            "completed": result.completed,
            "failed": result.failed,
        },
    }
    packet_id = canonical_json_sha256(stable_identity)

    files = [predictions_path, failures_path, timings_path, metrics_path]
    manifest = {
        "schema_version": "0.1",
        "packet_id": packet_id,
        **stable_identity,
        "benchmark": {
            "path": benchmark_path.as_posix(),
            "sha256": sha256_file(benchmark_path),
        },
        "files": build_file_manifest(files, root=output),
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def _write_predictions(path: Path, result: BaselineRunResult) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for prediction in result.predictions:
            payload = json.dumps(
                prediction.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            handle.write(payload)
            handle.write("\n")


def _write_json(path: Path, value: Any) -> None:
    payload = json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    path.write_text(payload + "\n", encoding="utf-8", newline="\n")


def _runtime_metadata() -> dict[str, str]:
    return {
        "python_implementation": sys.implementation.name,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
