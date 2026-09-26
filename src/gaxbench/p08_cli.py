from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from gaxbench.p08_freeze import (
    authorize_final_test,
    load_claim_ledger,
    load_freeze_manifest,
    publishable_claims,
    verify_authorization,
)
from gaxbench.p08_stats import (
    build_primary_report,
    evidence_ranking_metrics,
    load_comparison_result_set,
    load_primary_comparison_registry,
    paired_bootstrap_mean_difference,
    reliability_bins,
    serialize_primary_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gax-p08")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "freeze-validate",
        help="validate a P08 freeze manifest without opening final-test access",
    )
    validate.add_argument("--manifest", required=True)

    authorize = subparsers.add_parser(
        "freeze-authorize",
        help="write a digest-bound authorized copy of a sealed P08 freeze manifest",
    )
    authorize.add_argument("--manifest", required=True)
    authorize.add_argument("--output", required=True)

    claims = subparsers.add_parser(
        "claims-validate",
        help="validate the P08 claim ledger and summarize claim states",
    )
    claims.add_argument("--ledger", required=True)

    bootstrap = subparsers.add_parser(
        "stats-bootstrap",
        help="compute a deterministic paired percentile-bootstrap mean-difference CI",
    )
    bootstrap.add_argument("--input", required=True)
    bootstrap.add_argument("--replicates", type=int, required=True)
    bootstrap.add_argument("--seed", type=int, required=True)
    bootstrap.add_argument("--ci-level", type=float, required=True)

    evidence = subparsers.add_parser(
        "evidence-rank",
        help="compute deterministic evidence AUROC/AUPRC with explicit undefined cases",
    )
    evidence.add_argument("--input", required=True)

    reliability = subparsers.add_parser(
        "reliability",
        help="emit equal-width reliability bins including empty bins",
    )
    reliability.add_argument("--input", required=True)
    reliability.add_argument("--bins", type=int, required=True)

    report = subparsers.add_parser(
        "comparison-report",
        help="build a digest-bound deterministic primary-comparison report",
    )
    report.add_argument("--registry", required=True)
    report.add_argument("--results", required=True)
    report.add_argument("--repo-revision", required=True)
    report.add_argument("--output", required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "freeze-validate":
        manifest = load_freeze_manifest(args.manifest)
        ready = False
        authorization_error: str | None = None
        if manifest.test_access == "sealed":
            try:
                authorize_final_test(manifest)
                ready = True
            except ValueError as exc:
                authorization_error = str(exc)
        else:
            ready = verify_authorization(manifest)
        print(
            json.dumps(
                {
                    "schema_version": manifest.schema_version,
                    "experiment_revision": manifest.experiment_revision,
                    "test_access": manifest.test_access,
                    "authorization_valid": verify_authorization(manifest),
                    "ready_for_authorization": ready,
                    "authorization_error": authorization_error,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.command == "freeze-authorize":
        source = Path(args.manifest).resolve()
        output = Path(args.output).resolve()
        if source == output:
            parser.error("--output must differ from --manifest to preserve the sealed source")
        manifest = authorize_final_test(load_freeze_manifest(source))
        _write_json(output, manifest.model_dump(mode="json"))
        print(
            json.dumps(
                {
                    "output": output.as_posix(),
                    "authorization_digest": manifest.authorization_digest,
                    "test_access": manifest.test_access,
                },
                sort_keys=True,
            )
        )
        return

    if args.command == "claims-validate":
        ledger = load_claim_ledger(args.ledger)
        counts = Counter(entry.status for entry in ledger.entries)
        print(
            json.dumps(
                {
                    "schema_version": ledger.schema_version,
                    "experiment_revision": ledger.experiment_revision,
                    "counts": dict(sorted(counts.items())),
                    "publishable_claim_ids": [
                        entry.id for entry in publishable_claims(ledger)
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.command == "stats-bootstrap":
        payload = _read_object(args.input)
        values_a = _finite_number_list(payload.get("values_a"), "values_a")
        values_b = _finite_number_list(payload.get("values_b"), "values_b")
        result = paired_bootstrap_mean_difference(
            values_a,
            values_b,
            replicates=args.replicates,
            seed=args.seed,
            ci_level=args.ci_level,
        )
        print(json.dumps(asdict(result), indent=2, sort_keys=True, allow_nan=False))
        return

    if args.command == "evidence-rank":
        payload = _read_object(args.input)
        scores = _finite_number_list(payload.get("scores"), "scores")
        labels = _bool_list(payload.get("labels"), "labels")
        result = evidence_ranking_metrics(scores, labels)
        print(json.dumps(asdict(result), indent=2, sort_keys=True, allow_nan=False))
        return

    if args.command == "reliability":
        payload = _read_object(args.input)
        confidences = _finite_number_list(
            payload.get("confidences"),
            "confidences",
        )
        correctness = _bool_list(payload.get("correctness"), "correctness")
        result = reliability_bins(confidences, correctness, bins=args.bins)
        print(
            json.dumps(
                [asdict(entry) for entry in result],
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return

    if args.command == "comparison-report":
        registry_path = Path(args.registry).resolve()
        results_path = Path(args.results).resolve()
        output = Path(args.output).resolve()
        if output in {registry_path, results_path}:
            parser.error("--output must differ from --registry and --results")
        report = build_primary_report(
            load_primary_comparison_registry(registry_path),
            load_comparison_result_set(results_path),
            repo_revision=args.repo_revision,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            serialize_primary_report(report),
            encoding="utf-8",
            newline="\n",
        )
        print(
            json.dumps(
                {
                    "output": output.as_posix(),
                    "report_digest": report.report_digest,
                    "comparison_count": len(report.results),
                },
                sort_keys=True,
            )
        )
        return

    raise AssertionError(f"unhandled command: {args.command}")


def _read_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("input JSON must be an object")
    return payload


def _finite_number_list(value: Any, field: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    output: list[float] = []
    for entry in value:
        if isinstance(entry, bool) or not isinstance(entry, (int, float)):
            raise ValueError(f"{field} entries must be numbers")
        converted = float(entry)
        if not math.isfinite(converted):
            raise ValueError(f"{field} entries must be finite")
        output.append(converted)
    return output


def _bool_list(value: Any, field: str) -> list[bool]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    if any(not isinstance(entry, bool) for entry in value):
        raise ValueError(f"{field} entries must be booleans")
    return list(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
