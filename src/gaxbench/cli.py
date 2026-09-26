from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from gaxbench.audit import audit_split_integrity
from gaxbench.baseline_registry import load_baseline_registry
from gaxbench.baselines import (
    BaselineAdapter,
    LexicographicBaselineAdapter,
    PredictionFileAdapter,
    UniformBaselineAdapter,
)
from gaxbench.ecal import ExperimentContext
from gaxbench.evidence_packet import write_evidence_packet
from gaxbench.external_adapters import (
    JSONCommandAdapter,
    TypeSafeHTTPAdapter,
    TypeSafeHTTPConfig,
)
from gaxbench.fhir import (
    FHIRDecisionCase,
    FHIRRepresentation,
    fhir_case_to_benchmark_item,
)
from gaxbench.fhir_adapters import (
    BenchmarkName,
    external_task_to_case,
    load_external_fhir_export,
)
from gaxbench.intervention_run import build_intervention_run_manifest
from gaxbench.interventions import evaluate_interventions, load_intervention_manifest
from gaxbench.io import dump_jsonl, load_items, load_jsonl, load_predictions
from gaxbench.metrics import evaluate_abstention, evaluate_action_predictions
from gaxbench.runner import run_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gaxbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("evaluate", help="evaluate predictions against labeled items")
    evaluate.add_argument("--items", required=True)
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--ece-bins", type=int, default=15)

    audit = subparsers.add_parser("audit", help="audit exact cross-split leakage")
    audit.add_argument("--items", required=True)

    interventions = subparsers.add_parser(
        "interventions-evaluate",
        help="evaluate paired P06 evidence/counterfactual interventions",
    )
    interventions.add_argument("--items", required=True)
    interventions.add_argument("--predictions", required=True)
    interventions.add_argument("--manifest", required=True)
    interventions.add_argument("--stability-tv-threshold", type=float, default=0.05)
    interventions.add_argument("--git-sha", required=True)
    interventions.add_argument("--compute-provenance", required=True)

    fhir_convert = subparsers.add_parser(
        "fhir-convert",
        help="convert P07 FHIR decision cases into canonical GAXBench items",
    )
    fhir_convert.add_argument("--cases", required=True)
    fhir_convert.add_argument("--output", required=True)
    fhir_convert.add_argument(
        "--representation",
        default="canonical-structured",
        choices=[
            "canonical-structured",
            "canonical-with-narrative",
            "source-order-json",
            "flat-text",
        ],
    )

    fhir_external = subparsers.add_parser(
        "fhir-external-convert",
        help="convert a frozen local benchmark export into canonical GAXBench items",
    )
    fhir_external.add_argument("--export", required=True)
    fhir_external.add_argument(
        "--benchmark",
        required=True,
        choices=["MedAgentBench", "FHIR-AgentBench"],
    )
    fhir_external.add_argument("--output", required=True)
    fhir_external.add_argument(
        "--representation",
        default="canonical-structured",
        choices=[
            "canonical-structured",
            "canonical-with-narrative",
            "source-order-json",
            "flat-text",
        ],
    )

    baseline = subparsers.add_parser(
        "baseline-run",
        help="run one typed baseline and write a reproducible evidence packet",
    )
    baseline.add_argument("--items", required=True)
    baseline.add_argument(
        "--adapter",
        required=True,
        choices=[
            "uniform",
            "lexicographic",
            "prediction-file",
            "typesafe-http",
            "json-command",
        ],
    )
    baseline.add_argument("--prediction-file")
    baseline.add_argument("--adapter-name", default="external-predictions")
    baseline.add_argument("--adapter-version", default="0.1")
    baseline.add_argument("--model-id")
    baseline.add_argument("--model-revision")
    baseline.add_argument("--tokenizer-revision")
    baseline.add_argument("--source-revision")
    baseline.add_argument("--base-url")
    baseline.add_argument("--api-key-env")
    baseline.add_argument("--timeout-seconds", type=float, default=300.0)
    baseline.add_argument("--command-json")
    baseline.add_argument("--deterministic", action="store_true")
    baseline.add_argument("--output-dir", required=True)
    baseline.add_argument("--repo-revision", required=True)
    baseline.add_argument("--dirty-tree", action="store_true")
    baseline.add_argument("--seed", type=int, default=0)
    baseline.add_argument("--calibration-revision", default="none")
    baseline.add_argument("--ece-bins", type=int, default=15)

    registry = subparsers.add_parser(
        "baseline-registry",
        help="validate and summarize the baseline qualification registry",
    )
    registry.add_argument("--registry", default="registry/baselines.json")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "evaluate":
        items = load_items(args.items)
        predictions = load_predictions(args.predictions)
        action, records = evaluate_action_predictions(items, predictions, ece_bins=args.ece_bins)
        payload: dict[str, object] = {"action": asdict(action)}
        if all(record.gold_sufficient is not None for record in records):
            payload["abstention"] = asdict(evaluate_abstention(records, ece_bins=args.ece_bins))
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if args.command == "audit":
        report = audit_split_integrity(load_items(args.items))
        payload = {"ok": report.ok, **asdict(report)}
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if args.command == "interventions-evaluate":
        items = load_items(args.items)
        if any(item.split == "test" for item in items):
            parser.error(
                "P06 mechanism qualification rejects test items; final-test evaluation is P08 work"
            )
        predictions = load_predictions(args.predictions)
        intervention_manifest = load_intervention_manifest(args.manifest)
        context = ExperimentContext(
            git_sha=args.git_sha,
            compute_provenance=args.compute_provenance,
        )
        run_manifest = build_intervention_run_manifest(
            items,
            predictions,
            intervention_manifest,
            context=context,
            stability_tv_threshold=args.stability_tv_threshold,
        )
        intervention_result = evaluate_interventions(
            items,
            predictions,
            intervention_manifest,
            stability_tv_threshold=args.stability_tv_threshold,
        )
        print(
            json.dumps(
                {
                    "run_manifest": asdict(run_manifest),
                    "evaluation": asdict(intervention_result),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.command == "fhir-convert":
        cases = load_jsonl(args.cases, FHIRDecisionCase)
        if any(case.split == "test" for case in cases):
            parser.error("P07 rejects final-test FHIR cases; final evaluation is P08 work")
        representation: FHIRRepresentation = args.representation
        fhir_items = [
            fhir_case_to_benchmark_item(case, representation=representation)
            for case in cases
        ]
        dump_jsonl(args.output, fhir_items)
        print(
            json.dumps(
                {
                    "converted": len(fhir_items),
                    "representation": representation,
                    "output": args.output,
                },
                sort_keys=True,
            )
        )
        return

    if args.command == "fhir-external-convert":
        benchmark: BenchmarkName = args.benchmark
        representation: FHIRRepresentation = args.representation
        tasks = load_external_fhir_export(args.export, benchmark=benchmark)
        cases = [external_task_to_case(task) for task in tasks]
        fhir_items = [
            fhir_case_to_benchmark_item(case, representation=representation)
            for case in cases
        ]
        dump_jsonl(args.output, fhir_items)
        print(
            json.dumps(
                {
                    "benchmark": benchmark,
                    "converted": len(fhir_items),
                    "representation": representation,
                    "output": args.output,
                },
                sort_keys=True,
            )
        )
        return

    if args.command == "baseline-run":
        adapter = _build_baseline_adapter(parser, args)
        items = load_items(args.items)
        result = run_baseline(items, adapter, ece_bins=args.ece_bins)
        manifest = write_evidence_packet(
            args.output_dir,
            result,
            items_path=args.items,
            repo_revision=args.repo_revision,
            dirty_tree=args.dirty_tree,
            command=sys.argv,
            seed=args.seed,
            calibration_revision=args.calibration_revision,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        if result.failed or result.evaluation_error is not None:
            raise SystemExit(2)
        return

    if args.command == "baseline-registry":
        registry = load_baseline_registry(args.registry)
        payload = {
            "schema_version": registry.schema_version,
            "entries": [
                {
                    "id": entry.id,
                    "access_status": entry.access_status,
                    "qualification_status": entry.qualification_status,
                }
                for entry in registry.entries
            ],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    raise AssertionError(f"unhandled command: {args.command}")


def _build_baseline_adapter(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> BaselineAdapter:
    adapter: BaselineAdapter
    if args.adapter == "uniform":
        adapter = UniformBaselineAdapter()
    elif args.adapter == "lexicographic":
        adapter = LexicographicBaselineAdapter()
    elif args.adapter == "prediction-file":
        if args.prediction_file is None:
            parser.error("--prediction-file is required for --adapter prediction-file")
        adapter = PredictionFileAdapter(
            args.prediction_file,
            name=args.adapter_name,
            adapter_version=args.adapter_version,
            model_id=args.model_id,
            model_revision=args.model_revision,
            tokenizer_revision=args.tokenizer_revision,
            source_revision=args.source_revision,
        )
    elif args.adapter == "typesafe-http":
        if args.base_url is None:
            parser.error("--base-url is required for --adapter typesafe-http")
        if args.source_revision is None:
            parser.error("--source-revision is required for --adapter typesafe-http")
        adapter = TypeSafeHTTPAdapter(
            name=args.adapter_name,
            adapter_version=args.adapter_version,
            source_revision=args.source_revision,
            config=TypeSafeHTTPConfig(
                base_url=args.base_url,
                timeout_seconds=args.timeout_seconds,
                api_key_env=args.api_key_env,
            ),
            model_id=args.model_id,
            model_revision=args.model_revision,
            tokenizer_revision=args.tokenizer_revision,
            deterministic=args.deterministic,
        )
    elif args.adapter == "json-command":
        if args.command_json is None:
            parser.error("--command-json is required for --adapter json-command")
        if args.source_revision is None:
            parser.error("--source-revision is required for --adapter json-command")
        command = _parse_command_json(parser, args.command_json)
        adapter = JSONCommandAdapter(
            command,
            name=args.adapter_name,
            adapter_version=args.adapter_version,
            source_revision=args.source_revision,
            model_id=args.model_id,
            model_revision=args.model_revision,
            tokenizer_revision=args.tokenizer_revision,
            deterministic=args.deterministic,
            timeout_seconds=args.timeout_seconds,
        )
    else:
        raise AssertionError(f"unhandled adapter: {args.adapter}")
    return adapter


def _parse_command_json(
    parser: argparse.ArgumentParser,
    raw: str,
) -> list[str]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        parser.error("--command-json must be a JSON array of argv strings")
    if not isinstance(value, list) or not value or not all(
        isinstance(part, str) and part for part in value
    ):
        parser.error("--command-json must be a non-empty JSON array of non-empty strings")
    return [str(part) for part in value]


if __name__ == "__main__":
    main()
