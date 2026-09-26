from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from gaxbench.ecal import (
    ExperimentContext,
    build_p04_ablation_manifest,
    run_matched_ablation,
)
from gaxbench.gax_v0 import (
    GaxV0Adapter,
    GaxV0Config,
    load_gax_v0_checkpoint,
    save_gax_v0_checkpoint,
    train_gax_v0,
)
from gaxbench.io import load_items
from gaxbench.runner import run_baseline
from gaxbench.schema import BenchmarkItem
from gaxbench.selective import (
    SufficiencyConfig,
    build_p05_manifest,
    run_matched_selector_suite,
)

_ECAL_COMPONENTS = (
    "bidirectional",
    "hard-negative",
    "evidence",
    "proper-scoring",
    "replay",
)
_ItemTuple = tuple[BenchmarkItem, ...]
_EcalItemSets = tuple[_ItemTuple, _ItemTuple, _ItemTuple, _ItemTuple]
_SelectiveItemSets = tuple[_ItemTuple, _ItemTuple, _ItemTuple]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gax")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser(
        "train",
        help="train the deterministic GAX v0 reference model",
    )
    train.add_argument("--train-items", required=True)
    train.add_argument("--validation-items")
    train.add_argument("--checkpoint", required=True)
    _add_base_config_args(train)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="evaluate a GAX v0 checkpoint through the canonical GAXBench runner",
    )
    evaluate.add_argument("--items", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--ece-bins", type=int, default=15)

    manifest = subparsers.add_parser(
        "ecal-manifest",
        help="emit the deterministic P04 ECAL matched-ablation manifest",
    )
    _add_ecal_data_args(manifest)
    _add_base_config_args(manifest)
    _add_experiment_context_args(manifest)

    ablate = subparsers.add_parser(
        "ecal-ablate",
        help="run one P04 matched ECAL control/treatment ablation",
    )
    ablate.add_argument("component", choices=_ECAL_COMPONENTS)
    _add_ecal_data_args(ablate)
    _add_base_config_args(ablate)
    _add_experiment_context_args(ablate)
    ablate.add_argument("--ece-bins", type=int, default=15)

    selective_manifest = subparsers.add_parser(
        "selective-manifest",
        help="emit the deterministic P05 sufficiency/selective manifest",
    )
    _add_selective_data_args(selective_manifest)
    _add_selective_common_args(selective_manifest)
    _add_experiment_context_args(selective_manifest)

    selective_evaluate = subparsers.add_parser(
        "selective-evaluate",
        help="run matched P05 confidence and learned-sufficiency selectors",
    )
    _add_selective_data_args(selective_evaluate)
    _add_selective_common_args(selective_evaluate)
    _add_experiment_context_args(selective_evaluate)
    selective_evaluate.add_argument("--ece-bins", type=int, default=15)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "train":
        train_items = load_items(args.train_items)
        validation_items = _load_optional_items(args.validation_items)
        config = _base_config_from_args(args)
        training_result = train_gax_v0(
            train_items,
            config,
            validation_items=validation_items,
        )
        artifact_sha256 = save_gax_v0_checkpoint(args.checkpoint, training_result)
        payload = {
            "architecture_id": training_result.model.architecture_id,
            "feature_revision": training_result.model.feature_revision,
            "model_revision": training_result.model.model_revision,
            "checkpoint_sha256": artifact_sha256,
            "training_manifest_sha256": training_result.training_manifest_sha256,
            "epochs_completed": len(training_result.history),
            "first_epoch_nll": training_result.history[0].mean_nll,
            "final_epoch_nll": training_result.history[-1].mean_nll,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if args.command == "evaluate":
        model, artifact_sha256 = load_gax_v0_checkpoint(args.checkpoint)
        items = load_items(args.items)
        run_result = run_baseline(
            items,
            GaxV0Adapter(model, artifact_sha256=artifact_sha256),
            ece_bins=args.ece_bins,
        )
        payload = {
            "identity": asdict(run_result.identity),
            "requested": run_result.requested,
            "completed": run_result.completed,
            "failed": run_result.failed,
            "action_metrics": (
                None
                if run_result.action_metrics is None
                else asdict(run_result.action_metrics)
            ),
            "abstention_metrics": (
                None
                if run_result.abstention_metrics is None
                else asdict(run_result.abstention_metrics)
            ),
            "evaluation_error": run_result.evaluation_error,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        if run_result.failed or run_result.evaluation_error is not None:
            raise SystemExit(2)
        return

    if args.command == "ecal-manifest":
        ecal_train, ecal_validation, replay, retention = _load_ecal_items(args)
        payload = build_p04_ablation_manifest(
            ecal_train,
            ecal_validation,
            context=_experiment_context_from_args(args),
            replay_items=replay,
            retention_items=retention,
            base=_base_config_from_args(args),
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if args.command == "ecal-ablate":
        ecal_train, ecal_validation, replay, retention = _load_ecal_items(args)
        if args.component == "replay" and not replay:
            raise SystemExit("ecal-ablate replay requires --replay-items")
        result = run_matched_ablation(
            args.component,
            ecal_train,
            ecal_validation,
            context=_experiment_context_from_args(args),
            replay_items=replay,
            retention_items=retention,
            base=_base_config_from_args(args),
            ece_bins=args.ece_bins,
        )
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
        return

    if args.command in {"selective-manifest", "selective-evaluate"}:
        selective_train, calibration, validation = _load_selective_items(args)
        model, checkpoint_sha256 = load_gax_v0_checkpoint(args.checkpoint)
        sufficiency_config = _sufficiency_config_from_args(
            args,
            feature_dim=model.config.feature_dim,
        )
        context = _experiment_context_from_args(args)

        if args.command == "selective-manifest":
            manifest_payload = build_p05_manifest(
                selective_train,
                calibration,
                validation,
                model,
                context=context,
                sufficiency_config=sufficiency_config,
                target_coverage=args.target_coverage,
            )
            payload = {
                "checkpoint_sha256": checkpoint_sha256,
                "manifest": manifest_payload,
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
            return

        result = run_matched_selector_suite(
            selective_train,
            calibration,
            validation,
            model,
            context=context,
            sufficiency_config=sufficiency_config,
            target_coverage=args.target_coverage,
            ece_bins=args.ece_bins,
        )
        payload = {
            "checkpoint_sha256": checkpoint_sha256,
            "result": asdict(result),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    raise AssertionError(f"unhandled command: {args.command}")


def _add_base_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)


def _add_ecal_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-items", required=True)
    parser.add_argument("--validation-items", required=True)
    parser.add_argument("--replay-items")
    parser.add_argument("--retention-items")


def _add_selective_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-items", required=True)
    parser.add_argument("--calibration-items", required=True)
    parser.add_argument("--validation-items", required=True)
    parser.add_argument("--checkpoint", required=True)


def _add_selective_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target-coverage", type=float, default=0.80)
    parser.add_argument("--sufficiency-learning-rate", type=float, default=0.1)
    parser.add_argument("--sufficiency-epochs", type=int, default=80)
    parser.add_argument("--sufficiency-l2", type=float, default=0.0)
    parser.add_argument("--sufficiency-seed", type=int, default=0)


def _add_experiment_context_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--compute-provenance", required=True)


def _base_config_from_args(args: argparse.Namespace) -> GaxV0Config:
    return GaxV0Config(
        feature_dim=args.feature_dim,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        l2=args.l2,
        seed=args.seed,
    )


def _sufficiency_config_from_args(
    args: argparse.Namespace,
    *,
    feature_dim: int,
) -> SufficiencyConfig:
    return SufficiencyConfig(
        feature_dim=feature_dim,
        learning_rate=args.sufficiency_learning_rate,
        epochs=args.sufficiency_epochs,
        l2=args.sufficiency_l2,
        seed=args.sufficiency_seed,
    )


def _experiment_context_from_args(args: argparse.Namespace) -> ExperimentContext:
    return ExperimentContext(
        git_sha=args.git_sha,
        compute_provenance=args.compute_provenance,
    )


def _load_optional_items(path: str | None) -> _ItemTuple:
    return tuple(load_items(path)) if path is not None else ()


def _load_ecal_items(args: argparse.Namespace) -> _EcalItemSets:
    train_items = tuple(load_items(args.train_items))
    validation_items = tuple(load_items(args.validation_items))
    replay_items = _load_optional_items(args.replay_items)
    retention_items = _load_optional_items(args.retention_items)
    return train_items, validation_items, replay_items, retention_items


def _load_selective_items(args: argparse.Namespace) -> _SelectiveItemSets:
    train_items = tuple(load_items(args.train_items))
    calibration_items = tuple(load_items(args.calibration_items))
    validation_items = tuple(load_items(args.validation_items))
    return train_items, calibration_items, validation_items


if __name__ == "__main__":
    main()
