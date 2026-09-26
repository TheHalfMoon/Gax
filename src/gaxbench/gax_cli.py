from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from gaxbench.gax_v0 import (
    GaxV0Adapter,
    GaxV0Config,
    load_gax_v0_checkpoint,
    save_gax_v0_checkpoint,
    train_gax_v0,
)
from gaxbench.io import load_items
from gaxbench.runner import run_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gax")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="train the deterministic GAX v0 reference model")
    train.add_argument("--train-items", required=True)
    train.add_argument("--validation-items")
    train.add_argument("--checkpoint", required=True)
    train.add_argument("--feature-dim", type=int, default=32)
    train.add_argument("--learning-rate", type=float, default=0.2)
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--l2", type=float, default=0.0)
    train.add_argument("--seed", type=int, default=0)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="evaluate a GAX v0 checkpoint through the canonical GAXBench runner",
    )
    evaluate.add_argument("--items", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--ece-bins", type=int, default=15)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "train":
        train_items = load_items(args.train_items)
        validation_items = (
            load_items(args.validation_items) if args.validation_items is not None else ()
        )
        config = GaxV0Config(
            feature_dim=args.feature_dim,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            l2=args.l2,
            seed=args.seed,
        )
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

    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
