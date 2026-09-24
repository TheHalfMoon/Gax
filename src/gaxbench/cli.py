from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from gaxbench.io import load_items, load_predictions
from gaxbench.metrics import evaluate_abstention, evaluate_action_predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gaxbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("evaluate", help="evaluate predictions against labeled items")
    evaluate.add_argument("--items", required=True)
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--ece-bins", type=int, default=15)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "evaluate":
        items = load_items(args.items)
        predictions = load_predictions(args.predictions)
        action, records = evaluate_action_predictions(items, predictions, ece_bins=args.ece_bins)
        payload: dict[str, object] = {"action": asdict(action)}
        if all(record.gold_sufficient is not None for record in records):
            payload["abstention"] = asdict(evaluate_abstention(records, ece_bins=args.ece_bins))
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
