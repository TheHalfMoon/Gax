from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from gaxbench.p08_freeze import (
    authorize_final_test,
    load_claim_ledger,
    load_freeze_manifest,
    publishable_claims,
    verify_authorization,
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
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                manifest.model_dump(mode="json"),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
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

    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
