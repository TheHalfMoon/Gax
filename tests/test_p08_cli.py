from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def test_freeze_validate_reports_ready_without_mutation() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "gaxbench.p08_cli",
            "freeze-validate",
            "--manifest",
            str(FIXTURES / "p08_freeze_sealed.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["test_access"] == "sealed"
    assert payload["ready_for_authorization"] is True
    assert payload["authorization_valid"] is False


def test_freeze_authorize_writes_separate_digest_bound_copy(tmp_path: Path) -> None:
    output = tmp_path / "authorized.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "gaxbench.p08_cli",
            "freeze-authorize",
            "--manifest",
            str(FIXTURES / "p08_freeze_sealed.json"),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["test_access"] == "authorized"
    assert len(payload["authorization_digest"]) == 64


def test_claims_validate_preserves_non_supported_states() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "gaxbench.p08_cli",
            "claims-validate",
            "--ledger",
            str(FIXTURES / "p08_claims.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["counts"] == {"blocked": 1, "candidate": 1, "supported": 1}
    assert payload["publishable_claim_ids"] == ["supported-mechanics"]
