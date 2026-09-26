from __future__ import annotations

import pytest
from pydantic import ValidationError

from gaxbench.p08_freeze import (
    BenchmarkFreeze,
    CalibrationFreeze,
    ClaimEntry,
    ClaimLedger,
    P07CloseoutGate,
    P08FreezeManifest,
    ProtocolFreeze,
    StatisticsFreeze,
    SystemFreeze,
    authorize_final_test,
    publishable_claims,
    verify_authorization,
)

SHA256 = "0" * 64
GIT_SHA = "1" * 40


def make_manifest(*, dirty_tree: bool = False, frozen: bool = True) -> P08FreezeManifest:
    return P08FreezeManifest(
        experiment_revision="p08-freeze-test",
        repo_revision=GIT_SHA,
        dirty_tree=dirty_tree,
        p07_closeout=P07CloseoutGate(
            merge_sha="2" * 40,
            post_main_run_id=123,
            conclusion="success",
        ),
        benchmarks=[
            BenchmarkFreeze(
                id="synthetic",
                revision="1",
                license="CC0-1.0",
                redistribution="permitted",
                split_manifest_sha256=SHA256,
                leakage_audit_sha256="a" * 64,
                license_audit_sha256="b" * 64,
                test_manifest_frozen=frozen,
            )
        ],
        systems=[
            SystemFreeze(
                id="gax",
                role="gax",
                qualification_status="qualified",
                source_revision="gax-v0",
                adapter_revision="p08",
                seeds=[0, 1, 2],
            ),
            SystemFreeze(
                id="jev",
                role="baseline",
                qualification_status="blocked",
                adapter_revision="p08",
                blocked_reason="reproducible access unavailable",
            ),
        ],
        calibration=CalibrationFreeze(
            method="temperature-scaling",
            revision="p08-cal-v1",
            calibration_split_sha256="c" * 64,
            coverage_targets=[0.5, 0.8, 0.9],
        ),
        statistics=StatisticsFreeze(
            ci_level=0.95,
            bootstrap_replicates=10000,
            bootstrap_seed=17,
            primary_metrics=["risk@80", "accuracy", "brier"],
            primary_comparisons=["gax-vs-clinical-encoder"],
            multiplicity_policy="primary comparisons frozen; exploratory labeled",
        ),
        protocol=ProtocolFreeze(
            hardware_protocol_revision="p08-hw-v1",
            failure_policy_revision="p08-failure-v1",
            evidence_protocol_revision="p08-evidence-v1",
            intervention_revision="p06-v1",
            fhir_representation_revision="gax-fhir-v0.1",
        ),
    )


def test_authorization_is_digest_bound() -> None:
    authorized = authorize_final_test(make_manifest())
    assert authorized.test_access == "authorized"
    assert authorized.authorization_digest is not None
    assert verify_authorization(authorized)


def test_authorization_rejects_dirty_tree() -> None:
    with pytest.raises(ValueError, match="clean repository tree"):
        authorize_final_test(make_manifest(dirty_tree=True))


def test_authorization_rejects_unfrozen_test_manifest() -> None:
    with pytest.raises(ValueError, match="test manifests must be frozen"):
        authorize_final_test(make_manifest(frozen=False))


def test_tampered_authorization_digest_is_rejected() -> None:
    payload = authorize_final_test(make_manifest()).model_dump(mode="json")
    payload["experiment_revision"] = "tampered"
    with pytest.raises(ValidationError, match="authorization_digest"):
        P08FreezeManifest.model_validate(payload)


def test_qualified_system_requires_revision_identity() -> None:
    with pytest.raises(ValidationError, match="source_revision or model_revision"):
        SystemFreeze(
            id="missing",
            role="baseline",
            qualification_status="qualified",
            adapter_revision="p08",
        )


def test_blocked_system_requires_reason() -> None:
    with pytest.raises(ValidationError, match="blocked_reason"):
        SystemFreeze(
            id="blocked",
            role="baseline",
            qualification_status="blocked",
            adapter_revision="p08",
        )


def test_coverage_targets_must_be_sorted_and_unique() -> None:
    with pytest.raises(ValidationError, match="unique and sorted"):
        CalibrationFreeze(
            method="temperature-scaling",
            revision="1",
            calibration_split_sha256=SHA256,
            coverage_targets=[0.8, 0.5, 0.8],
        )


def test_supported_claim_requires_evidence() -> None:
    with pytest.raises(ValidationError, match="evidence_packet_ids"):
        ClaimEntry(id="claim", text="GAX is better", status="supported")


def test_terminal_negative_claim_requires_reason() -> None:
    with pytest.raises(ValidationError, match="reason"):
        ClaimEntry(id="claim", text="Mechanism helps", status="null")


def test_exploratory_claim_cannot_be_primary() -> None:
    with pytest.raises(ValidationError, match="primary"):
        ClaimEntry(
            id="claim",
            text="Exploratory slice improved",
            status="exploratory",
            primary_candidate=True,
        )


def test_publishable_claims_only_returns_supported_entries() -> None:
    ledger = ClaimLedger(
        experiment_revision="p08",
        entries=[
            ClaimEntry(
                id="supported",
                text="Supported claim",
                status="supported",
                evidence_packet_ids=["packet-1"],
                primary_candidate=True,
            ),
            ClaimEntry(
                id="null",
                text="Null mechanism",
                status="null",
                reason="confidence interval includes no improvement",
            ),
        ],
    )
    assert [entry.id for entry in publishable_claims(ledger)] == ["supported"]
