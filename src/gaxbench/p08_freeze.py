from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from gaxbench.provenance import canonical_json_sha256
from gaxbench.schema import StrictModel

Redistribution = Literal["permitted", "restricted", "prohibited", "unknown"]
SystemRole = Literal["gax", "baseline", "control"]
QualificationStatus = Literal["qualified", "blocked"]
TestAccess = Literal["sealed", "authorized"]
ClaimStatus = Literal[
    "candidate",
    "supported",
    "null",
    "rejected",
    "blocked",
    "exploratory",
]


class P07CloseoutGate(StrictModel):
    merge_sha: str
    post_main_run_id: int = Field(gt=0)
    conclusion: Literal["success"]

    @field_validator("merge_sha")
    @classmethod
    def validate_merge_sha(cls, value: str) -> str:
        _require_git_sha(value, "p07_closeout.merge_sha")
        return value


class BenchmarkFreeze(StrictModel):
    id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    license: str = Field(min_length=1)
    redistribution: Redistribution
    split_manifest_sha256: str
    leakage_audit_sha256: str
    license_audit_sha256: str
    test_manifest_frozen: bool
    access_requirements: str | None = None

    @field_validator(
        "split_manifest_sha256",
        "leakage_audit_sha256",
        "license_audit_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        _require_sha256(value, "benchmark audit hash")
        return value


class SystemFreeze(StrictModel):
    id: str = Field(min_length=1)
    role: SystemRole
    qualification_status: QualificationStatus
    source_revision: str | None = None
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    adapter_revision: str = Field(min_length=1)
    seeds: list[int] = Field(default_factory=list)
    blocked_reason: str | None = None

    @model_validator(mode="after")
    def validate_identity(self) -> SystemFreeze:
        if self.qualification_status == "blocked":
            if not self.blocked_reason:
                raise ValueError("blocked systems require blocked_reason")
            return self
        if self.blocked_reason is not None:
            raise ValueError("qualified systems must not carry blocked_reason")
        if self.source_revision is None and self.model_revision is None:
            raise ValueError("qualified systems require source_revision or model_revision")
        return self


class CalibrationFreeze(StrictModel):
    method: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    calibration_split_sha256: str
    coverage_targets: list[float] = Field(min_length=1)
    test_tuning_forbidden: Literal[True] = True

    @field_validator("calibration_split_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        _require_sha256(value, "calibration split hash")
        return value

    @field_validator("coverage_targets")
    @classmethod
    def validate_targets(cls, value: list[float]) -> list[float]:
        if any(target <= 0.0 or target > 1.0 for target in value):
            raise ValueError("coverage targets must be in (0, 1]")
        if value != sorted(set(value)):
            raise ValueError("coverage targets must be unique and sorted")
        return value


class StatisticsFreeze(StrictModel):
    ci_level: float = Field(gt=0.0, lt=1.0)
    bootstrap_replicates: int = Field(ge=1000)
    bootstrap_seed: int
    primary_metrics: list[str] = Field(min_length=1)
    primary_comparisons: list[str] = Field(min_length=1)
    multiplicity_policy: str = Field(min_length=1)


class ProtocolFreeze(StrictModel):
    hardware_protocol_revision: str = Field(min_length=1)
    failure_policy_revision: str = Field(min_length=1)
    evidence_protocol_revision: str = Field(min_length=1)
    intervention_revision: str = Field(min_length=1)
    fhir_representation_revision: str = Field(min_length=1)


class P08FreezeManifest(StrictModel):
    schema_version: Literal["0.1"] = "0.1"
    experiment_revision: str = Field(min_length=1)
    repo_revision: str
    dirty_tree: bool
    p07_closeout: P07CloseoutGate
    benchmarks: list[BenchmarkFreeze] = Field(min_length=1)
    systems: list[SystemFreeze] = Field(min_length=2)
    calibration: CalibrationFreeze
    statistics: StatisticsFreeze
    protocol: ProtocolFreeze
    test_access: TestAccess = "sealed"
    authorization_digest: str | None = None

    @field_validator("repo_revision")
    @classmethod
    def validate_repo_revision(cls, value: str) -> str:
        _require_git_sha(value, "repo_revision")
        return value

    @model_validator(mode="after")
    def validate_manifest(self) -> P08FreezeManifest:
        benchmark_ids = [benchmark.id for benchmark in self.benchmarks]
        if len(benchmark_ids) != len(set(benchmark_ids)):
            raise ValueError("benchmark ids must be unique")
        system_ids = [system.id for system in self.systems]
        if len(system_ids) != len(set(system_ids)):
            raise ValueError("system ids must be unique")
        if not any(system.role == "gax" for system in self.systems):
            raise ValueError("freeze manifest requires at least one GAX system")
        if not any(system.role in {"baseline", "control"} for system in self.systems):
            raise ValueError("freeze manifest requires at least one comparison system")
        if self.test_access == "authorized":
            _validate_authorizable(self)
            if self.authorization_digest is None:
                raise ValueError("authorized manifest requires authorization_digest")
            _require_sha256(self.authorization_digest, "authorization_digest")
            if self.authorization_digest != _authorization_digest(self):
                raise ValueError("authorization_digest does not match frozen manifest")
        elif self.authorization_digest is not None:
            raise ValueError("sealed manifest must not carry authorization_digest")
        return self


class ClaimEntry(StrictModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    status: ClaimStatus
    primary_candidate: bool = False
    evidence_packet_ids: list[str] = Field(default_factory=list)
    reason: str | None = None

    @model_validator(mode="after")
    def validate_claim(self) -> ClaimEntry:
        if self.status == "supported" and not self.evidence_packet_ids:
            raise ValueError("supported claims require evidence_packet_ids")
        if self.status in {"null", "rejected", "blocked"} and not self.reason:
            raise ValueError(f"{self.status} claims require a reason")
        if self.status == "exploratory" and self.primary_candidate:
            raise ValueError("exploratory claims cannot be primary candidates")
        return self


class ClaimLedger(StrictModel):
    schema_version: Literal["0.1"] = "0.1"
    experiment_revision: str = Field(min_length=1)
    entries: list[ClaimEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> ClaimLedger:
        ids = [entry.id for entry in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("claim ids must be unique")
        return self


def authorize_final_test(manifest: P08FreezeManifest) -> P08FreezeManifest:
    if manifest.test_access != "sealed":
        raise ValueError("only a sealed manifest can be authorized")
    _validate_authorizable(manifest)
    payload = manifest.model_dump(mode="json")
    payload["test_access"] = "authorized"
    payload["authorization_digest"] = None
    payload["authorization_digest"] = canonical_json_sha256(payload)
    return P08FreezeManifest.model_validate(payload)


def verify_authorization(manifest: P08FreezeManifest) -> bool:
    if manifest.test_access != "authorized" or manifest.authorization_digest is None:
        return False
    return manifest.authorization_digest == _authorization_digest(manifest)


def load_freeze_manifest(path: str | Path) -> P08FreezeManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return P08FreezeManifest.model_validate(payload)


def load_claim_ledger(path: str | Path) -> ClaimLedger:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ClaimLedger.model_validate(payload)


def publishable_claims(ledger: ClaimLedger) -> list[ClaimEntry]:
    return [entry for entry in ledger.entries if entry.status == "supported"]


def _validate_authorizable(manifest: P08FreezeManifest) -> None:
    if manifest.dirty_tree:
        raise ValueError("final-test authorization requires a clean repository tree")
    if not all(benchmark.test_manifest_frozen for benchmark in manifest.benchmarks):
        raise ValueError("all benchmark test manifests must be frozen")
    for system in manifest.systems:
        if system.qualification_status == "blocked" and not system.blocked_reason:
            raise ValueError("blocked systems require an explicit reason")


def _authorization_digest(manifest: P08FreezeManifest) -> str:
    payload = manifest.model_dump(mode="json")
    payload["authorization_digest"] = None
    return canonical_json_sha256(payload)


def _require_sha256(value: str, field: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


def _require_git_sha(value: str, field: str) -> None:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a 40-character lowercase hexadecimal git SHA")
