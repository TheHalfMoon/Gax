from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from gaxbench.provenance import canonical_json_sha256, sha256_file
from gaxbench.schema import StrictModel

OutcomeStatus = Literal[
    "success",
    "timeout",
    "oom",
    "parse_failure",
    "inference_failure",
    "evaluation_failure",
    "blocked",
]
MetricDirection = Literal["higher", "lower"]
ArtifactKind = Literal["table", "figure", "report"]
ComparisonStatus = Literal["complete", "blocked", "undefined"]


@dataclass(frozen=True)
class ConfidenceInterval:
    level: float
    lower: float
    upper: float


@dataclass(frozen=True)
class PairedBootstrapResult:
    n: int
    point_estimate: float
    confidence_interval: ConfidenceInterval
    replicates: int
    seed: int


@dataclass(frozen=True)
class RankingMetric:
    value: float | None
    defined: bool
    reason: str | None


@dataclass(frozen=True)
class EvidenceRankingMetrics:
    n: int
    positives: int
    negatives: int
    auroc: RankingMetric
    auprc: RankingMetric


@dataclass(frozen=True)
class ReliabilityBin:
    index: int
    lower: float
    upper: float
    upper_inclusive: bool
    count: int
    mean_confidence: float | None
    empirical_accuracy: float | None
    absolute_gap: float | None


class RunOutcome(StrictModel):
    item_id: str = Field(min_length=1)
    status: OutcomeStatus
    value: float | None = None
    detail: str | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> RunOutcome:
        if self.status == "success":
            if self.value is None or not math.isfinite(self.value):
                raise ValueError("successful outcomes require a finite value")
            if self.detail is not None:
                raise ValueError("successful outcomes must not carry failure detail")
            return self
        if self.value is not None:
            raise ValueError("failed outcomes must not carry a metric value")
        if not self.detail:
            raise ValueError("failed outcomes require detail")
        return self


@dataclass(frozen=True)
class FailurePreservingAggregate:
    requested: int
    completed: int
    failed: int
    complete: bool
    status_counts: dict[str, int]
    mean_value: float | None
    failure_item_ids: tuple[str, ...]


class DerivedArtifactManifest(StrictModel):
    schema_version: Literal["0.1"] = "0.1"
    artifact_id: str = Field(min_length=1)
    artifact_kind: ArtifactKind
    generator_revision: str = Field(min_length=1)
    source_run_ids: list[str] = Field(default_factory=list)
    evidence_packet_ids: list[str] = Field(default_factory=list)
    source_file_sha256: dict[str, str] = Field(default_factory=dict)
    output_sha256: str
    manifest_digest: str | None = None

    @field_validator("output_sha256")
    @classmethod
    def validate_output_hash(cls, value: str) -> str:
        _require_sha256(value, "output_sha256")
        return value

    @field_validator("source_file_sha256")
    @classmethod
    def validate_source_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        for key, digest in value.items():
            if not key:
                raise ValueError("source_file_sha256 keys must not be empty")
            _require_sha256(digest, f"source_file_sha256[{key!r}]")
        return value

    @model_validator(mode="after")
    def validate_manifest(self) -> DerivedArtifactManifest:
        _require_unique_sorted(self.source_run_ids, "source_run_ids")
        _require_unique_sorted(self.evidence_packet_ids, "evidence_packet_ids")
        if self.manifest_digest is not None:
            _require_sha256(self.manifest_digest, "manifest_digest")
            if self.manifest_digest != _derived_manifest_digest(self):
                raise ValueError("manifest_digest does not match manifest contents")
        return self


class PrimaryComparison(StrictModel):
    id: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    system_a: str = Field(min_length=1)
    system_b: str = Field(min_length=1)
    direction: MetricDirection
    paired: Literal[True] = True
    multiplicity_family: str = Field(min_length=1)

    @model_validator(mode="after")
    def different_systems(self) -> PrimaryComparison:
        if self.system_a == self.system_b:
            raise ValueError("primary comparison systems must differ")
        return self


class PrimaryComparisonRegistry(StrictModel):
    schema_version: Literal["0.1"] = "0.1"
    experiment_revision: str = Field(min_length=1)
    comparisons: list[PrimaryComparison] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_registry(self) -> PrimaryComparisonRegistry:
        ids = [comparison.id for comparison in self.comparisons]
        if len(ids) != len(set(ids)):
            raise ValueError("primary comparison ids must be unique")
        if ids != sorted(ids):
            raise ValueError("primary comparisons must be sorted by id")
        return self


class ComparisonResult(StrictModel):
    comparison_id: str = Field(min_length=1)
    status: ComparisonStatus
    requested: int = Field(ge=0)
    completed: int = Field(ge=0)
    failed: int = Field(ge=0)
    estimate_a: float | None = None
    estimate_b: float | None = None
    difference_a_minus_b: float | None = None
    ci_level: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    evidence_packet_ids: list[str] = Field(default_factory=list)
    reason: str | None = None

    @model_validator(mode="after")
    def validate_result(self) -> ComparisonResult:
        if self.requested != self.completed + self.failed:
            raise ValueError("requested must equal completed + failed")
        _require_unique_sorted(self.evidence_packet_ids, "evidence_packet_ids")
        numeric = (
            self.estimate_a,
            self.estimate_b,
            self.difference_a_minus_b,
            self.ci_level,
            self.ci_lower,
            self.ci_upper,
        )
        if any(value is not None and not math.isfinite(value) for value in numeric):
            raise ValueError("comparison numeric fields must be finite")
        if self.status == "complete":
            if self.failed != 0 or self.completed == 0:
                raise ValueError(
                    "complete comparisons require nonzero completed and zero failed"
                )
            if any(value is None for value in numeric):
                raise ValueError(
                    "complete comparisons require estimates and confidence interval"
                )
            assert self.ci_level is not None
            assert self.ci_lower is not None
            assert self.ci_upper is not None
            if not 0.0 < self.ci_level < 1.0:
                raise ValueError("ci_level must be in (0, 1)")
            if self.ci_lower > self.ci_upper:
                raise ValueError("ci_lower must not exceed ci_upper")
            if self.reason is not None:
                raise ValueError("complete comparisons must not carry reason")
            return self
        if not self.reason:
            raise ValueError("blocked/undefined comparisons require reason")
        if any(value is not None for value in numeric):
            raise ValueError(
                "blocked/undefined comparisons must not carry estimates"
            )
        return self


class ComparisonResultSet(StrictModel):
    schema_version: Literal["0.1"] = "0.1"
    experiment_revision: str = Field(min_length=1)
    results: list[ComparisonResult] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_results(self) -> ComparisonResultSet:
        ids = [result.comparison_id for result in self.results]
        if len(ids) != len(set(ids)):
            raise ValueError("comparison results must have unique ids")
        if ids != sorted(ids):
            raise ValueError("comparison results must be sorted by comparison_id")
        return self


class PrimaryComparisonReport(StrictModel):
    schema_version: Literal["0.1"] = "0.1"
    experiment_revision: str = Field(min_length=1)
    repo_revision: str
    results: list[ComparisonResult] = Field(min_length=1)
    report_digest: str | None = None

    @field_validator("repo_revision")
    @classmethod
    def validate_repo_revision(cls, value: str) -> str:
        _require_git_sha(value, "repo_revision")
        return value

    @model_validator(mode="after")
    def validate_report(self) -> PrimaryComparisonReport:
        ids = [result.comparison_id for result in self.results]
        if len(ids) != len(set(ids)):
            raise ValueError("comparison results must have unique ids")
        if ids != sorted(ids):
            raise ValueError("comparison results must be sorted by comparison_id")
        if self.report_digest is not None:
            _require_sha256(self.report_digest, "report_digest")
            if self.report_digest != _report_digest(self):
                raise ValueError("report_digest does not match report contents")
        return self


def paired_bootstrap_mean_difference(
    values_a: Sequence[float],
    values_b: Sequence[float],
    *,
    replicates: int,
    seed: int,
    ci_level: float,
) -> PairedBootstrapResult:
    _validate_paired_values(values_a, values_b)
    if replicates < 1000:
        raise ValueError("paired bootstrap requires at least 1000 replicates")
    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level must be in (0, 1)")

    differences = [a - b for a, b in zip(values_a, values_b, strict=True)]
    point_estimate = sum(differences) / len(differences)
    rng = random.Random(seed)
    samples: list[float] = []
    n = len(differences)
    for _ in range(replicates):
        total = 0.0
        for _ in range(n):
            total += differences[rng.randrange(n)]
        samples.append(total / n)

    alpha = (1.0 - ci_level) / 2.0
    interval = ConfidenceInterval(
        level=ci_level,
        lower=_quantile(samples, alpha),
        upper=_quantile(samples, 1.0 - alpha),
    )
    return PairedBootstrapResult(
        n=n,
        point_estimate=point_estimate,
        confidence_interval=interval,
        replicates=replicates,
        seed=seed,
    )


def evidence_ranking_metrics(
    scores: Sequence[float], labels: Sequence[bool]
) -> EvidenceRankingMetrics:
    if len(scores) != len(labels) or not scores:
        raise ValueError("scores and labels must be non-empty and equally sized")
    for score in scores:
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("evidence scores must be finite and in [0, 1]")

    positives = sum(labels)
    negatives = len(labels) - positives
    auroc = _auroc(scores, labels, positives=positives, negatives=negatives)
    auprc = _auprc(scores, labels, positives=positives)
    return EvidenceRankingMetrics(
        n=len(labels),
        positives=positives,
        negatives=negatives,
        auroc=auroc,
        auprc=auprc,
    )


def reliability_bins(
    confidences: Sequence[float], correctness: Sequence[bool], *, bins: int
) -> tuple[ReliabilityBin, ...]:
    if len(confidences) != len(correctness) or not confidences:
        raise ValueError(
            "confidences and correctness must be non-empty and equally sized"
        )
    if bins < 2:
        raise ValueError("bins must be >= 2")

    confidence_sum = [0.0] * bins
    correct_sum = [0] * bins
    counts = [0] * bins
    for confidence, correct in zip(confidences, correctness, strict=True):
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1]")
        index = min(int(confidence * bins), bins - 1)
        confidence_sum[index] += confidence
        correct_sum[index] += int(correct)
        counts[index] += 1

    output: list[ReliabilityBin] = []
    for index in range(bins):
        count = counts[index]
        mean_confidence = confidence_sum[index] / count if count else None
        empirical_accuracy = correct_sum[index] / count if count else None
        absolute_gap = (
            abs(empirical_accuracy - mean_confidence)
            if mean_confidence is not None and empirical_accuracy is not None
            else None
        )
        output.append(
            ReliabilityBin(
                index=index,
                lower=index / bins,
                upper=(index + 1) / bins,
                upper_inclusive=index == bins - 1,
                count=count,
                mean_confidence=mean_confidence,
                empirical_accuracy=empirical_accuracy,
                absolute_gap=absolute_gap,
            )
        )
    return tuple(output)


def aggregate_run_outcomes(
    outcomes: Sequence[RunOutcome],
) -> FailurePreservingAggregate:
    if not outcomes:
        raise ValueError("outcomes must not be empty")
    ids = [outcome.item_id for outcome in outcomes]
    if len(ids) != len(set(ids)):
        raise ValueError("outcome item ids must be unique")

    counts = Counter(outcome.status for outcome in outcomes)
    completed = counts.get("success", 0)
    failed = len(outcomes) - completed
    complete = failed == 0
    values = [
        outcome.value for outcome in outcomes if outcome.status == "success"
    ]
    mean_value: float | None = None
    if complete:
        assert all(value is not None for value in values)
        mean_value = sum(value for value in values if value is not None) / len(values)

    failure_item_ids = tuple(
        sorted(
            outcome.item_id
            for outcome in outcomes
            if outcome.status != "success"
        )
    )
    return FailurePreservingAggregate(
        requested=len(outcomes),
        completed=completed,
        failed=failed,
        complete=complete,
        status_counts=dict(sorted(counts.items())),
        mean_value=mean_value,
        failure_item_ids=failure_item_ids,
    )


def build_derived_artifact_manifest(
    *,
    artifact_id: str,
    artifact_kind: ArtifactKind,
    generator_revision: str,
    output_path: str | Path,
    source_run_ids: Sequence[str],
    evidence_packet_ids: Sequence[str],
    source_files: Mapping[str, str | Path],
) -> DerivedArtifactManifest:
    run_ids = sorted(set(source_run_ids))
    packet_ids = sorted(set(evidence_packet_ids))
    if len(run_ids) != len(source_run_ids):
        raise ValueError("source_run_ids must be unique")
    if len(packet_ids) != len(evidence_packet_ids):
        raise ValueError("evidence_packet_ids must be unique")
    source_hashes = {
        key: sha256_file(path)
        for key, path in sorted(source_files.items(), key=lambda item: item[0])
    }
    manifest = DerivedArtifactManifest(
        artifact_id=artifact_id,
        artifact_kind=artifact_kind,
        generator_revision=generator_revision,
        source_run_ids=run_ids,
        evidence_packet_ids=packet_ids,
        source_file_sha256=source_hashes,
        output_sha256=sha256_file(output_path),
    )
    payload = manifest.model_dump(mode="json")
    payload["manifest_digest"] = _derived_manifest_digest(manifest)
    return DerivedArtifactManifest.model_validate(payload)


def load_primary_comparison_registry(
    path: str | Path,
) -> PrimaryComparisonRegistry:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return PrimaryComparisonRegistry.model_validate(payload)


def load_comparison_result_set(path: str | Path) -> ComparisonResultSet:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ComparisonResultSet.model_validate(payload)


def build_primary_report(
    registry: PrimaryComparisonRegistry,
    result_set: ComparisonResultSet,
    *,
    repo_revision: str,
) -> PrimaryComparisonReport:
    if registry.experiment_revision != result_set.experiment_revision:
        raise ValueError("registry and result-set experiment revisions must match")
    result_map = {result.comparison_id: result for result in result_set.results}
    expected = {comparison.id for comparison in registry.comparisons}
    if set(result_map) != expected:
        missing = sorted(expected - set(result_map))
        extra = sorted(set(result_map) - expected)
        raise ValueError(
            f"comparison result ids mismatch: missing={missing}, extra={extra}"
        )

    report = PrimaryComparisonReport(
        experiment_revision=registry.experiment_revision,
        repo_revision=repo_revision,
        results=[result_map[comparison_id] for comparison_id in sorted(result_map)],
    )
    payload = report.model_dump(mode="json")
    payload["report_digest"] = _report_digest(report)
    return PrimaryComparisonReport.model_validate(payload)


def serialize_primary_report(report: PrimaryComparisonReport) -> str:
    return (
        json.dumps(
            report.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _auroc(
    scores: Sequence[float],
    labels: Sequence[bool],
    *,
    positives: int,
    negatives: int,
) -> RankingMetric:
    if positives == 0:
        return RankingMetric(None, False, "AUROC undefined without positive labels")
    if negatives == 0:
        return RankingMetric(None, False, "AUROC undefined without negative labels")

    positive_scores = [
        score for score, label in zip(scores, labels, strict=True) if label
    ]
    negative_scores = [
        score for score, label in zip(scores, labels, strict=True) if not label
    ]
    wins = 0.0
    for positive in positive_scores:
        for negative in negative_scores:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return RankingMetric(wins / (positives * negatives), True, None)


def _auprc(
    scores: Sequence[float], labels: Sequence[bool], *, positives: int
) -> RankingMetric:
    if positives == 0:
        return RankingMetric(None, False, "AUPRC undefined without positive labels")

    groups: dict[float, list[bool]] = defaultdict(list)
    for score, label in zip(scores, labels, strict=True):
        groups[score].append(label)

    true_positive = 0
    false_positive = 0
    previous_recall = 0.0
    area = 0.0
    for score in sorted(groups, reverse=True):
        group = groups[score]
        positive_in_group = sum(group)
        true_positive += positive_in_group
        false_positive += len(group) - positive_in_group
        recall = true_positive / positives
        precision = true_positive / (true_positive + false_positive)
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return RankingMetric(area, True, None)


def _validate_paired_values(
    values_a: Sequence[float], values_b: Sequence[float]
) -> None:
    if len(values_a) != len(values_b) or not values_a:
        raise ValueError("paired values must be non-empty and equally sized")
    for value in (*values_a, *values_b):
        if not math.isfinite(value):
            raise ValueError("paired bootstrap values must be finite")


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile values must not be empty")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return (
        ordered[lower_index] * (1.0 - fraction)
        + ordered[upper_index] * fraction
    )


def _derived_manifest_digest(manifest: DerivedArtifactManifest) -> str:
    payload = manifest.model_dump(mode="json")
    payload["manifest_digest"] = None
    return canonical_json_sha256(payload)


def _report_digest(report: PrimaryComparisonReport) -> str:
    payload = report.model_dump(mode="json")
    payload["report_digest"] = None
    return canonical_json_sha256(payload)


def _require_unique_sorted(values: Sequence[str], field: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field} must be unique")
    if list(values) != sorted(values):
        raise ValueError(f"{field} must be sorted")


def _require_sha256(value: str, field: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


def _require_git_sha(value: str, field: str) -> None:
    if len(value) != 40 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(
            f"{field} must be a 40-character lowercase hexadecimal git SHA"
        )
