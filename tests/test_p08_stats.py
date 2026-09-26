from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from gaxbench.p08_stats import (
    ComparisonResult,
    PrimaryComparison,
    PrimaryComparisonRegistry,
    RunOutcome,
    aggregate_run_outcomes,
    build_derived_artifact_manifest,
    build_primary_report,
    evidence_ranking_metrics,
    paired_bootstrap_mean_difference,
    reliability_bins,
    serialize_primary_report,
)


def test_paired_bootstrap_constant_difference_is_exact() -> None:
    result = paired_bootstrap_mean_difference(
        [2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0],
        replicates=1000,
        seed=17,
        ci_level=0.95,
    )
    assert result.point_estimate == pytest.approx(1.0)
    assert result.confidence_interval.lower == pytest.approx(1.0)
    assert result.confidence_interval.upper == pytest.approx(1.0)
    assert result.replicates == 1000
    assert result.seed == 17


def test_paired_bootstrap_is_seed_deterministic() -> None:
    first = paired_bootstrap_mean_difference(
        [1.0, 2.0, 8.0, 4.0],
        [0.0, 3.0, 2.0, 5.0],
        replicates=1000,
        seed=9,
        ci_level=0.95,
    )
    second = paired_bootstrap_mean_difference(
        [1.0, 2.0, 8.0, 4.0],
        [0.0, 3.0, 2.0, 5.0],
        replicates=1000,
        seed=9,
        ci_level=0.95,
    )
    assert first == second


def test_paired_bootstrap_rejects_invalid_protocol() -> None:
    with pytest.raises(ValueError, match="at least 1000"):
        paired_bootstrap_mean_difference(
            [1.0],
            [1.0],
            replicates=999,
            seed=1,
            ci_level=0.95,
        )
    with pytest.raises(ValueError, match="equally sized"):
        paired_bootstrap_mean_difference(
            [1.0, 2.0],
            [1.0],
            replicates=1000,
            seed=1,
            ci_level=0.95,
        )


def test_evidence_ranking_ties_are_grouped_deterministically() -> None:
    metrics = evidence_ranking_metrics(
        [0.9, 0.8, 0.8, 0.1],
        [True, True, False, False],
    )
    assert metrics.auroc.defined
    assert metrics.auroc.value == pytest.approx(0.875)
    assert metrics.auprc.defined
    assert metrics.auprc.value == pytest.approx(5 / 6)


def test_evidence_ranking_reports_undefined_cases() -> None:
    metrics = evidence_ranking_metrics([0.9, 0.8], [True, True])
    assert not metrics.auroc.defined
    assert metrics.auroc.value is None
    assert metrics.auroc.reason == "AUROC undefined without negative labels"
    assert metrics.auprc.defined
    assert metrics.auprc.value == pytest.approx(1.0)

    no_positive = evidence_ranking_metrics([0.2, 0.1], [False, False])
    assert not no_positive.auroc.defined
    assert not no_positive.auprc.defined
    assert no_positive.auprc.value is None


def test_evidence_ranking_rejects_non_probability_scores() -> None:
    with pytest.raises(ValueError, match="in \[0, 1\]"):
        evidence_ranking_metrics([1.2, 0.2], [True, False])


def test_reliability_bins_preserve_empty_bins() -> None:
    bins = reliability_bins([0.1, 0.2, 0.9], [True, False, True], bins=4)
    assert len(bins) == 4
    assert bins[0].count == 2
    assert bins[0].mean_confidence == pytest.approx(0.15)
    assert bins[0].empirical_accuracy == pytest.approx(0.5)
    assert bins[1].count == 0
    assert bins[1].mean_confidence is None
    assert bins[3].count == 1
    assert bins[3].upper_inclusive


def test_run_outcome_contract_rejects_silent_failure_values() -> None:
    with pytest.raises(ValidationError, match="failed outcomes must not carry"):
        RunOutcome(item_id="x", status="timeout", value=0.5, detail="timeout")
    with pytest.raises(ValidationError, match="require detail"):
        RunOutcome(item_id="x", status="oom")
    with pytest.raises(ValidationError, match="finite value"):
        RunOutcome(item_id="x", status="success")


def test_failure_preserving_aggregation_never_drops_failures() -> None:
    aggregate = aggregate_run_outcomes(
        [
            RunOutcome(item_id="a", status="success", value=1.0),
            RunOutcome(item_id="b", status="timeout", detail="deadline exceeded"),
            RunOutcome(item_id="c", status="success", value=0.0),
        ]
    )
    assert aggregate.requested == 3
    assert aggregate.completed == 2
    assert aggregate.failed == 1
    assert not aggregate.complete
    assert aggregate.mean_value is None
    assert aggregate.status_counts == {"success": 2, "timeout": 1}
    assert aggregate.failure_item_ids == ("b",)


def test_complete_aggregation_allows_mean() -> None:
    aggregate = aggregate_run_outcomes(
        [
            RunOutcome(item_id="a", status="success", value=1.0),
            RunOutcome(item_id="b", status="success", value=0.0),
        ]
    )
    assert aggregate.complete
    assert aggregate.mean_value == pytest.approx(0.5)


def test_derived_artifact_manifest_binds_sources_and_output(tmp_path: Path) -> None:
    source = tmp_path / "raw.json"
    output = tmp_path / "table.json"
    source.write_text('{"raw":1}\n', encoding="utf-8")
    output.write_text('{"table":1}\n', encoding="utf-8")

    manifest = build_derived_artifact_manifest(
        artifact_id="table-main",
        artifact_kind="table",
        generator_revision="report-v0.1",
        output_path=output,
        source_run_ids=["run-b", "run-a"],
        evidence_packet_ids=["packet-b", "packet-a"],
        source_files={"raw": source},
    )
    assert manifest.source_run_ids == ["run-a", "run-b"]
    assert manifest.evidence_packet_ids == ["packet-a", "packet-b"]
    assert manifest.manifest_digest is not None
    assert len(manifest.output_sha256) == 64
    assert len(manifest.source_file_sha256["raw"]) == 64


def test_derived_manifest_rejects_duplicate_source_ids(tmp_path: Path) -> None:
    output = tmp_path / "out.json"
    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source_run_ids must be unique"):
        build_derived_artifact_manifest(
            artifact_id="x",
            artifact_kind="figure",
            generator_revision="v1",
            output_path=output,
            source_run_ids=["run", "run"],
            evidence_packet_ids=[],
            source_files={},
        )


def registry() -> PrimaryComparisonRegistry:
    return PrimaryComparisonRegistry(
        experiment_revision="p08-test",
        comparisons=[
            PrimaryComparison(
                id="cmp-action",
                metric="accuracy",
                system_a="gax",
                system_b="baseline",
                direction="higher",
                multiplicity_family="primary",
            )
        ],
    )


def complete_result() -> ComparisonResult:
    return ComparisonResult(
        comparison_id="cmp-action",
        status="complete",
        requested=20,
        completed=20,
        failed=0,
        estimate_a=0.8,
        estimate_b=0.7,
        difference_a_minus_b=0.1,
        ci_level=0.95,
        ci_lower=0.02,
        ci_upper=0.18,
        evidence_packet_ids=["packet-a", "packet-b"],
    )


def test_primary_report_is_deterministic_and_digest_bound() -> None:
    report = build_primary_report(
        registry(),
        [complete_result()],
        repo_revision="a" * 40,
    )
    encoded = serialize_primary_report(report)
    parsed = json.loads(encoded)
    assert parsed["report_digest"] == report.report_digest
    assert parsed["results"][0]["comparison_id"] == "cmp-action"
    assert serialize_primary_report(report) == encoded


def test_primary_report_requires_exact_registry_coverage() -> None:
    with pytest.raises(ValueError, match="ids mismatch"):
        build_primary_report(registry(), [], repo_revision="a" * 40)


def test_comparison_result_preserves_failed_denominator() -> None:
    with pytest.raises(ValidationError, match="complete comparisons"):
        ComparisonResult(
            comparison_id="cmp-action",
            status="complete",
            requested=20,
            completed=19,
            failed=1,
            estimate_a=0.8,
            estimate_b=0.7,
            difference_a_minus_b=0.1,
            ci_level=0.95,
            ci_lower=0.02,
            ci_upper=0.18,
            evidence_packet_ids=["packet-a"],
        )


def test_blocked_comparison_keeps_reason_without_fake_estimate() -> None:
    result = ComparisonResult(
        comparison_id="cmp-jev",
        status="blocked",
        requested=0,
        completed=0,
        failed=0,
        reason="reproducible evaluation access unavailable",
    )
    assert result.estimate_a is None
    assert result.reason is not None
