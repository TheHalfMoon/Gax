from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from gaxbench.provenance import canonical_json_sha256
from gaxbench.schema import BenchmarkItem


@dataclass(frozen=True)
class CrossSplitFinding:
    key: str
    splits: tuple[str, ...]
    item_ids: tuple[str, ...]


@dataclass(frozen=True)
class AuditReport:
    duplicate_item_ids: tuple[str, ...]
    cross_split_source_ids: tuple[CrossSplitFinding, ...]
    cross_split_input_fingerprints: tuple[CrossSplitFinding, ...]
    cross_split_counterfactual_groups: tuple[CrossSplitFinding, ...]

    @property
    def ok(self) -> bool:
        return not (
            self.duplicate_item_ids
            or self.cross_split_source_ids
            or self.cross_split_input_fingerprints
            or self.cross_split_counterfactual_groups
        )


def audit_split_integrity(items: Sequence[BenchmarkItem]) -> AuditReport:
    """Detect exact leakage patterns that must be resolved before evaluation.

    This audit is intentionally exact and deterministic. It does not claim semantic
    near-duplicate detection; later benchmark grains can add embedding/fuzzy audits.
    """
    if not items:
        raise ValueError("items must not be empty")

    item_id_counts: dict[str, int] = defaultdict(int)
    source_groups: dict[str, list[BenchmarkItem]] = defaultdict(list)
    fingerprint_groups: dict[str, list[BenchmarkItem]] = defaultdict(list)
    counterfactual_groups: dict[str, list[BenchmarkItem]] = defaultdict(list)

    for item in items:
        item_id_counts[item.id] += 1
        source_groups[item.source_id].append(item)
        fingerprint_groups[_input_fingerprint(item)].append(item)
        if item.counterfactual_group is not None:
            counterfactual_groups[item.counterfactual_group].append(item)

    duplicate_item_ids = tuple(sorted(key for key, count in item_id_counts.items() if count > 1))
    return AuditReport(
        duplicate_item_ids=duplicate_item_ids,
        cross_split_source_ids=_cross_split_findings(source_groups),
        cross_split_input_fingerprints=_cross_split_findings(fingerprint_groups),
        cross_split_counterfactual_groups=_cross_split_findings(counterfactual_groups),
    )


def _input_fingerprint(item: BenchmarkItem) -> str:
    payload = {
        "task_family": item.task_family,
        "state": item.state,
        "actions": [action.model_dump(mode="json") for action in item.actions],
        "evidence": [evidence.model_dump(mode="json") for evidence in item.evidence],
    }
    return canonical_json_sha256(payload)


def _cross_split_findings(
    groups: dict[str, list[BenchmarkItem]],
) -> tuple[CrossSplitFinding, ...]:
    findings: list[CrossSplitFinding] = []
    for key, members in sorted(groups.items()):
        splits = tuple(sorted({member.split for member in members}))
        if len(splits) <= 1:
            continue
        findings.append(
            CrossSplitFinding(
                key=key,
                splits=splits,
                item_ids=tuple(sorted(member.id for member in members)),
            )
        )
    return tuple(findings)
