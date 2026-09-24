from __future__ import annotations

from gaxbench.audit import audit_split_integrity
from gaxbench.schema import Action, BenchmarkItem, Gold, Provenance


def make_item(
    item_id: str,
    source_id: str,
    split: str,
    state_value: int,
    *,
    counterfactual_group: str | None = None,
    task_family: str = "audit",
    reverse_actions: bool = False,
) -> BenchmarkItem:
    return BenchmarkItem(
        id=item_id,
        source_id=source_id,
        split=split,  # type: ignore[arg-type]
        task_family=task_family,
        state={"value": state_value},
        actions=(
            [Action(id="b", description="B"), Action(id="a", description="A")]
            if reverse_actions
            else [Action(id="a", description="A"), Action(id="b", description="B")]
        ),
        gold=Gold(action="a", sufficient=True),
        counterfactual_group=counterfactual_group,
        provenance=Provenance(
            dataset="synthetic",
            revision="1",
            license="CC0-1.0",
            transform_revision="1",
        ),
    )


def test_clean_entity_disjoint_splits_pass() -> None:
    report = audit_split_integrity(
        [
            make_item("a", "source-a", "train", 1),
            make_item("b", "source-b", "test", 2),
        ]
    )
    assert report.ok


def test_cross_split_source_and_exact_input_are_reported() -> None:
    report = audit_split_integrity(
        [
            make_item("a", "same-source", "train", 1),
            make_item("b", "same-source", "test", 1),
        ]
    )
    assert not report.ok
    assert report.cross_split_source_ids[0].key == "same-source"
    assert report.cross_split_input_fingerprints[0].splits == ("test", "train")


def test_counterfactual_group_must_not_cross_splits() -> None:
    report = audit_split_integrity(
        [
            make_item("a", "source-a", "train", 1, counterfactual_group="cf-1"),
            make_item("b", "source-b", "test", 2, counterfactual_group="cf-1"),
        ]
    )
    assert report.cross_split_counterfactual_groups[0].key == "cf-1"


def test_exact_input_fingerprint_ignores_action_order_and_task_family_label() -> None:
    report = audit_split_integrity(
        [
            make_item("a", "source-a", "train", 7, task_family="family-a"),
            make_item(
                "b",
                "source-b",
                "test",
                7,
                task_family="family-b",
                reverse_actions=True,
            ),
        ]
    )
    assert len(report.cross_split_input_fingerprints) == 1
    assert report.cross_split_input_fingerprints[0].splits == ("test", "train")
