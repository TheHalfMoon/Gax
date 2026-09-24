from __future__ import annotations

from pathlib import Path

from gaxbench.provenance import build_file_manifest, canonical_json_sha256, verify_file_manifest


def test_canonical_json_hash_ignores_key_order() -> None:
    assert canonical_json_sha256({"b": 2, "a": 1}) == canonical_json_sha256({"a": 1, "b": 2})


def test_manifest_detects_mutation(tmp_path: Path) -> None:
    first = tmp_path / "a.txt"
    second = tmp_path / "nested" / "b.txt"
    second.parent.mkdir()
    first.write_text("alpha\n", encoding="utf-8")
    second.write_text("beta\n", encoding="utf-8")

    manifest = build_file_manifest([second, first], root=tmp_path)
    assert list(manifest) == ["a.txt", "nested/b.txt"]
    assert verify_file_manifest(manifest, root=tmp_path) == []

    first.write_text("changed\n", encoding="utf-8")
    problems = verify_file_manifest(manifest, root=tmp_path)
    assert len(problems) == 1
    assert problems[0].startswith("sha256:a.txt:")
