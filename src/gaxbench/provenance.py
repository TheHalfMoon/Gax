from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_file_manifest(paths: Sequence[str | Path], *, root: str | Path) -> dict[str, str]:
    root_path = Path(root).resolve()
    entries: list[tuple[str, Path]] = []
    for raw in paths:
        raw_path = Path(raw).resolve()
        try:
            relative = raw_path.relative_to(root_path).as_posix()
        except ValueError as exc:
            raise ValueError(f"path {raw_path} is outside manifest root {root_path}") from exc
        entries.append((relative, raw_path))

    manifest: dict[str, str] = {}
    for relative, raw_path in sorted(entries, key=lambda entry: entry[0]):
        if relative in manifest:
            raise ValueError(f"duplicate manifest path: {relative}")
        manifest[relative] = sha256_file(raw_path)
    return manifest


def verify_file_manifest(manifest: dict[str, str], *, root: str | Path) -> list[str]:
    root_path = Path(root).resolve()
    problems: list[str] = []
    for relative, expected in sorted(manifest.items()):
        path = (root_path / relative).resolve()
        try:
            path.relative_to(root_path)
        except ValueError:
            problems.append(f"unsafe-path:{relative}")
            continue
        if not _is_sha256(expected):
            problems.append(f"invalid-sha256:{relative}:{expected}")
            continue
        if not path.is_file():
            problems.append(f"missing:{relative}")
            continue
        actual = sha256_file(path)
        if actual != expected:
            problems.append(f"sha256:{relative}:{expected}:{actual}")
    return problems


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)
