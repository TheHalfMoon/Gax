from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from gaxbench.schema import BenchmarkItem, Prediction

ModelT = TypeVar("ModelT", bound=BaseModel)


def load_jsonl(path: str | Path, model: type[ModelT]) -> list[ModelT]:
    path = Path(path)
    rows: list[ModelT] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                rows.append(model.model_validate(payload))
            except Exception as exc:  # noqa: BLE001 - preserve source line context
                raise ValueError(f"{path}:{line_number}: invalid record: {exc}") from exc
    if not rows:
        raise ValueError(f"{path}: no records")
    return rows


def dump_jsonl(path: str | Path, rows: list[BaseModel]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            payload = json.dumps(
                row.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            handle.write(payload)
            handle.write("\n")


def load_items(path: str | Path) -> list[BenchmarkItem]:
    return load_jsonl(path, BenchmarkItem)


def load_predictions(path: str | Path) -> list[Prediction]:
    return load_jsonl(path, Prediction)
