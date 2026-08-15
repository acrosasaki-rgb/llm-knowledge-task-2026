from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary_path.replace(output_path)


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()


def read_prompt_templates(path: str | Path) -> dict[str, str]:
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = csv.DictReader(stream)
        templates = {
            row["Relation"]: row["PromptTemplate"]
            for row in rows
            if row.get("Relation") and row.get("PromptTemplate")
        }
    if not templates:
        raise ValueError(f"no prompt templates found in {path}")
    return templates


def canonical_gold_labels(object_entities: Any) -> list[str]:
    if not isinstance(object_entities, list):
        return []
    labels: list[str] = []
    for entity in object_entities:
        if isinstance(entity, str):
            labels.append(entity)
        elif isinstance(entity, list) and entity and isinstance(entity[0], str):
            labels.append(entity[0])
    return labels
