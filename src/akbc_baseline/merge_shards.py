from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .data import read_jsonl, write_jsonl


def _identity(row: dict[str, Any]) -> tuple[Any, Any]:
    return row.get("SubjectEntity"), row.get("Relation")


def merge_rows(
    expected_input: list[dict[str, Any]],
    shard_paths: list[Path],
    *,
    prediction: bool,
) -> list[dict[str, Any]]:
    if not shard_paths:
        raise ValueError("no shard files matched")
    merged = [
        row
        for shard_path in sorted(shard_paths)
        for row in read_jsonl(shard_path)
    ]
    if len(merged) != len(expected_input):
        raise ValueError(
            f"expected {len(expected_input)} merged rows, found {len(merged)}"
        )
    for index, (expected, actual) in enumerate(
        zip(expected_input, merged, strict=True)
    ):
        if _identity(actual) != _identity(expected):
            raise ValueError(f"row identity/order mismatch at index {index}")
        if prediction and set(actual) != {
            "SubjectEntity",
            "Relation",
            "ObjectEntities",
        }:
            raise ValueError(f"prediction row {index} is not in official format")
        if prediction and not isinstance(actual["ObjectEntities"], list):
            raise ValueError(f"ObjectEntities must be a list at index {index}")
    return merged


def merge_metrics(metric_paths: list[Path], rows: int) -> dict[str, Any]:
    metrics = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(metric_paths)
    ]
    if not metrics:
        raise ValueError("no shard metric files matched")
    return {
        "name": metrics[0]["name"],
        "model_id": metrics[0]["model_id"],
        "parameter_count_billion": metrics[0]["parameter_count_billion"],
        "num_candidates": metrics[0]["num_candidates"],
        "rows": rows,
        "empty_predictions": sum(item["empty_predictions"] for item in metrics),
        "elapsed_seconds": sum(item["elapsed_seconds"] for item in metrics),
        "peak_cuda_memory_gib": max(
            item["peak_cuda_memory_gib"] for item in metrics
        ),
        "shards": len(metrics),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge ordered AKBC prediction and candidate shards"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--prediction-glob", required=True)
    parser.add_argument("--candidate-glob", required=True)
    parser.add_argument("--metrics-glob", required=True)
    parser.add_argument("--prediction-output", required=True)
    parser.add_argument("--candidate-output", required=True)
    parser.add_argument("--metrics-output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    expected_input = read_jsonl(args.input)
    predictions = merge_rows(
        expected_input,
        list(Path().glob(args.prediction_glob)),
        prediction=True,
    )
    candidates = merge_rows(
        expected_input,
        list(Path().glob(args.candidate_glob)),
        prediction=False,
    )
    metrics = merge_metrics(
        list(Path().glob(args.metrics_glob)),
        len(predictions),
    )
    write_jsonl(args.prediction_output, predictions)
    write_jsonl(args.candidate_output, candidates)
    metrics_path = Path(args.metrics_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"merged {len(predictions)} predictions and "
        f"{len(candidates)} candidate rows",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
