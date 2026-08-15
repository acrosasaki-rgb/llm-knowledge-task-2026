from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Sequence

from .aggregation import aggregate_candidates
from .area_clustering import (
    cluster_numeric_candidates,
    format_number,
)
from .area_unit_filter import filter_unit_collision_sources
from .data import read_jsonl, write_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Filter unit-conversion source candidates using cross-candidate "
            "collisions, then aggregate only the retained raw values"
        )
    )
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidates-output", required=True)
    parser.add_argument("--baseline-median-output")
    parser.add_argument("--baseline-dominant-output")
    parser.add_argument("--filtered-median-output")
    parser.add_argument("--metrics-output")
    parser.add_argument("--collision-tolerance", type=float, default=0.05)
    parser.add_argument("--cluster-tolerance", type=float, default=0.05)
    parser.add_argument("--limit", type=int)
    return parser


def _prediction(row: dict[str, Any], values: list[str]) -> dict[str, Any]:
    return {
        "SubjectEntity": row["SubjectEntity"],
        "Relation": row["Relation"],
        "ObjectEntities": values,
    }


def _dominant(candidates: list[list[str]], tolerance: float) -> list[str]:
    clusters = cluster_numeric_candidates(candidates, tolerance)
    if clusters:
        return [format_number(clusters[0].representative)]
    return aggregate_candidates(candidates, {"strategy": "median"})


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0 < args.collision_tolerance < 1:
        raise ValueError("--collision-tolerance must be in (0, 1)")
    if not 0 < args.cluster_tolerance < 1:
        raise ValueError("--cluster-tolerance must be in (0, 1)")
    rows = read_jsonl(args.candidates)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("candidate input is empty")
    if any(row.get("Relation") != "hasArea" for row in rows):
        raise ValueError("area unit filter accepts only hasArea rows")

    filtered_dominant_predictions: list[dict[str, Any]] = []
    filtered_median_predictions: list[dict[str, Any]] = []
    baseline_dominant_predictions: list[dict[str, Any]] = []
    baseline_median_predictions: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    rows_with_collisions = 0
    removed_counts: list[int] = []
    unit_counts = {"square_mile": 0, "hectare": 0, "acre": 0}

    for index, row in enumerate(rows):
        candidates = row.get("Candidates")
        if not isinstance(candidates, list) or not all(
            isinstance(candidate, list) for candidate in candidates
        ):
            raise ValueError(f"missing Candidates at row {index}")
        filtered, collisions, removed_indices = filter_unit_collision_sources(
            candidates, args.collision_tolerance
        )
        if collisions:
            rows_with_collisions += 1
        removed_counts.append(len(removed_indices))
        for collision in collisions:
            unit_counts[collision.assumed_unit] += 1

        baseline_median = aggregate_candidates(
            candidates, {"strategy": "median"}
        )
        baseline_dominant = _dominant(candidates, args.cluster_tolerance)
        filtered_median = aggregate_candidates(
            filtered, {"strategy": "median"}
        )
        filtered_dominant = _dominant(filtered, args.cluster_tolerance)

        baseline_median_predictions.append(_prediction(row, baseline_median))
        baseline_dominant_predictions.append(
            _prediction(row, baseline_dominant)
        )
        filtered_median_predictions.append(_prediction(row, filtered_median))
        filtered_dominant_predictions.append(
            _prediction(row, filtered_dominant)
        )
        output_row = dict(row)
        source_selection = output_row.pop("FinalSelection", None)
        if isinstance(source_selection, dict):
            output_row["SourceFinalSelection"] = source_selection
        removed_index_set = set(removed_indices)
        output_row["FinalSelection"] = {
            "strategy": "cross_candidate_unit_filter",
            "ObjectEntities": filtered_dominant,
            "collision_tolerance": args.collision_tolerance,
            "cluster_tolerance": args.cluster_tolerance,
            "removed_candidate_indices": removed_indices,
            "retained_candidate_indices": [
                candidate_index
                for candidate_index in range(len(candidates))
                if candidate_index not in removed_index_set
            ],
            "collisions": [collision.as_dict() for collision in collisions],
            "converted_values_added": False,
        }
        output_rows.append(output_row)

    write_jsonl(args.output, filtered_dominant_predictions)
    write_jsonl(args.candidates_output, output_rows)
    if args.baseline_median_output:
        write_jsonl(args.baseline_median_output, baseline_median_predictions)
    if args.baseline_dominant_output:
        write_jsonl(args.baseline_dominant_output, baseline_dominant_predictions)
    if args.filtered_median_output:
        write_jsonl(args.filtered_median_output, filtered_median_predictions)
    if args.metrics_output:
        metrics = {
            "strategy": "cross_candidate_unit_filter",
            "rows": len(rows),
            "collision_tolerance": args.collision_tolerance,
            "cluster_tolerance": args.cluster_tolerance,
            "rows_with_collisions": rows_with_collisions,
            "total_removed_candidates": sum(removed_counts),
            "mean_removed_candidates": statistics.mean(removed_counts),
            "max_removed_candidates": max(removed_counts),
            "collision_edges_by_assumed_unit": unit_counts,
            "converted_values_added": False,
        }
        path = Path(args.metrics_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
