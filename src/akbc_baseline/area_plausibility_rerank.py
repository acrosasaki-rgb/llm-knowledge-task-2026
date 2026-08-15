from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from .aggregation import aggregate_candidates
from .area_clustering import format_number
from .area_plausibility import (
    build_area_hypotheses,
    decimal_relation_records,
    dimension_estimate_messages,
    parse_dimension_estimate,
    parse_scale_estimate,
    scale_estimate_messages,
    select_area_hypothesis,
)
from .backends import create_backend
from .config import ModelConfig
from .data import read_jsonl, write_jsonl


STRATEGIES = {
    "scale-direct": (True, False, False),
    "scale-unit": (True, False, True),
    "dimension-direct": (False, True, False),
    "dimension-unit": (False, True, True),
    "combined-direct": (True, True, False),
    "combined-unit": (True, True, True),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rerank hasArea candidates with blind scale and dimensions"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidates-output", required=True)
    parser.add_argument("--raw-median-output")
    parser.add_argument("--dominant-output")
    for strategy in STRATEGIES:
        parser.add_argument(f"--{strategy}-output")
    parser.add_argument("--metrics-output")
    parser.add_argument("--limit", type=int)
    return parser


def _seed(seed: int, subject: str, relation: str, task: str) -> int:
    digest = hashlib.sha256(
        f"{seed}\0{relation}\0{subject}\0{task}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big")


def _prediction(row: dict[str, Any], values: list[str]) -> dict[str, Any]:
    return {
        "SubjectEntity": row["SubjectEntity"],
        "Relation": row["Relation"],
        "ObjectEntities": values,
    }


def _bounds(estimate: dict[str, Any]) -> tuple[float, float] | None:
    lower = estimate.get("lower_km2")
    upper = estimate.get("upper_km2")
    if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
        return float(lower), float(upper)
    return None


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ModelConfig.from_yaml(args.config)
    rows = read_jsonl(args.candidates)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("candidate input is empty")
    if any(row.get("Relation") != "hasArea" for row in rows):
        raise ValueError("area plausibility rerank accepts only hasArea rows")

    backend = create_backend(config)
    backend.reset_peak_memory_stats()
    predictions: dict[str, list[dict[str, Any]]] = {
        strategy: [] for strategy in STRATEGIES
    }
    raw_predictions: list[dict[str, Any]] = []
    dominant_predictions: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    override_counts: Counter[str] = Counter()
    selected_units: dict[str, Counter[str]] = {
        strategy: Counter() for strategy in STRATEGIES
    }
    scale_parsed = 0
    dimensions_parsed = 0
    rows_with_decimal_relations = 0
    started = time.monotonic()

    for index, row in enumerate(rows, start=1):
        subject = str(row["SubjectEntity"])
        relation = str(row["Relation"])
        candidates = row.get("Candidates")
        if not isinstance(candidates, list):
            raise ValueError(f"missing Candidates at row {index - 1}")

        scale_text = backend.generate(
            scale_estimate_messages(subject),
            seed=_seed(config.seed, subject, relation, "scale"),
            enable_thinking=False,
        )
        diagnostics_reader = getattr(backend, "last_generation_diagnostics", None)
        scale_diagnostics = diagnostics_reader() if diagnostics_reader else {}
        scale = parse_scale_estimate(scale_text)
        scale_parsed += int(scale["status"] == "parsed")

        dimension_text = backend.generate(
            dimension_estimate_messages(subject),
            seed=_seed(config.seed, subject, relation, "dimensions"),
            enable_thinking=False,
        )
        dimension_diagnostics = diagnostics_reader() if diagnostics_reader else {}
        dimensions = parse_dimension_estimate(dimension_text)
        dimensions_parsed += int(dimensions["status"] == "parsed")

        clusters, hypotheses = build_area_hypotheses(candidates)
        decimal_relations = decimal_relation_records(clusters)
        rows_with_decimal_relations += int(bool(decimal_relations))
        raw_values = aggregate_candidates(candidates, {"strategy": "median"})
        dominant_values = (
            [format_number(clusters[0].representative)] if clusters else raw_values
        )
        raw_predictions.append(_prediction(row, raw_values))
        dominant_predictions.append(_prediction(row, dominant_values))

        selections: dict[str, dict[str, Any]] = {}
        selected_values: dict[str, list[str]] = {}
        for strategy, (use_scale, use_dimensions, allow_units) in STRATEGIES.items():
            selected, details = select_area_hypothesis(
                hypotheses,
                scale_bounds=_bounds(scale) if use_scale else None,
                dimension_bounds=(
                    _bounds(dimensions) if use_dimensions else None
                ),
                allow_unit_conversion=allow_units,
            )
            values = (
                [format_number(selected.normalized_value_km2)]
                if selected is not None
                else dominant_values
            )
            details = {
                **details,
                "ObjectEntities": values,
                "selected_hypothesis": (
                    selected.as_dict() if selected is not None else None
                ),
            }
            selections[strategy] = details
            selected_values[strategy] = values
            predictions[strategy].append(_prediction(row, values))
            override_counts[strategy] += int(
                details.get("overrode_base") is True
            )
            if selected is not None:
                selected_units[strategy][selected.assumed_unit] += 1

        final_selection = {
            "strategy": "combined-unit-plausibility",
            "ObjectEntities": selected_values["combined-unit"],
            "used_fallback": False,
            "scale_estimate": scale,
            "dimension_estimate": dimensions,
            "decimal_relations": decimal_relations,
            "selections": selections,
        }
        output_row = dict(row)
        source_selection = output_row.pop("FinalSelection", None)
        if isinstance(source_selection, dict):
            output_row["SourceFinalSelection"] = source_selection
        output_row["ScaleEstimate"] = {
            "text": scale_text,
            "parsed": scale,
            "generation_diagnostics": scale_diagnostics,
        }
        output_row["DimensionEstimate"] = {
            "text": dimension_text,
            "parsed": dimensions,
            "generation_diagnostics": dimension_diagnostics,
        }
        output_row["AreaHypotheses"] = [
            hypothesis.as_dict() for hypothesis in hypotheses
        ]
        output_row["FinalSelection"] = final_selection
        output_rows.append(output_row)
        print(
            f"{config.name}: plausibility reranked {index}/{len(rows)} rows "
            f"in {time.monotonic() - started:.1f}s",
            flush=True,
        )

    write_jsonl(args.output, predictions["combined-unit"])
    write_jsonl(args.candidates_output, output_rows)
    if args.raw_median_output:
        write_jsonl(args.raw_median_output, raw_predictions)
    if args.dominant_output:
        write_jsonl(args.dominant_output, dominant_predictions)
    for strategy in STRATEGIES:
        path = getattr(args, f"{strategy.replace('-', '_')}_output")
        if path:
            write_jsonl(path, predictions[strategy])
    if args.metrics_output:
        metrics = {
            "name": config.name,
            "model_id": config.model_id,
            "parameter_count_billion": config.parameter_count_billion,
            "num_candidates": config.num_candidates,
            "rows": len(rows),
            "empty_predictions": sum(
                not row["ObjectEntities"]
                for row in predictions["combined-unit"]
            ),
            "scale_estimates_parsed": scale_parsed,
            "dimension_estimates_parsed": dimensions_parsed,
            "rows_with_decimal_relations": rows_with_decimal_relations,
            "override_counts": dict(override_counts),
            "selected_units": {
                strategy: dict(counts)
                for strategy, counts in selected_units.items()
            },
            "elapsed_seconds": time.monotonic() - started,
            "peak_cuda_memory_gib": backend.peak_cuda_memory_gib(),
        }
        metrics_path = Path(args.metrics_output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
