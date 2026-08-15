from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

from .aggregation import aggregate_candidates
from .area_clustering import (
    area_cluster_selection_messages,
    cluster_choice_records,
    cluster_numeric_candidates,
    format_number,
    match_cluster_choice,
)
from .backends import create_backend
from .config import ModelConfig
from .data import read_jsonl, write_jsonl
from .parsing import parse_object_entities_with_status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rerank saved hasArea candidates with clustered choices"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidates-output", required=True)
    parser.add_argument("--median-output")
    parser.add_argument("--dominant-output")
    parser.add_argument("--metrics-output")
    parser.add_argument("--limit", type=int)
    return parser


def _selection_seed(seed: int, subject: str, relation: str) -> int:
    digest = hashlib.sha256(
        f"{seed}\0{relation}\0{subject}\0cluster-choice".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big")


def _prediction(row: dict[str, Any], values: list[str]) -> dict[str, Any]:
    return {
        "SubjectEntity": row["SubjectEntity"],
        "Relation": row["Relation"],
        "ObjectEntities": values,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ModelConfig.from_yaml(args.config)
    policy = config.aggregation.get("hasArea", {})
    if policy.get("strategy") != "cluster_choice":
        raise ValueError("hasArea config must use cluster_choice")
    tolerance = float(policy.get("tolerance", 0.05))
    rows = read_jsonl(args.candidates)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("candidate input is empty")
    if any(row.get("Relation") != "hasArea" for row in rows):
        raise ValueError("area cluster rerank accepts only hasArea rows")

    backend = create_backend(config)
    backend.reset_peak_memory_stats()
    predictions: list[dict[str, Any]] = []
    median_predictions: list[dict[str, Any]] = []
    dominant_predictions: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    skipped = 0
    fallbacks = 0
    selection_calls = 0
    started = time.monotonic()

    for index, row in enumerate(rows, start=1):
        subject = str(row["SubjectEntity"])
        relation = str(row["Relation"])
        candidates = row.get("Candidates")
        if not isinstance(candidates, list):
            raise ValueError(f"missing Candidates at row {index - 1}")
        clusters = cluster_numeric_candidates(candidates, tolerance)
        median_values = aggregate_candidates(
            candidates, {"strategy": "median"}
        )
        dominant_values = (
            [format_number(clusters[0].representative)]
            if clusters
            else median_values
        )

        selection_text: str | None = None
        selection_diagnostics: dict[str, object] = {}
        parse_status = "skipped"
        used_fallback = False
        if len(clusters) <= 1:
            skipped += 1
            selected_values = dominant_values
        else:
            selection_calls += 1
            selection_text = backend.generate(
                area_cluster_selection_messages(subject, clusters),
                seed=_selection_seed(config.seed, subject, relation),
                enable_thinking=config.relation_thinking.get(relation),
            )
            diagnostics_reader = getattr(
                backend, "last_generation_diagnostics", None
            )
            selection_diagnostics = (
                diagnostics_reader() if diagnostics_reader else {}
            )
            parsed = parse_object_entities_with_status(selection_text, relation)
            parse_status = parsed.status
            chosen = match_cluster_choice(parsed.object_entities, clusters)
            used_fallback = chosen is None
            if chosen is None:
                chosen = clusters[0]
                fallbacks += 1
            selected_values = [format_number(chosen.representative)]

        final_selection = {
            "strategy": "cluster_choice",
            "ObjectEntities": selected_values,
            "text": selection_text,
            "parse_status": parse_status,
            "used_fallback": used_fallback,
            "skipped_single_cluster": len(clusters) <= 1,
            "tolerance": tolerance,
            "choices": cluster_choice_records(clusters),
            "generation_diagnostics": selection_diagnostics,
        }
        output_row = dict(row)
        source_selection = output_row.pop("FinalSelection", None)
        if isinstance(source_selection, dict):
            output_row["SourceFinalSelection"] = source_selection
        output_row["FinalSelection"] = final_selection
        output_rows.append(output_row)
        predictions.append(_prediction(row, selected_values))
        median_predictions.append(_prediction(row, median_values))
        dominant_predictions.append(_prediction(row, dominant_values))
        print(
            f"{config.name}: reranked {index}/{len(rows)} rows in "
            f"{time.monotonic() - started:.1f}s",
            flush=True,
        )

    write_jsonl(args.output, predictions)
    write_jsonl(args.candidates_output, output_rows)
    if args.median_output:
        write_jsonl(args.median_output, median_predictions)
    if args.dominant_output:
        write_jsonl(args.dominant_output, dominant_predictions)
    if args.metrics_output:
        metrics = {
            "name": config.name,
            "model_id": config.model_id,
            "parameter_count_billion": config.parameter_count_billion,
            "num_candidates": config.num_candidates,
            "rows": len(rows),
            "empty_predictions": sum(
                not prediction["ObjectEntities"] for prediction in predictions
            ),
            "tolerance": tolerance,
            "selection_calls": selection_calls,
            "single_cluster_skips": skipped,
            "fallbacks": fallbacks,
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
