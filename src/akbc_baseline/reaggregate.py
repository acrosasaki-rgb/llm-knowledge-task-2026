from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .aggregation import aggregate_candidates
from .config import ModelConfig
from .data import read_jsonl, write_jsonl


MODEL_ASSISTED = {"cluster_choice", "metadata_judge"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Re-run aggregation over a saved candidates file and emit official "
            "predictions without repeating model inference"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--input",
        help=(
            "official input JSONL; when given, row count, order, SubjectEntity "
            "and Relation are verified against it"
        ),
    )
    parser.add_argument(
        "--reuse-final-selection",
        action="store_true",
        help=(
            "keep the stored FinalSelection for relations whose strategy needs "
            "model assistance instead of failing"
        ),
    )
    return parser


def reaggregate_row(
    row: dict[str, Any],
    aggregation: dict[str, dict[str, Any]],
    *,
    reuse_final_selection: bool,
) -> list[str]:
    relation = row.get("Relation")
    policy = aggregation.get(relation)
    strategy = (policy or {}).get("strategy")
    if strategy in MODEL_ASSISTED:
        stored = row.get("FinalSelection")
        if reuse_final_selection and isinstance(stored, dict):
            values = stored.get("ObjectEntities")
            if isinstance(values, list):
                return [str(value) for value in values]
        raise ValueError(
            f"{strategy} for {relation} needs model assistance; rerun inference "
            "or pass --reuse-final-selection"
        )
    candidates = row.get("Candidates")
    if not isinstance(candidates, list) or not all(
        isinstance(candidate, list) for candidate in candidates
    ):
        raise ValueError(
            f"missing Candidates for {row.get('SubjectEntity')!r} / {relation}"
        )
    return aggregate_candidates(
        candidates, policy, diagnostics=row.get("CandidateDiagnostics")
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ModelConfig.from_yaml(args.config)
    rows = read_jsonl(args.candidates)
    if not rows:
        raise ValueError("candidate input is empty")
    if args.input:
        expected = read_jsonl(args.input)
        if len(expected) != len(rows):
            raise ValueError(
                f"expected {len(expected)} rows, found {len(rows)}"
            )
        for index, (want, got) in enumerate(zip(expected, rows)):
            if (
                want.get("SubjectEntity") != got.get("SubjectEntity")
                or want.get("Relation") != got.get("Relation")
            ):
                raise ValueError(f"row {index} does not match the official input")

    predictions = [
        {
            "SubjectEntity": row["SubjectEntity"],
            "Relation": row["Relation"],
            "ObjectEntities": reaggregate_row(
                row,
                config.aggregation,
                reuse_final_selection=args.reuse_final_selection,
            ),
        }
        for row in rows
    ]
    write_jsonl(args.output, predictions)
    counts: dict[str, int] = {}
    for row in predictions:
        counts[row["Relation"]] = counts.get(row["Relation"], 0) + 1
    print(json.dumps({"rows": len(predictions), "by_relation": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
