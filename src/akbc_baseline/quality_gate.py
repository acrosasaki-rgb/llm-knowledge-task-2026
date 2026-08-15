from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .data import read_jsonl


def assess_smoke(
    predictions: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    expected_rows: int,
    expected_candidates: int,
    require_final_selection: bool = False,
) -> dict[str, Any]:
    if len(predictions) != expected_rows:
        raise ValueError(
            f"expected {expected_rows} predictions, found {len(predictions)}"
        )
    if len(candidate_rows) != expected_rows:
        raise ValueError(
            f"expected {expected_rows} candidate rows, found {len(candidate_rows)}"
        )

    empty_predictions = 0
    empty_candidates = 0
    token_limit_candidates = 0
    thinking_budget_candidates = 0
    natural_think_end_candidates = 0
    forced_think_end_candidates = 0
    diagnostic_candidates = 0
    explicit_empty_candidates = 0
    parse_failure_candidates = 0
    final_selection_rows = 0
    final_selection_fallbacks = 0
    relation_diagnostics: dict[str, dict[str, int]] = {}
    total_candidates = expected_rows * expected_candidates
    for index, (prediction, candidate_row) in enumerate(
        zip(predictions, candidate_rows, strict=True)
    ):
        prediction_keys = set(prediction)
        if prediction_keys != {"SubjectEntity", "Relation", "ObjectEntities"}:
            raise ValueError(
                f"prediction row {index} has invalid keys: {sorted(prediction_keys)}"
            )
        if (
            prediction["SubjectEntity"] != candidate_row.get("SubjectEntity")
            or prediction["Relation"] != candidate_row.get("Relation")
        ):
            raise ValueError(f"prediction/candidate identity mismatch at row {index}")
        if not isinstance(prediction["ObjectEntities"], list):
            raise ValueError(f"ObjectEntities must be a list at row {index}")
        if not prediction["ObjectEntities"]:
            empty_predictions += 1

        final_selection = candidate_row.get("FinalSelection")
        if isinstance(final_selection, dict):
            final_selection_rows += 1
            final_selection_fallbacks += int(
                final_selection.get("used_fallback") is True
            )
            if final_selection.get("ObjectEntities") != prediction["ObjectEntities"]:
                raise ValueError(
                    f"final selection/prediction mismatch at row {index}"
                )
        elif require_final_selection:
            raise ValueError(f"missing FinalSelection at row {index}")

        candidates = candidate_row.get("Candidates")
        if not isinstance(candidates, list) or len(candidates) != expected_candidates:
            raise ValueError(
                f"candidate row {index} does not contain "
                f"{expected_candidates} candidates"
            )
        empty_candidates += sum(not candidate for candidate in candidates)

        diagnostics = candidate_row.get("CandidateDiagnostics", [])
        if diagnostics:
            if not isinstance(diagnostics, list) or len(diagnostics) != expected_candidates:
                raise ValueError(f"invalid CandidateDiagnostics at row {index}")
            diagnostic_candidates += len(diagnostics)
            relation = str(candidate_row.get("Relation"))
            relation_summary = relation_diagnostics.setdefault(
                relation,
                {
                    "thinking_enabled": 0,
                    "thinking_disabled": 0,
                    "has_think_end": 0,
                    "disabled_has_think_end": 0,
                    "explicit_empty": 0,
                    "parse_failure": 0,
                },
            )
            for item in diagnostics:
                if not isinstance(item, dict):
                    continue
                if item.get("enable_thinking") is True:
                    relation_summary["thinking_enabled"] += 1
                if item.get("enable_thinking") is False:
                    relation_summary["thinking_disabled"] += 1
                if item.get("has_think_end"):
                    relation_summary["has_think_end"] += 1
                    if item.get("enable_thinking") is False:
                        relation_summary["disabled_has_think_end"] += 1
                if item.get("parse_status") == "explicit_empty":
                    explicit_empty_candidates += 1
                    relation_summary["explicit_empty"] += 1
                if item.get("parse_status") == "parse_failure":
                    parse_failure_candidates += 1
                    relation_summary["parse_failure"] += 1
            token_limit_candidates += sum(
                bool(item.get("hit_token_limit"))
                for item in diagnostics
                if isinstance(item, dict)
            )
            thinking_budget_candidates += sum(
                bool(item.get("hit_thinking_budget"))
                for item in diagnostics
                if isinstance(item, dict)
            )
            natural_think_end_candidates += sum(
                bool(
                    item.get(
                        "natural_think_end",
                        item.get("has_think_end"),
                    )
                )
                for item in diagnostics
                if isinstance(item, dict)
            )
            forced_think_end_candidates += sum(
                bool(item.get("forced_think_end"))
                for item in diagnostics
                if isinstance(item, dict)
            )

    return {
        "rows": expected_rows,
        "candidates_per_row": expected_candidates,
        "total_candidates": total_candidates,
        "empty_predictions": empty_predictions,
        "empty_prediction_rate": empty_predictions / expected_rows,
        "empty_candidates": empty_candidates,
        "empty_candidate_rate": empty_candidates / total_candidates,
        "diagnostic_candidates": diagnostic_candidates,
        "token_limit_candidates": token_limit_candidates,
        "token_limit_rate": (
            token_limit_candidates / diagnostic_candidates
            if diagnostic_candidates
            else None
        ),
        "thinking_budget_candidates": thinking_budget_candidates,
        "thinking_budget_rate": (
            thinking_budget_candidates / diagnostic_candidates
            if diagnostic_candidates
            else None
        ),
        "natural_think_end_candidates": natural_think_end_candidates,
        "natural_think_end_rate": (
            natural_think_end_candidates / diagnostic_candidates
            if diagnostic_candidates
            else None
        ),
        "forced_think_end_candidates": forced_think_end_candidates,
        "forced_think_end_rate": (
            forced_think_end_candidates / diagnostic_candidates
            if diagnostic_candidates
            else None
        ),
        "explicit_empty_candidates": explicit_empty_candidates,
        "explicit_empty_rate": (
            explicit_empty_candidates / diagnostic_candidates
            if diagnostic_candidates
            else None
        ),
        "parse_failure_candidates": parse_failure_candidates,
        "parse_failure_rate": (
            parse_failure_candidates / diagnostic_candidates
            if diagnostic_candidates
            else None
        ),
        "final_selection_rows": final_selection_rows,
        "final_selection_fallbacks": final_selection_fallbacks,
        "relation_diagnostics": relation_diagnostics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the first reasoning shard before the full run"
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--expected-rows", type=int, default=50)
    parser.add_argument("--expected-candidates", type=int, default=20)
    parser.add_argument("--maximum-empty-prediction-rate", type=float, default=0.20)
    parser.add_argument("--maximum-empty-candidate-rate", type=float, default=0.25)
    parser.add_argument("--require-final-selection", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = assess_smoke(
        read_jsonl(args.predictions),
        read_jsonl(args.candidates),
        expected_rows=args.expected_rows,
        expected_candidates=args.expected_candidates,
        require_final_selection=args.require_final_selection,
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

    failures = []
    if report["empty_prediction_rate"] > args.maximum_empty_prediction_rate:
        failures.append(
            "empty prediction rate "
            f"{report['empty_prediction_rate']:.3f} exceeds "
            f"{args.maximum_empty_prediction_rate:.3f}"
        )
    if report["empty_candidate_rate"] > args.maximum_empty_candidate_rate:
        failures.append(
            "empty candidate rate "
            f"{report['empty_candidate_rate']:.3f} exceeds "
            f"{args.maximum_empty_candidate_rate:.3f}"
        )
    if failures:
        raise RuntimeError("; ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
