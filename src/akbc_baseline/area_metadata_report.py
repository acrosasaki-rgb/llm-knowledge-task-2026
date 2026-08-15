from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from .data import read_jsonl


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    unit_counts: Counter[str] = Counter()
    scope_counts: Counter[str] = Counter()
    reference_year_counts: Counter[str] = Counter()
    consistency_counts: Counter[str] = Counter()
    metadata_token_limits = 0
    final_status_counts: Counter[str] = Counter()
    final_token_limits = 0
    final_fallbacks = 0
    total = 0

    for row in rows:
        for diagnostic in row.get("CandidateDiagnostics", []):
            if not isinstance(diagnostic, dict):
                continue
            total += 1
            metadata = diagnostic.get("metadata")
            if not isinstance(metadata, dict):
                status_counts["missing"] += 1
            else:
                status_counts[str(metadata.get("status", "missing"))] += 1
                unit_counts[str(metadata.get("unit", "missing"))] += 1
                scope_counts[str(metadata.get("scope", "missing"))] += 1
                reference_year_counts[
                    str(metadata.get("reference_year", "missing"))
                ] += 1
                if metadata.get("value_matches_candidate") is True:
                    consistency_counts["value_match"] += 1
                else:
                    consistency_counts["value_mismatch_or_unknown"] += 1
                conversion = metadata.get("conversion_consistent")
                if conversion is True:
                    consistency_counts["conversion_consistent"] += 1
                elif conversion is False:
                    consistency_counts["conversion_inconsistent"] += 1
                else:
                    consistency_counts["conversion_unknown"] += 1
            generation = diagnostic.get("metadata_generation_diagnostics")
            if isinstance(generation, dict) and generation.get("hit_token_limit"):
                metadata_token_limits += 1

        final_selection = row.get("FinalSelection")
        if not isinstance(final_selection, dict):
            final_status_counts["missing"] += 1
            continue
        final_status_counts[str(final_selection.get("parse_status", "missing"))] += 1
        final_fallbacks += int(final_selection.get("used_fallback") is True)
        generation = final_selection.get("generation_diagnostics")
        if isinstance(generation, dict) and generation.get("hit_token_limit"):
            final_token_limits += 1

    return {
        "rows": len(rows),
        "total_metadata": total,
        "metadata_status_counts": dict(sorted(status_counts.items())),
        "unit_counts": dict(sorted(unit_counts.items())),
        "scope_counts": dict(sorted(scope_counts.items())),
        "reference_year_counts": dict(sorted(reference_year_counts.items())),
        "logical_consistency_counts": dict(sorted(consistency_counts.items())),
        "metadata_token_limit_candidates": metadata_token_limits,
        "final_selection_status_counts": dict(sorted(final_status_counts.items())),
        "final_selection_fallbacks": final_fallbacks,
        "final_selection_token_limits": final_token_limits,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize hasArea metadata estimates and final selections"
    )
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = summarize(read_jsonl(args.candidates))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
