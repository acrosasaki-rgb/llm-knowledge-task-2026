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
    rejection_counts: Counter[str] = Counter()
    accepted_per_row: list[int] = []
    audit_token_limits = 0
    total = 0
    for row in rows:
        accepted = 0
        diagnostics = row.get("CandidateDiagnostics", [])
        for diagnostic in diagnostics:
            if not isinstance(diagnostic, dict):
                continue
            total += 1
            audit = diagnostic.get("audit")
            if not isinstance(audit, dict):
                status_counts["missing"] += 1
                continue
            status_counts[str(audit.get("status", "missing"))] += 1
            unit_counts[str(audit.get("unit", "missing"))] += 1
            scope_counts[str(audit.get("scope", "missing"))] += 1
            for reason in audit.get("rejection_reasons", []):
                rejection_counts[str(reason)] += 1
            if audit.get("usable") is True:
                accepted += 1
            audit_generation = diagnostic.get("audit_generation_diagnostics")
            if isinstance(audit_generation, dict) and audit_generation.get(
                "hit_token_limit"
            ):
                audit_token_limits += 1
        accepted_per_row.append(accepted)
    return {
        "rows": len(rows),
        "total_audits": total,
        "status_counts": dict(sorted(status_counts.items())),
        "unit_counts": dict(sorted(unit_counts.items())),
        "scope_counts": dict(sorted(scope_counts.items())),
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "accepted_audits": sum(accepted_per_row),
        "accepted_audit_rate": (
            sum(accepted_per_row) / total if total else 0.0
        ),
        "rows_with_no_accepted_audit": sum(
            count == 0 for count in accepted_per_row
        ),
        "minimum_accepted_per_row": min(accepted_per_row, default=0),
        "maximum_accepted_per_row": max(accepted_per_row, default=0),
        "average_accepted_per_row": (
            sum(accepted_per_row) / len(accepted_per_row)
            if accepted_per_row
            else 0.0
        ),
        "audit_token_limit_candidates": audit_token_limits,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize hasArea candidate audits")
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
