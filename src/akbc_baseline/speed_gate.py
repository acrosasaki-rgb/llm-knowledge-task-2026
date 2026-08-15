from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def estimate_seconds(
    elapsed_seconds: float, measured_rows: int, target_rows: int
) -> float:
    if elapsed_seconds <= 0 or measured_rows <= 0 or target_rows <= 0:
        raise ValueError("elapsed time and row counts must be positive")
    return elapsed_seconds / measured_rows * target_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reject an inference run projected to exceed its CI budget"
    )
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--target-rows", required=True, type=int)
    parser.add_argument("--maximum-seconds", required=True, type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    projected = estimate_seconds(
        float(metrics["elapsed_seconds"]),
        int(metrics["rows"]),
        args.target_rows,
    )
    print(
        f"projected val+test runtime: {projected / 3600:.2f}h "
        f"for {args.target_rows} rows"
    )
    if projected > args.maximum_seconds:
        raise RuntimeError(
            f"projected runtime {projected:.0f}s exceeds "
            f"budget {args.maximum_seconds:.0f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
