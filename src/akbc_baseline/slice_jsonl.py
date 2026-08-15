from __future__ import annotations

import argparse
from typing import Sequence

from .data import read_jsonl, write_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write an ordered JSONL slice")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.offset < 0:
        raise ValueError("--offset must be zero or positive")
    if args.limit < 1:
        raise ValueError("--limit must be positive")
    rows = read_jsonl(args.input)
    selected = rows[args.offset : args.offset + args.limit]
    if len(selected) != args.limit:
        raise ValueError(
            f"expected {args.limit} rows at offset {args.offset}, "
            f"found {len(selected)}"
        )
    write_jsonl(args.output, selected)
    print(f"wrote {len(selected)} rows to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
