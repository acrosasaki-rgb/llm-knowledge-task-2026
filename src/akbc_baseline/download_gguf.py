from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download one pinned GGUF file")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--path-output", required=True)
    parser.add_argument("--minimum-gib", required=True, type=float)
    parser.add_argument("--maximum-gib", required=True, type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "install requirements-gguf.txt before downloading GGUF weights"
        ) from exc

    model_path = Path(
        hf_hub_download(
            repo_id=args.repo,
            filename=args.filename,
            revision=args.revision,
            cache_dir=args.cache_dir,
        )
    ).resolve()
    size_gib = model_path.stat().st_size / 1024**3
    if not args.minimum_gib <= size_gib <= args.maximum_gib:
        raise RuntimeError(
            f"unexpected GGUF size: {size_gib:.2f} GiB; "
            f"expected {args.minimum_gib}-{args.maximum_gib} GiB"
        )
    output_path = Path(args.path_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(str(model_path) + "\n", encoding="utf-8")
    print(
        f"downloaded {args.repo}@{args.revision}:{args.filename} "
        f"({size_gib:.2f} GiB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
