from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from .config import ModelConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record the manually selected submission model"
    )
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-ref", required=True)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--pipeline-url", required=True)
    parser.add_argument("--container-image", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config)
    model_config = ModelConfig.from_yaml(config_path)
    manifest = {
        "schema_version": 2,
        "model_key": args.model_key,
        "config": config_path.as_posix(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "num_candidates": model_config.num_candidates,
        "dataset_ref": args.dataset_ref,
        "commit_sha": args.commit_sha,
        "pipeline_url": args.pipeline_url,
        "execution_environment": "external-ssh-docker-gpu",
        "container_image": args.container_image,
        "selection_basis": "manual review of complete validation artifacts",
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
