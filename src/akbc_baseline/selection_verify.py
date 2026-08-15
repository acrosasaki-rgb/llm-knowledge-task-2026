from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from .config import ModelConfig


def verify_manifest(
    manifest: dict[str, Any],
    *,
    config_path: Path,
    model_key: str,
    dataset_ref: str,
    commit_sha: str,
    container_image: str,
    expected_candidates: int,
) -> None:
    model_config = ModelConfig.from_yaml(config_path)
    expected = {
        "schema_version": 2,
        "model_key": model_key,
        "config": config_path.as_posix(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "num_candidates": expected_candidates,
        "dataset_ref": dataset_ref,
        "commit_sha": commit_sha,
        "execution_environment": "external-ssh-docker-gpu",
        "container_image": container_image,
    }
    if model_config.num_candidates != expected_candidates:
        raise ValueError(
            "invalid selection config: "
            f"expected {expected_candidates} candidates, "
            f"found {model_config.num_candidates}"
        )
    mismatches = [
        f"{key}: expected {value!r}, found {manifest.get(key)!r}"
        for key, value in expected.items()
        if manifest.get(key) != value
    ]
    pipeline_url = manifest.get("pipeline_url")
    if not isinstance(pipeline_url, str) or not pipeline_url.startswith(
        ("http://", "https://")
    ):
        mismatches.append("pipeline_url must be an HTTP(S) URL")
    if (
        manifest.get("selection_basis")
        != "manual review of complete validation artifacts"
    ):
        mismatches.append("selection_basis does not record manual validation review")
    if mismatches:
        raise ValueError("invalid selection manifest: " + "; ".join(mismatches))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify a GitLab-to-external-host model selection manifest"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--dataset-ref", required=True)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--container-image", required=True)
    parser.add_argument("--expected-candidates", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("selection manifest must contain a JSON object")
    verify_manifest(
        manifest,
        config_path=Path(args.config),
        model_key=args.model_key,
        dataset_ref=args.dataset_ref,
        commit_sha=args.commit_sha,
        container_image=args.container_image,
        expected_candidates=args.expected_candidates,
    )
    print("selection manifest verified", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
