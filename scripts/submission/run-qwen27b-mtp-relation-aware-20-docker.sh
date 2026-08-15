#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

export AKBC_MODEL_KEY="qwen3.5-27b-mtp-relation-aware-20"
export AKBC_CONFIG="configs/experiment-qwen-3.5-27b-mtp-relation-aware-20.yaml"
export AKBC_EXPECTED_CANDIDATES="20"

exec bash "${repo_root}/scripts/submission/run-qwen27b-mtp-docker.sh" "$@"
