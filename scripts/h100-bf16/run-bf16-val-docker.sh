#!/usr/bin/env bash
# Host-side launcher for the BF16 validation experiment on a rented H100.
#
# Usage, from the repository root on the GPU host:
#   bash scripts/h100-bf16/run-bf16-val-docker.sh smoke
#   bash scripts/h100-bf16/run-bf16-val-docker.sh full
#
# smoke runs the first 16 validation rows across all slots and prints a
# projection for the full run; review its artifacts before starting full.
# This experiment is validation-only and independent of the GitLab CI jobs
# and of the test submission flow; it needs no selection manifest.
set -Eeuo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

mode="${1:?usage: run-bf16-val-docker.sh <smoke|full>}"
[[ "${mode}" == "smoke" || "${mode}" == "full" ]] || {
  echo "mode must be smoke or full, got: ${mode}" >&2
  exit 1
}

command -v git >/dev/null
command -v docker >/dev/null
docker info 2>/dev/null | grep -qi nvidia \
  || nvidia-smi >/dev/null 2>&1 \
  || { echo "no NVIDIA runtime visible on this host" >&2; exit 1; }

code_commit="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
if [[ -n "$(git status --porcelain --untracked-files=normal 2>/dev/null)" ]]; then
  echo "warning: working tree is not clean; recording commit ${code_commit}+dirty"
  code_commit="${code_commit}+dirty"
fi

mkdir -p outputs reports .cache/bf16-dataset .cache/bf16-huggingface

image_tag="akbc-qwen27b-bf16-val:${code_commit:0:12}"
build_args=(--build-arg "AKBC_CODE_COMMIT=${code_commit}")
for proxy_name in HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy; do
  if [[ -n "${!proxy_name:-}" ]]; then
    build_args+=(--build-arg "${proxy_name}")
  fi
done
docker build \
  --file docker/h100-bf16-val.Dockerfile \
  --tag "${image_tag}" \
  "${build_args[@]}" \
  .

run_args=(
  --rm
  --gpus all
  --env "AKBC_RUN_MODE=${mode}"
  --volume "${repo_root}/outputs:/workspace/run/outputs"
  --volume "${repo_root}/reports:/workspace/run/reports"
  --volume "${repo_root}/.cache/bf16-dataset:/cache/dataset2026"
  --volume "${repo_root}/.cache/bf16-huggingface:/cache/huggingface"
)
for variable_name in \
  AKBC_MODEL_KEY AKBC_CONFIG AKBC_EXPECTED_CANDIDATES \
  AKBC_PARALLEL AKBC_CTX_PER_SLOT HF_TOKEN \
  HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy; do
  if [[ -n "${!variable_name:-}" ]]; then
    run_args+=(--env "${variable_name}")
  fi
done
docker run "${run_args[@]}" "${image_tag}"
