#!/usr/bin/env bash
# Host-side launcher for the think-content sample (12 hasArea rows x 20
# candidates with raw thinking persisted). Reuses the BF16 validation image;
# only the container entrypoint differs.
#
# Usage, from the repository root on the GPU host:
#   bash scripts/h100-bf16/run-gemma-multiroute-area-docker.sh
set -Eeuo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

command -v docker >/dev/null
docker info 2>/dev/null | grep -qi nvidia \
  || nvidia-smi >/dev/null 2>&1 \
  || { echo "no NVIDIA runtime visible on this host" >&2; exit 1; }

code_commit="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

mkdir -p outputs reports .cache/bf16-dataset .cache/bf16-huggingface

image_tag="akbc-qwen27b-bf16-val:gemma-multiroute-area"
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
  --entrypoint bash
  --volume "${repo_root}/outputs:/workspace/run/outputs"
  --volume "${repo_root}/reports:/workspace/run/reports"
  --volume "${repo_root}/.cache/bf16-dataset:/cache/dataset2026"
  --volume "${repo_root}/.cache/bf16-huggingface:/cache/huggingface"
)
for variable_name in \
  AKBC_MODEL_KEY AKBC_CONFIG AKBC_PARALLEL AKBC_CLIENTS_PER_GPU AKBC_CTX_PER_SLOT HF_TOKEN \
  HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy; do
  if [[ -n "${!variable_name:-}" ]]; then
    run_args+=(--env "${variable_name}")
  fi
done
docker run "${run_args[@]}" "${image_tag}" \
  /opt/akbc/scripts/h100-bf16/run-gemma-multiroute-area-container.sh
