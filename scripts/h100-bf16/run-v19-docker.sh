#!/usr/bin/env bash
# Reproduce the submitted V19 system on an NVIDIA Docker host.
# Usage: AKBC_SPLIT=test AKBC_DATASET_DIR=../dataset2026 bash scripts/h100-bf16/run-v19-docker.sh
set -Eeuo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

split="${AKBC_SPLIT:-test}"
dataset_dir="$(cd -- "${AKBC_DATASET_DIR:-../dataset2026}" && pwd)"
dataset_ref="30d8cfaaa7af5b236054cfb361f57b7d0c1e6e57"
case "${split}" in val|test) ;; *) echo "AKBC_SPLIT must be val or test" >&2; exit 2 ;; esac
for path in data/train.jsonl data/val.jsonl "data/${split}.jsonl"; do
  test -f "${dataset_dir}/${path}" || { echo "missing ${dataset_dir}/${path}" >&2; exit 2; }
done
actual_dataset_ref="$(git -C "${dataset_dir}" rev-parse HEAD)"
if [[ "${actual_dataset_ref}" != "${dataset_ref}" ]]; then
  echo "dataset commit mismatch: expected ${dataset_ref}, got ${actual_dataset_ref}" >&2
  exit 2
fi

command -v docker >/dev/null
docker info >/dev/null
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi >/dev/null

code_commit="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
mkdir -p outputs reports .cache/v19-huggingface
image_tag="akbc-v19:gemma-3-27b-pt"
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
  --volume "${repo_root}/.cache/v19-huggingface:/cache/huggingface"
  --volume "${dataset_dir}:/workspace/dataset:ro"
  --env "AKBC_CODE_COMMIT=${code_commit}"
  --env "AKBC_SPLIT=${split}"
  --env "AKBC_DATA_FILE=/workspace/dataset/data/${split}.jsonl"
  --env "AKBC_TRAIN_FILE=/workspace/dataset/data/train.jsonl"
  --env "AKBC_VAL_FILE=/workspace/dataset/data/val.jsonl"
)
for variable_name in HF_TOKEN AKBC_GPU HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy; do
  if [[ -n "${!variable_name:-}" ]]; then
    run_args+=(--env "${variable_name}")
  fi
done
docker run "${run_args[@]}" "${image_tag}" \
  /opt/akbc/scripts/h100-bf16/run-v19-container.sh
