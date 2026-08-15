#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

manifest="${1:-reports/selection.json}"
model_key="${AKBC_MODEL_KEY:-qwen3.5-27b-mtp-thinking}"
config="${AKBC_CONFIG:-configs/experiment-qwen-3.5-27b-mtp-thinking.yaml}"
expected_candidates="${AKBC_EXPECTED_CANDIDATES:-5}"
dataset_ref="30d8cfaaa7af5b236054cfb361f57b7d0c1e6e57"
container_image="ghcr.io/ggml-org/llama.cpp:full-cuda@sha256:11b0e950e081777cf326598bb2eff2ab0815f02405bf95c6650b34027750114e"

test -f "${manifest}" || {
  echo "selection manifest is missing: ${manifest}" >&2
  exit 1
}
command -v git >/dev/null
command -v python3 >/dev/null
command -v docker >/dev/null

if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "the external submission checkout must be clean" >&2
  exit 1
fi

code_commit="$(git rev-parse HEAD)"
PYTHONPATH="${repo_root}/src" python3 -m akbc_baseline.selection_verify \
  --manifest "${manifest}" \
  --config "${config}" \
  --model-key "${model_key}" \
  --dataset-ref "${dataset_ref}" \
  --commit-sha "${code_commit}" \
  --container-image "${container_image}" \
  --expected-candidates "${expected_candidates}"

mkdir -p outputs reports .cache/submission-dataset .cache/submission-huggingface
manifest_path="$(cd -- "$(dirname -- "${manifest}")" && pwd)/$(basename -- "${manifest}")"
canonical_manifest="${repo_root}/reports/selection.json"
if [[ "${manifest_path}" != "${canonical_manifest}" ]]; then
  cp -- "${manifest_path}" "${canonical_manifest}"
fi

image_tag="akbc-qwen27b-mtp-submission:${code_commit:0:12}"
build_args=(--build-arg "AKBC_CODE_COMMIT=${code_commit}")
for proxy_name in HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy; do
  if [[ -n "${!proxy_name:-}" ]]; then
    build_args+=(--build-arg "${proxy_name}")
  fi
done
docker build \
  --file docker/submission-qwen27b-mtp.Dockerfile \
  --tag "${image_tag}" \
  "${build_args[@]}" \
  .

run_args=(
  --rm
  --gpus all
  --env "AKBC_MODEL_KEY=${model_key}"
  --env "AKBC_CONFIG=${config}"
  --env "AKBC_EXPECTED_CANDIDATES=${expected_candidates}"
  --volume "${repo_root}/outputs:/workspace/run/outputs"
  --volume "${repo_root}/reports:/workspace/run/reports"
  --volume "${repo_root}/.cache/submission-dataset:/cache/dataset2026"
  --volume "${repo_root}/.cache/submission-huggingface:/cache/huggingface"
  --volume "${canonical_manifest}:/selection/selection.json:ro"
)
for variable_name in HF_TOKEN HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy; do
  if [[ -n "${!variable_name:-}" ]]; then
    run_args+=(--env "${variable_name}")
  fi
done
docker run "${run_args[@]}" "${image_tag}"

echo "submission ready: outputs/${model_key}-test.jsonl"
