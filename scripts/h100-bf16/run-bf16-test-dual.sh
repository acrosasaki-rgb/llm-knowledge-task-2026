#!/usr/bin/env bash
# Dual-H100 test-split generation: one container per GPU, rows split in half,
# both halves of one budget variant run concurrently. Usage:
#   bash scripts/h100-bf16/run-bf16-test-dual.sh base    # budget 2048
#   bash scripts/h100-bf16/run-bf16-test-dual.sh think4096
# Weights must be pre-downloaded once into .cache/bf16-huggingface (see
# predownload in the same directory).
set -Eeuo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

variant="${1:?usage: run-bf16-test-dual.sh <base|think4096|val5c4096>}"
case "${variant}" in
  base)
    split=test
    total_rows=477
    model_key="qwen3.5-27b-bf16-thinking-empty-aware-20"
    config="configs/experiment-qwen-3.5-27b-bf16-thinking-empty-aware-20.yaml"
    budget=2048
    ctx_per_slot=10240
    ;;
  val5c4096)
    model_key="qwen3.5-27b-bf16-thinking-empty-aware-think4096"
    config="configs/experiment-qwen-3.5-27b-bf16-thinking-empty-aware-think4096.yaml"
    budget=4096
    ctx_per_slot=12288
    split=val
    total_rows=478
    ;;
  think4096)
    split=test
    total_rows=477
    model_key="qwen3.5-27b-bf16-thinking-empty-aware-20-think4096"
    config="configs/experiment-qwen-3.5-27b-bf16-thinking-empty-aware-20-think4096.yaml"
    budget=4096
    ctx_per_slot=12288
    ;;
  *) echo "unknown variant: ${variant}" >&2; exit 1 ;;
esac

split="${split:-test}"
total_rows="${total_rows:-477}"
half=$(( (total_rows + 1) / 2 ))   # 239 + 238
code_commit="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
image_tag="akbc-qwen27b-bf16-test:${code_commit:0:12}"

docker build \
  --file docker/h100-bf16-val.Dockerfile \
  --tag "${image_tag}" \
  --build-arg "AKBC_CODE_COMMIT=${code_commit}" \
  .

mkdir -p outputs reports .cache/bf16-dataset .cache/bf16-huggingface

run_one() {
  local gpu="$1" row_offset="$2" row_count="$3" shard_base="$4"
  docker run --rm \
    --gpus "\"device=${gpu}\"" \
    --entrypoint bash \
    --env "AKBC_MODEL_KEY=${model_key}" \
    --env "AKBC_CONFIG=${config}" \
    --env "AKBC_ROW_OFFSET=${row_offset}" \
    --env "AKBC_ROW_COUNT=${row_count}" \
    --env "AKBC_SHARD_BASE=${shard_base}" \
    --env "AKBC_REASONING_BUDGET=${budget}" \
    --env "AKBC_SPLIT=${split}" \
    --env "AKBC_CTX_PER_SLOT=${ctx_per_slot}" \
    --volume "${repo_root}/outputs:/workspace/run/outputs" \
    --volume "${repo_root}/reports:/workspace/run/reports" \
    --volume "${repo_root}/.cache/bf16-dataset:/cache/dataset2026" \
    --volume "${repo_root}/.cache/bf16-huggingface:/cache/huggingface" \
    --volume "${repo_root}/scripts/h100-bf16/run-bf16-test-container.sh:/opt/akbc/scripts/h100-bf16/run-bf16-test-container.sh:ro" \
    "${image_tag}" \
    /opt/akbc/scripts/h100-bf16/run-bf16-test-container.sh \
    > "reports/run-${variant}-gpu${gpu}.log" 2>&1
}

run_one 0 0 "${half}" 0 &
pid0=$!
run_one 1 "${half}" $(( total_rows - half )) 100 &
pid1=$!

status=0
wait "${pid0}" || { echo "gpu0 container failed" >&2; status=1; }
wait "${pid1}" || { echo "gpu1 container failed" >&2; status=1; }
exit "${status}"
