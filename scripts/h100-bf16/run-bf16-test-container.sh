#!/usr/bin/env bash
# BF16 test-split candidate generation for one GPU (Issue: dual-H100 test
# generation of the best 20-candidate thinking configuration at two thinking
# budgets). Generation only: no evaluator (test gold is a placeholder), no
# selection manifest (exploratory candidates, not the submission flow).
# The container sees exactly one GPU; each container serves its own
# llama-server on 127.0.0.1 and processes rows [AKBC_ROW_OFFSET,
# AKBC_ROW_OFFSET + AKBC_ROW_COUNT) of data/test.jsonl in AKBC_PARALLEL
# client shards named by global shard index for a later ordered merge.
set -Eeuo pipefail

cd /opt/akbc

model_key="${AKBC_MODEL_KEY:?}"
config="${AKBC_CONFIG:?}"
row_offset="${AKBC_ROW_OFFSET:?}"
row_count="${AKBC_ROW_COUNT:?}"
shard_base="${AKBC_SHARD_BASE:?}"
reasoning_budget="${AKBC_REASONING_BUDGET:-2048}"
split="${AKBC_SPLIT:-test}"
np="${AKBC_PARALLEL:-8}"
ctx_per_slot="${AKBC_CTX_PER_SLOT:-12288}"
ctx_total=$((np * ctx_per_slot))
dataset_ref="30d8cfaaa7af5b236054cfb361f57b7d0c1e6e57"
gguf_repo="unsloth/Qwen3.5-27B-GGUF"
gguf_revision="3221f178a6b842d04f1fb42f1c413534adcc0a6a"
gguf_part1="BF16/Qwen3.5-27B-BF16-00001-of-00002.gguf"
gguf_part2="BF16/Qwen3.5-27B-BF16-00002-of-00002.gguf"
dataset_dir="/cache/dataset2026/repository"
hf_home="/cache/huggingface"
run_dir="/workspace/run"
llama_cpp_url="http://127.0.0.1:8080"

mkdir -p "${run_dir}/outputs" "${run_dir}/reports"
bash scripts/ci/fetch-dataset.sh "${dataset_dir}" "${dataset_ref}"

snapshot_dir="${hf_home}/models--${gguf_repo//\//--}/snapshots/${gguf_revision}"
gguf_model_path="${snapshot_dir}/${gguf_part1}"
test -e "${gguf_model_path}" || {
  echo "BF16 weights are not pre-downloaded at ${gguf_model_path}" >&2
  exit 1
}
test -e "${snapshot_dir}/${gguf_part2}"

export LD_LIBRARY_PATH="/app${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
/app/llama-server \
  --model "${gguf_model_path}" \
  --alias "${gguf_repo}" \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx-size "${ctx_total}" \
  --parallel "${np}" \
  --n-gpu-layers 99 \
  --fit off \
  --flash-attn on \
  --batch-size 2048 \
  --ubatch-size 512 \
  --jinja \
  --reasoning on \
  --reasoning-format none \
  --reasoning-budget "${reasoning_budget}" \
  --reasoning-budget-message \
    "The reasoning budget is exhausted. Give the requested JSON array now using the best answer found so far." \
  --log-verbosity 4 \
  --metrics \
  --perf \
  > "${run_dir}/reports/llama-server-${model_key}-${shard_base}.log" 2>&1 &
llama_server_pid=$!
trap 'kill "${llama_server_pid}" 2>/dev/null || true' EXIT

python3 -m akbc_baseline.llama_preflight \
  --url "${llama_cpp_url}" \
  --log "${run_dir}/reports/llama-server-${model_key}-${shard_base}.log" \
  --server-pid "${llama_server_pid}" \
  --timeout-seconds 1800
nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader

per_shard=$(( (row_count + np - 1) / np ))
declare -a pids=() ids=()
for (( i = 0; i < np; i++ )); do
  offset=$(( row_offset + i * per_shard ))
  local_left=$(( row_offset + row_count - offset ))
  (( local_left > 0 )) || break
  limit="${per_shard}"
  (( limit <= local_left )) || limit="${local_left}"
  shard_id="$(printf '%03d' $(( shard_base + i )))"
  python3 -m akbc_baseline.run \
    --config "${config}" \
    --dataset-dir "${dataset_dir}" \
    --input "${dataset_dir}/data/${split}.jsonl" \
    --output "${run_dir}/outputs/${model_key}-${split}-shard-${shard_id}.jsonl" \
    --candidates-output "${run_dir}/outputs/${model_key}-candidates-${split}-shard-${shard_id}.jsonl" \
    --metrics-output "${run_dir}/reports/${model_key}-${split}-shard-${shard_id}-metrics.json" \
    --offset "${offset}" \
    --limit "${limit}" \
    --resume \
    > "${run_dir}/reports/${model_key}-${split}-shard-${shard_id}.log" 2>&1 &
  pids+=("$!")
  ids+=("${shard_id}")
done
echo "launched ${#pids[@]} shard workers (rows ${row_offset}..$((row_offset + row_count - 1)), budget ${reasoning_budget})"

failures=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[${index}]}"; then
    echo "shard ${ids[${index}]} failed" >&2
    failures=$((failures + 1))
  fi
done
(( failures == 0 )) || exit 1
echo "test generation container done: ${model_key} rows ${row_offset}+${row_count}"
