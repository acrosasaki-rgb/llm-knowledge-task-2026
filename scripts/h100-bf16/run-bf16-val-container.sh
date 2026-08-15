#!/usr/bin/env bash
# BF16 validation experiment for the 0.5662 thinking-empty-aware baseline.
#
# Runs Qwen3.5-27B BF16 (unquantized GGUF) through the unchanged prompt,
# sampling, and aggregation pipeline on a single H100 80GB. Serves llama.cpp
# with N parallel slots and fills them by launching N sequential
# akbc_baseline.run shard processes concurrently; candidate seeds are derived
# from (seed, subject, relation, candidate_index), so sharding does not change
# them. MTP is intentionally absent: batching and speculative decoding compete
# for the same idle compute, and the BF16 GGUF carries no MTP head.
#
# Modes (AKBC_RUN_MODE):
#   smoke - first 16 validation rows, 2 per slot, prints a full-run projection
#   full  - all validation rows in NP shards, merge, quality gate, evaluation
set -Eeuo pipefail

cd /opt/akbc

mode="${AKBC_RUN_MODE:?AKBC_RUN_MODE must be smoke or full}"
[[ "${mode}" == "smoke" || "${mode}" == "full" ]] || {
  echo "AKBC_RUN_MODE must be smoke or full, got: ${mode}" >&2
  exit 1
}

model_key="${AKBC_MODEL_KEY:-qwen3.5-27b-bf16-thinking-empty-aware}"
config="${AKBC_CONFIG:-configs/experiment-qwen-3.5-27b-bf16-thinking-empty-aware.yaml}"
expected_candidates="${AKBC_EXPECTED_CANDIDATES:-5}"
dataset_ref="30d8cfaaa7af5b236054cfb361f57b7d0c1e6e57"
gguf_repo="unsloth/Qwen3.5-27B-GGUF"
gguf_revision="3221f178a6b842d04f1fb42f1c413534adcc0a6a"
gguf_part1="BF16/Qwen3.5-27B-BF16-00001-of-00002.gguf"
gguf_part2="BF16/Qwen3.5-27B-BF16-00002-of-00002.gguf"
dataset_dir="/cache/dataset2026/repository"
hf_home="/cache/huggingface"
run_dir="/workspace/run"
llama_cpp_url="http://127.0.0.1:8080"

# Parallel slots and per-slot context. 10,240 tokens covers the worst
# awardWonBy prompt (~7k tokens with an unlucky few-shot draw) plus the
# 2,176-token generation budget. 8 x 10,240 KV in F16 plus ~50 GiB of BF16
# weights fits an 80 GB H100; on startup OOM reduce AKBC_PARALLEL before
# touching the per-slot context.
np="${AKBC_PARALLEL:-8}"
ctx_per_slot="${AKBC_CTX_PER_SLOT:-10240}"
ctx_total=$((np * ctx_per_slot))

mkdir -p "${run_dir}/outputs" "${run_dir}/reports"

bash scripts/ci/fetch-dataset.sh "${dataset_dir}" "${dataset_ref}"

python3 -m akbc_baseline.download_gguf \
  --repo "${gguf_repo}" \
  --filename "${gguf_part1}" \
  --revision "${gguf_revision}" \
  --cache-dir "${hf_home}" \
  --path-output "${run_dir}/reports/bf16-gguf-part1-path.txt" \
  --minimum-gib 40 \
  --maximum-gib 52
python3 -m akbc_baseline.download_gguf \
  --repo "${gguf_repo}" \
  --filename "${gguf_part2}" \
  --revision "${gguf_revision}" \
  --cache-dir "${hf_home}" \
  --path-output "${run_dir}/reports/bf16-gguf-part2-path.txt" \
  --minimum-gib 3 \
  --maximum-gib 5
# download_gguf resolves the snapshot symlink to its hash-named blob, which
# breaks llama.cpp's sibling discovery for split GGUFs (part 2 is found by
# filename pattern next to part 1). Point the server at the snapshot path,
# where both parts keep their canonical names.
snapshot_dir="${hf_home}/models--${gguf_repo//\//--}/snapshots/${gguf_revision}"
gguf_model_path="${snapshot_dir}/${gguf_part1}"
test -e "${gguf_model_path}" || {
  echo "snapshot path missing: ${gguf_model_path}" >&2
  exit 1
}
test -e "${snapshot_dir}/${gguf_part2}" || {
  echo "snapshot path missing: ${snapshot_dir}/${gguf_part2}" >&2
  exit 1
}

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
  --reasoning-budget 2048 \
  --reasoning-budget-message \
    "The reasoning budget is exhausted. Give the requested JSON array now using the best answer found so far." \
  --log-verbosity 4 \
  --metrics \
  --perf \
  > "${run_dir}/reports/llama-server.log" 2>&1 &
llama_server_pid=$!
trap 'kill "${llama_server_pid}" 2>/dev/null || true' EXIT

# No --require-mtp: this experiment runs without speculative decoding.
python3 -m akbc_baseline.llama_preflight \
  --url "${llama_cpp_url}" \
  --log "${run_dir}/reports/llama-server.log" \
  --server-pid "${llama_server_pid}" \
  --timeout-seconds 1800
nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader

total_rows="$(
  python3 -c \
    'import sys; from akbc_baseline.data import read_jsonl; print(len(read_jsonl(sys.argv[1])))' \
    "${dataset_dir}/data/val.jsonl"
)"

if [[ "${mode}" == "smoke" ]]; then
  run_rows=16
  (( run_rows <= total_rows )) || run_rows="${total_rows}"
else
  run_rows="${total_rows}"
fi
per_shard=$(( (run_rows + np - 1) / np ))

started_epoch="$(date +%s)"
declare -a shard_pids=()
declare -a shard_ids=()
for (( i = 0; i < np; i++ )); do
  offset=$(( i * per_shard ))
  (( offset < run_rows )) || break
  limit="${per_shard}"
  if (( offset + limit > run_rows )); then
    limit=$(( run_rows - offset ))
  fi
  shard_id="$(printf '%03d' "${i}")"
  python3 -m akbc_baseline.run \
    --config "${config}" \
    --dataset-dir "${dataset_dir}" \
    --input "${dataset_dir}/data/val.jsonl" \
    --output "${run_dir}/outputs/${model_key}-${mode}-shard-${shard_id}.jsonl" \
    --candidates-output "${run_dir}/outputs/${model_key}-candidates-${mode}-shard-${shard_id}.jsonl" \
    --metrics-output "${run_dir}/reports/${model_key}-${mode}-shard-${shard_id}-metrics.json" \
    --offset "${offset}" \
    --limit "${limit}" \
    --resume \
    > "${run_dir}/reports/${model_key}-${mode}-shard-${shard_id}.log" 2>&1 &
  shard_pids+=("$!")
  shard_ids+=("${shard_id}")
done

echo "launched ${#shard_pids[@]} shard workers (${per_shard} rows each, ${run_rows} rows total)"

failures=0
for index in "${!shard_pids[@]}"; do
  if ! wait "${shard_pids[${index}]}"; then
    echo "shard ${shard_ids[${index}]} failed;" \
      "see reports/${model_key}-${mode}-shard-${shard_ids[${index}]}.log" >&2
    failures=$((failures + 1))
  fi
done
(( failures == 0 )) || exit 1
elapsed=$(( "$(date +%s)" - started_epoch ))

cd "${run_dir}"

if [[ "${mode}" == "smoke" ]]; then
  # Shards cover disjoint ordered row ranges; concatenating them in shard
  # order restores the input order. merge_shards is reserved for the full run
  # because it validates against the complete validation split.
  cat outputs/"${model_key}-${mode}"-shard-*.jsonl \
    > "outputs/${model_key}-smoke-val.jsonl"
  cat outputs/"${model_key}-candidates-${mode}"-shard-*.jsonl \
    > "outputs/${model_key}-candidates-smoke-val.jsonl"
  python3 -m akbc_baseline.quality_gate \
    --predictions "outputs/${model_key}-smoke-val.jsonl" \
    --candidates "outputs/${model_key}-candidates-smoke-val.jsonl" \
    --report "reports/${model_key}-smoke-quality.json" \
    --expected-rows "${run_rows}" \
    --expected-candidates "${expected_candidates}" \
    --maximum-empty-prediction-rate 1 \
    --maximum-empty-candidate-rate 1
  python3 -m akbc_baseline.slice_jsonl \
    --input "${dataset_dir}/data/val.jsonl" \
    --output "outputs/val-head-${run_rows}.jsonl" \
    --offset 0 \
    --limit "${run_rows}"
  python3 -m akbc_baseline.compare \
    --evaluator "${dataset_dir}/evaluate.py" \
    --ground-truth "outputs/val-head-${run_rows}.jsonl" \
    --prediction "${model_key}=outputs/${model_key}-smoke-val.jsonl" \
    --json-output "reports/${model_key}-smoke-comparison.json" \
    --markdown-output "reports/${model_key}-smoke-comparison.md"
  python3 - "${elapsed}" "${run_rows}" "${total_rows}" <<'PROJECTION'
import sys
elapsed, run_rows, total_rows = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
projected = elapsed * total_rows / run_rows
print(f"smoke wall time: {elapsed} s for {run_rows} rows")
print(
    f"projected full validation ({total_rows} rows): "
    f"{projected / 3600:.2f} h at the smoke rate"
)
PROJECTION
  echo "smoke completed: review reports/${model_key}-smoke-comparison.md," \
    "reports/${model_key}-smoke-quality.json, and the projection above" \
    "before launching the full run"
else
  python3 -m akbc_baseline.merge_shards \
    --input "${dataset_dir}/data/val.jsonl" \
    --prediction-glob "outputs/${model_key}-${mode}-shard-*.jsonl" \
    --candidate-glob "outputs/${model_key}-candidates-${mode}-shard-*.jsonl" \
    --metrics-glob "reports/${model_key}-${mode}-shard-*-metrics.json" \
    --prediction-output "outputs/${model_key}-val.jsonl" \
    --candidate-output "outputs/${model_key}-candidates-val.jsonl" \
    --metrics-output "reports/${model_key}-val-metrics.json"
  python3 -m akbc_baseline.quality_gate \
    --predictions "outputs/${model_key}-val.jsonl" \
    --candidates "outputs/${model_key}-candidates-val.jsonl" \
    --report "reports/${model_key}-val-quality.json" \
    --expected-rows "${total_rows}" \
    --expected-candidates "${expected_candidates}" \
    --maximum-empty-prediction-rate 1 \
    --maximum-empty-candidate-rate 1
  python3 -m akbc_baseline.compare \
    --evaluator "${dataset_dir}/evaluate.py" \
    --ground-truth "${dataset_dir}/data/val.jsonl" \
    --prediction "${model_key}=outputs/${model_key}-val.jsonl" \
    --json-output "reports/${model_key}-val-comparison.json" \
    --markdown-output "reports/${model_key}-val-comparison.md"
  sha256sum "outputs/${model_key}-val.jsonl" \
    > "reports/${model_key}-val.jsonl.sha256"
  echo "full validation wall time: ${elapsed} s"
  echo "bf16 validation completed: outputs/${model_key}-val.jsonl"
fi
