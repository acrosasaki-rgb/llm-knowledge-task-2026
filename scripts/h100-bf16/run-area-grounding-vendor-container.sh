#!/usr/bin/env bash
# Grounding-prompt experiment on a multi-GPU host: all 100 hasArea validation
# rows x 20 candidates with the Wikidata/Wikipedia grounding instruction and
# raw thinking persisted. One llama-server per GPU (the BF16 27B fits on a
# single 80 GB device), AKBC_CLIENTS_PER_GPU shard clients per server.
set -Eeuo pipefail

cd /opt/akbc

model_key="${AKBC_MODEL_KEY:-qwen3.5-27b-bf16-area-grounding-vendor}"
config="${AKBC_CONFIG:-configs/experiment-qwen-3.5-27b-bf16-thinking-empty-aware-20-area-grounding-vendor.yaml}"
dataset_ref="30d8cfaaa7af5b236054cfb361f57b7d0c1e6e57"
gguf_repo="unsloth/Qwen3.5-27B-GGUF"
gguf_revision="3221f178a6b842d04f1fb42f1c413534adcc0a6a"
gguf_part1="BF16/Qwen3.5-27B-BF16-00001-of-00002.gguf"
gguf_part2="BF16/Qwen3.5-27B-BF16-00002-of-00002.gguf"
dataset_dir="/cache/dataset2026/repository"
hf_home="/cache/huggingface"
run_dir="/workspace/run"

num_gpus="$(nvidia-smi -L | wc -l)"
(( num_gpus >= 1 )) || { echo "no GPUs visible" >&2; exit 1; }
clients_per_gpu="${AKBC_CLIENTS_PER_GPU:-8}"
slots_per_server="${AKBC_PARALLEL:-8}"
ctx_per_slot="${AKBC_CTX_PER_SLOT:-10240}"
ctx_total=$((slots_per_server * ctx_per_slot))

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

input_path="${run_dir}/outputs/${model_key}-input.jsonl"
python3 - "${dataset_dir}/data/val.jsonl" "${input_path}" <<'PY'
import json, sys

val_path, output_path = sys.argv[1:3]
selected = []
with open(val_path, encoding="utf-8") as stream:
    for line in stream:
        row = json.loads(line)
        if row["Relation"] == "hasArea":
            selected.append(row)
if len(selected) != 100:
    raise SystemExit(f"expected 100 hasArea rows, found {len(selected)}")
with open(output_path, "w", encoding="utf-8") as stream:
    for row in selected:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"selected {len(selected)} hasArea rows")
PY

# One llama-server per GPU on port 8080+gpu, plus a per-port config copy so
# shard clients can address their server (llama_cpp_url lives in the config).
export LD_LIBRARY_PATH="/app${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
declare -a server_pids=()
for (( gpu = 0; gpu < num_gpus; gpu++ )); do
  port=$((8080 + gpu))
  CUDA_VISIBLE_DEVICES="${gpu}" /app/llama-server \
    --model "${gguf_model_path}" \
    --alias "${gguf_repo}" \
    --host 127.0.0.1 \
    --port "${port}" \
    --ctx-size "${ctx_total}" \
    --parallel "${slots_per_server}" \
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
    > "${run_dir}/reports/llama-server-gpu${gpu}.log" 2>&1 &
  server_pids+=("$!")
  sed "s|http://127.0.0.1:8080|http://127.0.0.1:${port}|" \
    "${config}" > "/tmp/config-port-${port}.yaml"
done
trap 'kill "${server_pids[@]}" 2>/dev/null || true' EXIT

for (( gpu = 0; gpu < num_gpus; gpu++ )); do
  python3 -m akbc_baseline.llama_preflight \
    --url "http://127.0.0.1:$((8080 + gpu))" \
    --log "${run_dir}/reports/llama-server-gpu${gpu}.log" \
    --server-pid "${server_pids[${gpu}]}" \
    --timeout-seconds 1800
done
nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader

run_rows="$(
  python3 -c \
    'import sys; from akbc_baseline.data import read_jsonl; print(len(read_jsonl(sys.argv[1])))' \
    "${input_path}"
)"
total_workers=$((num_gpus * clients_per_gpu))
per_shard=$(( (run_rows + total_workers - 1) / total_workers ))

started_epoch="$(date +%s)"
declare -a shard_pids=()
declare -a shard_ids=()
shard=0
for (( i = 0; i < total_workers; i++ )); do
  offset=$(( i * per_shard ))
  (( offset < run_rows )) || break
  limit="${per_shard}"
  if (( offset + limit > run_rows )); then
    limit=$(( run_rows - offset ))
  fi
  port=$((8080 + shard % num_gpus))
  shard_id="$(printf '%03d' "${i}")"
  python3 -m akbc_baseline.run \
    --config "/tmp/config-port-${port}.yaml" \
    --dataset-dir "${dataset_dir}" \
    --input "${input_path}" \
    --output "${run_dir}/outputs/${model_key}-shard-${shard_id}.jsonl" \
    --candidates-output "${run_dir}/outputs/${model_key}-candidates-shard-${shard_id}.jsonl" \
    --metrics-output "${run_dir}/reports/${model_key}-shard-${shard_id}-metrics.json" \
    --offset "${offset}" \
    --limit "${limit}" \
    --resume \
    > "${run_dir}/reports/${model_key}-shard-${shard_id}.log" 2>&1 &
  shard_pids+=("$!")
  shard_ids+=("${shard_id}")
  shard=$((shard + 1))
done

echo "launched ${#shard_pids[@]} shard workers across ${num_gpus} servers" \
  "(${per_shard} rows per shard, ${run_rows} rows total)"

failures=0
for index in "${!shard_pids[@]}"; do
  if ! wait "${shard_pids[${index}]}"; then
    echo "shard ${shard_ids[${index}]} failed;" \
      "see reports/${model_key}-shard-${shard_ids[${index}]}.log" >&2
    failures=$((failures + 1))
  fi
done
(( failures == 0 )) || exit 1
elapsed=$(( "$(date +%s)" - started_epoch ))

cd "${run_dir}"
cat outputs/"${model_key}"-candidates-shard-*.jsonl \
  > "outputs/${model_key}-candidates.jsonl"
cat outputs/"${model_key}"-shard-*.jsonl \
  > "outputs/${model_key}-predictions.jsonl"

python3 - "outputs/${model_key}-candidates.jsonl" "${run_rows}" <<'PY'
import json, sys

path, expected_rows = sys.argv[1], int(sys.argv[2])
rows = [json.loads(line) for line in open(path, encoding="utf-8")]
assert len(rows) == expected_rows, (len(rows), expected_rows)
for row in rows:
    assert len(row["Candidates"]) == 20, row["SubjectEntity"]
    missing = [
        index
        for index, diag in enumerate(row["CandidateDiagnostics"])
        if not diag.get("raw_text")
    ]
    assert not missing, (row["SubjectEntity"], missing)
print(f"sanity ok: {len(rows)} rows, 20 candidates each, raw_text present")
PY

echo "area-grounding-vendor wall time: ${elapsed} s"
echo "area-grounding-vendor completed: outputs/${model_key}-candidates.jsonl"
