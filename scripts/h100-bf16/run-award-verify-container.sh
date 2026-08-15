#!/usr/bin/env bash
# Award verification stage (#35): per-name yes/no probes over the distinct
# awardWonBy candidate names of an existing 20-candidate pool. Reuses the
# multi-GPU llama-server setup; the original header follows.
# all 100 hasArea validation
# rows x 20 candidates with the Wikidata/Wikipedia grounding instruction and
# raw thinking persisted. One llama-server per GPU (the BF16 27B fits on a
# single 80 GB device), AKBC_CLIENTS_PER_GPU shard clients per server.
set -Eeuo pipefail

cd /opt/akbc

model_key="${AKBC_MODEL_KEY:-qwen3.5-27b-bf16-award-verify}"
config="${AKBC_CONFIG:-configs/experiment-qwen-3.5-27b-bf16-thinking-empty-aware-20-area-grounding.yaml}"
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

pool_path="${AKBC_POOL_PATH:-${run_dir}/outputs/qwen3.5-27b-bf16-thinking-empty-aware-20-candidates-val.jsonl}"
test -e "${pool_path}" || {
  echo "missing candidates pool: ${pool_path}" >&2
  exit 1
}
ports="$(seq -s, 8080 $((8080 + num_gpus - 1)))"
python3 /opt/akbc/scripts/h100-bf16/award_verify.py   --candidates "${pool_path}"   --output "${run_dir}/outputs/${model_key}-votes.jsonl"   --votes "${AKBC_VERIFY_VOTES:-3}"   --ports "${ports}"
echo "award-verify completed: outputs/${model_key}-votes.jsonl"
