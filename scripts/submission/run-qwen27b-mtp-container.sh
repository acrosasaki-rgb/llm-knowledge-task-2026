#!/usr/bin/env bash
set -Eeuo pipefail

cd /opt/akbc

model_key="${AKBC_MODEL_KEY:-qwen3.5-27b-mtp-thinking}"
config="${AKBC_CONFIG:-configs/experiment-qwen-3.5-27b-mtp-thinking.yaml}"
expected_candidates="${AKBC_EXPECTED_CANDIDATES:-5}"
dataset_ref="30d8cfaaa7af5b236054cfb361f57b7d0c1e6e57"
container_image="ghcr.io/ggml-org/llama.cpp:full-cuda@sha256:11b0e950e081777cf326598bb2eff2ab0815f02405bf95c6650b34027750114e"
gguf_repo="unsloth/Qwen3.5-27B-MTP-GGUF"
gguf_filename="Qwen3.5-27B-Q4_K_M.gguf"
gguf_revision="88fb5663d646bc78e1140648e8d8cb7d3e849908"
dataset_dir="/cache/dataset2026/repository"
hf_home="/cache/huggingface"
run_dir="/workspace/run"
manifest="/selection/selection.json"
llama_cpp_url="http://127.0.0.1:8080"

test -n "${AKBC_CODE_COMMIT:-}" || {
  echo "AKBC_CODE_COMMIT is required" >&2
  exit 1
}
test -f "${manifest}"
mkdir -p "${run_dir}/outputs" "${run_dir}/reports"

python3 -m akbc_baseline.selection_verify \
  --manifest "${manifest}" \
  --config "${config}" \
  --model-key "${model_key}" \
  --dataset-ref "${dataset_ref}" \
  --commit-sha "${AKBC_CODE_COMMIT}" \
  --container-image "${container_image}" \
  --expected-candidates "${expected_candidates}"

bash scripts/ci/fetch-dataset.sh "${dataset_dir}" "${dataset_ref}"
python3 -m akbc_baseline.download_gguf \
  --repo "${gguf_repo}" \
  --filename "${gguf_filename}" \
  --revision "${gguf_revision}" \
  --cache-dir "${hf_home}" \
  --path-output "${run_dir}/reports/gguf-model-path.txt" \
  --minimum-gib 15 \
  --maximum-gib 18
gguf_model_path="$(cat "${run_dir}/reports/gguf-model-path.txt")"

export LD_LIBRARY_PATH="/app${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
/app/llama-server \
  --model "${gguf_model_path}" \
  --alias "${gguf_repo}" \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx-size 8192 \
  --parallel 1 \
  --n-gpu-layers 99 \
  --fit off \
  --flash-attn on \
  --batch-size 512 \
  --ubatch-size 128 \
  --jinja \
  --reasoning on \
  --reasoning-format none \
  --reasoning-budget 2048 \
  --reasoning-budget-message \
    "The reasoning budget is exhausted. Give the requested JSON array now using the best answer found so far." \
  --spec-type draft-mtp \
  --spec-draft-n-max 6 \
  --log-verbosity 4 \
  --metrics \
  --perf \
  > "${run_dir}/reports/llama-server.log" 2>&1 &
llama_server_pid=$!
trap 'kill "${llama_server_pid}" 2>/dev/null || true' EXIT

python3 -m akbc_baseline.llama_preflight \
  --url "${llama_cpp_url}" \
  --log "${run_dir}/reports/llama-server.log" \
  --server-pid "${llama_server_pid}" \
  --timeout-seconds 900 \
  --require-mtp
nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader

expected_rows="$(
  python3 -c \
    'import sys; from akbc_baseline.data import read_jsonl; print(len(read_jsonl(sys.argv[1])))' \
    "${dataset_dir}/data/test.jsonl"
)"
offset=0
shard_index=0
while (( offset < expected_rows )); do
  shard_id="$(printf '%03d' "${shard_index}")"
  python3 -m akbc_baseline.run \
    --config "${config}" \
    --dataset-dir "${dataset_dir}" \
    --input "${dataset_dir}/data/test.jsonl" \
    --output "${run_dir}/outputs/${model_key}-test-shard-${shard_id}.jsonl" \
    --candidates-output "${run_dir}/outputs/${model_key}-candidates-test-shard-${shard_id}.jsonl" \
    --metrics-output "${run_dir}/reports/${model_key}-test-shard-${shard_id}-metrics.json" \
    --offset "${offset}" \
    --limit 50 \
    --resume
  offset=$((offset + 50))
  shard_index=$((shard_index + 1))
done

cd "${run_dir}"
python3 -m akbc_baseline.merge_shards \
  --input "${dataset_dir}/data/test.jsonl" \
  --prediction-glob "outputs/${model_key}-test-shard-*.jsonl" \
  --candidate-glob "outputs/${model_key}-candidates-test-shard-*.jsonl" \
  --metrics-glob "reports/${model_key}-test-shard-*-metrics.json" \
  --prediction-output "outputs/${model_key}-test.jsonl" \
  --candidate-output "outputs/${model_key}-candidates-test.jsonl" \
  --metrics-output "reports/${model_key}-test-metrics.json"
python3 -m akbc_baseline.quality_gate \
  --predictions "outputs/${model_key}-test.jsonl" \
  --candidates "outputs/${model_key}-candidates-test.jsonl" \
  --report "reports/${model_key}-test-quality.json" \
  --expected-rows "${expected_rows}" \
  --expected-candidates "${expected_candidates}" \
  --maximum-empty-prediction-rate 1 \
  --maximum-empty-candidate-rate 1
cd /opt/akbc
python3 -m akbc_baseline.selection_verify \
  --manifest "${manifest}" \
  --config "${config}" \
  --model-key "${model_key}" \
  --dataset-ref "${dataset_ref}" \
  --commit-sha "${AKBC_CODE_COMMIT}" \
  --container-image "${container_image}" \
  --expected-candidates "${expected_candidates}"
cd "${run_dir}"
sha256sum "outputs/${model_key}-test.jsonl" \
  > "reports/${model_key}-test.jsonl.sha256"
echo "external submission completed: outputs/${model_key}-test.jsonl"
