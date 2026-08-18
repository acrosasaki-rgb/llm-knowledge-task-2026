#!/usr/bin/env bash
# Container entrypoint for the submitted V19 system.
set -Eeuo pipefail

cd /opt/akbc

model_key="gemma-3-27b-pt-v19"
gguf_repo="bluerain123/gemma-3-27b-pt-Q8_0-GGUF"
gguf_revision="71ed905c894b1d481e67a3bdbdfe06dd5805c6e9"
gguf_file="gemma-3-27b-pt-q8_0.gguf"
source_input_file="${AKBC_DATA_FILE:?set AKBC_DATA_FILE}"
train_file="${AKBC_TRAIN_FILE:?set AKBC_TRAIN_FILE}"
val_file="${AKBC_VAL_FILE:?set AKBC_VAL_FILE}"
split="${AKBC_SPLIT:-test}"
run_dir="/workspace/run"
hf_home="/cache/huggingface"

case "${split}" in val|test) ;; *) echo "AKBC_SPLIT must be val or test" >&2; exit 2 ;; esac
for path in "${source_input_file}" "${train_file}" "${val_file}"; do
  test -f "${path}" || { echo "required dataset file is missing: ${path}" >&2; exit 2; }
done

mkdir -p "${run_dir}/outputs" "${run_dir}/reports"
input_file="${source_input_file}"
if [[ "${split}" == "test" ]]; then
  input_file="${run_dir}/reports/${model_key}-input-test.jsonl"
  python3 scripts/h100-bf16/prepare_v19_input.py \
    --input "${source_input_file}" \
    --output "${input_file}"
fi
python3 -m akbc_baseline.download_gguf \
  --repo "${gguf_repo}" \
  --filename "${gguf_file}" \
  --revision "${gguf_revision}" \
  --cache-dir "${hf_home}" \
  --path-output "${run_dir}/reports/${model_key}-gguf-path.txt" \
  --minimum-gib 26 --maximum-gib 30
model_path="${hf_home}/models--${gguf_repo//\//--}/snapshots/${gguf_revision}/${gguf_file}"
test -f "${model_path}"

export LD_LIBRARY_PATH="/app${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
CUDA_VISIBLE_DEVICES="${AKBC_GPU:-0}" /app/llama-server \
  --model "${model_path}" \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 16384 --parallel 16 \
  --n-gpu-layers 99 --flash-attn on \
  --batch-size 2048 --ubatch-size 512 \
  > "${run_dir}/reports/${model_key}-llama-server.log" 2>&1 &
server_pid=$!
trap 'kill "${server_pid}" 2>/dev/null || true' EXIT
for _ in $(seq 1 180); do
  curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1 && break
  sleep 5
done
curl -sf http://127.0.0.1:8080/health >/dev/null

candidates="${run_dir}/outputs/${model_key}-candidates-${split}.jsonl"
predictions="${run_dir}/outputs/${model_key}-${split}.jsonl"
python3 scripts/h100-bf16/v19_infer.py \
  --input "${input_file}" \
  --output "${candidates}" \
  --seed 42 \
  --workers 16

python3 scripts/h100-bf16/compose_gemma.py \
  --rows "${input_file}" \
  --pool "${candidates}" \
  --out "${predictions}" \
  --alias-graph "${train_file}" "${val_file}" \
  --city-min-votes 14 \
  --strict

python3 scripts/h100-bf16/verify_v19.py \
  --input "${input_file}" \
  --candidates "${candidates}" \
  --predictions "${predictions}" \
  --sha256-output "${run_dir}/reports/${model_key}-${split}.sha256"

python3 - "${run_dir}/reports/${model_key}-${split}-manifest.json" <<'PY'
import json
import os
import sys

manifest = {
    "system": "gemma-3-27b-pt-v19",
    "split": os.environ.get("AKBC_SPLIT", "test"),
    "model_repo": "bluerain123/gemma-3-27b-pt-Q8_0-GGUF",
    "model_revision": "71ed905c894b1d481e67a3bdbdfe06dd5805c6e9",
    "model_file": "gemma-3-27b-pt-q8_0.gguf",
    "code_commit": os.environ.get("AKBC_CODE_COMMIT", "unknown"),
    "candidate_count": 20,
    "temperature": 0.6,
    "top_p": 0.95,
    "seed": 42,
    "city_min_votes": 14,
}
with open(sys.argv[1], "w", encoding="utf-8") as stream:
    json.dump(manifest, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
PY

echo "V19 ${split} inference completed: ${predictions}"
