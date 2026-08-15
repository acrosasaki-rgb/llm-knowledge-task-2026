#!/usr/bin/env bash
# Cross-unit raw-completion probe: hasArea val rows sampled in square miles
# and hectares registers (20 samples each), for the algebraic-consistency
# certificate. Same server setup as run-base-probe-container.sh.
set -Eeuo pipefail

cd /opt/akbc

model_key="${AKBC_MODEL_KEY:?set AKBC_MODEL_KEY}"
dataset_ref="30d8cfaaa7af5b236054cfb361f57b7d0c1e6e57"
gguf_repo="${AKBC_GGUF_REPO:?}"
gguf_revision="${AKBC_GGUF_REV:?}"
gguf_part1="${AKBC_GGUF_PART1:?}"
dataset_dir="/cache/dataset2026/repository"
hf_home="/cache/huggingface"
run_dir="/workspace/run"
split="${AKBC_SPLIT:-val}"

mkdir -p "${run_dir}/outputs" "${run_dir}/reports"
bash scripts/ci/fetch-dataset.sh "${dataset_dir}" "${dataset_ref}"

python3 -m akbc_baseline.download_gguf \
  --repo "${gguf_repo}" --filename "${gguf_part1}" \
  --revision "${gguf_revision}" --cache-dir "${hf_home}" \
  --path-output "${run_dir}/reports/unit-gguf-path.txt" \
  --minimum-gib "${AKBC_GGUF_MIN_GIB:-20}" --maximum-gib 158
gguf_model_path="${hf_home}/models--${gguf_repo//\//--}/snapshots/${gguf_revision}/${gguf_part1}"
test -e "${gguf_model_path}"

export LD_LIBRARY_PATH="/app${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
CUDA_VISIBLE_DEVICES=0 /app/llama-server \
  --model "${gguf_model_path}" \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 16384 --parallel 16 \
  --n-gpu-layers 99 --flash-attn on \
  --batch-size 2048 --ubatch-size 512 \
  > "${run_dir}/reports/unit-llama-server.log" 2>&1 &
server_pid=$!
trap 'kill "${server_pid}" 2>/dev/null || true' EXIT
for i in $(seq 1 180); do
  curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1 && break
  sleep 5
done
curl -sf http://127.0.0.1:8080/health >/dev/null

python3 - "${dataset_dir}/data/${split}.jsonl" "${run_dir}/outputs/${model_key}" <<'PY'
import concurrent.futures
import hashlib
import json
import re
import sys
import urllib.request

val_path, out_prefix = sys.argv[1], sys.argv[2]

# Exemplar values are unit-consistent conversions of Luxembourg (2586.4 km2)
# and Skye (1656 km2).
REGISTERS = {
    "sqmi": (
        "The Wikipedia infobox lists the total area of Luxembourg as 998.6 square miles.\n"
        "The Wikipedia infobox lists the total area of Skye as 639.4 square miles.\n"
        "The Wikipedia infobox lists the total area of {s} as"
    ),
    "hectare": (
        "The Wikipedia infobox lists the total area of Luxembourg as 258,640 hectares.\n"
        "The Wikipedia infobox lists the total area of Skye as 165,600 hectares.\n"
        "The Wikipedia infobox lists the total area of {s} as"
    ),
}

rows = [json.loads(line) for line in open(val_path, encoding="utf-8")]
rows = [r for r in rows if r["Relation"] == "hasArea"]
assert len(rows) == 100, len(rows)

def sample(prompt, seed):
    body = json.dumps({
        "prompt": prompt, "n_predict": 24, "temperature": 0.6,
        "top_p": 0.95, "seed": seed, "stop": ["\n"],
    }).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:8080/completion", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())["content"]

def run_row(row, register, template):
    prompt = template.format(s=row["SubjectEntity"])
    cands, diags = [], []
    for i in range(20):
        seed = int.from_bytes(hashlib.sha256(
            f"42|{row['SubjectEntity']}|{register}|{i}".encode()
        ).digest()[:4], "big")
        text = sample(prompt, seed)
        match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
        cands.append([match.group(0).replace(",", "")] if match else [])
        diags.append({"raw_text": text})
    return {"SubjectEntity": row["SubjectEntity"], "Relation": "hasArea",
            "Candidates": cands, "CandidateDiagnostics": diags}

for register, template in REGISTERS.items():
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(run_row, row, register, template) for row in rows]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    path = f"{out_prefix}-{register}-candidates.jsonl"
    with open(path, "w", encoding="utf-8") as stream:
        for row in results:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {path}: {len(results)} rows", flush=True)
PY

echo "unit-probe completed"
