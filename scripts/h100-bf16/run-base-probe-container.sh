#!/usr/bin/env bash
# Raw-completion probe for BASE (non-instruct) models: hasArea + hasCapacity
# val rows x 20 samples via few-shot completion prompts (no chat template).
set -Eeuo pipefail

cd /opt/akbc

model_key="${AKBC_MODEL_KEY:?set AKBC_MODEL_KEY}"
dataset_ref="30d8cfaaa7af5b236054cfb361f57b7d0c1e6e57"
gguf_repo="${AKBC_GGUF_REPO:?set AKBC_GGUF_REPO}"
gguf_revision="${AKBC_GGUF_REV:?set AKBC_GGUF_REV}"
gguf_part1="${AKBC_GGUF_PART1:?set AKBC_GGUF_PART1}"
dataset_dir="/cache/dataset2026/repository"
hf_home="/cache/huggingface"
run_dir="/workspace/run"
split="${AKBC_SPLIT:-val}"

mkdir -p "${run_dir}/outputs" "${run_dir}/reports"
bash scripts/ci/fetch-dataset.sh "${dataset_dir}" "${dataset_ref}"

python3 -m akbc_baseline.download_gguf \
  --repo "${gguf_repo}" \
  --filename "${gguf_part1}" \
  --revision "${gguf_revision}" \
  --cache-dir "${hf_home}" \
  --path-output "${run_dir}/reports/base-gguf-path.txt" \
  --minimum-gib "${AKBC_GGUF_MIN_GIB:-20}" \
  --maximum-gib 158
gguf_model_path="${hf_home}/models--${gguf_repo//\//--}/snapshots/${gguf_revision}/${gguf_part1}"
test -e "${gguf_model_path}"

export LD_LIBRARY_PATH="/app${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
CUDA_VISIBLE_DEVICES=0 /app/llama-server \
  --model "${gguf_model_path}" \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 16384 --parallel 16 \
  --n-gpu-layers 99 --flash-attn on \
  --batch-size 2048 --ubatch-size 512 \
  > "${run_dir}/reports/base-llama-server.log" 2>&1 &
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

PROMPTS = {
    "hasArea": (
        "The Wikipedia infobox lists the total area of Luxembourg as 2586.4 square kilometers.\n"
        "The Wikipedia infobox lists the total area of Skye as 1656 square kilometers.\n"
        "The Wikipedia infobox lists the total area of {s} as"
    ),
    "hasCapacity": (
        "The Wikipedia infobox lists the seating capacity of Camp Nou as 99354.\n"
        "The Wikipedia infobox lists the seating capacity of Mackay Stadium as 27000.\n"
        "The Wikipedia infobox lists the seating capacity of {s} as"
    ),
}

rows = [
    json.loads(line)
    for line in open(val_path, encoding="utf-8")
]
rows = [r for r in rows if r["Relation"] in PROMPTS]
assert len(rows) == 200, len(rows)

def sample(prompt, seed):
    body = json.dumps({
        "prompt": prompt,
        "n_predict": 24,
        "temperature": 0.6,
        "top_p": 0.95,
        "seed": seed,
        "stop": ["\n"],
    }).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:8080/completion",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())["content"]

def run_row(row, index):
    prompt = PROMPTS[row["Relation"]].format(s=row["SubjectEntity"])
    cands, diags = [], []
    for i in range(20):
        seed = int.from_bytes(
            hashlib.sha256(
                f"42|{row['SubjectEntity']}|{row['Relation']}|{i}".encode()
            ).digest()[:4],
            "big",
        )
        text = sample(prompt, seed)
        match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
        cands.append([match.group(0).replace(",", "")] if match else [])
        diags.append({"raw_text": text})
    return {
        "SubjectEntity": row["SubjectEntity"],
        "Relation": row["Relation"],
        "Candidates": cands,
        "CandidateDiagnostics": diags,
    }

results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
    futures = [pool.submit(run_row, row, i) for i, row in enumerate(rows)]
    for n, future in enumerate(concurrent.futures.as_completed(futures), 1):
        results.append(future.result())
        if n % 25 == 0:
            print(f"{n}/200 rows done", flush=True)

by_rel = {}
for result in results:
    by_rel.setdefault(result["Relation"], []).append(result)
for rel, rel_rows in by_rel.items():
    path = f"{out_prefix}-{rel}-candidates.jsonl"
    with open(path, "w", encoding="utf-8") as stream:
        for row in rel_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {path}: {len(rel_rows)} rows")
PY

echo "base-probe completed"
