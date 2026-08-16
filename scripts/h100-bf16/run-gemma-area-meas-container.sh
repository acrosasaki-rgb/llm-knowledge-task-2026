#!/usr/bin/env bash
# hasArea "measurement-first": let the model reproduce a remembered natural
# measurement sentence (metric or imperial), then parse the number offline.
# Two routes: km2 phrasing and sq-mi phrasing (converted deterministically).
set -Eeuo pipefail

cd /opt/akbc

model_key="${AKBC_MODEL_KEY:?}"
gguf_repo="${AKBC_GGUF_REPO:?}"
gguf_revision="${AKBC_GGUF_REV:?}"
gguf_part1="${AKBC_GGUF_PART1:?}"
input_file="${AKBC_DATA_FILE:?}"
hf_home="/cache/huggingface"
run_dir="/workspace/run"

mkdir -p "${run_dir}/outputs" "${run_dir}/reports"
python3 -m akbc_baseline.download_gguf \
  --repo "${gguf_repo}" --filename "${gguf_part1}" \
  --revision "${gguf_revision}" --cache-dir "${hf_home}" \
  --path-output "${run_dir}/reports/gmeas-gguf-path.txt" \
  --minimum-gib "${AKBC_GGUF_MIN_GIB:-20}" --maximum-gib 158
gguf_model_path="${hf_home}/models--${gguf_repo//\//--}/snapshots/${gguf_revision}/${gguf_part1}"
test -e "${gguf_model_path}"

export LD_LIBRARY_PATH="/app${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
CUDA_VISIBLE_DEVICES="${AKBC_GPU:-0}" /app/llama-server \
  --model "${gguf_model_path}" \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 16384 --parallel 16 \
  --n-gpu-layers 99 --flash-attn on \
  --batch-size 2048 --ubatch-size 512 \
  > "${run_dir}/reports/gmeas-llama-server.log" 2>&1 &
server_pid=$!
trap 'kill "${server_pid}" 2>/dev/null || true' EXIT
for i in $(seq 1 180); do
  curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1 && break
  sleep 5
done
curl -sf http://127.0.0.1:8080/health >/dev/null

python3 - "${input_file}" "${run_dir}/outputs/${model_key}" <<'PY'
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import urllib.request

rows_path, out_prefix = sys.argv[1], sys.argv[2]
SEED = os.environ.get("AKBC_SEED", "42")
ROUTE = os.environ.get("AKBC_AREA_ROUTE", "km")

REGS = {
    # remembered metric measurement sentence
    "km": (
        "Andros, Greece has a total area of approximately 381.4 square kilometres.\n"
        "Eritrea has a total area of approximately 117,600 square kilometres.\n"
        "Molokai has a total area of approximately 673.4 square kilometres.\n"
        "{s} has a total area of approximately"
    ),
    # remembered imperial measurement sentence (converted offline)
    "mi": (
        "Andros, Greece covers approximately 147.3 square miles.\n"
        "Eritrea covers approximately 45,406 square miles.\n"
        "Molokai covers approximately 260.0 square miles.\n"
        "{s} covers approximately"
    ),
}

rows = [json.loads(line) for line in open(rows_path, encoding="utf-8")]
rows = [r for r in rows if r["Relation"] == "hasArea"]
print(f"{len(rows)} rows route={ROUTE}", flush=True)


def sample(prompt, seed):
    body = json.dumps({
        "prompt": prompt, "n_predict": 24, "temperature": 0.6,
        "top_p": 0.95, "seed": seed, "stop": ["\n"],
    }).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:8080/completion", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())["content"]


def parse(text):
    m = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if not m:
        return []
    v = float(m.group(0).replace(",", ""))
    if ROUTE == "mi":
        v = v * 2.589988110336
    return [f"{v:g}"]


def run_row(row):
    subject = row["SubjectEntity"]
    prompt = REGS[ROUTE].format(s=subject)
    cands, diags = [], []
    for i in range(20):
        seed = int.from_bytes(hashlib.sha256(
            f"{SEED}|{subject}|area-{ROUTE}|{i}".encode()).digest()[:4], "big")
        text = sample(prompt, seed)
        cands.append(parse(text))
        diags.append({"raw_text": text})
    return {"SubjectEntity": subject, "Relation": "hasArea",
            "Candidates": cands, "CandidateDiagnostics": diags}


results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
    futures = [pool.submit(run_row, row) for row in rows]
    for n, future in enumerate(concurrent.futures.as_completed(futures), 1):
        results.append(future.result())
        if n % 25 == 0:
            print(f"{n}/{len(rows)} rows", flush=True)

path = f"{out_prefix}-meas-candidates.jsonl"
with open(path, "w", encoding="utf-8") as stream:
    for row in results:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"wrote {path}: {len(results)} rows")
print("gemma meas completed")
PY
