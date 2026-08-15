#!/usr/bin/env bash
# Answer-blind factual priming for hasCapacity: three fixed contexts that
# never mention capacity are generated first (greedy), then capacity is
# completed conditioned on each context (20 samples). The certificate is
# computed offline: switch only if all three routes land on the same
# existing minority cluster.
set -Eeuo pipefail

cd /opt/akbc

model_key="${AKBC_MODEL_KEY:?}"
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
  --path-output "${run_dir}/reports/capprime-gguf-path.txt" \
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
  > "${run_dir}/reports/capprime-llama-server.log" 2>&1 &
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

CONTEXTS = {
    "loc": "Venue: {s}\nLocation:",
    "hist": "Venue: {s}\nOpened:",
    "type": "Venue: {s}\nVenue type:",
}
FOLLOWUPS = {
    "loc": "\nHome team:",
    "hist": "\nMost recent renovation:",
    "type": "\nPrimary use:",
}

rows = [json.loads(line) for line in open(val_path, encoding="utf-8")]
rows = [r for r in rows if r["Relation"] == "hasCapacity"]
assert len(rows) in (98, 100), len(rows)

def complete(prompt, seed, n_predict, temperature):
    body = json.dumps({
        "prompt": prompt, "n_predict": n_predict,
        "temperature": temperature, "top_p": 0.95, "seed": seed,
        "stop": ["\n\n"],
    }).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:8080/completion", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())["content"]

def run_row(row):
    subject = row["SubjectEntity"]
    out = {"SubjectEntity": subject, "Relation": "hasCapacity", "routes": {}}
    for name, template in CONTEXTS.items():
        prefix = template.format(s=subject)
        ctx = complete(prefix, 7, 48, 0.2)
        ctx2 = complete(prefix + ctx + FOLLOWUPS[name], 7, 32, 0.2)
        primed = prefix + ctx + FOLLOWUPS[name] + ctx2 + "\nCapacity:"
        cands = []
        for i in range(20):
            seed = int.from_bytes(hashlib.sha256(
                f"42|{subject}|{name}|{i}".encode()).digest()[:4], "big")
            text = complete(primed, seed, 16, 0.6)
            match = re.search(r"\d[\d,]*", text)
            cands.append(match.group(0).replace(",", "") if match else None)
        out["routes"][name] = {"context": primed, "candidates": cands}
    return out

results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
    futures = [pool.submit(run_row, row) for row in rows]
    for n, future in enumerate(concurrent.futures.as_completed(futures), 1):
        results.append(future.result())
        if n % 25 == 0:
            print(f"{n}/{len(rows)} rows", flush=True)

path = f"{out_prefix}-capprime.jsonl"
with open(path, "w", encoding="utf-8") as stream:
    for row in results:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"wrote {path}: {len(results)} rows")
print("capprime completed")
PY
