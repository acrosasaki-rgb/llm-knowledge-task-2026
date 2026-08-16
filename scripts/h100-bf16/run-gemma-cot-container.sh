#!/usr/bin/env bash
# gemma-pt "About-first" (CoT-style) registers for the non-city relations:
# the model first writes a one-line description of the subject, then the
# answer field. Mirrors the city CoT register that beat the flat register.
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
  --path-output "${run_dir}/reports/gcot-gguf-path.txt" \
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
  > "${run_dir}/reports/gcot-llama-server.log" 2>&1 &
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

REGISTERS = {
    "hasArea": (
        "Place: Andros, Greece\n"
        "About: Greek island in the Cyclades archipelago, the northernmost of the group.\n"
        "Total area: 381.398 square kilometers\n"
        "Place: Eritrea\n"
        "About: Country in the Horn of Africa on the Red Sea coast, independent since 1993.\n"
        "Total area: 117600 square kilometers\n"
        "Place: Molokai\n"
        "About: Hawaiian island between Oahu and Maui, known for its sea cliffs.\n"
        "Total area: 673.4 square kilometers\n"
        "Place: {s}\n"
        "About:"
    ),
    "hasCapacity": (
        "Venue: HoHoKam Stadium in Arizona\n"
        "About: Baseball park in Mesa, Arizona, long used for Chicago Cubs spring training.\n"
        "Seating capacity: 12623\n"
        "Venue: Changwon Stadium in Changwon\n"
        "About: Multi-purpose stadium in Changwon, South Korea, used mainly for baseball.\n"
        "Seating capacity: 27085\n"
        "Venue: Al-Jalaa Stadium in Damascus\n"
        "About: Football stadium in Damascus, Syria, home of several Syrian league clubs.\n"
        "Seating capacity: 10000\n"
        "Venue: {s}\n"
        "About:"
    ),
    "companyTradesAtStockExchange": (
        "Company: All Nippon Airways\n"
        "About: Japanese flag-carrier airline headquartered in Tokyo; listed since 1961.\n"
        "Exchanges: Tokyo Stock Exchange; London Stock Exchange\n"
        "Company: Knipex\n"
        "About: German family-owned pliers manufacturer in Wuppertal; not publicly traded.\n"
        "Exchanges: none\n"
        "Company: BNP Paribas\n"
        "About: French international banking group headquartered in Paris.\n"
        "Exchanges: Euronext Paris\n"
        "Company: {s}\n"
        "About:"
    ),
}
FIELD = {"hasArea": "Total area:", "hasCapacity": "Seating capacity:",
         "companyTradesAtStockExchange": "Exchanges:"}
STOPS = ["\nPlace:", "\nVenue:", "\nCompany:", "\n\n"]

rows = [json.loads(line) for line in open(rows_path, encoding="utf-8")]
rows = [r for r in rows if r["Relation"] in REGISTERS]
print(f"{len(rows)} rows", flush=True)


def sample(prompt, seed):
    body = json.dumps({
        "prompt": prompt, "n_predict": 96, "temperature": 0.6,
        "top_p": 0.95, "seed": seed, "stop": STOPS,
    }).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:8080/completion", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())["content"]


def parse(relation, text):
    m = re.search(re.escape(FIELD[relation]) + r"\s*([^\n]*)", text)
    if not m:
        return []
    v = m.group(1).strip()
    if relation in ("hasArea", "hasCapacity"):
        num = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", v)
        return [num.group(0).replace(",", "")] if num else []
    if not v or v.lower().startswith("none"):
        return []
    return [x.strip().rstrip(".") for x in v.split(";") if 0 < len(x.strip()) < 60]


def run_row(row):
    relation, subject = row["Relation"], row["SubjectEntity"]
    prompt = REGISTERS[relation].format(s=subject)
    cands, diags = [], []
    for i in range(20):
        seed = int.from_bytes(hashlib.sha256(
            f"{SEED}|{subject}|{relation}|{i}".encode()).digest()[:4], "big")
        text = sample(prompt, seed)
        cands.append(parse(relation, text))
        diags.append({"raw_text": text})
    return {"SubjectEntity": subject, "Relation": relation,
            "Candidates": cands, "CandidateDiagnostics": diags}


results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
    futures = [pool.submit(run_row, row) for row in rows]
    for n, future in enumerate(concurrent.futures.as_completed(futures), 1):
        results.append(future.result())
        if n % 25 == 0:
            print(f"{n}/{len(rows)} rows", flush=True)

path = f"{out_prefix}-cot-candidates.jsonl"
with open(path, "w", encoding="utf-8") as stream:
    for row in results:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"wrote {path}: {len(results)} rows")
print("gemma cot completed")
PY
