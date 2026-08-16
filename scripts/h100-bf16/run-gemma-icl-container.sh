#!/usr/bin/env bash
# Gemma-pt train-gold fixed 4-shot ICL registers (city/company/capacity/area): all six relations in one session, raw
# completion registers, reading the row list from a mounted file so the new
# 475-row disambiguated split can be used.
set -Eeuo pipefail

cd /opt/akbc

model_key="${AKBC_MODEL_KEY:?}"
gguf_repo="${AKBC_GGUF_REPO:?}"
gguf_revision="${AKBC_GGUF_REV:?}"
gguf_part1="${AKBC_GGUF_PART1:?}"
input_file="${AKBC_DATA_FILE:?set AKBC_DATA_FILE (mounted rows file)}"
hf_home="/cache/huggingface"
run_dir="/workspace/run"

mkdir -p "${run_dir}/outputs" "${run_dir}/reports"
python3 -m akbc_baseline.download_gguf \
  --repo "${gguf_repo}" --filename "${gguf_part1}" \
  --revision "${gguf_revision}" --cache-dir "${hf_home}" \
  --path-output "${run_dir}/reports/gtest-gguf-path.txt" \
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
  > "${run_dir}/reports/gtest-llama-server.log" 2>&1 &
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
import os
import json
import re
import sys
import urllib.request

rows_path, out_prefix = sys.argv[1], sys.argv[2]
SEED = os.environ.get("AKBC_SEED", "42")

REGISTERS = {
    "hasArea": (
        "The Wikipedia infobox lists the total area of Andros, Greece as 381.398 square kilometers.\n"
        "The Wikipedia infobox lists the total area of Eritrea as 117600 square kilometers.\n"
        "The Wikipedia infobox lists the total area of Catanduanes as 1492.16 square kilometers.\n"
        "The Wikipedia infobox lists the total area of Molokaʻi as 673.4 square kilometers.\n"
        "The Wikipedia infobox lists the total area of {s} as"
    ),
    "hasCapacity": (
        "The Wikipedia infobox lists the seating capacity of HoHoKam Stadium in Arizona as 12623.\n"
        "The Wikipedia infobox lists the seating capacity of Changwon Stadium in Changwon as 27085.\n"
        "The Wikipedia infobox lists the seating capacity of Al-Jalaa Stadium in Damascus as 10000.\n"
        "The Wikipedia infobox lists the seating capacity of Christy Mathewson–Memorial Stadium in Pennsylvania as 13100.\n"
        "The Wikipedia infobox lists the seating capacity of {s} as"
    ),
    "personHasCityOfDeath": (
        "Name: Pavel Šrut\nOccupation: poet\nCity of death: Prague\n"
        "Name: George Akerlof\nOccupation: economist\nCity of death: NONE\n"
        "Name: Edward Soja\nOccupation: geographer\nCity of death: Los Angeles\n"
        "Name: Nobuyoshi Araki\nOccupation: photographer\nCity of death: NONE\n"
        "Name: {s}\nOccupation:"
    ),
    "companyTradesAtStockExchange": (
        "Company: All Nippon Airways\n"
        "Exchanges: Tokyo Stock Exchange; London Stock Exchange\n"
        "Company: Knipex\n"
        "Exchanges: none\n"
        "Company: BNP Paribas\n"
        "Exchanges: Euronext Paris\n"
        "Company: Tama Home\n"
        "Exchanges: Tokyo Stock Exchange; Fukuoka Stock Exchange\n"
        "Company: {s}\n"
        "Exchanges:"
    ),
}
NPRED = {"awardWonBy": 220}

rows = [json.loads(line) for line in open(rows_path, encoding="utf-8")]
rows = [r for r in rows if r["Relation"] in REGISTERS]
print(len(rows), "rows", flush=True)

def sample(prompt, seed, n_predict):
    body = json.dumps({
        "prompt": prompt, "n_predict": n_predict, "temperature": 0.6,
        "top_p": 0.95, "seed": seed,
        "stop": ["\nName:", "\nAward:", "\nCompany:", "\nCountry:", "\n\n", "\nThe Wikipedia"],
    }).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:8080/completion", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())["content"]

def parse(relation, text):
    text = text.strip()
    if relation in ("hasArea", "hasCapacity"):
        m = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
        return [m.group(0).replace(",", "")] if m else []
    if relation == "personHasCityOfDeath":
        if "alive" in text.lower() or re.search(r"City of death:\s*NONE", text):
            return []
        m = re.search(r"City of death:\s*([^\n,.(]+)", text)
        if m:
            city = m.group(1).strip()
            return [city] if city and "alive" not in city.lower() else []
        return []
    # list-valued relations
    first_line = text.split("\n")[0]
    if not first_line or first_line.lower().startswith("none"):
        return []
    names = [x.strip().rstrip(".") for x in first_line.split(";") if x.strip()]
    return [n for n in names if 0 < len(n) < 60]

def run_row(row):
    relation, subject = row["Relation"], row["SubjectEntity"]
    prompt = REGISTERS[relation].format(s=subject)
    n_predict = NPRED.get(relation, 32)
    cands, diags = [], []
    for i in range(20):
        seed = int.from_bytes(hashlib.sha256(
            f"{SEED}|{subject}|{relation}|{i}".encode()).digest()[:4], "big")
        text = sample(prompt, seed, n_predict)
        cands.append(parse(relation, text))
        diags.append({"raw_text": text})
    return {"SubjectEntity": subject, "Relation": relation,
            "Candidates": cands, "CandidateDiagnostics": diags}

results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
    futures = [pool.submit(run_row, row) for row in rows]
    for n, future in enumerate(concurrent.futures.as_completed(futures), 1):
        results.append(future.result())
        if n % 50 == 0:
            print(f"{n}/475 rows", flush=True)

path = f"{out_prefix}-icl-candidates.jsonl"
with open(path, "w", encoding="utf-8") as stream:
    for row in results:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"wrote {path}: {len(results)} rows")
print("gemma icl completed")
PY
