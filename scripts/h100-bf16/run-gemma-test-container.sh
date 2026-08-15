#!/usr/bin/env bash
# Gemma-pt full test-set generation: all six relations in one session, raw
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
CUDA_VISIBLE_DEVICES=0 /app/llama-server \
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
import json
import re
import sys
import urllib.request

rows_path, out_prefix = sys.argv[1], sys.argv[2]

REGISTERS = {
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
    "personHasCityOfDeath": (
        "Name: James Gandolfini\nOccupation: actor\nCity of death: Rome\n"
        "Name: Ada Lovelace\nOccupation: mathematician\nCity of death: London\n"
        "Name: Paul McCartney\nOccupation: musician\nCity of death: (still alive)\n"
        "Name: {s}\nOccupation:"
    ),
    "countryLandBordersCountry": (
        "Country: Portugal\nLand borders: Spain\n"
        "Country: Japan\nLand borders: none\n"
        "Country: Austria\nLand borders: Germany; Czech Republic; Slovakia; "
        "Hungary; Slovenia; Italy; Switzerland; Liechtenstein\n"
        "Country: {s}\nLand borders:"
    ),
    "companyTradesAtStockExchange": (
        "Company: Sony Group Corporation\n"
        "Exchanges: Tokyo Stock Exchange; New York Stock Exchange\n"
        "Company: Robert Bosch GmbH\n"
        "Exchanges: none (privately held)\n"
        "Company: Nike, Inc.\n"
        "Exchanges: New York Stock Exchange\n"
        "Company: {s}\n"
        "Exchanges:"
    ),
    "awardWonBy": (
        "Award: Hugo Award for Best Novel\n"
        "Recipients: Isaac Asimov; Frank Herbert; Ursula K. Le Guin; "
        "Arthur C. Clarke; Kim Stanley Robinson; N. K. Jemisin\n"
        "Award: {s}\n"
        "Recipients:"
    ),
}
NPRED = {"awardWonBy": 220}

rows = [json.loads(line) for line in open(rows_path, encoding="utf-8")]
assert len(rows) == 475, len(rows)

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
        if "alive" in text.lower():
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
            f"42|{subject}|{relation}|{i}".encode()).digest()[:4], "big")
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

path = f"{out_prefix}-test-all-candidates.jsonl"
with open(path, "w", encoding="utf-8") as stream:
    for row in results:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"wrote {path}: {len(results)} rows")
print("gemma test generation completed")
PY
