#!/usr/bin/env bash
# Small cross-family Base model ("aux") probe for the city / company
# certificates: four raw-completion registers, 20 samples each, over the
# personHasCityOfDeath and companyTradesAtStockExchange rows of the mounted
# row files (val and test in one session).
#
# Registers
#   lead   : Wikipedia-lead parenthesis "{s} (" -> (born YYYY) / (YYYY–YYYY)
#   bio    : Name/Born/Died record                -> Died: YYYY | (still alive)
#   city   : Name/City of death record            -> aux top-1 city
#   xchg   : Company/Stock exchange record        -> aux exchanges
set -Eeuo pipefail

cd /opt/akbc

model_key="${AKBC_MODEL_KEY:?}"
gguf_repo="${AKBC_GGUF_REPO:?}"
gguf_revision="${AKBC_GGUF_REV:?}"
gguf_part1="${AKBC_GGUF_PART1:?}"
row_files="${AKBC_DATA_FILE:?comma-separated row files}"
hf_home="/cache/huggingface"
run_dir="/workspace/run"

mkdir -p "${run_dir}/outputs" "${run_dir}/reports"
python3 -m akbc_baseline.download_gguf \
  --repo "${gguf_repo}" --filename "${gguf_part1}" \
  --revision "${gguf_revision}" --cache-dir "${hf_home}" \
  --path-output "${run_dir}/reports/aux-gguf-path.txt" \
  --minimum-gib "${AKBC_GGUF_MIN_GIB:-2}" --maximum-gib 158
gguf_model_path="${hf_home}/models--${gguf_repo//\//--}/snapshots/${gguf_revision}/${gguf_part1}"
test -e "${gguf_model_path}"

export LD_LIBRARY_PATH="/app${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
CUDA_VISIBLE_DEVICES="${AKBC_GPU:-0}" /app/llama-server \
  --model "${gguf_model_path}" \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 16384 --parallel 16 \
  --n-gpu-layers 99 --flash-attn on \
  --batch-size 2048 --ubatch-size 512 \
  > "${run_dir}/reports/aux-llama-server.log" 2>&1 &
server_pid=$!
trap 'kill "${server_pid}" 2>/dev/null || true' EXIT
for i in $(seq 1 180); do
  curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1 && break
  sleep 5
done
curl -sf http://127.0.0.1:8080/health >/dev/null

python3 - "${row_files}" "${run_dir}/outputs/${model_key}" <<'PY'
import concurrent.futures
import hashlib
import json
import sys
import urllib.request

row_files, out_prefix = sys.argv[1].split(","), sys.argv[2]

REGISTERS = {
    "lead": (
        "James Gandolfini (September 18, 1961 – June 19, 2013) was an American actor.\n"
        "Paul McCartney (born 18 June 1942) is an English singer and songwriter.\n"
        "Ada Lovelace (10 December 1815 – 27 November 1852) was an English mathematician.\n"
        "{s} ("
    ),
    "bio": (
        "Name: James Gandolfini\nBorn: 1961\nDied: 2013\n"
        "Name: Paul McCartney\nBorn: 1942\nDied: (still alive)\n"
        "Name: Ada Lovelace\nBorn: 1815\nDied: 1852\n"
        "Name: {s}\nBorn:"
    ),
    "city": (
        "Name: James Gandolfini\nCity of death: Rome\n"
        "Name: Ada Lovelace\nCity of death: London\n"
        "Name: Paul McCartney\nCity of death: (still alive)\n"
        "Name: {s}\nCity of death:"
    ),
    "xchg": (
        "Company: Sony Group Corporation\n"
        "Stock exchange: Tokyo Stock Exchange; New York Stock Exchange\n"
        "Company: Robert Bosch GmbH\n"
        "Stock exchange: none (privately held)\n"
        "Company: Nike, Inc.\n"
        "Stock exchange: New York Stock Exchange\n"
        "Company: {s}\n"
        "Stock exchange:"
    ),
}
BY_RELATION = {
    "personHasCityOfDeath": ["lead", "bio", "city"],
    "companyTradesAtStockExchange": ["xchg"],
}
STOPS = {
    "lead": ["\n"],
    "bio": ["\nName:", "\n\n"],
    "city": ["\nName:", "\n\n"],
    "xchg": ["\nCompany:", "\n\n"],
}

rows = []
seen = set()
for path in row_files:
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["Relation"] not in BY_RELATION:
            continue
        key = (r["Relation"], r["SubjectEntity"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(r)
print(f"{len(rows)} rows", flush=True)

def sample(prompt, seed, stop, n_predict=40):
    body = json.dumps({
        "prompt": prompt, "n_predict": n_predict, "temperature": 0.6,
        "top_p": 0.95, "seed": seed, "stop": stop,
    }).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:8080/completion", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())["content"]

def run_row(row):
    relation, subject = row["Relation"], row["SubjectEntity"]
    out = {"SubjectEntity": subject, "Relation": relation, "registers": {}}
    for reg in BY_RELATION[relation]:
        prompt = REGISTERS[reg].format(s=subject)
        texts = []
        for i in range(20):
            seed = int.from_bytes(hashlib.sha256(
                f"42|{subject}|{reg}|{i}".encode()).digest()[:4], "big")
            texts.append(sample(prompt, seed, STOPS[reg]))
        out["registers"][reg] = texts
    return out

results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
    futures = [pool.submit(run_row, row) for row in rows]
    for n, future in enumerate(concurrent.futures.as_completed(futures), 1):
        results.append(future.result())
        if n % 50 == 0:
            print(f"{n}/{len(rows)} rows", flush=True)

path = f"{out_prefix}-aux-raw.jsonl"
with open(path, "w", encoding="utf-8") as stream:
    for row in results:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"wrote {path}: {len(results)} rows")
print("aux probe completed")
PY
