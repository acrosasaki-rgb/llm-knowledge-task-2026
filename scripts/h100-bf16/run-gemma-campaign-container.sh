#!/usr/bin/env bash
# Gemma-pt campaign, phase 1: three city registers and one award register,
# raw completion, 20 samples per row per register.
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
  --path-output "${run_dir}/reports/campaign-gguf-path.txt" \
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
  > "${run_dir}/reports/campaign-llama-server.log" 2>&1 &
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

CITY_REGISTERS = {
    "bio": (
        "Name: James Gandolfini\nDied: 2013, Rome\n"
        "Name: Ada Lovelace\nDied: 1852, London\n"
        "Name: Paul McCartney\nDied: (still alive)\n"
        "Name: {s}\nDied:"
    ),
    "occ": (
        "Name: James Gandolfini\nOccupation: actor\nCity of death: Rome\n"
        "Name: Ada Lovelace\nOccupation: mathematician\nCity of death: London\n"
        "Name: Paul McCartney\nOccupation: musician\nCity of death: (still alive)\n"
        "Name: {s}\nOccupation:"
    ),
    "prose": (
        "James Gandolfini died on 19 June 2013 in Rome.\n"
        "Ada Lovelace died on 27 November 1852 in London.\n"
        "{s} died on"
    ),
}
AWARD_REGISTER = (
    "Award: Hugo Award for Best Novel\n"
    "Recipients: Isaac Asimov; Frank Herbert; Ursula K. Le Guin; "
    "Arthur C. Clarke; Kim Stanley Robinson; N. K. Jemisin\n"
    "Award: {s}\n"
    "Recipients:"
)

rows = [json.loads(line) for line in open(val_path, encoding="utf-8")]
city_rows = [r for r in rows if r["Relation"] == "personHasCityOfDeath"]
award_rows = [r for r in rows if r["Relation"] == "awardWonBy"]

def sample(prompt, seed, n_predict=32):
    body = json.dumps({
        "prompt": prompt, "n_predict": n_predict, "temperature": 0.6,
        "top_p": 0.95, "seed": seed, "stop": ["\nName:", "\nAward:", "\n\n"],
    }).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:8080/completion", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())["content"]

def parse_city(register, text):
    text = text.strip()
    if "alive" in text.lower() or text.startswith("("):
        return []
    if register == "bio":
        m = re.match(r"^\s*\d{3,4}\s*,\s*([^\n,.]+)", text)
        return [m.group(1).strip()] if m else []
    if register == "occ":
        m = re.search(r"City of death:\s*([^\n,.(]+)", text)
        if m:
            city = m.group(1).strip()
            return [city] if city and "alive" not in city.lower() else []
        return []
    m = re.search(r"\bin ([A-Z][^,.\n]*)", text)
    return [m.group(1).strip()] if m else []

def run_city(row, register, template):
    prompt = template.format(s=row["SubjectEntity"])
    cands, diags = [], []
    for i in range(20):
        seed = int.from_bytes(hashlib.sha256(
            f"42|{row['SubjectEntity']}|{register}|{i}".encode()
        ).digest()[:4], "big")
        text = sample(prompt, seed)
        cands.append(parse_city(register, text))
        diags.append({"raw_text": text})
    return {"SubjectEntity": row["SubjectEntity"],
            "Relation": "personHasCityOfDeath", "Candidates": cands,
            "CandidateDiagnostics": diags}

def run_award(row):
    prompt = AWARD_REGISTER.format(s=row["SubjectEntity"])
    cands, diags = [], []
    for i in range(20):
        seed = int.from_bytes(hashlib.sha256(
            f"42|{row['SubjectEntity']}|award|{i}".encode()
        ).digest()[:4], "big")
        text = sample(prompt, seed, n_predict=220)
        names = [x.strip().rstrip(".") for x in text.split(";") if x.strip()]
        cands.append([n for n in names if 0 < len(n) < 60])
        diags.append({"raw_text": text})
    return {"SubjectEntity": row["SubjectEntity"], "Relation": "awardWonBy",
            "Candidates": cands, "CandidateDiagnostics": diags}

with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
    for register, template in CITY_REGISTERS.items():
        results = list(pool.map(
            lambda r: run_city(r, register, template), city_rows))
        path = f"{out_prefix}-city-{register}-candidates.jsonl"
        with open(path, "w", encoding="utf-8") as stream:
            for row in results:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {path}: {len(results)} rows", flush=True)
    results = list(pool.map(run_award, award_rows))
    path = f"{out_prefix}-award-candidates.jsonl"
    with open(path, "w", encoding="utf-8") as stream:
        for row in results:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {path}: {len(results)} rows", flush=True)
print("campaign phase1 completed")
PY
