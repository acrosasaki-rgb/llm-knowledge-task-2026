#!/usr/bin/env bash
# gemma-pt city extras: (1) occ register on every city row of the mounted
# row files (covers the 20 renamed val subjects), (2) identity-dropout occ
# register: given names reduced to initials ("Kari Aronpuro" -> "K. Aronpuro")
# to test whether the answer survives without person identity.
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
  --path-output "${run_dir}/reports/gextra-gguf-path.txt" \
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
  > "${run_dir}/reports/gextra-llama-server.log" 2>&1 &
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
import os
import json
import re
import sys
import urllib.request

row_files, out_prefix = sys.argv[1].split(","), sys.argv[2]

REG_KEY = os.environ.get("AKBC_CITY_REG", "occ")
REGS = {
    "occ": (
        "Name: James Gandolfini\nOccupation: actor\nCity of death: Rome\n"
        "Name: Ada Lovelace\nOccupation: mathematician\nCity of death: London\n"
        "Name: Paul McCartney\nOccupation: musician\nCity of death: (still alive)\n"
        "Name: {s}\nOccupation:"
    ),
    "unk": (
        "Name: James Gandolfini\nOccupation: actor\nCity of death: Rome\n"
        "Name: Ada Lovelace\nOccupation: mathematician\nCity of death: London\n"
        "Name: Paul McCartney\nOccupation: musician\nCity of death: (still alive)\n"
        "Name: Halvard Brenneke\nOccupation: (unknown)\nCity of death: (unknown)\n"
        "Name: {s}\nOccupation:"
    ),
    "cot": (
        "Name: James Gandolfini\nAbout: American actor best known for playing Tony Soprano; he died of a heart attack while on holiday in Italy in 2013.\nCity of death: Rome\n"
        "Name: Ada Lovelace\nAbout: English mathematician who wrote the first algorithm intended for a machine; she died of uterine cancer in 1852.\nCity of death: London\n"
        "Name: Paul McCartney\nAbout: English singer-songwriter and former member of the Beatles; he is still alive and performing.\nCity of death: (still alive)\n"
        "Name: {s}\nAbout:"
    ),
    "sect": (
        "Article: James Gandolfini\nSections: Early life; Career; Personal life; Death; Filmography; Awards and nominations\nDeath section: Gandolfini died of a heart attack on 19 June 2013 while on holiday in Italy.\nCity of death: Rome\n"
        "Article: Ada Lovelace\nSections: Biography; Work; Death; Commemoration; Publications\nDeath section: Lovelace died of uterine cancer on 27 November 1852.\nCity of death: London\n"
        "Article: Paul McCartney\nSections: Early life; The Beatles; Solo career; Personal life; Musicianship; Legacy\nDeath section: (no Death section)\nCity of death: (still alive)\n"
        "Article: {s}\nSections:"
    ),
    "dsec": (
        "Article: James Gandolfini\n== Death ==\nOn 19 June 2013, while on holiday in Italy, Gandolfini died of a heart attack at the Boscolo Exedra Roma hotel.\nCity of death: Rome\n"
        "Article: Ada Lovelace\n== Death ==\nLovelace died of uterine cancer on 27 November 1852 at the age of 36, at her home in Marylebone.\nCity of death: London\n"
        "Article: Paul McCartney\n== Death ==\n(this article has no Death section; the subject is still alive)\nCity of death: (still alive)\n"
        "Article: {s}\n== Death ==\n"
    ),
    "obit": (
        "Name: James Gandolfini\nObituary: The American actor, best known as Tony Soprano, died on 19 June 2013 while on holiday in Italy.\nCity of death: Rome\n"
        "Name: Ada Lovelace\nObituary: The English mathematician, who wrote the first algorithm for the Analytical Engine, died on 27 November 1852 of uterine cancer.\nCity of death: London\n"
        "Name: Paul McCartney\nObituary: (no obituary; he is still alive)\nCity of death: (still alive)\n"
        "Name: {s}\nObituary:"
    ),
    "attr": (
        "Name: James Gandolfini\nOccupation: actor\nNationality: American\nLast years: lived in New York City; travelled to Italy on holiday in 2013\nDeath: heart attack, 19 June 2013\nCity of death: Rome\n"
        "Name: Ada Lovelace\nOccupation: mathematician\nNationality: English\nLast years: lived in London, working on the Analytical Engine\nDeath: uterine cancer, 27 November 1852\nCity of death: London\n"
        "Name: Paul McCartney\nOccupation: musician\nNationality: English\nLast years: still touring and recording\nDeath: (still alive)\nCity of death: (still alive)\n"
        "Name: {s}\nOccupation:"
    ),
    "final": (
        "Name: James Gandolfini\nAbout: American actor best known for playing Tony Soprano; he spent his last years in New York City, and died of a heart attack while on holiday in Rome in 2013.\nCity of death: Rome\n"
        "Name: Ada Lovelace\nAbout: English mathematician who wrote the first algorithm intended for a machine; she spent her last years in London, where she died of uterine cancer in 1852.\nCity of death: London\n"
        "Name: Paul McCartney\nAbout: English singer-songwriter and former member of the Beatles; he lives in England and is still alive.\nCity of death: (still alive)\n"
        "Name: {s}\nAbout:"
    ),
    "src": (
        "Name: James Gandolfini\nWikipedia: James Joseph Gandolfini Jr. (September 18, 1961 - June 19, 2013) was an American actor. He died of a heart attack while on holiday in Rome, Italy.\nCity of death: Rome\n"
        "Name: Ada Lovelace\nWikipedia: Augusta Ada King, Countess of Lovelace (10 December 1815 - 27 November 1852) was an English mathematician. She died of uterine cancer in London at the age of 36.\nCity of death: London\n"
        "Name: Paul McCartney\nWikipedia: Sir James Paul McCartney (born 18 June 1942) is an English singer, songwriter and musician who gained worldwide fame as a member of the Beatles. He is still alive.\nCity of death: (still alive)\n"
        "Name: {s}\nWikipedia:"
    ),
    "cite": (
        "Name: James Gandolfini\nAccording to his Wikipedia article: he was an American actor known for The Sopranos, and he died in Rome, Italy, on 19 June 2013.\nCity of death: Rome\n"
        "Name: Ada Lovelace\nAccording to her Wikipedia article: she was an English mathematician who worked on the Analytical Engine, and she died in London on 27 November 1852.\nCity of death: London\n"
        "Name: Paul McCartney\nAccording to his Wikipedia article: he is an English musician and a former member of the Beatles, and he is still alive.\nCity of death: (still alive)\n"
        "Name: {s}\nAccording to"
    ),
    "cot3": (
        "Name: James Gandolfini\nAbout: American actor best known for playing Tony Soprano.\nDied: 19 June 2013, of a heart attack while on holiday in Italy.\nCity of death: Rome\n"
        "Name: Ada Lovelace\nAbout: English mathematician who wrote the first algorithm intended for a machine.\nDied: 27 November 1852, of uterine cancer.\nCity of death: London\n"
        "Name: Paul McCartney\nAbout: English singer-songwriter and former member of the Beatles.\nDied: (still alive)\nCity of death: (still alive)\n"
        "Name: {s}\nAbout:"
    ),
    "cot4": (
        "Name: James Gandolfini\nAbout: American actor best known for playing Tony Soprano; he died of a heart attack while on holiday in Italy in 2013.\nCity of death: Rome\n"
        "Name: Ada Lovelace\nAbout: English mathematician who wrote the first algorithm intended for a machine; she died of uterine cancer in 1852.\nCity of death: London\n"
        "Name: Paul McCartney\nAbout: English singer-songwriter and former member of the Beatles; he is still alive and performing.\nCity of death: (still alive)\n"
        "Name: Vagif Mustafazadeh\nAbout: Azerbaijani jazz pianist and composer who fused jazz with mugham; he died of a heart attack on stage in 1979.\nCity of death: Tashkent\n"
        "Name: Tom Brokaw\nAbout: American television journalist and former anchor of NBC Nightly News; he is still alive and retired.\nCity of death: (still alive)\n"
        "Name: Nguyen Van Troi\nAbout: Vietnamese electrician and Viet Cong member executed by firing squad in 1964.\nCity of death: Ho Chi Minh City\n"
        "Name: {s}\nAbout:"
    ),
    "cot2": (
        "Name: James Gandolfini\nAbout: American actor best known for playing Tony Soprano; he died of a heart attack while on holiday in Italy in 2013.\nCity of death: Rome\n"
        "Name: Paul McCartney\nAbout: English singer-songwriter and former member of the Beatles; he is still alive and performing.\nCity of death: (still alive)\n"
        "Name: Ada Lovelace\nAbout: English mathematician who wrote the first algorithm intended for a machine; she died of uterine cancer in 1852.\nCity of death: London\n"
        "Name: Halvard Brenneke\nAbout: I do not have any information about this person.\nCity of death: (unknown)\n"
        "Name: {s}\nAbout:"
    ),
    "unk2": (
        "Name: James Gandolfini\nOccupation: actor\nCity of death: Rome\n"
        "Name: Halvard Brenneke\nOccupation: (unknown person)\nCity of death: (unknown)\n"
        "Name: Ada Lovelace\nOccupation: mathematician\nCity of death: London\n"
        "Name: Paul McCartney\nOccupation: musician\nCity of death: (still alive)\n"
        "Name: Rosalind Ekwueme\nOccupation: (unknown person)\nCity of death: (unknown)\n"
        "Name: {s}\nOccupation:"
    ),
}
OCC = REGS[REG_KEY]

def dropout(name):
    base = re.sub(r"\s*\(.*\)$", "", name).strip()
    parts = base.split()
    if len(parts) < 2:
        return None
    return parts[0][0] + ". " + " ".join(parts[1:])

rows, seen = [], set()
for path in row_files:
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["Relation"] != "personHasCityOfDeath":
            continue
        if r["SubjectEntity"] in seen:
            continue
        seen.add(r["SubjectEntity"])
        rows.append(r)
print(f"{len(rows)} city rows", flush=True)

def sample(prompt, seed):
    body = json.dumps({
        "prompt": prompt, "n_predict": (32 if REG_KEY in ("occ", "unk", "unk2") else (128 if REG_KEY in ("attr", "sect") else 96)), "temperature": float(os.environ.get("AKBC_TEMP", "0.6")),
        "top_p": 0.95, "seed": seed, "stop": ["\nName:", "\nArticle:", "\n\n"],
    }).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:8080/completion", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())["content"]

def parse(text):
    text = text.strip()
    if "alive" in text.lower() or "unknown" in text.lower():
        return []
    m = re.search(r"City of death:\s*([^\n,.(]+)", text)
    if m:
        city = m.group(1).strip()
        return [city] if city and "alive" not in city.lower() else []
    return []

def run(subject, tag, name):
    prompt = OCC.format(s=name)
    cands, diags = [], []
    for i in range(20):
        seed = int.from_bytes(hashlib.sha256(
            f"42|{subject}|{tag}|{i}".encode()).digest()[:4], "big")
        text = sample(prompt, seed)
        cands.append(parse(text))
        diags.append({"raw_text": text})
    return cands, diags

def run_row(row):
    s = row["SubjectEntity"]
    out = {"SubjectEntity": s, "Relation": "personHasCityOfDeath"}
    out["Candidates"], out["CandidateDiagnostics"] = run(s, "occ", s)
    d = dropout(s)
    out["DropoutName"] = d
    if d:
        out["DropoutCandidates"], out["DropoutDiagnostics"] = run(s, "drop", d)
    else:
        out["DropoutCandidates"], out["DropoutDiagnostics"] = None, None
    return out

results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
    futures = [pool.submit(run_row, row) for row in rows]
    for n, future in enumerate(concurrent.futures.as_completed(futures), 1):
        results.append(future.result())
        if n % 50 == 0:
            print(f"{n}/{len(rows)} rows", flush=True)

path = f"{out_prefix}-city-extra.jsonl"
with open(path, "w", encoding="utf-8") as stream:
    for row in results:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"wrote {path}: {len(results)} rows")
print("gemma city extra completed")
PY
