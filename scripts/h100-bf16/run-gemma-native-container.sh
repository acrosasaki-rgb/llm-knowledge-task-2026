#!/usr/bin/env bash
# gemma-pt: (1) native-language city-of-death registers (12 languages, the
# model transliterates the name itself), (2) company status / history
# registers for the delisting certificate. Runs over the city and company rows
# of the mounted row files (val + test), 20 samples per register.
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
  --path-output "${run_dir}/reports/gnative-gguf-path.txt" \
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
  > "${run_dir}/reports/gnative-llama-server.log" 2>&1 &
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
import os
import sys
import urllib.request

row_files, out_prefix = sys.argv[1].split(","), sys.argv[2]

# Each city register: three exemplars in the target language (dead/dead/alive),
# then "Name: {s}" in Latin script and the native-name line to be completed.
CITY = {
    "ru": ("Name: James Gandolfini\nИмя: Джеймс Гандольфини\nРод занятий: актёр\nМесто смерти: Рим\n"
           "Name: Ada Lovelace\nИмя: Ада Лавлейс\nРод занятий: математик\nМесто смерти: Лондон\n"
           "Name: Paul McCartney\nИмя: Пол Маккартни\nРод занятий: музыкант\nМесто смерти: (жив)\n"
           "Name: {s}\nИмя:", "Место смерти:", ["жив", "жива", "здравствует"]),
    "uk": ("Name: James Gandolfini\nІм'я: Джеймс Гандольфіні\nРід занять: актор\nМісце смерті: Рим\n"
           "Name: Ada Lovelace\nІм'я: Ада Лавлейс\nРід занять: математик\nМісце смерті: Лондон\n"
           "Name: Paul McCartney\nІм'я: Пол Маккартні\nРід занять: музикант\nМісце смерті: (живий)\n"
           "Name: {s}\nІм'я:", "Місце смерті:", ["жив", "жива"]),
    "pl": ("Name: James Gandolfini\nImię i nazwisko: James Gandolfini\nZawód: aktor\nMiejsce śmierci: Rzym\n"
           "Name: Ada Lovelace\nImię i nazwisko: Ada Lovelace\nZawód: matematyczka\nMiejsce śmierci: Londyn\n"
           "Name: Paul McCartney\nImię i nazwisko: Paul McCartney\nZawód: muzyk\nMiejsce śmierci: (żyje)\n"
           "Name: {s}\nImię i nazwisko:", "Miejsce śmierci:", ["żyje"]),
    "hu": ("Name: James Gandolfini\nNév: James Gandolfini\nFoglalkozás: színész\nHalálának helye: Róma\n"
           "Name: Ada Lovelace\nNév: Ada Lovelace\nFoglalkozás: matematikus\nHalálának helye: London\n"
           "Name: Paul McCartney\nNév: Paul McCartney\nFoglalkozás: zenész\nHalálának helye: (még él)\n"
           "Name: {s}\nNév:", "Halálának helye:", ["még él", "él"]),
    "ro": ("Name: James Gandolfini\nNume: James Gandolfini\nOcupație: actor\nLocul decesului: Roma\n"
           "Name: Ada Lovelace\nNume: Ada Lovelace\nOcupație: matematiciană\nLocul decesului: Londra\n"
           "Name: Paul McCartney\nNume: Paul McCartney\nOcupație: muzician\nLocul decesului: (în viață)\n"
           "Name: {s}\nNume:", "Locul decesului:", ["în viață"]),
    "de": ("Name: James Gandolfini\nName (deutsch): James Gandolfini\nBeruf: Schauspieler\nSterbeort: Rom\n"
           "Name: Ada Lovelace\nName (deutsch): Ada Lovelace\nBeruf: Mathematikerin\nSterbeort: London\n"
           "Name: Paul McCartney\nName (deutsch): Paul McCartney\nBeruf: Musiker\nSterbeort: (lebt noch)\n"
           "Name: {s}\nName (deutsch):", "Sterbeort:", ["lebt"]),
    "fr": ("Name: James Gandolfini\nNom: James Gandolfini\nProfession: acteur\nLieu de décès: Rome\n"
           "Name: Ada Lovelace\nNom: Ada Lovelace\nProfession: mathématicienne\nLieu de décès: Londres\n"
           "Name: Paul McCartney\nNom: Paul McCartney\nProfession: musicien\nLieu de décès: (toujours en vie)\n"
           "Name: {s}\nNom:", "Lieu de décès:", ["en vie", "vivant"]),
    "es": ("Name: James Gandolfini\nNombre: James Gandolfini\nOcupación: actor\nLugar de fallecimiento: Roma\n"
           "Name: Ada Lovelace\nNombre: Ada Lovelace\nOcupación: matemática\nLugar de fallecimiento: Londres\n"
           "Name: Paul McCartney\nNombre: Paul McCartney\nOcupación: músico\nLugar de fallecimiento: (sigue vivo)\n"
           "Name: {s}\nNombre:", "Lugar de fallecimiento:", ["vivo", "viva"]),
    "it": ("Name: James Gandolfini\nNome: James Gandolfini\nProfessione: attore\nLuogo di morte: Roma\n"
           "Name: Ada Lovelace\nNome: Ada Lovelace\nProfessione: matematica\nLuogo di morte: Londra\n"
           "Name: Paul McCartney\nNome: Paul McCartney\nProfessione: musicista\nLuogo di morte: (ancora vivo)\n"
           "Name: {s}\nNome:", "Luogo di morte:", ["vivo", "viva", "vivente"]),
    "fi": ("Name: James Gandolfini\nNimi: James Gandolfini\nAmmatti: näyttelijä\nKuolinpaikka: Rooma\n"
           "Name: Ada Lovelace\nNimi: Ada Lovelace\nAmmatti: matemaatikko\nKuolinpaikka: Lontoo\n"
           "Name: Paul McCartney\nNimi: Paul McCartney\nAmmatti: muusikko\nKuolinpaikka: (elossa)\n"
           "Name: {s}\nNimi:", "Kuolinpaikka:", ["elossa", "elää"]),
    "tr": ("Name: James Gandolfini\nAd: James Gandolfini\nMeslek: oyuncu\nÖlüm yeri: Roma\n"
           "Name: Ada Lovelace\nAd: Ada Lovelace\nMeslek: matematikçi\nÖlüm yeri: Londra\n"
           "Name: Paul McCartney\nAd: Paul McCartney\nMeslek: müzisyen\nÖlüm yeri: (hayatta)\n"
           "Name: {s}\nAd:", "Ölüm yeri:", ["hayatta", "yaşıyor"]),
    "ja": ("Name: James Gandolfini\n氏名: ジェームズ・ガンドルフィーニ\n職業: 俳優\n死没地: ローマ\n"
           "Name: Ada Lovelace\n氏名: エイダ・ラブレス\n職業: 数学者\n死没地: ロンドン\n"
           "Name: Paul McCartney\n氏名: ポール・マッカートニー\n職業: 音楽家\n死没地: (存命)\n"
           "Name: {s}\n氏名:", "死没地:", ["存命"]),
}

COMPANY = {
    "status": (
        "Company: Time Inc.\nStatus: acquired by Meredith Corporation in 2018; delisted from the New York Stock Exchange\n"
        "Company: Nike, Inc.\nStatus: publicly traded; listed on the New York Stock Exchange\n"
        "Company: Robert Bosch GmbH\nStatus: privately held; not listed\n"
        "Company: Compaq\nStatus: acquired by Hewlett-Packard in 2002; delisted from the New York Stock Exchange\n"
        "Company: {s}\nStatus:"
    ),
    "hist": (
        "Company: Compaq\nCorporate history: founded 1982; IPO 1983 on the New York Stock Exchange; acquired by Hewlett-Packard in 2002 (delisted)\n"
        "Company: Nike, Inc.\nCorporate history: founded 1964; IPO 1980; listed on the New York Stock Exchange (still listed)\n"
        "Company: {s}\nCorporate history:"
    ),
}

REGS = set(os.environ.get("AKBC_REGS", ",".join(list(CITY) + list(COMPANY))).split(","))
CITY = {k: v for k, v in CITY.items() if k in REGS}
COMPANY = {k: v for k, v in COMPANY.items() if k in REGS}
print("registers:", sorted(CITY), sorted(COMPANY), flush=True)

rows, seen = [], set()
for path in row_files:
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["Relation"] not in ("personHasCityOfDeath", "companyTradesAtStockExchange"):
            continue
        if (r["Relation"] == "personHasCityOfDeath" and not CITY) or (r["Relation"] != "personHasCityOfDeath" and not COMPANY):
            continue
        key = (r["Relation"], r["SubjectEntity"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(r)
print(f"{len(rows)} rows", flush=True)

def sample(prompt, seed, stop, n_predict):
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
    if relation == "personHasCityOfDeath":
        for lang, (template, _, _) in CITY.items():
            prompt = template.format(s=subject)
            texts = []
            for i in range(20):
                seed = int.from_bytes(hashlib.sha256(
                    f"42|{subject}|city-{lang}|{i}".encode()).digest()[:4], "big")
                texts.append(sample(prompt, seed, ["\nName:", "\n\n"], 48))
            out["registers"][lang] = texts
    else:
        for reg, template in COMPANY.items():
            prompt = template.format(s=subject)
            texts = []
            for i in range(20):
                seed = int.from_bytes(hashlib.sha256(
                    f"42|{subject}|co-{reg}|{i}".encode()).digest()[:4], "big")
                texts.append(sample(prompt, seed, ["\nCompany:", "\n\n"], 64))
            out["registers"][reg] = texts
    return out

results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
    futures = [pool.submit(run_row, row) for row in rows]
    for n, future in enumerate(concurrent.futures.as_completed(futures), 1):
        results.append(future.result())
        if n % 25 == 0:
            print(f"{n}/{len(rows)} rows", flush=True)

path = f"{out_prefix}-native-raw.jsonl"
with open(path, "w", encoding="utf-8") as stream:
    for row in results:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"wrote {path}: {len(results)} rows")
print("gemma native completed")
PY
