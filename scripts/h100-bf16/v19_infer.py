"""Generate the exact candidate pool used by the submitted V19 system.

The model server is supplied by ``run-v19-container.sh``.  This module keeps
the final registers, parsing, candidate count, and seed namespaces in a
testable Python file instead of spreading them over exploratory shell scripts.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import urllib.request
from pathlib import Path

CANDIDATE_COUNT = 20
TEMPERATURE = 0.6
TOP_P = 0.95

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
        "Name: James Gandolfini\nAbout: American actor best known for playing Tony Soprano; he died of a heart attack while on holiday in Italy in 2013.\nCity of death: Rome\n"
        "Name: Ada Lovelace\nAbout: English mathematician who wrote the first algorithm intended for a machine; she died of uterine cancer in 1852.\nCity of death: London\n"
        "Name: Paul McCartney\nAbout: English singer-songwriter and former member of the Beatles; he is still alive and performing.\nCity of death: (still alive)\n"
        "Name: {s}\nAbout:"
    ),
    "countryLandBordersCountry": (
        "Country: Portugal\nLand borders: Spain\n"
        "Country: Japan\nLand borders: none\n"
        "Country: Austria\nLand borders: Germany; Czech Republic; Slovakia; Hungary; Slovenia; Italy; Switzerland; Liechtenstein\n"
        "Country: {s}\nLand borders:"
    ),
    "companyTradesAtStockExchange": (
        "Company: Sony Group Corporation\nExchanges: Tokyo Stock Exchange; New York Stock Exchange\n"
        "Company: Robert Bosch GmbH\nExchanges: none (privately held)\n"
        "Company: Nike, Inc.\nExchanges: New York Stock Exchange\n"
        "Company: {s}\nExchanges:"
    ),
    "awardWonBy": (
        "Award: Hugo Award for Best Novel\n"
        "Recipients: Isaac Asimov; Frank Herbert; Ursula K. Le Guin; Arthur C. Clarke; Kim Stanley Robinson; N. K. Jemisin\n"
        "Award: {s}\nRecipients:"
    ),
}


def candidate_seed(base_seed: str, subject: str, relation: str, index: int) -> int:
    # The V19 city pool came from the historical city-extra script, whose
    # namespace remained "occ" when the register changed to About-first CoT.
    namespace = "occ" if relation == "personHasCityOfDeath" else relation
    payload = f"{base_seed}|{subject}|{namespace}|{index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def parse_candidate(relation: str, text: str) -> list[str]:
    text = text.strip()
    if relation in ("hasArea", "hasCapacity"):
        match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
        return [match.group(0).replace(",", "")] if match else []
    if relation == "personHasCityOfDeath":
        if "alive" in text.casefold() or "unknown" in text.casefold():
            return []
        match = re.search(r"City of death:\s*([^\n,.(]+)", text)
        if not match:
            return []
        city = match.group(1).strip()
        return [city] if city else []
    first_line = text.split("\n", 1)[0]
    if not first_line or first_line.casefold().startswith("none"):
        return []
    names = [part.strip().rstrip(".") for part in first_line.split(";")]
    return [name for name in names if 0 < len(name) < 60]


def complete(
    server_url: str,
    prompt: str,
    seed: int,
    n_predict: int,
    stop: list[str],
) -> str:
    payload = json.dumps({
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "seed": seed,
        "stop": stop,
    }).encode()
    request = urllib.request.Request(
        f"{server_url.rstrip('/')}/completion",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())["content"]


def generate_row(row: dict, server_url: str, base_seed: str) -> dict:
    relation = row["Relation"]
    subject = row["SubjectEntity"]
    prompt = REGISTERS[relation].format(s=subject)
    n_predict = 220 if relation == "awardWonBy" else (96 if relation == "personHasCityOfDeath" else 32)
    stop = (
        ["\nName:", "\nArticle:", "\n\n"]
        if relation == "personHasCityOfDeath"
        else ["\nName:", "\nAward:", "\nCompany:", "\nCountry:", "\n\n", "\nThe Wikipedia"]
    )
    candidates, diagnostics = [], []
    for index in range(CANDIDATE_COUNT):
        raw = complete(
            server_url,
            prompt,
            candidate_seed(base_seed, subject, relation, index),
            n_predict,
            stop,
        )
        candidates.append(parse_candidate(relation, raw))
        diagnostics.append({"raw_text": raw})
    return {
        "SubjectEntity": subject,
        "Relation": relation,
        "Candidates": candidates,
        "CandidateDiagnostics": diagnostics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:8080")
    parser.add_argument("--seed", default="42")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.input, encoding="utf-8")]
    unknown = sorted({row["Relation"] for row in rows} - set(REGISTERS))
    if unknown:
        raise SystemExit(f"unsupported relations: {unknown}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        # executor.map preserves official input order while still running rows concurrently.
        results = pool.map(
            lambda row: generate_row(row, args.server_url, args.seed), rows
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="\n") as stream:
            for index, result in enumerate(results, 1):
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                print(f"{index}/{len(rows)} rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
