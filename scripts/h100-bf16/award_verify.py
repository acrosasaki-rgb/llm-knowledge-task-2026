"""Per-name verification votes for awardWonBy candidates (#35).

Reads a candidates JSONL, collects the distinct candidate names per award,
and asks the model, for each (award, name) pair, whether that name actually
received the exact award. Votes are written per name; threshold selection
and scoring happen offline. No thinking: the verification is a short
closed-book yes/no probe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, build_opener, ProxyHandler

sys.path.insert(0, "/opt/akbc/src")
from akbc_baseline.aggregation import normalize_vote  # noqa: E402

SYSTEM = (
    "You are checking recorded award recipients. Answer with exactly one "
    "word: yes or no."
)
QUESTION = (
    "According to Wikipedia and Wikidata, did {name} receive the award "
    "\"{award}\"? Consider only this exact award, not similarly named or "
    "predecessor awards. Answer yes or no."
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--votes", type=int, default=3)
    parser.add_argument("--ports", default="8080,8081,8082,8083,8084,8085,8086,8087")
    parser.add_argument("--model", default="unsloth/Qwen3.5-27B-GGUF")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ports = [int(p) for p in args.ports.split(",")]
    opener = build_opener(ProxyHandler({}))

    rows = []
    for line in open(args.candidates, encoding="utf-8"):
        row = json.loads(line)
        if row["Relation"] == "awardWonBy":
            rows.append(row)

    tasks = []
    per_row: dict[str, dict[str, dict]] = {}
    for row in rows:
        subject = row["SubjectEntity"]
        names: dict[str, str] = {}
        for candidate in row["Candidates"]:
            for value in candidate:
                key = normalize_vote(value)
                if key and key not in names:
                    names[key] = value
        per_row[subject] = {
            key: {"surface": surface, "yes": 0, "votes": 0}
            for key, surface in names.items()
        }
        for key in names:
            tasks.append((subject, key))
    print(f"{len(rows)} award rows, {len(tasks)} (award, name) pairs")

    def ask(index: int, subject: str, key: str) -> tuple[str, str, int, int]:
        surface = per_row[subject][key]["surface"]
        yes = 0
        total = 0
        for vote in range(args.votes):
            digest = hashlib.sha256(
                f"{args.seed}\0{subject}\0{key}\0{vote}".encode("utf-8")
            ).digest()
            payload = {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {
                        "role": "user",
                        "content": QUESTION.format(name=surface, award=subject),
                    },
                ],
                "max_tokens": 8,
                "temperature": 0.6,
                "top_p": 0.95,
                "seed": int.from_bytes(digest[:4], "big"),
                "chat_template_kwargs": {"enable_thinking": False},
            }
            port = ports[(index + vote) % len(ports)]
            request = Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with opener.open(request, timeout=300) as response:
                decoded = json.loads(response.read().decode("utf-8"))
            content = decoded["choices"][0]["message"]["content"]
            answer = content.rsplit("</think>", 1)[-1].strip().lower()
            total += 1
            if answer.startswith("yes"):
                yes += 1
        return subject, key, yes, total

    with ThreadPoolExecutor(max_workers=64) as pool:
        futures = [
            pool.submit(ask, index, subject, key)
            for index, (subject, key) in enumerate(tasks)
        ]
        done = 0
        for future in futures:
            subject, key, yes, total = future.result()
            per_row[subject][key]["yes"] = yes
            per_row[subject][key]["votes"] = total
            done += 1
            if done % 200 == 0:
                print(f"verified {done}/{len(tasks)}")

    with open(args.output, "w", encoding="utf-8") as stream:
        for subject, names in per_row.items():
            stream.write(
                json.dumps(
                    {
                        "SubjectEntity": subject,
                        "Relation": "awardWonBy",
                        "names": list(names.values()),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
