"""Validate V19 candidate and submission JSONL against the official input."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

OFFICIAL_KEYS = {"SubjectEntity", "Relation", "ObjectEntities"}


def load(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def verify(rows: list[dict], candidates: list[dict], predictions: list[dict]) -> None:
    if not (len(rows) == len(candidates) == len(predictions)):
        raise ValueError(
            f"row count mismatch: input={len(rows)} candidates={len(candidates)} predictions={len(predictions)}"
        )
    for index, (source, candidate, prediction) in enumerate(
        zip(rows, candidates, predictions), 1
    ):
        identity = (source["SubjectEntity"], source["Relation"])
        if (candidate.get("SubjectEntity"), candidate.get("Relation")) != identity:
            raise ValueError(f"candidate order mismatch at line {index}")
        if (prediction.get("SubjectEntity"), prediction.get("Relation")) != identity:
            raise ValueError(f"prediction order mismatch at line {index}")
        if len(candidate.get("Candidates", [])) != 20:
            raise ValueError(f"line {index} does not contain 20 candidates")
        if set(prediction) != OFFICIAL_KEYS:
            raise ValueError(f"prediction schema mismatch at line {index}: {sorted(prediction)}")
        if not isinstance(prediction["ObjectEntities"], list):
            raise ValueError(f"ObjectEntities is not a list at line {index}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--sha256-output", required=True)
    args = parser.parse_args()
    verify(load(args.input), load(args.candidates), load(args.predictions))
    digest = hashlib.sha256(Path(args.predictions).read_bytes()).hexdigest()
    Path(args.sha256_output).write_text(
        f"{digest}  {Path(args.predictions).name}\n", encoding="utf-8"
    )
    print(f"verified {args.predictions}: sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
