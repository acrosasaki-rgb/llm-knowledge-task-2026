"""Apply award verification votes to a predictions file (#35).

Drops awardWonBy predictions whose name received zero "yes" votes in the
per-name verification stage (unanimous rejection). Names absent from the
votes file are kept. Deterministic post-processing; all other relations pass
through unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from akbc_baseline.aggregation import normalize_vote  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--votes", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-yes", type=int, default=1)
    args = parser.parse_args()

    votes: dict[str, dict[str, int]] = {}
    for line in open(args.votes, encoding="utf-8"):
        row = json.loads(line)
        votes[row["SubjectEntity"]] = {
            normalize_vote(name["surface"]): name["yes"] for name in row["names"]
        }

    removed = 0
    with open(args.output, "w", encoding="utf-8") as stream:
        for line in open(args.predictions, encoding="utf-8"):
            row = json.loads(line)
            if row["Relation"] == "awardWonBy" and row["SubjectEntity"] in votes:
                yes = votes[row["SubjectEntity"]]
                before = len(row["ObjectEntities"])
                row["ObjectEntities"] = [
                    value
                    for value in row["ObjectEntities"]
                    if yes.get(normalize_vote(value), args.minimum_yes)
                    >= args.minimum_yes
                ]
                removed += before - len(row["ObjectEntities"])
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"removed {removed} unanimously rejected award predictions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
