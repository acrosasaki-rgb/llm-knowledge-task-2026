"""Merge updated hasArea/hasCapacity test predictions into the base test file.

Rebuilds the numeric relations from new candidate pools (hasArea: median over
the REAP-exemplar pool; hasCapacity: dominant_cluster over the P1083 pool) and
keeps every other relation's rows from the base predictions untouched.

Usage:
  python scripts/h100-bf16/merge_test_predictions.py \
    --base outputs/screening/mistral-small-24b-test-predictions.jsonl \
    --area-candidates outputs/screening/mistral-area-reap-test-candidates.jsonl \
    --cap-candidates outputs/screening/mistral-cap-grounding-test-candidates.jsonl \
    --output outputs/screening/mistral-small-24b-test-predictions-v2.jsonl
"""

import argparse
import json
import sys

sys.path.insert(0, "src")

from akbc_baseline.aggregation import aggregate_dominant_cluster, aggregate_median


def load_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--area-candidates", required=True)
    parser.add_argument("--cap-candidates", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    area = {
        row["SubjectEntity"]: row["Candidates"]
        for row in load_jsonl(args.area_candidates)
        if row["Relation"] == "hasArea"
    }
    cap = {
        row["SubjectEntity"]: row["Candidates"]
        for row in load_jsonl(args.cap_candidates)
        if row["Relation"] == "hasCapacity"
    }

    base_rows = load_jsonl(args.base)
    replaced = {"hasArea": 0, "hasCapacity": 0}
    with open(args.output, "w", encoding="utf-8") as stream:
        for row in base_rows:
            relation = row["Relation"]
            subject = row["SubjectEntity"]
            if relation == "hasArea":
                assert subject in area, f"missing hasArea pool for {subject}"
                value = aggregate_median(area[subject])
                row = dict(row, ObjectEntities=value or [])
                replaced[relation] += 1
            elif relation == "hasCapacity":
                assert subject in cap, f"missing hasCapacity pool for {subject}"
                value = aggregate_dominant_cluster(cap[subject])
                row = dict(row, ObjectEntities=value or [])
                replaced[relation] += 1
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    assert replaced == {"hasArea": 100, "hasCapacity": 100}, replaced
    print(f"wrote {args.output}: replaced {replaced}")


if __name__ == "__main__":
    main()
