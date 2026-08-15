"""Assemble v8 inputs for the August 2026 dataset-disambiguation release.

Rebuilds the base predictions and every candidate pool against the new
475-row test split: rows whose subjects were renamed are replaced with
freshly generated candidates under the new subject strings, and the two
dropped capacity venues are removed. Unchanged rows are carried over
verbatim so the v6-winning configuration is preserved everywhere else.

Inputs (in outputs/screening/):
  test-new-475.jsonl                 -- the e079024b test split
  renamed-base-candidates.jsonl      -- 15 renamed rows, base test config
  renamed-rq3-candidates.jsonl       -- 6 renamed hasArea rows, recall arm
  renamed-llama-candidates.jsonl     -- 6 renamed hasArea rows, arbiter
Outputs (suffix -v8.jsonl), ready for build_predictions_v3.py.
"""

import json
from collections import Counter

DIR = "outputs/screening/"

DROPPED = {
    ("hasCapacity", "Charger Stadium in Texas"),
    ("hasCapacity", "Cougars Den in Virginia"),
}
RENAMED = {  # old -> new subject strings (relation-unique)
    "Franklin Medal": "Franklin Medal (Franklin Institute)",
    "United Aircraft Corporation": "United Aircraft Corporation (Russia)",
    "Boa Vista": "Boa Vista, Cape Verde",
    "Brava": "Brava, Cape Verde",
    "Cabrera": "Cabrera, Balearic Islands",
    "Gorgona": "Gorgona, Italy",
    "Ireland": "Republic of Ireland",
    "Tortuga": "Tortuga, Haiti",
    "Jim Roberts": "Jim Roberts (ice hockey, born 1940)",
    "John Lewis": "John Lewis (civil rights leader)",
    "József Tóth": "József Tóth (photographer)",
    "Peter Cartwright": "Peter Cartwright (lawyer)",
    "Petr Hájek": "Petr Hájek (logician)",
    "Vladimir": "Vladimir (Ikim)",
    "William Owens": "William Owens (admiral)",
}
NEW_NAMES = set(RENAMED.values())


def load(path):
    with open(DIR + path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def dump(rows, path):
    with open(DIR + path, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {path}: {len(rows)} rows")


def rebuild_pool(old_path, renamed_rows, out_path, relations):
    kept = [
        row
        for row in load(old_path)
        if row["Relation"] in relations
        and (row["Relation"], row["SubjectEntity"]) not in DROPPED
        and row["SubjectEntity"] not in RENAMED
    ]
    fresh = [r for r in renamed_rows if r["Relation"] in relations]
    dump(kept + fresh, out_path)


def city_majority(cands, minimum_votes=10):
    counts = Counter(c[0].strip() if c else "" for c in cands)
    winner, votes = counts.most_common(1)[0]
    return [] if (winner == "" or votes < minimum_votes) else [winner]


def award_frequency(cands, threshold=0.1):
    counts = Counter()
    for cand in cands:
        for name in cand or []:
            counts[name.strip()] += 1
    return [n for n, v in counts.items() if v / len(cands) >= threshold]


def main():
    new_rows = load("test-new-475.jsonl")
    assert len(new_rows) == 475, len(new_rows)
    renamed_base = load("renamed-base-candidates.jsonl")
    renamed_by_subject = {r["SubjectEntity"]: r for r in renamed_base}
    assert set(renamed_by_subject) == NEW_NAMES, (
        NEW_NAMES - set(renamed_by_subject)
    )

    # Base predictions: carry over unchanged rows; fill renamed city/award
    # rows here (relations the builder does not recompute); leave renamed
    # company/area/capacity rows empty for the builder to recompute.
    old_pred = {
        (r["Relation"], r["SubjectEntity"]): r["ObjectEntities"]
        for r in load("mistral-small-24b-test-predictions.jsonl")
    }
    out = []
    for row in new_rows:
        relation, subject = row["Relation"], row["SubjectEntity"]
        if subject in NEW_NAMES:
            cands = renamed_by_subject[subject]["Candidates"]
            if relation == "personHasCityOfDeath":
                objects = city_majority(cands)
            elif relation == "awardWonBy":
                # NOTE: the unanimous-no verification filter is not re-run
                # for this single renamed award row.
                objects = award_frequency(cands)
            else:
                objects = []  # recomputed by build_predictions_v3.py
        else:
            objects = old_pred[(relation, subject)]
        out.append(
            {"SubjectEntity": subject, "Relation": relation,
             "ObjectEntities": objects}
        )
    dump(out, "mistral-small-24b-test-predictions-base-v8.jsonl")

    rebuild_pool(
        "mistral-small-24b-test-candidates.jsonl", renamed_base,
        "mistral-small-24b-test-candidates-v8.jsonl",
        {"hasArea", "hasCapacity", "companyTradesAtStockExchange",
         "personHasCityOfDeath", "awardWonBy",
         "countryLandBordersCountry"},
    )
    rebuild_pool(
        "mistral-area-rq3-test-candidates.jsonl",
        load("renamed-rq3-candidates.jsonl"),
        "mistral-area-rq3-test-candidates-v8.jsonl", {"hasArea"},
    )
    rebuild_pool(
        "llama31-8b-area-test-candidates.jsonl",
        load("renamed-llama-candidates.jsonl"),
        "llama31-8b-area-test-candidates-v8.jsonl", {"hasArea"},
    )
    for pool in (
        "mistral-cap-grounding-test-candidates.jsonl",
        "mistral-cap-step-test-candidates.jsonl",
        "llama31-8b-cap-test-candidates.jsonl",
        "ministral-8b-cap-test-candidates.jsonl",
    ):
        rebuild_pool(pool, [], pool.replace(".jsonl", "-v8.jsonl"),
                     {"hasCapacity"})


if __name__ == "__main__":
    main()
