"""Convert the public 477-row test checkout to the submitted 475-row input.

The official Codabench test input was disambiguated after the dataset commit
used by the original development harness: 15 subjects gained qualifiers and
two obsolete capacity rows were removed.  Keeping this deterministic migration
in code makes the submitted V19 input reconstructible from the public commit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DROPPED = {
    ("hasCapacity", "Charger Stadium in Texas"),
    ("hasCapacity", "Cougars Den in Virginia"),
}

RENAMED = {
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


def prepare(rows: list[dict]) -> list[dict]:
    output = []
    renamed = set()
    dropped = set()
    for source in rows:
        row = dict(source)
        identity = (row["Relation"], row["SubjectEntity"])
        if identity in DROPPED:
            dropped.add(identity)
            continue
        old_name = row["SubjectEntity"]
        if old_name in RENAMED:
            row["SubjectEntity"] = RENAMED[old_name]
            renamed.add(old_name)
        output.append(row)

    # Accept either the original public 477-row input or an already migrated
    # 475-row input, but reject partial/mixed migrations.
    if len(rows) == 477:
        if renamed != set(RENAMED) or dropped != DROPPED:
            raise ValueError("477-row input does not match the V19 migration contract")
    elif len(rows) == 475:
        expected_names = set(RENAMED.values())
        actual_names = {row["SubjectEntity"] for row in output}
        if not expected_names <= actual_names or dropped:
            raise ValueError("475-row input is not the submitted V19 input")
    else:
        raise ValueError(f"expected 477 pre-migration or 475 V19 rows, got {len(rows)}")
    if len(output) != 475:
        raise ValueError(f"V19 input must contain 475 rows, got {len(output)}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in open(args.input, encoding="utf-8")]
    output = prepare(rows)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in output:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"prepared V19 input: {len(rows)} -> {len(output)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
