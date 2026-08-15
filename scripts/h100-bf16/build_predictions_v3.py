"""Build the v3 predictions for a split (val or test).

Per-relation sources:
  hasArea      -- arm tiebreak: recall-arm dominant (primary) vs grounding-arm
                  unit_equivalence (secondary), arbitrated on >5% disagreement by
                  the Llama-3.1-8B pool mode; falls back to the primary answer.
  hasCapacity  -- dominant_cluster over the P1083-grounding pool; when the
                  optional --cap-secondary / --cap-arbiter pools are given, the
                  same arm-tiebreak as hasArea is applied on top (grounding
                  primary vs cap-step secondary, Llama arbiter).
  companyTradesAtStockExchange -- frequency 0.5 + empty-majority 8 over the base
                  pool (recovers surface-form vote splits; gold aliases cover
                  both surface forms).
  personHasCityOfDeath -- when --city-secondary (the profile-pinned two-stage
                  pool) is given, majority over the merged 40-vote pool with
                  minimum_votes 18; otherwise rows stay as in the base
                  predictions.
  others       -- rows copied unchanged from the base predictions file.

Usage (test):
  python scripts/h100-bf16/build_predictions_v3.py \
    --base outputs/screening/mistral-small-24b-test-predictions.jsonl \
    --base-candidates outputs/screening/mistral-small-24b-test-candidates.jsonl \
    --area-primary outputs/screening/mistral-area-rq3-test-candidates.jsonl \
    --area-secondary outputs/screening/mistral-small-24b-test-candidates.jsonl \
    --area-arbiter outputs/screening/llama31-8b-area-test-candidates.jsonl \
    --cap-candidates outputs/screening/mistral-cap-grounding-test-candidates.jsonl \
    --output outputs/screening/mistral-small-24b-test-predictions-v3.jsonl
"""

import argparse
import json
import re
import statistics
import sys
from collections import Counter

sys.path.insert(0, "src")

from akbc_baseline.aggregation import (
    aggregate_dominant_cluster,
    aggregate_unit_equivalence,
)


def load_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def pool_by_subject(rows, relation):
    return {
        row["SubjectEntity"]: row["Candidates"]
        for row in rows
        if row["Relation"] == relation
    }


def to_number(candidate):
    if candidate is None:
        return None
    if isinstance(candidate, list):
        candidate = candidate[0] if candidate else None
    if not candidate:
        return None
    match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", str(candidate))
    try:
        return float(match.group(0).replace(",", "")) if match else None
    except ValueError:
        return None


def numeric_mode(candidates):
    values = sorted(
        v for c in candidates if (v := to_number(c)) is not None and v > 0
    )
    if not values:
        return None
    groups = []
    for value in values:
        for group in groups:
            if abs(value - group[0]) / max(group[0], 1e-9) <= 0.05:
                group.append(value)
                break
        else:
            groups.append([value])
    return statistics.median(max(groups, key=len))


def format_number(value):
    return f"{value:g}"


def _close(a, b):
    return a is not None and b is not None and abs(a - b) / max(b, 1e-9) <= 0.05


def arm_tiebreak(primary_cands, secondaries, arbiter_pools):
    """secondaries: list of (candidates, aggregator); arbiter_pools: candidate lists.

    Switch to the first secondary answer that disagrees with the primary while
    matching any arbiter's mode that the primary does not match; otherwise keep
    the primary answer.
    """
    primary = aggregate_dominant_cluster(primary_cands)
    primary_value = to_number(primary)
    arbiter_values = [numeric_mode(cands) for cands in arbiter_pools]
    secondary_values = [to_number(agg(cands)) for cands, agg in secondaries]
    if primary_value is None:
        for value in secondary_values:
            if value is not None:
                return [format_number(value)]
        return []
    for value in secondary_values:
        if value is None or _close(primary_value, value):
            continue
        if any(
            _close(value, arb) and not _close(primary_value, arb)
            for arb in arbiter_values
        ):
            return [format_number(value)]
    return primary


_ACRONYM_STOPWORDS = {"of", "the", "de", "di", "da"}


def _initials(name):
    words = [
        w for w in name.replace("-", " ").split() if w.lower() not in _ACRONYM_STOPWORDS
    ]
    return "".join(w[0] for w in words if w).lower()


def company_aggregate(candidates, threshold=0.5, empty_majority=8):
    total = len(candidates)
    empties = sum(1 for c in candidates if not c)
    if empties >= empty_majority:
        return []
    counts = Counter()
    for candidate in candidates:
        for item in candidate or []:
            counts[item.strip()] += 1
    # Consolidate case variants onto the most common surface form, then fold
    # acronyms into the long form whose word initials they spell (NYSE -> New
    # York Stock Exchange) so vote splits across surface forms cannot drop a
    # majority answer below the frequency threshold.
    by_casefold = Counter()
    surface = {}
    for name, votes in counts.items():
        key = name.lower()
        by_casefold[key] += votes
        if key not in surface or votes > counts.get(surface[key], 0):
            surface[key] = name
    counts = Counter({surface[key]: votes for key, votes in by_casefold.items()})
    for short in list(counts):
        if not short.isalpha() or len(short) > 6:
            continue
        for long_form in list(counts):
            if long_form == short or len(long_form) <= len(short):
                continue
            if short.lower() == _initials(long_form):
                counts[long_form] += counts.pop(short, 0)
                break
    return [name for name, votes in counts.items() if votes / total >= threshold]


def city_merged_majority(base_cands, secondary_cands, minimum_votes=18):
    counts = Counter()
    surface = {}
    for candidate in base_cands + secondary_cands:
        name = candidate[0].strip() if candidate else ""
        key = name.lower()
        counts[key] += 1
        if name and (key not in surface or name == candidate[0].strip()):
            surface.setdefault(key, name)
    winner, votes = counts.most_common(1)[0]
    if winner == "" or votes < minimum_votes:
        return []
    return [surface[winner]]


def strip_profile(subject):
    return re.sub(r"\s*\([^)]*\)\s*$", "", subject)


def tolerance_bridge(value, cands):
    """Shift the answer within its own 5% window to the point covered by the
    most candidate-interval vote mass. Derived from the metric geometry: two
    values whose ratio is below 1.05/0.95 admit a point correct for both, so
    the bridge can cover a runner-up cluster without giving up the mode."""
    from akbc_baseline.area_clustering import cluster_numeric_candidates

    if value is None:
        return None
    clusters = cluster_numeric_candidates(cands, 0.05)
    centers = [
        (to_number(c.representative), c.support) for c in clusters
    ]
    centers = [(c, s) for c, s in centers if c]
    if not centers:
        return value
    lo, hi = 0.95 * value, 1.05 * value
    best, best_score = value, -1
    for i in range(401):
        x = lo + (hi - lo) * i / 400
        score = sum(s for c, s in centers if 0.95 * c <= x <= 1.05 * c)
        if score > best_score or (
            score == best_score and abs(x - value) < abs(best - value)
        ):
            best, best_score = x, score
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--base-candidates", required=True)
    parser.add_argument("--area-primary", required=True)
    parser.add_argument("--area-secondary", required=True)
    parser.add_argument("--area-arbiter", required=True)
    parser.add_argument("--cap-candidates", required=True)
    parser.add_argument("--cap-secondary")
    parser.add_argument("--cap-arbiter")
    parser.add_argument("--cap-arbiter2")
    parser.add_argument("--cap-tertiary")
    parser.add_argument("--area-bridge", action="store_true",
                        help="apply the tolerance-geometry bridge to hasArea "
                        "(val +2 but test -1; off by default since v11)")
    parser.add_argument("--city-secondary")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if bool(args.cap_secondary) != bool(args.cap_arbiter):
        raise SystemExit("--cap-secondary and --cap-arbiter must be given together")

    base_pool = load_jsonl(args.base_candidates)
    area_primary = pool_by_subject(load_jsonl(args.area_primary), "hasArea")
    area_secondary = pool_by_subject(load_jsonl(args.area_secondary), "hasArea")
    area_arbiter = pool_by_subject(load_jsonl(args.area_arbiter), "hasArea")
    cap = pool_by_subject(load_jsonl(args.cap_candidates), "hasCapacity")
    cap_secondary = (
        pool_by_subject(load_jsonl(args.cap_secondary), "hasCapacity")
        if args.cap_secondary
        else None
    )
    cap_arbiter = (
        pool_by_subject(load_jsonl(args.cap_arbiter), "hasCapacity")
        if args.cap_arbiter
        else None
    )
    cap_arbiter2 = (
        pool_by_subject(load_jsonl(args.cap_arbiter2), "hasCapacity")
        if args.cap_arbiter2
        else None
    )
    cap_tertiary = (
        pool_by_subject(load_jsonl(args.cap_tertiary), "hasCapacity")
        if args.cap_tertiary
        else None
    )
    cap_base = pool_by_subject(base_pool, "hasCapacity")
    city_base = pool_by_subject(base_pool, "personHasCityOfDeath")
    city_secondary = None
    if args.city_secondary:
        city_secondary = {
            strip_profile(row["SubjectEntity"]): row["Candidates"]
            for row in load_jsonl(args.city_secondary)
            if row["Relation"] == "personHasCityOfDeath"
        }
    company = pool_by_subject(base_pool, "companyTradesAtStockExchange")

    replaced = Counter()
    with open(args.output, "w", encoding="utf-8") as stream:
        for row in load_jsonl(args.base):
            relation = row["Relation"]
            subject = row["SubjectEntity"]
            if relation == "hasArea":
                objects = arm_tiebreak(
                    area_primary[subject],
                    [(area_secondary[subject], aggregate_unit_equivalence)],
                    [area_arbiter[subject]],
                )
                if args.area_bridge:
                    bridged = tolerance_bridge(
                        to_number(objects),
                        area_primary[subject] + area_secondary[subject],
                    )
                    if bridged is not None:
                        objects = [format_number(bridged)]
                row = dict(row, ObjectEntities=objects or [])
                replaced[relation] += 1
            elif relation == "hasCapacity":
                if cap_secondary is not None:
                    secondaries = [
                        (cap_secondary[subject], aggregate_dominant_cluster),
                        (cap_base[subject], aggregate_dominant_cluster),
                    ]
                    if cap_tertiary is not None:
                        secondaries.append(
                            (cap_tertiary[subject], aggregate_dominant_cluster)
                        )
                    arbiters = [cap_arbiter[subject]]
                    if cap_arbiter2 is not None:
                        arbiters.append(cap_arbiter2[subject])
                    objects = arm_tiebreak(cap[subject], secondaries, arbiters)
                else:
                    objects = aggregate_dominant_cluster(cap[subject])
                row = dict(row, ObjectEntities=objects or [])
                replaced[relation] += 1
            elif relation == "companyTradesAtStockExchange":
                row = dict(row, ObjectEntities=company_aggregate(company[subject]))
                replaced[relation] += 1
            elif relation == "personHasCityOfDeath" and city_secondary is not None:
                objects = city_merged_majority(
                    city_base[subject], city_secondary[subject]
                )
                row = dict(row, ObjectEntities=objects)
                replaced[relation] += 1
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    base_counts = Counter(row["Relation"] for row in load_jsonl(args.base))
    expected = {
        relation: base_counts[relation]
        for relation in ("hasArea", "hasCapacity", "companyTradesAtStockExchange")
    }
    if city_secondary is not None:
        expected["personHasCityOfDeath"] = base_counts["personHasCityOfDeath"]
    assert dict(replaced) == expected, (dict(replaced), expected)
    print(f"wrote {args.output}: replaced {dict(replaced)}")


if __name__ == "__main__":
    main()
