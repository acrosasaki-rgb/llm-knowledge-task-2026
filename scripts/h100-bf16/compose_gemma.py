"""Compose gemma-pt predictions from candidate pools (val or test).

Usage:
  python scripts/h100-bf16/compose_gemma.py --rows data/val.jsonl \
      --pool outputs/screening/pool.jsonl [--pool ...] --out preds.jsonl \
      [--alias-graph train.jsonl val.jsonl] [--rename-map map.json]

Aggregations are the val-validated set behind v12: area/capacity =
dominant cluster; city = surface majority >= 12 on the occ register (v18; 8 before); award =
frequency >= 0.1; borders = frequency >= 0.3 with empty-majority 10; company
= casefold + acronym folding, frequency >= 0.5, empty-majority 12. With
--alias-graph, string surfaces are folded onto public gold entity classes
before voting (see alias_graph.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from akbc_baseline.aggregation import aggregate_dominant_cluster  # noqa: E402
from build_predictions_v3 import _initials  # noqa: E402
from alias_graph import AliasGraph  # noqa: E402


def norm(s: str) -> str:
    return s.strip().casefold()


def surface_city(cands, fold):
    cnt, surf = Counter(), {}
    for c in cands:
        if not c:
            cnt[""] += 1
            continue
        s = fold(c[0])
        k = norm(s)
        cnt[k] += 1
        surf.setdefault(k, s.strip())
    w, v = cnt.most_common(1)[0]
    return [surf[w]] if (w != "" and v >= 12) else []


def freq_multi(cands, th, fold, em=99):
    n = len(cands)
    if sum(1 for c in cands if not c) >= em:
        return []
    cnt, surf = Counter(), {}
    for c in cands:
        seen = set()
        for x in c or []:
            s = fold(x)
            k = norm(s)
            if k in seen:  # one vote per sample per entity
                continue
            seen.add(k)
            cnt[k] += 1
            surf.setdefault(k, s.strip())
    return [surf[k] for k, v in cnt.items() if v / n >= th]


def company_agg(cands, fold):
    n = len(cands)
    if sum(1 for c in cands if not c) >= 12:
        return []
    counts = Counter()
    for cand in cands:
        seen = set()
        for item in cand or []:
            s = fold(item).strip()
            if s.casefold() in seen:
                continue
            seen.add(s.casefold())
            counts[s] += 1
    cf, surface = Counter(), {}
    for k, v in counts.items():
        key = k.casefold()
        cf[key] += v
        if key not in surface or v > counts.get(surface[key], 0):
            surface[key] = k
    counts = Counter({surface[k]: v for k, v in cf.items()})
    for short in list(counts):
        if not short.isalpha() or len(short) > 6:
            continue
        for lf in list(counts):
            if lf == short or len(lf) <= len(short):
                continue
            if short.casefold() == _initials(lf):
                counts[lf] += counts.pop(short, 0)
                break
    return [k for k, v in counts.items() if v / n >= 0.5]


def load_pools(paths, rename):
    pools = {}
    for path in paths:
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            rel = r["Relation"]
            s = rename.get(f"{rel}|{r['SubjectEntity']}", r["SubjectEntity"])
            pools.setdefault((rel, s), r["Candidates"])
    return pools


def compose(rows, pools, graph):
    out, missing = [], []
    for r in rows:
        rel, s = r["Relation"], r["SubjectEntity"]
        cands = pools.get((rel, s))
        fold = (lambda x, rel=rel: graph.fold(rel, x)) if graph else (lambda x: x)
        if cands is None:
            missing.append((rel, s))
            obj = []
        elif rel in ("hasArea", "hasCapacity"):
            obj = aggregate_dominant_cluster(cands) or []
        elif rel == "personHasCityOfDeath":
            obj = surface_city(cands, fold)
        elif rel == "awardWonBy":
            obj = freq_multi(cands, 0.1, fold)
        elif rel == "countryLandBordersCountry":
            obj = freq_multi(cands, 0.3, fold, em=10)
        else:
            obj = company_agg(cands, fold)
        out.append({"SubjectEntity": s, "Relation": rel, "ObjectEntities": obj})
    return out, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--pool", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alias-graph", nargs="*", default=None,
                    help="gold files to build the public alias graph from")
    ap.add_argument("--rename-map", default=None,
                    help="json {relation|old_subject: new_subject}")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.rows, encoding="utf-8")]
    rename = json.load(open(a.rename_map, encoding="utf-8")) if a.rename_map else {}
    pools = load_pools(a.pool, rename)
    graph = AliasGraph.from_gold(*a.alias_graph) if a.alias_graph else None
    out, missing = compose(rows, pools, graph)
    if missing:
        print("WARNING missing pool rows:", missing, file=sys.stderr)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {a.out}: {len(out)} rows, {len(missing)} missing")


if __name__ == "__main__":
    main()
