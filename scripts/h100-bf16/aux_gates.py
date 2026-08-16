"""Cross-family certificate gates on top of the gemma-pt composition.

Inputs
  --pred        gemma predictions (compose_gemma output; alias-folded)
  --city-pool   gemma occ-register city pool (Candidates per subject); the
                city-extra file (with DropoutCandidates) is accepted too
  --company-pool gemma company pool
  --aux         aux raw file (run-aux-probe-container output)
  --gold        optional gold file -> prints per-gate +/- on val
  --alias-graph gold files for the public alias graph (entity folding)
  --apply       comma list of gates to apply: city_veto,city_rescue,company_rescue

Gates (all zero-downside: act only when the certificate holds, otherwise
keep the current prediction)
  city_veto     : current city non-empty AND aux explicit ALIVE (lead or bio
                  majority) AND no aux explicit DEAD AND gemma identity-dropout
                  keeps the same city (nationality prior)  -> []
  city_rescue   : current [] AND aux top-1 city (register 'city') equals
                  exactly one gemma non-empty candidate entity AND aux not
                  explicit ALIVE -> that city
  company_rescue: current [] AND aux exchange top-1 is among gemma non-empty
                  candidate entities and unique -> that exchange
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from alias_graph import AliasGraph  # noqa: E402

YEAR = r"(1[6-9]\d\d|20[0-2]\d)"


def lead_status(text: str) -> str:
    t = text.strip()
    if re.match(r"^\s*(born|b\.)\s+[^)]*" + YEAR, t, re.I):
        return "ALIVE"
    if re.match(r"^\s*(?:c\.\s*)?[^)]*?" + YEAR + r"\s*[–—-]\s*[^)]*?" + YEAR, t):
        return "DEAD"
    if re.match(r"^\s*(?:c\.\s*)?" + YEAR + r"\s*[)]", t):
        return "UNKNOWN"  # bare "(1972)" — ambiguous
    if re.match(r"^\s*(?:c\.\s*)?" + YEAR + r"\s*[–—-]\s*\)", t):
        return "ALIVE"  # "(1972–)"
    return "UNKNOWN"


def bio_status(text: str) -> str:
    m = re.search(r"Died:\s*([^\n]*)", text)
    if not m:
        return "UNKNOWN"
    d = m.group(1).strip()
    if "alive" in d.lower() or d.startswith("(still") or d.lower().startswith("n/a"):
        return "ALIVE"
    if re.search(YEAR, d):
        return "DEAD"
    return "UNKNOWN"


def parse_city(text: str):
    t = text.strip()
    if not t or "alive" in t.lower() or t.startswith("("):
        return None
    city = t.split("\n")[0].split(",")[0].strip().rstrip(".")
    return city if city and "<" not in city and len(city) < 60 else None


def parse_xchg(text: str):
    t = text.strip().split("\n")[0]
    if not t or t.lower().startswith("none"):
        return []
    return [x.strip().rstrip(".") for x in t.split(";") if 0 < len(x.strip()) < 60]


def aux_summary(row, fold):
    regs = row["registers"]
    out = {}
    if "lead" in regs:
        out["lead"] = Counter(lead_status(t) for t in regs["lead"])
    if "bio" in regs:
        out["bio"] = Counter(bio_status(t) for t in regs["bio"])
    if "city" in regs:
        cnt = Counter()
        for t in regs["city"]:
            c = parse_city(t)
            cnt[fold(c).casefold() if c else ""] += 1
        out["city"] = cnt
    if "xchg" in regs:
        cnt = Counter()
        for t in regs["xchg"]:
            items = {fold(x).casefold() for x in parse_xchg(t)}
            if not items:
                cnt[""] += 1
            for x in items:
                cnt[x] += 1
        out["xchg"] = cnt
    return out


def explicit(counter: Counter, label: str, k: int) -> bool:
    return counter.get(label, 0) >= k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--city-pool", required=True)
    ap.add_argument("--company-pool", required=True)
    ap.add_argument("--aux", required=True)
    ap.add_argument("--gold")
    ap.add_argument("--alias-graph", nargs="*", default=[])
    ap.add_argument("--apply", default="")
    ap.add_argument("--out")
    ap.add_argument("--k-alive", type=int, default=12,
                    help="aux samples (of 20) that must say ALIVE explicitly")
    ap.add_argument("--k-dead", type=int, default=3,
                    help="aux DEAD samples that block a veto")
    ap.add_argument("--k-aux-city", type=int, default=8,
                    help="aux city top-1 votes required for rescue")
    ap.add_argument("--k-aux-xchg", type=int, default=10)
    ap.add_argument("--dropout-min", type=int, default=8,
                    help="dropout votes for the same city that mark a prior")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    graph = AliasGraph.from_gold(*a.alias_graph) if a.alias_graph else None
    def fold_for(rel):
        return (lambda x: graph.fold(rel, x)) if graph else (lambda x: x)

    preds = {(r["Relation"], r["SubjectEntity"]): r
             for r in map(json.loads, open(a.pred, encoding="utf-8"))}
    city_pool = {r["SubjectEntity"]: r
                 for r in map(json.loads, open(a.city_pool, encoding="utf-8"))
                 if r["Relation"] == "personHasCityOfDeath"}
    comp_pool = {r["SubjectEntity"]: r
                 for r in map(json.loads, open(a.company_pool, encoding="utf-8"))
                 if r["Relation"] == "companyTradesAtStockExchange"}
    aux = {}
    for r in map(json.loads, open(a.aux, encoding="utf-8")):
        aux[(r["Relation"], r["SubjectEntity"])] = aux_summary(r, fold_for(r["Relation"]))
    gold = None
    if a.gold:
        gold = {}
        for r in map(json.loads, open(a.gold, encoding="utf-8")):
            gold[(r["Relation"], r["SubjectEntity"])] = [
                {x.casefold() for x in (o if isinstance(o, list) else [o])}
                for o in r["ObjectEntities"]]

    def f1(pred, gs):
        P = {p.casefold() for p in pred}
        if not gs and not P:
            return 1.0
        if not gs or not P:
            return 0.0
        tp = sum(1 for g in gs if g & P)
        hitp = sum(1 for p in P if any(p in g for g in gs))
        prec, rec = hitp / len(P), tp / len(gs)
        return 0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)

    apply = set(x for x in a.apply.split(",") if x)
    changes = {"city_veto": [], "city_rescue": [], "company_rescue": []}
    new_preds = {}
    for key, row in preds.items():
        rel, s = key
        cur = list(row["ObjectEntities"])
        new = cur
        ax = aux.get(key)
        if rel == "personHasCityOfDeath" and ax:
            fold = fold_for(rel)
            lead, bio, acity = ax.get("lead", Counter()), ax.get("bio", Counter()), ax.get("city", Counter())
            alive = explicit(lead, "ALIVE", a.k_alive) or explicit(bio, "ALIVE", a.k_alive)
            dead = lead.get("DEAD", 0) + bio.get("DEAD", 0) >= a.k_dead
            pool_row = city_pool.get(s)
            if cur and alive and not dead:
                # nationality-prior check via identity dropout
                prior = True  # if no dropout data, do not require it
                if pool_row and pool_row.get("DropoutCandidates"):
                    dc = Counter(fold(c[0]).casefold() for c in pool_row["DropoutCandidates"] if c)
                    prior = dc.get(fold(cur[0]).casefold(), 0) >= a.dropout_min
                if prior:
                    changes["city_veto"].append((s, cur, []))
                    if "city_veto" in apply:
                        new = []
            if not cur and pool_row and acity:
                top, votes = max(((k, v) for k, v in acity.items() if k), key=lambda kv: kv[1], default=("", 0))
                gem = Counter(fold(c[0]).casefold() for c in pool_row["Candidates"] if c)
                surf = {}
                for c in pool_row["Candidates"]:
                    if c:
                        surf.setdefault(fold(c[0]).casefold(), fold(c[0]))
                if top and votes >= a.k_aux_city and top in gem and not alive:
                    changes["city_rescue"].append((s, cur, [surf[top]]))
                    if "city_rescue" in apply:
                        new = [surf[top]]
        if rel == "companyTradesAtStockExchange" and ax and not cur:
            fold = fold_for(rel)
            ax_x = ax.get("xchg", Counter())
            pool_row = comp_pool.get(s)
            if pool_row:
                gem = Counter(); surf = {}
                for c in pool_row["Candidates"]:
                    for x in c or []:
                        k = fold(x).casefold(); gem[k] += 1; surf.setdefault(k, fold(x))
                top, votes = max(((k, v) for k, v in ax_x.items() if k), key=lambda kv: kv[1], default=("", 0))
                if top and votes >= a.k_aux_xchg and top in gem:
                    changes["company_rescue"].append((s, cur, [surf[top]]))
                    if "company_rescue" in apply:
                        new = [surf[top]]
        new_preds[key] = {**row, "ObjectEntities": new}

    for gate, rows in changes.items():
        if gold:
            delta = sum(f1(n, gold[("personHasCityOfDeath" if gate.startswith("city") else "companyTradesAtStockExchange", s)]) -
                        f1(c, gold[("personHasCityOfDeath" if gate.startswith("city") else "companyTradesAtStockExchange", s)])
                        for s, c, n in rows)
            plus = sum(1 for s, c, n in rows if f1(n, gold[("personHasCityOfDeath" if gate.startswith("city") else "companyTradesAtStockExchange", s)]) > f1(c, gold[("personHasCityOfDeath" if gate.startswith("city") else "companyTradesAtStockExchange", s)]))
            minus = sum(1 for s, c, n in rows if f1(n, gold[("personHasCityOfDeath" if gate.startswith("city") else "companyTradesAtStockExchange", s)]) < f1(c, gold[("personHasCityOfDeath" if gate.startswith("city") else "companyTradesAtStockExchange", s)]))
            print(f"{gate}: fires {len(rows)}  +{plus}/-{minus}  net {delta:+.2f}")
        else:
            print(f"{gate}: fires {len(rows)}")
        if a.verbose:
            for s, c, n in rows:
                g = ""
                if gold:
                    rel = "personHasCityOfDeath" if gate.startswith("city") else "companyTradesAtStockExchange"
                    g = " gold=" + str([sorted(x)[0] for x in gold[(rel, s)]][:3])
                print(f"    {s[:32]:32s} {str(c)[:30]:30s} -> {str(n)[:30]:30s}{g}")

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            for r in map(json.loads, open(a.pred, encoding="utf-8")):
                f.write(json.dumps(new_preds[(r["Relation"], r["SubjectEntity"])], ensure_ascii=False) + "\n")
        print("wrote", a.out)


if __name__ == "__main__":
    main()
