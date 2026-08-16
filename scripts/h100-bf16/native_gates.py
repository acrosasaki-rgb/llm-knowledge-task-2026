"""Zero-downside gates from the gemma native-language / company-status
registers (run-gemma-native-container.sh output).

  city_rescue   : current [] AND >= K_LANGS Latin-script language registers
                  each have a >= K_VOTES majority for the same folded city
                  -> that city (English label via alias graph when known)
  delist_veto   : current non-empty AND the company `Status:` register says
                  acquired/merged/delisted with a year in >= K_DELIST of 20
                  samples -> []
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

LABEL = {"ru": "Место смерти:", "uk": "Місце смерті:", "pl": "Miejsce śmierci:",
         "hu": "Halálának helye:", "ro": "Locul decesului:", "de": "Sterbeort:",
         "fr": "Lieu de décès:", "es": "Lugar de fallecimiento:", "it": "Luogo di morte:",
         "fi": "Kuolinpaikka:", "tr": "Ölüm yeri:", "ja": "死没地:"}
ALIVE = {"ru": ["жив", "здравств"], "uk": ["жив"], "pl": ["żyje"], "hu": ["él"],
         "ro": ["în viață"], "de": ["lebt"], "fr": ["en vie", "vivant"],
         "es": ["vivo", "viva"], "it": ["vivo", "viva", "vivente"],
         "fi": ["elossa", "elää"], "tr": ["hayatta", "yaşıyor"], "ja": ["存命"]}
LATIN = ["pl", "hu", "ro", "de", "fr", "es", "it", "fi", "tr"]


def parse_city(lang, text):
    m = re.search(re.escape(LABEL[lang]) + r"\s*([^\n]*)", text)
    if not m:
        return None
    v = m.group(1).strip()
    if any(a in v.lower() for a in ALIVE[lang]) or v.startswith("("):
        return ""
    v = re.split(r"[,(（]", v)[0].strip().rstrip(".。")
    return v if 0 < len(v) < 60 else None


def status_delisted(text):
    t = text.lower()
    has_year = re.search(r"(19|20)\d\d", t) is not None
    event = re.search(r"(acquired by|merged (with|into)|taken private|liquidated|"
                      r"bankrupt|dissolved|defunct|delisted)", t) is not None
    return event and has_year


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--native", action="append", required=True)
    ap.add_argument("--alias-graph", nargs="*", default=[])
    ap.add_argument("--gold")
    ap.add_argument("--apply", default="city_rescue,delist_veto")
    ap.add_argument("--out")
    ap.add_argument("--k-langs", type=int, default=5)
    ap.add_argument("--k-votes", type=int, default=10)
    ap.add_argument("--k-delist", type=int, default=16)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    graph = AliasGraph.from_gold(*a.alias_graph) if a.alias_graph else None
    fold_city = (lambda x: graph.fold("personHasCityOfDeath", x)) if graph else (lambda x: x)

    regs = {}
    for path in a.native:
        for r in map(json.loads, open(path, encoding="utf-8")):
            regs.setdefault((r["Relation"], r["SubjectEntity"]), {}).update(r["registers"])
    preds = [json.loads(l) for l in open(a.pred, encoding="utf-8")]
    gold = None
    if a.gold:
        gold = {(r["Relation"], r["SubjectEntity"]): [
            {x.casefold() for x in (o if isinstance(o, list) else [o])} for o in r["ObjectEntities"]]
            for r in map(json.loads, open(a.gold, encoding="utf-8"))}

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

    apply = set(a.apply.split(","))
    fired = {"city_rescue": [], "delist_veto": []}
    out = []
    for row in preds:
        key = (row["Relation"], row["SubjectEntity"])
        cur = list(row["ObjectEntities"])
        new = cur
        rg = regs.get(key)
        if rg and key[0] == "personHasCityOfDeath" and not cur:
            support = Counter()
            surface = {}
            for lang in LATIN:
                if lang not in rg:
                    continue
                cnt = Counter(parse_city(lang, t) for t in rg[lang])
                cnt.pop(None, None)
                if not cnt:
                    continue
                top, votes = cnt.most_common(1)[0]
                if top and votes >= a.k_votes:
                    k = fold_city(top).casefold()
                    support[k] += 1
                    surface.setdefault(k, fold_city(top))
            if support:
                best, n = support.most_common(1)[0]
                if n >= a.k_langs:
                    fired["city_rescue"].append((key, cur, [surface[best]]))
                    if "city_rescue" in apply:
                        new = [surface[best]]
        if rg and key[0] == "companyTradesAtStockExchange" and cur and "status" in rg:
            d = sum(1 for t in rg["status"] if status_delisted(t))
            if d >= a.k_delist:
                fired["delist_veto"].append((key, cur, []))
                if "delist_veto" in apply:
                    new = []
        out.append({**row, "ObjectEntities": new})

    for gate, rows in fired.items():
        if gold:
            plus = sum(1 for k, c, n in rows if f1(n, gold[k]) > f1(c, gold[k]))
            minus = sum(1 for k, c, n in rows if f1(n, gold[k]) < f1(c, gold[k]))
            print(f"{gate}: fires {len(rows)}  +{plus}/-{minus}")
        else:
            print(f"{gate}: fires {len(rows)}")
        if a.verbose:
            for k, c, n in rows:
                g = f"  gold={[sorted(x)[0] for x in gold[k]][:2]}" if gold else ""
                print(f"    {k[1][:34]:34s} {str(c)[:28]:28s} -> {str(n)[:24]:24s}{g}")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("wrote", a.out)


if __name__ == "__main__":
    main()
