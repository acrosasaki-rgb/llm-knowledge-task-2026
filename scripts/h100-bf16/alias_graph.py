"""Public alias graph built from the organiser-provided gold alias lists.

Every gold object in train/val is an alias list for one Wikidata entity. We
union alias lists that share any surface (casefold) into one entity class per
relation, then keep only aliases that belong to exactly one class
(``degree(alias) == 1``). Candidate surfaces are folded to the class canonical
(the first alias of the first list seen, i.e. the English label) before
voting, so split votes such as ``Borsa Italiana`` / ``Euronext Milan`` /
``Italian Stock Exchange`` count for one entity, and only one surface is
emitted per entity (the evaluator treats two surfaces of one entity as an
extra prediction).

A small hand-written rename table covers renamed exchanges whose old name is
never a Wikidata alias (e.g. Jakarta Stock Exchange -> Indonesia Stock
Exchange). It is applied before the graph lookup and only when the target is
itself a known class.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

RENAMES = {
    "companyTradesAtStockExchange": {
        "jakarta stock exchange": "Indonesia Stock Exchange",
        "kuala lumpur stock exchange": "Bursa Malaysia",
        "milan stock exchange": "Borsa Italiana",
        "italian stock exchange": "Borsa Italiana",
        "bombay stock exchange": "BSE Limited",
        "paris stock exchange": "Euronext Paris",
        "paris bourse": "Euronext Paris",
        "amsterdam stock exchange": "Euronext Amsterdam",
        "brussels stock exchange": "Euronext Brussels",
        "lisbon stock exchange": "Euronext Lisbon",
        "stockholm stock exchange": "Nasdaq Stockholm",
        "helsinki stock exchange": "Nasdaq Helsinki",
        "copenhagen stock exchange": "Nasdaq Copenhagen",
        "oslo stock exchange": "Oslo Stock Exchange",
        "toronto stock exchange (tsx)": "Toronto Stock Exchange",
        "the stock exchange of hong kong": "Hong Kong Stock Exchange",
        "stock exchange of hong kong": "Hong Kong Stock Exchange",
        "sehk": "Hong Kong Stock Exchange",
        "hkex": "Hong Kong Stock Exchange",
        "nyse american": "NYSE American",
        "american stock exchange": "NYSE American",
        "amex": "NYSE American",
        "nasdaq stock market": "Nasdaq",
        "nasdaq stock exchange": "Nasdaq",
        "the nasdaq stock market": "Nasdaq",
        "korea exchange (kospi)": "Korea Exchange",
        "kospi": "Korea Exchange",
        "kosdaq": "KOSDAQ",
        "shenzhen stock exchange (szse)": "Shenzhen Stock Exchange",
        "shanghai stock exchange (sse)": "Shanghai Stock Exchange",
        "wiener börse": "Vienna Stock Exchange",
        "swiss exchange": "SIX Swiss Exchange",
        "six": "SIX Swiss Exchange",
        "börse frankfurt": "Frankfurt Stock Exchange",
        "frankfurter wertpapierbörse": "Frankfurt Stock Exchange",
        "deutsche börse": "Frankfurt Stock Exchange",
        "xetra": "Frankfurt Stock Exchange",
        "jse": "Johannesburg Stock Exchange",
        "jse limited": "Johannesburg Stock Exchange",
        "asx": "Australian Securities Exchange",
        "nzx": "New Zealand Exchange",
        "new zealand stock exchange": "New Zealand Exchange",
        "tadawul": "Saudi Exchange",
        "saudi stock exchange": "Saudi Exchange",
        "saudi stock exchange (tadawul)": "Saudi Exchange",
        "b3": "B3",
        "bovespa": "B3",
        "bm&fbovespa": "B3",
        "são paulo stock exchange": "B3",
        "bolsa mexicana de valores": "Mexican Stock Exchange",
        "moscow exchange": "Moscow Exchange",
        "micex": "Moscow Exchange",
        "tse": None,  # ambiguous: Tokyo / Toronto / Tehran
        "sse": None,  # ambiguous: Shanghai / Shenzhen (rare) / Stockholm
    },
}


class AliasGraph:
    def __init__(self, classes: dict[str, list[list[str]]]):
        # relation -> list of alias lists (each = one entity class)
        self.canon: dict[str, dict[str, str]] = {}
        self.classes = classes
        for relation, groups in classes.items():
            owner: dict[str, set[int]] = defaultdict(set)
            for idx, aliases in enumerate(groups):
                for alias in aliases:
                    owner[alias.casefold()].add(idx)
            table = {}
            for alias, idxs in owner.items():
                if len(idxs) == 1:
                    table[alias] = groups[next(iter(idxs))][0]
            self.canon[relation] = table

    @classmethod
    def from_gold(cls, *paths: str | Path) -> "AliasGraph":
        groups: dict[str, list[list[str]]] = defaultdict(list)
        for path in paths:
            for line in open(path, encoding="utf-8"):
                row = json.loads(line)
                for obj in row["ObjectEntities"]:
                    aliases = obj if isinstance(obj, list) else [obj]
                    groups[row["Relation"]].append([a for a in aliases if a])
        merged = {rel: _union(lists) for rel, lists in groups.items()}
        return cls(merged)

    def fold(self, relation: str, surface: str) -> str:
        """Return the canonical surface for ``surface`` if it is an
        unambiguous alias of a known entity (after renames); otherwise the
        surface itself, stripped."""
        s = surface.strip()
        key = s.casefold()
        table = self.canon.get(relation, {})
        renames = RENAMES.get(relation, {})
        if key in renames:
            target = renames[key]
            if target is None:
                return s
            if target.casefold() in table:  # only rename onto a known class
                return table[target.casefold()]
            return s
        return table.get(key, s)

    def known(self, relation: str, surface: str) -> bool:
        return self.fold(relation, surface).casefold() in self.canon.get(relation, {})


MIN_ALIAS_LEN = 5  # shorter aliases are acronyms (TSE, SA, PSE...) that collide


def _union(lists: list[list[str]]) -> list[list[str]]:
    """Group gold alias lists by their English label (first alias). Two lists
    are the same entity only if their labels agree; sharing a secondary alias
    is NOT evidence (multilingual lists share acronyms across entities)."""
    groups: dict[str, list[str]] = {}
    for aliases in lists:
        if not aliases:
            continue
        label = aliases[0]
        bucket = groups.setdefault(label.casefold(), [label])
        seen = {x.casefold() for x in bucket}
        for alias in aliases:
            if len(alias) >= MIN_ALIAS_LEN and alias.casefold() not in seen:
                bucket.append(alias)
                seen.add(alias.casefold())
    return list(groups.values())
