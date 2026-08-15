# Multi-route elicitation for the string relations (#31)

## Scope

Applies the #30 multi-route method (4 independently prompted reasoning
routes × 5 samples, raw thinking persisted) to the four string relations —
awardWonBy (10 rows), countryLandBordersCountry (68), 
companyTradesAtStockExchange (100), personHasCityOfDeath (100); 278 rows,
5,560 generations. Sampling, seeds, few-shot, and the thinking budget match
the tuned baseline; the only variable is the per-candidate route
instruction.

## Routes (relation-specific)

| Relation | A | B | C | D |
|---|---|---|---|---|
| awardWonBy | direct recall | field/period cues | decade walk (backward in time) | check notable figures of the field against the exact award |
| countryLandBorders | direct | region/shape cues | border walk (trace boundary clockwise) | list all plausible neighbors, drop maritime |
| companyTrades | direct | HQ/sector/ticker cues | peer companies' exchanges, then verify | primary listing from incorporation, then cross-listings; empty if unlisted |
| personHasCityOfDeath | direct | occupation/era/death-year cues | trace final years to the place of death | recall the obituary/encyclopedia phrasing "died in ..." |

FinalSelection uses the tuned aggregation (award frequency 0.05, company
0.6 + empty_majority 10, borders 0.4 + 3, city majority explicit-empty-only)
so the run doubles as a deployment-style measurement; route-level and
cross-route analyses are offline reaggregations over the saved candidates.

## Motivation from #27–#30

The numeric arm showed route diversity broadens pools but dilutes majority
voting. The string relations differ in the relevant regime: awardWonBy is
recall-bound (oracle-recall 0.217 at n=20, sets averaging 146 entities) and
enumeration order plausibly gates what surfaces, so decomposition routes act
on the *coverage* axis rather than the value-selection axis; borders and
company are precision-shaped, where verification-style routes may reduce
over-generation; city-of-death tests whether cue/timeline routes shift the
empty-vs-value decision.

## Results

Pending — will be added after the generation run.
