# 32B-class association screening and the Mistral adoption (#38)

## Screening (12-row broken-association probe, correct candidates /20)

| row (gold) | Qwen3.5 | Qwen3.6 | Gemma31B | GLM-Z1-32B | Mistral-24B |
|---|---:|---:|---:|---:|---:|
| Basque Country (20870) | 0 | 3 | 9 | 1 | **16** |
| Corfu (626) | 0 | 0 | 0 | 6 | 8 |
| Hopen (47) | 0 | 0 | 0 | 6 | **18** |
| Okinotorishima (0.0085) | 0 | 0 | 0 | 0 | 0 |
| Hashima (0.063) | 14 | 10 | 18 | 16 | 11 |
| Itsukushima (30.33) | 1 | 20 | 0 | 14 | 8 |
| Lake Biel (39.3) | 1 | 19 | 0 | 20 | 20 |
| Borðoy (95) | 1 | 0 | 0 | 7 | 5 |
| Vygozero (1250) | 1 | 0 | 0 | 0 | 8 |
| Lake Zaysan (1860) | 6 | 0 | 0 | 15 | 20 |
| Mafia Island (435) | 0 | 1 | 2 | 1 | 1 |
| South Uist (320) | 0 | 1 | 3 | 0 | 15 |
| any / mode correct | 6/1 | 6/2 | 4/1 | 9/4 | **11/6** |

GLM-4-32B (no-think control) and Qwen3-32B were skipped once Mistral's
margin was clear. Mistral solves every probe row except the
scope-definition one (Okinotorishima); the label→value association health
that REAP's leaderboard entry implied is confirmed directly.

## Mistral Small 3.2 24B, full validation (478 rows, tuned per-pool)

| relation | Qwen3.5 tuned | Mistral retuned | setting |
|---|---:|---:|---|
| awardWonBy | 0.172 | 0.208 | min-votes 2 |
| companyTrades | 0.769 | 0.785 | threshold 0.5, empty>=8 |
| countryLandBorders | 0.993 | 0.981 | unchanged |
| hasArea | 0.620 | **0.720** (oracle 89) | unit_equivalence |
| hasCapacity | 0.280 | 0.250 | only loss |
| personHasCityOfDeath | 0.450 | **0.530** (P 0.77) | majority, minimum_votes 10 |
| **All (row-weighted)** | **0.5881** | **≈0.622** | |

The abstention gate (`minimum_votes`) is now a first-class aggregation
parameter. Retuning magnitudes (+0.05..0.08 per relation) sit far above the
forced-weight transfer-noise band (±0.006).

## Thinking-sibling check (Magistral Small 2509)

Magistral's [THINK] is prompt-conditioned (default system prompt carries the
trigger), unlike Qwen (template-enforced via enable_thinking) or GLM-Z1
(RL-intrinsic). Replacing the system message silences it; restoring a
trigger prefix reactivates it only partially because the few-shot pairs
demonstrate bare-JSON answers. Results (hasArea 100 rows): no-think 85/69,
partial-think 86/73 vs plain Mistral 89/72 — all within the ±5 fragile-tail
band. Thinking adds nothing when the mode is already correct, closing the
"training wheels" reading of the thinking effect.

## Test artifacts

`mistral-small-24b-test-predictions.jsonl` (477 rows, sha256 51b8c527…),
generated with the retuned aggregation; structural profile matches the
validation run (city abstains on 53 rows, company empty on 42).
