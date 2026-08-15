# awardWonBy per-name verification stage (#35)

## Problem

Test-set awardWonBy sits at P 0.320 / R 0.325 while every team above us has
higher precision (0.36-0.81) at similar recall. Vote-count filtering cannot
help: sweeping min-votes 1..10 on the 20-candidate pool raises P only to
0.36 while recall collapses (F1 monotonically falls from 0.172 to 0.102),
because hallucinated names are repeated across candidates as consistently as
real ones (correlated errors).

## Method

For each award row, collect the distinct candidate names from the baseline
20-candidate pool (910 (award, name) pairs over 10 validation rows) and ask
the model directly, no thinking, 3 votes each: "According to Wikipedia and
Wikidata, did {name} receive the award '{award}'? Consider only this exact
award." Names are then filtered by their yes-vote count
(`scripts/h100-bf16/award_verify.py`; ~2,700 short calls, minutes of GPU).

## Results (10 validation award rows, macro)

| Selection | P | R | F1 |
|---|---:|---:|---:|
| baseline freq>=1 (shipped) | 0.241 | 0.217 | 0.172 |
| **drop unanimous-no (yes>=1)** | **0.280** | 0.212 | **0.186** |
| yes>=2 | 0.297 | 0.182 | 0.179 |
| yes==3 | 0.315 | 0.158 | 0.175 |
| freq>=3 OR yes>=2 | 0.293 | 0.198 | 0.184 |

Yes-vote distribution over correct vs wrong names: yes=0 catches 230 wrong
names against only 8 correct ones; yes=3 contains 147 correct but also 187
wrong. **The verification signal is asymmetric: confident rejection is
informative, confident acceptance is not** — consistent with the
self-confirmation bias found in the thinking analyses (#27): "yes" is cheap
for a model reviewing its own outputs, but a name it rejects 3/3 times is
almost always fabricated.

## Verdict — adopted into the main line

Adopted: award F1 0.172 -> 0.186 on validation, recall cost 0.005, no other
relation touched. `scripts/h100-bf16/apply_award_verify.py` applies the
filter deterministically to any predictions file given the votes JSONL; the
deployment flow is generate votes once (`run-award-verify-docker.sh`, ~10
GPU-minutes) then filter the aggregated predictions.

Weighting caveat discovered during integration: the official "All
Relations" macro averages over rows, not relations, so the 10 award rows
carry 10/478 of the total — the filter moves the overall validation score
by only +0.0003 (0.5881 -> 0.5884). The adoption is justified by being
strictly positive and risk-free rather than by its aggregate size; the same
weighting means hasArea (100 rows) dominates the leaderboard gap analysis.
