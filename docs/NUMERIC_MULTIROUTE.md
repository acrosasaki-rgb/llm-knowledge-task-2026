# Multi-route cross-consensus for numeric relations (#30)

## Idea

Sampling the same prompt 20 times measures the stability of one recall path;
a wrong anchor repeated 20 times still wins every vote (#27). Instead,
generate candidates through **four independently prompted reasoning routes**
(5 samples each) and prefer the value cluster supported by the most
*distinct routes*, treating cross-route convergence as stronger evidence
than within-route repetition. Inference-time only: no fine-tuning, no
learned verifier.

## Routes (candidate_instructions, contiguous blocks of 5)

- **A direct recall** — retrieve the memorized value, no derivation.
- **B cued recall** — recall 2–4 relevant facts as retrieval cues first.
- **C comparative estimation** — infer from reference entities and relative
  ordering; do not repeat a memorized answer.
- **D independent estimation** — Fermi-style, from dimensions/scale/ratios
  with an order-of-magnitude sanity check.

Routes share no conversation state; per-candidate seeds differ as usual.
Route identity of candidate *i* is `i // samples_per_route`.

## Aggregation: `route_consensus`

Numeric candidates are mapped to log10. Clusters form by distance to the
cluster's log-median (≤ `log_threshold`, default 0.05 ≈ ×1.12) — assignment
against the median stops single-linkage chains (100→110→121→133). The
winning cluster is chosen lexicographically:

1. most distinct supporting routes,
2. most samples,
3. smallest log-space MAD,
4. deterministic tie-break.

The final answer is the value median of the winning cluster. Unit tests:
`tests/test_route_consensus.py`.

## Experiment

All 200 numeric validation rows (hasArea + hasCapacity) × 20 candidates on
the 8-GPU harness (`scripts/h100-bf16/run-numeric-multiroute-docker.sh`),
raw thinking persisted. Planned ablations, all pure reaggregations of the
same pool:

1. 20-candidate median (route-mixed),
2. log clustering with sample-count priority,
3. full route-priority selection (this method),
compared against the single-prompt baseline pool (hasArea median 62,
hasCapacity dominant_cluster 28) and per-route accuracy/oracle breakdowns.

## Results (correct rows)

| Selector | hasArea /100 | hasCapacity /100 |
|---|---:|---:|
| oracle (any of 20) | 79 | **58** |
| route_consensus | 60 | 23 |
| dominant_cluster | 61 | 23 |
| median | 59 | 21 |
| shipped tuned baseline | 62 | 28 |

Per-route breakdown (candidate accuracy / row-oracle over the route's 5
samples):

| Route | hasArea | hasCapacity |
|---|---|---|
| A direct | 60% / 67 | 21% / 39 |
| B cued | 58% / 71 | 21% / 36 |
| C compare | 57% / 64 | 18% / 33 |
| D fermi | 60% / **75** | 21% / 39 |

**Within-run route diversity does not beat the single-prompt tuned
configuration.** The estimation routes broaden the pool tail (hasCapacity
oracle 55 → 58; route D alone reaches row-oracle 75 on hasArea) but add more
diverse noise than correct votes, so every selector lands below the shipped
62/28. Cross-route support also cannot repair rows whose direct-recall
association is stably wrong: an estimation route that lands near the truth
contributes 1–2 votes, while the wrong anchor keeps 10+.

Pool-merging comparison (routes = pools of 20, `route_consensus`):

- hasArea, 4 pools (baseline + #28 + #29 + this): union oracle 84,
  consensus/median 64 — the 3 direct-recall pools alone remain best (65).
- hasCapacity, 2 pools (baseline + this): union oracle 55 → **65**, but all
  selectors ≤ 26 vs the shipped 28.

Conclusion: prompt-diverse *direct-recall* pools (#29) are the productive
axis of diversity; adding estimation routes dilutes selection faster than it
enriches the pool. The selection-information limit observed in #27–#29
(minority-correct candidates indistinguishable from minority-wrong) applies
unchanged.
