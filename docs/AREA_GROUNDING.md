# hasArea Wikidata/Wikipedia grounding prompt (#29)

## Basis

The dataset README states that hasArea gold values are the Wikidata
preferred-rank `P2046` value converted to km² (and hasCapacity is `P1083`).
Three residual error classes from #27/#28 map onto that provenance:

1. **Referent choice** — Basque Country: the model writes the gold Euskal
   Herria figure inside its thinking 20/20 times, then answers with the
   autonomous community. The gold referent is the Wikidata entity carrying
   the subject's exact label.
2. **Surface-form recall** — Hashima: sub-km² areas appear in sources as
   ha/m², so km²-form recall drops digits.
3. **Over-conversion** — the #28 instruction made the model hypothesize a
   unit for a remembered number and convert correct km² values away
   (Saint Kitts 261 → 675.99).

## Method

`configs/experiment-qwen-3.5-27b-bf16-thinking-empty-aware-20-area-grounding.yaml`
instructs: the ground truth is the Wikidata P2046 value (as shown in the
Wikipedia infobox) *for the entity with exactly this name*; recall that
stored value in its original unit; convert **only if** the unit is not
already km². All 100 hasArea rows × 20 candidates, raw thinking persisted,
sampling otherwise identical to the baseline pool.

Runs on a multi-GPU host via `scripts/h100-bf16/run-area-grounding-docker.sh`:
one llama-server per GPU (the BF16 27B fits on one 80 GB device), 8 slots and
8 shard clients per server — 64 concurrent streams on an 8-GPU machine.

## Results (correct rows / 100)

| Pool | oracle | unit_eq | dominant | median | candidate-level |
|---|---:|---:|---:|---:|---:|
| baseline | 78 | 62 | 62 | 62 | 1181/2000 |
| unit-recall (#28) | 76 | 58 | 63 | 64 | 1174/2000 |
| **grounding (#29)** | **82** | 59 | 61 | 63 | 1157/2000 |

The grounding prompt produces the **best pool so far** (oracle 78 → 82) but
no selector captures the gain: candidate-level correctness *drops*
(1181 → 1157), i.e. correct values now exist in more rows but as smaller
minorities, widening the selection gap (82 − 63 = 19 rows).

Per class: Hashima recovers as in #28 (9 → 14/20 correct); Mainland
(Shetland) improves 2 → 9/20; Maui reaches 20/20. The referent cue fails:
Basque Country stays 0/20 (autonomous community chosen every time despite
the "entity with exactly this name" instruction), Corfu and Okinotorishima
stay 0/20.

## Cross-prompt consensus (reaggregation only)

Treating the three pools (baseline / unit-recall / grounding) as three
"routes" of 20 samples and applying `route_consensus` (#30) over the merged
60 candidates:

| Selector over 60 candidates | correct |
|---|---:|
| union oracle | 84 |
| median | 64 |
| dominant_cluster | 62 |
| route_consensus (log_threshold 0.05) | 64 |
| route_consensus (log_threshold 0.10) | **65** |

**+3 over the shipped 62 with zero additional GPU cost.** Prompt diversity
behaves like route diversity: each instruction shifts which error class
dominates, and cross-prompt agreement recovers rows that no single pool's
majority vote can. The remaining 19-row gap to the union oracle is the
target of the dedicated multi-route experiment (#30).
