# hasArea unit-native recall prompt (#28)

## Hypothesis

From #27: correct candidates recall the figure in its source-native unit
(hectares for sub-km² subjects) and convert, while wrong ones recall a km²
surface form with a misplaced decimal. Instructing the model to recall in
the source unit first, then convert, should fix the digit-error class.

## Method

`configs/experiment-qwen-3.5-27b-bf16-thinking-empty-aware-20-area-unit-recall.yaml`
adds a hasArea-only system instruction ("recall in the source unit, then
convert; keep the decimal exact"). All 100 hasArea validation rows × 20
candidates, BF16, sampling/seeds/few-shot/thinking budget identical to the
baseline pool. Raw thinking persisted.

## Results (correct rows / 100)

| Selector | baseline pool | unit-recall pool |
|---|---:|---:|
| oracle (any correct among 20) | 78 | 76 |
| unit_equivalence (shipped) | 62 | 58 |
| dominant_cluster | 62 | 63 |
| median | 62 | 64 |
| candidate-level correct | 1181/2000 | 1174/2000 |

The targeted class is fixed: Hashima 8/20 → 15/20 correct candidates, with
17/20 thinkings now taking the hectare path. But the instruction *moves*
errors instead of removing them:

- **Unit re-attribution**: the correct numeral is relabeled and converted
  away (Saint Kitts: "261 square miles" → 261 × 2.589988 = 675.99; the true
  area is 261 km²). 2/20 votes.
- **Conversion twins**: independently recalled alternates that sit exactly at
  a unit-conversion of the majority value (Queimada Grande 0.43 vs 1.13).
- **Decimal-shift pairs**: fabricated self-consistent pairs like Lake
  Zaysan's "18,480 km² (7,140 sq mi)" (gold 1,860).

## Aggregation defect exposed (applies to the baseline too)

`unit_equivalence` transfers support along `source × factor ≈ target`
edges, always treating the source as the wrong unit. When the majority
cluster is correct and a 1-vote conversion twin exists, the majority's votes
are transferred onto the twin: unit-recall pool Queimada (19 × 0.43 loses to
1 × 1.13) and Saint Kitts (18 × 261 loses to 675/676); baseline Maui
(19 × 1883 loses to 1 × 4909). Redirecting the merge toward the
higher-support cluster scores 63 on the baseline pool (Maui fixed, no
regressions) and 62 on this pool.

## Conclusion

Do not ship the prompt (oracle −2, net gain within noise). Fix the transfer
direction in `unit_equivalence` instead — a pure reaggregation gain. For the
paper this is a clean negative: elicitation-side unit guidance relocates the
error mass rather than shrinking it.
