# hasArea temperature arms: uniform 0.2 and a 0.2-1.1 ladder (#34)

## Question

Production QA guidance and single-shot studies favor low temperatures for
factual answers, while the self-consistency literature places vote diversity
at 0.7-0.9. Which regime serves a 20-candidate majority-vote pipeline, and
does mixing temperatures inside one pool combine their strengths? Prompt is
fixed to the grounding instruction (#29); only sampling temperature varies.

## Results (100 hasArea rows x 20 candidates)

| pool | oracle | cand-correct | unit_eq | median | dominant |
|---|---:|---:|---:|---:|---:|
| uniform 0.6 (#29 reference) | 82 | 1157 | 59 | 63 | 61 |
| uniform 0.2 | 72 | 1186 | 62 | 62 | 63 |
| ladder 0.2/0.5/0.8/1.1 x5 | 78 | — | 58 | 60 | 60 |

Ladder per-band (5 samples each): candidate accuracy 59/58/58/54%,
row-oracle 68/70/69/67, unique-row contributions 0/2/1/2.

## Conclusions

- **Low temperature sharpens the mode and shrinks the pool.** 0.2 raises
  candidate-level correctness (+29) and shortens thinking (budget-hit
  80% -> 74%) but loses 10 oracle rows; selection stays within noise. In a
  majority-vote pipeline low temperature only reinforces rows the mode
  already wins.
- **The ladder buys nothing.** Bands behave almost identically (the anchor
  distribution barely moves between 0.2 and 0.8; only 1.1 degrades), unique
  contributions are 0-2 rows, and splitting the sample budget four ways
  costs more oracle than band diversity returns (78 vs 82 at uniform 0.6).
- The shipped temperature 0.6 sits at the practical optimum for this
  pipeline. The temperature axis is closed; `candidate_temperatures` remains
  available for future mixed-sampling experiments.
