# hasArea cross-candidate unit-collision filter

## Question

Test the proposed conservative unit handling on the existing 20 raw candidates:

1. Generate 20 candidates.
2. Convert each raw value under square-mile, hectare, and acre hypotheses.
3. Check whether a converted value matches a different original raw value.
4. If it does, remove only the conversion source from the original candidates.
5. Never add a converted value; aggregate only the retained original values.

All collision edges are found against the complete original candidate set before
any source is removed. The factors to square kilometers are 2.589988110336 for
square miles, 0.01 for hectares, and 0.0040468564224 for acres.

## Evaluation

The experiment reuses the first 50 `hasArea` rows and their 20 no-thinking
candidates from Job 580. No model inference is rerun. Official numeric matching
uses five-percent relative tolerance. Collision tolerances of one, two, and five
percent are reported without selecting a threshold from the gold labels.

Two aggregation rules are compared after filtering:

- dominant: representative of the largest five-percent cluster;
- median: median of all retained raw numeric candidates.

## Results

| Collision tolerance | Rows with a collision | Raw candidates removed | Dominant correct | Median correct |
|---:|---:|---:|---:|---:|
| No filter | 0 | 0 | 25/50 | 24/50 |
| 1% | 10 | 29 | 25/50 | 26/50 |
| 2% | 14 | 45 | 26/50 | 26/50 |
| 5% | 16 | 75 | 26/50 | 26/50 |

At two percent, dominant aggregation improves Hashima Island from 0.63 to the
correct 0.063 square kilometers. There are no correct-to-wrong regressions in
this 50-row shard. At one percent the two changed dominant predictions are both
wrong-to-wrong, while at five percent four of the five changed dominant
predictions are wrong-to-wrong.

Median aggregation improves two rows at every tested tolerance: Hashima Island
and Icaria. This raises its score from 24/50 to 26/50 without a regression.

At five percent, the filter finds 47 square-mile and 28 hectare source edges;
it finds no acre edge. Although 75 of 1,000 raw candidates are removed, the
dominant output changes on only five rows because most removed values were not
decisive for the winning cluster.

## Interpretation

This experiment tests the intended collision filter, unlike the earlier unit
expansion experiment that inserted every converted value into the selectable
hypothesis set. The conservative filter avoids the large degradation caused by
those added alternatives and produces a small gain on this shard.

The filter detects a unit inconsistency, not which fact has the correct scope.
For example, it recognizes square-mile-like relations among Rùm candidates, but
removing only the individually matching sources still leaves another wrong
cluster dominant. Hopen, Vasilyevsky Island, and Ayon Island similarly change to
different wrong values at five percent. Therefore the result supports retaining
the filter as a modest preprocessing signal, but not treating a collision as a
standalone correctness proof. The 2% threshold is the least aggressive tested
setting that improves dominant aggregation, but the 50-row result is too small
to establish it as a final threshold.

## Reproduction

Run `akbc_baseline.area_unit_filter_rerank` once per collision tolerance, then
score its filtered-dominant and filtered-median JSONL files with the official
evaluator. The detailed candidates output records the source index, assumed
unit, converted value, matched target index and value, and relative error for
every collision. It also records `converted_values_added: false` so this method
cannot be confused with selectable unit expansion.
