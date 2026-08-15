# hasArea metadata extraction and final judge

## Hypothesis

The candidate-by-candidate conversation audit in Issue #13 retained 81.3% of
wrong candidates and did not improve the first-50 score. Asking the same
conversation whether its own answer should be kept creates a self-confirmation
failure. This experiment separates description from judgment.

## Generation flow

For each validation row, Qwen3.5-27B first generates the same 20 independent
non-thinking numeric candidates as the baseline. Each candidate conversation
then receives one metadata-only follow-up. It returns:

- the unchanged previous value;
- inferred underlying unit;
- inferred area scope;
- inferred reference year;
- inferred value normalized to square kilometers.

The metadata prompt explicitly forbids `keep`, correctness, validity,
confidence, and recommendation fields. Its result is stored even when the
value, unit, conversion, scope, or year is internally inconsistent. No
candidate is filtered at this stage.

After all 20 metadata turns, a new independent prompt receives all 20 original
answers and all parsed attribute estimates. It is asked to choose one final
square-kilometer value while comparing:

1. original answer versus the metadata's copied value;
2. inferred unit versus normalized square-kilometer arithmetic;
3. inferred scope versus the requested total-area or lake-surface scope;
4. current/general versus historical or dated values;
5. agreement across independent candidates.

The final judge is told that attributes are unverified, must not be treated as
retention verdicts, and that numerical majority is not sufficient when its
attributes are inconsistent. It must select a value supported directly or by
a unit conversion from at least one candidate rather than retrieve an
unrelated new fact. Reasoning remains disabled to preserve the existing
no-thinking comparison.

Each row therefore uses 41 model generations: 20 initial values, 20 attribute
estimates, and one final judge. If the final answer is empty, non-numeric, or
contains multiple values, the system records the failure and falls back to the
raw 20-candidate median.

## Validation boundary

The dedicated validation-only child pipeline runs the first 50 `hasArea` rows,
saves raw candidates, metadata responses, final-judge response and generation
diagnostics, and blocks at a manual artifact-review gate. Only after approval
does it run offsets 50--99 and merge an exact 100-row slice. It defines no test
inference, selection manifest, or deployment job.

## Measurements

- metadata parse and token-limit rates;
- unit, scope, and reference-year distributions;
- copied-value and unit-conversion logical consistency;
- final-judge parse/fallback rates;
- final-judge Macro F1 versus raw median and largest 5% numeric cluster;
- per-row improvements and regressions, especially where a wrong numerical
  majority conflicts with inferred attributes.

## First-50 smoke result

GPU job 580 completed offsets 0--49 in 1,941.0 seconds with 17.26 GiB peak
CUDA memory. All 1,000 initial candidates, 1,000 metadata turns, and 50 final
judge turns completed without an empty answer or token-limit failure.

All 1,000 metadata arrays parsed. Their distributions were:

- unit: `square_kilometer` 1,000;
- scope: total geographic 675, water surface 112, land only 202,
  administrative subarea 4, historical 7;
- reference year: current 756, general 184, unknown 54, and 6 malformed
  `historical_area` values in the year position.

Every copied value matched its original candidate and every reported
normalization was arithmetically consistent, because the metadata estimator
classified every candidate as already expressed in square kilometers. Thus
this shard contains scope and year conflicts but no inferred unit or conversion
conflict for the final judge to resolve.

The final judge produced a JSON array on all 50 rows. Three arrays contained a
candidate record object rather than one numeric value; the guarded raw-median
fallback handled Tinos, Ile Saint-Paul, and Folegandros. The official and
same-candidate comparisons were:

| Selector | Correct rows | Macro F1 |
| --- | ---: | ---: |
| Raw 20-candidate median | 24/50 | 0.48 |
| Metadata-assisted final judge | 25/50 | 0.50 |
| Post-hoc largest 5% cluster | 25/50 | 0.50 |

The judge changed 21 raw-median values: five already-correct rows remained
correct, fifteen wrong rows remained wrong, Icaria improved, and no row
regressed. Icaria has only 8/20 correct candidates; the judge selected an exact
candidate value of 250 km2, while the largest-cluster median selected 252.8 km2.
Both are within five percent of the 255.32 km2 ground truth. The chosen 250
records were labeled total geographic area/current, but the same row's numeric
5% clustering succeeds without metadata.

Across all rows, the judge selected an exact original candidate 50/50 times and
selected inside the largest 5% cluster on 45/50 rows. Its correct-row set is
identical to the numeric cluster selector: neither method has a unique correct
row. Therefore the one-row improvement demonstrates that a separate judge can
recover a correct minority cluster without regressions on this shard, but does
not demonstrate that the extracted attributes caused the recovery. The
metadata was anchored to the original square-kilometer question and supplied
no unit-conversion signal.

Initial candidate arrays match the preceding conversation-audit smoke on all
50 rows and the original baseline on 48/50 rows; the two minor samples do not
change any selector outcome. The manual gate for offsets 50--99 remains closed
pending review. Running the second shard is reasonable to measure whether the
judge separates from the cheaper numeric cluster on all 100 rows, but the
first-50 evidence does not yet justify replacing the cluster selector.
