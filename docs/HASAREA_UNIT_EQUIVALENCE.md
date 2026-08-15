# hasArea cluster-level unit equivalence

## Method

The selector takes the twenty saved `hasArea` candidates and never calls the
model again. It is deterministic.

1. Parse a number from each candidate and build greedy dense clusters at five
   percent relative tolerance. A cluster's representative is the median of its
   members and its support is the member count.
2. Multiply each cluster representative by every unit factor. When the converted
   value lands within five percent of a different cluster representative, record
   an equivalence edge from the source cluster to the target cluster. Only the
   closest target is kept per source and unit.
3. Drop any pair that points at each other in both directions, because the
   conversion direction is then undetermined.
4. Process the edges in ascending error and move the source cluster's support to
   the end of its conversion chain. Nothing is removed and no converted value
   becomes selectable.
5. Emit the representative of the cluster with the greatest merged support.

## Unit factors and the digit-error boundary

Only factors that are not powers of ten are used:

| Unit | Factor to square kilometres |
|---|---:|
| square mile | 2.589988110336 |
| acre | 0.0040468564224 |

Hectare (0.01) is deliberately excluded. A power-of-ten factor cannot be told
apart from a misplaced decimal point, so accepting it as unit evidence moves
support onto the wrong decade. `validate_factors` rejects any power-of-ten
factor, and `test_decade_related_clusters_are_not_merged` pins the behaviour.
Order-of-magnitude mistakes are a separate category that this method does not
attempt to resolve.

## Why support transfer rather than removal

The earlier filter in `area_unit_filter.py` matched individual candidates and
removed the conversion source. Only the candidates that happened to convert
within tolerance disappeared, so the remaining members of the same mistaken
cluster kept it dominant.

Rùm is the case that separates the two. Its clusters are 37.65 with fourteen
candidates and 102.12 with three. Because 37.65 square miles is 97.51, which is
4.72 percent from 102.12, the whole cluster is one reading of the other. Support
transfer moves fourteen votes and 102.12 wins with seventeen, matching the gold
value of 104. Candidate-level removal deletes only 38.6 and 38.7, leaving twelve
votes behind and the wrong cluster in front.

The five percent edge tolerance is required. At one or two percent the Rùm edge
is never created.

## Validation results

Applied to the twenty no-thinking candidates from Job 407 without rerunning
inference. Other relations are unchanged.

| Selector | hasArea correct |
|---|---:|
| median (previous baseline) | 44/100 |
| dominant cluster, no unit handling | 49/100 |
| cluster unit equivalence with support transfer | 50/100 |
| rows whose twenty candidates contain a correct value | 69/100 |

| Relation | median | unit equivalence |
|---|---:|---:|
| awardWonBy | 0.1089 | 0.1089 |
| companyTradesAtStockExchange | 0.7252 | 0.7252 |
| countryLandBordersCountry | 0.9832 | 0.9832 |
| hasArea | 0.4400 | 0.5000 |
| hasCapacity | 0.1600 | 0.1600 |
| personHasCityOfDeath | 0.4100 | 0.4100 |
| All Relations | 0.5052 | 0.5177 |

Six rows improve and none regress: Rùm, Icaria, Flinders Island, Seram, Lake
Bohinj and Markermeer. Five of the six come from preferring the dominant cluster
over the median; the unit equivalence itself accounts for Rùm alone. Equivalence
edges were found on thirty of the hundred rows.

## Known limitation

On Vygozero a chain of spurious square-mile edges collected four votes onto the
lone candidate 390, which then became dominant. Both the previous and the new
prediction are wrong, so the official score is unaffected, but linking clusters
with very low support can merge unrelated numbers by coincidence. A minimum
source support or a chain-length limit should be evaluated before this method is
applied to a relation with sparser candidates.

Of the fifty rows still wrong, nineteen have a correct value among the twenty
candidates and thirty-one have none. The ceiling for any post-selection method on
this candidate set is therefore 69/100.

## Reproduction

```
python -m akbc_baseline.reaggregate \
  --config configs/experiment-qwen-3.5-27b-mtp-relation-aware-20-no-thinking-unit-equivalence.yaml \
  --candidates <candidates-val.jsonl> \
  --input .cache/dataset2026-latest/data/val.jsonl \
  --output outputs/<system>-val.jsonl
```

`reaggregate` reruns aggregation over saved candidates and emits the official
three-key JSONL. It refuses strategies that need model assistance unless
`--reuse-final-selection` is given, and it verifies row count, order,
`SubjectEntity` and `Relation` against the official input when `--input` is
supplied.
