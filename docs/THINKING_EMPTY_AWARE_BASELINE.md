# Thinking baseline with empty-aware aggregation

`experiment-qwen-3.5-27b-mtp-thinking-empty-aware.yaml` is the current best
validation configuration. It is the shared-prompt thinking configuration
(`experiment-qwen-3.5-27b-mtp-thinking.yaml`, official Macro F1 0.5593) with
only the aggregation block changed; generation is identical, so its existing
candidates can be re-scored without new inference.

## What changed and why

| Relation | Before | After | Reason |
|---|---|---|---|
| companyTradesAtStockExchange | frequency 0.4 | + `empty_aware`, `empty_majority: 3` | Plain frequency cannot elect the empty set: three explicit `[]` answers out of five were outvoted by two value answers. 36 of the 100 validation golds are empty. |
| countryLandBordersCountry | frequency 0.4 | + `empty_aware`, `empty_majority: 3` | Same rule as the no-thinking baseline; no rows change on Job 309 candidates. |
| personHasCityOfDeath | majority | + `explicit_empty_only` | Parse failures no longer count as empty votes; no rows change on Job 309 candidates. |
| hasArea | median | `unit_equivalence` | The cluster-level unit filter also helps the five thinking candidates (0.58 to 0.59). |
| awardWonBy | frequency 0.2 | unchanged | 0.2 beats 0.4 on the thinking arm (0.1487 vs 0.1111). |

The prediction changes on Job 309 candidates come almost entirely from five
companyTradesAtStockExchange rows where three or more of the five candidates
explicitly answered `[]`: three gold-empty rows are fixed, one correct answer
(Questerre Energy) is lost, one wrong answer changes to a different wrong
answer.

## Validation scores (Job 309 candidates, official evaluator)

| Relation | Macro F1 |
|---|---:|
| awardWonBy | 0.1487 |
| companyTradesAtStockExchange | 0.7452 |
| countryLandBordersCountry | 0.9949 |
| hasArea | 0.5900 |
| hasCapacity | 0.2200 |
| personHasCityOfDeath | 0.4600 |
| **All Relations** | **0.5662** |

The previous best submitted score was 0.5593 with the same candidates and the
naive aggregation.

## Reproduction

```
python -m akbc_baseline.reaggregate \
  --config configs/experiment-qwen-3.5-27b-mtp-thinking-empty-aware.yaml \
  --candidates <job309 qwen3.5-27b-mtp-thinking-candidates-val.jsonl> \
  --input <dataset>/data/val.jsonl \
  --output outputs/qwen3.5-27b-mtp-thinking-empty-aware-val.jsonl
```

A fresh inference run through the standard validation child pipeline
(`experiment:model:qwen3.5-27b-mtp-thinking-empty-aware`) reproduces the
candidates themselves; it costs about 21 GPU hours.
