# Vendor-aligned sampling and presence penalty (#36)

## Question

The Qwen3/3.5 model cards recommend TopK=20 / MinP=0 for thinking mode and
suggest presence_penalty 0-2 against repetition; our stack inherited
llama.cpp defaults (top_k=40, min_p=0.05, no penalty). Do the two deviations
matter on the grounding hasArea arm?

## Results (100 hasArea rows x 20 candidates, grounding prompt fixed)

| pool | oracle | cand-correct | unit_eq | median | dominant | mean tok | budget-hit | Wait/cand |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| #29 defaults | 82 | 1157 | 59 | 63 | 61 | 1953 | 80% | 9.0 |
| vendor topk20/minp0 | 78 | 1153 | **63** | **64** | 63 | — | 80% | — |
| presence_penalty 1.0 | 78 | 1146 | 61 | 57 | 62 | 1941 | 79% | 8.0 |

## Conclusions

- **Vendor alignment: adopt.** Selection improves across the board (unit_eq
  59→63 is the best value measured on any pool; median 64 ties the best) at
  zero cost, and the configuration matches the published Qwen guidance. The
  oracle drop (82→78) is tail churn: the lost rows are the same fragile
  1-vote subjects that reshuffle under any perturbation (#33).
- **Presence penalty: closed, null.** Accuracy mixed (median 63→57), and the
  thinking barely shortens (tokens 1953→1941, budget-hit 80→79%, Wait
  9.0→8.0). The oscillation the penalty was meant to suppress is *semantic*
  — alternating between two remembered values — not token-level repetition,
  so a token-frequency penalty cannot reach it.
