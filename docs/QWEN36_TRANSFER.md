# Qwen3.6-27B generation transfer on hasArea (#37)

## Design

Model-only A/B against the #29 grounding arm: unsloth/Qwen3.6-27B-GGUF
@82d411a (BF16), identical prompt, sampling (0.6/0.95), seeds, budget,
100 hasArea rows x 20 candidates. Pre-registered outcomes: oracle above the
82+-5 band = associations repairable by generation update; inside the band =
associations persist. The measured result is a third outcome.

## Results

| | Qwen3.5 (#29) | Qwen3.6 |
|---|---:|---:|
| oracle | 82 | 78 (+7 rows / -11 rows) |
| unit_eq / median / dominant | 59 / 63 / 61 | 60 / 58 / 59 |
| candidate-level correct | 1157 | 1086 |

Association churn, not repair:

- Fixed: Itsukushima 0/20 -> 20/20 (the 70.x anchor vanished), Lake Biel
  2/20 -> 19/20, Basque Country loosened (7234 x17, gold-range values appear
  as answers for the first time). Four of the seven newly solved rows are
  exactly the rows only Gemma could solve — broken associations are
  training-data artifacts and move when the data moves.
- Newly broken: 11 rows (Borðoy, Vygozero, Lake Zaysan, Yell, Filicudi, …);
  Hashima gains a new digit-error variant (0.0063 x8); Corfu grows more
  confidently wrong (592.9 x10).
- Cross-generation union oracle: 3.5-grounding ∪ 3.6 = **89/100**
  (3.5 all-pools ∪ 3.6 = 90/100).

## Conclusion

A generation update rotates which label→value associations are broken; the
single-model inference-time ceiling (oracle 82+-5) does not move. The
knowledge exists collectively across model generations (union 89-90), but
the closed-book single-model regime cannot reach it. Deployment stays on
Qwen3.5; the churn result becomes the closing claim of the paper
(paper/paper-memo.md §6).
