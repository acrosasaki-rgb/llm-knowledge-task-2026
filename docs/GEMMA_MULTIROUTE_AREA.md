# Gemma 4 31B thinking multi-route check on hasArea (#32)

## Question

Are the reasoning habits measured on Qwen3.5-27B (#27–#30) — the
near-universal "commonly cited: X" retrieval template, the single-shot
anchor, and the confabulated verification ritual — properties of LLM recall
in general, or artifacts of Qwen's trained thinking style?

## Setup

`unsloth/gemma-4-31B-it-GGUF` @ `c1ac76e` (unquantized BF16 split GGUF),
thinking enabled. hasArea 100 rows × 20 candidates with the exact #30
multi-route candidate instructions, sampling, seeds, and budget. Runs on the
8-GPU harness (`scripts/h100-bf16/run-gemma-multiroute-area-docker.sh`).
Gemma's reasoning markup is `<|channel>thought … <channel|>` (not
`</think>`), so analysis re-extracts answers from `raw_text` offline; the
stored candidates disagreed with re-extraction in only 9/2000 cases.

## Accuracy (hasArea, correct rows / 100)

| | oracle | route_consensus | median | dominant |
|---|---:|---:|---:|---:|
| Gemma 4 31B thinking (this run) | 50 | 38 | 39 | 37 |
| Gemma 4 31B no-thinking (#17 artifact) | 43 | — | — | 32 (unit_eq) |
| Qwen3.5-27B thinking multiroute (#30) | 79 | 60 | 59 | 61 |

Thinking moves the pool for Gemma too (+7 oracle), replicating the Qwen
finding (69 → 78), but the knowledge deficit dominates: the Gemma line stays
closed for deployment.

## Reasoning style, same measurements as Qwen

| Metric (per candidate) | Qwen (#30) | Gemma (this run) |
|---|---|---|
| "commonly/often cited" template | 95–98% (all routes) | 75–92% |
| Median thinking length | ~8,000+ chars (budget-bound, 68–87% exhaustion) | **393–1,134 chars**, budget never reached |
| "Wait" loops | 8.3–9.5 | 0.9–1.8 |
| Wikipedia/Britannica/Wikidata mentions | 15–29 | **0.1–0.3** |
| Route effect on structure | anchor position 17% → 47–50% (B/C) | length doubles for C/D; anchor stays early in absolute terms |

Example (Corfu, direct route): "Approximate area: ~580 km². Commonly cited
values: 579.8 km², 580 km²." — anchor first, one-line justification, no
citation theater. Gemma's stable wrong anchor for Corfu is ~580 (Qwen's is
593; gold 626): a different wrong association, same mechanism.

## Conclusions

1. **The retrieval template and single-shot anchor replicate across model
   families.** Gemma also opens with a "commonly cited: X" lookup whose
   sampled value decides the candidate; route instructions add content
   without changing the retrieval distribution (route accuracies within 2pp
   of each other, exactly as in Qwen).
2. **The confabulation ritual does not replicate.** The fabricated
   Wikipedia/Q-ID citations, cross-language "verification", and Wait-loop
   oscillation that consume Qwen's 2,048-token budget are Qwen's trained
   thinking style, not a general property: Gemma emits ~50× fewer source
   mentions and finishes thinking in a few hundred characters.
3. Stable-wrong-anchor rows (Corfu-type) exist in both models with
   *different* wrong values — further evidence the failure lives in each
   model's label→value association, not in the reasoning layer above it.
