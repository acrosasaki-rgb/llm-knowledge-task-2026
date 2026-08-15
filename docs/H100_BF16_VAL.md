# H100 BF16 validation experiment

`scripts/h100-bf16/` runs the current best validation configuration
(`experiment-qwen-3.5-27b-mtp-thinking-empty-aware.yaml`, Macro F1 0.5662)
with unquantized BF16 weights on one rented H100 80GB. The goal is to
separate quantization loss from missing model knowledge: prompts, few-shot
selection, seeds, sampling parameters, thinking budget, and aggregation are
identical to the Q4_K_M run; only the weight precision changes.

This flow is deliberately outside GitLab CI and outside the test submission
flow. It infers `data/val.jsonl` only, needs no selection manifest, and never
touches `data/test.jsonl`.

## What differs from the Q4_K_M baseline, and why

| Item | Q4_K_M baseline | BF16 experiment | Reason |
|---|---|---|---|
| Weights | `unsloth/Qwen3.5-27B-MTP-GGUF` Q4_K_M (16.7 GB) | `unsloth/Qwen3.5-27B-GGUF` BF16 split GGUF (~54 GB) | The variable under test |
| MTP | enabled | absent | The BF16 GGUF has no MTP head, and speculative decoding competes with batching for the same idle compute |
| Server slots | `--parallel 1` | `--parallel 8` (`AKBC_PARALLEL`) | A single BF16 stream on H100 decodes no faster than Q4+MTP on the 3090; throughput comes from batching |
| Context | `-c 8192` total | `10240 x 8` (`AKBC_CTX_PER_SLOT`) | llama.cpp splits `--ctx-size` evenly across slots; 10,240 per slot covers the worst awardWonBy prompt (~7k tokens) plus the 2,176-token budget, which `8192 / 1` only just covered |
| Client | sequential rows | 8 concurrent shard processes | `akbc_baseline.run` issues one blocking request at a time, so server slots alone would stay idle; N processes over disjoint `--offset/--limit` ranges fill them without code changes |
| Batch sizes | 512 / 128 | 2048 / 512 | Prefill throughput on H100; does not affect sampling semantics |

Candidate seeds are derived from `(seed, subject, relation,
candidate_index)`, so sharding reproduces exactly the seeds a sequential run
would use. Note that concurrent batching can still change low-level floating
point reduction order inside llama.cpp, so token-identical output versus a
1-slot run is not guaranteed even at the same precision; the comparison is
statistical, not bitwise.

## Procedure

On the GPU host, from a checkout of this repository:

```bash
bash scripts/h100-bf16/run-bf16-val-docker.sh smoke
```

Smoke covers the first 16 validation rows (2 per slot, 80 generations),
runs the quality gate and the official evaluator on that slice, and prints a
wall-time projection for the full run. Review
`reports/qwen3.5-27b-bf16-thinking-empty-aware-smoke-comparison.md`,
`...-smoke-quality.json` (empty rates, budget-hit rates), and the projection.
Then:

```bash
bash scripts/h100-bf16/run-bf16-val-docker.sh full
```

Full mode runs all validation rows in 8 shards, merges them with order
validation, applies the quality gate, scores with the official evaluator, and
writes a sha256. Shards use `--resume`, so a crashed run continues where it
stopped. Keep the instance alive between smoke and full: weights and dataset
are cached under `.cache/` and are not re-downloaded.

Artifacts land in `outputs/` and `reports/` with the
`qwen3.5-27b-bf16-thinking-empty-aware` prefix.

## Cost expectation (1x H100 80GB, $4.29/h)

| Phase | Expectation |
|---|---|
| Image build + dataset + 54 GB weights (hf_transfer) | ~0.3–0.5 h |
| Smoke (80 generations) | ~10–20 min |
| Full validation (2,390 generations, batched) | ~1–3 h depending on realized thinking length |
| Total | roughly $9–16; keep $20 of headroom |

The dominant uncertainty is mean thinking length: the Q4 run averaged ~1,530
thinking tokens with 54.8% budget exhaustion, and BF16 may think shorter or
longer. The smoke projection replaces this guess with a measurement — if the
projection is far above ~4 h, stop and reconsider before paying for the full
run.

## Failure handling

- Server OOM at startup (visible in `reports/llama-server.log`): lower
  `AKBC_PARALLEL` (8 → 6 → 4) before lowering `AKBC_CTX_PER_SLOT`; the
  per-slot context protects awardWonBy rows.
- A failed shard exits the container with the shard log path on stderr;
  rerunning the same mode resumes completed rows.
- The preflight requires all layers offloaded to the GPU but, unlike the
  Q4 jobs, does not require MTP.

## Interpreting the result

Compare against the Q4_K_M thinking-empty-aware validation artifacts (Job 309
candidates, Macro F1 0.5662). Two axes:

1. Official Macro F1 per relation (`...-val-comparison.md`).
2. Oracle recall — for each relation, rows where any of the 5 candidates
   contains a correct answer. Compute it locally from
   `outputs/qwen3.5-27b-bf16-thinking-empty-aware-candidates-val.jsonl`
   against the same-config Q4 candidates; this is the metric that separates
   "the quantized model lost the knowledge" (oracle recall rises with BF16)
   from "the model never knew" (oracle recall unchanged).
