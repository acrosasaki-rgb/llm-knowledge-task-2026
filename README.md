# AKBC Shared Task 2026 baseline

This is the public code release accompanying our AKBC 2026 shared task system
paper (`paper/main.tex`, `paper/main.pdf`). The submitted V19 system's
step-by-step reproduction recipe is
[`docs/V19_REPRODUCE.md`](docs/V19_REPRODUCE.md), and the
per-experiment records are the remaining files under `docs/`. Internal CI
definitions and infrastructure notes from our development repository are not
part of this release.

Qwen3.5-9B and gpt-oss-20b are evaluated with the same AKBC prompt and
few-shot selection policy. The inference job is intended to run on a
GPU-enabled GitLab Runner and writes official JSONL predictions plus a Markdown
comparison report.

The challenge dataset is not copied into this repository. Local runs may use
the sibling `../dataset2026` checkout. CI fetches the exact official commit
specified by each reproduction recipe.

## Submitted V19 inference

On a Linux NVIDIA Docker host, after checking out the dataset commit listed in
the V19 recipe:

```bash
AKBC_DATASET_DIR=../dataset2026 AKBC_SPLIT=test \
  bash scripts/h100-bf16/run-v19-docker.sh
```

This single command runs the pinned Gemma-3-27B-pt Q8_0 model, generates the
20-candidate V19 pool, applies the final relation-specific aggregation, and
validates the official submission schema. See
[`docs/V19_REPRODUCE.md`](docs/V19_REPRODUCE.md) for hardware, model revision,
output, and digest details.

## Local verification

```bash
python -m pip install -e ".[dev]"
bash ./scripts/ci/verify.sh
```

The verification suite does not download model weights or require a GPU.

## Local inference

Install inference dependencies in an environment that already provides a
CUDA-enabled PyTorch build:

```bash
python -m pip install -e . -r requirements-inference.txt
python -m akbc_baseline.preflight
python -m akbc_baseline.run \
  --config configs/baseline-qwen-3.5-9b.yaml \
  --dataset-dir ../dataset2026 \
  --input ../dataset2026/data/val.jsonl \
  --output outputs/qwen3.5-9b-val.jsonl
```

Use `configs/baseline-gpt-oss-20b.yaml` for gpt-oss. gpt-oss uses its
Transformers chat template, which applies the required Harmony response format.
Only the final channel is parsed and saved; reasoning text is not persisted.

To compare completed prediction files with the official evaluator:

```bash
python -m akbc_baseline.compare \
  --evaluator ../dataset2026/evaluate.py \
  --ground-truth ../dataset2026/data/val.jsonl \
  --prediction qwen3.5-9b=outputs/qwen3.5-9b-val.jsonl \
  --prediction gpt-oss-20b=outputs/gpt-oss-20b-val.jsonl \
  --json-output reports/comparison.json \
  --markdown-output reports/comparison.md
```

## GPU CI

Model inference ran on a GPU CI runner in our development repository; the
pipeline definitions are not part of this public release. The job sequence was:

1. fetch the pinned official dataset;
2. verify CUDA and print non-sensitive GPU diagnostics;
3. run Qwen3.5-9B and gpt-oss-20b on the validation split;
4. evaluate both outputs with the official evaluator;
5. retain validation predictions and comparison reports as job artifacts.

The Runner must expose an NVIDIA GPU to Docker. The inference requirements pin
a compatible Transformers/kernels pair so gpt-oss-20b keeps its MXFP4 weights
instead of silently expanding them to roughly 48 GB of BF16 weights. Loading
those weights also requires a C compiler for Triton JIT. The CI job installs
`build-essential`, and the preflight check rejects an incompatible environment
before downloading either model.
Set the protected/masked CI variable `HF_TOKEN` only if Hugging Face access
requires it. By default the job caches model weights under the Runner's
persistent `/cache` volume; `AKBC_HF_HOME` can override that location.

## Prediction artifact contract

Validation and test predictions are produced in separate trust boundaries:

- `outputs/<system>-val.jsonl` is the validation prediction used for local
  scoring with the official evaluator on the GitLab Runner;
- `outputs/<system>-test.jsonl` is the submission-ready prediction generated
  from the official `data/test.jsonl` on an external SSH-accessible Docker GPU
  host after manual model selection.

GitLab CI contains no test inference job. After a human reviews the complete
validation artifacts, a CPU-only manual selection job emits
`reports/selection.json`. The external launcher checks the selected config
SHA-256, dataset commit, code commit, validation Pipeline URL, and pinned
container image before it starts test inference. The test split contains
placeholder `ObjectEntities` values, so it is never passed to the evaluator.
Files containing `candidates` are raw generations for analysis and resume;
only the aggregated `*-test.jsonl` file is intended for Codabench submission.
See [`docs/EXTERNAL_SUBMISSION.md`](docs/EXTERNAL_SUBMISSION.md) for the SSH
handoff and Docker GPU procedure.

The relation-aware 20-candidate deployment is kept as a separate experiment:
`configs/experiment-qwen-3.5-27b-mtp-relation-aware-20.yaml` uses an
`awardWonBy` frequency threshold of 0.4, and
`scripts/submission/run-qwen27b-mtp-relation-aware-20-docker.sh` verifies a
20-candidate selection manifest before starting external test inference.

The manual
`experiment:model:qwen3.5-27b-mtp-relation-aware-20-no-thinking` job is a
validation-only ablation of that experiment. It keeps the same MTP model,
relation instructions, sampling parameters, 20 candidates, and aggregation
policies while disabling thinking for every relation and limiting each response
to 128 final-answer tokens. It has no selection or test-prediction job.

The validation-only
`experiment:model:qwen3.5-27b-mtp-award-era-no-thinking` job keeps that
non-thinking 20-candidate setup and partitions only `awardWonBy` by award-date
period. Candidate outputs are normalized, deduplicated, and unioned. Its smoke
starts at validation offset 200 and covers all ten `awardWonBy` validation rows.

The validation-only
`experiment:model:qwen3.5-27b-mtp-award-reverse-conversation` job tests two
independent conversations of ten turns each. Within each conversation the
model enumerates recipients from the newest period to the oldest while keeping
the earlier turns as context. Same-period answers from the two conversations
are aggregated together, and the ten period results are then unioned. Its smoke
also covers exactly the ten `awardWonBy` validation rows and does not create a
selection manifest or test prediction.
See [`docs/REASONING_ABLATION.md`](docs/REASONING_ABLATION.md)
for the controlled reasoning comparison and experiment rationale.

The validation-only
`experiment:model:qwen3.5-27b-mtp-hasarea-conversation-audit` job keeps each
of the 20 initial `hasArea` generations unchanged, then continues that same
conversation with a structured value/unit/scope/reference-year audit. Supported
units are normalized to square kilometers and incompatible scopes are excluded
before median aggregation. Its first 50 rows are a smoke shard; a manual review
gate blocks the remaining 50 rows. See
[`docs/HASAREA_AUDIT.md`](docs/HASAREA_AUDIT.md).

The follow-up
`experiment:model:qwen3.5-27b-mtp-hasarea-metadata-judge` removes correctness
and retention decisions from each candidate conversation. It extracts only
value/unit/scope/reference-year metadata, then sends all 20 candidate records to
a separate final judge that resolves logical inconsistencies and returns one
square-kilometer value. The same first-50/manual-review boundary applies. See
[`docs/HASAREA_METADATA_JUDGE.md`](docs/HASAREA_METADATA_JUDGE.md).

The current Runner has an RTX 3090 (24 GB, compute capability 8.6). The pinned
Transformers 5.14.1 MXFP4 path supports compute capability 7.5 or newer, so the
Runner can execute both models as long as the dependency preflight succeeds.

## Relation-wise self-consistency experiment

The manual `experiment:self-consistency:qwen-20` job compares the deterministic
Qwen3.5-9B baseline with 20 sampled candidates from the same model. It varies
few-shot examples and generation seeds, then applies relation-aware aggregation:

- frequency thresholds for set-valued relations;
- majority voting, including an empty answer vote, for city of death;
- median aggregation for numeric relations.

The GitLab job retains raw candidate JSONL and aggregated official prediction
JSONL for validation only. Test prediction is deliberately excluded from
GitLab CI. No prediction from another model is used in this experiment.

## Single-model reasoning comparison

The following manual jobs evaluate one candidate model at a time. Their
validation scores are compared with the non-thinking Qwen3.5-9B baseline:

- `experiment:model:qwen3.5-27b` uses Unsloth's Qwen3.5-27B Q4_K_M GGUF;
- `experiment:model:deepseek-r1-qwen3-8b` uses the official 8B DeepSeek
  reasoning distillation;
- `experiment:model:qwen3.5-9b-thinking` enables Qwen3.5-9B thinking mode.

There is no cross-model ensemble. Reasoning text is discarded before parsing,
and only the final answer is scored. Qwen thinking uses a two-stage generation:
`max_new_tokens` is the thinking budget, and `final_answer_tokens` reserves a
separate continuation for the requested JSON answer. When the first stage
exhausts its budget without `</think>`, the backend closes the reasoning block
before continuing. Candidate diagnostics distinguish natural and forced
thinking completion. Each GitLab job keeps raw candidates, validation
predictions, relation-level scores, elapsed time, empty-prediction count, and
sampled CUDA process memory.

The checked-in `parameter_count_billion` is rejected when it exceeds the
shared-task 32B total-parameter limit. The jobs are closed-book: they use only
the published model weights, the supplied train examples, and the validation
input; they perform no retrieval, external search, or additional training.
Quantization is used only to fit the 27B model on the 24 GiB Runner GPU and
does not change its declared 27B parameter count.

The 27B job runs the pinned `Qwen3.5-27B-Q4_K_M.gguf` file (about 16.7 GB)
with the digest-pinned official `llama.cpp:full-cuda` image. The model revision
and image digest are fixed in `.gitlab-ci.yml`. `llama-server` runs with an
8192-token context, one parallel slot, Flash Attention, `--n-gpu-layers 99`,
`--fit off`, and reasoning disabled. Trace-level server logging is retained so
the job can require both a healthy local API and a log entry proving that every
model layer was offloaded to the GPU before inference. A
one-row, 20-candidate smoke run projects the combined 955-row validation and
test runtime; the full run starts only when that projection fits within an
11-hour budget. The GGUF weights remain in the Runner's persistent Hugging
Face cache. Because AKBC inputs are text-only, the optional multimodal
projection GGUF is not downloaded.

The Q4_K_M job uses `llama.cpp` rather than the Transformers GPTQ path. This
avoids the generic GPTQ kernel and CPU/disk offload that made the published
GPTQ checkpoint too slow on the RTX 3090. No `llama.cpp` package or model is
installed on the GitLab/Runner host itself; both are supplied by the job
container and persistent `/cache` volume. Its report scores the 27B prediction
by itself; the baseline reference remains available from the baseline job so
the two runtimes do not have to coexist in one container. The job adds `/app`
to `LD_LIBRARY_PATH` because the pinned full CUDA image places its server
implementation shared library beside the executable.
