# Reproducing the submitted V19 system

V19 is the system submitted by team `acro-sasaki`. It uses one
Gemma-3-27B pretrained model and does not call a second model or an arbiter.
The public entrypoint below generates all 20 candidates per row, composes the
final predictions, and verifies the Codabench JSONL contract.

## Fixed inputs and runtime

- Public dataset base: `lm-kbc/dataset2026` commit
  `30d8cfaaa7af5b236054cfb361f57b7d0c1e6e57`
- Model: `bluerain123/gemma-3-27b-pt-Q8_0-GGUF`
- Model revision: `71ed905c894b1d481e67a3bdbdfe06dd5805c6e9`
- Model file: `gemma-3-27b-pt-q8_0.gguf` (Q8_0)
- Runtime image: the digest-pinned llama.cpp CUDA image in
  `docker/h100-bf16-val.Dockerfile`
- Sampling: 20 candidates, temperature 0.6, top-p 0.95, base seed 42
- Hardware: Linux x86_64, NVIDIA Container Toolkit, and an NVIDIA GPU with
  enough memory for the 28.7 GB GGUF plus KV cache (the submitted run used an
  H100 80 GB)

The Gemma license applies to the model weights. Hugging Face may require an
accepted license and `HF_TOKEN`; the token is passed to Docker but is never
written to an output file.

## Dataset checkout

From the parent directory of this repository:

```bash
git clone https://github.com/lm-kbc/dataset2026.git
git -C dataset2026 checkout 30d8cfaaa7af5b236054cfb361f57b7d0c1e6e57
```

The public commit contains the original 477-row test input. Before inference,
`prepare_v19_input.py` deterministically applies the official Codabench update
used by V19: 15 ambiguous subjects receive qualifiers and two obsolete
capacity rows are removed. The resulting 475-row input is retained in
`reports/gemma-3-27b-pt-v19-input-test.jsonl`. The launcher rejects any input
that is neither the exact pre-migration shape nor the already-migrated shape.
V19's test-time alias graph is built only from the public train and validation
labels.

## Run the submitted test system

From this repository root on the GPU host:

```bash
export AKBC_DATASET_DIR=../dataset2026
export AKBC_SPLIT=test
# export HF_TOKEN=...   # only when Hugging Face requires it
bash scripts/h100-bf16/run-v19-docker.sh
```

The command builds the pinned CUDA image, downloads the pinned Q8_0 file into
`.cache/v19-huggingface`, launches llama.cpp with full GPU offload, and runs
the following V19 relation policy:

- `hasArea`, `hasCapacity`: fixed four-example train-gold registers followed
  by dominant numeric-cluster aggregation;
- `personHasCityOfDeath`: About-first biography register, alive/unknown to
  empty, surface majority with at least 14 of 20 votes;
- `companyTradesAtStockExchange`: case and acronym folding, frequency at
  least 0.5, empty-majority gate at 12;
- `countryLandBordersCountry`: frequency at least 0.3, empty-majority gate at
  10;
- `awardWonBy`: frequency at least 0.1;
- public train+validation alias graph for string-valued relations.

## Outputs and checks

The completed run writes:

- `outputs/gemma-3-27b-pt-v19-candidates-test.jsonl`: raw candidates and
  completion diagnostics;
- `outputs/gemma-3-27b-pt-v19-test.jsonl`: Codabench submission JSONL;
- `reports/gemma-3-27b-pt-v19-test.sha256`: generated submission digest;
- `reports/gemma-3-27b-pt-v19-input-test.jsonl`: reconstructed 475-row
  submission input;
- `reports/gemma-3-27b-pt-v19-test-manifest.json`: model, sampling, code, and
  aggregation settings;
- `reports/gemma-3-27b-pt-v19-llama-server.log`: runtime log.

The final verification fails if any input row is missing, the candidate count
is not 20, row order or `(SubjectEntity, Relation)` changes, or the prediction
contains keys outside the official `SubjectEntity`, `Relation`, and
`ObjectEntities` schema.

For provenance, the submitted V19 file retained in the development archive was
`gemma-test-predictions-v19.jsonl` with SHA-256
`6ea6f871d7c0b013c9987a21ec058d80b91cfc733051d05c681bae9121f5067f`.
The corresponding submission ZIP had SHA-256
`69af59ae25df4b45b45882e11022baad9a8edfacaf0d5135b74a5eef5a4abb5c`.
The JSONL, not ZIP container metadata, is the meaningful prediction digest.

## Validation-only check

To exercise the complete path without producing a submission file, set
`AKBC_SPLIT=val`. The same V19 composition is used. Do not score `test.jsonl`:
its `ObjectEntities` values are placeholders, not gold labels.

The historical Mistral predecessor recipe remains available in
[`V4_REPRODUCE.md`](V4_REPRODUCE.md); it is not the submitted V19 system.
