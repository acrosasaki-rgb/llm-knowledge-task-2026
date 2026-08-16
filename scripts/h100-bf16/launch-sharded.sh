#!/usr/bin/env bash
# Run one generation job across N GPUs by sharding the rows file.
#
# Required env: AKBC_MODEL_KEY  (output key), AKBC_ENTRYPOINT (container script under
#               /workspace/run/outputs/), AKBC_DATA_FILE (host path, e.g. outputs/val-new-475.jsonl),
#               AKBC_GGUF_REPO / AKBC_GGUF_REV / AKBC_GGUF_PART1, AKBC_SUFFIX (output suffix the
#               container script writes, e.g. test-all-candidates | icl-candidates | city-extra | native-raw)
# Optional:     NSHARDS (default 8), GPUS (default "0 1 2 3 4 5 6 7"), AKBC_SEED, AKBC_REGS, AKBC_GGUF_MIN_GIB
#
# Usage (on host, from repo root):
#   AKBC_MODEL_KEY=g4pt-test AKBC_ENTRYPOINT=/workspace/run/outputs/run-gemma-test-container.sh \
#   AKBC_DATA_FILE=outputs/test-new-475.jsonl AKBC_SUFFIX=test-all-candidates \
#   AKBC_GGUF_REPO=... AKBC_GGUF_REV=... AKBC_GGUF_PART1=... bash scripts/h100-bf16/launch-sharded.sh
set -Eeuo pipefail
cd "$(dirname "$0")/../.."

: "${AKBC_MODEL_KEY:?}" "${AKBC_ENTRYPOINT:?}" "${AKBC_DATA_FILE:?}" "${AKBC_SUFFIX:?}"
NSHARDS="${NSHARDS:-8}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
key="${AKBC_MODEL_KEY}"
mkdir -p outputs/shards
python3 - "${AKBC_DATA_FILE}" "${NSHARDS}" "outputs/shards/${key}" <<'PY'
import sys
src, n, prefix = sys.argv[1], int(sys.argv[2]), sys.argv[3]
rows = [l for l in open(src, encoding="utf-8") if l.strip()]
outs = [open(f"{prefix}-{i}.jsonl", "w", encoding="utf-8") for i in range(n)]
for i, line in enumerate(rows):
    outs[i % n].write(line if line.endswith("\n") else line + "\n")
for o in outs:
    o.close()
print(f"{len(rows)} rows -> {n} shards")
PY

pids=()
i=0
for g in ${GPUS}; do
  [ "${i}" -ge "${NSHARDS}" ] && break
  AKBC_MODEL_KEY="${key}-sh${i}" AKBC_GPU="${g}" \
  AKBC_DATA_FILE="/workspace/run/outputs/shards/${key}-${i}.jsonl" \
  bash scripts/h100-bf16/run-model-screening-docker.sh > "arm-${key}-sh${i}.log" 2>&1 &
  pids+=($!)
  i=$((i+1))
  sleep 3
done
echo "launched ${i} shards for ${key}"
fail=0
for p in "${pids[@]}"; do wait "${p}" || fail=1; done
cat $(for j in $(seq 0 $((i-1))); do echo "outputs/${key}-sh${j}-${AKBC_SUFFIX}.jsonl"; done) > "outputs/${key}-${AKBC_SUFFIX}.jsonl"
echo "merged -> outputs/${key}-${AKBC_SUFFIX}.jsonl ($(wc -l < outputs/${key}-${AKBC_SUFFIX}.jsonl) rows) fail=${fail}"
echo "sharded job completed"
