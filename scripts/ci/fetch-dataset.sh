#!/usr/bin/env bash
set -Eeuo pipefail

target="${1:?dataset target directory is required}"
ref="${2:?dataset git ref is required}"
repository="https://github.com/lm-kbc/dataset2026.git"

if [[ -e "${target}" && ! -d "${target}/.git" ]]; then
  echo "dataset target exists but is not a git checkout: ${target}" >&2
  exit 1
fi

if [[ ! -d "${target}/.git" ]]; then
  mkdir -p "$(dirname -- "${target}")"
  git clone --filter=blob:none --no-checkout "${repository}" "${target}"
fi

git -C "${target}" fetch --depth 1 origin "${ref}"
git -C "${target}" checkout --detach FETCH_HEAD

actual_ref="$(git -C "${target}" rev-parse HEAD)"
test "${actual_ref}" = "${ref}" || {
  echo "dataset ref mismatch: expected ${ref}, got ${actual_ref}" >&2
  exit 1
}

test -f "${target}/data/val.jsonl"
test -f "${target}/data/test.jsonl"
test -f "${target}/evaluate.py"
echo "dataset2026 ready at ${actual_ref}"
