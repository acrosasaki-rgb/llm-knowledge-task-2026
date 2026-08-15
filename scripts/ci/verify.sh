#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

required_files=(
  README.md
  scripts/ci/verify.sh
)

for file in "${required_files[@]}"; do
  test -f "${file}" || {
    echo "required file is missing: ${file}" >&2
    exit 1
  }
done

while IFS= read -r -d '' script; do
  bash -n "${script}"
done < <(find scripts -type f -name '*.sh' -print0)

credential_matches="$(
  find . -type f ! -path './.git/*' ! -name '.env' \
    -exec grep -EHn \
      '((glpat|glrtr|glrt)-[[:alnum:]]|PRIVATE-TOKEN[[:space:]]*:|SSH_PASSWORD[[:space:]]*=)' {} + \
    || true
)"

if [[ -n "${credential_matches}" ]]; then
  printf '%s\n' "${credential_matches}"
  echo "credential-like value was found in repository content" >&2
  exit 1
fi

project_verifier="scripts/ci/project-verify.sh"
if [[ -f "${project_verifier}" ]]; then
  bash "${project_verifier}"
else
  manifests=(
    package.json
    pyproject.toml
    requirements.txt
    go.mod
    Cargo.toml
    pom.xml
    build.gradle
    composer.json
    Gemfile
  )

  for manifest in "${manifests[@]}"; do
    if [[ -f "${manifest}" ]]; then
      echo "${manifest} exists, but ${project_verifier} is not configured" >&2
      exit 1
    fi
  done

  echo "No application manifest was detected; policy checks only."
fi

echo "repository verification passed"
