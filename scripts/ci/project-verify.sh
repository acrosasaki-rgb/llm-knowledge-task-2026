#!/usr/bin/env bash
set -Eeuo pipefail

if command -v python >/dev/null 2>&1 && python -c 'import sys; raise SystemExit(sys.version_info < (3, 11))'; then
  python_cmd=(python)
elif command -v python3 >/dev/null 2>&1 && python3 -c 'import sys; raise SystemExit(sys.version_info < (3, 11))'; then
  python_cmd=(python3)
elif [[ -x /mnt/c/Windows/py.exe ]]; then
  # The Windows workspace is sometimes verified through WSL, whose system
  # Python is older than the project's supported runtime.
  python_cmd=(/mnt/c/Windows/py.exe -3.11)
else
  echo "Python 3.11 or newer is required" >&2
  exit 1
fi

"${python_cmd[@]}" -m pip install --disable-pip-version-check -e ".[dev]"
"${python_cmd[@]}" -m compileall -q src tests
if "${python_cmd[@]}" -c 'import sys; raise SystemExit(sys.platform != "win32")'; then
  # Running the long-lived concurrency test after the other 144 tests can
  # receive a spurious KeyboardInterrupt from the Windows Python runtime.
  # Separate processes preserve the same coverage and keep local verification stable.
  "${python_cmd[@]}" -m pytest --ignore=tests/test_run.py
  "${python_cmd[@]}" -m pytest tests/test_run.py
else
  "${python_cmd[@]}" -m pytest
fi
bash -n \
  scripts/h100-bf16/run-v19-docker.sh \
  scripts/h100-bf16/run-v19-container.sh \
  scripts/submission/run-qwen27b-mtp-docker.sh \
  scripts/submission/run-qwen27b-mtp-container.sh \
  scripts/submission/run-qwen27b-mtp-relation-aware-20-docker.sh
