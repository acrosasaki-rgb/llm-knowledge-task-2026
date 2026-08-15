from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Sequence
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


OFFLOAD_PATTERN = re.compile(
    r"offloaded\s+(?P<offloaded>\d+)/(?P<total>\d+)\s+layers to GPU",
    re.IGNORECASE,
)
MTP_PATTERN = re.compile(
    r"(?:draft-mtp|speculative decoding context initialized|"
    r"speculative.*method.*mtp)",
    re.IGNORECASE,
)


def validate_full_gpu_offload(log_text: str) -> tuple[int, int]:
    matches = list(OFFLOAD_PATTERN.finditer(log_text))
    if not matches:
        raise RuntimeError(
            "llama.cpp log does not report the number of GPU-offloaded layers"
        )
    offloaded = int(matches[-1].group("offloaded"))
    total = int(matches[-1].group("total"))
    if offloaded != total:
        raise RuntimeError(
            f"llama.cpp only offloaded {offloaded}/{total} layers to GPU"
        )
    return offloaded, total


def validate_mtp_enabled(log_text: str) -> None:
    if not MTP_PATTERN.search(log_text):
        raise RuntimeError("llama.cpp log does not confirm MTP speculative decoding")


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_until_healthy(
    url: str,
    timeout_seconds: float,
    *,
    server_pid: int | None = None,
    log_path: str | Path | None = None,
) -> None:
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if server_pid is not None and not process_exists(server_pid):
            log_tail = ""
            if log_path is not None and Path(log_path).exists():
                log_tail = Path(log_path).read_text(
                    encoding="utf-8", errors="replace"
                )[-4000:]
            raise RuntimeError(
                f"llama.cpp server process {server_pid} exited before health "
                f"check succeeded:\n{log_tail}"
            )
        try:
            request = Request(f"{url.rstrip('/')}/health", method="GET")
            with opener.open(request, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if response.status == 200 and payload.get("status") == "ok":
                return
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(2)
    raise RuntimeError(
        f"llama.cpp server did not become healthy within {timeout_seconds}s: "
        f"{last_error}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Require a healthy, fully GPU-offloaded llama.cpp server"
    )
    parser.add_argument("--url", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--timeout-seconds", type=float, default=900)
    parser.add_argument("--require-mtp", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    wait_until_healthy(
        args.url,
        args.timeout_seconds,
        server_pid=args.server_pid,
        log_path=args.log,
    )
    log_text = Path(args.log).read_text(encoding="utf-8", errors="replace")
    offloaded, total = validate_full_gpu_offload(log_text)
    if args.require_mtp:
        validate_mtp_enabled(log_text)
    print(f"llama.cpp is healthy; full GPU offload confirmed: {offloaded}/{total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
