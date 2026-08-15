import pytest

from akbc_baseline.llama_preflight import (
    process_exists,
    validate_full_gpu_offload,
    validate_mtp_enabled,
)


def test_accepts_full_llama_cpp_gpu_offload() -> None:
    assert validate_full_gpu_offload(
        "load_tensors: offloaded 65/65 layers to GPU\n"
    ) == (65, 65)


def test_rejects_partial_llama_cpp_gpu_offload() -> None:
    with pytest.raises(RuntimeError, match="only offloaded 60/65"):
        validate_full_gpu_offload(
            "load_tensors: offloaded 60/65 layers to GPU\n"
        )


def test_rejects_missing_llama_cpp_gpu_offload_report() -> None:
    with pytest.raises(RuntimeError, match="does not report"):
        validate_full_gpu_offload("model loaded\n")


def test_accepts_native_mtp_log_marker() -> None:
    validate_mtp_enabled("speculative decoding context initialized: draft-mtp\n")


def test_rejects_missing_native_mtp_log_marker() -> None:
    with pytest.raises(RuntimeError, match="does not confirm MTP"):
        validate_mtp_enabled("model loaded without speculative decoding\n")


def test_detects_current_process() -> None:
    import os

    assert process_exists(os.getpid())
