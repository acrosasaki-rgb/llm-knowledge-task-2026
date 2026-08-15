import pytest

from akbc_baseline.preflight import validate_c_compiler, validate_mxfp4_dependencies


def test_accepts_available_c_compiler() -> None:
    validate_c_compiler("/usr/bin/cc")


def test_rejects_missing_c_compiler() -> None:
    with pytest.raises(RuntimeError, match="build-essential"):
        validate_c_compiler(None)


def test_accepts_compatible_mxfp4_dependencies() -> None:
    validate_mxfp4_dependencies(
        kernels_available=True,
        triton_available=True,
        kernels_version="0.15.2",
        triton_version="3.4.0",
    )


@pytest.mark.parametrize(
    ("kernels_available", "triton_available", "message"),
    [(False, True, "kernels=0.16.0"), (True, False, "triton=3.3.0")],
)
def test_rejects_dependencies_that_force_bf16_fallback(
    kernels_available: bool, triton_available: bool, message: str
) -> None:
    with pytest.raises(RuntimeError, match=message):
        validate_mxfp4_dependencies(
            kernels_available=kernels_available,
            triton_available=triton_available,
            kernels_version="0.16.0" if not kernels_available else "0.15.2",
            triton_version="3.3.0" if not triton_available else "3.4.0",
        )
