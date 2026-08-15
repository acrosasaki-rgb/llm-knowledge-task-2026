from __future__ import annotations

import os
import shutil
from importlib.metadata import PackageNotFoundError, version


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def validate_mxfp4_dependencies(
    *,
    kernels_available: bool,
    triton_available: bool,
    kernels_version: str,
    triton_version: str,
) -> None:
    if not kernels_available:
        raise RuntimeError(
            "Transformers cannot use the installed kernels package "
            f"(kernels={kernels_version}); install the version constrained in "
            "requirements-inference.txt to avoid a 48 GiB BF16 fallback"
        )
    if not triton_available:
        raise RuntimeError(
            "MXFP4 requires Triton >= 3.4.0 "
            f"(triton={triton_version})"
        )


def validate_c_compiler(compiler: str | None) -> None:
    if compiler is None:
        raise RuntimeError(
            "MXFP4 weight loading requires a C compiler for Triton JIT; "
            "install build-essential or set CC to an available compiler"
        )


def main() -> int:
    try:
        import torch
        import transformers
        from transformers.utils import is_kernels_available, is_triton_available
    except ImportError as exc:
        raise RuntimeError("inference dependencies are not installed") from exc

    print(f"torch={torch.__version__}")
    print(f"transformers={transformers.__version__}")
    kernels_version = _package_version("kernels")
    triton_version = _package_version("triton")
    print(f"kernels={kernels_version}")
    print(f"triton={triton_version}")
    validate_mxfp4_dependencies(
        kernels_available=is_kernels_available(),
        triton_available=is_triton_available("3.4.0"),
        kernels_version=kernels_version,
        triton_version=triton_version,
    )
    compiler_name = os.environ.get("CC", "cc")
    compiler = shutil.which(compiler_name)
    print(f"c_compiler={compiler or 'not-found'}")
    validate_c_compiler(compiler)
    print(f"cuda_available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("GitLab Runner does not expose a CUDA GPU")

    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        memory_gib = properties.total_memory / 1024**3
        print(
            f"gpu[{index}]={properties.name}; "
            f"capability={properties.major}.{properties.minor}; "
            f"memory_gib={memory_gib:.1f}"
        )
        if properties.major < 7 or (properties.major == 7 and properties.minor < 5):
            raise RuntimeError(
                "the pinned Transformers MXFP4 kernels require compute "
                "capability 7.5 or newer"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
