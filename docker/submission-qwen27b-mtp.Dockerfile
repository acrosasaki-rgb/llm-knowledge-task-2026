ARG LLAMA_CPP_IMAGE=ghcr.io/ggml-org/llama.cpp:full-cuda@sha256:11b0e950e081777cf326598bb2eff2ab0815f02405bf95c6650b34027750114e
FROM ${LLAMA_CPP_IMAGE}

ARG AKBC_CODE_COMMIT
LABEL org.opencontainers.image.revision="${AKBC_CODE_COMMIT}"
ENV AKBC_CODE_COMMIT="${AKBC_CODE_COMMIT}" \
    PYTHONUNBUFFERED=1

WORKDIR /opt/akbc
COPY pyproject.toml setup.py requirements-gguf.txt ./
COPY src ./src
COPY configs ./configs
COPY scripts/ci/fetch-dataset.sh ./scripts/ci/fetch-dataset.sh
COPY scripts/submission/run-qwen27b-mtp-container.sh \
    ./scripts/submission/run-qwen27b-mtp-container.sh

RUN python3 -m pip install \
      --break-system-packages \
      --disable-pip-version-check \
      -e . \
      -r requirements-gguf.txt

ENTRYPOINT ["bash", "/opt/akbc/scripts/submission/run-qwen27b-mtp-container.sh"]
