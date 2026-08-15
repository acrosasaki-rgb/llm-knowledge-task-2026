ARG LLAMA_CPP_IMAGE=ghcr.io/ggml-org/llama.cpp:full-cuda@sha256:11b0e950e081777cf326598bb2eff2ab0815f02405bf95c6650b34027750114e
FROM ${LLAMA_CPP_IMAGE}

ARG AKBC_CODE_COMMIT=unknown
LABEL org.opencontainers.image.revision="${AKBC_CODE_COMMIT}"
ENV AKBC_CODE_COMMIT="${AKBC_CODE_COMMIT}" \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /opt/akbc
COPY pyproject.toml setup.py requirements-gguf.txt ./
COPY src ./src
COPY configs ./configs
COPY scripts/ci/fetch-dataset.sh ./scripts/ci/fetch-dataset.sh
COPY scripts/h100-bf16 ./scripts/h100-bf16

# pandas is required by the official dataset2026 evaluate.py that the final
# compare step imports; hf_transfer accelerates the ~54 GB BF16 download.
RUN python3 -m pip install \
      --break-system-packages \
      --disable-pip-version-check \
      -e . \
      -r requirements-gguf.txt \
      hf_transfer \
      pandas

ENTRYPOINT ["bash", "/opt/akbc/scripts/h100-bf16/run-bf16-val-container.sh"]
