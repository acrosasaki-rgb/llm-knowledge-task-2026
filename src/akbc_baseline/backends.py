from __future__ import annotations

import json
import subprocess
import threading
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from .config import ModelConfig
from .parsing import extract_final_text


THINKING_BUDGET_STOP_TEXT = (
    "\n\nThe reasoning budget is exhausted. Give the requested JSON array "
    "now using the best answer found so far.\n</think>\n\n"
)
REASONING_BUDGET_STOP_MESSAGE = (
    "The reasoning budget is exhausted. Give the requested JSON array now "
    "using the best answer found so far."
)


class TransformersBackend:
    """Lazy Transformers backend supporting the two official model families."""

    def __init__(self, config: ModelConfig) -> None:
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForMultimodalLM,
                AutoProcessor,
                AutoTokenizer,
                GPTQConfig,
            )
        except ImportError as exc:
            raise RuntimeError(
                "inference dependencies are missing; install requirements-inference.txt"
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required for baseline inference")

        self.config = config
        self.torch = torch
        model_kwargs = {
            "device_map": config.device_map,
            "torch_dtype": config.torch_dtype,
            "low_cpu_mem_usage": True,
        }
        model_kwargs.update(config.model_load)
        if config.quantization_backend is not None:
            model_kwargs["quantization_config"] = GPTQConfig(
                bits=4,
                backend=config.quantization_backend,
            )
        if config.backend == "multimodal":
            self.processor = AutoProcessor.from_pretrained(config.model_id)
            self.model = AutoModelForMultimodalLM.from_pretrained(
                config.model_id, **model_kwargs
            )
        else:
            self.processor = AutoTokenizer.from_pretrained(config.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_id, **model_kwargs
            )
        self.model.eval()
        self._last_generation_diagnostics: dict[str, Any] = {}

    def generate(
        self,
        messages: list[dict[str, str]],
        seed: int | None = None,
        enable_thinking: bool | None = None,
    ) -> str:
        template_kwargs: dict[str, Any] = {}
        thinking = (
            self.config.enable_thinking
            if enable_thinking is None
            else enable_thinking
        )
        if thinking is not None:
            template_kwargs["enable_thinking"] = thinking

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **template_kwargs,
        )
        inputs = inputs.to(self.model.device)

        generation_kwargs = dict(self.config.generation)
        generation_kwargs["max_new_tokens"] = self.config.max_new_tokens
        if seed is not None:
            self.torch.manual_seed(seed)
            self.torch.cuda.manual_seed_all(seed)
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, **generation_kwargs)

        prompt_length = inputs["input_ids"].shape[-1]
        completion_ids = generated[0][prompt_length:]
        decoded = self.processor.decode(
            completion_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        generated_tokens = int(completion_ids.shape[-1])
        first_stage_hit_limit = generated_tokens >= self.config.max_new_tokens
        natural_think_end = "</think>" in decoded
        forced_think_end = False
        final_stage_tokens = 0
        final_stage_hit_limit = False

        if (
            self.config.final_answer_tokens is not None
            and first_stage_hit_limit
        ):
            continuation_input_ids = generated
            if not natural_think_end:
                tokenizer = getattr(self.processor, "tokenizer", self.processor)
                stopping = tokenizer(
                    THINKING_BUDGET_STOP_TEXT,
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_tensors="pt",
                )
                stopping_ids = stopping["input_ids"].to(self.model.device)
                continuation_input_ids = self.torch.cat(
                    [continuation_input_ids, stopping_ids], dim=-1
                )
                forced_think_end = True

            continuation_kwargs = dict(self.config.generation)
            continuation_kwargs["max_new_tokens"] = (
                self.config.final_answer_tokens
            )
            attention_mask = self.torch.ones_like(continuation_input_ids)
            with self.torch.inference_mode():
                continued = self.model.generate(
                    input_ids=continuation_input_ids,
                    attention_mask=attention_mask,
                    **continuation_kwargs,
                )
            continuation_length = continuation_input_ids.shape[-1]
            final_ids = continued[0][continuation_length:]
            final_decoded = self.processor.decode(
                final_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            final_stage_tokens = int(final_ids.shape[-1])
            final_stage_hit_limit = (
                final_stage_tokens >= self.config.final_answer_tokens
            )
            if forced_think_end:
                decoded = decoded + THINKING_BUDGET_STOP_TEXT + final_decoded
            else:
                decoded = decoded + final_decoded

        final_text = extract_final_text(decoded)
        hit_token_limit = (
            final_stage_hit_limit
            if self.config.final_answer_tokens is not None
            else first_stage_hit_limit
        )
        self._last_generation_diagnostics = {
            "generated_tokens": generated_tokens + final_stage_tokens,
            "first_stage_generated_tokens": generated_tokens,
            "final_stage_generated_tokens": final_stage_tokens,
            "hit_token_limit": hit_token_limit,
            "hit_thinking_budget": forced_think_end,
            "hit_final_token_limit": final_stage_hit_limit,
            "natural_think_end": natural_think_end,
            "forced_think_end": forced_think_end,
            "has_think_end": natural_think_end or forced_think_end,
            "final_text_empty": not final_text.strip(),
        }
        if self.config.save_raw_text:
            self._last_generation_diagnostics["raw_text"] = decoded
        return final_text

    def last_generation_diagnostics(self) -> dict[str, Any]:
        return dict(self._last_generation_diagnostics)

    def reset_peak_memory_stats(self) -> None:
        self.torch.cuda.reset_peak_memory_stats()

    def peak_cuda_memory_gib(self) -> float:
        return self.torch.cuda.max_memory_allocated() / 1024**3


class LlamaCppServerBackend:
    """Client for a local llama.cpp OpenAI-compatible CUDA server."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.url = config.llama_cpp_url.rstrip("/")
        # CI jobs define an outbound proxy. Never route localhost inference
        # requests through it, even if NO_PROXY is misconfigured.
        self.opener = build_opener(ProxyHandler({}))
        # Concurrent candidate generation shares this client, so shared
        # diagnostic state is guarded by a lock.
        self._lock = threading.Lock()
        self._peak_cuda_memory_gib = 0.0
        self._last_generation_diagnostics: dict[str, Any] = {}
        self._request("/health", method="GET")

    def _request(
        self, path: str, *, method: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        data = None
        headers: dict[str, str] = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(
            f"{self.url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with self.opener.open(request, timeout=300) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError(f"llama.cpp request failed: {path}: {exc}") from exc
        if not isinstance(decoded, dict):
            raise RuntimeError(f"llama.cpp returned a non-object response: {path}")
        return decoded

    def generate(
        self,
        messages: list[dict[str, str]],
        seed: int | None = None,
        enable_thinking: bool | None = None,
        temperature: float | None = None,
    ) -> str:
        final_text, _ = self.generate_with_diagnostics(
            messages,
            seed=seed,
            enable_thinking=enable_thinking,
            temperature=temperature,
        )
        return final_text

    def generate_with_diagnostics(
        self,
        messages: list[dict[str, str]],
        seed: int | None = None,
        enable_thinking: bool | None = None,
        temperature: float | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Thread-safe generation returning the per-request diagnostics."""

        generation = dict(self.config.generation)
        generation.pop("do_sample", None)
        if temperature is not None:
            generation["temperature"] = float(temperature)
        thinking = (
            self.config.enable_thinking
            if enable_thinking is None
            else enable_thinking
        )
        payload: dict[str, Any] = {
            "model": self.config.model_id,
            "messages": messages,
            "max_tokens": self.config.max_new_tokens,
            **generation,
        }
        if thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": thinking}
        if seed is not None:
            payload["seed"] = seed
        response = self._request(
            "/v1/chat/completions", method="POST", payload=payload
        )
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("llama.cpp response has no assistant content") from exc
        if not isinstance(content, str):
            raise RuntimeError("llama.cpp assistant content is not text")
        usage = response.get("usage", {})
        completion_tokens = usage.get("completion_tokens")
        finish_reason = response.get("choices", [{}])[0].get("finish_reason")
        final_text = extract_final_text(content)
        forced_think_end = REASONING_BUDGET_STOP_MESSAGE in content
        has_think_end = "</think>" in content
        diagnostics = {
            "generated_tokens": (
                int(completion_tokens)
                if isinstance(completion_tokens, int)
                else None
            ),
            "finish_reason": finish_reason,
            "hit_token_limit": finish_reason == "length",
            "hit_thinking_budget": forced_think_end,
            "natural_think_end": has_think_end and not forced_think_end,
            "forced_think_end": forced_think_end,
            "has_think_end": has_think_end,
            "final_text_empty": not final_text.strip(),
        }
        if self.config.save_raw_text:
            diagnostics["raw_text"] = content
        with self._lock:
            self._last_generation_diagnostics = diagnostics
        self._sample_cuda_memory()
        return final_text, dict(diagnostics)

    def last_generation_diagnostics(self) -> dict[str, Any]:
        return dict(self._last_generation_diagnostics)

    def _sample_cuda_memory(self) -> None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            used_mib = sum(
                float(line.strip())
                for line in result.stdout.splitlines()
                if line.strip().replace(".", "", 1).isdigit()
            )
        except (OSError, subprocess.SubprocessError, ValueError):
            return
        with self._lock:
            self._peak_cuda_memory_gib = max(
                self._peak_cuda_memory_gib, used_mib / 1024
            )

    def reset_peak_memory_stats(self) -> None:
        self._peak_cuda_memory_gib = 0.0
        self._sample_cuda_memory()

    def peak_cuda_memory_gib(self) -> float:
        return self._peak_cuda_memory_gib


def create_backend(config: ModelConfig) -> TransformersBackend | LlamaCppServerBackend:
    if config.backend == "llama_cpp_server":
        return LlamaCppServerBackend(config)
    return TransformersBackend(config)
