import json
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

from akbc_baseline import backends
from akbc_baseline.config import ModelConfig


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.status = 200

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class _Opener:
    def __init__(self) -> None:
        self.requests: list[object] = []

    def open(self, request: object, timeout: int) -> _Response:
        self.requests.append(request)
        if request.full_url.endswith("/health"):
            return _Response({"status": "ok"})
        return _Response(
            {
                "choices": [
                    {
                        "message": {"content": "</think>\n[\"Paris\"]"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"completion_tokens": 12},
            }
        )


def _config() -> ModelConfig:
    return ModelConfig(
        name="gguf",
        model_id="test/gguf",
        backend="llama_cpp_server",
        prompt_templates_file="prompts.csv",
        train_data_file="train.jsonl",
        max_new_tokens=32,
        generation={"do_sample": True, "temperature": 0.7, "top_p": 0.9},
    )


def test_llama_cpp_backend_uses_openai_chat_api(
    monkeypatch,
) -> None:
    opener = _Opener()
    monkeypatch.setattr(backends, "build_opener", lambda *args: opener)
    backend = backends.LlamaCppServerBackend(_config())
    monkeypatch.setattr(backend, "_sample_cuda_memory", lambda: None)

    result = backend.generate(
        [{"role": "user", "content": "Where?"}], seed=123
    )

    assert result == '["Paris"]'
    request = opener.requests[-1]
    payload = json.loads(request.data.decode("utf-8"))
    assert request.full_url.endswith("/v1/chat/completions")
    assert payload["seed"] == 123
    assert payload["max_tokens"] == 32
    assert payload["temperature"] == 0.7
    assert "do_sample" not in payload
    assert backend.last_generation_diagnostics() == {
        "generated_tokens": 12,
        "finish_reason": "stop",
        "hit_token_limit": False,
        "hit_thinking_budget": False,
        "natural_think_end": True,
        "forced_think_end": False,
        "has_think_end": True,
        "final_text_empty": False,
    }


def test_generate_with_diagnostics_returns_the_request_diagnostics(
    monkeypatch,
) -> None:
    opener = _Opener()
    monkeypatch.setattr(backends, "build_opener", lambda *args: opener)
    backend = backends.LlamaCppServerBackend(_config())
    monkeypatch.setattr(backend, "_sample_cuda_memory", lambda: None)

    text, diagnostics = backend.generate_with_diagnostics(
        [{"role": "user", "content": "Where?"}], seed=7
    )

    assert text == '["Paris"]'
    assert diagnostics["generated_tokens"] == 12
    assert diagnostics == backend.last_generation_diagnostics()
    # The returned mapping is a copy, not shared mutable state.
    diagnostics["generated_tokens"] = 0
    assert backend.last_generation_diagnostics()["generated_tokens"] == 12


def test_llama_cpp_backend_enables_thinking_in_chat_template(
    monkeypatch,
) -> None:
    opener = _Opener()
    monkeypatch.setattr(backends, "build_opener", lambda *args: opener)
    config = replace(_config(), enable_thinking=True)
    backend = backends.LlamaCppServerBackend(config)
    monkeypatch.setattr(backend, "_sample_cuda_memory", lambda: None)

    backend.generate([{"role": "user", "content": "Where?"}], seed=123)

    request = opener.requests[-1]
    payload = json.loads(request.data.decode("utf-8"))
    assert payload["chat_template_kwargs"] == {"enable_thinking": True}


def test_llama_cpp_backend_allows_per_request_thinking_override(
    monkeypatch,
) -> None:
    opener = _Opener()
    monkeypatch.setattr(backends, "build_opener", lambda *args: opener)
    config = replace(_config(), enable_thinking=True)
    backend = backends.LlamaCppServerBackend(config)
    monkeypatch.setattr(backend, "_sample_cuda_memory", lambda: None)

    backend.generate(
        [{"role": "user", "content": "Where?"}],
        seed=123,
        enable_thinking=False,
    )

    request = opener.requests[-1]
    payload = json.loads(request.data.decode("utf-8"))
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_backend_factory_selects_llama_cpp(monkeypatch) -> None:
    monkeypatch.setattr(
        backends,
        "LlamaCppServerBackend",
        lambda config: SimpleNamespace(kind="llama"),
    )

    assert backends.create_backend(_config()).kind == "llama"


class _FakeTensor:
    def __init__(self, values: list[int], *, batched: bool = True) -> None:
        self.values = values
        self.batched = batched

    @property
    def shape(self) -> tuple[int, ...]:
        return (1, len(self.values)) if self.batched else (len(self.values),)

    def __getitem__(self, key: int | slice) -> "_FakeTensor":
        if self.batched and key == 0:
            return _FakeTensor(self.values, batched=False)
        if not self.batched and isinstance(key, slice):
            return _FakeTensor(self.values[key], batched=False)
        raise IndexError(key)

    def to(self, device: str) -> "_FakeTensor":
        return self


class _FakeBatch(dict[str, _FakeTensor]):
    def to(self, device: str) -> "_FakeBatch":
        return self


class _FakeProcessor:
    tokenizer: "_FakeProcessor"

    def __init__(self) -> None:
        self.tokenizer = self

    def apply_chat_template(self, *args: object, **kwargs: object) -> _FakeBatch:
        return _FakeBatch(
            input_ids=_FakeTensor([1, 2]),
            attention_mask=_FakeTensor([1, 1]),
        )

    def __call__(self, *args: object, **kwargs: object) -> dict[str, _FakeTensor]:
        return {"input_ids": _FakeTensor([90, 91])}

    def decode(self, ids: _FakeTensor, **kwargs: object) -> str:
        if ids.values == [10, 11]:
            return "<think>still working"
        if ids.values == [20]:
            return '["Paris"]<|im_end|>'
        raise AssertionError(f"unexpected token ids: {ids.values}")


class _FakeModel:
    device = "cuda:0"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs: object) -> _FakeTensor:
        self.calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        assert isinstance(input_ids, _FakeTensor)
        if len(self.calls) == 1:
            return _FakeTensor(input_ids.values + [10, 11])
        return _FakeTensor(input_ids.values + [20])


def test_transformers_backend_forces_final_answer_after_thinking_budget() -> None:
    config = ModelConfig(
        name="thinking",
        model_id="test/thinking",
        backend="causal",
        prompt_templates_file="prompts.csv",
        train_data_file="train.jsonl",
        max_new_tokens=2,
        final_answer_tokens=3,
        enable_thinking=True,
        generation={"do_sample": True},
    )
    model = _FakeModel()
    fake_torch = SimpleNamespace(
        manual_seed=lambda seed: None,
        cuda=SimpleNamespace(manual_seed_all=lambda seed: None),
        inference_mode=nullcontext,
        cat=lambda tensors, dim: _FakeTensor(
            [value for tensor in tensors for value in tensor.values]
        ),
        ones_like=lambda tensor: _FakeTensor([1] * len(tensor.values)),
    )
    backend = object.__new__(backends.TransformersBackend)
    backend.config = config
    backend.torch = fake_torch
    backend.processor = _FakeProcessor()
    backend.model = model
    backend._last_generation_diagnostics = {}

    result = backend.generate(
        [{"role": "user", "content": "Where?"}],
        seed=123,
    )

    assert result == '["Paris"]'
    assert len(model.calls) == 2
    assert model.calls[0]["max_new_tokens"] == 2
    assert model.calls[1]["max_new_tokens"] == 3
    assert model.calls[1]["input_ids"].values == [1, 2, 10, 11, 90, 91]
    assert backend.last_generation_diagnostics() == {
        "generated_tokens": 3,
        "first_stage_generated_tokens": 2,
        "final_stage_generated_tokens": 1,
        "hit_token_limit": False,
        "hit_thinking_budget": True,
        "hit_final_token_limit": False,
        "natural_think_end": False,
        "forced_think_end": True,
        "has_think_end": True,
        "final_text_empty": False,
    }
