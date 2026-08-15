from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


BackendName = Literal["causal", "multimodal", "llama_cpp_server"]


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model_id: str
    backend: BackendName
    prompt_templates_file: str
    train_data_file: str
    few_shot: int = 5
    seed: int = 42
    max_new_tokens: int = 64
    final_answer_tokens: int | None = None
    num_candidates: int = 1
    parameter_count_billion: float | None = None
    enable_thinking: bool | None = None
    relation_thinking: dict[str, bool] = field(default_factory=dict)
    system_prefix: str | None = None
    relation_instructions: dict[str, str] = field(default_factory=dict)
    candidate_instructions: dict[str, list[str]] = field(default_factory=dict)
    conversation_chains: dict[str, int] = field(default_factory=dict)
    candidate_audits: dict[str, dict[str, Any]] = field(default_factory=dict)
    torch_dtype: str = "auto"
    device_map: str = "auto"
    quantization_backend: str | None = None
    llama_cpp_url: str = "http://127.0.0.1:8080"
    model_load: dict[str, Any] = field(default_factory=dict)
    generation: dict[str, Any] = field(default_factory=dict)
    aggregation: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Persist the full model output (thinking included) into each candidate's
    # diagnostics. Off by default: 20 candidates x ~1,500 thinking tokens per
    # row makes candidate artifacts roughly 50x larger.
    save_raw_text: bool = False
    # Optional per-candidate sampling temperature (one entry per candidate
    # index, all relations). Overrides generation.temperature for that
    # candidate; llama_cpp_server only.
    candidate_temperatures: list[float] | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        with Path(path).open(encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
        if not isinstance(raw, dict):
            raise ValueError("model config must contain a YAML mapping")
        config = cls(**raw)
        if config.backend not in {"causal", "multimodal", "llama_cpp_server"}:
            raise ValueError(f"unsupported backend: {config.backend}")
        if not isinstance(config.save_raw_text, bool):
            raise ValueError("save_raw_text must be a boolean")
        if config.candidate_temperatures is not None:
            if config.backend != "llama_cpp_server":
                raise ValueError(
                    "candidate_temperatures requires the llama_cpp_server backend"
                )
            if len(config.candidate_temperatures) != config.num_candidates:
                raise ValueError(
                    "candidate_temperatures must contain exactly "
                    f"{config.num_candidates} entries"
                )
            for value in config.candidate_temperatures:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("candidate_temperatures must be numbers")
                if not 0 < float(value) <= 2:
                    raise ValueError(
                        "candidate_temperatures must be in (0, 2]"
                    )
        if config.few_shot < 0:
            raise ValueError("few_shot must be non-negative")
        if config.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if config.final_answer_tokens is not None:
            if config.final_answer_tokens < 1:
                raise ValueError("final_answer_tokens must be positive")
            if config.enable_thinking is not True:
                raise ValueError(
                    "final_answer_tokens requires enable_thinking: true"
                )
            if config.backend == "llama_cpp_server":
                raise ValueError(
                    "final_answer_tokens is not supported by llama_cpp_server"
                )
        if config.num_candidates < 1:
            raise ValueError("num_candidates must be positive")
        for relation, enabled in config.relation_thinking.items():
            if not isinstance(enabled, bool):
                raise ValueError(
                    f"relation_thinking for {relation} must be a boolean"
                )
        for relation, instructions in config.relation_instructions.items():
            if not isinstance(instructions, str) or not instructions.strip():
                raise ValueError(
                    f"relation_instructions for {relation} must be non-empty"
                )
        for relation, instructions in config.candidate_instructions.items():
            if not isinstance(instructions, list) or not instructions:
                raise ValueError(
                    f"candidate_instructions for {relation} must be a non-empty list"
                )
            if len(instructions) != config.num_candidates:
                raise ValueError(
                    f"candidate_instructions for {relation} must contain "
                    f"exactly {config.num_candidates} entries"
                )
            if any(
                not isinstance(instruction, str) or not instruction.strip()
                for instruction in instructions
            ):
                raise ValueError(
                    f"candidate_instructions for {relation} must be non-empty strings"
                )
        for relation, chains in config.conversation_chains.items():
            if not isinstance(chains, int) or isinstance(chains, bool) or chains < 2:
                raise ValueError(
                    f"conversation_chains for {relation} must be an integer >= 2"
                )
            if relation not in config.candidate_instructions:
                raise ValueError(
                    f"conversation_chains for {relation} requires candidate_instructions"
                )
            if config.num_candidates % chains:
                raise ValueError(
                    f"num_candidates must be divisible by conversation_chains "
                    f"for {relation}"
                )
        for relation, audit in config.candidate_audits.items():
            if not isinstance(audit, dict) or audit.get("kind") not in {
                "area_metadata",
                "area_metadata_only",
            }:
                raise ValueError(
                    f"candidate_audits for {relation} must use kind "
                    "area_metadata or area_metadata_only"
                )
            if relation != "hasArea":
                raise ValueError("area_metadata audit is supported only for hasArea")
            if relation in config.conversation_chains:
                raise ValueError(
                    f"candidate_audits and conversation_chains cannot both target {relation}"
                )
        if config.parameter_count_billion is not None:
            if config.parameter_count_billion <= 0:
                raise ValueError("parameter_count_billion must be positive")
            if config.parameter_count_billion > 32:
                raise ValueError(
                    "parameter_count_billion exceeds the shared-task 32B limit"
                )
        if (
            config.quantization_backend is not None
            and not config.quantization_backend.strip()
        ):
            raise ValueError("quantization_backend must not be empty")
        if config.backend == "llama_cpp_server" and not config.llama_cpp_url.startswith(
            ("http://", "https://")
        ):
            raise ValueError("llama_cpp_url must be an HTTP(S) URL")
        from .aggregation import SUPPORTED_STRATEGIES

        for relation, policy in config.aggregation.items():
            strategy = policy.get("strategy")
            if strategy not in SUPPORTED_STRATEGIES:
                raise ValueError(
                    f"unsupported aggregation strategy for {relation}: {strategy}"
                )
            if strategy in {"frequency", "grouped_frequency"}:
                threshold = float(policy.get("threshold", 0.5))
                if not 0 < threshold <= 1:
                    raise ValueError(
                        f"frequency threshold for {relation} must be in (0, 1]"
                    )
            if strategy == "grouped_frequency":
                groups = int(policy.get("groups", 0))
                if groups < 1 or config.num_candidates % groups:
                    raise ValueError(
                        f"grouped_frequency groups for {relation} must divide "
                        f"num_candidates"
                    )
            if strategy == "audited_median" and relation not in config.candidate_audits:
                raise ValueError(
                    f"audited_median for {relation} requires candidate_audits"
                )
            if strategy == "metadata_judge":
                audit = config.candidate_audits.get(relation)
                if not isinstance(audit, dict) or audit.get("kind") != "area_metadata_only":
                    raise ValueError(
                        f"metadata_judge for {relation} requires "
                        "candidate_audits kind area_metadata_only"
                    )
            if strategy == "unit_equivalence":
                if relation != "hasArea":
                    raise ValueError(
                        "unit_equivalence is supported only for hasArea"
                    )
                for key in ("cluster_tolerance", "unit_tolerance"):
                    tolerance = float(policy.get(key, 0.05))
                    if not 0 < tolerance < 1:
                        raise ValueError(
                            f"unit_equivalence {key} must be in (0, 1)"
                        )
            if strategy == "cluster_choice":
                if relation != "hasArea":
                    raise ValueError(
                        "cluster_choice is supported only for hasArea"
                    )
                tolerance = float(policy.get("tolerance", 0.05))
                if not 0 < tolerance < 1:
                    raise ValueError(
                        "cluster_choice tolerance must be in (0, 1)"
                    )
        return config
