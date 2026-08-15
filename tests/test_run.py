import json
from pathlib import Path

import pytest

from akbc_baseline import run
from akbc_baseline.config import ModelConfig


class _PromptBuilder:
    def __init__(self, **kwargs: object) -> None:
        pass

    def build(
        self, subject: str, relation: str, candidate_index: int
    ) -> list[dict[str, str]]:
        return [{"role": "user", "content": f"{subject}:{relation}"}]

    def candidate_instruction_for(
        self, relation: str, candidate_index: int
    ) -> str | None:
        return None


class _Backend:
    generated_thinking: list[bool | None] = []

    def __init__(self, config: ModelConfig) -> None:
        self.generated_thinking = []
        _Backend.generated_thinking = self.generated_thinking
        pass

    def reset_peak_memory_stats(self) -> None:
        pass

    def peak_cuda_memory_gib(self) -> float:
        return 7.5

    def generate(
        self,
        messages: list[dict[str, str]],
        seed: int | None = None,
        enable_thinking: bool | None = None,
    ) -> str:
        self.generated_thinking.append(enable_thinking)
        return '["Paris"]'

    def last_generation_diagnostics(self) -> dict[str, object]:
        return {
            "generated_tokens": 10,
            "hit_token_limit": False,
            "has_think_end": True,
            "final_text_empty": False,
        }


class _ConcurrentBackend(_Backend):
    def generate_with_diagnostics(
        self,
        messages: list[dict[str, str]],
        seed: int | None = None,
        enable_thinking: bool | None = None,
    ) -> tuple[str, dict[str, object]]:
        text = self.generate(messages, seed=seed, enable_thinking=enable_thinking)
        return text, self.last_generation_diagnostics()


class _ConversationPromptBuilder(_PromptBuilder):
    instructions = ["recent", "old", "recent", "old"]

    def build(
        self, subject: str, relation: str, candidate_index: int
    ) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": "system"},
            {"role": "user", "content": self.instructions[candidate_index]},
        ]

    def conversation_followup(
        self, subject: str, relation: str, candidate_index: int
    ) -> dict[str, str]:
        return {
            "role": "user",
            "content": self.instructions[candidate_index],
        }

    def candidate_instruction_for(
        self, relation: str, candidate_index: int
    ) -> str | None:
        return self.instructions[candidate_index]


class _ConversationBackend(_Backend):
    calls: list[list[dict[str, str]]] = []

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.calls = []
        _ConversationBackend.calls = self.calls

    def generate(
        self,
        messages: list[dict[str, str]],
        seed: int | None = None,
        enable_thinking: bool | None = None,
    ) -> str:
        self.calls.append([dict(message) for message in messages])
        return json.dumps([f"Winner {len(self.calls)}"])


class _AuditPromptBuilder(_PromptBuilder):
    def area_audit_followup(self, subject: str) -> dict[str, str]:
        return {"role": "user", "content": f"audit:{subject}"}


class _AuditBackend(_Backend):
    calls: list[list[dict[str, str]]] = []

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.calls = []
        _AuditBackend.calls = self.calls

    def generate(
        self,
        messages: list[dict[str, str]],
        seed: int | None = None,
        enable_thinking: bool | None = None,
    ) -> str:
        self.calls.append([dict(message) for message in messages])
        if len(self.calls) == 1:
            return '["99"]'
        if len(self.calls) == 2:
            return (
                '["value=99","unit=square_mile",'
                '"scope=total_geographic_area","reference_year=general",'
                '"normalized_value_km2=256.41","keep=true"]'
            )
        if len(self.calls) == 3:
            return '["255"]'
        return (
            '["value=255","unit=square_kilometer",'
            '"scope=total_geographic_area","reference_year=general",'
            '"normalized_value_km2=255","keep=true"]'
        )


class _MetadataPromptBuilder(_PromptBuilder):
    def area_metadata_followup(self, subject: str) -> dict[str, str]:
        return {"role": "user", "content": f"metadata:{subject}"}

    def area_metadata_selection(
        self,
        subject: str,
        candidates: list[list[str]],
        diagnostics: list[dict[str, object]],
    ) -> list[dict[str, str]]:
        assert len(candidates) == 2
        assert all("metadata" in item for item in diagnostics)
        return [{"role": "user", "content": f"select:{subject}:{len(candidates)}"}]


class _MetadataBackend(_Backend):
    calls: list[list[dict[str, str]]] = []

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.calls = []
        _MetadataBackend.calls = self.calls

    def generate(
        self,
        messages: list[dict[str, str]],
        seed: int | None = None,
        enable_thinking: bool | None = None,
    ) -> str:
        self.calls.append([dict(message) for message in messages])
        outputs = [
            '["99"]',
            '["99","square_mile","land_area_only","general","256.41"]',
            '["255"]',
            '["255","square_kilometer","total_geographic_area","general","255"]',
            '["256.41"]',
        ]
        return outputs[len(self.calls) - 1]


class _ClusterPromptBuilder(_PromptBuilder):
    def area_cluster_selection(
        self, subject: str, clusters: list[object]
    ) -> list[dict[str, str]]:
        assert len(clusters) == 2
        return [{"role": "user", "content": f"cluster-select:{subject}"}]


class _ClusterBackend(_Backend):
    calls: list[list[dict[str, str]]] = []

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.calls = []
        _ClusterBackend.calls = self.calls

    def generate(
        self,
        messages: list[dict[str, str]],
        seed: int | None = None,
        enable_thinking: bool | None = None,
    ) -> str:
        self.calls.append([dict(message) for message in messages])
        return ['["100"]', '["102"]', '["200"]', '["200"]'][
            len(self.calls) - 1
        ]


def test_writes_runtime_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ModelConfig(
        name="test-model",
        model_id="test/model",
        backend="causal",
        parameter_count_billion=8,
        prompt_templates_file="prompts.csv",
        train_data_file="train.jsonl",
    )
    input_row = {
        "SubjectEntity": "Person",
        "Relation": "personHasCityOfDeath",
    }
    monkeypatch.setattr(
        run.ModelConfig,
        "from_yaml",
        classmethod(lambda cls, path: config),
    )
    monkeypatch.setattr(run, "read_jsonl", lambda path: [input_row])
    monkeypatch.setattr(run, "read_prompt_templates", lambda path: {})
    monkeypatch.setattr(run, "PromptBuilder", _PromptBuilder)
    monkeypatch.setattr(run, "create_backend", _Backend)

    prediction_path = tmp_path / "prediction.jsonl"
    metrics_path = tmp_path / "reports" / "metrics.json"
    assert run.main(
        [
            "--config",
            "unused.yaml",
            "--dataset-dir",
            str(tmp_path),
            "--input",
            "input.jsonl",
            "--output",
            str(prediction_path),
            "--metrics-output",
            str(metrics_path),
        ]
    ) == 0

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["rows"] == 1
    assert metrics["input_offset"] == 0
    assert metrics["empty_predictions"] == 0
    assert metrics["parameter_count_billion"] == 8
    assert metrics["peak_cuda_memory_gib"] == 7.5


def test_offset_limits_input_and_writes_candidate_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ModelConfig(
        name="test-model",
        model_id="test/model",
        backend="causal",
        prompt_templates_file="prompts.csv",
        train_data_file="train.jsonl",
        num_candidates=2,
        relation_thinking={"personHasCityOfDeath": False},
    )
    input_rows = [
        {
            "SubjectEntity": f"Person {index}",
            "Relation": "personHasCityOfDeath",
        }
        for index in range(4)
    ]
    monkeypatch.setattr(
        run.ModelConfig,
        "from_yaml",
        classmethod(lambda cls, path: config),
    )
    monkeypatch.setattr(
        run,
        "read_jsonl",
        lambda path: input_rows if str(path).endswith("input.jsonl") else [],
    )
    monkeypatch.setattr(run, "read_prompt_templates", lambda path: {})
    monkeypatch.setattr(run, "PromptBuilder", _PromptBuilder)
    monkeypatch.setattr(run, "create_backend", _Backend)

    prediction_path = tmp_path / "prediction.jsonl"
    candidates_path = tmp_path / "candidates.jsonl"
    assert run.main(
        [
            "--config",
            "unused.yaml",
            "--dataset-dir",
            str(tmp_path),
            "--input",
            "input.jsonl",
            "--output",
            str(prediction_path),
            "--candidates-output",
            str(candidates_path),
            "--offset",
            "1",
            "--limit",
            "2",
        ]
    ) == 0

    predictions = [
        json.loads(line)
        for line in prediction_path.read_text(encoding="utf-8").splitlines()
    ]
    candidates = [
        json.loads(line)
        for line in candidates_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["SubjectEntity"] for row in predictions] == [
        "Person 1",
        "Person 2",
    ]
    assert len(candidates) == 2
    assert len(candidates[0]["CandidateDiagnostics"]) == 2
    assert candidates[0]["CandidateDiagnostics"][0]["parse_status"] == "json"
    assert candidates[0]["CandidateDiagnostics"][0]["enable_thinking"] is False
    assert _Backend.generated_thinking == [False, False, False, False]


def test_conversation_candidates_keep_history_within_each_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ModelConfig(
        name="conversation",
        model_id="test/model",
        backend="causal",
        prompt_templates_file="prompts.csv",
        train_data_file="train.jsonl",
        num_candidates=4,
        candidate_instructions={
            "awardWonBy": ["recent", "old", "recent", "old"]
        },
        conversation_chains={"awardWonBy": 2},
        aggregation={
            "awardWonBy": {
                "strategy": "grouped_frequency",
                "groups": 2,
                "threshold": 0.5,
            }
        },
    )
    input_row = {"SubjectEntity": "Award", "Relation": "awardWonBy"}
    monkeypatch.setattr(
        run.ModelConfig,
        "from_yaml",
        classmethod(lambda cls, path: config),
    )
    monkeypatch.setattr(run, "read_jsonl", lambda path: [input_row])
    monkeypatch.setattr(run, "read_prompt_templates", lambda path: {})
    monkeypatch.setattr(run, "PromptBuilder", _ConversationPromptBuilder)
    monkeypatch.setattr(run, "create_backend", _ConversationBackend)

    candidates_path = tmp_path / "candidates.jsonl"
    assert run.main(
        [
            "--config",
            "unused.yaml",
            "--dataset-dir",
            str(tmp_path),
            "--input",
            "input.jsonl",
            "--output",
            str(tmp_path / "prediction.jsonl"),
            "--candidates-output",
            str(candidates_path),
        ]
    ) == 0

    calls = _ConversationBackend.calls
    assert len(calls) == 4
    assert calls[0] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "recent"},
    ]
    assert calls[1][-2:] == [
        {"role": "assistant", "content": '["Winner 1"]'},
        {"role": "user", "content": "old"},
    ]
    assert len(calls[2]) == 2
    assert calls[3][-2:] == [
        {"role": "assistant", "content": '["Winner 3"]'},
        {"role": "user", "content": "old"},
    ]
    saved = json.loads(candidates_path.read_text(encoding="utf-8").splitlines()[0])
    diagnostics = saved["CandidateDiagnostics"]
    assert [item["conversation_chain"] for item in diagnostics] == [0, 0, 1, 1]
    assert [item["conversation_turn"] for item in diagnostics] == [0, 1, 0, 1]


def test_area_candidate_audit_continues_each_independent_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ModelConfig(
        name="area-audit",
        model_id="test/model",
        backend="causal",
        prompt_templates_file="prompts.csv",
        train_data_file="train.jsonl",
        num_candidates=2,
        relation_thinking={"hasArea": False},
        candidate_audits={"hasArea": {"kind": "area_metadata"}},
        aggregation={"hasArea": {"strategy": "audited_median"}},
    )
    input_row = {"SubjectEntity": "Icaria", "Relation": "hasArea"}
    monkeypatch.setattr(
        run.ModelConfig,
        "from_yaml",
        classmethod(lambda cls, path: config),
    )
    monkeypatch.setattr(run, "read_jsonl", lambda path: [input_row])
    monkeypatch.setattr(run, "read_prompt_templates", lambda path: {})
    monkeypatch.setattr(run, "PromptBuilder", _AuditPromptBuilder)
    monkeypatch.setattr(run, "create_backend", _AuditBackend)

    prediction_path = tmp_path / "prediction.jsonl"
    candidates_path = tmp_path / "candidates.jsonl"
    assert run.main(
        [
            "--config",
            "unused.yaml",
            "--dataset-dir",
            str(tmp_path),
            "--input",
            "input.jsonl",
            "--output",
            str(prediction_path),
            "--candidates-output",
            str(candidates_path),
        ]
    ) == 0

    calls = _AuditBackend.calls
    assert len(calls) == 4
    assert calls[1][-2:] == [
        {"role": "assistant", "content": '["99"]'},
        {"role": "user", "content": "audit:Icaria"},
    ]
    assert len(calls[2]) == 1
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    assert prediction["ObjectEntities"] == ["255.705"]
    saved = json.loads(candidates_path.read_text(encoding="utf-8"))
    assert saved["Candidates"] == [["99"], ["255"]]
    assert all(
        diagnostic["audit"]["usable"]
        for diagnostic in saved["CandidateDiagnostics"]
    )
    assert all(
        "audit_generation_diagnostics" in diagnostic
        for diagnostic in saved["CandidateDiagnostics"]
    )


def test_area_metadata_is_descriptive_then_judged_in_separate_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ModelConfig(
        name="area-metadata-judge",
        model_id="test/model",
        backend="causal",
        prompt_templates_file="prompts.csv",
        train_data_file="train.jsonl",
        num_candidates=2,
        relation_thinking={"hasArea": False},
        candidate_audits={"hasArea": {"kind": "area_metadata_only"}},
        aggregation={"hasArea": {"strategy": "metadata_judge"}},
    )
    input_row = {"SubjectEntity": "Icaria", "Relation": "hasArea"}
    monkeypatch.setattr(
        run.ModelConfig,
        "from_yaml",
        classmethod(lambda cls, path: config),
    )
    monkeypatch.setattr(run, "read_jsonl", lambda path: [input_row])
    monkeypatch.setattr(run, "read_prompt_templates", lambda path: {})
    monkeypatch.setattr(run, "PromptBuilder", _MetadataPromptBuilder)
    monkeypatch.setattr(run, "create_backend", _MetadataBackend)

    prediction_path = tmp_path / "prediction.jsonl"
    candidates_path = tmp_path / "candidates.jsonl"
    assert run.main(
        [
            "--config",
            "unused.yaml",
            "--dataset-dir",
            str(tmp_path),
            "--input",
            "input.jsonl",
            "--output",
            str(prediction_path),
            "--candidates-output",
            str(candidates_path),
        ]
    ) == 0

    calls = _MetadataBackend.calls
    assert len(calls) == 5
    assert calls[1][-2:] == [
        {"role": "assistant", "content": '["99"]'},
        {"role": "user", "content": "metadata:Icaria"},
    ]
    assert calls[4] == [{"role": "user", "content": "select:Icaria:2"}]
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    assert prediction["ObjectEntities"] == ["256.41"]
    saved = json.loads(candidates_path.read_text(encoding="utf-8"))
    assert saved["Candidates"] == [["99"], ["255"]]
    assert all("metadata" in item for item in saved["CandidateDiagnostics"])
    assert all("audit" not in item for item in saved["CandidateDiagnostics"])
    assert saved["FinalSelection"]["ObjectEntities"] == ["256.41"]
    assert saved["FinalSelection"]["used_fallback"] is False


def test_area_metadata_resume_reuses_saved_final_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ModelConfig(
        name="area-metadata-resume",
        model_id="test/model",
        backend="causal",
        prompt_templates_file="prompts.csv",
        train_data_file="train.jsonl",
        num_candidates=2,
        relation_thinking={"hasArea": False},
        candidate_audits={"hasArea": {"kind": "area_metadata_only"}},
        aggregation={"hasArea": {"strategy": "metadata_judge"}},
    )
    input_row = {"SubjectEntity": "Icaria", "Relation": "hasArea"}
    saved_row = {
        **input_row,
        "Candidates": [["99"], ["255"]],
        "CandidateDiagnostics": [
            {"metadata": {"status": "parsed"}},
            {"metadata": {"status": "parsed"}},
        ],
        "FinalSelection": {
            "ObjectEntities": ["256.41"],
            "text": '["256.41"]',
            "parse_status": "json",
            "used_fallback": False,
            "generation_diagnostics": {},
        },
    }
    candidates_path = tmp_path / "candidates.jsonl"
    candidates_path.write_text(json.dumps(saved_row) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        run.ModelConfig,
        "from_yaml",
        classmethod(lambda cls, path: config),
    )
    monkeypatch.setattr(
        run,
        "read_jsonl",
        lambda path: (
            [input_row]
            if str(path).endswith("input.jsonl")
            else [saved_row]
            if Path(path) == candidates_path
            else []
        ),
    )
    monkeypatch.setattr(run, "read_prompt_templates", lambda path: {})
    monkeypatch.setattr(run, "PromptBuilder", _MetadataPromptBuilder)
    monkeypatch.setattr(run, "create_backend", _MetadataBackend)

    prediction_path = tmp_path / "prediction.jsonl"
    assert run.main(
        [
            "--config",
            "unused.yaml",
            "--dataset-dir",
            str(tmp_path),
            "--input",
            "input.jsonl",
            "--output",
            str(prediction_path),
            "--candidates-output",
            str(candidates_path),
            "--resume",
        ]
    ) == 0

    assert _MetadataBackend.calls == []
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    assert prediction["ObjectEntities"] == ["256.41"]
    assert len(candidates_path.read_text(encoding="utf-8").splitlines()) == 1


def test_area_cluster_choice_generates_candidates_then_selects_a_cluster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ModelConfig(
        name="area-cluster-choice",
        model_id="test/model",
        backend="causal",
        prompt_templates_file="prompts.csv",
        train_data_file="train.jsonl",
        num_candidates=3,
        relation_thinking={"hasArea": False},
        aggregation={
            "hasArea": {"strategy": "cluster_choice", "tolerance": 0.05}
        },
    )
    input_row = {"SubjectEntity": "Example Island", "Relation": "hasArea"}
    monkeypatch.setattr(
        run.ModelConfig,
        "from_yaml",
        classmethod(lambda cls, path: config),
    )
    monkeypatch.setattr(run, "read_jsonl", lambda path: [input_row])
    monkeypatch.setattr(run, "read_prompt_templates", lambda path: {})
    monkeypatch.setattr(run, "PromptBuilder", _ClusterPromptBuilder)
    monkeypatch.setattr(run, "create_backend", _ClusterBackend)
    prediction_path = tmp_path / "prediction.jsonl"
    candidates_path = tmp_path / "candidates.jsonl"

    assert run.main(
        [
            "--config",
            "unused.yaml",
            "--dataset-dir",
            str(tmp_path),
            "--input",
            "input.jsonl",
            "--output",
            str(prediction_path),
            "--candidates-output",
            str(candidates_path),
        ]
    ) == 0

    assert len(_ClusterBackend.calls) == 4
    assert _ClusterBackend.calls[-1] == [
        {"role": "user", "content": "cluster-select:Example Island"}
    ]
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    assert prediction["ObjectEntities"] == ["200"]
    saved = json.loads(candidates_path.read_text(encoding="utf-8"))
    assert saved["FinalSelection"]["strategy"] == "cluster_choice"
    assert saved["FinalSelection"]["used_fallback"] is False
    assert len(saved["FinalSelection"]["choices"]) == 2


def _base_config(**overrides: object) -> ModelConfig:
    fields = dict(
        name="test-model",
        model_id="test/model",
        backend="causal",
        prompt_templates_file="prompts.csv",
        train_data_file="train.jsonl",
        num_candidates=3,
    )
    fields.update(overrides)
    return ModelConfig(**fields)


def _run_with(monkeypatch, tmp_path, config, backend_cls, extra_args):
    input_rows = [
        {"SubjectEntity": "Paris City", "Relation": "personHasCityOfDeath"}
    ]
    monkeypatch.setattr(
        run.ModelConfig, "from_yaml", classmethod(lambda cls, path: config)
    )
    monkeypatch.setattr(
        run,
        "read_jsonl",
        lambda path: input_rows if str(path).endswith("input.jsonl") else [],
    )
    monkeypatch.setattr(run, "read_prompt_templates", lambda path: {})
    monkeypatch.setattr(run, "PromptBuilder", _PromptBuilder)
    monkeypatch.setattr(run, "create_backend", backend_cls)
    prediction_path = tmp_path / "prediction.jsonl"
    candidates_path = tmp_path / "candidates.jsonl"
    code = run.main(
        [
            "--config", "unused.yaml",
            "--dataset-dir", str(tmp_path),
            "--input", "input.jsonl",
            "--output", str(prediction_path),
            "--candidates-output", str(candidates_path),
            *extra_args,
        ]
    )
    candidates = [
        json.loads(line)
        for line in candidates_path.read_text(encoding="utf-8").splitlines()
    ]
    return code, candidates


def test_concurrent_generation_matches_sequential_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _base_config()
    _, sequential = _run_with(
        monkeypatch, tmp_path / "seq", config, _ConcurrentBackend, []
    )
    (tmp_path / "seq").mkdir(exist_ok=True)
    _, concurrent = _run_with(
        monkeypatch, tmp_path / "par", config, _ConcurrentBackend,
        ["--concurrency", "3"],
    )

    assert sequential[0]["Candidates"] == concurrent[0]["Candidates"]
    assert len(concurrent[0]["CandidateDiagnostics"]) == 3
    for diagnostics in concurrent[0]["CandidateDiagnostics"]:
        assert diagnostics["parse_status"] == "json"
        assert diagnostics["conversation_chain"] is None


def test_concurrency_rejects_invalid_values_and_configs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="--concurrency must be positive"):
        _run_with(
            monkeypatch, tmp_path, _base_config(), _ConcurrentBackend,
            ["--concurrency", "0"],
        )
    with pytest.raises(ValueError, match="conversation_chains"):
        _run_with(
            monkeypatch, tmp_path, _base_config(
                num_candidates=4,
                conversation_chains={"personHasCityOfDeath": 2},
            ),
            _ConcurrentBackend, ["--concurrency", "2"],
        )
    with pytest.raises(ValueError, match="llama_cpp_server backend"):
        _run_with(
            monkeypatch, tmp_path, _base_config(), _Backend,
            ["--concurrency", "2"],
        )
