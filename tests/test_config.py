from pathlib import Path

import pytest
import yaml

from akbc_baseline.config import ModelConfig


def test_loads_checked_in_configs() -> None:
    for path in Path("configs").glob("*.yaml"):
        config = ModelConfig.from_yaml(path)
        assert config.name
        assert config.model_id


def test_rejects_negative_few_shot(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        "name: bad\nmodel_id: bad/model\nbackend: causal\n"
        "prompt_templates_file: prompts.csv\ntrain_data_file: train.jsonl\n"
        "few_shot: -1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="few_shot"):
        ModelConfig.from_yaml(config_path)


def test_rejects_invalid_candidate_count_and_aggregation(tmp_path: Path) -> None:
    common = (
        "name: bad\nmodel_id: bad/model\nbackend: causal\n"
        "prompt_templates_file: prompts.csv\ntrain_data_file: train.jsonl\n"
    )
    config_path = tmp_path / "bad-candidates.yaml"
    config_path.write_text(common + "num_candidates: 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="num_candidates"):
        ModelConfig.from_yaml(config_path)

    config_path.write_text(
        common
        + "aggregation:\n"
        + "  relation:\n"
        + "    strategy: frequency\n"
        + "    threshold: 0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="threshold"):
        ModelConfig.from_yaml(config_path)

    config_path.write_text(
        common
        + "num_candidates: 2\n"
        + "candidate_instructions:\n"
        + "  relation: ['only one']\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly 2"):
        ModelConfig.from_yaml(config_path)

    config_path.write_text(
        common
        + "aggregation:\n"
        + "  hasArea:\n"
        + "    strategy: metadata_judge\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="area_metadata_only"):
        ModelConfig.from_yaml(config_path)


def test_enforces_shared_task_parameter_limit(tmp_path: Path) -> None:
    config_path = tmp_path / "too-large.yaml"
    config_path.write_text(
        "name: bad\nmodel_id: bad/model\nbackend: causal\n"
        "prompt_templates_file: prompts.csv\ntrain_data_file: train.jsonl\n"
        "parameter_count_billion: 33\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="32B"):
        ModelConfig.from_yaml(config_path)


def test_rejects_empty_quantization_backend(tmp_path: Path) -> None:
    config_path = tmp_path / "empty-backend.yaml"
    config_path.write_text(
        "name: bad\nmodel_id: bad/model\nbackend: causal\n"
        "prompt_templates_file: prompts.csv\ntrain_data_file: train.jsonl\n"
        "quantization_backend: ' '\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="quantization_backend"):
        ModelConfig.from_yaml(config_path)


def test_qwen_27b_uses_llama_cpp_gguf_backend() -> None:
    config = ModelConfig.from_yaml("configs/experiment-qwen-3.5-27b.yaml")

    assert config.backend == "llama_cpp_server"
    assert config.model_id == "unsloth/Qwen3.5-27B-GGUF"
    assert config.quantization_backend is None


def test_rejects_non_http_llama_cpp_url(tmp_path: Path) -> None:
    config_path = tmp_path / "bad-llama-url.yaml"
    config_path.write_text(
        "name: bad\nmodel_id: bad/model\nbackend: llama_cpp_server\n"
        "prompt_templates_file: prompts.csv\ntrain_data_file: train.jsonl\n"
        "llama_cpp_url: file:///tmp/server\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="HTTP"):
        ModelConfig.from_yaml(config_path)


def test_two_stage_generation_requires_thinking_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "bad-two-stage.yaml"
    config_path.write_text(
        "name: bad\nmodel_id: bad/model\nbackend: causal\n"
        "prompt_templates_file: prompts.csv\ntrain_data_file: train.jsonl\n"
        "final_answer_tokens: 64\nenable_thinking: false\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="enable_thinking"):
        ModelConfig.from_yaml(config_path)


def test_relation_aware_qwen_27b_config_declares_thinking_and_empty_policies() -> None:
    config = ModelConfig.from_yaml(
        "configs/experiment-qwen-3.5-27b-mtp-relation-aware.yaml"
    )

    assert config.num_candidates == 5
    assert config.relation_thinking == {
        "countryLandBordersCountry": True,
        "companyTradesAtStockExchange": True,
        "hasArea": True,
        "personHasCityOfDeath": False,
        "hasCapacity": False,
        "awardWonBy": False,
    }
    assert config.aggregation["countryLandBordersCountry"]["empty_aware"] is True
    assert config.aggregation["companyTradesAtStockExchange"]["empty_aware"] is True
    assert (
        config.aggregation["personHasCityOfDeath"]["explicit_empty_only"] is True
    )


def test_relation_aware_twenty_candidate_submission_config_is_separate() -> None:
    config = ModelConfig.from_yaml(
        "configs/experiment-qwen-3.5-27b-mtp-relation-aware-20.yaml"
    )

    assert config.num_candidates == 20
    assert config.aggregation["awardWonBy"] == {
        "strategy": "frequency",
        "threshold": 0.4,
    }
    assert config.relation_thinking["awardWonBy"] is False


def test_no_thinking_ablation_only_changes_reasoning_controls() -> None:
    thinking_path = Path(
        "configs/experiment-qwen-3.5-27b-mtp-relation-aware-20.yaml"
    )
    no_thinking_path = Path(
        "configs/experiment-qwen-3.5-27b-mtp-relation-aware-20-no-thinking.yaml"
    )
    thinking = yaml.safe_load(thinking_path.read_text(encoding="utf-8"))
    no_thinking = yaml.safe_load(no_thinking_path.read_text(encoding="utf-8"))

    assert no_thinking["model_id"] == thinking["model_id"]
    assert no_thinking["num_candidates"] == thinking["num_candidates"] == 20
    assert no_thinking["enable_thinking"] is False
    assert set(no_thinking["relation_thinking"].values()) == {False}
    assert no_thinking["max_new_tokens"] == 128

    allowed_differences = {
        "name",
        "max_new_tokens",
        "enable_thinking",
        "relation_thinking",
    }
    assert {
        key: value
        for key, value in no_thinking.items()
        if key not in allowed_differences
    } == {
        key: value
        for key, value in thinking.items()
        if key not in allowed_differences
    }


def test_award_era_config_partitions_twenty_non_thinking_candidates() -> None:
    config = ModelConfig.from_yaml(
        "configs/experiment-qwen-3.5-27b-mtp-award-era-no-thinking.yaml"
    )

    assert config.num_candidates == 20
    assert config.max_new_tokens == 128
    assert config.enable_thinking is False
    assert config.relation_thinking["awardWonBy"] is False
    assert len(config.candidate_instructions["awardWonBy"]) == 20
    assert "before 1800" in config.candidate_instructions["awardWonBy"][0]
    assert "2026 or later" in config.candidate_instructions["awardWonBy"][-1]
    assert config.aggregation["awardWonBy"] == {"strategy": "union"}


def test_award_reverse_conversation_config_uses_two_chains_of_ten() -> None:
    config = ModelConfig.from_yaml(
        "configs/experiment-qwen-3.5-27b-mtp-award-reverse-conversation.yaml"
    )

    assert config.num_candidates == 20
    assert config.enable_thinking is False
    assert config.conversation_chains == {"awardWonBy": 2}
    instructions = config.candidate_instructions["awardWonBy"]
    assert len(instructions) == 20
    assert instructions[:10] == instructions[10:]
    assert "2020 through 2026" in instructions[0]
    assert "before 1900" in instructions[9]
    assert config.aggregation["awardWonBy"] == {
        "strategy": "grouped_frequency",
        "groups": 10,
        "threshold": 0.5,
    }


def test_hasarea_conversation_audit_config_is_validation_ablation() -> None:
    config = ModelConfig.from_yaml(
        "configs/experiment-qwen-3.5-27b-mtp-hasarea-conversation-audit.yaml"
    )

    assert config.num_candidates == 20
    assert config.enable_thinking is False
    assert config.relation_thinking["hasArea"] is False
    assert config.candidate_audits == {
        "hasArea": {"kind": "area_metadata"}
    }
    assert config.aggregation["hasArea"] == {"strategy": "audited_median"}

    baseline = yaml.safe_load(
        Path(
            "configs/experiment-qwen-3.5-27b-mtp-relation-aware-20-no-thinking.yaml"
        ).read_text(encoding="utf-8")
    )
    audit = yaml.safe_load(
        Path(
            "configs/experiment-qwen-3.5-27b-mtp-hasarea-conversation-audit.yaml"
        ).read_text(encoding="utf-8")
    )
    assert {
        key: value
        for key, value in audit.items()
        if key not in {"name", "candidate_audits", "aggregation"}
    } == {
        key: value
        for key, value in baseline.items()
        if key not in {"name", "aggregation"}
    }
    assert {
        relation: policy
        for relation, policy in audit["aggregation"].items()
        if relation != "hasArea"
    } == {
        relation: policy
        for relation, policy in baseline["aggregation"].items()
        if relation != "hasArea"
    }


def test_hasarea_metadata_judge_config_preserves_initial_generation_settings() -> None:
    config = ModelConfig.from_yaml(
        "configs/experiment-qwen-3.5-27b-mtp-hasarea-metadata-judge.yaml"
    )

    assert config.num_candidates == 20
    assert config.enable_thinking is False
    assert config.relation_thinking["hasArea"] is False
    assert config.candidate_audits == {
        "hasArea": {"kind": "area_metadata_only"}
    }
    assert config.aggregation["hasArea"] == {"strategy": "metadata_judge"}

    baseline = yaml.safe_load(
        Path(
            "configs/experiment-qwen-3.5-27b-mtp-relation-aware-20-no-thinking.yaml"
        ).read_text(encoding="utf-8")
    )
    experiment = yaml.safe_load(
        Path(
            "configs/experiment-qwen-3.5-27b-mtp-hasarea-metadata-judge.yaml"
        ).read_text(encoding="utf-8")
    )
    assert {
        key: value
        for key, value in experiment.items()
        if key not in {"name", "candidate_audits", "aggregation"}
    } == {
        key: value
        for key, value in baseline.items()
        if key not in {"name", "aggregation"}
    }
    assert {
        relation: policy
        for relation, policy in experiment["aggregation"].items()
        if relation != "hasArea"
    } == {
        relation: policy
        for relation, policy in baseline["aggregation"].items()
        if relation != "hasArea"
    }


def test_hasarea_cluster_choice_config_preserves_initial_generation_settings() -> None:
    config = ModelConfig.from_yaml(
        "configs/experiment-qwen-3.5-27b-mtp-hasarea-cluster-choice.yaml"
    )

    assert config.num_candidates == 20
    assert config.enable_thinking is False
    assert config.relation_thinking["hasArea"] is False
    assert config.candidate_audits == {}
    assert config.aggregation["hasArea"] == {
        "strategy": "cluster_choice",
        "tolerance": 0.05,
    }

    baseline = yaml.safe_load(
        Path(
            "configs/experiment-qwen-3.5-27b-mtp-relation-aware-20-no-thinking.yaml"
        ).read_text(encoding="utf-8")
    )
    experiment = yaml.safe_load(
        Path(
            "configs/experiment-qwen-3.5-27b-mtp-hasarea-cluster-choice.yaml"
        ).read_text(encoding="utf-8")
    )
    assert {
        key: value
        for key, value in experiment.items()
        if key not in {"name", "aggregation"}
    } == {
        key: value
        for key, value in baseline.items()
        if key not in {"name", "aggregation"}
    }
    assert {
        relation: policy
        for relation, policy in experiment["aggregation"].items()
        if relation != "hasArea"
    } == {
        relation: policy
        for relation, policy in baseline["aggregation"].items()
        if relation != "hasArea"
    }


def test_thinking_empty_aware_config_only_changes_aggregation() -> None:
    path = "configs/experiment-qwen-3.5-27b-mtp-thinking-empty-aware.yaml"
    config = ModelConfig.from_yaml(path)

    assert config.enable_thinking is True
    assert config.num_candidates == 5
    assert config.aggregation == {
        "awardWonBy": {"strategy": "frequency", "threshold": 0.2},
        "companyTradesAtStockExchange": {
            "strategy": "frequency",
            "threshold": 0.4,
            "empty_aware": True,
            "empty_majority": 3,
        },
        "countryLandBordersCountry": {
            "strategy": "frequency",
            "threshold": 0.4,
            "empty_aware": True,
            "empty_majority": 3,
        },
        "hasArea": {
            "strategy": "unit_equivalence",
            "cluster_tolerance": 0.05,
            "unit_tolerance": 0.05,
        },
        "hasCapacity": {"strategy": "median"},
        "personHasCityOfDeath": {
            "strategy": "majority",
            "explicit_empty_only": True,
        },
    }

    base = yaml.safe_load(
        Path("configs/experiment-qwen-3.5-27b-mtp-thinking.yaml").read_text(
            encoding="utf-8"
        )
    )
    updated = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    assert set(updated) == set(base)
    for key in base:
        if key in {"name", "aggregation"}:
            continue
        assert updated[key] == base[key], key


def test_unit_equivalence_config_only_changes_hasarea_aggregation() -> None:
    config = ModelConfig.from_yaml(
        "configs/experiment-qwen-3.5-27b-mtp-relation-aware-20-no-thinking"
        "-unit-equivalence.yaml"
    )

    assert config.num_candidates == 20
    assert config.enable_thinking is False
    assert config.aggregation["hasArea"] == {
        "strategy": "unit_equivalence",
        "cluster_tolerance": 0.05,
        "unit_tolerance": 0.05,
    }

    baseline = yaml.safe_load(
        Path(
            "configs/experiment-qwen-3.5-27b-mtp-relation-aware-20-no-thinking.yaml"
        ).read_text(encoding="utf-8")
    )
    experiment = yaml.safe_load(
        Path(
            "configs/experiment-qwen-3.5-27b-mtp-relation-aware-20-no-thinking"
            "-unit-equivalence.yaml"
        ).read_text(encoding="utf-8")
    )
    assert baseline["aggregation"]["hasArea"] == {"strategy": "median"}
    for key in set(baseline) | set(experiment):
        if key in {"name", "aggregation"}:
            continue
        assert experiment[key] == baseline[key], key
    assert {
        relation: policy
        for relation, policy in experiment["aggregation"].items()
        if relation != "hasArea"
    } == {
        relation: policy
        for relation, policy in baseline["aggregation"].items()
        if relation != "hasArea"
    }


def test_hasarea_plausibility_config_matches_no_thinking_baseline() -> None:
    config = ModelConfig.from_yaml(
        "configs/experiment-qwen-3.5-27b-mtp-hasarea-plausibility.yaml"
    )

    assert config.num_candidates == 20
    assert config.enable_thinking is False
    assert config.aggregation["hasArea"] == {"strategy": "median"}

    baseline = yaml.safe_load(
        Path(
            "configs/experiment-qwen-3.5-27b-mtp-relation-aware-20-no-thinking.yaml"
        ).read_text(encoding="utf-8")
    )
    experiment = yaml.safe_load(
        Path(
            "configs/experiment-qwen-3.5-27b-mtp-hasarea-plausibility.yaml"
        ).read_text(encoding="utf-8")
    )
    assert {
        key: value for key, value in experiment.items() if key != "name"
    } == {
        key: value for key, value in baseline.items() if key != "name"
    }
