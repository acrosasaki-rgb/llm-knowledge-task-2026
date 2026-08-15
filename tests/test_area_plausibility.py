import json
import math
from pathlib import Path

import pytest

from akbc_baseline import area_plausibility_rerank
from akbc_baseline.area_plausibility import (
    build_area_hypotheses,
    decimal_relation_records,
    dimension_estimate_messages,
    parse_dimension_estimate,
    parse_scale_estimate,
    scale_estimate_messages,
    select_area_hypothesis,
)
from akbc_baseline.config import ModelConfig


def test_candidate_blind_prompts_do_not_contain_area_choices() -> None:
    scale = scale_estimate_messages("Hashima Island")
    dimensions = dimension_estimate_messages("Hashima Island")

    assert "Without seeing any candidate" in scale[1]["content"]
    assert "Do not state or calculate" in dimensions[0]["content"]
    assert "Hashima Island" in scale[1]["content"]
    assert "Hashima Island" in dimensions[1]["content"]


def test_parses_scale_and_derives_dimension_area_range() -> None:
    scale = parse_scale_estimate(
        '["entity_type=island","area_bin=0.01_0.1"]'
    )
    dimensions = parse_dimension_estimate(
        '["length_km=0.48","width_km=0.16","shape=elongated"]'
    )

    assert scale == {
        "status": "parsed",
        "entity_type": "island",
        "area_bin": "0.01_0.1",
        "lower_km2": 0.01,
        "upper_km2": 0.1,
    }
    assert dimensions["status"] == "parsed"
    assert math.isclose(dimensions["lower_km2"], 0.00768)
    assert math.isclose(dimensions["upper_km2"], 0.04608)


def test_expands_units_and_detects_decimal_relations() -> None:
    clusters, hypotheses = build_area_hypotheses(
        [["0.063"], ["0.63"], ["800"]]
    )

    assert len(hypotheses) == 1 + 3 * 4
    square_mile = next(
        hypothesis
        for hypothesis in hypotheses
        if hypothesis.source_value == 800
        and hypothesis.assumed_unit == "square_mile"
    )
    assert math.isclose(square_mile.normalized_value_km2, 2071.9904882688)
    assert decimal_relation_records(clusters) == [
        {
            "lower": "0.063",
            "higher": "0.63",
            "power_of_ten": 1,
            "ratio": 10.0,
        }
    ]


def test_scale_evidence_conservatively_corrects_decimal_direction() -> None:
    _, hypotheses = build_area_hypotheses(
        [["0.63"], ["0.63"], ["0.063"]]
    )

    selected, details = select_area_hypothesis(
        hypotheses,
        scale_bounds=(0.01, 0.1),
        dimension_bounds=None,
        allow_unit_conversion=False,
    )

    assert selected is not None
    assert selected.normalized_value_km2 == 0.063
    assert selected.assumed_unit == "square_kilometer"
    assert details["overrode_base"] is True


def test_missing_evidence_keeps_dominant_direct_value() -> None:
    _, hypotheses = build_area_hypotheses([["10"], ["10"], ["100"]])

    selected, details = select_area_hypothesis(
        hypotheses,
        scale_bounds=None,
        dimension_bounds=None,
        allow_unit_conversion=True,
    )

    assert selected is not None
    assert selected.normalized_value_km2 == 10
    assert selected.assumed_unit == "square_kilometer"
    assert details["overrode_base"] is False


class _Backend:
    def __init__(self, config: ModelConfig) -> None:
        self.calls = 0

    def reset_peak_memory_stats(self) -> None:
        pass

    def peak_cuda_memory_gib(self) -> float:
        return 2.0

    def generate(
        self,
        messages: list[dict[str, str]],
        seed: int | None = None,
        enable_thinking: bool | None = None,
    ) -> str:
        self.calls += 1
        if self.calls == 1:
            return '["entity_type=island","area_bin=0.01_0.1"]'
        return '["length_km=0.48","width_km=0.16","shape=elongated"]'

    def last_generation_diagnostics(self) -> dict[str, object]:
        return {"hit_token_limit": False}


def test_rerank_combines_blind_scale_dimensions_and_unit_hypotheses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ModelConfig(
        name="plausibility",
        model_id="test/model",
        backend="causal",
        prompt_templates_file="prompts.csv",
        train_data_file="train.jsonl",
        num_candidates=3,
        aggregation={"hasArea": {"strategy": "median"}},
    )
    candidate = {
        "SubjectEntity": "Hashima Island",
        "Relation": "hasArea",
        "Candidates": [["0.63"], ["0.63"], ["0.063"]],
        "CandidateDiagnostics": [{}, {}, {}],
    }
    input_path = tmp_path / "source.jsonl"
    input_path.write_text(json.dumps(candidate) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        area_plausibility_rerank.ModelConfig,
        "from_yaml",
        classmethod(lambda cls, path: config),
    )
    monkeypatch.setattr(area_plausibility_rerank, "create_backend", _Backend)
    output_path = tmp_path / "prediction.jsonl"
    candidates_path = tmp_path / "candidates.jsonl"
    metrics_path = tmp_path / "metrics.json"

    assert area_plausibility_rerank.main(
        [
            "--config",
            "unused.yaml",
            "--candidates",
            str(input_path),
            "--output",
            str(output_path),
            "--candidates-output",
            str(candidates_path),
            "--metrics-output",
            str(metrics_path),
        ]
    ) == 0

    prediction = json.loads(output_path.read_text(encoding="utf-8"))
    assert prediction["ObjectEntities"] == ["0.063"]
    saved = json.loads(candidates_path.read_text(encoding="utf-8"))
    assert saved["ScaleEstimate"]["parsed"]["area_bin"] == "0.01_0.1"
    assert saved["FinalSelection"]["decimal_relations"][0]["power_of_ten"] == 1
    assert (
        saved["FinalSelection"]["selections"]["combined-unit"][
            "overrode_base"
        ]
        is True
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["scale_estimates_parsed"] == 1
    assert metrics["dimension_estimates_parsed"] == 1
