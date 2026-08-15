import json
from pathlib import Path

import pytest

from akbc_baseline import area_cluster_rerank
from akbc_baseline.area_clustering import (
    area_cluster_selection_messages,
    cluster_choice_records,
    cluster_numeric_candidates,
    match_cluster_choice,
)
from akbc_baseline.config import ModelConfig


def test_builds_disjoint_five_percent_choices_with_dominant_first() -> None:
    clusters = cluster_numeric_candidates(
        [["100"], ["102"], ["104"], ["200"], ["204"], ["1000"]]
    )

    assert [cluster.representative for cluster in clusters] == [102, 202, 1000]
    assert [cluster.support for cluster in clusters] == [3, 2, 1]
    assert cluster_choice_records(clusters) == [
        {
            "choice_id": 1,
            "value_km2": "102",
            "support": 3,
            "minimum_km2": "100",
            "maximum_km2": "104",
            "dominant": True,
        },
        {
            "choice_id": 2,
            "value_km2": "202",
            "support": 2,
            "minimum_km2": "200",
            "maximum_km2": "204",
            "dominant": False,
        },
        {
            "choice_id": 3,
            "value_km2": "1000",
            "support": 1,
            "minimum_km2": "1000",
            "maximum_km2": "1000",
            "dominant": False,
        },
    ]
    assert match_cluster_choice(["202"], clusters) == clusters[1]
    assert match_cluster_choice(["201"], clusters) is None


def test_cluster_selection_prompt_requires_copying_one_choice() -> None:
    clusters = cluster_numeric_candidates([["100"], ["200"]])
    messages = area_cluster_selection_messages("Example Island", clusters)

    assert "total geographic area" in messages[1]["content"]
    assert "1. 100 km2" in messages[1]["content"]
    assert "2. 200 km2" in messages[1]["content"]
    full_prompt = "\n".join(message["content"] for message in messages)
    for hidden_signal in (
        "cluster",
        "support",
        "dominant",
        "minimum",
        "maximum",
        "20 independently",
    ):
        assert hidden_signal not in full_prompt.lower()


class _Backend:
    calls: list[list[dict[str, str]]] = []

    def __init__(self, config: ModelConfig) -> None:
        self.calls = []
        _Backend.calls = self.calls

    def reset_peak_memory_stats(self) -> None:
        pass

    def peak_cuda_memory_gib(self) -> float:
        return 1.5

    def generate(
        self,
        messages: list[dict[str, str]],
        seed: int | None = None,
        enable_thinking: bool | None = None,
    ) -> str:
        self.calls.append(messages)
        return '["200"]'

    def last_generation_diagnostics(self) -> dict[str, object]:
        return {"hit_token_limit": False}


def test_rerank_skips_single_cluster_and_asks_only_for_multiple_choices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ModelConfig(
        name="cluster-choice",
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
    rows = [
        {
            "SubjectEntity": "One cluster",
            "Relation": "hasArea",
            "Candidates": [["100"], ["101"], ["102"]],
            "CandidateDiagnostics": [{}, {}, {}],
        },
        {
            "SubjectEntity": "Two clusters",
            "Relation": "hasArea",
            "Candidates": [["100"], ["102"], ["200"]],
            "CandidateDiagnostics": [{}, {}, {}],
            "FinalSelection": {"ObjectEntities": ["100"]},
        },
    ]
    input_path = tmp_path / "source.jsonl"
    input_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    monkeypatch.setattr(
        area_cluster_rerank.ModelConfig,
        "from_yaml",
        classmethod(lambda cls, path: config),
    )
    monkeypatch.setattr(area_cluster_rerank, "create_backend", _Backend)
    prediction_path = tmp_path / "prediction.jsonl"
    candidates_path = tmp_path / "candidates.jsonl"
    metrics_path = tmp_path / "metrics.json"

    assert area_cluster_rerank.main(
        [
            "--config",
            "unused.yaml",
            "--candidates",
            str(input_path),
            "--output",
            str(prediction_path),
            "--candidates-output",
            str(candidates_path),
            "--metrics-output",
            str(metrics_path),
        ]
    ) == 0

    predictions = [
        json.loads(line)
        for line in prediction_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["ObjectEntities"] for row in predictions] == [["101"], ["200"]]
    assert len(_Backend.calls) == 1
    saved = [
        json.loads(line)
        for line in candidates_path.read_text(encoding="utf-8").splitlines()
    ]
    assert saved[0]["FinalSelection"]["skipped_single_cluster"] is True
    assert saved[1]["FinalSelection"]["used_fallback"] is False
    assert saved[1]["SourceFinalSelection"] == {"ObjectEntities": ["100"]}
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["selection_calls"] == 1
    assert metrics["single_cluster_skips"] == 1
