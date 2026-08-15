import json
from pathlib import Path

import pytest

from akbc_baseline.data import read_jsonl, write_jsonl
from akbc_baseline.merge_shards import main, merge_rows


def _row(index: int) -> dict[str, object]:
    return {
        "SubjectEntity": f"Subject {index}",
        "Relation": "personHasCityOfDeath",
        "ObjectEntities": [f"City {index}"],
    }


def test_merges_ordered_prediction_shards(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    expected = [_row(index) for index in range(3)]
    write_jsonl("input.jsonl", expected)
    write_jsonl("outputs/model-val-shard-000.jsonl", expected[:2])
    write_jsonl("outputs/model-val-shard-001.jsonl", expected[2:])
    candidate_rows = [
        {
            "SubjectEntity": row["SubjectEntity"],
            "Relation": row["Relation"],
            "Candidates": [row["ObjectEntities"]],
        }
        for row in expected
    ]
    write_jsonl("outputs/model-candidates-val-shard-000.jsonl", candidate_rows[:2])
    write_jsonl("outputs/model-candidates-val-shard-001.jsonl", candidate_rows[2:])
    for index, (offset, rows) in enumerate(((0, 2), (2, 1))):
        path = Path(f"reports/model-val-shard-{index:03d}-metrics.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "name": "model",
                    "model_id": "test/model",
                    "parameter_count_billion": 8,
                    "num_candidates": 20,
                    "input_offset": offset,
                    "rows": rows,
                    "empty_predictions": 0,
                    "elapsed_seconds": rows * 10,
                    "peak_cuda_memory_gib": 8 + index,
                }
            ),
            encoding="utf-8",
        )

    assert main(
        [
            "--input",
            "input.jsonl",
            "--prediction-glob",
            "outputs/model-val-shard-*.jsonl",
            "--candidate-glob",
            "outputs/model-candidates-val-shard-*.jsonl",
            "--metrics-glob",
            "reports/model-val-shard-*-metrics.json",
            "--prediction-output",
            "outputs/model-val.jsonl",
            "--candidate-output",
            "outputs/model-candidates-val.jsonl",
            "--metrics-output",
            "reports/model-val-metrics.json",
        ]
    ) == 0
    assert read_jsonl("outputs/model-val.jsonl") == expected
    metrics = json.loads(
        Path("reports/model-val-metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["rows"] == 3
    assert metrics["shards"] == 2
    assert metrics["elapsed_seconds"] == 30
    assert metrics["peak_cuda_memory_gib"] == 9


def test_merge_rejects_out_of_order_shards(tmp_path: Path) -> None:
    expected = [_row(0), _row(1)]
    shard = tmp_path / "shard.jsonl"
    write_jsonl(shard, list(reversed(expected)))

    with pytest.raises(ValueError, match="identity/order mismatch"):
        merge_rows(expected, [shard], prediction=True)
