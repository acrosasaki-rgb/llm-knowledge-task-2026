import json
from pathlib import Path

import pytest

from akbc_baseline import reaggregate


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _config(tmp_path: Path, hasarea_strategy: str = "unit_equivalence") -> Path:
    return _text(
        tmp_path / "config.yaml",
        f"""
name: test
model_id: test/model
backend: causal
prompt_templates_file: prompts.csv
train_data_file: train.jsonl
num_candidates: 3
aggregation:
  hasArea:
    strategy: {hasarea_strategy}
  countryLandBordersCountry:
    strategy: frequency
    threshold: 0.4
""",
    )


def _text(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


CANDIDATE_ROWS = [
    {
        "SubjectEntity": "Rùm",
        "Relation": "hasArea",
        "Candidates": [["37.65"], ["37.65"], ["37.65"], ["37.65"], ["101.6"],
                       ["102.12"], ["104"]],
    },
    {
        "SubjectEntity": "France",
        "Relation": "countryLandBordersCountry",
        "Candidates": [["Spain", "Italy"], ["Spain"], ["Spain", "Italy"]],
    },
]


def test_reaggregates_hasarea_with_unit_equivalence(tmp_path: Path) -> None:
    config = _config(tmp_path)
    candidates = _write(tmp_path / "candidates.jsonl", CANDIDATE_ROWS)
    output = tmp_path / "predictions.jsonl"

    assert reaggregate.main(
        ["--config", str(config), "--candidates", str(candidates),
         "--output", str(output)]
    ) == 0

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["ObjectEntities"] for row in rows] == [["102.12"], ["Spain", "Italy"]]
    assert set(rows[0]) == {"SubjectEntity", "Relation", "ObjectEntities"}


def test_median_config_reproduces_the_previous_baseline(tmp_path: Path) -> None:
    config = _config(tmp_path, "median")
    candidates = _write(tmp_path / "candidates.jsonl", CANDIDATE_ROWS)
    output = tmp_path / "predictions.jsonl"

    reaggregate.main(
        ["--config", str(config), "--candidates", str(candidates),
         "--output", str(output)]
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["ObjectEntities"] == ["37.65"]


def test_official_input_order_is_verified(tmp_path: Path) -> None:
    config = _config(tmp_path)
    candidates = _write(tmp_path / "candidates.jsonl", CANDIDATE_ROWS)
    official = _write(
        tmp_path / "input.jsonl",
        [{"SubjectEntity": "France", "Relation": "countryLandBordersCountry"},
         {"SubjectEntity": "Rùm", "Relation": "hasArea"}],
    )

    with pytest.raises(ValueError, match="does not match the official input"):
        reaggregate.main(
            ["--config", str(config), "--candidates", str(candidates),
             "--output", str(tmp_path / "out.jsonl"), "--input", str(official)]
        )


def test_row_count_mismatch_is_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    candidates = _write(tmp_path / "candidates.jsonl", CANDIDATE_ROWS)
    official = _write(
        tmp_path / "input.jsonl",
        [{"SubjectEntity": "Rùm", "Relation": "hasArea"}],
    )

    with pytest.raises(ValueError, match="expected 1 rows"):
        reaggregate.main(
            ["--config", str(config), "--candidates", str(candidates),
             "--output", str(tmp_path / "out.jsonl"), "--input", str(official)]
        )


def test_model_assisted_strategy_requires_stored_selection(tmp_path: Path) -> None:
    config = _config(tmp_path, "cluster_choice")
    candidates = _write(tmp_path / "candidates.jsonl", CANDIDATE_ROWS[:1])
    output = tmp_path / "out.jsonl"

    with pytest.raises(ValueError, match="needs model assistance"):
        reaggregate.main(
            ["--config", str(config), "--candidates", str(candidates),
             "--output", str(output)]
        )

    stored = [dict(CANDIDATE_ROWS[0], FinalSelection={"ObjectEntities": ["104"]})]
    _write(tmp_path / "candidates.jsonl", stored)
    reaggregate.main(
        ["--config", str(config), "--candidates", str(candidates),
         "--output", str(output), "--reuse-final-selection"]
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["ObjectEntities"] == ["104"]


def test_missing_candidates_is_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    candidates = _write(
        tmp_path / "candidates.jsonl",
        [{"SubjectEntity": "Rùm", "Relation": "hasArea"}],
    )

    with pytest.raises(ValueError, match="missing Candidates"):
        reaggregate.main(
            ["--config", str(config), "--candidates", str(candidates),
             "--output", str(tmp_path / "out.jsonl")]
        )
