from pathlib import Path

import pytest

from akbc_baseline import quality_gate
from akbc_baseline.data import write_jsonl


def _prediction(index: int, entities: list[str]) -> dict[str, object]:
    return {
        "SubjectEntity": f"Subject {index}",
        "Relation": "personHasCityOfDeath",
        "ObjectEntities": entities,
    }


def _candidate_row(
    index: int,
    candidates: list[list[str]],
    diagnostics: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "SubjectEntity": f"Subject {index}",
        "Relation": "personHasCityOfDeath",
        "Candidates": candidates,
        "CandidateDiagnostics": diagnostics
        or [{"hit_token_limit": False} for _ in candidates],
    }


def test_smoke_gate_reports_candidate_quality() -> None:
    report = quality_gate.assess_smoke(
        [_prediction(0, ["Paris"]), _prediction(1, [])],
        [
            _candidate_row(0, [["Paris"], ["Paris"]]),
            _candidate_row(1, [[], ["London"]]),
        ],
        expected_rows=2,
        expected_candidates=2,
    )

    assert report["empty_prediction_rate"] == 0.5
    assert report["empty_candidate_rate"] == 0.25
    assert report["diagnostic_candidates"] == 4


def test_smoke_gate_reports_two_stage_generation_quality() -> None:
    report = quality_gate.assess_smoke(
        [_prediction(0, ["Paris"])],
        [
            _candidate_row(
                0,
                [["Paris"], ["Lyon"]],
                [
                    {
                        "hit_token_limit": False,
                        "hit_thinking_budget": True,
                        "natural_think_end": False,
                        "forced_think_end": True,
                    },
                    {
                        "hit_token_limit": False,
                        "hit_thinking_budget": False,
                        "natural_think_end": True,
                        "forced_think_end": False,
                    },
                ],
            )
        ],
        expected_rows=1,
        expected_candidates=2,
    )

    assert report["token_limit_rate"] == 0
    assert report["thinking_budget_rate"] == 0.5
    assert report["natural_think_end_rate"] == 0.5
    assert report["forced_think_end_rate"] == 0.5


def test_smoke_gate_fails_when_empty_candidates_exceed_limit(
    tmp_path: Path,
) -> None:
    predictions = tmp_path / "predictions.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    write_jsonl(predictions, [_prediction(0, [])])
    write_jsonl(candidates, [_candidate_row(0, [[], []])])

    with pytest.raises(RuntimeError, match="empty candidate rate"):
        quality_gate.main(
            [
                "--predictions",
                str(predictions),
                "--candidates",
                str(candidates),
                "--report",
                str(tmp_path / "report.json"),
                "--expected-rows",
                "1",
                "--expected-candidates",
                "2",
                "--maximum-empty-prediction-rate",
                "1",
                "--maximum-empty-candidate-rate",
                "0.25",
            ]
        )


def test_smoke_gate_requires_matching_final_selection() -> None:
    prediction = _prediction(0, ["255"])
    candidate = _candidate_row(0, [["99"], ["255"]])

    with pytest.raises(ValueError, match="missing FinalSelection"):
        quality_gate.assess_smoke(
            [prediction],
            [candidate],
            expected_rows=1,
            expected_candidates=2,
            require_final_selection=True,
        )

    candidate["FinalSelection"] = {
        "ObjectEntities": ["255"],
        "used_fallback": False,
    }
    report = quality_gate.assess_smoke(
        [prediction],
        [candidate],
        expected_rows=1,
        expected_candidates=2,
        require_final_selection=True,
    )
    assert report["final_selection_rows"] == 1
    assert report["final_selection_fallbacks"] == 0
