import json
from pathlib import Path

from akbc_baseline import area_unit_filter_rerank
from akbc_baseline.area_unit_filter import (
    filter_unit_collision_sources,
    find_unit_collisions,
)


def test_filters_conversion_source_without_adding_converted_value() -> None:
    candidates = [["38.4"], ["101.6"], ["102.12"], ["104"]]

    filtered, collisions, removed = filter_unit_collision_sources(candidates)

    assert removed == [0]
    assert filtered == [["101.6"], ["102.12"], ["104"]]
    assert {item.assumed_unit for item in collisions} == {"square_mile"}
    assert all(item.converted_value_km2 < 100 for item in collisions)
    assert ["99.455543437"] not in filtered


def test_detects_hectare_collision_against_original_candidate() -> None:
    collisions = find_unit_collisions([["6.3"], ["0.063"], ["0.064"]], 0.02)

    assert len(collisions) == 1
    assert collisions[0].source_index == 0
    assert collisions[0].target_index == 1
    assert collisions[0].assumed_unit == "hectare"


def test_no_observed_target_keeps_source_and_does_not_expand() -> None:
    candidates = [["19.9"], ["54.052"], ["55"]]

    filtered, collisions, removed = filter_unit_collision_sources(
        candidates, 0.02
    )

    assert filtered == candidates
    assert collisions == []
    assert removed == []


def test_cli_aggregates_only_retained_raw_candidates(tmp_path: Path) -> None:
    row = {
        "SubjectEntity": "Rùm",
        "Relation": "hasArea",
        "Candidates": [["38.4"], ["101.6"], ["102.12"], ["104"]],
        "FinalSelection": {"ObjectEntities": ["70"]},
    }
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    output = tmp_path / "prediction.jsonl"
    audit = tmp_path / "audit.jsonl"
    metrics = tmp_path / "metrics.json"

    assert area_unit_filter_rerank.main(
        [
            "--candidates",
            str(source),
            "--output",
            str(output),
            "--candidates-output",
            str(audit),
            "--metrics-output",
            str(metrics),
        ]
    ) == 0

    prediction = json.loads(output.read_text(encoding="utf-8"))
    assert prediction["ObjectEntities"] == ["102.12"]
    saved = json.loads(audit.read_text(encoding="utf-8"))
    assert saved["SourceFinalSelection"] == {"ObjectEntities": ["70"]}
    assert saved["FinalSelection"]["removed_candidate_indices"] == [0]
    assert saved["FinalSelection"]["converted_values_added"] is False
    report = json.loads(metrics.read_text(encoding="utf-8"))
    assert report["rows_with_collisions"] == 1
    assert report["total_removed_candidates"] == 1
