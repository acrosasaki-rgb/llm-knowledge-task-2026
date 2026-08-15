import pytest

from akbc_baseline.aggregation import (
    aggregate_audited_median,
    aggregate_candidates,
    aggregate_empty_aware_frequency,
    aggregate_frequency,
    aggregate_grouped_frequency,
    aggregate_majority,
    aggregate_median,
    aggregate_union,
    normalize_vote,
)


def test_normalizes_surface_forms_for_voting() -> None:
    assert normalize_vote("São-Paulo") == normalize_vote("sao paulo")
    assert normalize_vote("Kauaʻi") == normalize_vote("Kauai")
    assert normalize_vote("O'Brien") == normalize_vote("Obrien")
    assert normalize_vote("Canal+") == normalize_vote("Canal")


def test_frequency_uses_one_vote_per_candidate_and_keeps_representative() -> None:
    candidates = [
        ["NASDAQ", "Nasdaq"],
        ["Nasdaq"],
        ["New York Stock Exchange"],
        [],
    ]
    assert aggregate_frequency(candidates, threshold=0.5) == ["Nasdaq"]


def test_majority_treats_empty_as_a_vote_and_breaks_ties_conservatively() -> None:
    assert aggregate_majority([["Paris"], [], ["paris"]]) == ["Paris"]
    assert aggregate_majority([["Paris"], []]) == []


def test_empty_aware_frequency_uses_explicit_empty_majority() -> None:
    diagnostics = [
        {"parse_status": "explicit_empty"},
        {"parse_status": "explicit_empty"},
        {"parse_status": "explicit_empty"},
        {"parse_status": "json"},
        {"parse_status": "json"},
    ]
    assert aggregate_empty_aware_frequency(
        [[], [], [], ["A"], ["B"]], 0.4, diagnostics=diagnostics
    ) == []


def test_empty_aware_frequency_excludes_empty_from_frequency_denominator() -> None:
    diagnostics = [
        {"parse_status": "explicit_empty"},
        {"parse_status": "explicit_empty"},
        {"parse_status": "json"},
        {"parse_status": "json"},
        {"parse_status": "json"},
    ]
    assert aggregate_empty_aware_frequency(
        [[], [], ["A"], ["A"], ["A"]], 0.7, diagnostics=diagnostics
    ) == ["A"]


def test_empty_aware_frequency_does_not_count_parse_failure_as_empty() -> None:
    diagnostics = [
        {"parse_status": "explicit_empty"},
        {"parse_status": "explicit_empty"},
        {"parse_status": "parse_failure"},
        {"parse_status": "json"},
        {"parse_status": "json"},
    ]
    assert aggregate_empty_aware_frequency(
        [[], [], [], ["A"], ["A"]], 1.0, diagnostics=diagnostics
    ) == ["A"]


def test_relation_specific_aggregation_policies() -> None:
    assert aggregate_candidates(
        [[], [], [], ["Winner"], ["Other"]],
        {"strategy": "frequency", "threshold": 0.2},
        diagnostics=[
            {"parse_status": "explicit_empty"},
            {"parse_status": "explicit_empty"},
            {"parse_status": "explicit_empty"},
            {"parse_status": "json"},
            {"parse_status": "json"},
        ],
    ) == ["Winner", "Other"]
    assert aggregate_candidates(
        [[], [], [], ["Paris"], ["Paris"]],
        {"strategy": "majority", "explicit_empty_only": True},
        diagnostics=[
            {"parse_status": "explicit_empty"},
            {"parse_status": "explicit_empty"},
            {"parse_status": "explicit_empty"},
            {"parse_status": "json"},
            {"parse_status": "json"},
        ],
    ) == []


def test_numeric_aggregation_uses_median() -> None:
    assert aggregate_median([["10,000"], ["12000"], ["11000"], []]) == ["11000"]
    assert aggregate_median([["1.5"], ["2.5"]]) == ["2"]


def test_union_deduplicates_era_candidates_and_skips_parse_failures() -> None:
    candidates = [
        ["Alice", "Bob"],
        ["alice", "Carol"],
        ["Should not survive"],
    ]
    diagnostics = [
        {"parse_status": "json"},
        {"parse_status": "json"},
        {"parse_status": "parse_failure"},
    ]

    assert aggregate_union(candidates, diagnostics=diagnostics) == [
        "Alice",
        "Bob",
        "Carol",
    ]
    assert aggregate_candidates(
        candidates, {"strategy": "union"}, diagnostics=diagnostics
    ) == ["Alice", "Bob", "Carol"]


def test_grouped_frequency_pairs_same_turn_across_conversation_chains() -> None:
    candidates = [
        ["Recent A", "Shared recent"],
        ["Old A", "Shared old"],
        ["Recent B", "Shared recent"],
        ["Old B", "Shared old"],
    ]

    assert aggregate_grouped_frequency(
        candidates, groups=2, threshold=0.5
    ) == [
        "Shared recent",
        "Recent A",
        "Recent B",
        "Shared old",
        "Old A",
        "Old B",
    ]
    assert aggregate_grouped_frequency(
        candidates, groups=2, threshold=1.0
    ) == ["Shared recent", "Shared old"]


def test_audited_median_uses_normalized_usable_values_and_falls_back() -> None:
    candidates = [["99"], ["255"], ["1000"]]
    diagnostics = [
        {"audit": {"usable": True, "normalized_value_km2": 256.41}},
        {"audit": {"usable": True, "normalized_value_km2": 255.0}},
        {"audit": {"usable": False, "normalized_value_km2": 1000.0}},
    ]

    assert aggregate_audited_median(
        candidates, diagnostics=diagnostics
    ) == ["255.705"]
    assert aggregate_candidates(
        candidates,
        {"strategy": "audited_median"},
        diagnostics=diagnostics,
    ) == ["255.705"]
    assert aggregate_audited_median(candidates, diagnostics=[]) == ["255"]


def test_dispatches_and_validates_strategy() -> None:
    assert aggregate_candidates([["A"], ["a"]], {"strategy": "frequency"}) == [
        "A"
    ]
    with pytest.raises(ValueError, match="unsupported"):
        aggregate_candidates([["A"]], {"strategy": "unknown"})
    with pytest.raises(ValueError, match="model-assisted"):
        aggregate_candidates([["1"]], {"strategy": "metadata_judge"})
