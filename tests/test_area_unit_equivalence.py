import pytest

from akbc_baseline.aggregation import aggregate_candidates
from akbc_baseline.area_clustering import cluster_numeric_candidates
from akbc_baseline.area_unit_equivalence import (
    UNIT_EQUIVALENCE_FACTORS,
    find_cluster_unit_equivalences,
    is_power_of_ten,
    merge_cluster_support,
    select_unit_equivalent_value,
    validate_factors,
)


def _candidates(values: list[str]) -> list[list[str]]:
    return [[value] for value in values]


def test_power_of_ten_factor_is_rejected_as_a_digit_error() -> None:
    assert is_power_of_ten(0.01)
    assert is_power_of_ten(100.0)
    assert not is_power_of_ten(2.589988110336)
    assert not is_power_of_ten(0.0040468564224)
    with pytest.raises(ValueError):
        validate_factors({"hectare": 0.01})


def test_hectare_is_not_a_configured_unit_factor() -> None:
    assert set(UNIT_EQUIVALENCE_FACTORS) == {"square_mile", "acre"}
    validate_factors(UNIT_EQUIVALENCE_FACTORS)


def test_decade_related_clusters_are_not_merged() -> None:
    # 0.63 and 0.063 differ by a factor of ten. That is a digit error and must
    # not transfer support, otherwise the wrong decade wins.
    candidates = _candidates(["0.63"] * 13 + ["0.063"] * 3 + ["0.0063"] * 2)
    clusters = cluster_numeric_candidates(candidates, 0.05)

    edges = find_cluster_unit_equivalences(clusters)

    assert edges == []
    assert merge_cluster_support(clusters, edges) == [
        cluster.support for cluster in clusters
    ]


def test_support_moves_to_the_square_kilometre_cluster() -> None:
    candidates = _candidates(["37.65"] * 4 + ["101.6", "102.12", "104"])

    value, diagnostics = select_unit_equivalent_value(candidates)

    assert value == "102.12"
    assert diagnostics["changed_dominant"] is True
    assert diagnostics["converted_values_added"] is False
    assert [edge["assumed_unit"] for edge in diagnostics["equivalences"]] == [
        "square_mile"
    ]
    assert sum(diagnostics["merged_support"]) == sum(
        diagnostics["original_support"]
    )


def test_no_candidate_is_removed_and_no_converted_value_is_selectable() -> None:
    candidates = _candidates(["37.65"] * 4 + ["101.6", "102.12", "104"])

    value, diagnostics = select_unit_equivalent_value(candidates)

    assert diagnostics["clusters"] == len(diagnostics["cluster_representatives"])
    assert value in diagnostics["cluster_representatives"]
    converted = diagnostics["equivalences"][0]["converted_value_km2"]
    assert converted not in diagnostics["cluster_representatives"]


def test_mutually_pointing_clusters_are_dropped() -> None:
    # Reciprocal factors make both directions valid, so neither may be used.
    clusters = cluster_numeric_candidates(_candidates(["1", "2"]), 0.001)

    edges = find_cluster_unit_equivalences(
        clusters, 0.05, {"double": 2.0, "half": 0.5}
    )

    assert edges == []


def test_without_an_equivalence_the_dominant_cluster_is_kept() -> None:
    candidates = _candidates(["19.9"] * 5 + ["30"])

    value, diagnostics = select_unit_equivalent_value(candidates)

    assert value == "19.9"
    assert diagnostics["equivalences"] == []
    assert diagnostics["changed_dominant"] is False


def test_a_minority_square_mile_reading_can_win_by_transfer() -> None:
    # 19.9 square miles is 51.5, matching the lone 51.5 candidate. The five
    # supporting candidates move onto it rather than being deleted.
    candidates = _candidates(["19.9"] * 5 + ["51.5"])

    value, diagnostics = select_unit_equivalent_value(candidates)

    assert value == "51.5"
    assert diagnostics["merged_support"] == [0, 6]


def test_empty_and_unparsable_candidates_are_safe() -> None:
    assert select_unit_equivalent_value([])[0] is None
    assert select_unit_equivalent_value([["not a number"]])[0] is None


def test_aggregate_candidates_exposes_the_strategy() -> None:
    candidates = _candidates(["37.65"] * 4 + ["101.6", "102.12", "104"])

    assert aggregate_candidates(candidates, {"strategy": "unit_equivalence"}) == [
        "102.12"
    ]


def test_aggregate_candidates_honours_tolerances() -> None:
    candidates = _candidates(["37.65"] * 4 + ["101.6", "102.12", "104"])

    # 37.65 square miles is 97.5, which is 4.7 percent away from 102.12. A one
    # percent unit tolerance must not create the edge.
    assert aggregate_candidates(
        candidates, {"strategy": "unit_equivalence", "unit_tolerance": 0.01}
    ) == ["37.65"]


def test_invalid_tolerance_is_rejected() -> None:
    with pytest.raises(ValueError):
        select_unit_equivalent_value([["1"]], cluster_tolerance=0)
    with pytest.raises(ValueError):
        find_cluster_unit_equivalences(
            cluster_numeric_candidates([["1"]], 0.05), 1.5
        )
