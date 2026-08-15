import pytest

from akbc_baseline.aggregation import (
    aggregate_candidates,
    aggregate_dominant_cluster,
)
from akbc_baseline.config import ModelConfig


def test_dominant_cluster_prefers_the_largest_cluster_over_the_median() -> None:
    candidates = (
        [["15000"]] * 6 + [["30000"]] * 7 + [["20000"]] * 3 + [["12000"]]
    )
    assert aggregate_dominant_cluster(candidates) == ["30000"]


def test_dominant_cluster_matches_median_extraction_and_formatting() -> None:
    assert aggregate_dominant_cluster([["1,000 people"], ["1000"]]) == ["1000"]
    assert aggregate_dominant_cluster([["2.5"], ["2.5"], ["9"]]) == ["2.5"]
    assert aggregate_dominant_cluster([["no number"], []]) == []


def test_dominant_cluster_uses_relative_tolerance() -> None:
    # 100 and 104 fall inside one 5% window; 200 stays alone.
    candidates = [["100"], ["104"], ["200"]]
    assert aggregate_dominant_cluster(candidates) == ["102"]


def test_dominant_cluster_rejects_a_non_positive_tolerance() -> None:
    with pytest.raises(ValueError):
        aggregate_dominant_cluster([["1"]], cluster_tolerance=0)


def test_dispatcher_routes_the_dominant_cluster_policy() -> None:
    candidates = [["10"], ["10"], ["30"]]
    policy = {"strategy": "dominant_cluster", "cluster_tolerance": 0.05}
    assert aggregate_candidates(candidates, policy) == ["10"]


def test_tuned_20_candidate_config_adopts_dominant_cluster() -> None:
    config = ModelConfig.from_yaml(
        "configs/experiment-qwen-3.5-27b-thinking-empty-aware-20-tuned.yaml"
    )
    assert config.num_candidates == 20
    assert config.aggregation["hasCapacity"]["strategy"] == "dominant_cluster"
    assert config.aggregation["awardWonBy"]["threshold"] == 0.05
    company = config.aggregation["companyTradesAtStockExchange"]
    assert company["threshold"] == 0.6
    assert company["empty_majority"] == 10
