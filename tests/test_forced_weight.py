import pytest

from akbc_baseline.aggregation import (
    aggregate_candidates,
    aggregate_dominant_cluster,
    aggregate_majority,
)
from akbc_baseline.config import ModelConfig


def diags(candidates, forced):
    """parse_status mirrors the runtime: explicit_empty for [] answers."""
    return [
        {
            "parse_status": "explicit_empty" if not cand else "json",
            "forced_think_end": bool(f),
        }
        for cand, f in zip(candidates, forced)
    ]


def test_majority_default_weight_keeps_legacy_behavior() -> None:
    candidates = [["Moscow"], [], [], ["Moscow"], []]
    d = diags(candidates, [0, 1, 1, 0, 1])
    unweighted = aggregate_majority(
        candidates, diagnostics=d, explicit_empty_only=True
    )
    assert unweighted == []  # 3 empty votes beat 2 Moscow votes
    assert aggregate_majority(
        candidates, diagnostics=d, explicit_empty_only=True, forced_weight=1.0
    ) == unweighted


def test_majority_down_weights_forced_empty_votes() -> None:
    # Two natural candidates name the city; three forced candidates gave up.
    candidates = [["Moscow"], [], [], ["Moscow"], []]
    d = diags(candidates, [0, 1, 1, 0, 1])
    assert aggregate_majority(
        candidates, diagnostics=d, explicit_empty_only=True, forced_weight=0.4
    ) == ["Moscow"]


def test_majority_weight_applies_to_forced_value_votes_too() -> None:
    # Three forced candidates guess a city; two natural candidates are empty.
    candidates = [["Berlin"], ["Berlin"], ["Berlin"], [], []]
    d = diags(candidates, [1, 1, 1, 0, 0])
    assert aggregate_majority(
        candidates, diagnostics=d, explicit_empty_only=True, forced_weight=0.4
    ) == []


def test_majority_rejects_out_of_range_weight() -> None:
    with pytest.raises(ValueError):
        aggregate_majority([["x"]], forced_weight=1.5)


def test_dominant_cluster_weighted_mass_moves_the_cluster() -> None:
    # Forced majority cluster at 15000 loses to two natural votes at 30000.
    candidates = [["15000"]] * 3 + [["30000"]] * 2
    d = diags(candidates, [1, 1, 1, 0, 0])
    assert aggregate_dominant_cluster(
        candidates, diagnostics=d, forced_weight=0.4
    ) == ["30000"]
    assert aggregate_dominant_cluster(
        candidates, diagnostics=d, forced_weight=1.0
    ) == ["15000"]


def test_dispatcher_passes_forced_weight() -> None:
    candidates = [["15000"]] * 3 + [["30000"]] * 2
    d = diags(candidates, [1, 1, 1, 0, 0])
    policy = {
        "strategy": "dominant_cluster",
        "cluster_tolerance": 0.05,
        "forced_weight": 0.4,
    }
    assert aggregate_candidates(candidates, policy, d) == ["30000"]


def test_tuned_config_adopts_forced_weight_for_two_relations() -> None:
    config = ModelConfig.from_yaml(
        "configs/experiment-qwen-3.5-27b-thinking-empty-aware-20-tuned.yaml"
    )
    assert config.aggregation["personHasCityOfDeath"]["forced_weight"] == 0.4
    assert config.aggregation["hasCapacity"]["forced_weight"] == 0.4
    assert "forced_weight" not in config.aggregation["companyTradesAtStockExchange"]
    assert "forced_weight" not in config.aggregation["hasArea"]
