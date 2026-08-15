from akbc_baseline.aggregation import aggregate_candidates, aggregate_route_consensus


def _wrap(values):
    return [[str(value)] for value in values]


def test_cross_route_support_beats_single_route_repetition():
    # Route blocks of 5: route 0 repeats 378000 five times; routes 1-3 each
    # contribute one sample near 550000. Three routes beat one route even
    # though the single-route cluster has more samples.
    values = (
        ["378000"] * 5
        + ["550000", "9", "13", "17", "21"]
        + ["551000", "33", "41", "47", "53"]
        + ["549000", "71", "79", "83", "89"]
    )
    result = aggregate_route_consensus(_wrap(values))
    assert result == ["550000"]


def test_sample_count_breaks_route_ties():
    # Both clusters span two routes; the 300000 cluster has more samples.
    values = (
        ["100000", "100500", "300000", "300500", "301000"]
        + ["100200", "299500", "300200", "1", "2"]
        + ["3", "4", "5", "6", "7"]
        + ["8", "9", "11", "12", "13"]
    )
    result = aggregate_route_consensus(_wrap(values))
    assert result == ["300200"]


def test_median_center_assignment_resists_chaining():
    # 100 -> 110 -> 121 -> 133 are pairwise-adjacent in log space, but
    # assignment against the cluster median stops the chain from absorbing
    # everything into one wide cluster.
    values = ["100", "110", "121", "133"] + ["1"] * 16
    result = aggregate_route_consensus(_wrap(values), samples_per_route=5)
    assert result == ["1"]


def test_non_numeric_and_nonpositive_candidates_are_ignored():
    values = ["not a number", "-5", "0", "550000", "551000"]
    result = aggregate_route_consensus(_wrap(values), samples_per_route=1)
    assert result == ["550500"]


def test_empty_when_no_numeric_candidate():
    assert aggregate_route_consensus([["x"], []]) == []


def test_dispatch_through_aggregate_candidates():
    values = ["550000"] * 20
    result = aggregate_candidates(
        _wrap(values),
        {"strategy": "route_consensus", "log_threshold": 0.05, "samples_per_route": 5},
    )
    assert result == ["550000"]


def test_candidate_temperatures_validation():
    import pytest
    from akbc_baseline.config import ModelConfig

    base = dict(
        name="t", model_id="m", backend="llama_cpp_server",
        prompt_templates_file="p.csv", train_data_file="t.jsonl",
        num_candidates=4,
    )
    config = ModelConfig(**base, candidate_temperatures=[0.2, 0.5, 0.8, 1.1])
    assert config.candidate_temperatures[0] == 0.2
    import yaml, tempfile, os
    for bad in ([0.2, 0.5], [0.2, 0.5, 0.8, 0.0], [0.2, 0.5, 0.8, "x"]):
        raw = dict(base, candidate_temperatures=bad)
        path = tempfile.mktemp(suffix=".yaml")
        with open(path, "w", encoding="utf-8") as h:
            yaml.safe_dump(raw, h)
        with pytest.raises(ValueError):
            ModelConfig.from_yaml(path)
        os.unlink(path)


def test_majority_minimum_votes_abstains():
    from akbc_baseline.aggregation import aggregate_majority

    candidates = [["Paris"]] * 9 + [[]] * 2
    assert aggregate_majority(candidates, minimum_votes=10) == []
    assert aggregate_majority(candidates, minimum_votes=9) == ["Paris"]
