from __future__ import annotations

import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from typing import Any


SUPPORTED_STRATEGIES = {
    "audited_median",
    "cluster_choice",
    "dominant_cluster",
    "frequency",
    "grouped_frequency",
    "majority",
    "median",
    "metadata_judge",
    "route_consensus",
    "union",
    "unit_equivalence",
}
APOSTROPHE_LIKE = set("'’‘ʻʼʹ`´")
ASCII_SYMBOLS = set("+$<=>|~^")


def normalize_vote(value: str) -> str:
    value = "".join(
        character for character in value.strip() if character not in APOSTROPHE_LIKE
    )
    decomposed = unicodedata.normalize("NFKD", value).casefold()
    normalized = []
    for character in decomposed:
        if character in APOSTROPHE_LIKE or unicodedata.combining(character):
            continue
        if (
            character in ASCII_SYMBOLS
            or unicodedata.category(character).startswith("P")
        ):
            normalized.append(" ")
        else:
            normalized.append(character)
    return " ".join("".join(normalized).split())


def _parse_status(
    diagnostics: list[dict[str, Any]] | None, index: int
) -> str | None:
    if diagnostics is None or index >= len(diagnostics):
        return None
    item = diagnostics[index]
    if not isinstance(item, dict):
        return None
    status = item.get("parse_status")
    return status if isinstance(status, str) else None


def _representatives(
    candidates: list[list[str]],
) -> tuple[Counter[str], dict[str, Counter[str]], dict[str, int]]:
    votes: Counter[str] = Counter()
    surfaces: dict[str, Counter[str]] = defaultdict(Counter)
    first_seen: dict[str, int] = {}
    position = 0
    for candidate in candidates:
        seen_in_candidate: set[str] = set()
        for value in candidate:
            if not isinstance(value, str) or not value.strip():
                continue
            surface = value.strip()
            key = normalize_vote(surface)
            if not key:
                continue
            surfaces[key][surface] += 1
            if key in seen_in_candidate:
                continue
            seen_in_candidate.add(key)
            votes[key] += 1
            first_seen.setdefault(key, position)
            position += 1
    return votes, surfaces, first_seen


def _surface_for(key: str, surfaces: dict[str, Counter[str]]) -> str:
    return sorted(
        surfaces[key].items(),
        key=lambda item: (-item[1], len(item[0]), item[0].casefold()),
    )[0][0]


def aggregate_frequency(
    candidates: list[list[str]], threshold: float
) -> list[str]:
    if not 0 < threshold <= 1:
        raise ValueError("frequency threshold must be in (0, 1]")
    if not candidates:
        return []
    votes, surfaces, first_seen = _representatives(candidates)
    minimum_votes = max(1, math.ceil(len(candidates) * threshold))
    accepted = [key for key, count in votes.items() if count >= minimum_votes]
    accepted.sort(key=lambda key: (-votes[key], first_seen[key], key))
    return [_surface_for(key, surfaces) for key in accepted]


def aggregate_union(
    candidates: list[list[str]],
    *,
    diagnostics: list[dict[str, Any]] | None = None,
) -> list[str]:
    usable = [
        candidate
        for index, candidate in enumerate(candidates)
        if _parse_status(diagnostics, index) != "parse_failure"
    ]
    _, surfaces, first_seen = _representatives(usable)
    accepted = sorted(first_seen, key=lambda key: (first_seen[key], key))
    return [_surface_for(key, surfaces) for key in accepted]


def aggregate_grouped_frequency(
    candidates: list[list[str]],
    *,
    groups: int,
    threshold: float,
    diagnostics: list[dict[str, Any]] | None = None,
) -> list[str]:
    if groups < 1 or len(candidates) % groups:
        raise ValueError("groups must evenly divide candidates")
    chains = len(candidates) // groups
    accepted_by_group: list[list[str]] = []
    for group_index in range(groups):
        indices = [
            chain_index * groups + group_index
            for chain_index in range(chains)
        ]
        group_candidates = [
            [] if _parse_status(diagnostics, index) == "parse_failure"
            else candidates[index]
            for index in indices
        ]
        accepted_by_group.append(
            aggregate_frequency(group_candidates, threshold)
        )
    return aggregate_union(accepted_by_group)


def aggregate_empty_aware_frequency(
    candidates: list[list[str]],
    threshold: float,
    *,
    diagnostics: list[dict[str, Any]] | None = None,
    empty_majority: int = 3,
) -> list[str]:
    if not 0 < threshold <= 1:
        raise ValueError("frequency threshold must be in (0, 1]")
    explicit_empty_votes = 0
    usable_candidates: list[list[str]] = []
    for index, candidate in enumerate(candidates):
        status = _parse_status(diagnostics, index)
        if not candidate:
            if status == "explicit_empty":
                explicit_empty_votes += 1
            elif status is None:
                explicit_empty_votes += 1
            continue
        if status == "parse_failure":
            continue
        usable_candidates.append(candidate)
    if explicit_empty_votes >= empty_majority:
        return []
    if not usable_candidates:
        return []
    return aggregate_frequency(usable_candidates, threshold)


def _forced_candidate_weight(
    diagnostics: list[dict[str, Any]] | None, index: int, forced_weight: float
) -> float:
    """Weight for one candidate: forced_weight when its thinking hit the
    budget (forced_think_end), 1.0 otherwise. Issue #25: budget-exhausted
    candidates skew toward given-up empty answers, so down-weighting them
    recalibrates empty-versus-value votes without changing the default path
    (forced_weight 1.0 keeps integer votes and legacy behavior)."""
    if forced_weight == 1.0 or diagnostics is None or index >= len(diagnostics):
        return 1.0
    item = diagnostics[index]
    if isinstance(item, dict) and item.get("forced_think_end"):
        return forced_weight
    return 1.0


def aggregate_majority(
    candidates: list[list[str]],
    *,
    diagnostics: list[dict[str, Any]] | None = None,
    explicit_empty_only: bool = False,
    forced_weight: float = 1.0,
    minimum_votes: float = 0.0,
) -> list[str]:
    if not 0 <= forced_weight <= 1:
        raise ValueError("forced_weight must be within [0, 1]")
    if minimum_votes < 0:
        raise ValueError("minimum_votes must be non-negative")
    if not candidates:
        return []
    votes: dict[str | None, float] = defaultdict(float)
    surfaces: dict[str, Counter[str]] = defaultdict(Counter)
    first_seen: dict[str, int] = {}
    for candidate_index, candidate in enumerate(candidates):
        status = _parse_status(diagnostics, candidate_index)
        if explicit_empty_only and status == "parse_failure":
            continue
        weight = _forced_candidate_weight(
            diagnostics, candidate_index, forced_weight
        )
        first_value = next(
            (
                value.strip()
                for value in candidate
                if isinstance(value, str) and value.strip()
            ),
            None,
        )
        if first_value is None:
            if not explicit_empty_only or status in {None, "explicit_empty"}:
                votes[None] += weight
            continue
        key = normalize_vote(first_value)
        if not key:
            if not explicit_empty_only or status in {None, "explicit_empty"}:
                votes[None] += weight
            continue
        votes[key] += weight
        surfaces[key][first_value] += 1
        first_seen.setdefault(key, candidate_index)

    empty_votes = votes.get(None, 0.0)
    nonempty = [key for key in votes if key is not None]
    if not nonempty:
        return []
    winner = sorted(
        nonempty,
        key=lambda key: (-votes[key], first_seen[key], key),
    )[0]
    if votes[winner] <= empty_votes:
        return []
    # Abstention gate: without a sufficiently dominant winner, an empty
    # answer scores better than a low-confidence guess on the empty-heavy
    # relations (personHasCityOfDeath sweep, #38).
    if votes[winner] < minimum_votes:
        return []
    return [_surface_for(winner, surfaces)]


def _parse_number(value: str) -> float | None:
    match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", value)
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def aggregate_median(candidates: list[list[str]]) -> list[str]:
    values: list[float] = []
    for candidate in candidates:
        parsed = next(
            (
                number
                for value in candidate
                if isinstance(value, str)
                for number in [_parse_number(value)]
                if number is not None
            ),
            None,
        )
        if parsed is not None:
            values.append(parsed)
    if not values:
        return []
    median = statistics.median(values)
    if median.is_integer():
        return [str(int(median))]
    return [format(median, ".15g")]


def aggregate_dominant_cluster(
    candidates: list[list[str]],
    *,
    cluster_tolerance: float = 0.05,
    diagnostics: list[dict[str, Any]] | None = None,
    forced_weight: float = 1.0,
) -> list[str]:
    """Median of the largest relative-tolerance cluster of candidate values.

    Unlike unit_equivalence this hypothesizes no unit conversions, so it is
    suitable for numeric relations whose ambiguity has no fixed conversion
    factor (Issue #24: hasCapacity). Value extraction and number formatting
    match aggregate_median. With few candidates the largest cluster is
    unstable; on the 5-candidate validation pool it scored below the plain
    median, so only many-candidate configurations should adopt it.
    """
    if cluster_tolerance <= 0:
        raise ValueError("cluster_tolerance must be positive")
    if not 0 <= forced_weight <= 1:
        raise ValueError("forced_weight must be within [0, 1]")
    pairs: list[tuple[float, float]] = []
    for candidate_index, candidate in enumerate(candidates):
        parsed = next(
            (
                number
                for value in candidate
                if isinstance(value, str)
                for number in [_parse_number(value)]
                if number is not None
            ),
            None,
        )
        if parsed is not None:
            pairs.append(
                (
                    parsed,
                    _forced_candidate_weight(
                        diagnostics, candidate_index, forced_weight
                    ),
                )
            )
    if not pairs:
        return []
    ordered = sorted(pairs)
    best_mass = -1.0
    best: list[float] = []
    for anchor, _ in ordered:
        cluster = [
            (value, weight)
            for value, weight in ordered
            if anchor <= value <= anchor * (1 + cluster_tolerance)
        ]
        mass = sum(weight for _, weight in cluster)
        if mass > best_mass:
            best_mass = mass
            best = [value for value, _ in cluster]
    selected = statistics.median(best)
    if selected.is_integer():
        return [str(int(selected))]
    return [format(selected, ".15g")]


def aggregate_route_consensus(
    candidates: list[list[str]],
    *,
    log_threshold: float = 0.05,
    samples_per_route: int = 5,
) -> list[str]:
    """Cross-route consensus over numeric candidates in log10 space.

    Candidate index i is generated by route i // samples_per_route (routes are
    contiguous blocks of candidate_instructions). Values are clustered by
    distance to the cluster's log-median (no single-linkage chaining), then
    the cluster is chosen lexicographically: most distinct supporting routes,
    most samples, smallest log-space MAD. Agreement across independently
    prompted routes is treated as stronger evidence than repetition within
    one route. The cluster's value median is returned.
    """
    if log_threshold <= 0:
        raise ValueError("log_threshold must be positive")
    if samples_per_route < 1:
        raise ValueError("samples_per_route must be positive")
    entries: list[tuple[float, float, int]] = []
    for index, candidate in enumerate(candidates):
        parsed = next(
            (
                number
                for value in candidate
                if isinstance(value, str)
                for number in [_parse_number(value)]
                if number is not None
            ),
            None,
        )
        if parsed is not None and parsed > 0 and math.isfinite(parsed):
            entries.append((math.log10(parsed), parsed, index // samples_per_route))
    if not entries:
        return []
    clusters: list[list[tuple[float, float, int]]] = []
    for entry in sorted(entries):
        best = None
        best_distance = math.inf
        for cluster in clusters:
            center = statistics.median(member[0] for member in cluster)
            distance = abs(entry[0] - center)
            if distance <= log_threshold and distance < best_distance:
                best = cluster
                best_distance = distance
        if best is None:
            clusters.append([entry])
        else:
            best.append(entry)

    def rank(cluster: list[tuple[float, float, int]]) -> tuple:
        logs = [member[0] for member in cluster]
        center = statistics.median(logs)
        dispersion = statistics.median(abs(value - center) for value in logs)
        route_count = len({member[2] for member in cluster})
        # Final component makes ties deterministic regardless of formation
        # order: prefer the cluster with the larger representative value only
        # as a last resort.
        return (-route_count, -len(cluster), dispersion, -center)

    selected = min(clusters, key=rank)
    final = statistics.median(member[1] for member in selected)
    if float(final).is_integer():
        return [str(int(final))]
    return [format(final, ".15g")]


def aggregate_audited_median(
    candidates: list[list[str]],
    *,
    diagnostics: list[dict[str, Any]] | None = None,
) -> list[str]:
    audited: list[list[str]] = []
    for index in range(len(candidates)):
        if diagnostics is None or index >= len(diagnostics):
            continue
        diagnostic = diagnostics[index]
        audit = diagnostic.get("audit") if isinstance(diagnostic, dict) else None
        if not isinstance(audit, dict) or audit.get("usable") is not True:
            continue
        normalized = audit.get("normalized_value_km2")
        if isinstance(normalized, (int, float)) and not isinstance(normalized, bool):
            audited.append([str(normalized)])
    return aggregate_median(audited) if audited else aggregate_median(candidates)


def aggregate_unit_equivalence(
    candidates: list[list[str]],
    *,
    cluster_tolerance: float = 0.05,
    unit_tolerance: float = 0.05,
) -> list[str]:
    from .area_unit_equivalence import select_unit_equivalent_value

    value, _ = select_unit_equivalent_value(
        candidates,
        cluster_tolerance=cluster_tolerance,
        unit_tolerance=unit_tolerance,
    )
    return [] if value is None else [value]


def aggregate_candidates(
    candidates: list[list[str]],
    policy: dict[str, Any] | None,
    diagnostics: list[dict[str, Any]] | None = None,
) -> list[str]:
    if not candidates:
        return []
    if policy is None:
        return candidates[0]
    strategy = policy.get("strategy")
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(f"unsupported aggregation strategy: {strategy}")
    if strategy == "frequency":
        if policy.get("empty_aware"):
            return aggregate_empty_aware_frequency(
                candidates,
                float(policy.get("threshold", 0.5)),
                diagnostics=diagnostics,
                empty_majority=int(policy.get("empty_majority", 3)),
            )
        return aggregate_frequency(candidates, float(policy.get("threshold", 0.5)))
    if strategy == "grouped_frequency":
        return aggregate_grouped_frequency(
            candidates,
            groups=int(policy["groups"]),
            threshold=float(policy.get("threshold", 0.5)),
            diagnostics=diagnostics,
        )
    if strategy == "majority":
        return aggregate_majority(
            candidates,
            diagnostics=diagnostics,
            explicit_empty_only=bool(policy.get("explicit_empty_only", False)),
            forced_weight=float(policy.get("forced_weight", 1.0)),
            minimum_votes=float(policy.get("minimum_votes", 0.0)),
        )
    if strategy == "union":
        return aggregate_union(candidates, diagnostics=diagnostics)
    if strategy == "audited_median":
        return aggregate_audited_median(candidates, diagnostics=diagnostics)
    if strategy == "dominant_cluster":
        return aggregate_dominant_cluster(
            candidates,
            cluster_tolerance=float(policy.get("cluster_tolerance", 0.05)),
            diagnostics=diagnostics,
            forced_weight=float(policy.get("forced_weight", 1.0)),
        )
    if strategy == "unit_equivalence":
        return aggregate_unit_equivalence(
            candidates,
            cluster_tolerance=float(policy.get("cluster_tolerance", 0.05)),
            unit_tolerance=float(policy.get("unit_tolerance", 0.05)),
        )
    if strategy == "route_consensus":
        return aggregate_route_consensus(
            candidates,
            log_threshold=float(policy.get("log_threshold", 0.05)),
            samples_per_route=int(policy.get("samples_per_route", 5)),
        )
    if strategy in {"cluster_choice", "metadata_judge"}:
        raise ValueError(f"{strategy} requires model-assisted selection")
    return aggregate_median(candidates)
