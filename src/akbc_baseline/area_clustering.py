from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass
from typing import Any


NUMBER_PATTERN = re.compile(
    r"[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?"
)


@dataclass(frozen=True)
class NumericCluster:
    representative: float
    members: tuple[float, ...]

    @property
    def support(self) -> int:
        return len(self.members)

    @property
    def spread(self) -> float:
        minimum = min(self.members)
        if minimum <= 0:
            return math.inf
        return max(self.members) / minimum


def format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return format(value, ".15g")


def parse_candidate_number(candidate: list[str]) -> float | None:
    for value in candidate:
        if not isinstance(value, str):
            continue
        match = NUMBER_PATTERN.search(value)
        if match is None:
            continue
        try:
            return float(match.group(0).replace(",", ""))
        except ValueError:
            continue
    return None


def cluster_numeric_candidates(
    candidates: list[list[str]], tolerance: float = 0.05
) -> list[NumericCluster]:
    """Build disjoint dense clusters, returning the dominant cluster first.

    Each iteration finds the observed center whose five-percent neighborhood has
    the most remaining values, emits that neighborhood, and removes its members.
    This preserves the previous post-hoc dominant-cluster definition while also
    producing alternatives for a subsequent multiple-choice question.
    """

    if not 0 < tolerance < 1:
        raise ValueError("cluster tolerance must be in (0, 1)")
    values = [
        value
        for candidate in candidates
        for value in [parse_candidate_number(candidate)]
        if value is not None and math.isfinite(value)
    ]
    if not values:
        return []

    remaining = list(enumerate(values))
    overall_median = statistics.median(values)
    clusters: list[NumericCluster] = []
    while remaining:
        best: tuple[
            tuple[int, float, float, float], list[tuple[int, float]]
        ] | None = None
        for _, center in remaining:
            denominator = max(abs(center), 1e-12)
            members = [
                item
                for item in remaining
                if abs(item[1] - center) / denominator <= tolerance
            ]
            member_values = [value for _, value in members]
            representative = statistics.median(member_values)
            minimum = min(member_values)
            spread = (
                max(member_values) / minimum if minimum > 0 else math.inf
            )
            rank = (
                len(members),
                -spread,
                -abs(representative - overall_median),
                -representative,
            )
            if best is None or rank > best[0]:
                best = (rank, members)
        assert best is not None
        selected = best[1]
        selected_ids = {index for index, _ in selected}
        member_values = tuple(sorted(value for _, value in selected))
        clusters.append(
            NumericCluster(
                representative=float(statistics.median(member_values)),
                members=member_values,
            )
        )
        remaining = [item for item in remaining if item[0] not in selected_ids]
    return clusters


def cluster_choice_records(
    clusters: list[NumericCluster],
) -> list[dict[str, Any]]:
    dominant = clusters[0] if clusters else None
    records = []
    for choice_index, cluster in enumerate(
        sorted(clusters, key=lambda item: item.representative), start=1
    ):
        records.append(
            {
                "choice_id": choice_index,
                "value_km2": format_number(cluster.representative),
                "support": cluster.support,
                "minimum_km2": format_number(min(cluster.members)),
                "maximum_km2": format_number(max(cluster.members)),
                "dominant": cluster is dominant,
            }
        )
    return records


def area_cluster_selection_messages(
    subject: str, clusters: list[NumericCluster]
) -> list[dict[str, str]]:
    choices = sorted(clusters, key=lambda item: item.representative)
    choice_lines = "\n".join(
        f"{index}. {format_number(cluster.representative)} km2"
        for index, cluster in enumerate(choices, start=1)
    )
    return [
        {
            "role": "system",
            "content": (
                "Answer the supplied area question using knowledge already "
                "contained in the model. Return exactly one JSON array containing "
                "the numeric value copied from one choice and no explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"What is the total geographic area of {subject!r} in square "
                "kilometers? For a lake, use water-surface area. Choose exactly "
                "one of the following values and copy that numeric value into the "
                "JSON array.\n\n"
                + choice_lines
            ),
        },
    ]


def match_cluster_choice(
    values: list[str], clusters: list[NumericCluster]
) -> NumericCluster | None:
    if len(values) != 1:
        return None
    parsed = parse_candidate_number(values)
    if parsed is None:
        return None
    for cluster in clusters:
        if math.isclose(
            parsed, cluster.representative, rel_tol=1e-12, abs_tol=1e-12
        ):
            return cluster
    return None
