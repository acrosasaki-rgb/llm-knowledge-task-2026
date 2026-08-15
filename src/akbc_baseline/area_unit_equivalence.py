from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .area_clustering import NumericCluster, cluster_numeric_candidates, format_number


# Only factors that are not powers of ten belong here. A power-of-ten factor
# such as hectare (0.01) cannot be told apart from a misplaced decimal point,
# so treating it as unit evidence would move support onto the wrong decade.
# Order-of-magnitude mistakes are a separate category and are not handled here.
UNIT_EQUIVALENCE_FACTORS = {
    "square_mile": 2.589988110336,
    "acre": 0.0040468564224,
}


def is_power_of_ten(factor: float, tolerance: float = 1e-9) -> bool:
    if factor <= 0:
        return False
    exponent = math.log10(factor)
    return abs(exponent - round(exponent)) <= tolerance


def validate_factors(factors: dict[str, float]) -> None:
    for unit, factor in factors.items():
        if factor <= 0:
            raise ValueError(f"unit factor must be positive: {unit}")
        if is_power_of_ten(factor):
            raise ValueError(
                f"power-of-ten factor is a digit error, not a unit error: {unit}"
            )


validate_factors(UNIT_EQUIVALENCE_FACTORS)


@dataclass(frozen=True)
class UnitEquivalence:
    """One cluster read as another cluster expressed in a different unit."""

    source_index: int
    source_value: float
    assumed_unit: str
    converted_value_km2: float
    target_index: int
    target_value: float
    relative_error: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_index": self.source_index,
            "source_value": format_number(self.source_value),
            "assumed_unit": self.assumed_unit,
            "converted_value_km2": format_number(self.converted_value_km2),
            "target_index": self.target_index,
            "target_value": format_number(self.target_value),
            "relative_error": self.relative_error,
        }


def find_cluster_unit_equivalences(
    clusters: list[NumericCluster],
    tolerance: float = 0.05,
    factors: dict[str, float] | None = None,
) -> list[UnitEquivalence]:
    """Link clusters whose representatives differ by one unit conversion.

    Only the closest target is kept per source and unit. A pair that points at
    each other in both directions is dropped, because the conversion direction
    is then undetermined.
    """

    if not 0 < tolerance < 1:
        raise ValueError("unit equivalence tolerance must be in (0, 1)")
    factors = UNIT_EQUIVALENCE_FACTORS if factors is None else factors
    validate_factors(factors)
    reps = [cluster.representative for cluster in clusters]
    edges: list[UnitEquivalence] = []
    for source_index, source_value in enumerate(reps):
        if not math.isfinite(source_value) or source_value <= 0:
            continue
        for unit, factor in factors.items():
            converted = source_value * factor
            best: tuple[float, int, float] | None = None
            for target_index, target_value in enumerate(reps):
                if target_index == source_index:
                    continue
                if not math.isfinite(target_value) or target_value <= 0:
                    continue
                relative_error = abs(target_value - converted) / max(
                    abs(converted), 1e-12
                )
                if relative_error <= tolerance and (
                    best is None or relative_error < best[0]
                ):
                    best = (relative_error, target_index, target_value)
            if best is not None:
                relative_error, target_index, target_value = best
                edges.append(
                    UnitEquivalence(
                        source_index=source_index,
                        source_value=source_value,
                        assumed_unit=unit,
                        converted_value_km2=converted,
                        target_index=target_index,
                        target_value=target_value,
                        relative_error=relative_error,
                    )
                )
    pairs = {(edge.source_index, edge.target_index) for edge in edges}
    return [
        edge for edge in edges if (edge.target_index, edge.source_index) not in pairs
    ]


def merge_cluster_support(
    clusters: list[NumericCluster], edges: list[UnitEquivalence]
) -> list[int]:
    """Move each source cluster's support to the end of its conversion chain.

    No cluster is removed and no converted value becomes selectable. Support is
    transferred only, so the surviving representative is always an observed
    cluster representative.
    """

    support = [cluster.support for cluster in clusters]
    merged = list(support)
    moved = [False] * len(clusters)
    for edge in sorted(edges, key=lambda item: item.relative_error):
        source = edge.source_index
        if moved[source]:
            continue
        sink = edge.target_index
        seen = {source}
        while True:
            following = next(
                (
                    item.target_index
                    for item in edges
                    if item.source_index == sink and item.target_index not in seen
                ),
                None,
            )
            if following is None:
                break
            seen.add(sink)
            sink = following
        if sink == source:
            continue
        merged[sink] += support[source]
        merged[source] = 0
        moved[source] = True
    return merged


def rank_clusters(
    clusters: list[NumericCluster], merged_support: list[int]
) -> list[int]:
    return sorted(
        range(len(clusters)),
        key=lambda index: (
            merged_support[index],
            -clusters[index].spread,
            -clusters[index].representative,
        ),
        reverse=True,
    )


def select_unit_equivalent_value(
    candidates: list[list[str]],
    cluster_tolerance: float = 0.05,
    unit_tolerance: float = 0.05,
    factors: dict[str, float] | None = None,
) -> tuple[str | None, dict[str, Any]]:
    """Pick the representative of the cluster with the most merged support."""

    if not 0 < cluster_tolerance < 1:
        raise ValueError("cluster tolerance must be in (0, 1)")
    clusters = cluster_numeric_candidates(candidates, cluster_tolerance)
    if not clusters:
        return None, {
            "strategy": "unit_equivalence",
            "clusters": 0,
            "equivalences": [],
            "converted_values_added": False,
        }
    edges = find_cluster_unit_equivalences(clusters, unit_tolerance, factors)
    merged = merge_cluster_support(clusters, edges)
    order = rank_clusters(clusters, merged)
    winner = order[0]
    diagnostics = {
        "strategy": "unit_equivalence",
        "clusters": len(clusters),
        "cluster_tolerance": cluster_tolerance,
        "unit_tolerance": unit_tolerance,
        "cluster_representatives": [
            format_number(cluster.representative) for cluster in clusters
        ],
        "original_support": [cluster.support for cluster in clusters],
        "merged_support": merged,
        "selected_index": winner,
        "dominant_index": 0,
        "changed_dominant": winner != 0,
        "equivalences": [edge.as_dict() for edge in edges],
        "converted_values_added": False,
    }
    return format_number(clusters[winner].representative), diagnostics
