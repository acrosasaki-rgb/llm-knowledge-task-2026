from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .area_clustering import format_number, parse_candidate_number


UNIT_FACTORS_TO_KM2 = {
    "square_mile": 2.589988110336,
    "hectare": 0.01,
    "acre": 0.0040468564224,
}


@dataclass(frozen=True)
class UnitCollision:
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


def find_unit_collisions(
    candidates: list[list[str]], tolerance: float = 0.05
) -> list[UnitCollision]:
    """Find conversions of one raw candidate that match another raw candidate.

    The converted values are evidence only. They never become output candidates.
    For each source/unit pair, only the closest observed target is retained.
    """

    if not 0 < tolerance < 1:
        raise ValueError("unit collision tolerance must be in (0, 1)")
    parsed = [parse_candidate_number(candidate) for candidate in candidates]
    positive = [
        (index, value)
        for index, value in enumerate(parsed)
        if value is not None and math.isfinite(value) and value > 0
    ]
    collisions: list[UnitCollision] = []
    for source_index, source_value in positive:
        for unit, factor in UNIT_FACTORS_TO_KM2.items():
            converted = source_value * factor
            matches = []
            for target_index, target_value in positive:
                if target_index == source_index:
                    continue
                relative_error = abs(target_value - converted) / max(
                    abs(converted), 1e-12
                )
                if relative_error <= tolerance:
                    matches.append(
                        (relative_error, target_index, target_value)
                    )
            if matches:
                relative_error, target_index, target_value = min(matches)
                collisions.append(
                    UnitCollision(
                        source_index=source_index,
                        source_value=source_value,
                        assumed_unit=unit,
                        converted_value_km2=converted,
                        target_index=target_index,
                        target_value=target_value,
                        relative_error=relative_error,
                    )
                )
    return collisions


def filter_unit_collision_sources(
    candidates: list[list[str]], tolerance: float = 0.05
) -> tuple[list[list[str]], list[UnitCollision], list[int]]:
    """Remove raw candidates acting as conversion sources in a collision.

    All collision edges are calculated against the original candidate set before
    any removal. If the rule would remove every candidate, the original set is
    retained as a defensive fallback.
    """

    collisions = find_unit_collisions(candidates, tolerance)
    removed_indices = sorted({item.source_index for item in collisions})
    removed = set(removed_indices)
    filtered = [
        candidate for index, candidate in enumerate(candidates) if index not in removed
    ]
    if not filtered:
        return list(candidates), collisions, []
    return filtered, collisions, removed_indices
