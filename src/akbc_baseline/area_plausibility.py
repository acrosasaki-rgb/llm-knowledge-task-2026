from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

from .area_clustering import (
    NumericCluster,
    cluster_numeric_candidates,
    format_number,
    parse_candidate_number,
)


AREA_BINS: dict[str, tuple[float, float]] = {
    "lt_0.01": (1e-12, 0.01),
    "0.01_0.1": (0.01, 0.1),
    "0.1_1": (0.1, 1.0),
    "1_10": (1.0, 10.0),
    "10_100": (10.0, 100.0),
    "100_1000": (100.0, 1000.0),
    "1000_10000": (1000.0, 10000.0),
    "10000_100000": (10000.0, 100000.0),
    "gte_100000": (100000.0, 1e12),
}
UNIT_FACTORS = {
    "square_kilometer": 1.0,
    "square_mile": 2.589988110336,
    "hectare": 0.01,
    "acre": 0.0040468564224,
}
UNIT_PENALTIES = {
    "square_kilometer": 0.0,
    "square_mile": 0.03,
    "hectare": 0.08,
    "acre": 0.08,
}
SHAPE_FILL_RANGES = {
    "compact": (0.45, 0.9),
    "moderate": (0.25, 0.8),
    "elongated": (0.1, 0.6),
    "fragmented": (0.03, 0.5),
    "unknown": (0.05, 1.0),
}
NUMBER_PATTERN = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class AreaHypothesis:
    normalized_value_km2: float
    source_value: float
    assumed_unit: str
    support: int
    dominant_source: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "normalized_value_km2": format_number(self.normalized_value_km2),
            "source_value": format_number(self.source_value),
            "assumed_unit": self.assumed_unit,
            "support": self.support,
            "dominant_source": self.dominant_source,
        }


def scale_estimate_messages(subject: str) -> list[dict[str, str]]:
    bins = ", ".join(AREA_BINS)
    return [
        {
            "role": "system",
            "content": (
                "Estimate only the broad physical scale of a geographic entity. "
                "Do not provide its exact area. Return exactly one JSON array of "
                "two strings and no explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Without seeing any candidate area values, classify {subject!r}. "
                "For a land entity estimate total geographic area; for a lake "
                "estimate water-surface area. Return "
                '["entity_type=island|lake|region|other|unknown",'
                f'"area_bin={bins}"]. Choose exactly one listed area_bin in km2.'
            ),
        },
    ]


def dimension_estimate_messages(subject: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Estimate geographic dimensions independently of any area "
                "candidates. Do not state or calculate the entity's area. Return "
                "exactly one JSON array of three strings and no explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Estimate the approximate maximum length and maximum width of "
                f"{subject!r} in kilometers. Use unknown when they cannot be "
                "recalled independently. Classify its footprint shape. Return "
                '["length_km=number|unknown","width_km=number|unknown",'
                '"shape=compact|moderate|elongated|fragmented|unknown"].'
            ),
        },
    ]


def _json_strings(text: str) -> list[str]:
    try:
        value = json.loads(text.strip())
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\[[\s\S]*?\]", text)
        if match is None:
            return []
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for item in _json_strings(text):
        key, separator, value = item.partition("=")
        if separator:
            fields[key.strip()] = value.strip()
    return fields


def _number(value: str | None) -> float | None:
    if value is None or value.casefold() == "unknown":
        return None
    match = NUMBER_PATTERN.search(value)
    if match is None:
        return None
    try:
        parsed = float(match.group(0).replace(",", ""))
    except ValueError:
        return None
    return parsed if parsed > 0 and math.isfinite(parsed) else None


def parse_scale_estimate(text: str) -> dict[str, Any]:
    fields = _fields(text)
    entity_type = fields.get("entity_type", "unknown")
    if entity_type not in {"island", "lake", "region", "other", "unknown"}:
        entity_type = "unknown"
    area_bin = fields.get("area_bin", "unknown")
    bounds = AREA_BINS.get(area_bin)
    return {
        "status": "parsed" if bounds is not None else "invalid",
        "entity_type": entity_type,
        "area_bin": area_bin,
        "lower_km2": bounds[0] if bounds else None,
        "upper_km2": bounds[1] if bounds else None,
    }


def parse_dimension_estimate(text: str) -> dict[str, Any]:
    fields = _fields(text)
    length = _number(fields.get("length_km"))
    width = _number(fields.get("width_km"))
    shape = fields.get("shape", "unknown")
    if shape not in SHAPE_FILL_RANGES:
        shape = "unknown"
    lower = upper = None
    if length is not None and width is not None:
        fill_lower, fill_upper = SHAPE_FILL_RANGES[shape]
        bounding_area = length * width
        lower = bounding_area * fill_lower
        upper = bounding_area * fill_upper
    return {
        "status": "parsed" if lower is not None else "invalid",
        "length_km": length,
        "width_km": width,
        "shape": shape,
        "lower_km2": lower,
        "upper_km2": upper,
    }


def build_area_hypotheses(
    candidates: list[list[str]], tolerance: float = 0.05
) -> tuple[list[NumericCluster], list[AreaHypothesis]]:
    clusters = cluster_numeric_candidates(candidates, tolerance)
    if not clusters:
        return clusters, []
    raw_values = [
        value
        for candidate in candidates
        for value in [parse_candidate_number(candidate)]
        if value is not None and math.isfinite(value) and value > 0
    ]
    dominant = clusters[0]
    hypotheses = [
        AreaHypothesis(
            normalized_value_km2=dominant.representative,
            source_value=dominant.representative,
            assumed_unit="square_kilometer",
            support=dominant.support,
            dominant_source=True,
        )
    ]
    for source_value in raw_values:
        source_support = sum(
            abs(other - source_value) / max(abs(source_value), 1e-12)
            <= tolerance
            for other in raw_values
        )
        dominant_source = (
            abs(source_value - dominant.representative)
            / max(abs(dominant.representative), 1e-12)
            <= tolerance
        )
        hypotheses.extend(
            AreaHypothesis(
                normalized_value_km2=source_value * factor,
                source_value=source_value,
                assumed_unit=unit,
                support=source_support,
                dominant_source=dominant_source,
            )
            for unit, factor in UNIT_FACTORS.items()
        )
    return clusters, hypotheses


def decimal_relation_records(
    clusters: list[NumericCluster], tolerance: float = 0.05
) -> list[dict[str, Any]]:
    records = []
    values = sorted(cluster.representative for cluster in clusters)
    for index, lower in enumerate(values):
        if lower <= 0:
            continue
        for higher in values[index + 1 :]:
            ratio = higher / lower
            for exponent in range(1, 7):
                expected = 10**exponent
                if abs(ratio - expected) / expected <= tolerance:
                    records.append(
                        {
                            "lower": format_number(lower),
                            "higher": format_number(higher),
                            "power_of_ten": exponent,
                            "ratio": ratio,
                        }
                    )
                    break
    return records


def _interval_distance(
    value: float, bounds: tuple[float, float] | None
) -> float:
    if bounds is None:
        return 0.0
    lower, upper = bounds
    if lower <= value <= upper:
        return 0.0
    boundary = lower if value < lower else upper
    return abs(math.log10(value / boundary))


def select_area_hypothesis(
    hypotheses: list[AreaHypothesis],
    *,
    scale_bounds: tuple[float, float] | None,
    dimension_bounds: tuple[float, float] | None,
    allow_unit_conversion: bool,
    minimum_improvement: float = 0.15,
) -> tuple[AreaHypothesis | None, dict[str, Any]]:
    usable = [
        hypothesis
        for hypothesis in hypotheses
        if allow_unit_conversion
        or hypothesis.assumed_unit == "square_kilometer"
    ]
    if not usable:
        return None, {"status": "no_hypotheses"}
    base = next(
        (
            hypothesis
            for hypothesis in usable
            if hypothesis.dominant_source
            and hypothesis.assumed_unit == "square_kilometer"
        ),
        usable[0],
    )

    def evidence_score(hypothesis: AreaHypothesis) -> float:
        distances = []
        if scale_bounds is not None:
            distances.append(
                _interval_distance(hypothesis.normalized_value_km2, scale_bounds)
            )
        if dimension_bounds is not None:
            distances.append(
                _interval_distance(
                    hypothesis.normalized_value_km2, dimension_bounds
                )
            )
        return sum(distances) / len(distances) if distances else 0.0

    base_score = evidence_score(base)
    ranked = sorted(
        usable,
        key=lambda hypothesis: (
            evidence_score(hypothesis) + UNIT_PENALTIES[hypothesis.assumed_unit],
            not hypothesis.dominant_source,
            -hypothesis.support,
            hypothesis.normalized_value_km2,
        ),
    )
    proposed = ranked[0]
    proposed_score = evidence_score(proposed) + UNIT_PENALTIES[proposed.assumed_unit]
    improvement = base_score - proposed_score
    selected = proposed if improvement >= minimum_improvement else base
    return selected, {
        "status": "selected",
        "base_value_km2": format_number(base.normalized_value_km2),
        "base_score": base_score,
        "proposed_value_km2": format_number(proposed.normalized_value_km2),
        "proposed_unit": proposed.assumed_unit,
        "proposed_score": proposed_score,
        "improvement": improvement,
        "overrode_base": selected is proposed and proposed is not base,
    }
