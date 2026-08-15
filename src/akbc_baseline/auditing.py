from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

from .parsing import extract_final_text


SUPPORTED_UNITS = {
    "square_kilometer": 1.0,
    "square_mile": 2.589988110336,
    "hectare": 0.01,
    "acre": 0.0040468564224,
}
ACCEPTED_SCOPES = {"total_geographic_area", "water_surface_area"}
SUPPORTED_SCOPES = ACCEPTED_SCOPES | {
    "land_area_only",
    "administrative_subarea",
    "metropolitan_area",
    "historical_area",
    "unknown",
}
AREA_AUDIT_FIELDS = (
    "value",
    "unit",
    "scope",
    "reference_year",
    "normalized_value_km2",
    "keep",
)
AREA_METADATA_FIELDS = (
    "value",
    "unit",
    "scope",
    "reference_year",
    "normalized_value_km2",
)


@dataclass(frozen=True)
class AreaAuditResult:
    metadata: dict[str, Any]


def _first_json_value(text: str) -> Any:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, (dict, list)):
            return value
    return None


def _audit_mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, list):
        return None
    result: dict[str, Any] = {}
    if len(value) == len(AREA_AUDIT_FIELDS) and all(
        isinstance(item, str) and "=" not in item for item in value
    ):
        result = dict(zip(AREA_AUDIT_FIELDS, value, strict=True))
    else:
        for item in value:
            if not isinstance(item, str) or "=" not in item:
                return None
            key, field_value = item.split("=", 1)
            result[key.strip()] = field_value.strip()
    keep = result.get("keep")
    if isinstance(keep, str) and keep.casefold() in {"true", "false"}:
        result["keep"] = keep.casefold() == "true"
    return result


def _metadata_mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, list):
        return None
    if len(value) == len(AREA_METADATA_FIELDS) and all(
        isinstance(item, str) and "=" not in item for item in value
    ):
        return dict(zip(AREA_METADATA_FIELDS, value, strict=True))
    result: dict[str, Any] = {}
    for item in value:
        if not isinstance(item, str) or "=" not in item:
            return None
        key, field_value = item.split("=", 1)
        result[key.strip()] = field_value.strip()
    return result


def _number(value: Any) -> float | None:
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return None
    match = re.search(
        r"[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?", str(value)
    )
    if not match:
        return None
    try:
        parsed = float(match.group(0).replace(",", ""))
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def _close(left: float, right: float, tolerance: float = 0.05) -> bool:
    return abs(left - right) / max(abs(right), 1e-12) <= tolerance


def parse_area_audit(text: str, original_entities: list[str]) -> AreaAuditResult:
    cleaned = extract_final_text(text)
    parsed = _audit_mapping(_first_json_value(cleaned))
    if parsed is None:
        return AreaAuditResult({"status": "parse_failure", "usable": False})

    original_value = _number(original_entities[0]) if original_entities else None
    value = _number(parsed.get("value"))
    normalized = _number(parsed.get("normalized_value_km2"))
    unit = parsed.get("unit")
    scope = parsed.get("scope")
    reference_year = parsed.get("reference_year")
    keep = parsed.get("keep")
    unit = unit.strip().casefold() if isinstance(unit, str) else None
    scope = scope.strip().casefold() if isinstance(scope, str) else None
    reference_year = (
        str(reference_year).strip()
        if isinstance(reference_year, (str, int))
        and not isinstance(reference_year, bool)
        else None
    )

    metadata: dict[str, Any] = {
        "status": "parsed",
        "usable": False,
        "value": value,
        "unit": unit,
        "scope": scope,
        "reference_year": reference_year,
        "normalized_value_km2": normalized,
        "keep": keep if isinstance(keep, bool) else None,
    }
    rejection_reasons: list[str] = []
    if original_value is None or value is None or not _close(value, original_value):
        rejection_reasons.append("value_mismatch")
    multiplier = SUPPORTED_UNITS.get(unit or "")
    if multiplier is None:
        rejection_reasons.append("unsupported_unit")
    if normalized is None or (
        value is not None
        and multiplier is not None
        and not _close(normalized, value * multiplier)
    ):
        rejection_reasons.append("conversion_mismatch")
    if scope not in SUPPORTED_SCOPES:
        rejection_reasons.append("unsupported_scope")
    elif scope not in ACCEPTED_SCOPES:
        rejection_reasons.append("unexpected_scope")
    if not reference_year:
        rejection_reasons.append("missing_reference_year")
    elif reference_year.casefold() == "unknown":
        rejection_reasons.append("unknown_reference_year")
    elif reference_year.casefold() not in {"current", "general"} and not re.fullmatch(
        r"\d{4}", reference_year
    ):
        rejection_reasons.append("unsupported_reference_year")
    if keep is not True:
        rejection_reasons.append("keep_false")

    metadata["rejection_reasons"] = rejection_reasons
    metadata["usable"] = not rejection_reasons
    metadata["status"] = "accepted" if not rejection_reasons else "rejected"
    return AreaAuditResult(metadata)


def parse_area_metadata(text: str, original_entities: list[str]) -> AreaAuditResult:
    cleaned = extract_final_text(text)
    parsed = _metadata_mapping(_first_json_value(cleaned))
    if parsed is None:
        return AreaAuditResult({"status": "parse_failure"})

    original_value = _number(original_entities[0]) if original_entities else None
    value = _number(parsed.get("value"))
    normalized = _number(parsed.get("normalized_value_km2"))
    unit = parsed.get("unit")
    scope = parsed.get("scope")
    reference_year = parsed.get("reference_year")
    unit = unit.strip().casefold() if isinstance(unit, str) else None
    scope = scope.strip().casefold() if isinstance(scope, str) else None
    reference_year = (
        str(reference_year).strip()
        if isinstance(reference_year, (str, int))
        and not isinstance(reference_year, bool)
        else None
    )
    multiplier = SUPPORTED_UNITS.get(unit or "")
    conversion_consistent = (
        _close(normalized, value * multiplier)
        if normalized is not None and value is not None and multiplier is not None
        else None
    )
    return AreaAuditResult(
        {
            "status": "parsed",
            "value": value,
            "unit": unit,
            "scope": scope,
            "reference_year": reference_year,
            "normalized_value_km2": normalized,
            "value_matches_candidate": (
                _close(value, original_value)
                if value is not None and original_value is not None
                else False
            ),
            "conversion_consistent": conversion_consistent,
        }
    )
