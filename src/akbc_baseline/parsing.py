from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


NUMERIC_RELATIONS = {"hasArea", "hasCapacity"}


@dataclass(frozen=True)
class ParseResult:
    object_entities: list[str]
    status: str


def extract_final_text(decoded: str) -> str:
    """Discard model reasoning and retain only the user-facing answer."""
    text = decoded
    harmony_marker = "<|channel|>final<|message|>"
    if harmony_marker in text:
        text = text.rsplit(harmony_marker, 1)[1]
        text = re.split(r"<\|(?:return|end|call)\|>", text, maxsplit=1)[0]
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<\|[^>]+\|>", "", text)
    return text.strip()


def _first_json_array(text: str) -> list[Any] | None:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "[":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            return value
    return None


def parse_object_entities(text: str, relation: str) -> list[str]:
    return parse_object_entities_with_status(text, relation).object_entities


def parse_object_entities_with_status(text: str, relation: str) -> ParseResult:
    cleaned = extract_final_text(text)
    lowered = cleaned.strip().lower()
    if lowered in {"", "none", "null", "unknown"}:
        return ParseResult([], "parse_failure")
    if lowered == "[]":
        return ParseResult([], "explicit_empty")

    parsed = _first_json_array(cleaned)
    if parsed is not None:
        values = [str(value).strip() for value in parsed if str(value).strip()]
        deduplicated = _deduplicate(values)
        status = "json" if deduplicated else "explicit_empty"
        return ParseResult(deduplicated, status)

    if relation in NUMERIC_RELATIONS:
        number = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", cleaned)
        return (
            ParseResult([number.group(0)], "numeric_fallback")
            if number
            else ParseResult([], "parse_failure")
        )

    values = re.split(r"\s*(?:,|;|\n)\s*", cleaned)
    normalized = []
    for value in values:
        value = re.sub(r"^[-*\d.)\s]+", "", value).strip(" \t\r\n\"'")
        if value and value.lower() not in {"none", "null", "unknown"}:
            normalized.append(value)
    deduplicated = _deduplicate(normalized)
    return ParseResult(
        deduplicated,
        "text_fallback" if deduplicated else "parse_failure",
    )


def _deduplicate(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = " ".join(value.casefold().split())
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result
