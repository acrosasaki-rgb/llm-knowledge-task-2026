from akbc_baseline.parsing import (
    extract_final_text,
    parse_object_entities,
    parse_object_entities_with_status,
)


def test_extracts_harmony_final_without_reasoning() -> None:
    decoded = (
        "<|channel|>analysis<|message|>private reasoning"
        '<|channel|>final<|message|>["Haiti"]<|return|>'
    )
    assert extract_final_text(decoded) == '["Haiti"]'
    assert parse_object_entities(decoded, "countryLandBordersCountry") == ["Haiti"]


def test_extracts_qwen_answer_after_thinking() -> None:
    decoded = '<think>private reasoning</think>\n["Warsaw"]'
    assert parse_object_entities(decoded, "personHasCityOfDeath") == ["Warsaw"]


def test_parses_empty_and_numeric_fallback() -> None:
    assert parse_object_entities("[]", "companyTradesAtStockExchange") == []
    empty = parse_object_entities_with_status(
        "[]", "companyTradesAtStockExchange"
    )
    assert empty.object_entities == []
    assert empty.status == "explicit_empty"
    assert parse_object_entities("The answer is 60,206 people", "hasCapacity") == [
        "60,206"
    ]
    assert (
        parse_object_entities_with_status(
            "The answer is 60,206 people", "hasCapacity"
        ).status
        == "numeric_fallback"
    )


def test_distinguishes_parse_failure_from_explicit_empty() -> None:
    parsed = parse_object_entities_with_status("", "companyTradesAtStockExchange")

    assert parsed.object_entities == []
    assert parsed.status == "parse_failure"


def test_deduplicates_case_insensitively() -> None:
    assert parse_object_entities(
        '["NASDAQ", "nasdaq"]', "companyTradesAtStockExchange"
    ) == ["NASDAQ"]
