import json

from akbc_baseline.prompting import PromptBuilder


def test_prompt_selection_is_deterministic_and_uses_alias_head() -> None:
    train_rows = [
        {
            "SubjectEntity": f"Person {index}",
            "Relation": "personHasCityOfDeath",
            "ObjectEntities": [[f"City {index}", f"Alias {index}"]],
        }
        for index in range(8)
    ]
    builder = PromptBuilder(
        train_rows,
        {"personHasCityOfDeath": "Where did {subject_entity} die?"},
        few_shot=3,
        seed=42,
    )

    first = builder.build("Target Person", "personHasCityOfDeath")
    second = builder.build("Target Person", "personHasCityOfDeath")
    alternate = builder.build(
        "Target Person", "personHasCityOfDeath", candidate_index=1
    )

    assert first == second
    assert first != alternate
    assert first[-1]["content"] == "Where did Target Person die?"
    assistant_messages = [m for m in first if m["role"] == "assistant"]
    assert len(assistant_messages) == 3
    assert all(json.loads(m["content"])[0].startswith("City ") for m in assistant_messages)


def test_relation_instructions_are_added_to_system_message() -> None:
    builder = PromptBuilder(
        [],
        {"countryLandBordersCountry": "Which countries border {subject_entity}?"},
        few_shot=0,
        seed=42,
        relation_instructions={
            "countryLandBordersCountry": "Exclude maritime borders."
        },
    )

    messages = builder.build("France", "countryLandBordersCountry")

    assert messages[0]["role"] == "system"
    assert "Exclude maritime borders." in messages[0]["content"]


def test_candidate_instruction_scopes_only_the_final_question() -> None:
    builder = PromptBuilder(
        [
            {
                "SubjectEntity": "Old award",
                "Relation": "awardWonBy",
                "ObjectEntities": [["Alice"]],
            }
        ],
        {"awardWonBy": "Who won {subject_entity}?"},
        few_shot=1,
        seed=42,
        candidate_instructions={
            "awardWonBy": ["Use 1900 through 1909.", "Use 1910 through 1919."]
        },
    )

    messages = builder.build("Target award", "awardWonBy", candidate_index=1)

    assert "Candidate-specific scope" not in messages[1]["content"]
    assert messages[-1]["content"].endswith(
        "Candidate-specific scope: Use 1910 through 1919."
    )
    assert (
        builder.candidate_instruction_for("awardWonBy", 0)
        == "Use 1900 through 1909."
    )


def test_conversation_followup_continues_backward_without_repeating() -> None:
    builder = PromptBuilder(
        [],
        {"awardWonBy": "Who won {subject_entity}?"},
        few_shot=0,
        seed=42,
        candidate_instructions={
            "awardWonBy": ["Use 2020 through 2026.", "Use 2010 through 2019."]
        },
    )

    followup = builder.conversation_followup(
        "Turing Award", "awardWonBy", candidate_index=1
    )

    assert followup["role"] == "user"
    assert "Continue the same backward chronological enumeration" in followup["content"]
    assert "do not repeat" in followup["content"]
    assert "Who won Turing Award?" in followup["content"]
    assert followup["content"].endswith(
        "Candidate-specific scope: Use 2010 through 2019."
    )


def test_area_audit_followup_requests_structured_self_check() -> None:
    builder = PromptBuilder(
        [],
        {"hasArea": "What is the area of {subject_entity} in square kilometers?"},
        few_shot=0,
        seed=42,
    )

    messages = builder.build("Icaria", "hasArea")
    followup = builder.area_audit_followup("Icaria")

    assert followup["role"] == "user"
    assert '"unit=square_kilometer|square_mile|hectare|acre|unknown"' in followup["content"]
    assert '"normalized_value_km2=number or unknown"' in followup["content"]
    assert "Do not silently replace" in followup["content"]


def test_area_metadata_followup_has_no_correctness_or_retention_field() -> None:
    builder = PromptBuilder(
        [],
        {"hasArea": "What is the area of {subject_entity} in square kilometers?"},
        few_shot=0,
        seed=42,
    )

    followup = builder.area_metadata_followup("Icaria")

    assert '"reference_year=current|general|YYYY|unknown"' in followup["content"]
    assert '"normalized_value_km2=number or unknown"' in followup["content"]
    assert '"keep=' not in followup["content"]
    assert "metadata estimation only" in followup["content"]


def test_area_metadata_selection_receives_all_candidates_and_attributes() -> None:
    builder = PromptBuilder([], {}, few_shot=0, seed=42)
    messages = builder.area_metadata_selection(
        "Icaria",
        [["99"], ["255"]],
        [
            {"metadata": {"unit": "square_mile", "scope": "land_area_only"}},
            {"metadata": {"unit": "square_kilometer", "scope": "total_geographic_area"}},
        ],
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "one numeric string" in messages[0]["content"]
    assert messages[1]["content"].count('"candidate_id"') == 2
    assert '"original_answer":"99"' in messages[1]["content"]
    assert '"unit":"square_mile"' in messages[1]["content"]
    assert "do not blindly follow" in messages[1]["content"]
