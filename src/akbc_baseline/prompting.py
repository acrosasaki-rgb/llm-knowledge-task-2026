from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from typing import Any

from .data import canonical_gold_labels
from .area_clustering import NumericCluster, area_cluster_selection_messages


BASE_SYSTEM_MESSAGE = (
    "Answer the final question using only knowledge already contained in the model. "
    "Return exactly one JSON array of strings and no explanation. "
    "Include every correct object you know. Return [] when there is no answer or "
    "when the answer is unknown. For numeric answers, return one numeric string."
)


class PromptBuilder:
    def __init__(
        self,
        train_rows: list[dict[str, Any]],
        templates: dict[str, str],
        few_shot: int,
        seed: int,
        system_prefix: str | None = None,
        relation_instructions: dict[str, str] | None = None,
        candidate_instructions: dict[str, list[str]] | None = None,
    ) -> None:
        self.templates = templates
        self.few_shot = few_shot
        self.seed = seed
        self.system_prefix = system_prefix
        self.relation_instructions = relation_instructions or {}
        self.candidate_instructions = candidate_instructions or {}
        self.by_relation: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in train_rows:
            self.by_relation[row["Relation"]].append(row)

    def _examples_for(
        self, subject: str, relation: str, candidate_index: int = 0
    ) -> list[dict[str, Any]]:
        candidates = [
            row
            for row in self.by_relation.get(relation, [])
            if row.get("SubjectEntity") != subject
        ]
        digest = hashlib.sha256(
            f"{self.seed}\0{relation}\0{subject}\0{candidate_index}".encode("utf-8")
        ).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        count = min(self.few_shot, len(candidates))
        return rng.sample(candidates, count)

    def build(
        self, subject: str, relation: str, candidate_index: int = 0
    ) -> list[dict[str, str]]:
        try:
            template = self.templates[relation]
        except KeyError as exc:
            raise ValueError(f"no prompt template for relation {relation}") from exc

        system_message = BASE_SYSTEM_MESSAGE
        if self.system_prefix:
            system_message = f"{self.system_prefix}\n{system_message}"
        relation_instruction = self.relation_instructions.get(relation)
        if relation_instruction:
            system_message = f"{system_message}\n\n{relation_instruction.strip()}"

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_message}
        ]
        for example in self._examples_for(subject, relation, candidate_index):
            messages.extend(
                [
                    {
                        "role": "user",
                        "content": template.format(
                            subject_entity=example["SubjectEntity"]
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            canonical_gold_labels(example.get("ObjectEntities")),
                            ensure_ascii=False,
                        ),
                    },
                ]
            )
        final_question = template.format(subject_entity=subject)
        candidate_instruction = self.candidate_instruction_for(
            relation, candidate_index
        )
        if candidate_instruction:
            final_question = (
                f"{final_question}\n\n"
                f"Candidate-specific scope: {candidate_instruction}"
            )
        messages.append(
            {
                "role": "user",
                "content": final_question,
            }
        )
        return messages

    def candidate_instruction_for(
        self, relation: str, candidate_index: int
    ) -> str | None:
        instructions = self.candidate_instructions.get(relation)
        if instructions is None:
            return None
        if candidate_index < 0 or candidate_index >= len(instructions):
            raise ValueError(
                f"candidate index {candidate_index} is out of range for {relation}"
            )
        return instructions[candidate_index].strip()

    def conversation_followup(
        self, subject: str, relation: str, candidate_index: int
    ) -> dict[str, str]:
        try:
            template = self.templates[relation]
        except KeyError as exc:
            raise ValueError(f"no prompt template for relation {relation}") from exc
        instruction = self.candidate_instruction_for(relation, candidate_index)
        question = template.format(subject_entity=subject)
        return {
            "role": "user",
            "content": (
                "Continue the same backward chronological enumeration. "
                "Use the recipients already discussed as associative memory cues, "
                "but do not repeat them in this answer.\n\n"
                f"{question}\n\nCandidate-specific scope: {instruction}"
            ),
        }

    def area_audit_followup(self, subject: str) -> dict[str, str]:
        return {
            "role": "user",
            "content": (
                "Audit the numeric answer you just gave for the exact subject "
                f"{subject!r}. Do not assume its real unit is square kilometers "
                "merely because the question requested that unit. Identify which "
                "real-world area fact the number most likely represents, then "
                "convert it to square kilometers when the unit is supported.\n\n"
                "Return exactly one JSON array containing these six strings in "
                "this order:\n"
                '["value=number from your previous answer",'
                '"unit=square_kilometer|square_mile|hectare|acre|unknown",'
                '"scope=total_geographic_area|water_surface_area|land_area_only|'
                'administrative_subarea|metropolitan_area|historical_area|unknown",'
                '"reference_year=current|general|YYYY|unknown",'
                '"normalized_value_km2=number or unknown",'
                '"keep=true|false"]\n\n'
                "Set keep=true only when the fact refers to this exact subject and "
                "the ordinary requested scope: total geographic area for a land "
                "entity or water-surface area for a lake. Set keep=false for a "
                "different entity, subarea, land-only value when total was asked, "
                "metropolitan area, historical extent, or unresolved ambiguity. "
                "Do not silently replace the previous value with a different fact."
            ),
        }

    def area_metadata_followup(self, subject: str) -> dict[str, str]:
        return {
            "role": "user",
            "content": (
                "Describe the attributes of the numeric answer you just gave for "
                f"{subject!r}. This is metadata estimation only: do not decide "
                "whether the answer is correct, whether it should be retained, or "
                "whether it satisfies the requested hasArea relation. Do not "
                "replace it with a preferred value. Infer what real-world area "
                "quantity that exact number most likely denotes. Use unknown for "
                "an attribute that cannot be inferred.\n\n"
                "Return exactly one JSON array containing these five strings in "
                "this order:\n"
                '["value=number from your previous answer",'
                '"unit=square_kilometer|square_mile|hectare|acre|unknown",'
                '"scope=total_geographic_area|water_surface_area|land_area_only|'
                'administrative_subarea|metropolitan_area|historical_area|unknown",'
                '"reference_year=current|general|YYYY|unknown",'
                '"normalized_value_km2=number or unknown"]\n\n'
                "The normalized value must only convert the previous number's "
                "estimated unit; do not substitute a different area fact. Do not "
                "include keep, confidence, correct, valid, or any recommendation."
            ),
        }

    def area_metadata_selection(
        self,
        subject: str,
        candidates: list[list[str]],
        diagnostics: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        records = []
        for index, candidate in enumerate(candidates):
            diagnostic = diagnostics[index] if index < len(diagnostics) else {}
            metadata = diagnostic.get("metadata")
            attribute_estimate = (
                {
                    field: metadata.get(field)
                    for field in (
                        "status",
                        "value",
                        "unit",
                        "scope",
                        "reference_year",
                        "normalized_value_km2",
                    )
                }
                if isinstance(metadata, dict)
                else {"status": "missing"}
            )
            records.append(
                {
                    "candidate_id": index + 1,
                    "original_answer": candidate[0] if candidate else None,
                    "attribute_estimate": attribute_estimate,
                }
            )
        return [
            {
                "role": "system",
                "content": (
                    "Select one final numeric answer from supplied candidates. "
                    "Return exactly one JSON array containing one numeric string "
                    "in square kilometers and no explanation."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Determine the requested hasArea value for {subject!r}. The "
                    "target is total geographic area for a land entity, or water-"
                    "surface area for a lake, expressed in square kilometers.\n\n"
                    "Below are 20 independent numeric answers followed by separately "
                    "estimated attributes. Attribute estimates are unverified and "
                    "may themselves be wrong. Compare all records and resolve logical "
                    "inconsistencies: the original number versus the estimated value, "
                    "the unit versus its square-kilometer conversion, requested scope "
                    "versus estimated scope, and current/general area versus historical "
                    "or dated area. Use agreement across independent candidates as "
                    "evidence, but do not blindly follow the numerical majority when "
                    "its attributes are inconsistent. Do not treat any attribute as a "
                    "correctness or retention verdict.\n\n"
                    "Choose a value supported by at least one candidate record, either "
                    "directly in square kilometers or by a logically consistent unit "
                    "conversion. Do not retrieve or invent an unrelated new area value. "
                    "Always return exactly one numeric answer.\n\nCandidate records:\n"
                    + json.dumps(records, ensure_ascii=False, separators=(",", ":"))
                ),
            },
        ]

    def area_cluster_selection(
        self, subject: str, clusters: list[NumericCluster]
    ) -> list[dict[str, str]]:
        return area_cluster_selection_messages(subject, clusters)
