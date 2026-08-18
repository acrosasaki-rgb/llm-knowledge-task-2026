from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v19_registers_and_seed_namespace_are_fixed():
    inference = load_module("v19_infer", "scripts/h100-bf16/v19_infer.py")
    assert inference.CANDIDATE_COUNT == 20
    assert inference.TEMPERATURE == 0.6
    assert inference.TOP_P == 0.95
    assert "Andros, Greece" in inference.REGISTERS["hasArea"]
    assert "HoHoKam Stadium" in inference.REGISTERS["hasCapacity"]
    assert "About:" in inference.REGISTERS["personHasCityOfDeath"]
    assert inference.candidate_seed("42", "Ada Lovelace", "personHasCityOfDeath", 0) == inference.candidate_seed("42", "Ada Lovelace", "occ", 0)


def test_city_threshold_is_v19_configurable():
    compose = load_module("compose_gemma", "scripts/h100-bf16/compose_gemma.py")
    candidates = [["Rome"]] * 13 + [[]] * 7
    assert compose.surface_city(candidates, lambda value: value, 12) == ["Rome"]
    assert compose.surface_city(candidates, lambda value: value, 14) == []
    assert compose.surface_city([["Rome"]] * 14 + [[]] * 6, lambda value: value, 14) == ["Rome"]


def test_v19_verifier_rejects_candidate_count_and_order():
    verifier = load_module("verify_v19", "scripts/h100-bf16/verify_v19.py")
    source = [{"SubjectEntity": "A", "Relation": "hasArea"}]
    candidate = [{"SubjectEntity": "A", "Relation": "hasArea", "Candidates": [[]] * 20}]
    prediction = [{"SubjectEntity": "A", "Relation": "hasArea", "ObjectEntities": []}]
    verifier.verify(source, candidate, prediction)
    candidate[0]["Candidates"] = [[]] * 19
    with pytest.raises(ValueError, match="20 candidates"):
        verifier.verify(source, candidate, prediction)


def test_v19_input_migration_renames_15_and_drops_two():
    prepare = load_module("prepare_v19_input", "scripts/h100-bf16/prepare_v19_input.py")
    rows = [
        {"SubjectEntity": subject, "Relation": relation, "ObjectEntities": []}
        for relation, subject in prepare.DROPPED
    ]
    rows.extend(
        {"SubjectEntity": old, "Relation": "hasArea", "ObjectEntities": []}
        for old in prepare.RENAMED
    )
    rows.extend(
        {"SubjectEntity": f"unchanged-{index}", "Relation": "hasArea", "ObjectEntities": []}
        for index in range(460)
    )
    migrated = prepare.prepare(rows)
    assert len(rows) == 477
    assert len(migrated) == 475
    names = {row["SubjectEntity"] for row in migrated}
    assert set(prepare.RENAMED.values()) <= names
    assert not set(prepare.RENAMED) & names
    assert not {subject for _, subject in prepare.DROPPED} & names


def test_v19_launcher_pins_model_and_gpu_runtime():
    host = (ROOT / "scripts/h100-bf16/run-v19-docker.sh").read_text(encoding="utf-8")
    container = (ROOT / "scripts/h100-bf16/run-v19-container.sh").read_text(encoding="utf-8")
    assert "--gpus all" in host
    assert "AKBC_DATA_FILE=/workspace/dataset/data/${split}.jsonl" in host
    assert "30d8cfaaa7af5b236054cfb361f57b7d0c1e6e57" in host
    assert "bluerain123/gemma-3-27b-pt-Q8_0-GGUF" in container
    assert "71ed905c894b1d481e67a3bdbdfe06dd5805c6e9" in container
    assert "--city-min-votes 14" in container
    assert "--strict" in container
