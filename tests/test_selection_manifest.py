import hashlib
import json
from pathlib import Path

import pytest

from akbc_baseline.selection_manifest import main
from akbc_baseline.selection_verify import verify_manifest


CONTAINER_IMAGE = (
    "ghcr.io/ggml-org/llama.cpp:full-cuda@sha256:"
    "11b0e950e081777cf326598bb2eff2ab0815f02405bf95c6650b34027750114e"
)


def test_writes_pinned_selection_manifest(tmp_path: Path) -> None:
    config = tmp_path / "model.yaml"
    config.write_text(
        "name: selected\n"
        "model_id: example/model\n"
        "backend: causal\n"
        "prompt_templates_file: prompts.csv\n"
        "train_data_file: train.jsonl\n"
        "num_candidates: 20\n",
        encoding="utf-8",
    )
    output = tmp_path / "selection.json"

    assert main(
        [
            "--model-key",
            "selected",
            "--config",
            str(config),
            "--dataset-ref",
            "dataset-sha",
            "--commit-sha",
            "commit-sha",
            "--pipeline-url",
            "https://gitlab.example/pipelines/1",
            "--container-image",
            CONTAINER_IMAGE,
            "--output",
            str(output),
        ]
    ) == 0

    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["model_key"] == "selected"
    assert len(manifest["config_sha256"]) == 64
    assert manifest["dataset_ref"] == "dataset-sha"
    assert manifest["schema_version"] == 2
    assert manifest["num_candidates"] == 20
    assert manifest["execution_environment"] == "external-ssh-docker-gpu"
    assert manifest["container_image"] == CONTAINER_IMAGE

    verify_manifest(
        manifest,
        config_path=config,
        model_key="selected",
        dataset_ref="dataset-sha",
        commit_sha="commit-sha",
        container_image=CONTAINER_IMAGE,
        expected_candidates=20,
    )


def test_rejects_modified_config_after_selection(tmp_path: Path) -> None:
    config = tmp_path / "model.yaml"
    config.write_text(
        "name: selected\n"
        "model_id: example/model\n"
        "backend: causal\n"
        "prompt_templates_file: prompts.csv\n"
        "train_data_file: train.jsonl\n"
        "num_candidates: 5\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 2,
        "model_key": "selected",
        "config": config.as_posix(),
        "config_sha256": "0" * 64,
        "num_candidates": 5,
        "dataset_ref": "dataset-sha",
        "commit_sha": "commit-sha",
        "pipeline_url": "https://gitlab.example/pipelines/1",
        "execution_environment": "external-ssh-docker-gpu",
        "container_image": CONTAINER_IMAGE,
        "selection_basis": "manual review of complete validation artifacts",
    }

    with pytest.raises(ValueError, match="config_sha256"):
        verify_manifest(
            manifest,
            config_path=config,
            model_key="selected",
            dataset_ref="dataset-sha",
            commit_sha="commit-sha",
            container_image=CONTAINER_IMAGE,
            expected_candidates=5,
        )


def test_rejects_candidate_count_mismatch_before_external_run(
    tmp_path: Path,
) -> None:
    config = tmp_path / "model.yaml"
    config.write_text(
        "name: selected\n"
        "model_id: example/model\n"
        "backend: causal\n"
        "prompt_templates_file: prompts.csv\n"
        "train_data_file: train.jsonl\n"
        "num_candidates: 5\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 2,
        "model_key": "selected",
        "config": config.as_posix(),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "num_candidates": 5,
        "dataset_ref": "dataset-sha",
        "commit_sha": "commit-sha",
        "pipeline_url": "https://gitlab.example/pipelines/1",
        "execution_environment": "external-ssh-docker-gpu",
        "container_image": CONTAINER_IMAGE,
        "selection_basis": "manual review of complete validation artifacts",
    }

    with pytest.raises(ValueError, match="expected 20 candidates"):
        verify_manifest(
            manifest,
            config_path=config,
            model_key="selected",
            dataset_ref="dataset-sha",
            commit_sha="commit-sha",
            container_image=CONTAINER_IMAGE,
            expected_candidates=20,
        )
