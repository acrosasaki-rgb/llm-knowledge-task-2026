from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

from .aggregation import aggregate_candidates
from .area_clustering import (
    cluster_choice_records,
    cluster_numeric_candidates,
    format_number,
    match_cluster_choice,
)
from .auditing import parse_area_audit, parse_area_metadata
from .backends import create_backend
from .config import ModelConfig
from .data import append_jsonl, read_jsonl, read_prompt_templates, write_jsonl
from .parsing import parse_object_entities_with_status
from .prompting import PromptBuilder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an AKBC baseline model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidates-output")
    parser.add_argument("--metrics-output")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help=(
            "number of candidate requests issued concurrently; requires a "
            "llama.cpp server started with a matching --parallel slot count"
        ),
    )
    return parser


def _candidate_seed(
    base_seed: int, subject: str, relation: str, candidate_index: int
) -> int:
    digest = hashlib.sha256(
        f"{base_seed}\0{relation}\0{subject}\0{candidate_index}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big")


def _is_single_numeric_answer(values: list[str]) -> bool:
    return len(values) == 1 and re.fullmatch(
        r"[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?", values[0].strip()
    ) is not None


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ModelConfig.from_yaml(args.config)
    dataset_dir = Path(args.dataset_dir).resolve()
    train_rows = read_jsonl(dataset_dir / config.train_data_file)
    templates = read_prompt_templates(dataset_dir / config.prompt_templates_file)
    input_rows = read_jsonl(args.input)
    if args.offset < 0:
        raise ValueError("--offset must be zero or positive")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        input_rows = input_rows[args.offset : args.offset + args.limit]
    else:
        input_rows = input_rows[args.offset :]

    prompt_builder = PromptBuilder(
        train_rows=train_rows,
        templates=templates,
        few_shot=config.few_shot,
        seed=config.seed,
        system_prefix=config.system_prefix,
        relation_instructions=config.relation_instructions,
        candidate_instructions=config.candidate_instructions,
    )
    if config.num_candidates > 1 and not args.candidates_output:
        raise ValueError("--candidates-output is required when num_candidates > 1")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be positive")
    if args.concurrency > 1:
        # Conversation chains and per-candidate audits are inherently
        # sequential, so concurrent generation refuses those configs instead
        # of silently changing their semantics.
        if config.conversation_chains:
            raise ValueError(
                "--concurrency does not support conversation_chains configs"
            )
        if config.candidate_audits:
            raise ValueError(
                "--concurrency does not support candidate_audits configs"
            )

    candidates_path = Path(args.candidates_output) if args.candidates_output else None
    completed_candidates: dict[tuple[str, str], list[list[str]]] = {}
    completed_diagnostics: dict[tuple[str, str], list[dict[str, object]]] = {}
    completed_final_selections: dict[tuple[str, str], dict[str, object]] = {}
    if candidates_path is not None:
        if args.resume and candidates_path.exists():
            for saved in read_jsonl(candidates_path):
                key = (saved["SubjectEntity"], saved["Relation"])
                raw_candidates = saved.get("Candidates")
                if isinstance(raw_candidates, list):
                    completed_candidates[key] = raw_candidates
                raw_diagnostics = saved.get("CandidateDiagnostics")
                if isinstance(raw_diagnostics, list):
                    completed_diagnostics[key] = raw_diagnostics
                raw_final_selection = saved.get("FinalSelection")
                if isinstance(raw_final_selection, dict):
                    completed_final_selections[key] = raw_final_selection
        else:
            write_jsonl(candidates_path, [])

    backend = create_backend(config)
    if args.concurrency > 1 and not hasattr(backend, "generate_with_diagnostics"):
        raise ValueError(
            "--concurrency requires the llama_cpp_server backend"
        )
    backend.reset_peak_memory_stats()
    predictions = []
    started = time.monotonic()

    for index, row in enumerate(input_rows, start=1):
        subject = row["SubjectEntity"]
        relation = row["Relation"]
        key = (subject, relation)
        candidates = completed_candidates.get(key)
        candidate_diagnostics = completed_diagnostics.get(key, [])
        generated_candidates = (
            candidates is None or len(candidates) != config.num_candidates
        )
        if generated_candidates and args.concurrency > 1:
            relation_enable_thinking = config.relation_thinking.get(relation)
            effective_thinking = (
                config.enable_thinking
                if relation_enable_thinking is None
                else relation_enable_thinking
            )

            def _generate_candidate(
                candidate_index: int,
            ) -> tuple[list[str], dict[str, object]]:
                candidate_messages = prompt_builder.build(
                    subject, relation, candidate_index=candidate_index
                )
                generation_kwargs: dict[str, object] = {}
                if config.candidate_temperatures is not None:
                    generation_kwargs["temperature"] = (
                        config.candidate_temperatures[candidate_index]
                    )
                text, diagnostics = backend.generate_with_diagnostics(
                    candidate_messages,
                    seed=_candidate_seed(
                        config.seed, subject, relation, candidate_index
                    ),
                    enable_thinking=relation_enable_thinking,
                    **generation_kwargs,
                )
                parsed = parse_object_entities_with_status(text, relation)
                return parsed.object_entities, {
                    **diagnostics,
                    "candidate_temperature": generation_kwargs.get("temperature"),
                    "parse_status": parsed.status,
                    "enable_thinking": effective_thinking,
                    "candidate_instruction": (
                        prompt_builder.candidate_instruction_for(
                            relation, candidate_index
                        )
                    ),
                    "conversation_chain": None,
                    "conversation_turn": None,
                }

            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                results = list(
                    pool.map(_generate_candidate, range(config.num_candidates))
                )
            candidates = [values for values, _ in results]
            candidate_diagnostics = [
                diagnostics for _, diagnostics in results
            ]
        elif generated_candidates:
            candidates = []
            candidate_diagnostics = []
            conversation_chains = config.conversation_chains.get(relation)
            turns_per_chain = (
                config.num_candidates // conversation_chains
                if conversation_chains is not None
                else config.num_candidates
            )
            chain_count = conversation_chains or 1
            for chain_index in range(chain_count):
                messages: list[dict[str, str]] | None = None
                previous_final_text: str | None = None
                for turn_index in range(turns_per_chain):
                    candidate_index = chain_index * turns_per_chain + turn_index
                    if messages is None or conversation_chains is None:
                        messages = prompt_builder.build(
                            subject, relation, candidate_index=candidate_index
                        )
                    else:
                        if previous_final_text is None:
                            raise RuntimeError("conversation has no previous answer")
                        messages.extend(
                            [
                                {
                                    "role": "assistant",
                                    "content": previous_final_text,
                                },
                                prompt_builder.conversation_followup(
                                    subject,
                                    relation,
                                    candidate_index,
                                ),
                            ]
                        )
                    relation_enable_thinking = config.relation_thinking.get(relation)
                    generation_kwargs = {}
                    if config.candidate_temperatures is not None:
                        generation_kwargs["temperature"] = (
                            config.candidate_temperatures[candidate_index]
                        )
                    final_text = backend.generate(
                        messages,
                        seed=_candidate_seed(
                            config.seed, subject, relation, candidate_index
                        ),
                        enable_thinking=relation_enable_thinking,
                        **generation_kwargs,
                    )
                    previous_final_text = final_text
                    parsed = parse_object_entities_with_status(final_text, relation)
                    candidates.append(parsed.object_entities)
                    diagnostics_reader = getattr(
                        backend, "last_generation_diagnostics", None
                    )
                    diagnostics = diagnostics_reader() if diagnostics_reader else {}
                    secondary_fields: dict[str, object] = {}
                    audit_config = config.candidate_audits.get(relation)
                    if audit_config is not None:
                        audit_kind = audit_config["kind"]
                        followup = (
                            prompt_builder.area_metadata_followup(subject)
                            if audit_kind == "area_metadata_only"
                            else prompt_builder.area_audit_followup(subject)
                        )
                        audit_messages = [
                            *messages,
                            {"role": "assistant", "content": final_text},
                            followup,
                        ]
                        audit_text = backend.generate(
                            audit_messages,
                            seed=_candidate_seed(
                                config.seed,
                                subject,
                                relation,
                                config.num_candidates + candidate_index,
                            ),
                            enable_thinking=relation_enable_thinking,
                        )
                        audit_diagnostics = (
                            diagnostics_reader() if diagnostics_reader else {}
                        )
                        if audit_kind == "area_metadata_only":
                            secondary_fields = {
                                "metadata": parse_area_metadata(
                                    audit_text, parsed.object_entities
                                ).metadata,
                                "metadata_text": audit_text,
                                "metadata_generation_diagnostics": audit_diagnostics,
                            }
                        else:
                            secondary_fields = {
                                "audit": parse_area_audit(
                                    audit_text, parsed.object_entities
                                ).metadata,
                                "audit_text": audit_text,
                                "audit_generation_diagnostics": audit_diagnostics,
                            }
                    effective_thinking = (
                        config.enable_thinking
                        if relation_enable_thinking is None
                        else relation_enable_thinking
                    )
                    candidate_diagnostics.append(
                        {
                            **diagnostics,
                            "candidate_temperature": generation_kwargs.get(
                                "temperature"
                            ),
                            "parse_status": parsed.status,
                            "enable_thinking": effective_thinking,
                            "candidate_instruction": (
                                prompt_builder.candidate_instruction_for(
                                    relation, candidate_index
                                )
                            ),
                            "conversation_chain": (
                                chain_index if conversation_chains is not None else None
                            ),
                            "conversation_turn": (
                                turn_index if conversation_chains is not None else None
                            ),
                            **secondary_fields,
                        }
                    )
        aggregation_policy = config.aggregation.get(relation)
        aggregation_strategy = (
            aggregation_policy.get("strategy")
            if aggregation_policy is not None
            else None
        )
        final_selection: dict[str, object] | None = None
        if (
            aggregation_strategy in {"cluster_choice", "metadata_judge"}
        ):
            saved_selection = completed_final_selections.get(key)
            saved_entities = (
                saved_selection.get("ObjectEntities")
                if isinstance(saved_selection, dict)
                else None
            )
            saved_strategy = (
                saved_selection.get("strategy")
                if isinstance(saved_selection, dict)
                else None
            )
            reusable_strategy = saved_strategy == aggregation_strategy or (
                aggregation_strategy == "metadata_judge"
                and saved_strategy is None
            )
            if isinstance(saved_entities, list) and _is_single_numeric_answer(
                saved_entities
            ) and reusable_strategy:
                object_entities = saved_entities
                final_selection = saved_selection
            elif aggregation_strategy == "cluster_choice":
                tolerance = float(aggregation_policy.get("tolerance", 0.05))
                clusters = cluster_numeric_candidates(candidates, tolerance)
                selection_text: str | None = None
                selection_diagnostics: dict[str, object] = {}
                selected_status = "skipped"
                used_fallback = False
                skipped_single_cluster = len(clusters) <= 1
                if len(clusters) > 1:
                    selection_text = backend.generate(
                        prompt_builder.area_cluster_selection(subject, clusters),
                        seed=_candidate_seed(
                            config.seed,
                            subject,
                            relation,
                            config.num_candidates * 2,
                        ),
                        enable_thinking=config.relation_thinking.get(relation),
                    )
                    diagnostics_reader = getattr(
                        backend, "last_generation_diagnostics", None
                    )
                    selection_diagnostics = (
                        diagnostics_reader() if diagnostics_reader else {}
                    )
                    selected = parse_object_entities_with_status(
                        selection_text, relation
                    )
                    selected_status = selected.status
                    chosen_cluster = match_cluster_choice(
                        selected.object_entities, clusters
                    )
                    used_fallback = chosen_cluster is None
                    if chosen_cluster is None:
                        chosen_cluster = clusters[0]
                    object_entities = [
                        format_number(chosen_cluster.representative)
                    ]
                elif clusters:
                    object_entities = [
                        format_number(clusters[0].representative)
                    ]
                else:
                    object_entities = aggregate_candidates(
                        candidates,
                        {"strategy": "median"},
                        diagnostics=candidate_diagnostics,
                    )
                    used_fallback = True
                final_selection = {
                    "strategy": "cluster_choice",
                    "ObjectEntities": object_entities,
                    "text": selection_text,
                    "parse_status": selected_status,
                    "used_fallback": used_fallback,
                    "skipped_single_cluster": skipped_single_cluster,
                    "tolerance": tolerance,
                    "choices": cluster_choice_records(clusters),
                    "generation_diagnostics": selection_diagnostics,
                }
            else:
                selection_text = backend.generate(
                    prompt_builder.area_metadata_selection(
                        subject, candidates, candidate_diagnostics
                    ),
                    seed=_candidate_seed(
                        config.seed,
                        subject,
                        relation,
                        config.num_candidates * 2,
                    ),
                    enable_thinking=config.relation_thinking.get(relation),
                )
                diagnostics_reader = getattr(
                    backend, "last_generation_diagnostics", None
                )
                selection_diagnostics = (
                    diagnostics_reader() if diagnostics_reader else {}
                )
                selected = parse_object_entities_with_status(
                    selection_text, relation
                )
                used_fallback = not _is_single_numeric_answer(
                    selected.object_entities
                )
                object_entities = (
                    aggregate_candidates(
                        candidates,
                        {"strategy": "median"},
                        diagnostics=candidate_diagnostics,
                    )
                    if used_fallback
                    else selected.object_entities
                )
                final_selection = {
                    "strategy": "metadata_judge",
                    "ObjectEntities": object_entities,
                    "text": selection_text,
                    "parse_status": selected.status,
                    "used_fallback": used_fallback,
                    "generation_diagnostics": selection_diagnostics,
                }
        else:
            object_entities = aggregate_candidates(
                candidates,
                aggregation_policy,
                diagnostics=candidate_diagnostics,
            )
        if generated_candidates and candidates_path is not None:
            candidate_row = {
                "SubjectEntity": subject,
                "Relation": relation,
                "Candidates": candidates,
                "CandidateDiagnostics": candidate_diagnostics,
            }
            if final_selection is not None:
                candidate_row["FinalSelection"] = final_selection
            append_jsonl(candidates_path, candidate_row)
        predictions.append(
            {
                "SubjectEntity": subject,
                "Relation": relation,
                "ObjectEntities": object_entities,
            }
        )
        elapsed = time.monotonic() - started
        print(
            f"{config.name}: completed {index}/{len(input_rows)} "
            f"rows (dataset offset {args.offset + index - 1}) "
            f"in {elapsed:.1f}s",
            flush=True,
        )

    write_jsonl(args.output, predictions)
    elapsed = time.monotonic() - started
    if args.metrics_output:
        metrics_path = Path(args.metrics_output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics = {
            "name": config.name,
            "model_id": config.model_id,
            "parameter_count_billion": config.parameter_count_billion,
            "num_candidates": config.num_candidates,
            "input_offset": args.offset,
            "rows": len(predictions),
            "empty_predictions": sum(
                not prediction["ObjectEntities"] for prediction in predictions
            ),
            "elapsed_seconds": elapsed,
            "peak_cuda_memory_gib": backend.peak_cuda_memory_gib(),
        }
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"wrote {len(predictions)} predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
