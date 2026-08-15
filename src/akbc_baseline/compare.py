from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

from .data import read_jsonl


RELATION_ORDER = [
    "awardWonBy",
    "companyTradesAtStockExchange",
    "countryLandBordersCountry",
    "hasArea",
    "hasCapacity",
    "personHasCityOfDeath",
]
ALL_RELATIONS_KEY = "*** All Relations ***"


def _load_evaluator(path: str | Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("akbc_official_evaluator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load evaluator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate_prediction(
    evaluator: ModuleType, ground_truth: str | Path, prediction: str | Path
) -> dict[str, Any]:
    # The official script omits an encoding in its file reader, which fails on
    # non-UTF-8 Windows locales. Keep its scoring logic but read JSONL as UTF-8.
    gt_rows = read_jsonl(ground_truth)
    pred_rows = read_jsonl(prediction)
    scores = evaluator.evaluate_per_sr_pair(
        pred_rows, gt_rows, evaluator.RELATION_TYPE, tolerance=0.05
    )
    macro = evaluator.macro_average_per_relation(scores)
    return {
        "macro": macro,
        "micro": evaluator.micro_average_per_relation(scores),
        "statistics": evaluator.prediction_statistics(scores),
        "category_scores": category_scores(macro, gt_rows, pred_rows),
        "empty_confusion": empty_confusion(gt_rows, pred_rows),
        "non_empty_entity": non_empty_entity_scores(scores),
    }


def _is_empty_gold(row: dict[str, Any]) -> bool:
    values = row.get("ObjectEntities")
    return not isinstance(values, list) or len(values) == 0


def _is_empty_prediction(row: dict[str, Any] | None) -> bool:
    if row is None:
        return True
    values = row.get("ObjectEntities")
    return not isinstance(values, list) or len(values) == 0


def _f1(precision: float, recall: float) -> float:
    return (2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def empty_confusion(
    gt_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]]
) -> dict[str, int]:
    pred_by_key = {
        (row.get("SubjectEntity"), row.get("Relation")): row for row in pred_rows
    }
    counts = {
        "true_empty": 0,
        "false_empty": 0,
        "missed_empty": 0,
        "true_non_empty": 0,
    }
    for row in gt_rows:
        predicted_empty = _is_empty_prediction(
            pred_by_key.get((row.get("SubjectEntity"), row.get("Relation")))
        )
        gold_empty = _is_empty_gold(row)
        if predicted_empty and gold_empty:
            counts["true_empty"] += 1
        elif predicted_empty:
            counts["false_empty"] += 1
        elif gold_empty:
            counts["missed_empty"] += 1
        else:
            counts["true_non_empty"] += 1
    return counts


def zero_object_scores(
    gt_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]]
) -> dict[str, float]:
    counts = empty_confusion(gt_rows, pred_rows)
    predicted_empty = counts["true_empty"] + counts["false_empty"]
    actual_empty = counts["true_empty"] + counts["missed_empty"]
    precision = counts["true_empty"] / predicted_empty if predicted_empty else 1.0
    recall = counts["true_empty"] / actual_empty if actual_empty else 1.0
    return {
        "macro-p": precision,
        "macro-r": recall,
        "macro-f1": _f1(precision, recall),
    }


def category_scores(
    macro: dict[str, dict[str, float]],
    gt_rows: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for relation in RELATION_ORDER:
        values = macro.get(relation)
        if values is None:
            continue
        rows.append(
            {
                "Relation": relation,
                "Macro Average Precision": values["macro-p"],
                "Macro Average Recall": values["macro-r"],
                "Macro Average F1-score": values["macro-f1"],
            }
        )
    all_values = macro[ALL_RELATIONS_KEY]
    rows.append(
        {
            "Relation": "All Relations",
            "Macro Average Precision": all_values["macro-p"],
            "Macro Average Recall": all_values["macro-r"],
            "Macro Average F1-score": all_values["macro-f1"],
        }
    )
    zero_values = zero_object_scores(gt_rows, pred_rows)
    rows.append(
        {
            "Relation": "Zero-object cases*",
            "Macro Average Precision": zero_values["macro-p"],
            "Macro Average Recall": zero_values["macro-r"],
            "Macro Average F1-score": zero_values["macro-f1"],
        }
    )
    return rows


def non_empty_entity_scores(scores: list[dict[str, Any]]) -> dict[str, float]:
    non_empty = [score for score in scores if score.get("total_gt", 0) > 0]
    if not non_empty:
        return {"macro-p": 1.0, "macro-r": 1.0, "macro-f1": 1.0}
    return {
        "macro-p": sum(score["p"] for score in non_empty) / len(non_empty),
        "macro-r": sum(score["r"] for score in non_empty) / len(non_empty),
        "macro-f1": sum(score["f1"] for score in non_empty) / len(non_empty),
    }


def _parse_prediction_argument(value: str) -> tuple[str, str]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("prediction must be NAME=PATH")
    return name, path


def render_markdown(results: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# AKBC validation comparison",
        "",
        "| Model | Macro P | Macro R | Macro F1 | Micro F1 | Avg predictions | Empty predictions |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    all_key = ALL_RELATIONS_KEY
    for name, metrics in results.items():
        macro = metrics["macro"][all_key]
        micro = metrics["micro"][all_key]
        stats = metrics["statistics"][all_key]
        lines.append(
            f"| {name} | {macro['macro-p']:.3f} | {macro['macro-r']:.3f} | "
            f"{macro['macro-f1']:.3f} | {micro['micro-f1']:.3f} | "
            f"{stats['avg. #preds']:.3f} | {stats['#empty preds']} |"
        )

    relations = sorted(
        key for key in next(iter(results.values()))["macro"] if key != all_key
    )
    lines.extend(
        [
            "",
            "## Macro F1 by relation",
            "",
            "| Relation | " + " | ".join(results) + " |",
            "|---|" + "---:|" * len(results),
        ]
    )
    for relation in relations:
        values = [
            f"{metrics['macro'][relation]['macro-f1']:.3f}"
            for metrics in results.values()
        ]
        lines.append(f"| {relation} | " + " | ".join(values) + " |")
    for name, metrics in results.items():
        lines.extend(
            [
                "",
                f"## Per-relation scores: {name}",
                "",
                "| Relation | Macro Average Precision | Macro Average Recall | Macro Average F1-score |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in metrics["category_scores"]:
            lines.append(
                f"| {row['Relation']} | "
                f"{row['Macro Average Precision']:.4f} | "
                f"{row['Macro Average Recall']:.4f} | "
                f"{row['Macro Average F1-score']:.4f} |"
            )
        lines.extend(
            [
                "",
                "*: 'Zero-object cases' are for reference only and will not be included in the final competition results.",
            ]
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare AKBC predictions")
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument(
        "--prediction", action="append", required=True, type=_parse_prediction_argument
    )
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--markdown-output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evaluator = _load_evaluator(args.evaluator)
    results = {
        name: evaluate_prediction(evaluator, args.ground_truth, path)
        for name, path in args.prediction
    }

    json_path = Path(args.json_output)
    markdown_path = Path(args.markdown_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown = render_markdown(results)
    markdown_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
