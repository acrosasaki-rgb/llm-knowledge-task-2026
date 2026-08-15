from akbc_baseline.compare import category_scores, empty_confusion, render_markdown


def test_renders_comparison_markdown() -> None:
    metrics = {
        "macro": {
            "relation": {"macro-p": 0.5, "macro-r": 0.4, "macro-f1": 0.45},
            "*** All Relations ***": {
                "macro-p": 0.5,
                "macro-r": 0.4,
                "macro-f1": 0.45,
            },
        },
        "micro": {"*** All Relations ***": {"micro-f1": 0.35}},
        "statistics": {
            "*** All Relations ***": {"avg. #preds": 1.2, "#empty preds": 3}
        },
        "category_scores": [
            {
                "Relation": "relation",
                "Macro Average Precision": 0.5,
                "Macro Average Recall": 0.4,
                "Macro Average F1-score": 0.45,
            },
            {
                "Relation": "All Relations",
                "Macro Average Precision": 0.5,
                "Macro Average Recall": 0.4,
                "Macro Average F1-score": 0.45,
            },
            {
                "Relation": "Zero-object cases*",
                "Macro Average Precision": 1.0,
                "Macro Average Recall": 1.0,
                "Macro Average F1-score": 1.0,
            },
        ],
    }
    markdown = render_markdown({"model-a": metrics, "model-b": metrics})
    assert "| model-a | 0.500 | 0.400 | 0.450 |" in markdown
    assert "| relation | 0.450 | 0.450 |" in markdown
    assert "## Per-relation scores: model-a" in markdown
    assert "| Zero-object cases* | 1.0000 | 1.0000 | 1.0000 |" in markdown


def test_category_scores_match_per_relation_report_layout() -> None:
    macro = {
        "awardWonBy": {"macro-p": 0.25, "macro-r": 0.5, "macro-f1": 0.33},
        "*** All Relations ***": {
            "macro-p": 0.25,
            "macro-r": 0.5,
            "macro-f1": 0.33,
        },
    }
    gt_rows = [
        {"SubjectEntity": "A", "Relation": "awardWonBy", "ObjectEntities": []},
        {
            "SubjectEntity": "B",
            "Relation": "awardWonBy",
            "ObjectEntities": [["Prize"]],
        },
    ]
    pred_rows = [
        {"SubjectEntity": "A", "Relation": "awardWonBy", "ObjectEntities": []},
        {"SubjectEntity": "B", "Relation": "awardWonBy", "ObjectEntities": []},
    ]

    rows = category_scores(macro, gt_rows, pred_rows)

    assert rows[0]["Relation"] == "awardWonBy"
    assert rows[-2]["Relation"] == "All Relations"
    assert rows[-1]["Relation"] == "Zero-object cases*"
    assert rows[-1]["Macro Average Precision"] == 0.5
    assert rows[-1]["Macro Average Recall"] == 1.0
    assert empty_confusion(gt_rows, pred_rows) == {
        "true_empty": 1,
        "false_empty": 1,
        "missed_empty": 0,
        "true_non_empty": 0,
    }
