from akbc_baseline.area_metadata_report import summarize


def test_summarizes_metadata_and_final_selection_diagnostics() -> None:
    rows = [
        {
            "CandidateDiagnostics": [
                {
                    "metadata": {
                        "status": "parsed",
                        "unit": "square_mile",
                        "scope": "land_area_only",
                        "reference_year": "1999",
                        "value_matches_candidate": True,
                        "conversion_consistent": False,
                    },
                    "metadata_generation_diagnostics": {"hit_token_limit": True},
                },
                {
                    "metadata": {"status": "parse_failure"},
                    "metadata_generation_diagnostics": {"hit_token_limit": False},
                },
            ],
            "FinalSelection": {
                "parse_status": "json",
                "used_fallback": False,
                "generation_diagnostics": {"hit_token_limit": False},
            },
        },
        {
            "CandidateDiagnostics": [],
            "FinalSelection": {
                "parse_status": "parse_failure",
                "used_fallback": True,
                "generation_diagnostics": {"hit_token_limit": True},
            },
        },
    ]

    report = summarize(rows)

    assert report["rows"] == 2
    assert report["total_metadata"] == 2
    assert report["metadata_status_counts"] == {
        "parse_failure": 1,
        "parsed": 1,
    }
    assert report["logical_consistency_counts"] == {
        "conversion_inconsistent": 1,
        "conversion_unknown": 1,
        "value_match": 1,
        "value_mismatch_or_unknown": 1,
    }
    assert report["metadata_token_limit_candidates"] == 1
    assert report["final_selection_status_counts"] == {
        "json": 1,
        "parse_failure": 1,
    }
    assert report["final_selection_fallbacks"] == 1
    assert report["final_selection_token_limits"] == 1
