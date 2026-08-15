from akbc_baseline.area_audit_report import summarize


def test_summarizes_area_audit_acceptance_and_rejections() -> None:
    rows = [
        {
            "CandidateDiagnostics": [
                {
                    "audit": {
                        "status": "accepted",
                        "usable": True,
                        "unit": "square_kilometer",
                        "scope": "total_geographic_area",
                        "rejection_reasons": [],
                    },
                    "audit_generation_diagnostics": {"hit_token_limit": False},
                },
                {
                    "audit": {
                        "status": "rejected",
                        "usable": False,
                        "unit": "unknown",
                        "scope": "unknown",
                        "rejection_reasons": ["unsupported_unit", "unexpected_scope"],
                    },
                    "audit_generation_diagnostics": {"hit_token_limit": True},
                },
            ]
        },
        {"CandidateDiagnostics": [{"audit": {"status": "parse_failure", "usable": False}}]},
    ]

    report = summarize(rows)

    assert report["rows"] == 2
    assert report["total_audits"] == 3
    assert report["accepted_audits"] == 1
    assert report["rows_with_no_accepted_audit"] == 1
    assert report["status_counts"] == {
        "accepted": 1,
        "parse_failure": 1,
        "rejected": 1,
    }
    assert report["rejection_reason_counts"] == {
        "unexpected_scope": 1,
        "unsupported_unit": 1,
    }
    assert report["audit_token_limit_candidates"] == 1
