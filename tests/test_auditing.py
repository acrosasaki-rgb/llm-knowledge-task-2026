from akbc_baseline.auditing import parse_area_audit, parse_area_metadata


def test_accepts_supported_unit_conversion_and_expected_scope() -> None:
    result = parse_area_audit(
        '["value=99","unit=square_mile",'
        '"scope=total_geographic_area","reference_year=general",'
        '"normalized_value_km2=256.41","keep=true"]',
        ["99"],
    )

    assert result.metadata["status"] == "accepted"
    assert result.metadata["usable"] is True
    assert result.metadata["normalized_value_km2"] == 256.41


def test_accepts_six_positional_strings_in_documented_order() -> None:
    result = parse_area_audit(
        '["4.4","hectare","total_geographic_area","current","0.044","true"]',
        ["4.4"],
    )

    assert result.metadata["status"] == "accepted"
    assert result.metadata["usable"] is True
    assert result.metadata["unit"] == "hectare"
    assert result.metadata["normalized_value_km2"] == 0.044


def test_rejects_mismatched_value_scope_and_conversion() -> None:
    result = parse_area_audit(
        '["value=90","unit=hectare",'
        '"scope=land_area_only","reference_year=2020",'
        '"normalized_value_km2=90","keep=true"]',
        ["99"],
    )

    assert result.metadata["status"] == "rejected"
    assert result.metadata["usable"] is False
    assert set(result.metadata["rejection_reasons"]) == {
        "value_mismatch",
        "conversion_mismatch",
        "unexpected_scope",
    }


def test_rejects_unparseable_audit() -> None:
    result = parse_area_audit("not json", ["99"])

    assert result.metadata == {"status": "parse_failure", "usable": False}


def test_rejects_unknown_or_invalid_reference_year() -> None:
    unknown = parse_area_audit(
        '["99","square_kilometer","total_geographic_area","unknown","99","true"]',
        ["99"],
    )
    invalid = parse_area_audit(
        '["99","square_kilometer","total_geographic_area","historical_area","99","true"]',
        ["99"],
    )

    assert unknown.metadata["rejection_reasons"] == ["unknown_reference_year"]
    assert invalid.metadata["rejection_reasons"] == ["unsupported_reference_year"]


def test_area_metadata_describes_attributes_without_acceptance_decision() -> None:
    result = parse_area_metadata(
        '["99","square_mile","land_area_only","1999","256.41"]',
        ["99"],
    )

    assert result.metadata == {
        "status": "parsed",
        "value": 99.0,
        "unit": "square_mile",
        "scope": "land_area_only",
        "reference_year": "1999",
        "normalized_value_km2": 256.41,
        "value_matches_candidate": True,
        "conversion_consistent": True,
    }
    assert "usable" not in result.metadata
    assert "keep" not in result.metadata


def test_area_metadata_preserves_logical_inconsistency_for_final_judge() -> None:
    result = parse_area_metadata(
        '["value=90","unit=hectare","scope=historical_area",'
        '"reference_year=unknown","normalized_value_km2=90"]',
        ["99"],
    )

    assert result.metadata["status"] == "parsed"
    assert result.metadata["value_matches_candidate"] is False
    assert result.metadata["conversion_consistent"] is False
    assert result.metadata["scope"] == "historical_area"
    assert "rejection_reasons" not in result.metadata
