import json
from pathlib import Path

import pytest

from akbc_baseline.slice_jsonl import main


def _write_rows(path: Path, count: int) -> None:
    path.write_text(
        "".join(json.dumps({"index": index}) + "\n" for index in range(count)),
        encoding="utf-8",
    )


def test_writes_exact_ordered_slice(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "slice.jsonl"
    _write_rows(source, 8)

    assert main(
        [
            "--input",
            str(source),
            "--output",
            str(output),
            "--offset",
            "2",
            "--limit",
            "3",
        ]
    ) == 0

    rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert rows == [{"index": 2}, {"index": 3}, {"index": 4}]


def test_rejects_incomplete_slice(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "slice.jsonl"
    _write_rows(source, 2)

    with pytest.raises(ValueError, match="expected 3 rows"):
        main(
            [
                "--input",
                str(source),
                "--output",
                str(output),
                "--limit",
                "3",
            ]
        )
