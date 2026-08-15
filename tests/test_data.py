import json
from pathlib import Path

from akbc_baseline.data import read_jsonl


def test_read_jsonl_uses_utf8_on_non_utf8_default_locales(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    expected = {"SubjectEntity": "Lošinj", "Relation": "hasArea"}
    path.write_text(json.dumps(expected, ensure_ascii=False) + "\n", encoding="utf-8")
    assert read_jsonl(path) == [expected]
