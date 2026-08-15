import pytest

from akbc_baseline.speed_gate import estimate_seconds


def test_estimates_full_dataset_runtime() -> None:
    assert estimate_seconds(10, 2, 100) == 500


@pytest.mark.parametrize(
    ("elapsed", "measured", "target"),
    [(0, 1, 1), (1, 0, 1), (1, 1, 0)],
)
def test_rejects_invalid_speed_measurement(
    elapsed: float, measured: int, target: int
) -> None:
    with pytest.raises(ValueError, match="positive"):
        estimate_seconds(elapsed, measured, target)
