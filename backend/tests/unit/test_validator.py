import pytest

from backend.app.pipeline.validator import ValidationConfig


@pytest.mark.parametrize(
    "kwargs",
    [
        {"duration_tolerance_seconds": -1},
        {"duration_tolerance_frames": -1},
        {"fps_absolute_tolerance": -1},
        {"aspect_relative_tolerance": -1},
        {"frame_count_tolerance": -1},
        {"decode_timeout_seconds": 0},
        {"fps_relative_tolerance": float("nan")},
    ],
)
def test_validation_configuration_rejects_invalid_tolerances(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        ValidationConfig(**kwargs)  # type: ignore[arg-type]
