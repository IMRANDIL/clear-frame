from fractions import Fraction
from pathlib import Path

import pytest

from backend.app.domain.video import VideoMetadata
from backend.app.pipeline.planner import plan_target_resolution


def make_metadata(
    width: int,
    height: int,
    *,
    sample_aspect_ratio: Fraction = Fraction(1, 1),
    rotation_degrees: int = 0,
) -> VideoMetadata:
    return VideoMetadata(
        input_path=Path("fixture.mp4"),
        file_size=1024,
        width=width,
        height=height,
        fps=Fraction(30, 1),
        duration_seconds=10.0,
        video_stream_index=0,
        codec_name="h264",
        pixel_format="yuv420p",
        container_format="mov,mp4,m4a,3gp,3g2,mj2",
        sample_aspect_ratio=sample_aspect_ratio,
        rotation_degrees=rotation_degrees,
    )


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        (make_metadata(640, 360), (1920, 1080)),
        (make_metadata(640, 480), (1440, 1080)),
        (make_metadata(720, 1280), (1080, 1920)),
        (make_metadata(480, 480), (1080, 1080)),
        (make_metadata(1920, 1080, rotation_degrees=90), (1080, 1920)),
        (
            make_metadata(720, 576, sample_aspect_ratio=Fraction(16, 15)),
            (1440, 1080),
        ),
    ],
)
def test_standard_target_resolutions(
    metadata: VideoMetadata,
    expected: tuple[int, int],
) -> None:
    target = plan_target_resolution(metadata)

    assert (target.width, target.height) == expected
    assert target.relative_aspect_error == pytest.approx(0.0)


def test_nonstandard_ratio_rounds_to_nearest_even_width() -> None:
    target = plan_target_resolution(make_metadata(853, 480))

    assert (target.width, target.height) == (1920, 1080)
    assert target.relative_aspect_error < 0.001


@pytest.mark.parametrize("target_axis", [0, -2, 1079])
def test_target_axis_must_be_positive_and_even(target_axis: int) -> None:
    with pytest.raises(ValueError, match="positive even"):
        plan_target_resolution(make_metadata(640, 360), target_axis=target_axis)
