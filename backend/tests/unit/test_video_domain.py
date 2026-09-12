from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from backend.app.domain.video import FrameBatch, VideoMetadata


def make_metadata(**overrides: object) -> VideoMetadata:
    values: dict[str, object] = {
        "input_path": Path("fixture.mp4"),
        "file_size": 1024,
        "width": 640,
        "height": 360,
        "fps": Fraction(30, 1),
        "duration_seconds": 10.0,
        "video_stream_index": 0,
        "codec_name": "h264",
        "pixel_format": "yuv420p",
        "container_format": "mov,mp4,m4a,3gp,3g2,mj2",
    }
    values.update(overrides)
    return VideoMetadata(**values)  # type: ignore[arg-type]


def test_display_aspect_ratio_includes_sample_aspect_ratio() -> None:
    metadata = make_metadata(
        width=720,
        height=576,
        sample_aspect_ratio=Fraction(16, 15),
    )

    assert metadata.display_aspect_ratio == Fraction(4, 3)


def test_rotation_swaps_display_aspect_ratio() -> None:
    metadata = make_metadata(width=1920, height=1080, rotation_degrees=-90)

    assert metadata.rotation_degrees == 270
    assert metadata.display_aspect_ratio == Fraction(9, 16)


def test_different_average_and_nominal_rates_mark_vfr() -> None:
    metadata = make_metadata(fps=Fraction(24000, 1001), nominal_fps=Fraction(30, 1))

    assert metadata.is_variable_frame_rate


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("file_size", 0, "file size"),
        ("width", 0, "dimensions"),
        ("fps", Fraction(0, 1), "FPS"),
        ("duration_seconds", float("nan"), "duration"),
        ("rotation_degrees", 45, "rotation"),
    ],
)
def test_invalid_video_metadata_is_rejected(field: str, value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make_metadata(**{field: value})


def test_frame_batch_uses_exclusive_end_index() -> None:
    frames = [np.zeros((8, 12, 3), dtype=np.uint8) for _ in range(2)]

    batch = FrameBatch(frames=frames, start_frame=5, end_frame=7)

    assert len(batch.frames) == 2


def test_frame_batch_rejects_mismatched_range() -> None:
    frame = np.zeros((8, 12, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="range"):
        FrameBatch(frames=[frame], start_frame=5, end_frame=7)


def test_frame_batch_rejects_non_bgr_shape() -> None:
    frame = np.zeros((8, 12), dtype=np.uint8)

    with pytest.raises(ValueError, match="HxWx3"):
        FrameBatch(frames=[frame], start_frame=0, end_frame=1)
