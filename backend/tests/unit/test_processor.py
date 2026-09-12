from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from backend.app.domain.errors import ClearFrameError, ErrorCode
from backend.app.domain.video import FrameBatch, TargetResolution, VideoMetadata
from backend.app.models.base import EnhancementModel, ModelRuntime
from backend.app.pipeline.processor import orient_frame, restore_frame_batches


class IdentityModel(EnhancementModel):
    name = "identity"
    version = "test"
    temporal = False

    def __init__(self, *, loaded: bool = True) -> None:
        self._loaded = loaded

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def runtime(self) -> ModelRuntime | None:
        return ModelRuntime("cpu", "fp32", 1, 0) if self.loaded else None

    def load(self) -> None:
        self._loaded = True

    def enhance(self, batch: FrameBatch) -> FrameBatch:
        return batch

    def unload(self) -> None:
        self._loaded = False


def metadata(*, rotation_degrees: int = 0) -> VideoMetadata:
    return VideoMetadata(
        input_path=Path("fixture.mp4"),
        file_size=1,
        width=4,
        height=2,
        fps=Fraction(1, 1),
        duration_seconds=1,
        video_stream_index=0,
        codec_name="h264",
        pixel_format="yuv420p",
        container_format="mp4",
        rotation_degrees=rotation_degrees,
    )


def test_orient_frame_applies_display_rotation() -> None:
    frame = np.zeros((2, 4, 3), dtype=np.uint8)
    frame[:, :, 0] = [[1, 2, 3, 4], [5, 6, 7, 8]]

    oriented = orient_frame(frame, 90)

    assert oriented[:, :, 0].tolist() == [[4, 8], [3, 7], [2, 6], [1, 5]]
    assert oriented.flags.c_contiguous


def test_restore_batches_orients_and_resizes_exactly() -> None:
    frame = np.zeros((2, 4, 3), dtype=np.uint8)
    source = [FrameBatch(frames=[frame], start_frame=0, end_frame=1)]
    target = TargetResolution(4, 8, Fraction(1, 2))

    result = list(
        restore_frame_batches(
            source,
            metadata=metadata(rotation_degrees=90),
            target=target,
            model=IdentityModel(),
        )
    )

    assert result[0].frames[0].shape == (8, 4, 3)
    assert (result[0].start_frame, result[0].end_frame) == (0, 1)


def test_restore_batches_requires_loaded_model() -> None:
    with pytest.raises(ClearFrameError) as caught:
        list(
            restore_frame_batches(
                [],
                metadata=metadata(),
                target=TargetResolution(4, 2, Fraction(2, 1)),
                model=IdentityModel(loaded=False),
            )
        )

    assert caught.value.code is ErrorCode.INFERENCE_FAILED


def test_restore_batches_rejects_non_contiguous_input() -> None:
    frame = np.zeros((2, 4, 3), dtype=np.uint8)
    source = [FrameBatch(frames=[frame], start_frame=1, end_frame=2)]

    with pytest.raises(ClearFrameError, match="Non-contiguous"):
        list(
            restore_frame_batches(
                source,
                metadata=metadata(),
                target=TargetResolution(4, 2, Fraction(2, 1)),
                model=IdentityModel(),
            )
        )
