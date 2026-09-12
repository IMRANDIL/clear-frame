from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from backend.app.domain.errors import ClearFrameError, ErrorCode
from backend.app.domain.video import ColorMetadata, FrameBatch, TargetResolution
from backend.app.pipeline.encoder import EncoderConfig, encode_video


@pytest.mark.parametrize("crf", [-1, 52])
def test_encoder_rejects_invalid_crf(crf: int) -> None:
    with pytest.raises(ValueError, match="CRF"):
        EncoderConfig(crf=crf)


def test_encoder_rejects_wrong_frame_dimensions_before_output_survives(tmp_path: Path) -> None:
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    output = tmp_path / "bad.mp4"

    with pytest.raises(ClearFrameError) as caught:
        encode_video(
            [FrameBatch([frame], 0, 1)],
            output_path=output,
            target=TargetResolution(8, 8, Fraction(1, 1)),
            fps=Fraction(1, 1),
            color=ColorMetadata(),
        )

    assert caught.value.code is ErrorCode.ENCODE_FAILED
    assert not output.exists()


def test_encoder_rejects_non_contiguous_batches(tmp_path: Path) -> None:
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    with pytest.raises(ClearFrameError, match="Non-contiguous"):
        encode_video(
            [FrameBatch([frame], 1, 2)],
            output_path=tmp_path / "bad.mp4",
            target=TargetResolution(4, 4, Fraction(1, 1)),
            fps=Fraction(1, 1),
            color=ColorMetadata(),
        )


def test_encoder_removes_zero_frame_output(tmp_path: Path) -> None:
    output = tmp_path / "empty.mp4"

    with pytest.raises(ClearFrameError):
        encode_video(
            [],
            output_path=output,
            target=TargetResolution(4, 4, Fraction(1, 1)),
            fps=Fraction(1, 1),
            color=ColorMetadata(),
        )

    assert not output.exists()
