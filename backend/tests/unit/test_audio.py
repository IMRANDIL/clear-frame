from pathlib import Path

import pytest

from backend.app.domain.errors import ClearFrameError, ErrorCode
from backend.app.pipeline.audio import AudioMuxConfig, mux_source_audio


def test_audio_mux_rejects_missing_video_stream(tmp_path: Path) -> None:
    with pytest.raises(ClearFrameError) as caught:
        mux_source_audio(
            tmp_path / "missing.mp4",
            source=None,  # type: ignore[arg-type]
            output_path=tmp_path / "output.mp4",
        )

    assert caught.value.code is ErrorCode.AUDIO_MUX_FAILED


def test_audio_mux_bitrate_must_be_positive() -> None:
    with pytest.raises(ValueError, match="bitrate"):
        AudioMuxConfig(aac_bitrate_kbps=0)
