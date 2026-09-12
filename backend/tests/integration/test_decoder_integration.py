from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from backend.app.pipeline.decoder import DecoderConfig, decode_frame_batches
from backend.app.video.ffprobe import probe_video


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_decoder_streams_bounded_batches(tmp_path: Path) -> None:
    fixture = tmp_path / "five frames.mp4"
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x36:rate=5",
            "-frames:v",
            "5",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(fixture),
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    metadata = probe_video(fixture)

    batches = list(decode_frame_batches(metadata, config=DecoderConfig(batch_size=2)))

    assert [(batch.start_frame, batch.end_frame) for batch in batches] == [(0, 2), (2, 4), (4, 5)]
    assert all(len(batch.frames) <= 2 for batch in batches)
    assert all(frame.shape == (36, 64, 3) for batch in batches for frame in batch.frames)
    assert all(frame.dtype == np.uint8 for batch in batches for frame in batch.frames)
