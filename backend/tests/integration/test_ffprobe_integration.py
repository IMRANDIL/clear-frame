from __future__ import annotations

import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

from backend.app.video.ffprobe import probe_video


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_probe_generated_mp4_with_audio_and_path_spaces(tmp_path: Path) -> None:
    fixture_directory = tmp_path / "path with spaces"
    fixture_directory.mkdir()
    fixture = fixture_directory / "probe fixture.mp4"
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=30000/1001",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:sample_rate=48000",
            "-t",
            "1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(fixture),
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr

    metadata = probe_video(fixture)

    assert metadata.input_path == fixture.resolve()
    assert (metadata.width, metadata.height) == (320, 180)
    assert metadata.fps == Fraction(30000, 1001)
    assert metadata.duration_seconds == pytest.approx(1.0, abs=0.05)
    assert metadata.codec_name == "h264"
    assert metadata.pixel_format == "yuv420p"
    assert metadata.audio is not None
    assert metadata.audio.codec_name == "aac"
