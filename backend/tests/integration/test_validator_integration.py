from __future__ import annotations

import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

from backend.app.domain.errors import ClearFrameError, ErrorCode
from backend.app.domain.video import TargetResolution
from backend.app.pipeline.validator import validate_output
from backend.app.video.ffmpeg import get_ffmpeg_version
from backend.app.video.ffprobe import probe_video


def generate_video(path: Path) -> None:
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
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:sample_rate=48000",
            "-frames:v",
            "5",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_validator_probes_and_fully_decodes_output(tmp_path: Path) -> None:
    source_path = tmp_path / "source.mp4"
    output_path = tmp_path / "output.mp4"
    generate_video(source_path)
    shutil.copyfile(source_path, output_path)
    source = probe_video(source_path)
    target = TargetResolution(64, 36, Fraction(16, 9))

    result = validate_output(
        output_path,
        source=source,
        target=target,
        expected_frames=5,
    )

    assert result.decoded_frames == 5
    assert {check.name for check in result.checks} >= {
        "resolution",
        "fps",
        "duration_seconds",
        "audio_presence",
        "decoded_frame_count",
    }


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_validator_rejects_wrong_resolution(tmp_path: Path) -> None:
    source_path = tmp_path / "source.mp4"
    output_path = tmp_path / "output.mp4"
    generate_video(source_path)
    shutil.copyfile(source_path, output_path)
    source = probe_video(source_path)

    with pytest.raises(ClearFrameError) as caught:
        validate_output(
            output_path,
            source=source,
            target=TargetResolution(128, 72, Fraction(16, 9)),
            expected_frames=5,
        )

    assert caught.value.code is ErrorCode.OUTPUT_VALIDATION_FAILED
    assert "resolution" in str(caught.value)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_ffmpeg_version_is_available_for_metrics() -> None:
    assert get_ffmpeg_version().startswith("ffmpeg version")
