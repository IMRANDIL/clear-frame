from __future__ import annotations

import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from backend.app.domain.video import ColorMetadata, FrameBatch, TargetResolution
from backend.app.pipeline.audio import mux_source_audio
from backend.app.pipeline.encoder import EncoderConfig, encode_video
from backend.app.video.ffprobe import probe_video


def generated_batches(frame_count: int, width: int, height: int) -> list[FrameBatch]:
    frames = []
    for index in range(frame_count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = index * 20
        frame[:, :, 1] = np.arange(width, dtype=np.uint8)
        frames.append(frame)
    return [FrameBatch(frames=frames, start_frame=0, end_frame=frame_count)]


def generate_source(path: Path, *, audio_codec: str, container_codec: str) -> None:
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
            "-t",
            "1",
            "-c:v",
            container_codec,
            "-c:a",
            audio_codec,
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
def test_encode_h264_and_copy_aac_audio(tmp_path: Path) -> None:
    source_path = tmp_path / "source with aac.mp4"
    video_only = tmp_path / "video-only.mp4"
    output = tmp_path / "restored.mp4"
    generate_source(source_path, audio_codec="aac", container_codec="libx264")
    source = probe_video(source_path)
    target = TargetResolution(64, 36, Fraction(16, 9))

    encoded = encode_video(
        generated_batches(5, 64, 36),
        output_path=video_only,
        target=target,
        fps=Fraction(5, 1),
        color=ColorMetadata(space="bt709", primaries="bt709", transfer="bt709", range="tv"),
        config=EncoderConfig(crf=18),
    )
    muxed = mux_source_audio(video_only, source=source, output_path=output)
    metadata = probe_video(output)

    assert encoded.frames_encoded == 5
    assert muxed.audio_mode == "copy"
    assert metadata.codec_name == "h264"
    assert metadata.pixel_format == "yuv420p"
    assert metadata.fps == Fraction(5, 1)
    assert metadata.frame_count == 5
    assert metadata.audio is not None
    assert metadata.audio.codec_name == "aac"
    assert metadata.color.space == "bt709"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_mux_transcodes_opus_audio_to_aac(tmp_path: Path) -> None:
    source_path = tmp_path / "source.webm"
    video_only = tmp_path / "video-only.mp4"
    output = tmp_path / "restored.mp4"
    generate_source(source_path, audio_codec="libopus", container_codec="libvpx-vp9")
    source = probe_video(source_path)
    target = TargetResolution(64, 36, Fraction(16, 9))
    encode_video(
        generated_batches(5, 64, 36),
        output_path=video_only,
        target=target,
        fps=Fraction(5, 1),
        color=ColorMetadata(),
    )

    muxed = mux_source_audio(video_only, source=source, output_path=output)
    metadata = probe_video(output)

    assert muxed.audio_mode == "transcode"
    assert muxed.source_audio_codec == "opus"
    assert muxed.output_audio_codec == "aac"
    assert metadata.audio is not None
    assert metadata.audio.codec_name == "aac"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_mux_preserves_video_without_adding_audio(tmp_path: Path) -> None:
    video_only = tmp_path / "video-only.mp4"
    output = tmp_path / "restored.mp4"
    target = TargetResolution(64, 36, Fraction(16, 9))
    encode_video(
        generated_batches(5, 64, 36),
        output_path=video_only,
        target=target,
        fps=Fraction(5, 1),
        color=ColorMetadata(),
    )
    source = probe_video(video_only)

    muxed = mux_source_audio(video_only, source=source, output_path=output)
    metadata = probe_video(output)

    assert muxed.audio_mode == "none"
    assert metadata.audio is None
    assert metadata.frame_count == 5
