from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import BinaryIO

import numpy as np

from backend.app.domain.errors import ClearFrameError, ErrorCode
from backend.app.domain.video import ColorMetadata, Frame, FrameBatch, TargetResolution
from backend.app.video.ffmpeg import FFmpegProcess

X264_PRESETS = frozenset(
    {
        "ultrafast",
        "superfast",
        "veryfast",
        "faster",
        "fast",
        "medium",
        "slow",
        "slower",
        "veryslow",
    }
)
SUPPORTED_COLOR_SPACES = frozenset(
    {"bt709", "fcc", "bt470bg", "smpte170m", "smpte240m", "bt2020nc", "bt2020c"}
)
PROCESS_EXIT_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True, slots=True)
class EncoderConfig:
    ffmpeg_path: str | os.PathLike[str] = "ffmpeg"
    codec: str = "libx264"
    crf: int = 18
    preset: str = "medium"
    pixel_format: str = "yuv420p"

    def __post_init__(self) -> None:
        if not self.codec:
            raise ValueError("video codec cannot be empty")
        if not 0 <= self.crf <= 51:
            raise ValueError("H.264 CRF must be between 0 and 51")
        if self.codec == "libx264" and self.preset not in X264_PRESETS:
            raise ValueError(f"unsupported x264 preset: {self.preset}")
        if not self.pixel_format:
            raise ValueError("output pixel format cannot be empty")


@dataclass(frozen=True, slots=True)
class EncodingResult:
    output_path: Path
    frames_encoded: int
    codec: str
    crf: int
    preset: str
    pixel_format: str


def _resolved_color(color: ColorMetadata) -> tuple[str, str, str, str]:
    space = color.space if color.space in SUPPORTED_COLOR_SPACES else "bt709"
    primaries = color.primaries or "bt709"
    transfer = color.transfer or "bt709"
    color_range = "pc" if color.range == "pc" else "tv"
    return space, primaries, transfer, color_range


def _write_frame(stream: BinaryIO, frame: Frame) -> None:
    data = memoryview(frame.tobytes(order="C"))
    offset = 0
    while offset < len(data):
        written = stream.write(data[offset:])
        if not written:
            raise BrokenPipeError("FFmpeg stopped accepting video frames")
        offset += written


def encode_video(
    batches: Iterable[FrameBatch],
    *,
    output_path: Path,
    target: TargetResolution,
    fps: Fraction,
    color: ColorMetadata,
    config: EncoderConfig | None = None,
) -> EncodingResult:
    """Stream exact-size BGR24 frames into a video-only MP4."""

    resolved_config = config or EncoderConfig()
    if fps <= 0:
        raise ValueError("encoding FPS must be positive")
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("video encoder output must use the .mp4 extension")
    if output_path.exists():
        raise ClearFrameError(ErrorCode.ENCODE_FAILED, f"Output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    space, primaries, transfer, color_range = _resolved_color(color)
    arguments = [
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-video_size",
        f"{target.width}x{target.height}",
        "-framerate",
        f"{fps.numerator}/{fps.denominator}",
        "-i",
        "pipe:0",
        "-an",
        "-vf",
        f"scale=in_range=full:out_range={color_range}:out_color_matrix={space}",
        "-c:v",
        resolved_config.codec,
        "-preset",
        resolved_config.preset,
        "-crf",
        str(resolved_config.crf),
        "-pix_fmt",
        resolved_config.pixel_format,
        "-colorspace",
        space,
        "-color_primaries",
        primaries,
        "-color_trc",
        transfer,
        "-color_range",
        color_range,
        "-movflags",
        "+faststart",
        "-n",
        os.fspath(output_path),
    ]
    process = FFmpegProcess(
        arguments,
        ffmpeg_path=resolved_config.ffmpeg_path,
        error_code=ErrorCode.ENCODE_FAILED,
        pipe_stdin=True,
    )
    assert process.process.stdin is not None
    succeeded = False
    frames_encoded = 0
    expected_start = 0

    try:
        for batch in batches:
            if batch.start_frame != expected_start:
                raise ClearFrameError(
                    ErrorCode.ENCODE_FAILED,
                    "Non-contiguous encode batch: "
                    f"expected {expected_start}, got {batch.start_frame}",
                )
            for frame in batch.frames:
                _validate_frame(frame, target)
                _write_frame(process.process.stdin, frame)
                frames_encoded += 1
            expected_start = batch.end_frame

        process.process.stdin.close()
        return_code = process.wait(timeout_seconds=PROCESS_EXIT_TIMEOUT_SECONDS)
        if return_code != 0:
            raise ClearFrameError(
                ErrorCode.ENCODE_FAILED,
                f"FFmpeg H.264 encoding failed: {process.stderr_detail}",
            )
        if frames_encoded == 0:
            raise ClearFrameError(ErrorCode.ENCODE_FAILED, "Cannot encode a video with zero frames")
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise ClearFrameError(
                ErrorCode.ENCODE_FAILED,
                "FFmpeg exited successfully but did not create a non-empty video",
            )
        succeeded = True
    except OSError as exc:
        raise ClearFrameError(
            ErrorCode.ENCODE_FAILED,
            f"Could not write restored frames to FFmpeg: {exc}; {process.stderr_detail}",
        ) from exc
    finally:
        if not succeeded:
            process.terminate()
            output_path.unlink(missing_ok=True)
        process.close()

    return EncodingResult(
        output_path=output_path,
        frames_encoded=frames_encoded,
        codec=resolved_config.codec,
        crf=resolved_config.crf,
        preset=resolved_config.preset,
        pixel_format=resolved_config.pixel_format,
    )


def _validate_frame(frame: Frame, target: TargetResolution) -> None:
    if frame.dtype != np.uint8:
        raise ClearFrameError(ErrorCode.ENCODE_FAILED, "Encoder frames must use uint8 samples")
    if frame.shape != (target.height, target.width, 3):
        raise ClearFrameError(
            ErrorCode.ENCODE_FAILED,
            f"Encoder frame has shape {frame.shape}; expected {(target.height, target.width, 3)}",
        )
