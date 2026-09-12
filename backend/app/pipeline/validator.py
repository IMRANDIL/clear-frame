from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from backend.app.domain.errors import ClearFrameError, ErrorCode
from backend.app.domain.video import TargetResolution, VideoMetadata
from backend.app.video.ffmpeg import FFmpegProcess
from backend.app.video.ffprobe import probe_video


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    ffmpeg_path: str | os.PathLike[str] = "ffmpeg"
    ffprobe_path: str | os.PathLike[str] = "ffprobe"
    duration_tolerance_seconds: float = 0.5
    duration_tolerance_frames: int = 2
    fps_absolute_tolerance: float = 0.01
    fps_relative_tolerance: float = 0.001
    aspect_relative_tolerance: float = 0.001
    frame_count_tolerance: int = 1
    decode_timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        numeric_values = (
            self.duration_tolerance_seconds,
            self.fps_absolute_tolerance,
            self.fps_relative_tolerance,
            self.aspect_relative_tolerance,
            self.decode_timeout_seconds,
        )
        if any(not math.isfinite(value) or value < 0 for value in numeric_values):
            raise ValueError("validation tolerances cannot be negative")
        if self.duration_tolerance_frames < 0 or self.frame_count_tolerance < 0:
            raise ValueError("validation count tolerances cannot be negative")
        if self.decode_timeout_seconds == 0:
            raise ValueError("decode timeout must be positive")


@dataclass(frozen=True, slots=True)
class ValidationCheck:
    name: str
    expected: str
    actual: str


@dataclass(frozen=True, slots=True)
class OutputValidationResult:
    metadata: VideoMetadata
    decoded_frames: int
    checks: tuple[ValidationCheck, ...]


class _ProgressCollector(threading.Thread):
    def __init__(self, stream: BinaryIO) -> None:
        super().__init__(name="ffmpeg-validation-progress", daemon=True)
        self._stream = stream
        self.decoded_frames = 0

    def run(self) -> None:
        while line := self._stream.readline():
            key, separator, value = line.decode("utf-8", errors="replace").strip().partition("=")
            if separator and key == "frame":
                try:
                    self.decoded_frames = max(self.decoded_frames, int(value))
                except ValueError:
                    continue


def _full_decode(
    output_path: Path,
    *,
    ffmpeg_path: str | os.PathLike[str],
    timeout_seconds: float,
) -> int:
    process = FFmpegProcess(
        [
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-xerror",
            "-err_detect",
            "explode",
            "-i",
            output_path,
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-progress",
            "pipe:1",
            "-nostats",
            "-f",
            "null",
            "-",
        ],
        ffmpeg_path=ffmpeg_path,
        error_code=ErrorCode.OUTPUT_VALIDATION_FAILED,
        pipe_stdout=True,
    )
    assert process.process.stdout is not None
    progress = _ProgressCollector(process.process.stdout)
    progress.start()
    succeeded = False
    try:
        return_code = process.wait(timeout_seconds=timeout_seconds)
        progress.join(timeout=2)
        if return_code != 0:
            raise ClearFrameError(
                ErrorCode.OUTPUT_VALIDATION_FAILED,
                f"Output failed full decode: {process.stderr_detail}",
            )
        if progress.decoded_frames <= 0:
            raise ClearFrameError(
                ErrorCode.OUTPUT_VALIDATION_FAILED,
                "Output full decode produced zero video frames",
            )
        succeeded = True
        return progress.decoded_frames
    finally:
        if not succeeded:
            process.terminate()
        progress.join(timeout=2)
        process.close()


def _assert_check(
    condition: bool,
    *,
    name: str,
    expected: object,
    actual: object,
    checks: list[ValidationCheck],
) -> None:
    if not condition:
        raise ClearFrameError(
            ErrorCode.OUTPUT_VALIDATION_FAILED,
            f"Output validation failed for {name}: expected {expected}, got {actual}",
        )
    checks.append(ValidationCheck(name=name, expected=str(expected), actual=str(actual)))


def validate_output(
    output_path: Path,
    *,
    source: VideoMetadata,
    target: TargetResolution,
    expected_frames: int,
    config: ValidationConfig | None = None,
) -> OutputValidationResult:
    resolved_config = config or ValidationConfig()
    if expected_frames <= 0:
        raise ValueError("expected frame count must be positive")
    if not output_path.is_file() or output_path.stat().st_size <= 0:
        raise ClearFrameError(
            ErrorCode.OUTPUT_VALIDATION_FAILED,
            f"Output does not exist or is empty: {output_path}",
        )

    try:
        output = probe_video(output_path, ffprobe_path=resolved_config.ffprobe_path)
    except ClearFrameError as exc:
        raise ClearFrameError(
            ErrorCode.OUTPUT_VALIDATION_FAILED,
            f"Could not probe completed output: {exc}",
        ) from exc

    checks: list[ValidationCheck] = []
    _assert_check(
        (output.width, output.height) == (target.width, target.height),
        name="resolution",
        expected=f"{target.width}x{target.height}",
        actual=f"{output.width}x{output.height}",
        checks=checks,
    )
    aspect_error = abs(float(output.display_aspect_ratio / target.source_display_aspect_ratio) - 1)
    _assert_check(
        aspect_error <= resolved_config.aspect_relative_tolerance,
        name="display_aspect_ratio",
        expected=target.source_display_aspect_ratio,
        actual=output.display_aspect_ratio,
        checks=checks,
    )

    fps_difference = abs(float(output.fps - source.fps))
    fps_relative_error = fps_difference / float(source.fps)
    _assert_check(
        fps_difference <= resolved_config.fps_absolute_tolerance
        or fps_relative_error <= resolved_config.fps_relative_tolerance,
        name="fps",
        expected=source.fps,
        actual=output.fps,
        checks=checks,
    )

    duration_tolerance = max(
        resolved_config.duration_tolerance_seconds,
        resolved_config.duration_tolerance_frames / float(source.fps),
    )
    duration_difference = abs(output.duration_seconds - source.duration_seconds)
    _assert_check(
        duration_difference <= duration_tolerance,
        name="duration_seconds",
        expected=f"{source.duration_seconds:.6f} +/- {duration_tolerance:.6f}",
        actual=f"{output.duration_seconds:.6f}",
        checks=checks,
    )
    _assert_check(
        output.audio_present == source.audio_present,
        name="audio_presence",
        expected=source.audio_present,
        actual=output.audio_present,
        checks=checks,
    )
    if source.audio_present:
        assert output.audio is not None
        _assert_check(
            output.audio.codec_name == "aac",
            name="audio_codec",
            expected="aac",
            actual=output.audio.codec_name,
            checks=checks,
        )
    _assert_check(
        output.codec_name == "h264",
        name="video_codec",
        expected="h264",
        actual=output.codec_name,
        checks=checks,
    )
    _assert_check(
        output.pixel_format == "yuv420p",
        name="pixel_format",
        expected="yuv420p",
        actual=output.pixel_format,
        checks=checks,
    )
    _assert_check(
        output.rotation_degrees == 0,
        name="rotation",
        expected=0,
        actual=output.rotation_degrees,
        checks=checks,
    )
    _assert_check(
        "mp4" in output.container_format.split(","),
        name="container",
        expected="mp4",
        actual=output.container_format,
        checks=checks,
    )

    for name, expected, actual in (
        ("color_primaries", source.color.primaries, output.color.primaries),
        ("color_transfer", source.color.transfer, output.color.transfer),
        ("color_space", source.color.space, output.color.space),
        ("color_range", source.color.range, output.color.range),
    ):
        if expected is not None:
            _assert_check(
                actual == expected,
                name=name,
                expected=expected,
                actual=actual,
                checks=checks,
            )

    if output.frame_count is not None:
        _assert_check(
            abs(output.frame_count - expected_frames) <= resolved_config.frame_count_tolerance,
            name="reported_frame_count",
            expected=expected_frames,
            actual=output.frame_count,
            checks=checks,
        )

    decoded_frames = _full_decode(
        output_path,
        ffmpeg_path=resolved_config.ffmpeg_path,
        timeout_seconds=resolved_config.decode_timeout_seconds,
    )
    _assert_check(
        abs(decoded_frames - expected_frames) <= resolved_config.frame_count_tolerance,
        name="decoded_frame_count",
        expected=expected_frames,
        actual=decoded_frames,
        checks=checks,
    )
    return OutputValidationResult(
        metadata=output,
        decoded_frames=decoded_frames,
        checks=tuple(checks),
    )
