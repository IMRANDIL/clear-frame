from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

type Frame = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class ColorMetadata:
    primaries: str | None = None
    transfer: str | None = None
    space: str | None = None
    range: str | None = None


@dataclass(frozen=True, slots=True)
class AudioStreamMetadata:
    stream_index: int
    codec_name: str
    sample_rate: int | None = None
    channels: int | None = None
    bitrate: int | None = None

    def __post_init__(self) -> None:
        if self.stream_index < 0:
            raise ValueError("audio stream index cannot be negative")
        if not self.codec_name:
            raise ValueError("audio codec name is required")
        if self.sample_rate is not None and self.sample_rate <= 0:
            raise ValueError("audio sample rate must be positive")
        if self.channels is not None and self.channels <= 0:
            raise ValueError("audio channel count must be positive")
        if self.bitrate is not None and self.bitrate < 0:
            raise ValueError("audio bitrate cannot be negative")


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    input_path: Path
    file_size: int
    width: int
    height: int
    fps: Fraction
    duration_seconds: float
    video_stream_index: int
    codec_name: str
    pixel_format: str
    container_format: str
    frame_count: int | None = None
    bitrate: int | None = None
    nominal_fps: Fraction | None = None
    sample_aspect_ratio: Fraction = field(default_factory=lambda: Fraction(1, 1))
    rotation_degrees: int = 0
    audio: AudioStreamMetadata | None = None
    color: ColorMetadata = field(default_factory=ColorMetadata)

    def __post_init__(self) -> None:
        if self.file_size <= 0:
            raise ValueError("video file size must be positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("video dimensions must be positive")
        if self.fps <= 0:
            raise ValueError("video FPS must be positive")
        if not math.isfinite(self.duration_seconds) or self.duration_seconds <= 0:
            raise ValueError("video duration must be finite and positive")
        if self.video_stream_index < 0:
            raise ValueError("video stream index cannot be negative")
        if not self.codec_name:
            raise ValueError("video codec name is required")
        if not self.pixel_format:
            raise ValueError("video pixel format is required")
        if not self.container_format:
            raise ValueError("container format is required")
        if self.frame_count is not None and self.frame_count <= 0:
            raise ValueError("video frame count must be positive when provided")
        if self.bitrate is not None and self.bitrate < 0:
            raise ValueError("video bitrate cannot be negative")
        if self.nominal_fps is not None and self.nominal_fps <= 0:
            raise ValueError("nominal video FPS must be positive when provided")
        if self.sample_aspect_ratio <= 0:
            raise ValueError("sample aspect ratio must be positive")
        if self.rotation_degrees % 90 != 0:
            raise ValueError("video rotation must be a multiple of 90 degrees")

        object.__setattr__(self, "rotation_degrees", self.rotation_degrees % 360)

    @property
    def filename(self) -> str:
        return self.input_path.name

    @property
    def audio_present(self) -> bool:
        return self.audio is not None

    @property
    def is_variable_frame_rate(self) -> bool:
        return self.nominal_fps is not None and self.nominal_fps != self.fps

    @property
    def display_aspect_ratio(self) -> Fraction:
        encoded_ratio = Fraction(
            self.width * self.sample_aspect_ratio.numerator,
            self.height * self.sample_aspect_ratio.denominator,
        )
        if self.rotation_degrees in {90, 270}:
            return 1 / encoded_ratio
        return encoded_ratio


@dataclass(slots=True)
class FrameBatch:
    """A contiguous, non-empty range where end_frame is exclusive."""

    frames: list[Frame]
    start_frame: int
    end_frame: int

    def __post_init__(self) -> None:
        if self.start_frame < 0:
            raise ValueError("batch start frame cannot be negative")
        if self.end_frame <= self.start_frame:
            raise ValueError("batch end frame must be greater than its start frame")
        if len(self.frames) != self.end_frame - self.start_frame:
            raise ValueError("batch frame range must match the number of frames")

        for frame in self.frames:
            if frame.dtype != np.uint8:
                raise ValueError("frames must use uint8 samples")
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError("frames must have HxWx3 shape")


@dataclass(frozen=True, slots=True)
class TargetResolution:
    width: int
    height: int
    source_display_aspect_ratio: Fraction

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("target dimensions must be positive")
        if self.width % 2 != 0 or self.height % 2 != 0:
            raise ValueError("target dimensions must be even for yuv420p encoding")
        if self.source_display_aspect_ratio <= 0:
            raise ValueError("source display aspect ratio must be positive")

    @property
    def display_aspect_ratio(self) -> Fraction:
        return Fraction(self.width, self.height)

    @property
    def relative_aspect_error(self) -> float:
        difference = abs(self.display_aspect_ratio - self.source_display_aspect_ratio)
        return float(difference / self.source_display_aspect_ratio)
