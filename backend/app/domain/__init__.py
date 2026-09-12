"""Core video processing domain types."""

from backend.app.domain.errors import ClearFrameError, ErrorCode
from backend.app.domain.video import (
    AudioStreamMetadata,
    ColorMetadata,
    Frame,
    FrameBatch,
    TargetResolution,
    VideoMetadata,
)

__all__ = [
    "AudioStreamMetadata",
    "ClearFrameError",
    "ColorMetadata",
    "ErrorCode",
    "Frame",
    "FrameBatch",
    "TargetResolution",
    "VideoMetadata",
]
