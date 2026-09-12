from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator

import cv2
import numpy as np

from backend.app.domain.errors import ClearFrameError, ErrorCode
from backend.app.domain.video import Frame, FrameBatch, TargetResolution, VideoMetadata
from backend.app.models.base import EnhancementModel


def orient_frame(frame: Frame, rotation_degrees: int) -> Frame:
    if rotation_degrees % 90 != 0:
        raise ValueError("frame rotation must be a multiple of 90 degrees")
    quarter_turns = (rotation_degrees % 360) // 90
    if quarter_turns == 0:
        return frame
    return np.ascontiguousarray(np.rot90(frame, k=quarter_turns))


def resize_frame(frame: Frame, target: TargetResolution) -> Frame:
    if (frame.shape[1], frame.shape[0]) == (target.width, target.height):
        return frame
    resized = cv2.resize(
        frame,
        (target.width, target.height),
        interpolation=cv2.INTER_LANCZOS4,
    )
    return np.ascontiguousarray(resized)


def restore_frame_batches(
    batches: Iterable[FrameBatch],
    *,
    metadata: VideoMetadata,
    target: TargetResolution,
    model: EnhancementModel,
    inference_observer: Callable[[int, float], None] | None = None,
) -> Iterator[FrameBatch]:
    """Orient, restore, and resize one bounded batch at a time."""

    if not model.loaded:
        raise ClearFrameError(ErrorCode.INFERENCE_FAILED, f"Model {model.name} is not loaded")

    expected_start = 0
    for batch in batches:
        if batch.start_frame != expected_start:
            raise ClearFrameError(
                ErrorCode.INFERENCE_FAILED,
                f"Non-contiguous frame batch: expected {expected_start}, got {batch.start_frame}",
            )

        oriented = FrameBatch(
            frames=[orient_frame(frame, metadata.rotation_degrees) for frame in batch.frames],
            start_frame=batch.start_frame,
            end_frame=batch.end_frame,
        )
        inference_started = time.perf_counter()
        restored = model.enhance(oriented)
        inference_seconds = time.perf_counter() - inference_started
        if inference_observer is not None:
            inference_observer(len(oriented.frames), inference_seconds)
        yield FrameBatch(
            frames=[resize_frame(frame, target) for frame in restored.frames],
            start_frame=restored.start_frame,
            end_frame=restored.end_frame,
        )
        expected_start = batch.end_frame
