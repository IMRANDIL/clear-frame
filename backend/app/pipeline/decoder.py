from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import BinaryIO

import numpy as np

from backend.app.domain.errors import ClearFrameError, ErrorCode
from backend.app.domain.video import Frame, FrameBatch, VideoMetadata
from backend.app.video.ffmpeg import FFmpegProcess

BYTES_PER_BGR24_PIXEL = 3
MAX_BATCH_SIZE = 8
MAX_DECODED_FRAME_BYTES = 256 * 1024 * 1024
PROCESS_EXIT_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class DecoderConfig:
    ffmpeg_path: str | os.PathLike[str] = "ffmpeg"
    batch_size: int = 1

    def __post_init__(self) -> None:
        if not 1 <= self.batch_size <= MAX_BATCH_SIZE:
            raise ValueError(f"decoder batch size must be between 1 and {MAX_BATCH_SIZE}")


def _read_frame(stream: BinaryIO, frame_bytes: int) -> bytes | None:
    buffer = bytearray(frame_bytes)
    view = memoryview(buffer)
    offset = 0
    while offset < frame_bytes:
        count = stream.readinto(view[offset:])
        if not count:
            if offset == 0:
                return None
            raise ClearFrameError(
                ErrorCode.FFMPEG_DECODE_FAILED,
                f"FFmpeg emitted a partial frame: expected {frame_bytes} bytes, got {offset}",
            )
        offset += count
    return bytes(buffer)


def decode_frame_batches(
    metadata: VideoMetadata,
    *,
    config: DecoderConfig | None = None,
) -> Iterator[FrameBatch]:
    """Yield bounded BGR24 frame batches without materializing the full video."""

    resolved_config = config or DecoderConfig()
    frame_bytes = metadata.width * metadata.height * BYTES_PER_BGR24_PIXEL
    if frame_bytes > MAX_DECODED_FRAME_BYTES:
        raise ClearFrameError(
            ErrorCode.INVALID_VIDEO,
            f"Decoded frame requires {frame_bytes} bytes, above the safety limit",
        )

    arguments = [
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-noautorotate",
        "-i",
        os.fspath(metadata.input_path),
        "-map",
        f"0:{metadata.video_stream_index}",
        "-an",
        "-sn",
        "-dn",
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "pipe:1",
    ]
    process = FFmpegProcess(
        arguments,
        ffmpeg_path=resolved_config.ffmpeg_path,
        error_code=ErrorCode.FFMPEG_DECODE_FAILED,
        pipe_stdout=True,
    )
    assert process.process.stdout is not None
    completed = False
    frame_index = 0
    batch_frames: list[Frame] = []
    batch_start = 0

    try:
        while raw_frame := _read_frame(process.process.stdout, frame_bytes):
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
                metadata.height,
                metadata.width,
                BYTES_PER_BGR24_PIXEL,
            )
            batch_frames.append(frame.copy())
            frame_index += 1

            if len(batch_frames) == resolved_config.batch_size:
                yield FrameBatch(
                    frames=batch_frames,
                    start_frame=batch_start,
                    end_frame=frame_index,
                )
                batch_frames = []
                batch_start = frame_index

        if batch_frames:
            yield FrameBatch(
                frames=batch_frames,
                start_frame=batch_start,
                end_frame=frame_index,
            )

        return_code = process.wait(timeout_seconds=PROCESS_EXIT_TIMEOUT_SECONDS)
        completed = True
        if return_code != 0:
            raise ClearFrameError(
                ErrorCode.FFMPEG_DECODE_FAILED,
                f"FFmpeg frame decoding failed: {process.stderr_detail}",
            )
        if frame_index == 0:
            raise ClearFrameError(
                ErrorCode.FFMPEG_DECODE_FAILED,
                "FFmpeg decoded zero video frames",
            )
    finally:
        if not completed:
            process.terminate()
        process.close()
