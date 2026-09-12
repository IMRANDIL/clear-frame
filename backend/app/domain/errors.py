from __future__ import annotations

from enum import StrEnum


class ErrorCode(StrEnum):
    INVALID_IMAGE = "INVALID_IMAGE"
    IMAGE_DECODE_FAILED = "IMAGE_DECODE_FAILED"
    IMAGE_ENCODE_FAILED = "IMAGE_ENCODE_FAILED"
    INVALID_VIDEO = "INVALID_VIDEO"
    UNSUPPORTED_CODEC = "UNSUPPORTED_CODEC"
    SOURCE_CORRUPT = "SOURCE_CORRUPT"
    FFPROBE_FAILED = "FFPROBE_FAILED"
    FFMPEG_DECODE_FAILED = "FFMPEG_DECODE_FAILED"
    MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
    CUDA_OUT_OF_MEMORY = "CUDA_OUT_OF_MEMORY"
    INFERENCE_FAILED = "INFERENCE_FAILED"
    ENCODE_FAILED = "ENCODE_FAILED"
    AUDIO_MUX_FAILED = "AUDIO_MUX_FAILED"
    OUTPUT_VALIDATION_FAILED = "OUTPUT_VALIDATION_FAILED"
    CANCELLED = "CANCELLED"


class ClearFrameError(RuntimeError):
    def __init__(self, code: ErrorCode, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
