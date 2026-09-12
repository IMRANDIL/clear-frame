from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path


def _environment_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _environment_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


def parse_tile_size(value: str) -> int | None:
    if value.strip().lower() == "auto":
        return None
    try:
        tile_size = int(value)
    except ValueError as exc:
        raise ValueError("tile size must be 'auto' or a non-negative integer") from exc
    if tile_size < 0:
        raise ValueError("tile size must be 'auto' or a non-negative integer")
    return tile_size


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    model_name: str = "realesrgan-x4plus"
    model_directory: Path = Path("data/models")
    temporary_directory: Path = Path("data/temp")
    target_axis: int = 1080
    device: str = "auto"
    tile_size: int | None = None
    batch_size: int = 1
    use_half: bool | None = None
    video_codec: str = "libx264"
    crf: int = 18
    encoder_preset: str = "medium"
    pixel_format: str = "yuv420p"
    aac_bitrate_kbps: int = 192
    overwrite: bool = False

    def __post_init__(self) -> None:
        if not self.ffmpeg_path or not self.ffprobe_path:
            raise ValueError("FFmpeg and FFprobe paths cannot be empty")
        if not self.model_name:
            raise ValueError("model name cannot be empty")
        if self.target_axis <= 0 or self.target_axis % 2 != 0:
            raise ValueError("target axis must be a positive even integer")
        if self.tile_size is not None and self.tile_size < 0:
            raise ValueError("tile size cannot be negative")
        if not 1 <= self.batch_size <= 8:
            raise ValueError("batch size must be between 1 and 8")
        if not 0 <= self.crf <= 51:
            raise ValueError("H.264 CRF must be between 0 and 51")
        if self.aac_bitrate_kbps <= 0:
            raise ValueError("AAC bitrate must be positive")

    @classmethod
    def from_environment(cls) -> PipelineConfig:
        return cls(
            ffmpeg_path=os.getenv("CLEARFRAME_FFMPEG", "ffmpeg"),
            ffprobe_path=os.getenv("CLEARFRAME_FFPROBE", "ffprobe"),
            model_name=os.getenv("CLEARFRAME_MODEL", "realesrgan-x4plus"),
            model_directory=Path(os.getenv("CLEARFRAME_MODEL_DIR", "data/models")),
            temporary_directory=Path(os.getenv("CLEARFRAME_TEMP_DIR", "data/temp")),
            device=os.getenv("CLEARFRAME_DEVICE", "auto"),
            tile_size=parse_tile_size(os.getenv("CLEARFRAME_TILE_SIZE", "auto")),
            batch_size=_environment_int("CLEARFRAME_BATCH_SIZE", 1),
            crf=_environment_int("CLEARFRAME_CRF", 18),
            encoder_preset=os.getenv("CLEARFRAME_ENCODER_PRESET", "medium"),
            aac_bitrate_kbps=_environment_int("CLEARFRAME_AAC_BITRATE_KBPS", 192),
        )


@dataclass(frozen=True, slots=True)
class ImagePipelineConfig:
    model_name: str = "realesrgan-x4plus"
    model_directory: Path = Path("data/models")
    device: str = "auto"
    tile_size: int | None = None
    use_half: bool | None = None
    output_scale: float = 4.0
    lighting: str = "auto"
    jpeg_quality: int = 95
    png_compression: int = 3
    overwrite: bool = False

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError("model name cannot be empty")
        if self.tile_size is not None and self.tile_size < 0:
            raise ValueError("tile size cannot be negative")
        if not math.isfinite(self.output_scale) or not 1 <= self.output_scale <= 4:
            raise ValueError("image output scale must be between 1 and 4")
        if self.lighting not in {"auto", "force", "off"}:
            raise ValueError("image lighting must be auto, force, or off")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("JPEG quality must be between 1 and 100")
        if not 0 <= self.png_compression <= 9:
            raise ValueError("PNG compression must be between 0 and 9")

    @classmethod
    def from_environment(cls) -> ImagePipelineConfig:
        return cls(
            model_name=os.getenv("CLEARFRAME_MODEL", "realesrgan-x4plus"),
            model_directory=Path(os.getenv("CLEARFRAME_MODEL_DIR", "data/models")),
            device=os.getenv("CLEARFRAME_DEVICE", "auto"),
            tile_size=parse_tile_size(os.getenv("CLEARFRAME_TILE_SIZE", "auto")),
            output_scale=_environment_float("CLEARFRAME_IMAGE_SCALE", 4.0),
            lighting=os.getenv("CLEARFRAME_IMAGE_LIGHTING", "auto").strip().lower(),
            jpeg_quality=_environment_int("CLEARFRAME_JPEG_QUALITY", 95),
            png_compression=_environment_int("CLEARFRAME_PNG_COMPRESSION", 3),
        )
