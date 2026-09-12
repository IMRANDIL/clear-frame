from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib import metadata as package_metadata
from pathlib import Path

import torch

from backend.app.domain.video import TargetResolution, VideoMetadata
from backend.app.models.base import ModelRuntime
from backend.app.models.registry import ModelSpec
from backend.app.pipeline.audio import AudioMuxResult
from backend.app.pipeline.encoder import EncodingResult
from backend.app.pipeline.validator import OutputValidationResult

PIPELINE_VERSION = "clearframe-poc-v1"
HASH_CHUNK_SIZE = 1024 * 1024
RECORDED_PACKAGES = ("torch", "torchvision", "numpy", "opencv-python-headless")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in RECORDED_PACKAGES:
        try:
            versions[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _nvidia_driver_version() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.splitlines()[0].strip() if result.stdout.strip() else None


def _gpu_metrics() -> dict[str, object] | None:
    if not torch.cuda.is_available():
        return None
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    return {
        "device_index": device_index,
        "name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_memory_mib": round(properties.total_memory / 1024**2, 2),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device_index) / 1024**2, 2),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(device_index) / 1024**2, 2),
        "cuda_runtime": torch.version.cuda,
        "driver": _nvidia_driver_version(),
    }


def _video_metadata(metadata: VideoMetadata) -> dict[str, object]:
    return {
        "path": str(metadata.input_path),
        "file_size": metadata.file_size,
        "width": metadata.width,
        "height": metadata.height,
        "fps": str(metadata.fps),
        "duration_seconds": metadata.duration_seconds,
        "frame_count": metadata.frame_count,
        "codec": metadata.codec_name,
        "pixel_format": metadata.pixel_format,
        "bitrate": metadata.bitrate,
        "audio_codec": metadata.audio.codec_name if metadata.audio else None,
        "audio_present": metadata.audio_present,
        "sample_aspect_ratio": str(metadata.sample_aspect_ratio),
        "display_aspect_ratio": str(metadata.display_aspect_ratio),
        "rotation_degrees": metadata.rotation_degrees,
        "color_primaries": metadata.color.primaries,
        "color_transfer": metadata.color.transfer,
        "color_space": metadata.color.space,
        "color_range": metadata.color.range,
    }


class MetricsRecorder:
    def __init__(self, *, pipeline_version: str = PIPELINE_VERSION) -> None:
        self.run_id = uuid.uuid4().hex
        self.pipeline_version = pipeline_version
        self.started_at = datetime.now(UTC)
        self._started_counter = time.perf_counter()
        self._stage_seconds: dict[str, float] = {}
        self._inference_seconds = 0.0
        self._frames_processed = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            self._stage_seconds[name] = self._stage_seconds.get(name, 0.0) + elapsed

    def record_inference(self, frames: int, seconds: float) -> None:
        if frames <= 0 or seconds < 0:
            raise ValueError("inference metrics require positive frames and non-negative time")
        self._frames_processed += frames
        self._inference_seconds += seconds

    def build_report(
        self,
        *,
        source: VideoMetadata,
        target: TargetResolution,
        model: ModelSpec,
        model_runtime: ModelRuntime,
        encoding: EncodingResult,
        audio: AudioMuxResult,
        validation: OutputValidationResult,
        ffmpeg_version: str,
    ) -> dict[str, object]:
        source_sha256 = _sha256(source.input_path)
        output_sha256 = _sha256(validation.metadata.input_path)
        finished_at = datetime.now(UTC)
        wall_seconds = time.perf_counter() - self._started_counter
        output_metadata = _video_metadata(validation.metadata)
        output_metadata["sha256"] = output_sha256
        inference_fps = (
            self._frames_processed / self._inference_seconds
            if self._inference_seconds > 0
            else None
        )
        processing_fps = self._frames_processed / wall_seconds if wall_seconds > 0 else None
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "pipeline_version": self.pipeline_version,
            "started_at": self.started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "wall_clock_seconds": wall_seconds,
            "stage_seconds": dict(self._stage_seconds),
            "frames_processed": self._frames_processed,
            "inference_seconds": self._inference_seconds,
            "inference_fps": inference_fps,
            "processing_fps": processing_fps,
            "source": {**_video_metadata(source), "sha256": source_sha256},
            "target": {
                "width": target.width,
                "height": target.height,
                "display_aspect_ratio": str(target.display_aspect_ratio),
                "relative_aspect_error": target.relative_aspect_error,
            },
            "output": output_metadata,
            "model": {
                "name": model.name,
                "release": model.release,
                "sha256": model.sha256,
                "size_bytes": model.size_bytes,
                "realesrgan_source_commit": model.realesrgan_source_commit,
                "basicsr_source_commit": model.basicsr_source_commit,
                "device": model_runtime.device,
                "precision": model_runtime.precision,
                "scale": model_runtime.scale,
                "tile_size": model_runtime.tile_size,
            },
            "encoding": {
                "codec": encoding.codec,
                "crf": encoding.crf,
                "preset": encoding.preset,
                "pixel_format": encoding.pixel_format,
                "frames_encoded": encoding.frames_encoded,
                "audio_mode": audio.audio_mode,
                "source_audio_codec": audio.source_audio_codec,
                "output_audio_codec": audio.output_audio_codec,
            },
            "validation": {
                "decoded_frames": validation.decoded_frames,
                "checks": [
                    {"name": check.name, "expected": check.expected, "actual": check.actual}
                    for check in validation.checks
                ],
            },
            "environment": {
                "python": platform.python_version(),
                "packages": _package_versions(),
                "ffmpeg": ffmpeg_version,
                "gpu": _gpu_metrics(),
            },
        }


def write_report(report: Mapping[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.partial")
    try:
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
