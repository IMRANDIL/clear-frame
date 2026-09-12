from __future__ import annotations

import os
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from backend.app.config import ImagePipelineConfig
from backend.app.domain.errors import ClearFrameError, ErrorCode
from backend.app.domain.video import Frame, FrameBatch
from backend.app.image.processing import LightingResult, adjust_low_light, resize_restored_image
from backend.app.models.base import ModelRuntime
from backend.app.models.realesrgan import RealESRGANConfig, RealESRGANModel
from backend.app.models.registry import ensure_model_weights, load_model_spec, sha256_file
from backend.app.observability.metrics import write_report

SUPPORTED_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})
type ProgressObserver = Callable[[str, Mapping[str, object]], None]


@dataclass(frozen=True, slots=True)
class DecodedImage:
    pixels: Frame
    alpha: np.ndarray | None
    grayscale: bool
    format: str

    @property
    def width(self) -> int:
        return self.pixels.shape[1]

    @property
    def height(self) -> int:
        return self.pixels.shape[0]

    @property
    def has_alpha(self) -> bool:
        return self.alpha is not None


@dataclass(frozen=True, slots=True)
class ImageValidationResult:
    width: int
    height: int
    format: str
    has_alpha: bool
    file_size: int


@dataclass(frozen=True, slots=True)
class ImageEnhancementResult:
    output_path: Path
    report_path: Path
    validation: ImageValidationResult
    lighting_applied: bool
    report: Mapping[str, object]


def _emit(observer: ProgressObserver | None, event: str, **fields: object) -> None:
    if observer is not None:
        observer(event, fields)


def _validate_paths(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    overwrite: bool,
) -> None:
    if input_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise ClearFrameError(
            ErrorCode.INVALID_IMAGE,
            "Input image must use a .png, .jpg, or .jpeg extension",
        )
    if output_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise ClearFrameError(
            ErrorCode.IMAGE_ENCODE_FAILED,
            "Output image must use a .png, .jpg, or .jpeg extension",
        )
    if input_path == output_path:
        raise ClearFrameError(ErrorCode.INVALID_IMAGE, "Input and output paths must be different")
    if report_path in {input_path, output_path}:
        raise ClearFrameError(
            ErrorCode.IMAGE_ENCODE_FAILED,
            "Report path must differ from image paths",
        )
    if not overwrite and output_path.exists():
        raise ClearFrameError(
            ErrorCode.IMAGE_ENCODE_FAILED,
            f"Output already exists: {output_path}",
        )
    if not overwrite and report_path.exists():
        raise ClearFrameError(
            ErrorCode.IMAGE_ENCODE_FAILED,
            f"Report already exists: {report_path}",
        )


def decode_image(path: Path) -> DecodedImage:
    if not path.is_file():
        raise ClearFrameError(ErrorCode.INVALID_IMAGE, f"Input image does not exist: {path}")
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
        mode = cv2.IMREAD_UNCHANGED if path.suffix.lower() == ".png" else cv2.IMREAD_COLOR
        decoded = cv2.imdecode(encoded, mode)
    except (OSError, cv2.error) as exc:
        raise ClearFrameError(
            ErrorCode.IMAGE_DECODE_FAILED,
            f"Could not decode image {path}: {exc}",
        ) from exc
    if decoded is None:
        raise ClearFrameError(ErrorCode.IMAGE_DECODE_FAILED, f"Could not decode image: {path}")
    if decoded.dtype != np.uint8:
        raise ClearFrameError(
            ErrorCode.INVALID_IMAGE,
            f"Only 8-bit images are supported; decoded {decoded.dtype}",
        )

    alpha: np.ndarray | None = None
    grayscale = decoded.ndim == 2
    if grayscale:
        pixels = cv2.cvtColor(decoded, cv2.COLOR_GRAY2BGR)
    elif decoded.ndim == 3 and decoded.shape[2] == 3:
        pixels = decoded
    elif decoded.ndim == 3 and decoded.shape[2] == 4:
        pixels = decoded[:, :, :3]
        alpha = np.ascontiguousarray(decoded[:, :, 3])
    else:
        raise ClearFrameError(
            ErrorCode.INVALID_IMAGE,
            f"Unsupported image channel layout: {decoded.shape}",
        )
    return DecodedImage(
        pixels=np.ascontiguousarray(pixels),
        alpha=alpha,
        grayscale=grayscale,
        format=path.suffix.lower().lstrip("."),
    )


def _encode_image(
    path: Path,
    image: Frame,
    *,
    alpha: np.ndarray | None,
    grayscale: bool,
    config: ImagePipelineConfig,
) -> None:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"} and alpha is not None:
        raise ClearFrameError(
            ErrorCode.IMAGE_ENCODE_FAILED,
            "JPEG cannot preserve transparency; choose a PNG output path",
        )

    output: np.ndarray = image
    if alpha is not None:
        output = np.dstack((image, alpha))
    elif grayscale:
        output = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    encode_suffix = ".jpg" if suffix == ".jpeg" else suffix
    parameters = (
        [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality, cv2.IMWRITE_JPEG_OPTIMIZE, 1]
        if suffix in {".jpg", ".jpeg"}
        else [cv2.IMWRITE_PNG_COMPRESSION, config.png_compression]
    )
    try:
        success, encoded = cv2.imencode(encode_suffix, output, parameters)
        if not success:
            raise ClearFrameError(
                ErrorCode.IMAGE_ENCODE_FAILED,
                f"Image encoder rejected {suffix} output",
            )
        path.write_bytes(encoded.tobytes())
    except ClearFrameError:
        raise
    except (OSError, cv2.error) as exc:
        raise ClearFrameError(
            ErrorCode.IMAGE_ENCODE_FAILED,
            f"Could not encode image {path}: {exc}",
        ) from exc


def _validate_output(
    path: Path,
    *,
    expected_width: int,
    expected_height: int,
    expected_alpha: bool,
) -> ImageValidationResult:
    decoded = decode_image(path)
    if (decoded.width, decoded.height) != (expected_width, expected_height):
        raise ClearFrameError(
            ErrorCode.OUTPUT_VALIDATION_FAILED,
            "Image dimensions changed during encoding: "
            f"expected {expected_width}x{expected_height}, got {decoded.width}x{decoded.height}",
        )
    if decoded.has_alpha != expected_alpha:
        raise ClearFrameError(
            ErrorCode.OUTPUT_VALIDATION_FAILED,
            f"Image transparency mismatch: expected {expected_alpha}, got {decoded.has_alpha}",
        )
    return ImageValidationResult(
        width=decoded.width,
        height=decoded.height,
        format=decoded.format,
        has_alpha=decoded.has_alpha,
        file_size=path.stat().st_size,
    )


def _build_report(
    *,
    run_id: str,
    started_at: datetime,
    started_counter: float,
    stages: Mapping[str, float],
    source_path: Path,
    source: DecodedImage,
    output_path: Path,
    validation: ImageValidationResult,
    lighting: LightingResult,
    config: ImagePipelineConfig,
    model_runtime: ModelRuntime,
    model_name: str,
    model_release: str,
    model_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "pipeline_version": "clearframe-image-v1",
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "wall_clock_seconds": time.perf_counter() - started_counter,
        "stage_seconds": dict(stages),
        "source": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "file_size": source_path.stat().st_size,
            "width": source.width,
            "height": source.height,
            "format": source.format,
            "grayscale": source.grayscale,
            "has_alpha": source.has_alpha,
        },
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "file_size": validation.file_size,
            "width": validation.width,
            "height": validation.height,
            "format": validation.format,
            "has_alpha": validation.has_alpha,
        },
        "processing": {
            "output_scale": config.output_scale,
            "lighting_mode": config.lighting,
            "lighting_applied": lighting.applied,
            "source_luminance_median": lighting.source_median,
            "gamma": lighting.gamma,
            "jpeg_quality": config.jpeg_quality,
            "png_compression": config.png_compression,
        },
        "model": {
            "name": model_name,
            "release": model_release,
            "sha256": model_sha256,
            "device": model_runtime.device,
            "precision": model_runtime.precision,
            "scale": model_runtime.scale,
            "tile_size": model_runtime.tile_size,
        },
        "validation": {
            "decoded": True,
            "dimensions_match": True,
            "transparency_matches": True,
        },
    }


def run_image_pipeline(
    input_path: Path,
    output_path: Path,
    *,
    config: ImagePipelineConfig | None = None,
    report_path: Path | None = None,
    observer: ProgressObserver | None = None,
) -> ImageEnhancementResult:
    resolved_config = config or ImagePipelineConfig.from_environment()
    source_path = input_path.expanduser().resolve()
    final_path = output_path.expanduser().resolve()
    final_report_path = (
        report_path.expanduser().resolve()
        if report_path is not None
        else Path(f"{final_path}.report.json")
    )
    _validate_paths(
        source_path,
        final_path,
        final_report_path,
        overwrite=resolved_config.overwrite,
    )

    run_id = uuid.uuid4().hex
    started_at = datetime.now(UTC)
    started_counter = time.perf_counter()
    stages: dict[str, float] = {}

    def run_stage(name: str, operation: Callable[[], object]) -> object:
        started = time.perf_counter()
        try:
            return operation()
        finally:
            stages[name] = time.perf_counter() - started

    _emit(observer, "image_pipeline_started", run_id=run_id, input=str(source_path))
    source = run_stage("decoding", lambda: decode_image(source_path))
    assert isinstance(source, DecodedImage)
    if source.has_alpha and final_path.suffix.lower() != ".png":
        raise ClearFrameError(
            ErrorCode.IMAGE_ENCODE_FAILED,
            "JPEG cannot preserve transparency; choose a PNG output path",
        )
    _emit(
        observer,
        "image_analyzed",
        width=source.width,
        height=source.height,
        format=source.format,
        has_alpha=source.has_alpha,
    )

    lighting = run_stage(
        "lighting",
        lambda: adjust_low_light(source.pixels, resolved_config.lighting),
    )
    assert isinstance(lighting, LightingResult)
    _emit(
        observer,
        "lighting_adjusted",
        applied=lighting.applied,
        source_median=lighting.source_median,
        gamma=lighting.gamma,
    )

    model_spec = load_model_spec(resolved_config.model_name)
    weights_path = run_stage(
        "model_preparation",
        lambda: ensure_model_weights(model_spec, resolved_config.model_directory),
    )
    assert isinstance(weights_path, Path)
    model = RealESRGANModel(
        model_spec,
        RealESRGANConfig(
            weights_path=weights_path,
            device=resolved_config.device,
            tile_size=resolved_config.tile_size,
            use_half=resolved_config.use_half,
        ),
    )

    target_width = max(1, round(source.width * resolved_config.output_scale))
    target_height = max(1, round(source.height * resolved_config.output_scale))
    partial_path = final_path.with_name(
        f".{final_path.stem}.{uuid.uuid4().hex}.partial{final_path.suffix.lower()}"
    )
    model_runtime: ModelRuntime | None = None
    try:
        run_stage("model_loading", model.load)

        def enhance() -> Frame:
            result = model.enhance(
                FrameBatch(frames=[lighting.image], start_frame=0, end_frame=1)
            )
            return resize_restored_image(result.frames[0], target_width, target_height)

        restored = run_stage("inference", enhance)
        assert isinstance(restored, np.ndarray)
        model_runtime = model.runtime
        model.unload()
        if model_runtime is None:
            raise ClearFrameError(
                ErrorCode.INFERENCE_FAILED,
                "Model runtime details were unavailable after image restoration",
            )
        _emit(
            observer,
            "image_restored",
            width=target_width,
            height=target_height,
            model=model_spec.name,
        )

        alpha = None
        if source.alpha is not None:
            alpha = cv2.resize(
                source.alpha,
                (target_width, target_height),
                interpolation=cv2.INTER_LANCZOS4,
            )
        final_path.parent.mkdir(parents=True, exist_ok=True)
        run_stage(
            "encoding",
            lambda: _encode_image(
                partial_path,
                restored,
                alpha=alpha,
                grayscale=source.grayscale,
                config=resolved_config,
            ),
        )
        validation = run_stage(
            "validating",
            lambda: _validate_output(
                partial_path,
                expected_width=target_width,
                expected_height=target_height,
                expected_alpha=source.has_alpha,
            ),
        )
        assert isinstance(validation, ImageValidationResult)
        os.replace(partial_path, final_path)
        validation = ImageValidationResult(
            width=validation.width,
            height=validation.height,
            format=validation.format,
            has_alpha=validation.has_alpha,
            file_size=validation.file_size,
        )
        _emit(observer, "image_output_validated", width=target_width, height=target_height)

        report = _build_report(
            run_id=run_id,
            started_at=started_at,
            started_counter=started_counter,
            stages=stages,
            source_path=source_path,
            source=source,
            output_path=final_path,
            validation=validation,
            lighting=lighting,
            config=resolved_config,
            model_runtime=model_runtime,
            model_name=model_spec.name,
            model_release=model_spec.release,
            model_sha256=model_spec.sha256,
        )
        write_report(report, final_report_path)
    finally:
        if model.loaded:
            model.unload()
        partial_path.unlink(missing_ok=True)

    _emit(
        observer,
        "image_pipeline_completed",
        output=str(final_path),
        report=str(final_report_path),
    )
    return ImageEnhancementResult(
        output_path=final_path,
        report_path=final_report_path,
        validation=validation,
        lighting_applied=lighting.applied,
        report=report,
    )
