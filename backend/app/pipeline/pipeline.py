from __future__ import annotations

import os
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from backend.app.config import PipelineConfig
from backend.app.domain.errors import ClearFrameError, ErrorCode
from backend.app.models.base import ModelRuntime
from backend.app.models.realesrgan import RealESRGANConfig, RealESRGANModel
from backend.app.models.registry import ensure_model_weights, load_model_spec
from backend.app.observability.metrics import MetricsRecorder, write_report
from backend.app.pipeline.audio import AudioMuxConfig, AudioMuxResult, mux_source_audio
from backend.app.pipeline.decoder import DecoderConfig, decode_frame_batches
from backend.app.pipeline.encoder import EncoderConfig, EncodingResult, encode_video
from backend.app.pipeline.planner import plan_target_resolution
from backend.app.pipeline.processor import restore_frame_batches
from backend.app.pipeline.validator import (
    OutputValidationResult,
    ValidationConfig,
    validate_output,
)
from backend.app.video.ffmpeg import get_ffmpeg_version
from backend.app.video.ffprobe import probe_video

type ProgressObserver = Callable[[str, Mapping[str, object]], None]

HDR_TRANSFER_CHARACTERISTICS = frozenset({"smpte2084", "arib-std-b67"})


@dataclass(frozen=True, slots=True)
class PipelineResult:
    output_path: Path
    report_path: Path
    encoding: EncodingResult
    audio: AudioMuxResult
    validation: OutputValidationResult
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
    if output_path.suffix.lower() != ".mp4":
        raise ClearFrameError(ErrorCode.ENCODE_FAILED, "Output path must use the .mp4 extension")
    if input_path == output_path:
        raise ClearFrameError(ErrorCode.INVALID_VIDEO, "Input and output paths must be different")
    if report_path in {input_path, output_path}:
        raise ClearFrameError(ErrorCode.ENCODE_FAILED, "Report path must differ from video paths")
    if not overwrite and output_path.exists():
        raise ClearFrameError(ErrorCode.ENCODE_FAILED, f"Output already exists: {output_path}")
    if not overwrite and report_path.exists():
        raise ClearFrameError(ErrorCode.ENCODE_FAILED, f"Report already exists: {report_path}")


def run_pipeline(
    input_path: Path,
    output_path: Path,
    *,
    config: PipelineConfig | None = None,
    report_path: Path | None = None,
    observer: ProgressObserver | None = None,
) -> PipelineResult:
    resolved_config = config or PipelineConfig.from_environment()
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

    metrics = MetricsRecorder()
    _emit(observer, "pipeline_started", run_id=metrics.run_id, input=str(source_path))
    with metrics.stage("analyzing"):
        source = probe_video(source_path, ffprobe_path=resolved_config.ffprobe_path)
    _emit(
        observer,
        "source_analyzed",
        width=source.width,
        height=source.height,
        fps=str(source.fps),
        frames_total=source.frame_count,
        audio_present=source.audio_present,
    )

    if source.color.transfer in HDR_TRANSFER_CHARACTERISTICS:
        raise ClearFrameError(
            ErrorCode.INVALID_VIDEO,
            f"HDR transfer {source.color.transfer} is not supported by the SDR baseline",
        )
    if source.is_variable_frame_rate:
        _emit(
            observer,
            "variable_frame_rate_detected",
            average_fps=str(source.fps),
            nominal_fps=str(source.nominal_fps),
        )

    with metrics.stage("planning"):
        target = plan_target_resolution(source, target_axis=resolved_config.target_axis)
        model_spec = load_model_spec(resolved_config.model_name)
    _emit(
        observer,
        "restoration_planned",
        target_width=target.width,
        target_height=target.height,
        model=model_spec.name,
        crf=resolved_config.crf,
    )

    partial_path = final_path.with_name(f".{final_path.stem}.{uuid.uuid4().hex}.partial.mp4")
    with metrics.stage("model_preparation"):
        weights_path = ensure_model_weights(model_spec, resolved_config.model_directory)
        model = RealESRGANModel(
            model_spec,
            RealESRGANConfig(
                weights_path=weights_path,
                device=resolved_config.device,
                tile_size=resolved_config.tile_size,
                use_half=resolved_config.use_half,
            ),
        )
    model_runtime: ModelRuntime | None = None
    frames_processed = 0
    last_progress_emitted = 0.0

    def observe_inference(frame_count: int, seconds: float) -> None:
        nonlocal frames_processed, last_progress_emitted
        frames_processed += frame_count
        metrics.record_inference(frame_count, seconds)
        now = time.monotonic()
        if (
            now - last_progress_emitted < 1.0
            and frames_processed != source.frame_count
        ):
            return
        last_progress_emitted = now
        _emit(
            observer,
            "frames_restored",
            frames_processed=frames_processed,
            frames_total=source.frame_count,
            batch_inference_fps=frame_count / seconds if seconds > 0 else None,
        )

    try:
        with metrics.stage("model_loading"):
            model.load()
        final_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_config.temporary_directory.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(
            prefix=f"clearframe-{metrics.run_id[:8]}-",
            dir=resolved_config.temporary_directory,
        ) as temporary:
            video_only_path = Path(temporary) / "restored-video-only.mp4"
            try:
                with metrics.stage("decode_restore_encode"):
                    restored_batches = restore_frame_batches(
                        decode_frame_batches(
                            source,
                            config=DecoderConfig(
                                ffmpeg_path=resolved_config.ffmpeg_path,
                                batch_size=resolved_config.batch_size,
                            ),
                        ),
                        metadata=source,
                        target=target,
                        model=model,
                        inference_observer=observe_inference,
                    )
                    encoding = encode_video(
                        restored_batches,
                        output_path=video_only_path,
                        target=target,
                        fps=source.fps,
                        color=source.color,
                        config=EncoderConfig(
                            ffmpeg_path=resolved_config.ffmpeg_path,
                            codec=resolved_config.video_codec,
                            crf=resolved_config.crf,
                            preset=resolved_config.encoder_preset,
                            pixel_format=resolved_config.pixel_format,
                        ),
                    )
                model_runtime = model.runtime
            finally:
                model.unload()

            if model_runtime is None:
                raise ClearFrameError(
                    ErrorCode.INFERENCE_FAILED,
                    "Model runtime details were unavailable after restoration",
                )

            _emit(observer, "video_encoded", frames_encoded=encoding.frames_encoded)
            with metrics.stage("muxing_audio"):
                audio = mux_source_audio(
                    video_only_path,
                    source=source,
                    output_path=partial_path,
                    config=AudioMuxConfig(
                        ffmpeg_path=resolved_config.ffmpeg_path,
                        aac_bitrate_kbps=resolved_config.aac_bitrate_kbps,
                    ),
                )
            _emit(observer, "audio_muxed", mode=audio.audio_mode)

            with metrics.stage("validating"):
                validation = validate_output(
                    partial_path,
                    source=source,
                    target=target,
                    expected_frames=encoding.frames_encoded,
                    config=ValidationConfig(
                        ffmpeg_path=resolved_config.ffmpeg_path,
                        ffprobe_path=resolved_config.ffprobe_path,
                    ),
                )
            _emit(observer, "output_validated", decoded_frames=validation.decoded_frames)

            with metrics.stage("finalizing"):
                os.replace(partial_path, final_path)
                validation = replace(
                    validation,
                    metadata=replace(validation.metadata, input_path=final_path),
                )

            finalized_audio = replace(audio, output_path=final_path)
            report = metrics.build_report(
                source=source,
                target=target,
                model=model_spec,
                model_runtime=model_runtime,
                encoding=encoding,
                audio=finalized_audio,
                validation=validation,
                ffmpeg_version=get_ffmpeg_version(resolved_config.ffmpeg_path),
            )
            write_report(report, final_report_path)
    finally:
        if model.loaded:
            model.unload()
        partial_path.unlink(missing_ok=True)

    _emit(
        observer,
        "pipeline_completed",
        output=str(final_path),
        report=str(final_report_path),
        frames_processed=frames_processed,
    )
    return PipelineResult(
        output_path=final_path,
        report_path=final_report_path,
        encoding=encoding,
        audio=finalized_audio,
        validation=validation,
        report=report,
    )
