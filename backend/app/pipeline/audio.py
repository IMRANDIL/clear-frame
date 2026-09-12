from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from backend.app.domain.errors import ClearFrameError, ErrorCode
from backend.app.domain.video import VideoMetadata
from backend.app.video.ffmpeg import FFmpegProcess

AUDIO_COPY_CODECS = frozenset({"aac"})
PROCESS_EXIT_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True, slots=True)
class AudioMuxConfig:
    ffmpeg_path: str | os.PathLike[str] = "ffmpeg"
    aac_bitrate_kbps: int = 192

    def __post_init__(self) -> None:
        if self.aac_bitrate_kbps <= 0:
            raise ValueError("AAC bitrate must be positive")


@dataclass(frozen=True, slots=True)
class AudioMuxResult:
    output_path: Path
    audio_mode: str
    source_audio_codec: str | None
    output_audio_codec: str | None


def mux_source_audio(
    video_only_path: Path,
    *,
    source: VideoMetadata,
    output_path: Path,
    config: AudioMuxConfig | None = None,
) -> AudioMuxResult:
    """Mux the selected source audio stream, transcoding incompatible audio to AAC."""

    resolved_config = config or AudioMuxConfig()
    if not video_only_path.is_file():
        raise ClearFrameError(
            ErrorCode.AUDIO_MUX_FAILED,
            f"Encoded video stream does not exist: {video_only_path}",
        )
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("audio mux output must use the .mp4 extension")
    if output_path.exists():
        raise ClearFrameError(ErrorCode.AUDIO_MUX_FAILED, f"Output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arguments: list[str | os.PathLike[str]] = [
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        video_only_path,
    ]
    source_codec = source.audio.codec_name if source.audio is not None else None
    if source.audio is None:
        audio_mode = "none"
        output_codec = None
        arguments.extend(["-map", "0:v:0", "-c:v", "copy"])
    else:
        arguments.extend(
            [
                "-i",
                source.input_path,
                "-map",
                "0:v:0",
                "-map",
                f"1:{source.audio.stream_index}",
                "-c:v",
                "copy",
            ]
        )
        if source.audio.codec_name in AUDIO_COPY_CODECS:
            audio_mode = "copy"
            output_codec = source.audio.codec_name
            arguments.extend(["-c:a", "copy"])
        else:
            audio_mode = "transcode"
            output_codec = "aac"
            arguments.extend(
                ["-c:a", "aac", "-b:a", f"{resolved_config.aac_bitrate_kbps}k"]
            )
        arguments.append("-shortest")

    arguments.extend(
        [
            "-map_metadata",
            "-1",
            "-movflags",
            "+faststart",
            "-n",
            output_path,
        ]
    )
    process = FFmpegProcess(
        arguments,
        ffmpeg_path=resolved_config.ffmpeg_path,
        error_code=ErrorCode.AUDIO_MUX_FAILED,
    )
    succeeded = False
    try:
        return_code = process.wait(timeout_seconds=PROCESS_EXIT_TIMEOUT_SECONDS)
        if return_code != 0:
            raise ClearFrameError(
                ErrorCode.AUDIO_MUX_FAILED,
                f"FFmpeg audio mux failed: {process.stderr_detail}",
            )
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise ClearFrameError(
                ErrorCode.AUDIO_MUX_FAILED,
                "FFmpeg exited successfully but did not create a non-empty muxed video",
            )
        succeeded = True
    finally:
        if not succeeded:
            process.terminate()
        process.close()
        if not succeeded:
            output_path.unlink(missing_ok=True)

    return AudioMuxResult(
        output_path=output_path,
        audio_mode=audio_mode,
        source_audio_codec=source_codec,
        output_audio_codec=output_codec,
    )
