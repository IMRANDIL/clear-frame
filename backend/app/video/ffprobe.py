from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

from backend.app.domain.errors import ClearFrameError, ErrorCode
from backend.app.domain.video import AudioStreamMetadata, ColorMetadata, VideoMetadata

DEFAULT_PROBE_TIMEOUT_SECONDS = 30.0
MAX_ERROR_DETAIL_LENGTH = 2_000


def _optional_int(value: object) -> int | None:
    if value in {None, "", "N/A"}:
        return None
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return None
    return parsed


def _positive_int(value: object) -> int | None:
    parsed = _optional_int(value)
    return parsed if parsed is not None and parsed > 0 else None


def _positive_float(value: object) -> float | None:
    if value in {None, "", "N/A"}:
        return None
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _positive_fraction(value: object, *, separators: tuple[str, ...]) -> Fraction | None:
    if value in {None, "", "N/A", "0/0", "0:1"}:
        return None

    text = str(value)
    for separator in separators:
        if separator in text:
            numerator, denominator = text.split(separator, maxsplit=1)
            try:
                parsed = Fraction(int(numerator), int(denominator))
            except (ValueError, ZeroDivisionError):
                return None
            return parsed if parsed > 0 else None

    try:
        parsed = Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None
    return parsed if parsed > 0 else None


def _required_text(data: Mapping[str, object], key: str, label: str) -> str:
    value = data.get(key)
    if value in {None, "", "N/A"}:
        raise ValueError(f"{label} is missing")
    return str(value)


def _rotation_degrees(stream: Mapping[str, object]) -> int:
    side_data = stream.get("side_data_list")
    if isinstance(side_data, list):
        for item in side_data:
            if not isinstance(item, Mapping) or item.get("rotation") is None:
                continue
            try:
                return int(round(float(str(item["rotation"]))))
            except ValueError:
                break

    tags = stream.get("tags")
    if isinstance(tags, Mapping) and tags.get("rotate") is not None:
        try:
            return int(round(float(str(tags["rotate"]))))
        except ValueError:
            return 0
    return 0


def _stream_duration(stream: Mapping[str, object], format_data: Mapping[str, object]) -> float:
    duration = _positive_float(stream.get("duration"))
    if duration is not None:
        return duration

    duration_ts = _positive_int(stream.get("duration_ts"))
    time_base = _positive_fraction(stream.get("time_base"), separators=("/",))
    if duration_ts is not None and time_base is not None:
        return float(duration_ts * time_base)

    duration = _positive_float(format_data.get("duration"))
    if duration is None:
        raise ValueError("video duration is missing or invalid")
    return duration


def _default_audio_stream(streams: list[Mapping[str, object]]) -> Mapping[str, object] | None:
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
    for stream in audio_streams:
        if _has_disposition(stream, "default"):
            return stream
    return audio_streams[0] if audio_streams else None


def _default_video_stream(streams: list[Mapping[str, object]]) -> Mapping[str, object] | None:
    video_streams = [
        stream
        for stream in streams
        if stream.get("codec_type") == "video" and not _has_disposition(stream, "attached_pic")
    ]
    for stream in video_streams:
        if _has_disposition(stream, "default"):
            return stream
    return video_streams[0] if video_streams else None


def _has_disposition(stream: Mapping[str, object], name: str) -> bool:
    disposition = stream.get("disposition")
    return isinstance(disposition, Mapping) and disposition.get(name) in {1, "1"}


def _parse_audio(stream: Mapping[str, object] | None) -> AudioStreamMetadata | None:
    if stream is None:
        return None

    stream_index = _optional_int(stream.get("index"))
    if stream_index is None:
        raise ValueError("audio stream index is missing")

    return AudioStreamMetadata(
        stream_index=stream_index,
        codec_name=_required_text(stream, "codec_name", "audio codec name"),
        sample_rate=_positive_int(stream.get("sample_rate")),
        channels=_positive_int(stream.get("channels")),
        bitrate=_positive_int(stream.get("bit_rate")),
    )


def parse_probe_payload(
    payload: Mapping[str, object],
    *,
    input_path: Path,
    file_size: int,
) -> VideoMetadata:
    try:
        raw_streams = payload.get("streams")
        if not isinstance(raw_streams, list):
            raise ValueError("stream list is missing")
        streams = [stream for stream in raw_streams if isinstance(stream, Mapping)]

        video_stream = _default_video_stream(streams)
        if video_stream is None:
            raise ValueError("video stream is missing")

        format_data = payload.get("format")
        if not isinstance(format_data, Mapping):
            raise ValueError("container metadata is missing")

        width = _positive_int(video_stream.get("width"))
        height = _positive_int(video_stream.get("height"))
        stream_index = _optional_int(video_stream.get("index"))
        if width is None or height is None:
            raise ValueError("video dimensions are missing or invalid")
        if stream_index is None:
            raise ValueError("video stream index is missing")

        average_fps = _positive_fraction(
            video_stream.get("avg_frame_rate"),
            separators=("/",),
        )
        nominal_fps = _positive_fraction(
            video_stream.get("r_frame_rate"),
            separators=("/",),
        )
        fps = average_fps or nominal_fps
        if fps is None:
            raise ValueError("video frame rate is missing or invalid")

        sample_aspect_ratio = _positive_fraction(
            video_stream.get("sample_aspect_ratio"),
            separators=(":", "/"),
        ) or Fraction(1, 1)

        return VideoMetadata(
            input_path=input_path,
            file_size=file_size,
            width=width,
            height=height,
            fps=fps,
            nominal_fps=nominal_fps,
            duration_seconds=_stream_duration(video_stream, format_data),
            video_stream_index=stream_index,
            codec_name=_required_text(video_stream, "codec_name", "video codec name"),
            pixel_format=_required_text(video_stream, "pix_fmt", "video pixel format"),
            container_format=_required_text(format_data, "format_name", "container format"),
            frame_count=_positive_int(video_stream.get("nb_frames")),
            bitrate=_positive_int(video_stream.get("bit_rate")),
            sample_aspect_ratio=sample_aspect_ratio,
            rotation_degrees=_rotation_degrees(video_stream),
            audio=_parse_audio(_default_audio_stream(streams)),
            color=ColorMetadata(
                primaries=_optional_text(video_stream.get("color_primaries")),
                transfer=_optional_text(video_stream.get("color_transfer")),
                space=_optional_text(video_stream.get("color_space")),
                range=_optional_text(video_stream.get("color_range")),
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ClearFrameError(ErrorCode.INVALID_VIDEO, f"Invalid video metadata: {exc}") from exc


def _optional_text(value: object) -> str | None:
    return None if value in {None, "", "N/A", "unknown"} else str(value)


def probe_video(
    input_path: str | os.PathLike[str],
    *,
    ffprobe_path: str | os.PathLike[str] = "ffprobe",
    timeout_seconds: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
) -> VideoMetadata:
    path = Path(input_path).expanduser()
    if not path.is_file():
        raise ClearFrameError(ErrorCode.INVALID_VIDEO, f"Input video does not exist: {path}")

    file_size = path.stat().st_size
    if file_size <= 0:
        raise ClearFrameError(ErrorCode.INVALID_VIDEO, f"Input video is empty: {path}")
    if timeout_seconds <= 0:
        raise ValueError("FFprobe timeout must be positive")

    command = [
        os.fspath(ffprobe_path),
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-print_format",
        "json",
        os.fspath(path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise ClearFrameError(
            ErrorCode.FFPROBE_FAILED,
            f"FFprobe executable was not found: {ffprobe_path}",
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ClearFrameError(
            ErrorCode.FFPROBE_FAILED,
            f"FFprobe timed out after {timeout_seconds:g} seconds",
            retryable=True,
        ) from exc
    except OSError as exc:
        raise ClearFrameError(ErrorCode.FFPROBE_FAILED, f"Could not start FFprobe: {exc}") from exc

    if result.returncode != 0:
        detail = (result.stderr or "no FFprobe error detail").strip()
        detail = detail[-MAX_ERROR_DETAIL_LENGTH:]
        raise ClearFrameError(
            ErrorCode.FFPROBE_FAILED,
            f"FFprobe rejected the input: {detail}",
        )

    try:
        payload: Any = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ClearFrameError(
            ErrorCode.FFPROBE_FAILED,
            "FFprobe returned malformed JSON",
        ) from exc
    if not isinstance(payload, Mapping):
        raise ClearFrameError(ErrorCode.FFPROBE_FAILED, "FFprobe returned an invalid JSON root")

    return parse_probe_payload(payload, input_path=path.resolve(), file_size=file_size)
