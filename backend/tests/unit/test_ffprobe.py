from fractions import Fraction
from pathlib import Path

import pytest

from backend.app.domain.errors import ClearFrameError, ErrorCode
from backend.app.video.ffprobe import parse_probe_payload, probe_video


def probe_payload() -> dict[str, object]:
    return {
        "streams": [
            {
                "index": 0,
                "codec_name": "h264",
                "codec_type": "video",
                "width": 720,
                "height": 576,
                "pix_fmt": "yuv420p",
                "r_frame_rate": "30/1",
                "avg_frame_rate": "24000/1001",
                "time_base": "1/24000",
                "duration_ts": "240240",
                "bit_rate": "750000",
                "nb_frames": "240",
                "sample_aspect_ratio": "16:15",
                "color_range": "tv",
                "color_space": "smpte170m",
                "color_transfer": "bt709",
                "color_primaries": "bt709",
                "side_data_list": [{"side_data_type": "Display Matrix", "rotation": -90}],
            },
            {
                "index": 1,
                "codec_name": "mp3",
                "codec_type": "audio",
                "sample_rate": "44100",
                "channels": 2,
                "disposition": {"default": 0},
            },
            {
                "index": 2,
                "codec_name": "aac",
                "codec_type": "audio",
                "sample_rate": "48000",
                "channels": 2,
                "bit_rate": "128000",
                "disposition": {"default": 1},
            },
        ],
        "format": {
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "duration": "10.01",
        },
    }


def test_parse_probe_payload_preserves_exact_metadata() -> None:
    metadata = parse_probe_payload(
        probe_payload(),
        input_path=Path("source.mp4"),
        file_size=12345,
    )

    assert metadata.width == 720
    assert metadata.height == 576
    assert metadata.fps == Fraction(24000, 1001)
    assert metadata.nominal_fps == Fraction(30, 1)
    assert metadata.is_variable_frame_rate
    assert metadata.duration_seconds == pytest.approx(10.01)
    assert metadata.frame_count == 240
    assert metadata.sample_aspect_ratio == Fraction(16, 15)
    assert metadata.rotation_degrees == 270
    assert metadata.display_aspect_ratio == Fraction(3, 4)
    assert metadata.audio is not None
    assert metadata.audio.stream_index == 2
    assert metadata.audio.codec_name == "aac"
    assert metadata.audio.sample_rate == 48000
    assert metadata.color.space == "smpte170m"


def test_stream_duration_uses_time_base_before_container_duration() -> None:
    payload = probe_payload()
    metadata = parse_probe_payload(
        payload,
        input_path=Path("source.mp4"),
        file_size=12345,
    )

    assert metadata.duration_seconds == pytest.approx(10.01)


def test_missing_sample_aspect_ratio_defaults_to_square_pixels() -> None:
    payload = probe_payload()
    video_stream = payload["streams"][0]  # type: ignore[index]
    video_stream.pop("sample_aspect_ratio")  # type: ignore[union-attr]

    metadata = parse_probe_payload(
        payload,
        input_path=Path("source.mp4"),
        file_size=12345,
    )

    assert metadata.sample_aspect_ratio == Fraction(1, 1)


def test_attached_picture_is_not_selected_as_video() -> None:
    payload = probe_payload()
    attached_picture = {
        "index": 3,
        "codec_name": "mjpeg",
        "codec_type": "video",
        "width": 600,
        "height": 600,
        "pix_fmt": "yuvj420p",
        "avg_frame_rate": "1/1",
        "disposition": {"attached_pic": 1},
    }
    payload["streams"].insert(0, attached_picture)  # type: ignore[union-attr]

    metadata = parse_probe_payload(
        payload,
        input_path=Path("source.mp4"),
        file_size=12345,
    )

    assert metadata.video_stream_index == 0
    assert (metadata.width, metadata.height) == (720, 576)


@pytest.mark.parametrize("streams", [[], [{"codec_type": "audio", "index": 0}]])
def test_missing_video_stream_is_invalid(streams: list[dict[str, object]]) -> None:
    payload = probe_payload()
    payload["streams"] = streams

    with pytest.raises(ClearFrameError) as caught:
        parse_probe_payload(payload, input_path=Path("source.mp4"), file_size=1)

    assert caught.value.code is ErrorCode.INVALID_VIDEO


def test_probe_rejects_missing_file_before_starting_process(tmp_path: Path) -> None:
    with pytest.raises(ClearFrameError) as caught:
        probe_video(tmp_path / "missing.mp4")

    assert caught.value.code is ErrorCode.INVALID_VIDEO


def test_probe_reports_missing_executable(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"not empty")

    with pytest.raises(ClearFrameError) as caught:
        probe_video(source, ffprobe_path=tmp_path / "missing-ffprobe.exe")

    assert caught.value.code is ErrorCode.FFPROBE_FAILED
