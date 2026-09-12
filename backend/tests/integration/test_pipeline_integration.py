from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from backend.app.config import PipelineConfig
from backend.app.domain.errors import ClearFrameError, ErrorCode
from backend.app.pipeline.pipeline import run_pipeline

WEIGHTS_DIRECTORY = Path("data/models").resolve()
WEIGHTS_PATH = WEIGHTS_DIRECTORY / "RealESRGAN_x4plus.pth"


@dataclass(frozen=True, slots=True)
class PipelineCase:
    name: str
    suffix: str
    width: int
    height: int
    video_codec: str
    audio_codec: str | None
    target_width: int
    target_height: int
    audio_mode: str


CASES = (
    PipelineCase("landscape-aac", ".mp4", 32, 16, "libx264", "aac", 128, 64, "copy"),
    PipelineCase("portrait-silent", ".mov", 16, 32, "libx264", None, 64, 128, "none"),
    PipelineCase("square-opus", ".webm", 16, 16, "libvpx-vp9", "libopus", 64, 64, "transcode"),
)


def generate_source(path: Path, case: PipelineCase) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={case.width}x{case.height}:rate=2",
    ]
    if case.audio_codec is not None:
        command.extend(
            [
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=1000:sample_rate=48000",
            ]
        )
    command.extend(
        [
            "-t",
            "1",
            "-c:v",
            case.video_codec,
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-colorspace",
            "bt709",
        ]
    )
    if case.video_codec == "libx264":
        command.extend(["-pix_fmt", "yuv420p"])
    if case.audio_codec is not None:
        command.extend(["-c:a", case.audio_codec, "-shortest"])
    command.append(str(path))

    result = subprocess.run(
        command,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def pipeline_config(tmp_path: Path, **overrides: object) -> PipelineConfig:
    values: dict[str, object] = {
        "model_directory": WEIGHTS_DIRECTORY,
        "temporary_directory": tmp_path / "temp",
        "target_axis": 64,
        "device": "cuda:0",
        "tile_size": 64,
        "encoder_preset": "ultrafast",
    }
    values.update(overrides)
    return PipelineConfig(**values)  # type: ignore[arg-type]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(not WEIGHTS_PATH.is_file(), reason="Real-ESRGAN weights are not downloaded")
@pytest.mark.model
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_complete_pipeline_matrix(tmp_path: Path, case: PipelineCase) -> None:
    source = tmp_path / f"source{case.suffix}"
    output = tmp_path / f"{case.name}.mp4"
    events: list[tuple[str, dict[str, object]]] = []
    generate_source(source, case)

    result = run_pipeline(
        source,
        output,
        config=pipeline_config(tmp_path),
        observer=lambda event, fields: events.append((event, dict(fields))),
    )

    assert result.output_path == output.resolve()
    assert result.output_path.is_file()
    assert result.report_path.is_file()
    assert (result.validation.metadata.width, result.validation.metadata.height) == (
        case.target_width,
        case.target_height,
    )
    assert result.validation.metadata.frame_count == 2
    assert result.validation.decoded_frames == 2
    assert result.audio.audio_mode == case.audio_mode
    assert result.validation.metadata.audio_present == (case.audio_codec is not None)
    if case.audio_codec is not None:
        assert result.validation.metadata.audio is not None
        assert result.validation.metadata.audio.codec_name == "aac"

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["frames_processed"] == 2
    assert report["target"]["width"] == case.target_width
    assert report["target"]["height"] == case.target_height
    assert report["validation"]["decoded_frames"] == 2
    assert report["output"]["sha256"]
    assert {event for event, _ in events} >= {
        "pipeline_started",
        "frames_restored",
        "output_validated",
        "pipeline_completed",
    }
    assert not list(output.parent.glob(f".{output.stem}.*.partial.mp4"))
    assert not list((tmp_path / "temp").iterdir())


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="FFprobe is not installed")
def test_corrupt_input_leaves_no_output_or_report(tmp_path: Path) -> None:
    source = tmp_path / "corrupt.mp4"
    output = tmp_path / "output.mp4"
    source.write_bytes(b"not a video")

    with pytest.raises(ClearFrameError) as caught:
        run_pipeline(source, output, config=pipeline_config(tmp_path, device="cpu"))

    assert caught.value.code is ErrorCode.FFPROBE_FAILED
    assert not output.exists()
    assert not Path(f"{output.resolve()}.report.json").exists()
    assert not list(tmp_path.glob(f".{output.stem}.*.partial.mp4"))


def test_existing_output_is_not_overwritten_without_permission(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    report = Path(f"{output.resolve()}.report.json")
    output.write_bytes(b"existing output")
    report.write_bytes(b"existing report")

    with pytest.raises(ClearFrameError) as caught:
        run_pipeline(source, output, config=pipeline_config(tmp_path, device="cpu"))

    assert caught.value.code is ErrorCode.ENCODE_FAILED
    assert output.read_bytes() == b"existing output"
    assert report.read_bytes() == b"existing report"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
@pytest.mark.skipif(not WEIGHTS_PATH.is_file(), reason="Real-ESRGAN weights are not downloaded")
@pytest.mark.model
def test_downstream_failure_preserves_existing_overwrite_targets(tmp_path: Path) -> None:
    case = CASES[1]
    source = tmp_path / "source.mov"
    output = tmp_path / "output.mp4"
    report = Path(f"{output.resolve()}.report.json")
    generate_source(source, case)
    output.write_bytes(b"existing output")
    report.write_bytes(b"existing report")

    with pytest.raises(ClearFrameError) as caught:
        run_pipeline(
            source,
            output,
            config=pipeline_config(
                tmp_path,
                device="cpu",
                ffmpeg_path=str(tmp_path / "missing-ffmpeg.exe"),
                overwrite=True,
            ),
        )

    assert caught.value.code is ErrorCode.ENCODE_FAILED
    assert output.read_bytes() == b"existing output"
    assert report.read_bytes() == b"existing report"
    assert not list(tmp_path.glob(f".{output.stem}.*.partial.mp4"))
    assert not list((tmp_path / "temp").iterdir())
