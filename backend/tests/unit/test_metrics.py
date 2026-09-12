import json
from fractions import Fraction
from pathlib import Path

from backend.app.domain.video import ColorMetadata, TargetResolution, VideoMetadata
from backend.app.models.base import ModelRuntime
from backend.app.models.registry import load_model_spec
from backend.app.observability.metrics import MetricsRecorder, write_report
from backend.app.pipeline.audio import AudioMuxResult
from backend.app.pipeline.encoder import EncodingResult
from backend.app.pipeline.validator import OutputValidationResult, ValidationCheck


def metadata(path: Path) -> VideoMetadata:
    return VideoMetadata(
        input_path=path,
        file_size=path.stat().st_size,
        width=64,
        height=36,
        fps=Fraction(5, 1),
        duration_seconds=1,
        video_stream_index=0,
        codec_name="h264",
        pixel_format="yuv420p",
        container_format="mov,mp4",
        frame_count=5,
        color=ColorMetadata(space="bt709"),
    )


def test_metrics_report_records_reproducibility_and_performance(tmp_path: Path) -> None:
    source_path = tmp_path / "source.mp4"
    output_path = tmp_path / "output.mp4"
    source_path.write_bytes(b"source")
    output_path.write_bytes(b"output")
    source = metadata(source_path)
    output = metadata(output_path)
    recorder = MetricsRecorder()
    recorder.record_inference(5, 2.0)
    validation = OutputValidationResult(
        metadata=output,
        decoded_frames=5,
        checks=(ValidationCheck("resolution", "64x36", "64x36"),),
    )

    report = recorder.build_report(
        source=source,
        target=TargetResolution(64, 36, Fraction(16, 9)),
        model=load_model_spec("realesrgan-x4plus"),
        model_runtime=ModelRuntime("cuda:0", "fp16", 4, 256),
        encoding=EncodingResult(output_path, 5, "libx264", 18, "medium", "yuv420p"),
        audio=AudioMuxResult(output_path, "none", None, None),
        validation=validation,
        ffmpeg_version="ffmpeg test version",
    )
    report_path = tmp_path / "output.mp4.report.json"
    write_report(report, report_path)
    persisted = json.loads(report_path.read_text(encoding="utf-8"))

    assert persisted["frames_processed"] == 5
    assert persisted["inference_fps"] == 2.5
    assert persisted["model"]["sha256"] == load_model_spec("realesrgan-x4plus").sha256
    assert persisted["source"]["sha256"]
    assert persisted["output"]["sha256"]
    assert persisted["encoding"]["crf"] == 18
    assert persisted["encoding"]["preset"] == "medium"
    assert persisted["environment"]["packages"]["torch"]
