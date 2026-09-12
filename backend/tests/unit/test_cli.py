from pathlib import Path

from backend.app.config import ImagePipelineConfig, PipelineConfig
from scripts.enhance_image import build_parser as build_image_parser
from scripts.enhance_video import build_parser


def test_cli_uses_configurable_model_and_encoding_defaults() -> None:
    parser = build_parser(PipelineConfig(model_name="test-model", crf=21, tile_size=192))

    arguments = parser.parse_args(["--input", "input.mp4", "--output", "output.mp4"])

    assert arguments.input == Path("input.mp4")
    assert arguments.output == Path("output.mp4")
    assert arguments.model == "test-model"
    assert arguments.crf == 21
    assert arguments.tile == 192


def test_cli_accepts_auto_tile() -> None:
    parser = build_parser(PipelineConfig(tile_size=256))

    arguments = parser.parse_args(
        ["--input", "input.mp4", "--output", "output.mp4", "--tile", "auto"]
    )

    assert arguments.tile is None


def test_image_cli_uses_image_defaults() -> None:
    parser = build_image_parser(
        ImagePipelineConfig(output_scale=2, lighting="off", jpeg_quality=97)
    )

    arguments = parser.parse_args(["--input", "input.png", "--output", "output.jpg"])

    assert arguments.input == Path("input.png")
    assert arguments.output == Path("output.jpg")
    assert arguments.scale == 2
    assert arguments.lighting == "off"
    assert arguments.jpeg_quality == 97
