from pathlib import Path

import pytest

from backend.app.config import ImagePipelineConfig, PipelineConfig, parse_tile_size


@pytest.mark.parametrize(("value", "expected"), [("auto", None), ("256", 256), ("0", 0)])
def test_parse_tile_size(value: str, expected: int | None) -> None:
    assert parse_tile_size(value) == expected


@pytest.mark.parametrize("value", ["bad", "-1"])
def test_parse_tile_size_rejects_invalid_value(value: str) -> None:
    with pytest.raises(ValueError, match="tile size"):
        parse_tile_size(value)


def test_environment_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLEARFRAME_MODEL", "benchmark-model")
    monkeypatch.setenv("CLEARFRAME_CRF", "20")
    monkeypatch.setenv("CLEARFRAME_MODEL_DIR", "custom-models")

    config = PipelineConfig.from_environment()

    assert config.model_name == "benchmark-model"
    assert config.crf == 20
    assert config.model_directory == Path("custom-models")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_axis": 1079},
        {"tile_size": -1},
        {"batch_size": 0},
        {"batch_size": 9},
        {"crf": 52},
        {"aac_bitrate_kbps": 0},
    ],
)
def test_pipeline_configuration_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        PipelineConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"output_scale": 0.5},
        {"output_scale": 4.1},
        {"lighting": "aggressive"},
        {"jpeg_quality": 0},
        {"jpeg_quality": 101},
        {"png_compression": -1},
        {"png_compression": 10},
    ],
)
def test_image_configuration_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ImagePipelineConfig(**kwargs)  # type: ignore[arg-type]
