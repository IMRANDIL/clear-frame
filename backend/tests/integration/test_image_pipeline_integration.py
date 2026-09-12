from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from backend.app.config import ImagePipelineConfig
from backend.app.image.pipeline import run_image_pipeline

WEIGHTS_DIRECTORY = Path("data/models").resolve()
WEIGHTS_PATH = WEIGHTS_DIRECTORY / "RealESRGAN_x4plus.pth"


@dataclass(frozen=True, slots=True)
class ImageCase:
    suffix: str
    alpha: bool


CASES = (
    ImageCase(".png", True),
    ImageCase(".jpg", False),
    ImageCase(".jpeg", False),
)


def generate_source(path: Path, *, alpha: bool) -> None:
    channels = 4 if alpha else 3
    image = np.zeros((6, 8, channels), dtype=np.uint8)
    image[:, :, :3] = (24, 36, 48)
    image[1:5, 2:6, :3] = (60, 90, 120)
    if alpha:
        image[:, :, 3] = 192
    encode_suffix = ".jpg" if path.suffix == ".jpeg" else path.suffix
    success, encoded = cv2.imencode(encode_suffix, image)
    assert success
    path.write_bytes(encoded.tobytes())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(not WEIGHTS_PATH.is_file(), reason="Real-ESRGAN weights are not downloaded")
@pytest.mark.model
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.suffix.lstrip("."))
def test_complete_image_pipeline_matrix(tmp_path: Path, case: ImageCase) -> None:
    source = tmp_path / f"source{case.suffix}"
    output = tmp_path / f"enhanced{case.suffix}"
    events: list[str] = []
    generate_source(source, alpha=case.alpha)

    result = run_image_pipeline(
        source,
        output,
        config=ImagePipelineConfig(
            model_directory=WEIGHTS_DIRECTORY,
            device="cuda:0",
            tile_size=16,
            output_scale=2,
        ),
        observer=lambda event, _fields: events.append(event),
    )

    assert result.output_path == output.resolve()
    assert result.output_path.is_file()
    assert result.report_path.is_file()
    assert (result.validation.width, result.validation.height) == (16, 12)
    assert result.validation.has_alpha is case.alpha
    assert result.lighting_applied
    assert set(events) >= {
        "image_pipeline_started",
        "image_analyzed",
        "lighting_adjusted",
        "image_restored",
        "image_output_validated",
        "image_pipeline_completed",
    }
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["source"]["sha256"]
    assert report["output"]["sha256"]
    assert report["output"]["width"] == 16
    assert report["processing"]["lighting_applied"] is True
    assert report["validation"]["decoded"] is True
    assert not list(tmp_path.glob(f".{output.stem}.*.partial{case.suffix}"))
