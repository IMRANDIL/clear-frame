from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.app.config import ImagePipelineConfig
from backend.app.domain.errors import ClearFrameError, ErrorCode
from backend.app.image.pipeline import decode_image, run_image_pipeline


def write_image(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(path.suffix, image)
    assert success
    path.write_bytes(encoded.tobytes())


def test_decode_image_preserves_png_alpha(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    image = np.zeros((6, 8, 4), dtype=np.uint8)
    image[:, :, :3] = (10, 20, 30)
    image[:, :, 3] = np.arange(8, dtype=np.uint8) * 30
    write_image(source, image)

    decoded = decode_image(source)

    assert (decoded.width, decoded.height) == (8, 6)
    assert decoded.has_alpha
    assert decoded.alpha is not None
    assert np.array_equal(decoded.alpha, image[:, :, 3])
    assert np.array_equal(decoded.pixels, image[:, :, :3])


def test_decode_image_normalizes_grayscale_to_model_input(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    image = np.arange(48, dtype=np.uint8).reshape(6, 8)
    write_image(source, image)

    decoded = decode_image(source)

    assert decoded.grayscale
    assert decoded.pixels.shape == (6, 8, 3)
    assert np.array_equal(decoded.pixels[:, :, 0], image)


def test_corrupt_image_is_rejected_before_model_loading(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    source.write_bytes(b"not an image")

    with pytest.raises(ClearFrameError) as caught:
        run_image_pipeline(source, tmp_path / "output.jpg")

    assert caught.value.code is ErrorCode.IMAGE_DECODE_FAILED


def test_transparent_png_cannot_silently_lose_alpha(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    image = np.full((4, 4, 4), 127, dtype=np.uint8)
    write_image(source, image)

    with pytest.raises(ClearFrameError, match="transparency") as caught:
        run_image_pipeline(source, tmp_path / "output.jpg")

    assert caught.value.code is ErrorCode.IMAGE_ENCODE_FAILED


def test_existing_output_is_not_overwritten(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    output = tmp_path / "output.jpg"
    output.write_bytes(b"existing")

    with pytest.raises(ClearFrameError) as caught:
        run_image_pipeline(source, output, config=ImagePipelineConfig())

    assert caught.value.code is ErrorCode.IMAGE_ENCODE_FAILED
    assert output.read_bytes() == b"existing"
