import cv2
import numpy as np
import pytest

from backend.app.image.processing import adjust_low_light, resize_restored_image


def test_auto_lighting_brightens_a_dark_image() -> None:
    image = np.full((8, 12, 3), 32, dtype=np.uint8)

    result = adjust_low_light(image, "auto")

    assert result.applied
    assert result.source_median == 32
    assert 0.55 <= result.gamma < 1
    assert float(result.image.mean()) > float(image.mean())


def test_auto_lighting_leaves_a_bright_image_unchanged() -> None:
    image = np.full((8, 12, 3), 180, dtype=np.uint8)

    result = adjust_low_light(image, "auto")

    assert not result.applied
    assert result.gamma == 1
    assert np.array_equal(result.image, image)


def test_lighting_can_be_disabled() -> None:
    image = np.full((8, 12, 3), 16, dtype=np.uint8)

    result = adjust_low_light(image, "off")

    assert not result.applied
    assert np.array_equal(result.image, image)


def test_lighting_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError, match="lighting mode"):
        adjust_low_light(np.zeros((2, 2, 3), dtype=np.uint8), "aggressive")


def test_resize_restored_image_uses_exact_target_dimensions() -> None:
    image = np.zeros((16, 24, 3), dtype=np.uint8)
    image[4:12, 6:18] = (20, 80, 160)

    resized = resize_restored_image(image, 12, 8)

    assert resized.shape == (8, 12, 3)
    assert resized.dtype == np.uint8
    assert resized.flags.c_contiguous
    assert cv2.countNonZero(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)) > 0
