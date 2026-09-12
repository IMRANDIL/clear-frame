from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from backend.app.domain.video import Frame

AUTO_LOW_LIGHT_MEDIAN = 100.0
TARGET_LOW_LIGHT_MEDIAN = 110.0
MIN_GAMMA = 0.55
MAX_GAMMA = 0.95


@dataclass(frozen=True, slots=True)
class LightingResult:
    image: Frame
    applied: bool
    source_median: float
    gamma: float


def adjust_low_light(image: Frame, mode: str) -> LightingResult:
    """Apply conservative gamma correction when the image is globally dark."""

    if mode not in {"auto", "force", "off"}:
        raise ValueError("lighting mode must be auto, force, or off")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("low-light adjustment requires an HxWx3 uint8 image")

    luminance = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    source_median = float(np.median(luminance))
    if mode == "off" or (mode == "auto" and source_median >= AUTO_LOW_LIGHT_MEDIAN):
        return LightingResult(image, False, source_median, 1.0)

    normalized_median = max(source_median, 8.0) / 255.0
    target = TARGET_LOW_LIGHT_MEDIAN / 255.0
    gamma = math.log(target) / math.log(normalized_median)
    gamma = min(MAX_GAMMA, max(MIN_GAMMA, gamma))
    lookup = np.rint((np.arange(256, dtype=np.float32) / 255.0) ** gamma * 255.0)
    corrected = cv2.LUT(image, lookup.astype(np.uint8))
    return LightingResult(np.ascontiguousarray(corrected), True, source_median, gamma)


def resize_restored_image(image: Frame, width: int, height: int) -> Frame:
    if width <= 0 or height <= 0:
        raise ValueError("target image dimensions must be positive")
    if (image.shape[1], image.shape[0]) == (width, height):
        return image
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)
    return np.ascontiguousarray(resized)
