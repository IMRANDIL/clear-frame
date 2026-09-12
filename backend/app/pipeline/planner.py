from __future__ import annotations

from fractions import Fraction

from backend.app.domain.video import TargetResolution, VideoMetadata

DEFAULT_TARGET_AXIS = 1080


def _round_fraction_to_even(value: Fraction) -> int:
    if value <= 0:
        raise ValueError("dimension must be positive")

    half_dimension = value / 2
    quotient, remainder = divmod(half_dimension.numerator, half_dimension.denominator)
    if remainder * 2 >= half_dimension.denominator:
        quotient += 1
    return max(2, quotient * 2)


def plan_target_resolution(
    metadata: VideoMetadata,
    target_axis: int = DEFAULT_TARGET_AXIS,
) -> TargetResolution:
    """Plan a square-pixel target while preserving the source display aspect ratio."""

    if target_axis <= 0 or target_axis % 2 != 0:
        raise ValueError("target axis must be a positive even integer")

    source_ratio = metadata.display_aspect_ratio
    if source_ratio > 1:
        width = _round_fraction_to_even(target_axis * source_ratio)
        height = target_axis
    elif source_ratio < 1:
        width = target_axis
        height = _round_fraction_to_even(target_axis / source_ratio)
    else:
        width = target_axis
        height = target_axis

    return TargetResolution(
        width=width,
        height=height,
        source_display_aspect_ratio=source_ratio,
    )
