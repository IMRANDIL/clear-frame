from pathlib import Path

import pytest

from backend.app.models.realesrgan import RealESRGANConfig


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tile_size": -1},
        {"tile_fallbacks": ()},
        {"tile_fallbacks": (256, 0)},
        {"tile_padding": -1},
        {"pre_padding": -1},
    ],
)
def test_invalid_model_configuration_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RealESRGANConfig(weights_path=Path("weights.pth"), **kwargs)  # type: ignore[arg-type]
