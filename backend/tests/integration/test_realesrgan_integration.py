from pathlib import Path

import numpy as np
import pytest
import torch

from backend.app.domain.video import FrameBatch
from backend.app.models.realesrgan import RealESRGANConfig, RealESRGANModel
from backend.app.models.registry import load_model_spec

WEIGHTS_PATH = Path("data/models/RealESRGAN_x4plus.pth")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(not WEIGHTS_PATH.is_file(), reason="Real-ESRGAN weights are not downloaded")
@pytest.mark.model
def test_realesrgan_loads_once_and_enhances_on_cuda() -> None:
    spec = load_model_spec("realesrgan-x4plus")
    model = RealESRGANModel(
        spec,
        RealESRGANConfig(weights_path=WEIGHTS_PATH, device="cuda:0", tile_size=16),
    )
    frame = np.full((16, 16, 3), 127, dtype=np.uint8)

    model.load()
    loaded_network = model._network
    model.load()
    output = model.enhance(FrameBatch(frames=[frame], start_frame=0, end_frame=1))

    assert model._network is loaded_network
    assert model.runtime is not None
    assert model.runtime.device == "cuda:0"
    assert model.runtime.precision == "fp16"
    assert output.frames[0].shape == (64, 64, 3)
    assert output.frames[0].dtype == np.uint8

    model.unload()
    assert not model.loaded
