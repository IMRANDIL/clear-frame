from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from backend.app.domain.video import TargetResolution
from backend.app.models.realesrgan import RealESRGANConfig, RealESRGANModel
from backend.app.models.registry import load_model_spec
from backend.app.pipeline.decoder import decode_frame_batches
from backend.app.pipeline.processor import restore_frame_batches
from backend.app.video.ffprobe import probe_video

WEIGHTS_PATH = Path("data/models/RealESRGAN_x4plus.pth")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(not WEIGHTS_PATH.is_file(), reason="Real-ESRGAN weights are not downloaded")
@pytest.mark.model
def test_ffmpeg_frames_flow_through_realesrgan(tmp_path: Path) -> None:
    fixture = tmp_path / "single-frame.mp4"
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=16x16:rate=1",
            "-frames:v",
            "1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(fixture),
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr

    metadata = probe_video(fixture)
    target = TargetResolution(64, 64, metadata.display_aspect_ratio)
    model = RealESRGANModel(
        load_model_spec("realesrgan-x4plus"),
        RealESRGANConfig(weights_path=WEIGHTS_PATH, device="cuda:0", tile_size=16),
    )

    with model:
        restored = list(
            restore_frame_batches(
                decode_frame_batches(metadata),
                metadata=metadata,
                target=target,
                model=model,
            )
        )

    assert len(restored) == 1
    assert restored[0].frames[0].shape == (64, 64, 3)
