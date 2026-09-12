from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from backend.app.web.api import create_app
from backend.app.web.jobs import JobManager

WEIGHTS_PATH = Path("data/models/RealESRGAN_x4plus.pth")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(not WEIGHTS_PATH.is_file(), reason="Real-ESRGAN weights are not downloaded")
@pytest.mark.model
def test_real_image_job_from_upload_to_download(tmp_path: Path) -> None:
    image = np.full((6, 8, 3), 36, dtype=np.uint8)
    image[1:5, 2:6] = (80, 100, 140)
    success, encoded = cv2.imencode(".jpg", image)
    assert success

    with TestClient(create_app(job_manager=JobManager(tmp_path / "jobs"))) as client:
        response = client.post(
            "/api/jobs",
            data={"media_type": "image", "scale": "2", "lighting": "auto"},
            files={"file": ("gloomy.jpg", encoded.tobytes(), "image/jpeg")},
        )
        assert response.status_code == 202
        job = response.json()

        for _ in range(500):
            job = client.get(f"/api/jobs/{job['id']}").json()
            if job["status"] in {"completed", "failed"}:
                break
            time.sleep(0.01)

        assert job["status"] == "completed", job.get("error_message")
        output_response = client.get(job["output_url"])
        assert output_response.status_code == 200
        output = cv2.imdecode(
            np.frombuffer(output_response.content, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        assert output is not None
        assert output.shape == (12, 16, 3)
