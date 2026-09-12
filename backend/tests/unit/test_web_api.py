from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.web.api import create_app
from backend.app.web.jobs import JobManager


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    manager = JobManager(tmp_path / "jobs")
    with TestClient(create_app(job_manager=manager)) as test_client:
        yield test_client


def jpeg_bytes() -> bytes:
    image = np.full((6, 8, 3), 48, dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    assert success
    return encoded.tobytes()


def test_health_reports_local_runtime(client: TestClient) -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert isinstance(response.json()["cuda_available"], bool)


def test_upload_rejects_unsupported_media(client: TestClient) -> None:
    response = client.post(
        "/api/jobs",
        data={"media_type": "image"},
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 415
    assert "PNG" in response.json()["detail"]


def test_image_job_lifecycle_and_downloads(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_pipeline(
        input_path: Path,
        output_path: Path,
        *,
        report_path: Path,
        observer,
        **_kwargs: object,
    ) -> None:
        observer("image_analyzed", {"width": 8, "height": 6})
        output_path.write_bytes(input_path.read_bytes())
        report_path.write_text(json.dumps({"validation": True}), encoding="utf-8")
        observer("image_output_validated", {"width": 8, "height": 6})

    monkeypatch.setattr("backend.app.web.jobs.run_image_pipeline", fake_pipeline)
    created = client.post(
        "/api/jobs",
        data={"media_type": "image", "scale": "2", "lighting": "auto"},
        files={"file": ("night.jpg", jpeg_bytes(), "image/jpeg")},
    )

    assert created.status_code == 202
    job = created.json()
    for _ in range(100):
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["status"] == "completed":
            break
        time.sleep(0.01)

    assert job["status"] == "completed"
    assert job["progress"] == 1
    assert client.get(job["source_url"]).content == jpeg_bytes()
    output = client.get(job["output_url"])
    assert output.status_code == 200
    assert "night-enhanced.jpg" in output.headers["content-disposition"]
    assert client.get(job["report_url"]).json() == {"validation": True}

    deleted = client.delete(f"/api/jobs/{job['id']}")
    assert deleted.status_code == 204
    assert client.get(f"/api/jobs/{job['id']}").status_code == 404
