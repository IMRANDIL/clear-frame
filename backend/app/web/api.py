from __future__ import annotations

import os
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.config import ImagePipelineConfig
from backend.app.web.jobs import IMAGE_SUFFIXES, VIDEO_SUFFIXES, JobManager, JobOptions

MAX_UPLOAD_BYTES = 2 * 1024**3
UPLOAD_CHUNK_SIZE = 1024 * 1024
FRONTEND_DIRECTORY = Path(__file__).resolve().parents[3] / "frontend" / "dist"


def _manager(request: Request) -> JobManager:
    return request.app.state.job_manager


def _job_or_404(request: Request, job_id: str):
    job = _manager(request).get(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return job


async def _persist_upload(upload: UploadFile, suffix: str) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(prefix="clearframe-upload-", suffix=suffix)
    size = 0
    try:
        with os.fdopen(descriptor, "wb") as target:
            while chunk := await upload.read(UPLOAD_CHUNK_SIZE):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail="Upload exceeds the 2 GiB local limit",
                    )
                target.write(chunk)
        if size == 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Uploaded file is empty",
            )
        return Path(temporary_name)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    finally:
        await upload.close()


def create_app(*, job_manager: JobManager | None = None) -> FastAPI:
    manager = job_manager or JobManager(Path(os.getenv("CLEARFRAME_JOB_DIR", "data/jobs")))

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.job_manager = manager
        yield
        manager.close()

    application = FastAPI(
        title="ClearFrame Local API",
        version="0.2.0",
        lifespan=lifespan,
    )
    application.state.job_manager = manager

    @application.get("/api/health")
    def health() -> dict[str, object]:
        return {
            "status": "ready",
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }

    @application.post("/api/jobs", status_code=status.HTTP_202_ACCEPTED)
    async def create_job(
        request: Request,
        file: Annotated[UploadFile, File()],
        media_type: Annotated[str, Form()],
        scale: Annotated[float, Form()] = 4.0,
        lighting: Annotated[str, Form()] = "auto",
    ) -> dict[str, object]:
        original_name = Path(file.filename or "upload").name
        suffix = Path(original_name).suffix.lower()
        allowed = IMAGE_SUFFIXES if media_type == "image" else VIDEO_SUFFIXES
        if media_type not in {"image", "video"} or suffix not in allowed:
            await file.close()
            expected = "PNG, JPG, or JPEG" if media_type == "image" else "MP4, MOV, or WebM"
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Choose a supported {media_type} file ({expected})",
            )
        try:
            options = JobOptions(scale=scale, lighting=lighting)
            if media_type == "image":
                ImagePipelineConfig(output_scale=options.scale, lighting=options.lighting)
        except ValueError as exc:
            await file.close()
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc

        temporary = await _persist_upload(file, suffix)
        try:
            job = _manager(request).create(
                source_path=temporary,
                original_name=original_name,
                media_type=media_type,
                options=options,
            )
        finally:
            temporary.unlink(missing_ok=True)
        return job.public()

    @application.get("/api/jobs/{job_id}")
    def get_job(request: Request, job_id: str) -> dict[str, object]:
        return _job_or_404(request, job_id).public()

    @application.get("/api/jobs/{job_id}/source")
    def get_source(request: Request, job_id: str) -> FileResponse:
        job = _job_or_404(request, job_id)
        return FileResponse(job.input_path, filename=job.original_name)

    @application.get("/api/jobs/{job_id}/output")
    def get_output(request: Request, job_id: str) -> FileResponse:
        job = _job_or_404(request, job_id)
        if job.status != "completed" or not job.output_path.is_file():
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Output is not ready")
        source_stem = Path(job.original_name).stem
        return FileResponse(
            job.output_path,
            filename=f"{source_stem}-enhanced{job.output_path.suffix}",
        )

    @application.get("/api/jobs/{job_id}/report")
    def get_report(request: Request, job_id: str) -> FileResponse:
        job = _job_or_404(request, job_id)
        if job.status != "completed" or not job.report_path.is_file():
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Report is not ready")
        return FileResponse(job.report_path, filename=f"{job.id}-report.json")

    @application.delete("/api/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_job(request: Request, job_id: str) -> None:
        try:
            deleted = _manager(request).delete(job_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        if not deleted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    if FRONTEND_DIRECTORY.is_dir():
        application.mount(
            "/",
            StaticFiles(directory=FRONTEND_DIRECTORY, html=True),
            name="frontend",
        )

    return application


app = create_app()
