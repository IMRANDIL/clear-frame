from __future__ import annotations

import shutil
import threading
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from backend.app.config import ImagePipelineConfig, PipelineConfig
from backend.app.domain.errors import ClearFrameError
from backend.app.image.pipeline import run_image_pipeline
from backend.app.pipeline.pipeline import run_pipeline

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".webm"})
TERMINAL_STATES = frozenset({"completed", "failed"})


@dataclass(frozen=True, slots=True)
class JobOptions:
    scale: float = 4.0
    lighting: str = "auto"


@dataclass(slots=True)
class JobRecord:
    id: str
    media_type: str
    original_name: str
    status: str
    progress: float
    stage: str
    message: str
    input_path: Path
    output_path: Path
    report_path: Path
    created_at: str
    updated_at: str
    error_code: str | None = None
    error_message: str | None = None

    def public(self) -> dict[str, object]:
        data = asdict(self)
        data.pop("input_path")
        data.pop("output_path")
        data.pop("report_path")
        data["source_url"] = f"/api/jobs/{self.id}/source"
        data["output_url"] = (
            f"/api/jobs/{self.id}/output" if self.status == "completed" else None
        )
        data["report_url"] = (
            f"/api/jobs/{self.id}/report" if self.status == "completed" else None
        )
        return data


class JobManager:
    """Thread-safe local job registry with one GPU worker."""

    def __init__(self, root: Path = Path("data/jobs")) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="clearframe-gpu")

    def create(
        self,
        *,
        source_path: Path,
        original_name: str,
        media_type: str,
        options: JobOptions,
    ) -> JobRecord:
        suffix = source_path.suffix.lower()
        if media_type == "image" and suffix not in IMAGE_SUFFIXES:
            raise ValueError("Image jobs require a PNG, JPG, or JPEG file")
        if media_type == "video" and suffix not in VIDEO_SUFFIXES:
            raise ValueError("Video jobs require an MP4, MOV, or WebM file")
        if media_type not in {"image", "video"}:
            raise ValueError("media type must be image or video")

        job_id = uuid.uuid4().hex
        job_directory = self.root / job_id
        job_directory.mkdir(parents=False, exist_ok=False)
        input_path = job_directory / f"source{suffix}"
        shutil.move(source_path, input_path)
        output_suffix = suffix if media_type == "image" else ".mp4"
        output_path = job_directory / f"enhanced{output_suffix}"
        report_path = Path(f"{output_path}.report.json")
        now = datetime.now(UTC).isoformat()
        record = JobRecord(
            id=job_id,
            media_type=media_type,
            original_name=original_name,
            status="queued",
            progress=0.02,
            stage="queued",
            message="Waiting for the GPU",
            input_path=input_path,
            output_path=output_path,
            report_path=report_path,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._jobs[job_id] = record
        self._executor.submit(self._process, job_id, options)
        return record

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def delete(self, job_id: str) -> bool:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return False
            if record.status not in TERMINAL_STATES:
                raise RuntimeError("A running job cannot be deleted")
            del self._jobs[job_id]
        shutil.rmtree(record.input_path.parent, ignore_errors=True)
        return True

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _update(self, job_id: str, **changes: object) -> None:
        with self._lock:
            record = self._jobs[job_id]
            for name, value in changes.items():
                setattr(record, name, value)
            record.updated_at = datetime.now(UTC).isoformat()

    def _observe(self, job_id: str, event: str, fields: Mapping[str, object]) -> None:
        stages = {
            "image_analyzed": (0.12, "analyzing", "Reading image structure"),
            "lighting_adjusted": (0.24, "lighting", "Recovering shadow detail"),
            "image_restored": (0.88, "restoring", "Neural restoration complete"),
            "image_output_validated": (0.97, "validating", "Checking final image"),
            "source_analyzed": (0.08, "analyzing", "Reading video streams"),
            "restoration_planned": (0.12, "planning", "Preparing restoration pass"),
            "video_encoded": (0.91, "encoding", "Finalizing video stream"),
            "audio_muxed": (0.95, "audio", "Restoring source audio"),
            "output_validated": (0.98, "validating", "Checking every output frame"),
        }
        if event == "frames_restored":
            processed = int(fields.get("frames_processed", 0))
            total = int(fields.get("frames_total") or 0)
            ratio = processed / total if total else 0
            self._update(
                job_id,
                progress=min(0.89, 0.14 + ratio * 0.74),
                stage="restoring",
                message=f"Restoring frame {processed} of {total}" if total else "Restoring frames",
            )
        elif event in stages:
            progress, stage, message = stages[event]
            self._update(job_id, progress=progress, stage=stage, message=message)

    def _process(self, job_id: str, options: JobOptions) -> None:
        record = self.get(job_id)
        if record is None:
            return
        self._update(
            job_id,
            status="processing",
            progress=0.04,
            stage="starting",
            message="Starting local GPU pipeline",
        )
        def observer(event: str, fields: Mapping[str, object]) -> None:
            self._observe(job_id, event, fields)

        try:
            if record.media_type == "image":
                run_image_pipeline(
                    record.input_path,
                    record.output_path,
                    config=ImagePipelineConfig(
                        output_scale=options.scale,
                        lighting=options.lighting,
                    ),
                    report_path=record.report_path,
                    observer=observer,
                )
            else:
                run_pipeline(
                    record.input_path,
                    record.output_path,
                    config=PipelineConfig(),
                    report_path=record.report_path,
                    observer=observer,
                )
        except ClearFrameError as exc:
            self._update(
                job_id,
                status="failed",
                progress=1.0,
                stage="failed",
                message="Enhancement failed",
                error_code=str(exc.code),
                error_message=str(exc),
            )
            return
        except Exception as exc:
            self._update(
                job_id,
                status="failed",
                progress=1.0,
                stage="failed",
                message="Unexpected enhancement failure",
                error_code="UNEXPECTED_ERROR",
                error_message=str(exc),
            )
            return

        self._update(
            job_id,
            status="completed",
            progress=1.0,
            stage="completed",
            message="Your restored file is ready",
        )
