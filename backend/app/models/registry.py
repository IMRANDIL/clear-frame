from __future__ import annotations

import hashlib
import hmac
import json
import os
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from backend.app.domain.errors import ClearFrameError, ErrorCode

DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parents[3] / "config" / "models.json"
DOWNLOAD_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    display_name: str
    architecture: str
    filename: str
    scale: int
    temporal: bool
    release: str
    source_url: str
    size_bytes: int
    sha256: str
    realesrgan_source_commit: str
    basicsr_source_commit: str

    def __post_init__(self) -> None:
        if not self.name or not self.display_name or not self.architecture or not self.release:
            raise ValueError("model identity fields cannot be empty")
        if Path(self.filename).name != self.filename:
            raise ValueError("model filename must not contain a path")
        if self.scale <= 0:
            raise ValueError("model scale must be positive")
        if urlparse(self.source_url).scheme != "https":
            raise ValueError("model source URL must use HTTPS")
        if self.size_bytes <= 0:
            raise ValueError("model size must be positive")
        if len(self.sha256) != 64:
            raise ValueError("model SHA-256 must contain 64 hexadecimal characters")
        try:
            int(self.sha256, 16)
        except ValueError as exc:
            raise ValueError("model SHA-256 must be hexadecimal") from exc
        for source_name, commit in (
            ("Real-ESRGAN", self.realesrgan_source_commit),
            ("BasicSR", self.basicsr_source_commit),
        ):
            if len(commit) != 40:
                raise ValueError(f"{source_name} commit must contain 40 hexadecimal characters")
            try:
                int(commit, 16)
            except ValueError as exc:
                raise ValueError(f"{source_name} commit must be hexadecimal") from exc


def _required_value(data: Mapping[str, object], key: str) -> object:
    value = data.get(key)
    if value is None or value == "":
        raise ValueError(f"model manifest field is missing: {key}")
    return value


def _required_bool(data: Mapping[str, object], key: str) -> bool:
    value = _required_value(data, key)
    if not isinstance(value, bool):
        raise ValueError(f"model manifest field must be boolean: {key}")
    return value


def load_model_spec(
    name: str,
    *,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> ModelSpec:
    try:
        payload: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("model manifest root must be an object")
        models = payload.get("models")
        if not isinstance(models, Mapping):
            raise ValueError("model manifest must contain a models object")
        raw_spec = models.get(name)
        if not isinstance(raw_spec, Mapping):
            raise ValueError(f"unknown model: {name}")

        return ModelSpec(
            name=name,
            display_name=str(_required_value(raw_spec, "display_name")),
            architecture=str(_required_value(raw_spec, "architecture")),
            filename=str(_required_value(raw_spec, "filename")),
            scale=int(_required_value(raw_spec, "scale")),
            temporal=_required_bool(raw_spec, "temporal"),
            release=str(_required_value(raw_spec, "release")),
            source_url=str(_required_value(raw_spec, "source_url")),
            size_bytes=int(_required_value(raw_spec, "size_bytes")),
            sha256=str(_required_value(raw_spec, "sha256")).lower(),
            realesrgan_source_commit=str(
                _required_value(raw_spec, "realesrgan_source_commit")
            ),
            basicsr_source_commit=str(_required_value(raw_spec, "basicsr_source_commit")),
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ClearFrameError(
            ErrorCode.MODEL_LOAD_FAILED,
            f"Could not load model manifest {manifest_path}: {exc}",
        ) from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(DOWNLOAD_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model_weights(path: Path, spec: ModelSpec) -> None:
    if not path.is_file():
        raise ClearFrameError(ErrorCode.MODEL_LOAD_FAILED, f"Model weights do not exist: {path}")
    actual_size = path.stat().st_size
    if actual_size != spec.size_bytes:
        raise ClearFrameError(
            ErrorCode.MODEL_LOAD_FAILED,
            f"Model size mismatch for {path}; expected {spec.size_bytes}, got {actual_size}",
        )
    actual_hash = sha256_file(path)
    if not hmac.compare_digest(actual_hash, spec.sha256):
        raise ClearFrameError(
            ErrorCode.MODEL_LOAD_FAILED,
            f"Model checksum mismatch for {path}; expected {spec.sha256}, got {actual_hash}",
        )


def ensure_model_weights(spec: ModelSpec, model_directory: Path) -> Path:
    model_directory.mkdir(parents=True, exist_ok=True)
    destination = model_directory / spec.filename
    if destination.exists():
        verify_model_weights(destination, spec)
        return destination

    temporary = model_directory / f".{spec.filename}.{uuid.uuid4().hex}.partial"
    try:
        with (
            urllib.request.urlopen(spec.source_url, timeout=60) as response,  # noqa: S310
            temporary.open("xb") as target,
        ):
            while chunk := response.read(DOWNLOAD_CHUNK_SIZE):
                target.write(chunk)
        verify_model_weights(temporary, spec)
        os.replace(temporary, destination)
    except ClearFrameError:
        raise
    except (OSError, urllib.error.URLError) as exc:
        raise ClearFrameError(
            ErrorCode.MODEL_LOAD_FAILED,
            f"Could not download model weights from {spec.source_url}: {exc}",
            retryable=True,
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)

    return destination
