from __future__ import annotations

import os
import subprocess
import threading
from collections import deque
from collections.abc import Sequence
from typing import BinaryIO

from backend.app.domain.errors import ClearFrameError, ErrorCode

STDERR_TAIL_LINES = 100


def get_ffmpeg_version(ffmpeg_path: str | os.PathLike[str] = "ffmpeg") -> str:
    try:
        result = subprocess.run(
            [os.fspath(ffmpeg_path), "-version"],
            capture_output=True,
            check=False,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return "unknown"
    lines = (result.stdout or result.stderr).splitlines()
    return lines[0] if result.returncode == 0 and lines else "unknown"


class _StderrCollector(threading.Thread):
    def __init__(self, stream: BinaryIO) -> None:
        super().__init__(name="ffmpeg-stderr", daemon=True)
        self._stream = stream
        self._lines: deque[str] = deque(maxlen=STDERR_TAIL_LINES)

    @property
    def detail(self) -> str:
        return "\n".join(self._lines).strip()

    def run(self) -> None:
        while line := self._stream.readline():
            self._lines.append(line.decode("utf-8", errors="replace").rstrip())


class FFmpegProcess:
    """Managed FFmpeg process whose stderr is continuously drained into a bounded tail."""

    def __init__(
        self,
        arguments: Sequence[str | os.PathLike[str]],
        *,
        ffmpeg_path: str | os.PathLike[str],
        error_code: ErrorCode,
        pipe_stdin: bool = False,
        pipe_stdout: bool = False,
    ) -> None:
        self.error_code = error_code
        command = [os.fspath(ffmpeg_path), *(os.fspath(argument) for argument in arguments)]
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE if pipe_stdin else subprocess.DEVNULL,
                stdout=subprocess.PIPE if pipe_stdout else subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise ClearFrameError(
                error_code,
                f"FFmpeg executable was not found: {ffmpeg_path}",
            ) from exc
        except OSError as exc:
            raise ClearFrameError(error_code, f"Could not start FFmpeg: {exc}") from exc

        assert self.process.stderr is not None
        self._stderr = _StderrCollector(self.process.stderr)
        self._stderr.start()

    @property
    def stderr_detail(self) -> str:
        return self._stderr.detail or "no FFmpeg error detail"

    def wait(self, *, timeout_seconds: float | None = None) -> int:
        try:
            return_code = self.process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            self.terminate()
            raise ClearFrameError(
                self.error_code,
                f"FFmpeg did not exit within {timeout_seconds:g} seconds",
                retryable=True,
            ) from exc
        finally:
            self._stderr.join(timeout=2)
        return return_code

    def terminate(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self._stderr.join(timeout=2)

    def close(self) -> None:
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            if stream is not None:
                stream.close()
