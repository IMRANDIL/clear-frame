from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from importlib import metadata


def executable_version(name: str) -> dict[str, str | bool]:
    path = shutil.which(name)
    if path is None:
        return {"available": False, "path": "", "version": ""}

    result = subprocess.run(
        [path, "-version"],
        capture_output=True,
        check=False,
        text=True,
        timeout=15,
    )
    first_line = (result.stdout or result.stderr).splitlines()
    return {
        "available": result.returncode == 0,
        "path": path,
        "version": first_line[0] if first_line else "unknown",
    }


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def torch_environment() -> dict[str, object]:
    try:
        import torch
    except ImportError:
        return {"installed": False, "cuda_available": False}

    cuda_available = torch.cuda.is_available()
    details: dict[str, object] = {
        "installed": True,
        "version": torch.__version__,
        "built_cuda": torch.version.cuda,
        "cuda_available": cuda_available,
    }
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        details.update(
            {
                "device_name": properties.name,
                "device_count": torch.cuda.device_count(),
                "device_memory_mib": round(properties.total_memory / 1024**2),
                "compute_capability": f"{properties.major}.{properties.minor}",
            }
        )
    return details


def main() -> int:
    report = {
        "platform": platform.platform(),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "supported": sys.version_info[:2] == (3, 12),
        },
        "executables": {
            "ffmpeg": executable_version("ffmpeg"),
            "ffprobe": executable_version("ffprobe"),
        },
        "packages": {
            "torch": package_version("torch"),
            "torchvision": package_version("torchvision"),
        },
        "torch": torch_environment(),
    }
    print(json.dumps(report, indent=2))

    healthy = (
        report["python"]["supported"]
        and report["executables"]["ffmpeg"]["available"]
        and report["executables"]["ffprobe"]["available"]
        and report["torch"]["cuda_available"]
    )
    return 0 if healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
