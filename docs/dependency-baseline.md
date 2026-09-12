# Dependency Baseline

This document records the initial local inference environment. Runtime dependencies will be
added and locked only after they pass an isolated import and inference smoke test.

## Host

| Component | Version or value |
| --- | --- |
| Operating system | Windows |
| Python | 3.12.3 |
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop GPU |
| GPU memory | 4096 MiB |
| NVIDIA driver | 610.62 |
| Driver-reported CUDA support | 13.3 |
| FFmpeg | 2024-02-01 git-94422871fc |
| FFprobe | 2024-02-01 git-94422871fc |
| pip | 26.2.1 |

The NVIDIA CUDA Toolkit and `nvcc` are not required. PyTorch wheels include the CUDA runtime
needed for inference; the installed NVIDIA driver provides device support.

## Python Runtime

| Package | Exact version | Source |
| --- | --- | --- |
| torch | 2.14.0+cu130 | PyTorch CUDA 13.0 wheel index |
| torchvision | 0.29.0+cu130 | PyTorch CUDA 13.0 wheel index |
| opencv-python-headless | 4.14.0.94 | Python Package Index |

The global Python installation contains CPU-only PyTorch 2.4.0. ClearFrame must use `.venv`
and must not alter the global environment.

The environment was verified by allocating tensors on the RTX 3050 Ti and executing a CUDA
matrix multiplication. `pip check` reported no broken requirements. The complete transitive
bootstrap lock is stored in `requirements/bootstrap-cu130.lock.txt`.

Video-processing packages are separately pinned in `requirements/video.lock.txt` so the PyTorch
CUDA wheel index and the Python Package Index do not need to be combined in one install command.

## Restoration Sources

Real-ESRGAN and BasicSR have broad package-level imports intended for training and evaluation.
ClearFrame uses only the exact inference architecture and tiling behavior required by the
baseline, adapted behind its own model interface from these pinned upstream commits:

| Project | Candidate commit | Reason |
| --- | --- | --- |
| Real-ESRGAN | `a4abfb2979a7bbff3f69f58f58ae324608821e27` | Pinned upstream inference behavior |
| BasicSR | `8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a` | Uses the supported torchvision functional import |

The upstream Python packages are not runtime dependencies. Their licenses and source commits are
recorded in `THIRD_PARTY_NOTICES.md` and `config/models.json`.

## Model Compatibility Check

The official `RealESRGAN_x4plus` v0.1.0 weight artifact was downloaded and verified:

| Property | Verified value |
| --- | --- |
| Size | 67,040,989 bytes |
| SHA-256 | `4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1` |
| Device | NVIDIA GeForce RTX 3050 Ti Laptop GPU |
| Precision | FP16 |
| Tile size | 256 |
| Smoke-test input | 320x180 BGR frame |
| Smoke-test output | 1280x720 BGR frame |
| Inference time | 0.632 seconds |
| Peak allocated VRAM | 464.4 MiB |

The timing is a hardware-specific smoke-test measurement, excludes model loading, and is not yet
a pipeline benchmark.

## Development Tools

| Package | Exact version | Purpose |
| --- | --- | --- |
| pytest | 9.1.1 | Unit and integration tests |
| ruff | 0.16.7 | Linting and import checks |

The exact transitive development lock is stored in `requirements/dev.lock.txt`.
