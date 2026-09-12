# Third-Party Notices

ClearFrame's own project license has not been selected. This file records third-party software
and model artifacts used by the local proof-of-concept. It is not a substitute for legal review.

| Component | Pinned version | License | Project source |
| --- | --- | --- | --- |
| PyTorch | 2.14.0+cu130 | BSD-3-Clause | https://github.com/pytorch/pytorch |
| torchvision | 0.29.0+cu130 | BSD-3-Clause | https://github.com/pytorch/vision |
| NumPy | 2.5.2 | BSD-3-Clause | https://github.com/numpy/numpy |
| OpenCV Python Headless | 4.14.0.94 | Apache-2.0 | https://github.com/opencv/opencv-python |
| pytest | 9.1.1 | MIT | https://github.com/pytest-dev/pytest |
| Ruff | 0.16.7 | MIT | https://github.com/astral-sh/ruff |
| Real-ESRGAN inference behavior | Commit `a4abfb2979a7bbff3f69f58f58ae324608821e27` | BSD-3-Clause | https://github.com/xinntao/Real-ESRGAN |
| BasicSR RRDBNet architecture | Commit `8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a` | Apache-2.0 | https://github.com/XPixelGroup/BasicSR |
| FFmpeg | 2024-02-01 git-94422871fc, Gyan full build | GPL-enabled build | https://ffmpeg.org |
| x264 | Version bundled with the local FFmpeg build | GPL-2.0-or-later | https://www.videolan.org/developers/x264.html |
| RealESRGAN_x4plus weights | Release v0.1.0, SHA-256 recorded in `config/models.json` | Separate weight terms not identified | https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.1.0 |

Exact versions of the transitive Python packages installed by the CUDA bootstrap are recorded
in `requirements/bootstrap-cu130.lock.txt`. Their packaged license metadata and upstream notices
must remain available in the environment. A distributable release will require a generated
complete license bundle and a model-weight licensing review.

The project currently invokes the user's existing FFmpeg executable and does not redistribute
it. Preserve upstream notices if ClearFrame later distributes binaries, source, or model weights.

This table will be updated whenever a direct dependency or model is added.
