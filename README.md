# ClearFrame

ClearFrame restores low-quality images and video into perceptually cleaner, higher-resolution
results.

The current milestone is a local CLI proof-of-concept using FFmpeg, FFprobe, PyTorch, CUDA,
and Real-ESRGAN. No API, web application, database, authentication, or cloud infrastructure
is part of this milestone.

## Current Status

The local video and image CLI pipelines and generated end-to-end regression suite are implemented.
The video pipeline has passed a representative automated run; playback approval remains pending.

## Video Usage

```powershell
& ".venv\Scripts\python.exe" scripts\enhance_video.py `
  --input fixtures\low_resolution\test.mp4 `
  --output data\output\test-enhanced.mp4
```

The initial model is `RealESRGAN_x4plus`; H.264 output defaults to CRF 18. Use `--help` to inspect
configurable model, device, tile, batch, encoding, audio, and executable settings.

## Image Usage

PNG, JPG, and JPEG inputs and outputs are supported. The default 4x image path combines
Real-ESRGAN detail restoration with conservative automatic shadow recovery for globally dark
images.

```powershell
& ".venv\Scripts\python.exe" scripts\enhance_image.py `
  --input data\input\photo.jpg `
  --output data\output\photo-enhanced.jpg
```

Use `--scale 1` through `--scale 4` to control dimensions and `--lighting auto|force|off` to
control gloomy-image correction. JPEG output defaults to quality 95. PNG transparency is retained
when the output is PNG; transparent sources are rejected for JPEG output to prevent silent data
loss. Inputs must be 8-bit images. A reproducibility report is written next to each output.

## Environment

ClearFrame uses Python 3.12 in an isolated `.venv`. The global Python environment must not be
used for project dependencies.

```powershell
py -3.12 -m venv .venv
& ".venv\Scripts\python.exe" -m pip install --upgrade pip
& ".venv\Scripts\python.exe" -m pip install -r requirements\bootstrap-cu130.lock.txt
& ".venv\Scripts\python.exe" -m pip install -r requirements\video.lock.txt
& ".venv\Scripts\python.exe" scripts\check_environment.py
```

The environment check succeeds only when Python, FFmpeg, FFprobe, and CUDA-enabled PyTorch are
available.

## Initial Baseline

| Setting | Value |
| --- | --- |
| Model | RealESRGAN_x4plus |
| Image formats | PNG, JPG, JPEG |
| Image scale | 4x by default; configurable from 1x to 4x |
| Image lighting | Conservative automatic shadow recovery |
| Video codec | H.264 with libx264 |
| Quality | CRF 18 |
| Pixel format | yuv420p |
| Target | Aspect-preserving 1080p |

These are configurable defaults, not hard-coded pipeline behavior.
