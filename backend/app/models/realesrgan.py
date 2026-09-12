"""Real-ESRGAN x4 inference adapter.

Tiling and image conversion behavior are adapted from Real-ESRGAN commit
a4abfb2979a7bbff3f69f58f58ae324608821e27 under BSD-3-Clause.
"""

from __future__ import annotations

import gc
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional

from backend.app.domain.errors import ClearFrameError, ErrorCode
from backend.app.domain.video import Frame, FrameBatch
from backend.app.models.base import EnhancementModel, ModelRuntime
from backend.app.models.registry import ModelSpec, verify_model_weights
from backend.app.models.rrdbnet import RRDBNet

DEFAULT_TILE_FALLBACKS = (256, 192, 128, 64)


@dataclass(frozen=True, slots=True)
class RealESRGANConfig:
    weights_path: Path
    device: str = "auto"
    tile_size: int | None = None
    tile_fallbacks: tuple[int, ...] = DEFAULT_TILE_FALLBACKS
    tile_padding: int = 10
    pre_padding: int = 10
    use_half: bool | None = None

    def __post_init__(self) -> None:
        if self.tile_size is not None and self.tile_size < 0:
            raise ValueError("tile size cannot be negative")
        if not self.tile_fallbacks:
            raise ValueError("at least one tile fallback size is required")
        if any(tile <= 0 for tile in self.tile_fallbacks):
            raise ValueError("tile fallback sizes must be positive")
        if self.tile_padding < 0 or self.pre_padding < 0:
            raise ValueError("tile and pre-padding cannot be negative")


class RealESRGANModel(EnhancementModel):
    temporal = False

    def __init__(self, spec: ModelSpec, config: RealESRGANConfig) -> None:
        if spec.architecture != "RRDBNet" or spec.scale != 4:
            raise ValueError("RealESRGANModel requires the x4 RRDBNet model specification")
        self.name = spec.name
        self.version = spec.release
        self.spec = spec
        self.config = config
        self._network: RRDBNet | None = None
        self._device: torch.device | None = None
        self._half = False
        self._active_tile_size: int | None = None

    @property
    def loaded(self) -> bool:
        return self._network is not None

    @property
    def runtime(self) -> ModelRuntime | None:
        if self._device is None or self._active_tile_size is None:
            return None
        return ModelRuntime(
            device=str(self._device),
            precision="fp16" if self._half else "fp32",
            scale=self.spec.scale,
            tile_size=self._active_tile_size,
        )

    def load(self) -> None:
        if self.loaded:
            return

        device = self._resolve_device()
        use_half = (
            self.config.use_half if self.config.use_half is not None else device.type == "cuda"
        )
        if use_half and device.type != "cuda":
            raise ClearFrameError(
                ErrorCode.MODEL_LOAD_FAILED,
                "FP16 inference is supported only on CUDA for this baseline",
            )

        try:
            verify_model_weights(self.config.weights_path, self.spec)
            checkpoint = torch.load(
                self.config.weights_path,
                map_location="cpu",
                weights_only=True,
            )
            if not isinstance(checkpoint, Mapping):
                raise TypeError("model checkpoint must contain a mapping")
            parameters = checkpoint.get("params_ema", checkpoint.get("params"))
            if not isinstance(parameters, Mapping):
                raise KeyError("model checkpoint does not contain params_ema or params")

            network = RRDBNet(scale=self.spec.scale)
            network.load_state_dict(parameters, strict=True)
            network.eval()
            network.requires_grad_(False)
            network = network.to(device)
            if use_half:
                network = network.half()
        except torch.OutOfMemoryError as exc:
            self._release_cuda_cache()
            raise ClearFrameError(
                ErrorCode.CUDA_OUT_OF_MEMORY,
                f"CUDA ran out of memory while loading {self.name}",
            ) from exc
        except (OSError, RuntimeError, TypeError, KeyError) as exc:
            raise ClearFrameError(
                ErrorCode.MODEL_LOAD_FAILED,
                f"Could not load {self.name} weights from {self.config.weights_path}: {exc}",
            ) from exc

        self._network = network
        self._device = device
        self._half = use_half
        self._active_tile_size = self._tile_candidates()[0]

    def enhance(self, batch: FrameBatch) -> FrameBatch:
        if self._network is None or self._device is None:
            raise ClearFrameError(ErrorCode.INFERENCE_FAILED, f"Model {self.name} is not loaded")

        enhanced_frames = [self._enhance_frame_with_fallback(frame) for frame in batch.frames]
        return FrameBatch(
            frames=enhanced_frames,
            start_frame=batch.start_frame,
            end_frame=batch.end_frame,
        )

    def unload(self) -> None:
        self._network = None
        self._device = None
        self._active_tile_size = None
        self._half = False
        self._release_cuda_cache()

    def _resolve_device(self) -> torch.device:
        requested = self.config.device.strip().lower()
        if requested == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        try:
            device = torch.device(requested)
        except RuntimeError as exc:
            raise ClearFrameError(
                ErrorCode.MODEL_LOAD_FAILED,
                f"Invalid model device: {self.config.device}",
            ) from exc
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ClearFrameError(
                ErrorCode.MODEL_LOAD_FAILED,
                "CUDA was requested but is unavailable",
            )
        return device

    def _tile_candidates(self) -> tuple[int, ...]:
        first = self.config.tile_size
        candidates = list(self.config.tile_fallbacks if first is None else (first,))
        candidates.extend(
            tile for tile in self.config.tile_fallbacks if first is None or tile < first
        )
        return tuple(dict.fromkeys(candidates))

    def _enhance_frame_with_fallback(self, frame: Frame) -> Frame:
        candidates = self._tile_candidates()
        if self._active_tile_size in candidates:
            candidates = candidates[candidates.index(self._active_tile_size) :]

        last_error: RuntimeError | None = None
        for tile_size in candidates:
            try:
                output = self._enhance_frame(frame, tile_size)
                self._active_tile_size = tile_size
                return output
            except RuntimeError as exc:
                if not self._is_cuda_oom(exc):
                    raise ClearFrameError(
                        ErrorCode.INFERENCE_FAILED,
                        f"{self.name} inference failed: {exc}",
                    ) from exc
                last_error = exc
                self._release_cuda_cache()

        raise ClearFrameError(
            ErrorCode.CUDA_OUT_OF_MEMORY,
            f"{self.name} exhausted tile sizes {candidates}",
        ) from last_error

    @torch.inference_mode()
    def _enhance_frame(self, frame: Frame, tile_size: int) -> Frame:
        assert self._network is not None
        assert self._device is not None

        height, width = frame.shape[:2]
        rgb = np.ascontiguousarray(frame[:, :, ::-1].transpose(2, 0, 1))
        inputs = torch.from_numpy(rgb).unsqueeze(0).to(self._device)
        inputs = inputs.half() if self._half else inputs.float()
        inputs = inputs / 255.0

        if self.config.pre_padding:
            padding_mode = (
                "reflect"
                if height > self.config.pre_padding and width > self.config.pre_padding
                else "replicate"
            )
            inputs = functional.pad(
                inputs,
                (0, self.config.pre_padding, 0, self.config.pre_padding),
                mode=padding_mode,
            )

        output = self._infer_tiles(inputs, tile_size)
        output = output[:, :, : height * self.spec.scale, : width * self.spec.scale]
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        output = output.squeeze(0).float().clamp_(0, 1).cpu().permute(1, 2, 0).numpy()
        output = np.rint(output[:, :, ::-1] * 255.0).astype(np.uint8)
        return np.ascontiguousarray(output)

    def _infer_tiles(self, inputs: Tensor, tile_size: int) -> Tensor:
        assert self._network is not None
        if tile_size == 0:
            return self._network(inputs)

        _, _, height, width = inputs.shape
        scale = self.spec.scale
        output = inputs.new_empty((1, 3, height * scale, width * scale))
        for top in range(0, height, tile_size):
            for left in range(0, width, tile_size):
                bottom = min(top + tile_size, height)
                right = min(left + tile_size, width)
                padded_top = max(top - self.config.tile_padding, 0)
                padded_left = max(left - self.config.tile_padding, 0)
                padded_bottom = min(bottom + self.config.tile_padding, height)
                padded_right = min(right + self.config.tile_padding, width)

                tile = inputs[:, :, padded_top:padded_bottom, padded_left:padded_right]
                enhanced_tile = self._network(tile)

                output_top = top * scale
                output_left = left * scale
                output_bottom = bottom * scale
                output_right = right * scale
                tile_top = (top - padded_top) * scale
                tile_left = (left - padded_left) * scale
                tile_bottom = tile_top + (bottom - top) * scale
                tile_right = tile_left + (right - left) * scale
                output[:, :, output_top:output_bottom, output_left:output_right] = enhanced_tile[
                    :, :, tile_top:tile_bottom, tile_left:tile_right
                ]
        return output

    @staticmethod
    def _is_cuda_oom(error: RuntimeError) -> bool:
        return isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()

    @staticmethod
    def _release_cuda_cache() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
