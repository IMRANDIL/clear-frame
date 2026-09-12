from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from backend.app.domain.video import FrameBatch


@dataclass(frozen=True, slots=True)
class ModelRuntime:
    device: str
    precision: str
    scale: int
    tile_size: int


class EnhancementModel(ABC):
    name: str
    version: str
    temporal: bool

    @property
    @abstractmethod
    def loaded(self) -> bool:
        """Whether model weights are currently resident on the selected device."""

    @property
    @abstractmethod
    def runtime(self) -> ModelRuntime | None:
        """Resolved runtime details, available after loading."""

    @abstractmethod
    def load(self) -> None:
        """Load model weights once and retain them for subsequent batches."""

    @abstractmethod
    def enhance(self, batch: FrameBatch) -> FrameBatch:
        """Enhance one contiguous frame batch."""

    @abstractmethod
    def unload(self) -> None:
        """Release model resources."""

    def __enter__(self) -> EnhancementModel:
        self.load()
        return self

    def __exit__(self, *_: object) -> None:
        self.unload()
