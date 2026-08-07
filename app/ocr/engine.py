"""Shared, engine-neutral OCR adapter contracts and immutable output models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


class OCRPipelineError(RuntimeError):
    """Base exception for failures in the OCR stage."""


class OCREngineNotFoundError(OCRPipelineError):
    """Raised when a requested OCR engine cannot be executed."""


class OCRExecutionError(OCRPipelineError):
    """Raised when an OCR engine fails while processing an image."""


class OCRDataError(OCRPipelineError):
    """Raised when an OCR engine returns malformed data."""


@dataclass(frozen=True, slots=True)
class OCRToken:
    """One recognized text region on the common 0-100 confidence scale."""

    text: str
    confidence: float
    left: int
    top: int
    width: int
    height: int
    page_num: int | None
    block_num: int | None
    paragraph_num: int | None
    line_num: int | None


@dataclass(frozen=True, slots=True)
class OCRExtraction:
    """Engine-neutral recognized regions in deterministic reading order."""

    tokens: tuple[OCRToken, ...]
    engine_name: str
    engine_version: str | None


@runtime_checkable
class OCREngine(Protocol):
    """Interface implemented by selectable local OCR engines."""

    @property
    def name(self) -> str:
        """Return the stable display name of the engine."""
        ...

    @property
    def version(self) -> str | None:
        """Return the runtime version when it can be determined."""
        ...

    def extract(self, image: np.ndarray) -> OCRExtraction:
        """Recognize text from a preprocessed image."""
        ...
