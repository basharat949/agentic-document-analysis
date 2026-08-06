"""Centralized, immutable OCR engine configuration."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

_DEFAULT_ENGINE = "tesseract"
_SUPPORTED_ENGINES = frozenset(("tesseract", "paddle"))


class OCRConfigurationError(ValueError):
    """Raised when OCR engine configuration is invalid or unsupported."""


@dataclass(frozen=True, slots=True)
class OCRSettings:
    """Validated OCR settings resolved once at an application boundary."""

    engine: str = _DEFAULT_ENGINE

    def __post_init__(self) -> None:
        if not isinstance(self.engine, str):
            raise OCRConfigurationError(
                "OCR_ENGINE must be 'tesseract' or 'paddle'"
            )
        normalized = self.engine.strip().casefold()
        if normalized not in _SUPPORTED_ENGINES:
            raise OCRConfigurationError(
                f"Unsupported OCR_ENGINE value {self.engine!r}; expected "
                "'tesseract' or 'paddle'"
            )
        object.__setattr__(self, "engine", normalized)

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> OCRSettings:
        """Read and validate ``OCR_ENGINE`` once, defaulting to Tesseract."""
        source = os.environ if environ is None else environ
        return cls(engine=source.get("OCR_ENGINE", _DEFAULT_ENGINE))
