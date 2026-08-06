"""Explicit construction of supported OCR engine adapters."""

from __future__ import annotations

from app.ocr.config import OCRConfigurationError, OCRSettings
from app.ocr.engine import OCREngine
from app.ocr.paddle_engine import PaddleOCREngine
from app.ocr.tesseract_engine import TesseractEngine


class UnsupportedOCREngineError(OCRConfigurationError):
    """Raised when an unknown OCR engine name is explicitly requested."""


def create_ocr_engine(name: str) -> OCREngine:
    """Create an OCR adapter from an explicit engine name."""
    try:
        settings = OCRSettings(engine=name)
    except OCRConfigurationError as exc:
        raise UnsupportedOCREngineError(str(exc)) from exc
    return create_configured_ocr_engine(settings)


def create_configured_ocr_engine(settings: OCRSettings) -> OCREngine:
    """Create the adapter selected by centralized, validated settings."""
    if not isinstance(settings, OCRSettings):
        raise TypeError("settings must be an OCRSettings instance")
    if settings.engine == "tesseract":
        return TesseractEngine()
    if settings.engine == "paddle":
        return PaddleOCREngine()
    raise OCRConfigurationError(f"Unsupported configured OCR engine {settings.engine!r}")
