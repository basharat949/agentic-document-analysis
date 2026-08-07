"""Tesseract adapter preserving the original pipeline's extraction behavior."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pytesseract
from pytesseract import Output
from pytesseract.pytesseract import TesseractError, TesseractNotFoundError

from app.ocr.engine import (
    OCRDataError,
    OCREngineNotFoundError,
    OCRExecutionError,
    OCRExtraction,
    OCRToken,
)

LOGGER = logging.getLogger(__name__)


class TesseractEngine:
    """Local Tesseract OCR adapter."""

    @property
    def name(self) -> str:
        return "Tesseract"

    @property
    def version(self) -> str | None:
        try:
            return str(pytesseract.get_tesseract_version())
        except (TesseractNotFoundError, TesseractError, OSError):
            LOGGER.warning("Could not determine Tesseract version", exc_info=True)
            return None

    def extract(self, image: np.ndarray) -> OCRExtraction:
        data = _run_tesseract(image)
        return OCRExtraction(
            tokens=_parse_tesseract_data(data),
            engine_name=self.name,
            engine_version=self.version,
        )


def _run_tesseract(image: np.ndarray) -> Mapping[str, Sequence[Any]]:
    """Execute Tesseract and translate engine failures to public exceptions."""
    try:
        data = pytesseract.image_to_data(image, output_type=Output.DICT)
    except TesseractNotFoundError as exc:
        LOGGER.exception("Tesseract executable was not found")
        raise OCREngineNotFoundError(
            "Tesseract executable was not found; install it or configure "
            "pytesseract.pytesseract.tesseract_cmd"
        ) from exc
    except TesseractError as exc:
        LOGGER.exception("Tesseract failed while extracting OCR data")
        raise OCRExecutionError(f"Tesseract OCR execution failed: {exc}") from exc
    except OSError as exc:
        LOGGER.exception("Operating system failed to execute Tesseract")
        raise OCRExecutionError(f"Could not execute Tesseract: {exc}") from exc

    if not isinstance(data, Mapping):
        raise OCRDataError("Tesseract returned OCR data in an unexpected format")
    return data


def _parse_tesseract_data(
    data: Mapping[str, Sequence[Any]],
) -> tuple[OCRToken, ...]:
    """Convert pytesseract rows to the common immutable token model."""
    required_fields = ("text", "conf", "left", "top", "width", "height")
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise OCRDataError(
            "Tesseract OCR data is missing fields: " + ", ".join(missing_fields)
        )

    row_count = len(data["text"])
    if any(len(data[field]) != row_count for field in required_fields):
        raise OCRDataError("Tesseract OCR data columns have inconsistent lengths")

    tokens: list[OCRToken] = []
    for index in range(row_count):
        text = str(data["text"][index])
        if not text.strip():
            continue
        tokens.append(
            OCRToken(
                text=text,
                confidence=_parse_float(data["conf"][index], "conf", index),
                left=_parse_int(data["left"][index], "left", index),
                top=_parse_int(data["top"][index], "top", index),
                width=_parse_int(data["width"][index], "width", index),
                height=_parse_int(data["height"][index], "height", index),
                page_num=_optional_identifier(data, "page_num", index),
                block_num=_optional_identifier(data, "block_num", index),
                paragraph_num=_optional_identifier(data, "par_num", index),
                line_num=_optional_identifier(data, "line_num", index),
            )
        )
    return tuple(tokens)


def _parse_float(value: Any, field: str, index: int) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise OCRDataError(f"Invalid {field!r} value at OCR row {index}") from exc
    if not np.isfinite(result):
        raise OCRDataError(f"Non-finite {field!r} value at OCR row {index}")
    return result


def _parse_int(value: Any, field: str, index: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise OCRDataError(f"Invalid {field!r} value at OCR row {index}") from exc


def _optional_identifier(
    data: Mapping[str, Sequence[Any]], field: str, index: int
) -> int | None:
    values = data.get(field)
    if values is None or index >= len(values):
        return None
    value = values[index]
    if value is None or str(value).strip() == "":
        return None
    return _parse_int(value, field, index)
