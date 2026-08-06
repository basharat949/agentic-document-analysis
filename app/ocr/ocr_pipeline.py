"""Deterministic OCR extraction and sentence segmentation pipeline."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
import pytesseract

from app.ocr.engine import (
    OCRDataError,
    OCREngine,
    OCREngineNotFoundError,
    OCRExecutionError,
    OCRPipelineError,
    OCRToken,
)
from app.ocr.preprocessing import ImagePath, preprocess_image
from app.ocr.tesseract_engine import TesseractEngine

LOGGER = logging.getLogger(__name__)

__all__ = (
    "InvalidConfidenceThresholdError",
    "OCRDataError",
    "OCREngineNotFoundError",
    "OCRExecutionError",
    "OCRPipelineError",
    "OCRRegion",
    "OCRResult",
    "extract_text_and_sentences",
    "pytesseract",
)

_METADATA_LINE_LIMIT = 5
_METADATA_HEIGHT_RATIO = 0.20
_MAX_TITLE_WORDS = 8
_MAX_TITLE_CHARACTERS = 80

_LABELED_METADATA_PATTERN = re.compile(
    r"^\s*(?:name|date|title|subject)\s*:\s*\S.*$",
    re.IGNORECASE,
)
_DATE_PATTERN = re.compile(
    r"^\s*(?:"
    r"(?:0?[1-9]|[12]\d|3[01])[-/.](?:0?[1-9]|1[0-2])[-/.](?:\d{2}|\d{4})"
    r"|(?:\d{4})[-/.](?:0?[1-9]|1[0-2])[-/.](?:0?[1-9]|[12]\d|3[01])"
    r"|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\s+(?:0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?,?\s+\d{4}"
    r"|(?:0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?\s+"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\s+\d{4}"
    r")\s*$",
    re.IGNORECASE,
)
_TITLE_WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z'’-]*")
_SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+")


class InvalidConfidenceThresholdError(ValueError):
    """Raised when the configured OCR confidence threshold is invalid."""


@dataclass(frozen=True, slots=True)
class OCRRegion:
    """A non-empty OCR token whose confidence is below the threshold.

    Attributes:
        text: Token text exactly as supplied by Tesseract.
        confidence: Tesseract confidence score.
        left: Left edge of the token bounding box in pixels.
        top: Top edge of the token bounding box in pixels.
        width: Width of the token bounding box in pixels.
        height: Height of the token bounding box in pixels.
        page_num: Tesseract page identifier, when available.
        block_num: Tesseract block identifier, when available.
        paragraph_num: Tesseract paragraph identifier, when available.
        line_num: Tesseract line identifier, when available.
    """

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
class OCRResult:
    """Structured text and diagnostic output from the OCR pipeline.

    Attributes:
        raw_text: All non-empty OCR lines in reading order, including metadata.
        body_text: OCR lines after conservative metadata exclusion.
        sentences: Non-empty body sentences in reading order.
        low_confidence_regions: Every non-empty token below the threshold.
        excluded_metadata_lines: Lines omitted only from body and sentences.
        engine_name: Name of the OCR engine.
        engine_version: Engine version, or ``None`` when it cannot be reported.
    """

    raw_text: str
    body_text: str
    sentences: tuple[str, ...]
    low_confidence_regions: tuple[OCRRegion, ...]
    excluded_metadata_lines: tuple[str, ...]
    engine_name: str
    engine_version: str | None


@dataclass(frozen=True, slots=True)
class _OCRLine:
    """Internal ordered OCR line with its vertical position."""

    text: str
    top: int


def extract_text_and_sentences(
    image_path: ImagePath,
    confidence_threshold: float = 60.0,
    engine: OCREngine | None = None,
) -> OCRResult:
    """Preprocess an image and extract text, sentences, and uncertain regions.

    Args:
        image_path: Path accepted by :func:`preprocess_image`.
        confidence_threshold: Exclusive lower-bound confidence cutoff. Every
            non-empty token with a score below this value is flagged.
        engine: Explicit OCR adapter. Defaults to Tesseract for backward
            compatibility; low-level extraction never reads environment state.

    Returns:
        Immutable OCR output containing original text and derived body content.

    Raises:
        InvalidConfidenceThresholdError: If the threshold is not a finite number
            between 0 and 100 inclusive.
        OCREngineNotFoundError: If the Tesseract executable is unavailable.
        OCRExecutionError: If Tesseract cannot process the image.
        OCRDataError: If Tesseract returns malformed OCR data.
        ImagePreprocessingError: If the existing preprocessing stage fails.
    """
    threshold = _validate_confidence_threshold(confidence_threshold)
    LOGGER.info("Starting OCR extraction for %s", image_path)

    processed_image = preprocess_image(image_path)
    selected_engine = engine if engine is not None else TesseractEngine()
    extraction = selected_engine.extract(processed_image)
    lines, low_confidence_regions = _build_lines_and_regions(
        extraction.tokens, threshold
    )

    raw_text = "\n".join(line.text for line in lines)
    body_lines, excluded_lines = _exclude_metadata(lines, processed_image.shape[0])
    body_text = "\n".join(line.text for line in body_lines)
    sentences = _segment_sentences(body_text)
    LOGGER.info(
        "Finished OCR extraction for %s: %d lines, %d low-confidence regions",
        image_path,
        len(lines),
        len(low_confidence_regions),
    )
    return OCRResult(
        raw_text=raw_text,
        body_text=body_text,
        sentences=sentences,
        low_confidence_regions=low_confidence_regions,
        excluded_metadata_lines=excluded_lines,
        engine_name=extraction.engine_name,
        engine_version=extraction.engine_version,
    )


def _validate_confidence_threshold(value: float) -> float:
    """Validate and normalize the caller's confidence threshold."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidConfidenceThresholdError(
            "confidence_threshold must be a number between 0 and 100"
        )
    threshold = float(value)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 100.0:
        raise InvalidConfidenceThresholdError(
            "confidence_threshold must be between 0 and 100 inclusive"
        )
    return threshold


def _build_lines_and_regions(
    tokens: tuple[OCRToken, ...],
    confidence_threshold: float,
) -> tuple[tuple[_OCRLine, ...], tuple[OCRRegion, ...]]:
    """Build existing pipeline lines and low-confidence regions from tokens."""
    line_tokens: dict[tuple[int | None, ...], list[str]] = {}
    line_tops: dict[tuple[int | None, ...], int] = {}
    regions: list[OCRRegion] = []

    for token in tokens:
        fallback_top = token.top if token.line_num is None else None
        line_key = (
            token.page_num,
            token.block_num,
            token.paragraph_num,
            token.line_num,
            fallback_top,
        )
        line_tokens.setdefault(line_key, []).append(token.text)
        line_tops.setdefault(line_key, token.top)

        if token.confidence < confidence_threshold:
            regions.append(
                OCRRegion(
                    text=token.text,
                    confidence=token.confidence,
                    left=token.left,
                    top=token.top,
                    width=token.width,
                    height=token.height,
                    page_num=token.page_num,
                    block_num=token.block_num,
                    paragraph_num=token.paragraph_num,
                    line_num=token.line_num,
                )
            )

    lines = tuple(
        _OCRLine(text=" ".join(tokens), top=line_tops[key])
        for key, tokens in line_tokens.items()
    )
    return lines, tuple(regions)


def _exclude_metadata(
    lines: tuple[_OCRLine, ...],
    image_height: int,
) -> tuple[tuple[_OCRLine, ...], tuple[str, ...]]:
    """Conservatively exclude strong metadata matches near the page top."""
    body_lines: list[_OCRLine] = []
    excluded_lines: list[str] = []
    top_limit = image_height * _METADATA_HEIGHT_RATIO

    for index, line in enumerate(lines):
        is_candidate = index < _METADATA_LINE_LIMIT and line.top <= top_limit
        if is_candidate and _is_metadata_line(line.text, is_first=index == 0):
            excluded_lines.append(line.text)
        else:
            body_lines.append(line)

    return tuple(body_lines), tuple(excluded_lines)


def _is_metadata_line(text: str, *, is_first: bool) -> bool:
    """Match explicit labels, full dates, or a conservative first-page title."""
    stripped = text.strip()
    if _LABELED_METADATA_PATTERN.fullmatch(stripped):
        return True
    if _DATE_PATTERN.fullmatch(stripped):
        return True
    if not is_first or len(stripped) > _MAX_TITLE_CHARACTERS:
        return False
    if stripped.endswith((".", "?", "!", ";")):
        return False

    words = _TITLE_WORD_PATTERN.findall(stripped)
    if not 1 <= len(words) <= _MAX_TITLE_WORDS:
        return False
    return all(word.isupper() or word.istitle() for word in words)


def _segment_sentences(text: str) -> tuple[str, ...]:
    """Split body text at punctuation followed by whitespace."""
    if not text.strip():
        return ()
    return tuple(
        sentence.strip()
        for sentence in _SENTENCE_BOUNDARY_PATTERN.split(text)
        if sentence.strip()
    )
