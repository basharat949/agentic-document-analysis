"""Tests for the deterministic Tesseract OCR pipeline."""

from __future__ import annotations

import shutil
from typing import Any
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from pytesseract.pytesseract import TesseractNotFoundError

from app.ocr import ocr_pipeline
from app.ocr.ocr_pipeline import (
    InvalidConfidenceThresholdError,
    OCREngineNotFoundError,
    OCRResult,
    extract_text_and_sentences,
)


def _ocr_data(
    rows: list[tuple[str, float, int, int, int, int, int, int, int, int]],
) -> dict[str, list[Any]]:
    """Build a pytesseract ``Output.DICT``-shaped mapping."""
    fields = (
        "text",
        "conf",
        "left",
        "top",
        "width",
        "height",
        "page_num",
        "block_num",
        "par_num",
        "line_num",
    )
    return {field: [row[index] for row in rows] for index, field in enumerate(fields)}


def _configure_mocks(
    monkeypatch: pytest.MonkeyPatch,
    image: np.ndarray,
    data: dict[str, list[Any]],
) -> Mock:
    """Replace preprocessing and Tesseract with deterministic test doubles."""
    preprocess_mock = Mock(return_value=image)
    monkeypatch.setattr(ocr_pipeline, "preprocess_image", preprocess_mock)
    monkeypatch.setattr(ocr_pipeline.pytesseract, "image_to_data", Mock(return_value=data))
    monkeypatch.setattr(
        ocr_pipeline.pytesseract,
        "get_tesseract_version",
        Mock(return_value="5.5.0"),
    )
    return preprocess_mock


def test_clean_image_extracts_ordered_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clean image is preprocessed first and converted to ordered sentences."""
    image = np.full((200, 400), 255, dtype=np.uint8)
    data = _ocr_data(
        [
            ("This", 96, 10, 60, 35, 15, 1, 1, 1, 1),
            ("works.", 95, 50, 60, 50, 15, 1, 1, 1, 1),
            ("Next", 94, 10, 90, 35, 15, 1, 1, 1, 2),
            ("line.", 93, 50, 90, 35, 15, 1, 1, 1, 2),
        ]
    )
    preprocess_mock = _configure_mocks(monkeypatch, image, data)

    result = extract_text_and_sentences("clean.png")

    preprocess_mock.assert_called_once_with("clean.png")
    assert result.raw_text == "This works.\nNext line."
    assert result.body_text == result.raw_text
    assert result.sentences == ("This works.", "Next line.")
    assert result.engine_name == "Tesseract"
    assert result.engine_version == "5.5.0"


def test_skewed_image_uses_preprocessed_pixels(monkeypatch: pytest.MonkeyPatch) -> None:
    """The OCR engine receives the deskewed output of preprocessing unchanged."""
    deskewed = np.full((180, 360), 255, dtype=np.uint8)
    data = _ocr_data([("Readable.", 91, 20, 70, 80, 18, 1, 1, 1, 1)])
    _configure_mocks(monkeypatch, deskewed, data)

    extract_text_and_sentences("skewed.png")

    passed_image = ocr_pipeline.pytesseract.image_to_data.call_args.args[0]
    assert passed_image is deskewed


def test_low_contrast_image_is_not_postprocessed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OCR text is retained exactly rather than spelling-corrected."""
    processed = np.full((160, 320), 255, dtype=np.uint8)
    data = _ocr_data([("Helo", 72, 15, 60, 40, 16, 1, 1, 1, 1)])
    _configure_mocks(monkeypatch, processed, data)

    result = extract_text_and_sentences("low-contrast.png")

    assert result.raw_text == "Helo"
    assert result.body_text == "Helo"


def test_page_metadata_is_excluded_only_from_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strong top-page metadata stays in raw text but not sentence input."""
    image = np.full((500, 400), 255, dtype=np.uint8)
    data = _ocr_data(
        [
            ("Name:", 97, 10, 20, 45, 15, 1, 1, 1, 1),
            ("Aisha", 96, 60, 20, 45, 15, 1, 1, 1, 1),
            ("Body", 94, 10, 130, 40, 15, 1, 2, 1, 1),
            ("starts", 93, 55, 130, 45, 15, 1, 2, 1, 1),
            ("here.", 92, 105, 130, 40, 15, 1, 2, 1, 1),
        ]
    )
    _configure_mocks(monkeypatch, image, data)

    result = extract_text_and_sentences("metadata.png")

    assert result.raw_text == "Name: Aisha\nBody starts here."
    assert result.body_text == "Body starts here."
    assert result.excluded_metadata_lines == ("Name: Aisha",)
    assert result.sentences == ("Body starts here.",)


def test_low_confidence_tokens_are_flagged_not_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every non-empty token below the threshold remains visible and flagged."""
    image = np.full((200, 400), 255, dtype=np.uint8)
    data = _ocr_data(
        [
            ("Certain", 88, 10, 80, 55, 17, 1, 2, 3, 4),
            ("unclear", 42.5, 70, 80, 60, 17, 1, 2, 3, 4),
        ]
    )
    _configure_mocks(monkeypatch, image, data)

    result = extract_text_and_sentences("confidence.png", confidence_threshold=60)

    assert result.raw_text == "Certain unclear"
    assert len(result.low_confidence_regions) == 1
    region = result.low_confidence_regions[0]
    assert region.text == "unclear"
    assert region.confidence == 42.5
    assert (region.left, region.top, region.width, region.height) == (70, 80, 60, 17)
    assert (region.page_num, region.block_num, region.paragraph_num, region.line_num) == (
        1,
        2,
        3,
        4,
    )


@pytest.mark.parametrize("threshold", [-0.1, 100.1, float("nan"), True, "60"])
def test_invalid_confidence_threshold_is_rejected(
    threshold: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid thresholds fail before image preprocessing or OCR execution."""
    preprocess_mock = Mock()
    monkeypatch.setattr(ocr_pipeline, "preprocess_image", preprocess_mock)

    with pytest.raises(InvalidConfidenceThresholdError):
        extract_text_and_sentences("image.png", threshold)

    preprocess_mock.assert_not_called()


def test_missing_tesseract_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing executable produces a clear pipeline-specific exception."""
    image = np.full((100, 200), 255, dtype=np.uint8)
    monkeypatch.setattr(ocr_pipeline, "preprocess_image", Mock(return_value=image))
    monkeypatch.setattr(
        ocr_pipeline.pytesseract,
        "image_to_data",
        Mock(side_effect=TesseractNotFoundError()),
    )

    with pytest.raises(OCREngineNotFoundError, match="executable was not found"):
        extract_text_and_sentences("image.png")


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="Tesseract is not installed")
def test_real_tesseract_smoke(tmp_path: Any) -> None:
    """Run an optional end-to-end extraction when Tesseract is available."""
    image = np.full((160, 500, 3), 255, dtype=np.uint8)
    cv2.putText(
        image,
        "A test sentence.",
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    image_path = tmp_path / "tesseract-smoke.png"
    assert cv2.imwrite(str(image_path), image)

    result = extract_text_and_sentences(image_path)

    assert isinstance(result, OCRResult)
    assert result.engine_name == "Tesseract"
    assert result.raw_text
