"""Focused contract tests for the shared Tesseract OCR adapter."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest
from pytesseract.pytesseract import TesseractNotFoundError

from app.ocr.engine import OCREngine, OCREngineNotFoundError
from app.ocr.tesseract_engine import TesseractEngine


def test_tesseract_engine_satisfies_shared_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The adapter exposes the shared name, version, and extraction contract."""
    image = np.full((40, 80), 255, dtype=np.uint8)
    data = {
        "text": ["Exact", "text."],
        "conf": [91.5, 42.25],
        "left": [5, 35],
        "top": [10, 10],
        "width": [25, 30],
        "height": [12, 12],
        "page_num": [1, 1],
        "block_num": [2, 2],
        "par_num": [3, 3],
        "line_num": [4, 4],
    }
    image_to_data = Mock(return_value=data)
    monkeypatch.setattr(
        "app.ocr.tesseract_engine.pytesseract.image_to_data", image_to_data
    )
    monkeypatch.setattr(
        "app.ocr.tesseract_engine.pytesseract.get_tesseract_version",
        Mock(return_value="5.5.0"),
    )
    engine = TesseractEngine()

    extraction = engine.extract(image)

    assert isinstance(engine, OCREngine)
    assert extraction.engine_name == "Tesseract"
    assert extraction.engine_version == "5.5.0"
    assert [token.text for token in extraction.tokens] == ["Exact", "text."]
    assert [token.confidence for token in extraction.tokens] == [91.5, 42.25]
    assert (
        extraction.tokens[1].left,
        extraction.tokens[1].top,
        extraction.tokens[1].width,
        extraction.tokens[1].height,
    ) == (35, 10, 30, 12)
    assert (
        extraction.tokens[1].page_num,
        extraction.tokens[1].block_num,
        extraction.tokens[1].paragraph_num,
        extraction.tokens[1].line_num,
    ) == (1, 2, 3, 4)
    assert image_to_data.call_args.args[0] is image


def test_tesseract_adapter_preserves_missing_executable_wrapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The adapter retains the pipeline's actionable public exception."""
    monkeypatch.setattr(
        "app.ocr.tesseract_engine.pytesseract.image_to_data",
        Mock(side_effect=TesseractNotFoundError()),
    )

    with pytest.raises(OCREngineNotFoundError, match="executable was not found"):
        TesseractEngine().extract(np.zeros((10, 10), dtype=np.uint8))
