"""Tests for the root assessment runner without real OCR or providers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import run_all


def test_main_runs_offline_tasks_and_reports_non_runnable_work(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ocr = Mock(
        return_value=SimpleNamespace(
            engine_name="Tesseract",
            raw_text="Text.",
            sentences=("Text.",),
            low_confidence_regions=(),
        )
    )
    evaluation = Mock(
        return_value=(
            (object(),),
            SimpleNamespace(cer=0.1, wer=0.2, sentence_f1=0.3, composite_score=0.4),
        )
    )
    fidelity = Mock(return_value=0.5)
    monkeypatch.setattr(run_all, "extract_text_and_sentences", ocr)
    monkeypatch.setattr(run_all, "evaluate_file", evaluation)
    monkeypatch.setattr(run_all, "sfs", fidelity)

    assert run_all.main() == 0

    output = capsys.readouterr().out
    assert "[Section 1] PASS OCR: Tesseract" in output
    assert "[Section 1] PASS evaluation" in output
    assert "[Section 2] NOT RUN" in output
    assert "[Section 3] DOCUMENTATION ONLY" in output
    assert "[Section 4] PASS Source-Fidelity Score" in output
    assert "[Section 4] DOCUMENTATION ONLY" in output
    ocr.assert_called_once_with(run_all._OCR_INPUT)
    evaluation.assert_called_once_with(run_all._EVALUATION_INPUT)
    fidelity.assert_called_once_with("Please wait!!!", "Please wait!")
