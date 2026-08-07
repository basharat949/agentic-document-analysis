"""Tests for centralized OCR configuration and engine construction."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from app.ocr import cli
from app.ocr.config import OCRConfigurationError, OCRSettings
from app.ocr.engine import OCREngineNotFoundError
from app.ocr.factory import create_configured_ocr_engine, create_ocr_engine
from app.ocr.ocr_pipeline import OCRRegion, OCRResult
from app.ocr.paddle_engine import PaddleOCREngine, PaddleOCRNotInstalledError
from app.ocr.tesseract_engine import TesseractEngine


def test_missing_environment_value_defaults_to_tesseract() -> None:
    settings = OCRSettings.from_environment({})

    assert settings.engine == "tesseract"
    assert isinstance(create_configured_ocr_engine(settings), TesseractEngine)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("tesseract", "tesseract"),
        ("TESSERACT", "tesseract"),
        ("  TeSsErAcT  ", "tesseract"),
        ("paddle", "paddle"),
        ("PADDLE", "paddle"),
        ("  PaDdLe  ", "paddle"),
    ],
)
def test_environment_values_are_normalized(value: str, expected: str) -> None:
    assert OCRSettings.from_environment({"OCR_ENGINE": value}).engine == expected


def test_invalid_environment_value_fails_clearly() -> None:
    with pytest.raises(OCRConfigurationError, match="Unsupported OCR_ENGINE"):
        OCRSettings.from_environment({"OCR_ENGINE": "unknown"})


def test_invalid_explicit_factory_name_fails_clearly() -> None:
    with pytest.raises(OCRConfigurationError, match="Unsupported OCR_ENGINE"):
        create_ocr_engine("unknown")


def test_explicit_factory_supports_both_engines_without_initializing_paddle() -> None:
    assert isinstance(create_ocr_engine(" TESSERACT "), TesseractEngine)
    paddle = create_ocr_engine(" PADDLE ")

    assert isinstance(paddle, PaddleOCREngine)
    assert paddle.version is None


def test_cli_uses_central_settings_and_writes_configured_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "result.json"
    settings = OCRSettings("paddle")
    engine = Mock(name="configured-engine")
    result = OCRResult(
        raw_text="Exact text.",
        body_text="Body text.",
        sentences=("Exact text.",),
        low_confidence_regions=(
            OCRRegion("Exact", 42.0, 1, 2, 3, 4, 1, 0, 0, 1),
        ),
        excluded_metadata_lines=("Title",),
        engine_name="PaddleOCR",
        engine_version="3.7.0",
    )
    settings_loader = Mock(return_value=settings)
    factory = Mock(return_value=engine)
    extraction = Mock(return_value=result)
    monkeypatch.setattr(cli.OCRSettings, "from_environment", settings_loader)
    monkeypatch.setattr(cli, "create_configured_ocr_engine", factory)
    monkeypatch.setattr(cli, "extract_text_and_sentences", extraction)

    exit_code = cli.main(["page.png", "--output", str(output)])

    assert exit_code == 0
    settings_loader.assert_called_once_with()
    factory.assert_called_once_with(settings)
    assert extraction.call_args.kwargs["engine"] is engine
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["configured_engine"] == "paddle"
    assert payload["engine_name"] == "PaddleOCR"
    assert payload["engine_version"] == "3.7.0"
    assert payload["raw_text"] == "Exact text."
    assert payload["body_text"] == "Body text."
    assert payload["sentences"] == ["Exact text."]
    assert payload["excluded_metadata_lines"] == ["Title"]
    assert payload["low_confidence_regions"][0] == {
        "text": "Exact",
        "confidence": 42.0,
        "left": 1,
        "top": 2,
        "width": 3,
        "height": 4,
        "page_num": 1,
        "block_num": 0,
        "paragraph_num": 0,
        "line_num": 1,
    }


def test_cli_explicit_engine_overrides_environment_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "result.json"
    environment_loader = Mock(side_effect=AssertionError("environment was read"))
    factory = Mock(return_value=Mock(name="tesseract-engine"))
    monkeypatch.setattr(cli.OCRSettings, "from_environment", environment_loader)
    monkeypatch.setattr(cli, "create_configured_ocr_engine", factory)
    monkeypatch.setattr(
        cli,
        "extract_text_and_sentences",
        Mock(
            return_value=OCRResult(
                "",
                "",
                (),
                (),
                (),
                "Tesseract",
                "5.5.0",
            )
        ),
    )

    cli.main(
        ["page.png", "--engine", " TESSERACT ", "--output", str(output)]
    )

    environment_loader.assert_not_called()
    configured_settings = factory.call_args.args[0]
    assert configured_settings.engine == "tesseract"
    assert json.loads(output.read_text())["configured_engine"] == "tesseract"


@pytest.mark.parametrize(
    ("environment_value", "expected"),
    [("tesseract", "tesseract"), (" PADDLE ", "paddle")],
)
def test_cli_environment_selects_configured_engine_once(
    environment_value: str,
    expected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "result.json"
    monkeypatch.setenv("OCR_ENGINE", environment_value)
    factory = Mock(return_value=Mock(name="engine"))
    monkeypatch.setattr(cli, "create_configured_ocr_engine", factory)
    monkeypatch.setattr(
        cli,
        "extract_text_and_sentences",
        Mock(return_value=OCRResult("", "", (), (), (), "mock", "1")),
    )

    assert cli.main(["page.png", "--output", str(output)]) == 0

    assert factory.call_args.args[0].engine == expected
    assert json.loads(output.read_text())["configured_engine"] == expected


@pytest.mark.parametrize(
    ("argv", "error", "message"),
    [
        (
            ["page.png", "--engine", "invalid"],
            None,
            "Unsupported OCR_ENGINE",
        ),
        (
            ["page.png", "--engine", "paddle"],
            PaddleOCRNotInstalledError("Install with uv sync --extra paddle"),
            "uv sync --extra paddle",
        ),
        (
            ["page.png", "--engine", "tesseract"],
            OCREngineNotFoundError("Tesseract executable was not found"),
            "Tesseract executable was not found",
        ),
    ],
)
def test_cli_expected_errors_are_user_facing_without_tracebacks(
    argv: list[str],
    error: Exception | None,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "result.json"
    if error is not None:
        monkeypatch.setattr(
            cli, "extract_text_and_sentences", Mock(side_effect=error)
        )

    with pytest.raises(SystemExit) as caught:
        cli.main([*argv, "--output", str(output)])

    assert caught.value.code == 2
    stderr = capsys.readouterr().err
    assert message in stderr
    assert "Traceback" not in stderr
    assert not output.exists()


def test_cli_missing_image_is_reported_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "result.json"
    missing = tmp_path / "missing.png"

    with pytest.raises(SystemExit) as caught:
        cli.main([str(missing), "--output", str(output)])

    assert caught.value.code == 2
    stderr = capsys.readouterr().err
    assert "Image file does not exist" in stderr
    assert "Traceback" not in stderr


def test_cli_invalid_confidence_threshold_is_user_facing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "result.json"

    with pytest.raises(SystemExit) as caught:
        cli.main(
            [
                "page.png",
                "--confidence-threshold",
                "101",
                "--output",
                str(output),
            ]
        )

    assert caught.value.code == 2
    stderr = capsys.readouterr().err
    assert "confidence_threshold must be between 0 and 100" in stderr
    assert "Traceback" not in stderr


def test_cli_unreadable_image_is_reported_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "result.json"
    unreadable = tmp_path / "unreadable.png"
    unreadable.write_text("not an image", encoding="utf-8")

    with pytest.raises(SystemExit) as caught:
        cli.main([str(unreadable), "--output", str(output)])

    assert caught.value.code == 2
    stderr = capsys.readouterr().err
    assert "could not decode image" in stderr
    assert "Traceback" not in stderr
