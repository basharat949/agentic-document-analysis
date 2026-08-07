"""Command-line entry point for explicit local OCR engine selection."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from app.ocr.config import OCRConfigurationError, OCRSettings
from app.ocr.engine import OCRPipelineError
from app.ocr.factory import create_configured_ocr_engine
from app.ocr.ocr_pipeline import (
    InvalidConfidenceThresholdError,
    OCRResult,
    extract_text_and_sentences,
)
from app.ocr.paddle_engine import PaddleOCRNotInstalledError
from app.ocr.preprocessing import ImagePreprocessingError


def _payload(
    source_image: Path, configured_engine: str, result: OCRResult
) -> dict[str, object]:
    return {
        "source_image": str(source_image),
        "configured_engine": configured_engine,
        "engine_name": result.engine_name,
        "engine_version": result.engine_version,
        "raw_text": result.raw_text,
        "body_text": result.body_text,
        "sentences": list(result.sentences),
        "excluded_metadata_lines": list(result.excluded_metadata_lines),
        "low_confidence_regions": [
            asdict(region) for region in result.low_confidence_regions
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run OCR for one image and write the common result contract as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--engine",
        default=None,
        help="OCR adapter: tesseract or paddle "
        "(default: OCR_ENGINE or tesseract)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=60.0)
    args = parser.parse_args(argv)

    try:
        settings = (
            OCRSettings(engine=args.engine)
            if args.engine is not None
            else OCRSettings.from_environment()
        )
        engine = create_configured_ocr_engine(settings)
        result = extract_text_and_sentences(
            args.image,
            confidence_threshold=args.confidence_threshold,
            engine=engine,
        )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                _payload(args.image, settings.engine, result),
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    except (
        ImagePreprocessingError,
        InvalidConfidenceThresholdError,
        OCRConfigurationError,
        OCRPipelineError,
        PaddleOCRNotInstalledError,
    ) as exc:
        parser.error(str(exc))
    except OSError as exc:
        parser.error(f"Could not write output file {args.output}: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
