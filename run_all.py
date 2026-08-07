"""Run the assessment's offline-safe executable tasks from the repository root."""

from __future__ import annotations

from pathlib import Path

from app.ocr.ocr_pipeline import extract_text_and_sentences
from section1.eval import evaluate_file
from section4.sfs import sfs

_ROOT = Path(__file__).resolve().parent
_OCR_INPUT = _ROOT / "samples/testing_pages/page-1-small.png"
_EVALUATION_INPUT = _ROOT / "samples/results/paddle_evaluation.json"


def main() -> int:
    """Execute runnable local tasks and identify non-runnable design work."""
    print("Assessment runner (offline-safe, default OCR engine: Tesseract)")

    ocr_result = extract_text_and_sentences(_OCR_INPUT)
    print(
        "[Section 1] PASS OCR: "
        f"{ocr_result.engine_name}; {len(ocr_result.raw_text)} characters; "
        f"{len(ocr_result.sentences)} sentences; "
        f"{len(ocr_result.low_confidence_regions)} low-confidence regions"
    )

    evaluation_results, aggregate = evaluate_file(_EVALUATION_INPUT)
    print(
        "[Section 1] PASS evaluation: "
        f"{len(evaluation_results)} sample; CER={aggregate.cer:.4f}; "
        f"WER={aggregate.wer:.4f}; Sentence F1={aggregate.sentence_f1:.4f}; "
        f"Composite={aggregate.composite_score:.4f}"
    )

    print(
        "[Section 2] NOT RUN: deterministic orchestration requires injected "
        "classifier clients; no concrete LLM provider or offline demo adapter "
        "is included"
    )
    print(
        "[Section 3] DOCUMENTATION ONLY: production architecture, model "
        "migration, and fidelity trade-off designs"
    )

    fidelity_score = sfs("Please wait!!!", "Please wait!")
    print(f"[Section 4] PASS Source-Fidelity Score example: {fidelity_score:.4f}")
    print(
        "[Section 4] DOCUMENTATION ONLY: hybrid OCR/Vision fallback design"
    )
    print("Assessment runner completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
