"""Compare deterministic preprocessing profiles across Tesseract PSM modes.

Tesseract confidence is diagnostic metadata, not a reliable proxy for
transcription accuracy. When ground truth is available, configurations are
ranked by transcription metrics and Source-Fidelity Score instead. Conservative
ruled-line removal is experimental because it can erase genuine horizontal
character strokes.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from pytesseract.pytesseract import TesseractError, TesseractNotFoundError

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from section1.eval import EvaluationSample, evaluate_sample
from section4.sfs import sfs

LOGGER = logging.getLogger(__name__)

_SUPPORTED_IMAGE_SUFFIXES = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)
_PROFILE_NAMES = (
    "grayscale",
    "clahe_median_adaptive",
    "median_adaptive_no_clahe",
    "otsu",
    "adaptive_line_removal",
)
_PSM_MODES = (4, 6, 11, 12)
_DEFAULT_OUTPUT_DIRECTORY = Path("artifacts/ocr_profiles")
_DEFAULT_LOW_CONFIDENCE_THRESHOLD = 60.0


class OCRProfileError(RuntimeError):
    """Base exception for invalid inputs and experiment failures."""


class OCRProfileInputError(OCRProfileError, ValueError):
    """Raised when image or ground-truth input is invalid."""


class OCRProfileExecutionError(OCRProfileError):
    """Raised when OpenCV or Tesseract cannot complete an experiment."""


@dataclass(frozen=True, slots=True)
class OCRStatistics:
    """Confidence and token-count summary for one OCR response."""

    raw_text: str
    mean_confidence: float | None
    median_confidence: float | None
    low_confidence_token_ratio: float
    non_empty_token_count: int


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    """Serializable output for one page/profile/PSM experiment."""

    page: str
    source_image: str
    profile: str
    psm: int
    processed_image: str
    raw_text: str
    mean_confidence: float | None
    median_confidence: float | None
    low_confidence_token_ratio: float
    non_empty_token_count: int
    ground_truth_available: bool
    cer: float | None = None
    wer: float | None = None
    sentence_f1: float | None = None
    composite_score: float | None = None
    sfs: float | None = None
    page_rank: int | None = None
    overall_rank: int | None = None


def collect_image_paths(input_path: str | Path) -> tuple[Path, ...]:
    """Return one image or a filename-sorted directory of page images.

    Args:
        input_path: Image file or directory containing page images.

    Returns:
        Validated image paths in deterministic order.

    Raises:
        OCRProfileInputError: If the path is missing, unsupported, or contains
            no supported page images.
    """
    path = Path(input_path).expanduser()
    if not path.exists():
        raise OCRProfileInputError(f"Input path does not exist: {path}")

    if path.is_file():
        if path.suffix.casefold() not in _SUPPORTED_IMAGE_SUFFIXES:
            raise OCRProfileInputError(f"Unsupported image extension: {path.suffix}")
        return (path,)
    if not path.is_dir():
        raise OCRProfileInputError(f"Input path is not a file or directory: {path}")

    images = tuple(
        sorted(
            (
                candidate
                for candidate in path.iterdir()
                if candidate.is_file()
                and candidate.suffix.casefold() in _SUPPORTED_IMAGE_SUFFIXES
            ),
            key=lambda candidate: candidate.name.casefold(),
        )
    )
    if not images:
        raise OCRProfileInputError(
            f"Directory contains no supported page images: {path}"
        )
    page_ids = [image.stem for image in images]
    duplicates = sorted(
        page_id for page_id in set(page_ids) if page_ids.count(page_id) > 1
    )
    if duplicates:
        raise OCRProfileInputError(
            "Page image stems must be unique to avoid artifact overwrite: "
            + ", ".join(duplicates)
        )
    return images


def generate_profiles(image: np.ndarray) -> dict[str, np.ndarray]:
    """Generate the five required deterministic preprocessing variants.

    Args:
        image: Three-channel BGR ``uint8`` image.

    Returns:
        Profile names mapped to independent two-dimensional ``uint8`` arrays.

    Raises:
        OCRProfileInputError: If the image shape, type, or size is unsupported.
        OCRProfileExecutionError: If OpenCV fails during preprocessing.
    """
    _validate_bgr_image(image)
    try:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        median = cv2.medianBlur(grayscale, 3)
        block_size = _adaptive_block_size(grayscale.shape)
        adaptive_without_clahe = _adaptive_threshold(median, block_size)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(grayscale)
        enhanced_median = cv2.medianBlur(enhanced, 3)
        clahe_adaptive = _adaptive_threshold(enhanced_median, block_size)

        _, otsu = cv2.threshold(
            grayscale,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        line_removed = _remove_horizontal_lines(adaptive_without_clahe)
    except cv2.error as exc:
        raise OCRProfileExecutionError(
            "OpenCV failed while generating preprocessing profiles"
        ) from exc

    return {
        "grayscale": grayscale.copy(),
        "clahe_median_adaptive": clahe_adaptive,
        "median_adaptive_no_clahe": adaptive_without_clahe,
        "otsu": otsu,
        "adaptive_line_removal": line_removed,
    }


def summarize_ocr_data(
    data: Mapping[str, Sequence[Any]],
    *,
    low_confidence_threshold: float = _DEFAULT_LOW_CONFIDENCE_THRESHOLD,
) -> OCRStatistics:
    """Reconstruct ordered text and summarize non-empty token confidence.

    Tesseract confidence is engine-specific and uncalibrated. These statistics
    support diagnosis only and must not be treated as transcription accuracy.
    """
    _validate_confidence_threshold(low_confidence_threshold)
    required = ("text", "conf")
    if any(field not in data for field in required):
        raise OCRProfileExecutionError("Tesseract data is missing text or conf")
    row_count = len(data["text"])
    if len(data["conf"]) != row_count:
        raise OCRProfileExecutionError("Tesseract data columns have unequal lengths")

    line_tokens: dict[tuple[str, ...], list[str]] = {}
    confidences: list[float] = []
    for index in range(row_count):
        text = str(data["text"][index])
        if not text.strip():
            continue
        try:
            confidence = float(data["conf"][index])
        except (TypeError, ValueError) as exc:
            raise OCRProfileExecutionError(
                f"Invalid Tesseract confidence at row {index}"
            ) from exc
        if not np.isfinite(confidence):
            raise OCRProfileExecutionError(
                f"Non-finite Tesseract confidence at row {index}"
            )

        key = _line_key(data, index)
        line_tokens.setdefault(key, []).append(text)
        confidences.append(confidence)

    raw_text = "\n".join(" ".join(tokens) for tokens in line_tokens.values())
    if not confidences:
        return OCRStatistics(raw_text, None, None, 0.0, 0)
    low_count = sum(
        confidence < low_confidence_threshold for confidence in confidences
    )
    return OCRStatistics(
        raw_text=raw_text,
        mean_confidence=float(statistics.fmean(confidences)),
        median_confidence=float(statistics.median(confidences)),
        low_confidence_token_ratio=float(low_count / len(confidences)),
        non_empty_token_count=len(confidences),
    )


def load_ground_truth(path: str | Path) -> dict[str, str]:
    """Load page ground truth from an evaluation list or page mapping.

    Accepted forms are a mapping of page identifier to transcription, or a list
    of objects containing ``ground_truth`` and either ``sample_id`` or ``page``.
    File extensions in identifiers are ignored when matching page image stems.
    """
    ground_truth_path = Path(path).expanduser()
    try:
        payload = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise OCRProfileInputError(
            f"Could not read ground-truth JSON: {ground_truth_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise OCRProfileInputError(
            f"Malformed ground-truth JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc

    entries: list[tuple[object, object]] = []
    if isinstance(payload, dict):
        for identifier, value in payload.items():
            ground_truth = value.get("ground_truth") if isinstance(value, dict) else value
            entries.append((identifier, ground_truth))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            if not isinstance(value, dict):
                raise OCRProfileInputError(
                    f"Ground-truth entry {index} must be an object"
                )
            identifier = value.get("sample_id", value.get("page"))
            entries.append((identifier, value.get("ground_truth")))
    else:
        raise OCRProfileInputError(
            "Ground-truth JSON must be an object or list of objects"
        )

    result: dict[str, str] = {}
    for index, (identifier, transcription) in enumerate(entries):
        if not isinstance(identifier, str) or not identifier:
            raise OCRProfileInputError(
                f"Ground-truth entry {index} requires sample_id or page"
            )
        if not isinstance(transcription, str):
            raise OCRProfileInputError(
                f"Ground-truth entry {index} requires string ground_truth"
            )
        page_id = Path(identifier).stem
        if page_id in result:
            raise OCRProfileInputError(f"Duplicate ground truth for page: {page_id}")
        result[page_id] = transcription
    return result


def run_experiment(
    input_path: str | Path,
    *,
    output_directory: str | Path = _DEFAULT_OUTPUT_DIRECTORY,
    ground_truth_path: str | Path | None = None,
    low_confidence_threshold: float = _DEFAULT_LOW_CONFIDENCE_THRESHOLD,
) -> tuple[ExperimentRecord, ...]:
    """Run every profile/PSM combination and write artifacts and reports."""
    _validate_confidence_threshold(low_confidence_threshold)
    image_paths = collect_image_paths(input_path)
    ground_truth = (
        load_ground_truth(ground_truth_path) if ground_truth_path is not None else {}
    )
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    records: list[ExperimentRecord] = []
    for image_path in image_paths:
        page_id = image_path.stem
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise OCRProfileInputError(f"OpenCV could not decode image: {image_path}")
        profiles = generate_profiles(image)
        page_directory = output_path / page_id
        page_directory.mkdir(parents=True, exist_ok=True)

        for profile_name in _PROFILE_NAMES:
            processed = profiles[profile_name]
            processed_path = page_directory / f"{profile_name}.png"
            if not cv2.imwrite(str(processed_path), processed):
                raise OCRProfileExecutionError(
                    f"Could not write processed image: {processed_path}"
                )

            for psm in _PSM_MODES:
                LOGGER.info(
                    "Running Tesseract for page=%s profile=%s psm=%d",
                    page_id,
                    profile_name,
                    psm,
                )
                data = _run_tesseract(processed, psm)
                stats = summarize_ocr_data(
                    data,
                    low_confidence_threshold=low_confidence_threshold,
                )
                record = ExperimentRecord(
                    page=page_id,
                    source_image=str(image_path),
                    profile=profile_name,
                    psm=psm,
                    processed_image=str(processed_path),
                    raw_text=stats.raw_text,
                    mean_confidence=stats.mean_confidence,
                    median_confidence=stats.median_confidence,
                    low_confidence_token_ratio=stats.low_confidence_token_ratio,
                    non_empty_token_count=stats.non_empty_token_count,
                    ground_truth_available=page_id in ground_truth,
                )
                if page_id in ground_truth:
                    record = _add_ground_truth_metrics(
                        record,
                        ground_truth[page_id],
                    )
                records.append(record)

    ranked_records, overall_ranking = _rank_records(tuple(records))
    _write_reports(
        output_path,
        ranked_records,
        overall_ranking,
        low_confidence_threshold,
    )
    return ranked_records


def _validate_bgr_image(image: np.ndarray) -> None:
    """Validate the BGR input needed by all experimental profiles."""
    if not isinstance(image, np.ndarray):
        raise OCRProfileInputError("image must be a NumPy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise OCRProfileInputError(f"Expected BGR image, received shape {image.shape}")
    if image.dtype != np.uint8:
        raise OCRProfileInputError(f"Expected uint8 image, received {image.dtype}")
    if min(image.shape[:2]) < 3:
        raise OCRProfileInputError("Image must be at least 3x3 pixels")


def _adaptive_block_size(shape: tuple[int, ...]) -> int:
    """Return the largest valid odd adaptive-threshold window up to 31."""
    block_size = min(31, min(shape[:2]))
    if block_size % 2 == 0:
        block_size -= 1
    if block_size < 3:
        raise OCRProfileInputError("Image is too small for adaptive thresholding")
    return block_size


def _adaptive_threshold(image: np.ndarray, block_size: int) -> np.ndarray:
    """Apply the experiment's Gaussian adaptive threshold."""
    return cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        15,
    )


def _remove_horizontal_lines(binary: np.ndarray) -> np.ndarray:
    """Remove only long horizontal foreground runs from a binary image.

    Even this conservative operation can damage genuine handwriting strokes, so
    it is an experiment profile rather than a replacement for the assessment
    preprocessing pipeline.
    """
    foreground = cv2.bitwise_not(binary)
    kernel_width = min(binary.shape[1], max(25, round(binary.shape[1] * 0.08)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    horizontal_lines = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.subtract(foreground, horizontal_lines)
    return cv2.bitwise_not(cleaned)


def _run_tesseract(
    image: np.ndarray,
    psm: int,
) -> Mapping[str, Sequence[Any]]:
    """Run Tesseract once and translate executable failures clearly."""
    try:
        data = pytesseract.image_to_data(
            image,
            config=f"--psm {psm}",
            output_type=Output.DICT,
        )
    except TesseractNotFoundError as exc:
        raise OCRProfileExecutionError("Tesseract executable was not found") from exc
    except TesseractError as exc:
        raise OCRProfileExecutionError(f"Tesseract failed for PSM {psm}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise OCRProfileExecutionError("Tesseract returned an unexpected data type")
    return data


def _line_key(data: Mapping[str, Sequence[Any]], index: int) -> tuple[str, ...]:
    """Build a stable line key, falling back to row position when unavailable."""
    hierarchy_fields = ("page_num", "block_num", "par_num", "line_num")
    values: list[str] = []
    for field in hierarchy_fields:
        column = data.get(field)
        values.append(str(column[index]) if column is not None and index < len(column) else "")
    if not values[-1]:
        top_column = data.get("top")
        fallback = (
            str(top_column[index])
            if top_column is not None and index < len(top_column)
            else str(index)
        )
        values.append(fallback)
    return tuple(values)


def _validate_confidence_threshold(value: float) -> None:
    """Require a finite confidence threshold between zero and one hundred."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
        or not 0 <= value <= 100
    ):
        raise OCRProfileInputError(
            "low_confidence_threshold must be between 0 and 100"
        )


def _add_ground_truth_metrics(
    record: ExperimentRecord,
    ground_truth: str,
) -> ExperimentRecord:
    """Add existing evaluation metrics and SFS to one experiment record."""
    evaluation = evaluate_sample(
        EvaluationSample(record.page, ground_truth, record.raw_text)
    )
    return replace(
        record,
        cer=evaluation.cer,
        wer=evaluation.wer,
        sentence_f1=evaluation.sentence_f1,
        composite_score=evaluation.composite_score,
        sfs=sfs(ground_truth, record.raw_text),
    )


def _quality_sort_key(record: ExperimentRecord) -> tuple[Any, ...]:
    """Rank by ground-truth quality only; confidence is deliberately excluded."""
    assert record.composite_score is not None
    assert record.sfs is not None
    assert record.sentence_f1 is not None
    assert record.wer is not None
    assert record.cer is not None
    return (
        -record.composite_score,
        -record.sfs,
        -record.sentence_f1,
        record.wer,
        record.cer,
        record.profile,
        record.psm,
    )


def _rank_records(
    records: tuple[ExperimentRecord, ...],
) -> tuple[tuple[ExperimentRecord, ...], list[dict[str, Any]]]:
    """Assign per-page and macro configuration ranks from ground-truth metrics."""
    page_rank: dict[tuple[str, str, int], int] = {}
    by_page: dict[str, list[ExperimentRecord]] = defaultdict(list)
    for record in records:
        if record.ground_truth_available:
            by_page[record.page].append(record)
    for page, page_records in by_page.items():
        for rank, record in enumerate(sorted(page_records, key=_quality_sort_key), 1):
            page_rank[(page, record.profile, record.psm)] = rank

    by_configuration: dict[tuple[str, int], list[ExperimentRecord]] = defaultdict(list)
    for record in records:
        if record.ground_truth_available:
            by_configuration[(record.profile, record.psm)].append(record)

    aggregates: list[dict[str, Any]] = []
    for (profile, psm), configuration_records in by_configuration.items():
        aggregates.append(
            {
                "profile": profile,
                "psm": psm,
                "evaluated_pages": len(configuration_records),
                "mean_cer": statistics.fmean(
                    record.cer for record in configuration_records if record.cer is not None
                ),
                "mean_wer": statistics.fmean(
                    record.wer for record in configuration_records if record.wer is not None
                ),
                "mean_sentence_f1": statistics.fmean(
                    record.sentence_f1
                    for record in configuration_records
                    if record.sentence_f1 is not None
                ),
                "mean_composite_score": statistics.fmean(
                    record.composite_score
                    for record in configuration_records
                    if record.composite_score is not None
                ),
                "mean_sfs": statistics.fmean(
                    record.sfs for record in configuration_records if record.sfs is not None
                ),
            }
        )
    aggregates.sort(
        key=lambda aggregate: (
            -aggregate["mean_composite_score"],
            -aggregate["mean_sfs"],
            -aggregate["mean_sentence_f1"],
            aggregate["mean_wer"],
            aggregate["mean_cer"],
            aggregate["profile"],
            aggregate["psm"],
        )
    )
    overall_rank: dict[tuple[str, int], int] = {}
    for rank, aggregate in enumerate(aggregates, 1):
        aggregate["rank"] = rank
        overall_rank[(aggregate["profile"], aggregate["psm"])] = rank

    ranked = tuple(
        replace(
            record,
            page_rank=page_rank.get((record.page, record.profile, record.psm)),
            overall_rank=overall_rank.get((record.profile, record.psm)),
        )
        for record in records
    )
    return ranked, aggregates


def _write_reports(
    output_directory: Path,
    records: tuple[ExperimentRecord, ...],
    overall_ranking: list[dict[str, Any]],
    low_confidence_threshold: float,
) -> None:
    """Write complete JSON and flat CSV experiment reports."""
    json_path = output_directory / "results.json"
    csv_path = output_directory / "results.csv"
    payload = {
        "notes": [
            "Tesseract confidence is not a reliable proxy for transcription accuracy.",
            "Ruled-line removal can damage genuine character strokes.",
            "No winner is selected from confidence alone.",
        ],
        "low_confidence_threshold": low_confidence_threshold,
        "psm_modes": list(_PSM_MODES),
        "profiles": list(_PROFILE_NAMES),
        "ranking_basis": (
            "Ground-truth composite score, SFS, Sentence F1, WER, then CER. "
            "Ranking is omitted when ground truth is unavailable."
        ),
        "overall_ranking": overall_ranking,
        "results": [asdict(record) for record in records],
    }
    try:
        json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
            field_names = [field.name for field in fields(ExperimentRecord)]
            writer = csv.DictWriter(csv_file, fieldnames=field_names)
            writer.writeheader()
            writer.writerows(asdict(record) for record in records)
    except OSError as exc:
        raise OCRProfileExecutionError(
            f"Could not write experiment reports under {output_directory}"
        ) from exc


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare five OCR preprocessing profiles across Tesseract PSM "
            "4, 6, 11, and 12. Confidence is diagnostic, not an accuracy proxy."
        )
    )
    parser.add_argument("input", type=Path, help="Image or directory of page images")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        help="Optional JSON page ground truth for accuracy metrics and ranking",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIRECTORY,
        help="Artifact directory (default: artifacts/ocr_profiles)",
    )
    parser.add_argument(
        "--low-confidence-threshold",
        type=float,
        default=_DEFAULT_LOW_CONFIDENCE_THRESHOLD,
        help="Diagnostic token-confidence threshold (default: 60)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the profile experiment CLI and return a process exit code."""
    arguments = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        records = run_experiment(
            arguments.input,
            output_directory=arguments.output_dir,
            ground_truth_path=arguments.ground_truth,
            low_confidence_threshold=arguments.low_confidence_threshold,
        )
    except OCRProfileError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(
        f"Wrote {len(records)} experiment rows to "
        f"{arguments.output_dir / 'results.json'} and "
        f"{arguments.output_dir / 'results.csv'}"
    )
    if arguments.ground_truth is None:
        print("No ground truth supplied; no winner or accuracy ranking was selected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
