"""Deterministic tests for the OCR profile experiment harness."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from tools import compare_ocr_profiles
from tools.compare_ocr_profiles import (
    ExperimentRecord,
    OCRProfileInputError,
    collect_image_paths,
    generate_profiles,
    load_ground_truth,
    run_experiment,
    summarize_ocr_data,
)


def _tesseract_data(text: str = "hello!!! world") -> dict[str, list[Any]]:
    """Create deterministic pytesseract-style word data."""
    words = text.split()
    return {
        "text": words,
        "conf": [90.0, 30.0][: len(words)],
        "page_num": [1] * len(words),
        "block_num": [1] * len(words),
        "par_num": [1] * len(words),
        "line_num": [1] * len(words),
        "top": [10] * len(words),
    }


def test_generate_profiles_returns_all_required_variants() -> None:
    """All profiles preserve dimensions and use uint8 output."""
    image = np.full((120, 240, 3), 255, dtype=np.uint8)
    cv2.putText(image, "Text", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    profiles = generate_profiles(image)

    assert tuple(profiles) == (
        "grayscale",
        "clahe_median_adaptive",
        "median_adaptive_no_clahe",
        "otsu",
        "adaptive_line_removal",
    )
    assert all(profile.shape == (120, 240) for profile in profiles.values())
    assert all(profile.dtype == np.uint8 for profile in profiles.values())
    for name, profile in profiles.items():
        if name != "grayscale":
            assert set(np.unique(profile)).issubset({0, 255})


def test_collect_image_paths_is_sorted_and_rejects_empty_directory(
    tmp_path: Path,
) -> None:
    """Directory inputs are filtered and ordered deterministically."""
    (tmp_path / "b.png").write_bytes(b"placeholder")
    (tmp_path / "a.jpg").write_bytes(b"placeholder")
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")

    assert tuple(path.name for path in collect_image_paths(tmp_path)) == (
        "a.jpg",
        "b.png",
    )

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(OCRProfileInputError, match="no supported"):
        collect_image_paths(empty)


def test_summarize_ocr_data_reconstructs_text_and_confidence() -> None:
    """Only non-empty OCR tokens contribute to confidence statistics."""
    data = {
        "text": ["hello", "world", "", "next"],
        "conf": [90, 30, -1, 60],
        "page_num": [1, 1, 1, 1],
        "block_num": [1, 1, 1, 1],
        "par_num": [1, 1, 1, 1],
        "line_num": [1, 1, 1, 2],
    }

    result = summarize_ocr_data(data, low_confidence_threshold=60)

    assert result.raw_text == "hello world\nnext"
    assert result.mean_confidence == 60.0
    assert result.median_confidence == 60.0
    assert result.low_confidence_token_ratio == pytest.approx(1 / 3)
    assert result.non_empty_token_count == 3


def test_load_ground_truth_supports_evaluation_and_page_formats(tmp_path: Path) -> None:
    """Existing evaluation lists and page filenames map to image stems."""
    path = tmp_path / "truth.json"
    path.write_text(
        json.dumps(
            [
                {"sample_id": "page-1", "ground_truth": "first"},
                {"page": "page-2.png", "ground_truth": "second"},
            ]
        ),
        encoding="utf-8",
    )

    assert load_ground_truth(path) == {"page-1": "first", "page-2": "second"}


def test_ranking_uses_ground_truth_metrics_not_confidence() -> None:
    """A lower-confidence accurate row outranks a high-confidence inaccurate row."""
    accurate = ExperimentRecord(
        page="page",
        source_image="page.png",
        profile="accurate",
        psm=6,
        processed_image="accurate.png",
        raw_text="hello!!!",
        mean_confidence=10.0,
        median_confidence=10.0,
        low_confidence_token_ratio=1.0,
        non_empty_token_count=1,
        ground_truth_available=True,
        cer=0.0,
        wer=0.0,
        sentence_f1=1.0,
        composite_score=1.0,
        sfs=1.0,
    )
    inaccurate = ExperimentRecord(
        page="page",
        source_image="page.png",
        profile="inaccurate",
        psm=6,
        processed_image="inaccurate.png",
        raw_text="wrong",
        mean_confidence=99.0,
        median_confidence=99.0,
        low_confidence_token_ratio=0.0,
        non_empty_token_count=1,
        ground_truth_available=True,
        cer=1.0,
        wer=1.0,
        sentence_f1=0.0,
        composite_score=0.0,
        sfs=0.0,
    )

    ranked, _ = compare_ocr_profiles._rank_records((inaccurate, accurate))

    ranks = {record.profile: record.page_rank for record in ranked}
    assert ranks == {"inaccurate": 2, "accurate": 1}


def test_run_experiment_writes_images_json_and_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mocked one-page run emits five images and twenty OCR result rows."""
    image = np.full((100, 220, 3), 255, dtype=np.uint8)
    cv2.putText(image, "hello", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    input_path = tmp_path / "page-1.png"
    assert cv2.imwrite(str(input_path), image)
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(
        json.dumps([{"sample_id": "page-1", "ground_truth": "hello!!! world"}]),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        compare_ocr_profiles,
        "_run_tesseract",
        lambda _image, _psm: _tesseract_data(),
    )
    output_path = tmp_path / "artifacts" / "ocr_profiles"

    records = run_experiment(
        input_path,
        output_directory=output_path,
        ground_truth_path=truth_path,
    )

    assert len(records) == 20
    page_path = output_path / "page-1"
    assert len(tuple(page_path.glob("*.png"))) == 5
    payload = json.loads((output_path / "results.json").read_text(encoding="utf-8"))
    assert len(payload["results"]) == 20
    assert len(payload["overall_ranking"]) == 20
    assert "not a reliable proxy" in payload["notes"][0]
    assert "damage genuine character strokes" in payload["notes"][1]
    with (output_path / "results.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 20
    assert all(record.cer == 0.0 for record in records)
    assert all(record.sfs == 1.0 for record in records)
