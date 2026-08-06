"""Unit tests for optional PaddleOCR integration without model downloads."""

from __future__ import annotations

import importlib.util
import itertools
import os
from collections import UserDict
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from typing import Any
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from app.ocr import factory as ocr_factory
from app.ocr.engine import OCRToken
from app.ocr.factory import create_ocr_engine
from app.ocr.ocr_pipeline import extract_text_and_sentences
from app.ocr.paddle_engine import (
    InferenceImage,
    PaddleOCRDataError,
    PaddleOCREngine,
    PaddleOCRNotInstalledError,
    reconstruct_lines,
    reconstruct_reading_order,
    remap_box,
    resize_for_inference,
)


def _result(
    *,
    texts: list[str] | None = None,
    scores: list[Any] | None = None,
    boxes: list[list[int]] | None = None,
) -> list[dict[str, Any]]:
    return [
        {
            "res": {
                "rec_texts": texts if texts is not None else ["hello"],
                "rec_scores": scores if scores is not None else [0.42],
                "rec_boxes": boxes if boxes is not None else [[10, 20, 50, 40]],
            }
        }
    ]


def _engine_with_result(result: Any) -> tuple[PaddleOCREngine, Mock, Mock]:
    model = Mock()
    model.predict.return_value = result
    factory = Mock(return_value=model)
    return PaddleOCREngine(model_factory=factory), factory, model


class _JSONResult:
    def __init__(self, payload: Any) -> None:
        self._payload = payload
        self.json_calls = 0

    @property
    def json(self) -> Any:
        self.json_calls += 1
        return self._payload


def test_missing_optional_dependency_has_install_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.ocr.paddle_engine.importlib.import_module",
        Mock(side_effect=ModuleNotFoundError("paddleocr")),
    )
    engine = PaddleOCREngine()

    with pytest.raises(PaddleOCRNotInstalledError, match="uv sync --extra paddle"):
        engine.extract(np.zeros((10, 10), dtype=np.uint8))


def test_model_is_lazy_and_initialized_only_once() -> None:
    engine, factory, model = _engine_with_result(_result())
    assert factory.call_count == 0

    engine.extract(np.zeros((10, 10), dtype=np.uint8))
    engine.extract(np.zeros((10, 10), dtype=np.uint8))

    factory.assert_called_once()
    assert model.predict.call_count == 2


def test_initialization_is_thread_safe_and_reuses_one_model() -> None:
    model = Mock()
    model.predict.return_value = _result()
    factory_entered = Event()
    release_factory = Event()

    def factory(**_options: Any) -> Mock:
        factory_entered.set()
        assert release_factory.wait(timeout=2)
        return model

    engine = PaddleOCREngine(model_factory=Mock(side_effect=factory))
    image = np.zeros((10, 10), dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(engine.extract, image)
        assert factory_entered.wait(timeout=2)
        second = executor.submit(engine.extract, image)
        release_factory.set()
        first.result(timeout=2)
        second.result(timeout=2)

    assert engine._model_factory.call_count == 1
    assert model.predict.call_count == 2


def test_predict_calls_are_serialized_with_a_separate_lock() -> None:
    first_predict_entered = Event()
    release_first_predict = Event()
    state_lock = Lock()
    active_predictions = 0
    maximum_active = 0

    def predict(_image: np.ndarray) -> list[dict[str, Any]]:
        nonlocal active_predictions, maximum_active
        with state_lock:
            active_predictions += 1
            maximum_active = max(maximum_active, active_predictions)
            call_number = model.predict.call_count
        if call_number == 1:
            first_predict_entered.set()
            assert release_first_predict.wait(timeout=2)
        with state_lock:
            active_predictions -= 1
        return _result()

    model = Mock()
    model.predict.side_effect = predict
    engine = PaddleOCREngine(model_factory=Mock(return_value=model))
    image = np.zeros((10, 10), dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(engine.extract, image)
        assert first_predict_entered.wait(timeout=2)
        second = executor.submit(engine.extract, image)
        assert not second.done()
        assert model.predict.call_count == 1
        release_first_predict.set()
        first.result(timeout=2)
        second.result(timeout=2)

    assert maximum_active == 1


def test_confidence_boxes_and_neutral_identifiers_are_converted() -> None:
    engine, _, _ = _engine_with_result(_result())

    extraction = engine.extract(np.zeros((100, 100), dtype=np.uint8))

    token = extraction.tokens[0]
    assert token.confidence == 42.0
    assert (token.left, token.top, token.width, token.height) == (10, 20, 40, 20)
    assert (token.page_num, token.block_num, token.paragraph_num, token.line_num) == (
        1,
        0,
        0,
        1,
    )


@pytest.mark.parametrize(
    ("paddle_confidence", "normalized"),
    [
        (0.0, 0.0),
        (0.5, 50.0),
        (1.0, 100.0),
        (0.123456789, 12.3456789),
    ],
)
def test_confidence_is_normalized_to_shared_scale(
    paddle_confidence: float, normalized: float
) -> None:
    engine, _, _ = _engine_with_result(_result(scores=[paddle_confidence]))

    token = engine.extract(np.zeros((100, 100), dtype=np.uint8)).tokens[0]

    assert token.confidence == pytest.approx(normalized)


@pytest.mark.parametrize(
    "confidence",
    [True, False, "0.5", None, np.nan, np.inf, -np.inf, -0.01, 1.01],
)
def test_invalid_confidence_is_rejected(confidence: Any) -> None:
    engine, _, _ = _engine_with_result(_result(scores=[confidence]))

    with pytest.raises(PaddleOCRDataError, match="score"):
        engine.extract(np.zeros((100, 100), dtype=np.uint8))


def test_large_image_is_downscaled_without_mutating_source() -> None:
    image = np.arange(3000 * 1000, dtype=np.uint8).reshape(3000, 1000)
    original = image.copy()

    inference = resize_for_inference(image, 2000)

    assert inference.image.shape == (2000, 667)
    assert (inference.original_width, inference.original_height) == (1000, 3000)
    assert (inference.inference_width, inference.inference_height) == (667, 2000)
    assert inference.scale_x == pytest.approx(0.667)
    assert inference.scale_y == pytest.approx(2 / 3)
    assert np.array_equal(image, original)


def test_small_image_is_copied_without_upscaling() -> None:
    image = np.zeros((100, 200), dtype=np.uint8)
    inference = resize_for_inference(image, 2000)

    assert inference.image.shape == image.shape
    assert inference.image is not image
    assert (inference.scale_x, inference.scale_y) == (1.0, 1.0)


def test_large_landscape_image_preserves_aspect_ratio() -> None:
    inference = resize_for_inference(np.zeros((1000, 4000), dtype=np.uint8), 2000)

    assert inference.image.shape == (500, 2000)
    assert inference.scale_x == 0.5
    assert inference.scale_y == 0.5


def test_downscaling_uses_inter_area(monkeypatch: pytest.MonkeyPatch) -> None:
    resized = np.zeros((500, 2000), dtype=np.uint8)
    resize = Mock(return_value=resized)
    monkeypatch.setattr("app.ocr.paddle_engine.cv2.resize", resize)

    inference = resize_for_inference(np.zeros((1000, 4000), dtype=np.uint8), 2000)

    assert inference.image is resized
    assert resize.call_args.args[1] == (2000, 500)
    assert resize.call_args.kwargs["interpolation"] == cv2.INTER_AREA


@pytest.mark.parametrize("value", [0, -1, True, False, 2000.0, "2000"])
def test_invalid_max_side_is_rejected(value: Any) -> None:
    image = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError, match="positive integer"):
        resize_for_inference(image, value)
    with pytest.raises(ValueError, match="positive integer"):
        PaddleOCREngine(max_inference_side=value)


@pytest.mark.parametrize(
    "image",
    [
        np.empty((0, 10), dtype=np.uint8),
        np.empty((10, 0), dtype=np.uint8),
        np.zeros((10,), dtype=np.uint8),
        np.zeros((10, 10), dtype=np.float32),
        np.zeros((10, 10, 2), dtype=np.uint8),
    ],
)
def test_invalid_images_are_rejected(image: np.ndarray) -> None:
    with pytest.raises(ValueError):
        resize_for_inference(image)


def _geometry() -> InferenceImage:
    return InferenceImage(
        image=np.zeros((50, 100), dtype=np.uint8),
        original_width=200,
        original_height=100,
        inference_width=100,
        inference_height=50,
        scale_x=0.5,
        scale_y=0.5,
    )


@pytest.mark.parametrize(
    ("box", "expected"),
    [
        ([10, 5, 50, 25], (20, 10, 80, 40)),
        (np.asarray([10, 5, 50, 25]), (20, 10, 80, 40)),
        ([-10, -5, 20, 10], (0, 0, 40, 20)),
        ([80, 40, 120, 70], (160, 80, 40, 20)),
        ([110, 60, 130, 80], (200, 100, 0, 0)),
        ([-30, -20, -10, -5], (0, 0, 0, 0)),
    ],
)
def test_box_remapping_is_clamped_and_non_negative(
    box: Any, expected: tuple[int, int, int, int]
) -> None:
    assert remap_box(box, _geometry()) == expected


@pytest.mark.parametrize("box", [[0, 1], [0, 1, 2, 3, 4], ["x", 0, 1, 1]])
def test_malformed_boxes_are_rejected(box: Any) -> None:
    with pytest.raises(PaddleOCRDataError):
        remap_box(box, _geometry())


def test_boxes_are_scaled_back_to_source_coordinates() -> None:
    engine, _, model = _engine_with_result(
        _result(boxes=[[10, 20, 50, 40]])
    )
    image = np.zeros((4000, 2000), dtype=np.uint8)

    extraction = engine.extract(image)

    assert model.predict.call_args.args[0].shape == (2000, 1000, 3)
    token = extraction.tokens[0]
    assert (token.left, token.top, token.width, token.height) == (20, 40, 80, 40)


def test_single_region_forms_one_line() -> None:
    lines = reconstruct_lines((_token("Only", left=10, top=20),))

    assert len(lines) == 1
    assert lines[0].line_num == 1
    assert lines[0].text == "Only"


def test_reversed_fragments_are_joined_left_to_right_with_one_space() -> None:
    lines = reconstruct_lines(
        (
            _token("world", left=400, top=100),
            _token("Hello", left=50, top=100),
        )
    )

    assert lines[0].text == "Hello world"
    assert [region.text for region in lines[0].regions] == ["Hello", "world"]


def test_multiple_lines_are_ordered_top_to_bottom() -> None:
    ordered = reconstruct_reading_order(
        (
            _token("Second", left=50, top=200),
            _token("world", left=400, top=100),
            _token("Hello", left=50, top=100),
        )
    )

    assert [token.text for token in ordered] == ["Hello", "world", "Second"]
    assert [token.line_num for token in ordered] == [1, 1, 2]


def test_unique_geometry_is_independent_of_detection_order() -> None:
    regions = (
        _token("first", left=10, top=10),
        _token("second", left=80, top=12),
        _token("third", left=10, top=80),
    )
    expected = ("first", "second", "third")

    for shuffled in itertools.permutations(regions):
        assert tuple(
            token.text for token in reconstruct_reading_order(shuffled)
        ) == expected


def test_slight_vertical_misalignment_stays_on_one_line() -> None:
    lines = reconstruct_lines(
        (
            _token("left", left=10, top=100, height=30),
            _token("right", left=80, top=110, height=30),
        )
    )

    assert [line.text for line in lines] == ["left right"]


def test_nearby_separate_rows_are_not_merged() -> None:
    lines = reconstruct_lines(
        (
            _token("upper", left=10, top=100, height=20),
            _token("lower", left=10, top=119, height=20),
        )
    )

    assert [line.text for line in lines] == ["upper", "lower"]


def test_different_heights_group_when_they_vertically_overlap() -> None:
    lines = reconstruct_lines(
        (
            _token("Tall", left=10, top=90, height=50),
            _token("short", left=80, top=105, height=20),
        )
    )

    assert [line.text for line in lines] == ["Tall short"]


def test_duplicate_text_occurrences_are_preserved() -> None:
    ordered = reconstruct_reading_order(
        (
            _token("same", left=80, top=10),
            _token("same", left=10, top=10),
        )
    )

    assert [token.text for token in ordered] == ["same", "same"]
    assert [token.left for token in ordered] == [10, 80]


def test_identical_geometry_uses_original_occurrence_as_final_tie_breaker() -> None:
    first = _token("first occurrence", left=10, top=10)
    second = _token("second occurrence", left=10, top=10)

    ordered = reconstruct_reading_order((first, second))

    assert [token.text for token in ordered] == ["first occurrence", "second occurrence"]


def test_blank_regions_are_ignored_safely() -> None:
    lines = reconstruct_lines(
        (
            _token(" ", left=1, top=1),
            _token("kept", left=10, top=10),
        )
    )

    assert [line.text for line in lines] == ["kept"]


def test_empty_reconstruction_is_empty() -> None:
    assert reconstruct_lines(()) == ()
    assert reconstruct_reading_order(()) == ()


def test_repeated_reconstruction_is_identical() -> None:
    regions = (
        _token("b", left=80, top=10),
        _token("a", left=10, top=12),
        _token("c", left=10, top=100),
    )

    results = [reconstruct_reading_order(regions) for _ in range(10)]

    assert all(result == results[0] for result in results[1:])


def test_pipeline_joins_same_line_fragments_and_flags_low_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, _, _ = _engine_with_result(
        _result(
            texts=["right", "left"],
            scores=[0.4, 0.9],
            boxes=[[80, 20, 120, 40], [10, 18, 50, 42]],
        )
    )
    monkeypatch.setattr(
        "app.ocr.ocr_pipeline.preprocess_image",
        Mock(return_value=np.zeros((100, 200), dtype=np.uint8)),
    )

    result = extract_text_and_sentences("page.png", engine=engine)

    assert result.raw_text == "left right"
    assert result.engine_name == "PaddleOCR"
    assert [region.text for region in result.low_confidence_regions] == ["right"]


def test_shared_mapping_preserves_occurrences_order_boxes_and_hierarchy() -> None:
    engine, _, _ = _engine_with_result(
        _result(
            texts=["same", "same", "Exact!!!"],
            scores=[0.8, 0.7, 0.6],
            boxes=[[80, 10, 100, 20], [10, 10, 30, 20], [10, 60, 50, 80]],
        )
    )

    extraction = engine.extract(np.zeros((100, 120), dtype=np.uint8))

    assert [token.text for token in extraction.tokens] == ["same", "same", "Exact!!!"]
    assert [token.left for token in extraction.tokens] == [10, 80, 10]
    assert [token.line_num for token in extraction.tokens] == [1, 1, 2]
    assert all(token.page_num == 1 for token in extraction.tokens)
    assert all(token.block_num == 0 for token in extraction.tokens)
    assert all(token.paragraph_num == 0 for token in extraction.tokens)
    assert (
        extraction.tokens[2].left,
        extraction.tokens[2].top,
        extraction.tokens[2].width,
        extraction.tokens[2].height,
    ) == (10, 60, 40, 20)


def test_pipeline_maps_raw_body_sentences_confidence_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.ocr.paddle_engine.metadata.version", Mock(return_value="3.7.0")
    )
    engine, _, _ = _engine_with_result(
        _result(
            texts=["sentence.", "Title", "Body"],
            scores=[0.95, 0.99, 0.59],
            boxes=[[80, 100, 150, 120], [10, 10, 60, 30], [10, 100, 60, 120]],
        )
    )
    monkeypatch.setattr(
        "app.ocr.ocr_pipeline.preprocess_image",
        Mock(return_value=np.zeros((200, 200), dtype=np.uint8)),
    )

    result = extract_text_and_sentences(
        "page.png", confidence_threshold=60, engine=engine
    )

    assert result.raw_text == "Title\nBody sentence."
    assert result.excluded_metadata_lines == ("Title",)
    assert result.body_text == "Body sentence."
    assert result.sentences == ("Body sentence.",)
    assert result.engine_name == "PaddleOCR"
    assert result.engine_version == "3.7.0"
    assert len(result.low_confidence_regions) == 1
    region = result.low_confidence_regions[0]
    assert region.text == "Body"
    assert region.confidence == 59.0
    assert (region.left, region.top, region.width, region.height) == (10, 100, 50, 20)
    assert region.line_num == 2


def test_empty_result_is_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    engine, _, _ = _engine_with_result([])
    extraction = engine.extract(np.zeros((10, 10), dtype=np.uint8))
    assert extraction.tokens == ()
    assert extraction.engine_name == "PaddleOCR"
    monkeypatch.setattr(
        "app.ocr.ocr_pipeline.preprocess_image",
        Mock(return_value=np.zeros((10, 10), dtype=np.uint8)),
    )

    pipeline_result = extract_text_and_sentences("empty.png", engine=engine)

    assert pipeline_result.raw_text == ""
    assert pipeline_result.body_text == ""
    assert pipeline_result.sentences == ()


def test_python_lists_and_numpy_arrays_are_supported() -> None:
    list_engine, _, _ = _engine_with_result(_result(texts=["list text"]))
    array_engine, _, _ = _engine_with_result(
        [
            {
                "res": {
                    "rec_texts": np.asarray(["array text"]),
                    "rec_scores": np.asarray([0.75]),
                    "rec_boxes": np.asarray([[1, 2, 11, 12]]),
                }
            }
        ]
    )

    assert list_engine.extract(np.zeros((20, 20), dtype=np.uint8)).tokens[0].text == (
        "list text"
    )
    assert array_engine.extract(np.zeros((20, 20), dtype=np.uint8)).tokens[0].text == (
        "array text"
    )


@pytest.mark.parametrize(
    "page",
    [
        {
            "res": {
                "rec_texts": ["wrapped dict"],
                "rec_scores": [0.9],
                "rec_boxes": [[1, 2, 11, 12]],
            }
        },
        {
            "rec_texts": ["direct dict"],
            "rec_scores": [0.9],
            "rec_boxes": [[1, 2, 11, 12]],
        },
        UserDict(
            {
                "res": UserDict(
                    {
                        "rec_texts": ["wrapped mapping"],
                        "rec_scores": [0.9],
                        "rec_boxes": [[1, 2, 11, 12]],
                    }
                )
            }
        ),
        UserDict(
            {
                "rec_texts": ["direct mapping"],
                "rec_scores": [0.9],
                "rec_boxes": [[1, 2, 11, 12]],
            }
        ),
    ],
)
def test_wrapped_and_direct_mapping_result_shapes_are_supported(page: Any) -> None:
    engine, _, _ = _engine_with_result([page])

    extraction = engine.extract(np.zeros((20, 20), dtype=np.uint8))

    assert len(extraction.tokens) == 1


@pytest.mark.parametrize("wrapped", [True, False])
def test_json_result_supports_wrapped_and_direct_shapes(wrapped: bool) -> None:
    recognition = {
        "rec_texts": ["json text"],
        "rec_scores": [0.9],
        "rec_boxes": [[1, 2, 11, 12]],
    }
    page = _JSONResult({"res": recognition} if wrapped else recognition)
    engine, _, _ = _engine_with_result([page])

    extraction = engine.extract(np.zeros((20, 20), dtype=np.uint8))

    assert extraction.tokens[0].text == "json text"
    assert page.json_calls == 1


def test_wrapped_shape_has_canonical_precedence_over_direct_fields() -> None:
    page = {
        "rec_texts": ["direct"],
        "rec_scores": [0.9],
        "rec_boxes": [[1, 2, 11, 12]],
        "res": {
            "rec_texts": ["wrapped"],
            "rec_scores": [0.8],
            "rec_boxes": [[2, 3, 12, 13]],
        },
    }
    engine, _, _ = _engine_with_result([page])

    extraction = engine.extract(np.zeros((20, 20), dtype=np.uint8))

    assert extraction.tokens[0].text == "wrapped"


@pytest.mark.parametrize(
    "page",
    [
        {"rec_texts": [], "rec_scores": []},
        {"res": {"rec_texts": [], "rec_scores": []}},
        {"res": []},
    ],
)
def test_incomplete_direct_or_wrapped_shapes_are_rejected(page: Any) -> None:
    engine, _, _ = _engine_with_result([page])

    with pytest.raises(PaddleOCRDataError):
        engine.extract(np.zeros((20, 20), dtype=np.uint8))


@pytest.mark.parametrize(
    "page",
    [
        {"res": {"rec_texts": [], "rec_scores": [], "rec_boxes": []}},
        {"rec_texts": [], "rec_scores": [], "rec_boxes": []},
    ],
)
def test_empty_recognition_arrays_are_valid_for_both_shapes(page: Any) -> None:
    engine, _, _ = _engine_with_result([page])

    extraction = engine.extract(np.zeros((20, 20), dtype=np.uint8))

    assert extraction.tokens == ()


def test_blank_text_is_ignored_without_normalizing_other_text() -> None:
    engine, _, _ = _engine_with_result(
        _result(
            texts=["  ", "Exact!!!  text"],
            scores=[0.1, 0.9],
            boxes=[[0, 0, 1, 1], [2, 2, 12, 12]],
        )
    )

    extraction = engine.extract(np.zeros((20, 20), dtype=np.uint8))

    assert [token.text for token in extraction.tokens] == ["Exact!!!  text"]


@pytest.mark.parametrize(
    "result",
    [
        None,
        123,
        [{}],
        [{"res": None}],
        [{"res": {}}],
        [{"res": {"rec_texts": ["a"], "rec_scores": [], "rec_boxes": []}}],
        [
            {
                "res": {
                    "rec_texts": ["a"],
                    "rec_scores": [1.2],
                    "rec_boxes": [[0, 0, 1, 1]],
                }
            }
        ],
        [
            {
                "res": {
                    "rec_texts": ["a"],
                    "rec_scores": [0.5],
                    "rec_boxes": [[0, 1]],
                }
            }
        ],
    ],
)
def test_malformed_results_raise_data_error(result: Any) -> None:
    engine, _, _ = _engine_with_result(result)
    with pytest.raises(PaddleOCRDataError):
        engine.extract(np.zeros((10, 10), dtype=np.uint8))


def test_package_version_uses_metadata_without_model_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    version_lookup = Mock(return_value="3.7.0")
    monkeypatch.setattr("app.ocr.paddle_engine.metadata.version", version_lookup)
    factory = Mock()

    engine = PaddleOCREngine(model_factory=factory)

    assert engine.version == "3.7.0"
    version_lookup.assert_called_once_with("paddleocr")
    factory.assert_not_called()


def test_explicit_paddle_request_never_returns_tesseract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tesseract = Mock(side_effect=AssertionError("Tesseract fallback invoked"))
    monkeypatch.setattr(ocr_factory, "TesseractEngine", tesseract)
    engine = create_ocr_engine("paddle")
    assert isinstance(engine, PaddleOCREngine)
    assert engine.name == "PaddleOCR"
    tesseract.assert_not_called()


def test_default_pipeline_engine_remains_tesseract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.zeros((10, 10), dtype=np.uint8)
    monkeypatch.setattr("app.ocr.ocr_pipeline.preprocess_image", Mock(return_value=image))
    monkeypatch.setattr(
        "app.ocr.ocr_pipeline.pytesseract.image_to_data",
        Mock(
            return_value={
                "text": [],
                "conf": [],
                "left": [],
                "top": [],
                "width": [],
                "height": [],
            }
        ),
    )
    monkeypatch.setattr(
        "app.ocr.ocr_pipeline.pytesseract.get_tesseract_version",
        Mock(return_value="5"),
    )

    result = extract_text_and_sentences("page.png")
    assert result.engine_name == "Tesseract"


@pytest.mark.skipif(
    os.environ.get("RUN_PADDLE_OCR_SMOKE") != "1"
    or importlib.util.find_spec("paddleocr") is None,
    reason="requires PaddleOCR and RUN_PADDLE_OCR_SMOKE=1",
)
def test_real_paddle_smoke() -> None:
    result = extract_text_and_sentences(
        "samples/testing_pages/page-1-small.png",
        engine=create_ocr_engine("paddle"),
    )
    assert result.engine_name == "PaddleOCR"
    assert result.raw_text


def _token(
    text: str,
    *,
    left: int,
    top: int,
    width: int = 50,
    height: int = 30,
) -> OCRToken:
    return OCRToken(text, 90.0, left, top, width, height, 1, 0, 0, None)
