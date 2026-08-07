"""Optional PaddleOCR PP-OCRv5 adapter with bounded CPU inference."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from numbers import Real
from threading import Lock
from typing import Any

import cv2
import numpy as np

from app.ocr.engine import OCRDataError, OCRExecutionError, OCRExtraction, OCRToken

LOGGER = logging.getLogger(__name__)

_DEFAULT_MAX_INFERENCE_SIDE = 2000
_MIN_VERTICAL_OVERLAP_RATIO = 0.30
_MAX_CENTER_DISTANCE_RATIO = 0.50
_NEUTRAL_HIERARCHY_ID = 0
_RECOGNITION_FIELDS = ("rec_texts", "rec_scores", "rec_boxes")
_PADDLE_OPTIONS: dict[str, Any] = {
    "text_detection_model_name": "PP-OCRv5_mobile_det",
    "text_recognition_model_name": "PP-OCRv5_mobile_rec",
    "use_doc_orientation_classify": False,
    "use_doc_unwarping": False,
    "use_textline_orientation": False,
    "device": "cpu",
    "enable_mkldnn": False,
    "cpu_threads": 1,
    "text_det_limit_type": "max",
    "text_det_limit_side_len": 1600,
}


class PaddleOCRNotInstalledError(ImportError):
    """Raised when the explicitly requested optional Paddle runtime is absent."""


class PaddleOCRDataError(OCRDataError):
    """Raised when PaddleOCR returns a malformed or incomplete result."""


@dataclass(frozen=True, slots=True)
class InferenceImage:
    """Request-local inference copy and its geometry relative to the source."""

    image: np.ndarray
    original_width: int
    original_height: int
    inference_width: int
    inference_height: int
    scale_x: float
    scale_y: float


@dataclass(frozen=True, slots=True)
class OrderedOCRLine:
    """One deterministic visual line reconstructed from recognized regions."""

    line_num: int
    regions: tuple[OCRToken, ...]

    @property
    def text(self) -> str:
        """Join exact nonblank region text with one separator."""
        return " ".join(region.text for region in self.regions)


@dataclass(frozen=True, slots=True)
class _IndexedRegion:
    """A region plus its final tie-breaker occurrence index."""

    region: OCRToken
    occurrence: int


class PaddleOCREngine:
    """Lazy, serialized local PaddleOCR adapter.

    Model construction and prediction are guarded by separate locks. Prediction
    is serialized because the native Paddle runtime's concurrent safety is not
    assumed; all parsed output remains request-local and immutable.
    """

    def __init__(
        self,
        *,
        max_inference_side: int = _DEFAULT_MAX_INFERENCE_SIDE,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        if (
            isinstance(max_inference_side, bool)
            or not isinstance(max_inference_side, int)
            or max_inference_side < 1
        ):
            raise ValueError("max_inference_side must be a positive integer")
        self._max_inference_side = max_inference_side
        self._model_factory = model_factory
        self._model: Any | None = None
        self._version = _installed_package_version()
        self._initialization_lock = Lock()
        self._inference_lock = Lock()

    @property
    def name(self) -> str:
        return "PaddleOCR"

    @property
    def version(self) -> str | None:
        return self._version

    def extract(self, image: np.ndarray) -> OCRExtraction:
        _validate_image(image)
        paddle_image = (
            cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
        )
        inference = resize_for_inference(
            paddle_image, self._max_inference_side
        )
        model = self._get_model()
        try:
            with self._inference_lock:
                result = model.predict(inference.image)
        except Exception as exc:
            raise OCRExecutionError(f"PaddleOCR inference failed: {exc}") from exc

        regions = _parse_paddle_results(result, inference=inference)
        tokens = reconstruct_reading_order(regions)
        return OCRExtraction(
            tokens=tokens,
            engine_name=self.name,
            engine_version=self.version,
        )

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._initialization_lock:
            if self._model is not None:
                return self._model
            factory = self._model_factory
            if factory is None:
                try:
                    module = importlib.import_module("paddleocr")
                    factory = module.PaddleOCR
                    if self._version is None:
                        self._version = getattr(module, "__version__", None)
                except (ImportError, AttributeError) as exc:
                    raise PaddleOCRNotInstalledError(
                        "PaddleOCR is optional and was explicitly requested. "
                        "Install it with `uv sync --extra paddle`; model files "
                        "download on first use."
                    ) from exc
            try:
                self._model = factory(**_PADDLE_OPTIONS)
            except (ImportError, ModuleNotFoundError) as exc:
                raise PaddleOCRNotInstalledError(
                    "The optional PaddleOCR runtime is incomplete. Install it "
                    "with `uv sync --extra paddle`."
                ) from exc
            return self._model


def resize_for_inference(
    image: np.ndarray, max_side: int = _DEFAULT_MAX_INFERENCE_SIDE
) -> InferenceImage:
    """Return an uncropped inference copy with complete scaling geometry."""
    _validate_image(image)
    if isinstance(max_side, bool) or not isinstance(max_side, int) or max_side < 1:
        raise ValueError("max_side must be a positive integer")
    height, width = image.shape[:2]
    largest_side = max(height, width)
    if largest_side <= max_side:
        return InferenceImage(
            image=image.copy(),
            original_width=width,
            original_height=height,
            inference_width=width,
            inference_height=height,
            scale_x=1.0,
            scale_y=1.0,
        )

    ratio = max_side / largest_side
    target_width = max(1, round(width * ratio))
    target_height = max(1, round(height * ratio))
    resized = cv2.resize(
        image, (target_width, target_height), interpolation=cv2.INTER_AREA
    )
    return InferenceImage(
        image=resized,
        original_width=width,
        original_height=height,
        inference_width=target_width,
        inference_height=target_height,
        scale_x=target_width / width,
        scale_y=target_height / height,
    )


def remap_box(
    value: Any, inference: InferenceImage, index: int = 0
) -> tuple[int, int, int, int]:
    """Map one Paddle box to clamped source-space left/top/width/height."""
    left, top, right, bottom = _box_edges(value, index)
    original_left = _clamp(round(left / inference.scale_x), 0, inference.original_width)
    original_top = _clamp(round(top / inference.scale_y), 0, inference.original_height)
    original_right = _clamp(
        round(right / inference.scale_x), 0, inference.original_width
    )
    original_bottom = _clamp(
        round(bottom / inference.scale_y), 0, inference.original_height
    )
    return (
        original_left,
        original_top,
        max(0, original_right - original_left),
        max(0, original_bottom - original_top),
    )


def reconstruct_reading_order(
    regions: Iterable[OCRToken],
) -> tuple[OCRToken, ...]:
    """Return regions flattened in deterministic reconstructed line order."""
    return tuple(region for line in reconstruct_lines(regions) for region in line.regions)


def reconstruct_lines(regions: Iterable[OCRToken]) -> tuple[OrderedOCRLine, ...]:
    """Group regions using relative vertical overlap and center-distance rules.

    A region joins a line when vertical overlap covers at least 30% of the
    shorter height, or when their vertical centers differ by no more than 50%
    of the larger height. These ratios scale with image resolution.
    """
    indexed = [
        _IndexedRegion(region, occurrence)
        for occurrence, region in enumerate(regions)
        if region.text.strip()
    ]
    indexed.sort(key=_region_geometry_key)
    grouped: list[list[_IndexedRegion]] = []

    for candidate in indexed:
        compatible: list[tuple[tuple[float, ...], list[_IndexedRegion]]] = []
        for creation_index, line in enumerate(grouped):
            metrics = _line_compatibility(candidate, line)
            if metrics is None:
                continue
            overlap_ratio, center_distance = metrics
            line_top = min(item.region.top for item in line)
            line_left = min(item.region.left for item in line)
            compatible.append(
                (
                    (
                        -overlap_ratio,
                        center_distance,
                        float(line_top),
                        float(line_left),
                        float(creation_index),
                    ),
                    line,
                )
            )
        if compatible:
            min(compatible, key=lambda item: item[0])[1].append(candidate)
        else:
            grouped.append([candidate])

    grouped.sort(key=_line_geometry_key)
    lines: list[OrderedOCRLine] = []
    for line_num, line in enumerate(grouped, start=1):
        line.sort(key=_horizontal_geometry_key)
        ordered_regions = tuple(
            _with_line_number(item.region, line_num) for item in line
        )
        lines.append(OrderedOCRLine(line_num=line_num, regions=ordered_regions))
    return tuple(lines)


def _line_compatibility(
    candidate: _IndexedRegion, line: list[_IndexedRegion]
) -> tuple[float, float] | None:
    token = candidate.region
    line_top = min(item.region.top for item in line)
    line_bottom = max(item.region.top + item.region.height for item in line)
    line_height = max(1, line_bottom - line_top)
    token_height = max(1, token.height)
    overlap = max(
        0, min(token.top + token_height, line_bottom) - max(token.top, line_top)
    )
    overlap_ratio = overlap / min(token_height, line_height)
    center_distance = abs(
        (token.top + token_height / 2) - (line_top + line_height / 2)
    )
    center_tolerance = max(token_height, line_height) * _MAX_CENTER_DISTANCE_RATIO
    if (
        overlap_ratio < _MIN_VERTICAL_OVERLAP_RATIO
        and center_distance > center_tolerance
    ):
        return None
    return overlap_ratio, center_distance


def _region_geometry_key(item: _IndexedRegion) -> tuple[float, ...]:
    region = item.region
    return (
        float(region.top),
        float(region.left),
        float(region.top + region.height),
        float(region.left + region.width),
        float(item.occurrence),
    )


def _horizontal_geometry_key(item: _IndexedRegion) -> tuple[float, ...]:
    region = item.region
    return (
        float(region.left),
        float(region.top),
        float(region.left + region.width),
        float(region.top + region.height),
        float(item.occurrence),
    )


def _line_geometry_key(line: list[_IndexedRegion]) -> tuple[float, ...]:
    return (
        float(min(item.region.top for item in line)),
        float(min(item.region.left for item in line)),
        float(min(item.occurrence for item in line)),
    )


def _with_line_number(region: OCRToken, line_num: int) -> OCRToken:
    return OCRToken(
        text=region.text,
        confidence=region.confidence,
        left=region.left,
        top=region.top,
        width=region.width,
        height=region.height,
        page_num=1,
        block_num=_NEUTRAL_HIERARCHY_ID,
        paragraph_num=_NEUTRAL_HIERARCHY_ID,
        line_num=line_num,
    )


def _parse_paddle_results(
    result: Any, *, inference: InferenceImage
) -> tuple[OCRToken, ...]:
    if result is None:
        raise PaddleOCRDataError("PaddleOCR returned no result collection")
    if isinstance(result, (str, bytes, Mapping)):
        pages: Iterable[Any] = (result,)
    else:
        try:
            pages = iter(result)
        except TypeError as exc:
            raise PaddleOCRDataError(
                "PaddleOCR returned a non-iterable result"
            ) from exc

    tokens: list[OCRToken] = []
    for page_index, page in enumerate(pages):
        mapping = _result_mapping(page, page_index)
        missing_fields = [field for field in _RECOGNITION_FIELDS if field not in mapping]
        if missing_fields:
            raise PaddleOCRDataError(
                f"PaddleOCR result is missing fields at page {page_index}: "
                + ", ".join(missing_fields)
            )
        texts = mapping["rec_texts"]
        scores = mapping["rec_scores"]
        boxes = mapping["rec_boxes"]
        if not _is_sequence(texts) or not _is_sequence(scores) or not _is_sequence(boxes):
            raise PaddleOCRDataError(
                f"Malformed PaddleOCR recognition fields at page {page_index}"
            )
        if not (len(texts) == len(scores) == len(boxes)):
            raise PaddleOCRDataError(
                f"PaddleOCR result columns have inconsistent lengths at page {page_index}"
            )
        for index, (text_value, score_value, box_value) in enumerate(
            zip(texts, scores, boxes, strict=True)
        ):
            text = str(text_value)
            if not text.strip():
                continue
            score = _finite_float(score_value, "score", index)
            if not 0.0 <= score <= 1.0:
                raise PaddleOCRDataError(
                    f"PaddleOCR score outside [0, 1] at row {index}"
                )
            left, top, width, height = remap_box(box_value, inference, index)
            tokens.append(
                OCRToken(
                    text=text,
                    confidence=score * 100.0,
                    left=left,
                    top=top,
                    width=width,
                    height=height,
                    page_num=1,
                    # Paddle does not report semantic block/paragraph IDs.
                    block_num=_NEUTRAL_HIERARCHY_ID,
                    paragraph_num=_NEUTRAL_HIERARCHY_ID,
                    line_num=None,
                )
            )
    return tuple(tokens)


def _result_mapping(value: Any, page_index: int) -> Mapping[str, Any]:
    """Return a normalized recognition mapping for one Paddle result page.

    PaddleOCR has emitted both a wrapped ``{"res": ...}`` shape and a direct
    recognition-field shape. The wrapped shape is canonical when both exist.
    """
    if isinstance(value, Mapping):
        normalized = _normalize_result_mapping(value, page_index)
        if normalized is not None:
            return normalized

    payload = getattr(value, "json", None)
    if callable(payload):
        payload = payload()
    if not isinstance(payload, Mapping):
        raise PaddleOCRDataError(f"Malformed PaddleOCR result at page {page_index}")
    normalized = _normalize_result_mapping(payload, page_index)
    if normalized is None:
        raise PaddleOCRDataError(
            f"PaddleOCR result is missing recognition fields at page {page_index}"
        )
    return normalized


def _normalize_result_mapping(
    mapping: Mapping[str, Any], page_index: int
) -> Mapping[str, Any] | None:
    if "res" in mapping:
        nested = mapping["res"]
        if not isinstance(nested, Mapping):
            raise PaddleOCRDataError(
                f"PaddleOCR 'res' is malformed at page {page_index}"
            )
        return nested
    if all(field in mapping for field in _RECOGNITION_FIELDS):
        return mapping
    return None


def _box_edges(value: Any, index: int) -> tuple[float, float, float, float]:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise PaddleOCRDataError(f"Invalid PaddleOCR box at row {index}") from exc
    if array.shape == (4,):
        left, top, right, bottom = array.tolist()
    elif array.ndim == 2 and array.shape[1] == 2 and array.shape[0] >= 4:
        left = float(array[:, 0].min())
        top = float(array[:, 1].min())
        right = float(array[:, 0].max())
        bottom = float(array[:, 1].max())
    else:
        raise PaddleOCRDataError(
            f"Invalid PaddleOCR box shape at row {index}: {array.shape}"
        )
    if not np.isfinite(array).all() or right < left or bottom < top:
        raise PaddleOCRDataError(f"Invalid PaddleOCR box coordinates at row {index}")
    return float(left), float(top), float(right), float(bottom)


def _finite_float(value: Any, field: str, index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise PaddleOCRDataError(f"Invalid PaddleOCR {field} at row {index}")
    result = float(value)
    if not np.isfinite(result):
        raise PaddleOCRDataError(f"Non-finite PaddleOCR {field} at row {index}")
    return result


def _installed_package_version() -> str | None:
    """Read distribution metadata without importing or initializing PaddleOCR."""
    try:
        return metadata.version("paddleocr")
    except metadata.PackageNotFoundError:
        return None


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (Sequence, np.ndarray)) and not isinstance(
        value, (str, bytes)
    )


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return min(maximum, max(minimum, value))


def _validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
        raise ValueError("PaddleOCR image must be a two- or three-dimensional array")
    if image.shape[0] < 1 or image.shape[1] < 1:
        raise ValueError("PaddleOCR image dimensions must be non-zero")
    if image.dtype != np.uint8:
        raise ValueError("PaddleOCR image must use uint8 pixels")
    if image.ndim == 3 and image.shape[2] not in (1, 3):
        raise ValueError("PaddleOCR image must have one or three channels")
