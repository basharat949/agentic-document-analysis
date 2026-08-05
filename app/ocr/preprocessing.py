"""Image preprocessing for handwritten text recognition pipelines.

This module prepares document images for a downstream OCR engine. It does not
perform OCR itself and has no dependency on an OCR or machine-learning runtime.
"""

from __future__ import annotations

import logging
from os import PathLike
from pathlib import Path

import cv2
import numpy as np

type ImagePath = str | PathLike[str]

LOGGER = logging.getLogger(__name__)

_CLAHE_CLIP_LIMIT = 2.0
_CLAHE_TILE_GRID_SIZE = (8, 8)
_MEDIAN_KERNEL_SIZE = 3
_ADAPTIVE_BLOCK_SIZE = 31
_ADAPTIVE_CONSTANT = 15
_MORPHOLOGY_KERNEL_SIZE = (2, 2)


class ImagePreprocessingError(RuntimeError):
    """Base exception raised when an image cannot be preprocessed."""


class ImageLoadError(ImagePreprocessingError):
    """Raised when an image path cannot be decoded by OpenCV."""


class InvalidImageError(ImagePreprocessingError):
    """Raised when decoded image data is invalid or unsupported."""


def preprocess_image(image_path: ImagePath) -> np.ndarray:
    """Prepare a document image for handwritten OCR.

    The pipeline loads and validates the image, converts it to grayscale,
    enhances local contrast, removes impulse noise, creates a binary image,
    corrects skew, and closes small gaps in character shapes with morphology.

    Args:
        image_path: Path to an image that OpenCV can decode.

    Returns:
        A two-dimensional ``uint8`` NumPy array containing black foreground
        text on a white background.

    Raises:
        TypeError: If ``image_path`` is not a string or path-like object.
        ImageLoadError: If the path is missing, is not a file, or cannot be
            decoded as an image.
        InvalidImageError: If the decoded image has invalid dimensions,
            channels, data type, or pixel values.
        ImagePreprocessingError: If OpenCV fails during preprocessing.
    """
    path = _normalise_path(image_path)
    LOGGER.info("Starting image preprocessing for %s", path)

    image = _load_image(path)
    _validate_image(image)

    try:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        LOGGER.debug("Converted image to grayscale")

        clahe = cv2.createCLAHE(
            clipLimit=_CLAHE_CLIP_LIMIT,
            tileGridSize=_CLAHE_TILE_GRID_SIZE,
        )
        enhanced = clahe.apply(grayscale)
        LOGGER.debug("Enhanced local contrast with CLAHE")

        denoised = cv2.medianBlur(enhanced, _MEDIAN_KERNEL_SIZE)
        LOGGER.debug("Removed impulse noise with median blur")

        block_size = _adaptive_block_size(denoised.shape)
        binary = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            _ADAPTIVE_CONSTANT,
        )
        LOGGER.debug("Applied adaptive thresholding with block size %d", block_size)

        deskewed = _deskew(binary)
        processed = _apply_morphology(deskewed)
        processed = np.where(processed > 0, 255, 0).astype(np.uint8, copy=False)
        LOGGER.debug("Normalized output to binary uint8 values")
    except cv2.error as exc:
        LOGGER.exception("OpenCV failed while preprocessing %s", path)
        raise ImagePreprocessingError(
            f"OpenCV failed while preprocessing image: {path}"
        ) from exc

    LOGGER.info("Finished image preprocessing for %s", path)
    return processed


def _normalise_path(image_path: ImagePath) -> Path:
    """Convert a supported path value to ``Path`` without resolving it."""
    if not isinstance(image_path, (str, PathLike)):
        raise TypeError("image_path must be a string or path-like object")
    return Path(image_path).expanduser()


def _load_image(path: Path) -> np.ndarray:
    """Load a colour image from disk and provide actionable failures."""
    if not path.exists():
        raise ImageLoadError(f"Image file does not exist: {path}")
    if not path.is_file():
        raise ImageLoadError(f"Image path is not a regular file: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ImageLoadError(f"OpenCV could not decode image file: {path}")
    LOGGER.debug("Loaded image %s with shape %s", path, image.shape)
    return image


def _validate_image(image: np.ndarray) -> None:
    """Validate the structure and values expected from an OpenCV image."""
    if not isinstance(image, np.ndarray):
        raise InvalidImageError("Decoded image must be a NumPy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise InvalidImageError(
            f"Expected a three-channel BGR image, received shape {image.shape}"
        )
    if image.shape[0] < 3 or image.shape[1] < 3:
        raise InvalidImageError(
            f"Image must be at least 3x3 pixels, received {image.shape[:2]}"
        )
    if image.dtype != np.uint8:
        raise InvalidImageError(
            f"Expected uint8 image data, received {image.dtype}"
        )
    if not np.isfinite(image).all():
        raise InvalidImageError("Image contains non-finite pixel values")


def _adaptive_block_size(shape: tuple[int, ...]) -> int:
    """Choose the largest valid odd threshold window up to the configured size."""
    minimum_dimension = min(shape[:2])
    block_size = min(_ADAPTIVE_BLOCK_SIZE, minimum_dimension)
    if block_size % 2 == 0:
        block_size -= 1
    if block_size < 3:
        raise InvalidImageError(
            "Image dimensions are too small for adaptive thresholding"
        )
    return block_size


def _deskew(binary: np.ndarray) -> np.ndarray:
    """Estimate foreground orientation and rotate a binary image to correct it."""
    foreground = cv2.bitwise_not(binary)
    points = cv2.findNonZero(foreground)
    if points is None or len(points) < 2:
        LOGGER.warning("Skipping deskew because no usable foreground was detected")
        return binary.copy()

    (_, _), (width, height), angle = cv2.minAreaRect(points)
    if width < height:
        angle += 90.0
    correction_angle = -(((angle + 45.0) % 90.0) - 45.0)

    if abs(correction_angle) < 0.05:
        LOGGER.debug("Deskew not required; correction angle is %.3f", correction_angle)
        return binary.copy()

    image_height, image_width = binary.shape
    centre = (image_width / 2.0, image_height / 2.0)
    rotation = cv2.getRotationMatrix2D(centre, correction_angle, 1.0)
    deskewed = cv2.warpAffine(
        binary,
        rotation,
        (image_width, image_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    LOGGER.debug("Deskewed image by %.3f degrees", correction_angle)
    return deskewed


def _apply_morphology(binary: np.ndarray) -> np.ndarray:
    """Close small gaps in foreground strokes with light morphology."""
    foreground = cv2.bitwise_not(binary)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        _MORPHOLOGY_KERNEL_SIZE,
    )
    closed = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
    LOGGER.debug("Applied morphological closing")
    return cv2.bitwise_not(closed)
