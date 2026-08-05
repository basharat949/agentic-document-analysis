# Task 1.1 — Handwritten Image Preprocessing

The implementation in `app/ocr/preprocessing.py` prepares a document image for
a downstream handwritten OCR system. It deliberately stops at preprocessing and
does not perform text recognition.

## Why each pipeline step helps

1. **Load image:** Decoding through OpenCV produces a consistent BGR NumPy array
   for the rest of the pipeline. Explicit load errors prevent missing or corrupt
   files from silently reaching later stages.
2. **Validate image:** Checking dimensions, channels, data type, and minimum size
   rejects malformed input early. This makes later OpenCV operations predictable
   and gives callers an actionable error rather than a low-level failure.
3. **Convert to grayscale:** Colour is usually not needed to identify handwritten
   glyphs. A single intensity channel reduces computation and makes contrast and
   threshold operations less sensitive to ink colour.
4. **CLAHE contrast enhancement:** Contrast Limited Adaptive Histogram
   Equalization improves local ink-to-paper contrast under shadows, faded ink,
   and uneven lighting. Its contrast limit avoids strongly amplifying noise.
5. **Median-blur noise removal:** A small median filter suppresses isolated
   salt-and-pepper noise while preserving edges better than ordinary averaging.
   Preserved stroke boundaries are important for handwritten character shapes.
6. **Adaptive thresholding:** A locally calculated threshold separates ink from
   paper even when illumination varies across the page. The result is a binary
   image that makes character structure clearer to an OCR engine.
7. **Deskew with OpenCV:** The foreground orientation is estimated with
   `minAreaRect`, then corrected using an affine rotation. Straighter text lines
   improve line segmentation and keep adjacent characters in a more consistent
   reading order. Blank images are retained safely because they have no reliable
   angle to correct.
8. **Morphological opening and closing:** Opening removes small isolated
   foreground specks. Closing reconnects tiny breaks within pen strokes. The
   operations run on an inverted foreground mask so they affect ink rather than
   the white page, and use a conservative kernel to avoid distorting handwriting.
9. **Return a NumPy array:** The final two-dimensional `uint8` array uses black
   text on a white background, a conventional representation that can be passed
   directly to later OCR or inspection code without another file round trip.

## Intentionally omitted technique

**Skeletonization (thinning)** was considered but omitted. It can normalize
strokes to a single-pixel width, but handwritten text contains meaningful width,
loops, joins, and pressure variations. Thinning can introduce spurs, break faint
strokes, or deform small characters, especially after thresholding. Those losses
are irreversible and can reduce recognition quality, so this pipeline preserves
the original stroke structure and limits morphology to light cleanup.
