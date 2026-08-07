# Section 1 — Traditional OCR and Evaluation

## Task 1.1 — Handwriting preprocessing choices

The implemented pipeline in `app/ocr/preprocessing.py` applies these steps in
order:

- **Load and validate:** OpenCV decodes a three-channel `uint8` image, while
  explicit shape, size, and value checks stop missing, corrupt, or unsupported
  inputs before they produce misleading OCR.
- **Grayscale conversion:** removes unnecessary colour variation so ink and
  paper are represented by one intensity channel.
- **CLAHE:** improves local ink-to-paper contrast under shadows, uneven lighting,
  or faint handwriting while limiting noise amplification.
- **3×3 median blur:** suppresses isolated scan speckles and salt-and-pepper
  noise while retaining handwriting edges better than averaging.
- **Adaptive Gaussian thresholding:** separates ink from paper using local
  illumination, addressing gradients and nonuniform page brightness.
- **Deskew:** estimates foreground orientation with `minAreaRect` and rotates the
  binary page, addressing tilted baselines that disrupt line segmentation.
- **Light morphological closing:** reconnects tiny gaps in pen strokes with a
  2×2 elliptical kernel without applying aggressive shape changes.
- **Binary normalization:** guarantees a two-dimensional `uint8` result containing
  only black text (`0`) and white background (`255`), removing ambiguous
  intermediate intensity values.

**Considered but excluded:** skeletonization/thinning could normalize stroke
width, but it can introduce spurs, break faint strokes, and deform small loops or
joins. Those irreversible artifacts are especially risky for handwriting, so
the implementation preserves stroke structure.

## Task 1.2 — OCR engine choice

Tesseract, through `pytesseract`, is the implemented default baseline because it
runs locally, keeps the lightweight environment small, and returns token text,
confidence, bounding boxes, and hierarchy needed by the shared pipeline. It is
deterministic and readily testable, but its supplied-page handwriting output was
largely unusable and it is not presented as highly accurate.

PaddleOCR PP-OCRv5 is an optional, explicitly selected candidate because its
mobile detection and recognition models recovered substantially more recognizable
content from the supplied handwriting. It remains outside the default install
because Paddle has a heavier native runtime, platform-specific constraints, and
external model downloads. The adapter adds safe downscaling, coordinate
remapping, confidence normalization, and geometric reading order, but the
single-page experiment does not demonstrate production readiness or broad
handwriting accuracy. Neither engine silently falls back to the other.

## Task 1.3 — Composite score analysis

The composite combines character quality (`1 − CER`) at 35%, word quality
(`1 − WER`) at 35%, and Sentence F1 at 30%, with bounded component quality. It
therefore rewards exact character and token transcription while retaining a
sentence-level view of structural recovery.

It does not measure semantic correctness, layout or reading order, confidence
calibration, error importance, or performance across writers and document types.
Different damaging edits can also receive similar aggregate scores.

Source-Fidelity Score is the additional implemented verbatim-transcription
metric. It aligns token occurrences and measures exact preservation of
fidelity-sensitive forms such as unusual capitalization, punctuation,
alphanumeric text, repeated characters, and non-ASCII text. CER and WER count
edit distance uniformly, so they cannot specifically reveal fluent-looking
“corrections” to unusual but intentional source text. SFS complements rather
than replaces CER, WER, or human-ground-truth review.
