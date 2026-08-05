# Task 1.2 — OCR Extraction Pipeline

## OCR engine choice (131 words)

Tesseract, accessed through `pytesseract`, was selected because it is a mature,
local, deterministic OCR engine with no model-service dependency. Its
`image_to_data` interface exposes token text, confidence, bounding boxes, and
page hierarchy in one call, which directly supports auditable low-confidence
handling. It also keeps document content on the machine and has modest runtime
requirements. Tesseract was preferred over EasyOCR and PaddleOCR for this
environment because both alternatives introduce larger deep-learning dependency
stacks and commonly depend on Torch or other model runtimes, which are explicitly
out of scope. Tesseract integrates cleanly with the existing OpenCV/NumPy
preprocessing pipeline and is easy to mock in fast deterministic tests. The main
tradeoff is that its default models are stronger on printed text than
unconstrained handwriting, so preprocessing helps but does not guarantee strong
handwriting recognition.

## Sentence segmentation

The implementation uses a deterministic regex fallback that splits only after
`.`, `!`, or `?` followed by whitespace. It requires no downloaded language
model and introduces no shared mutable request state. OCR tokens are joined only
to reconstruct their Tesseract line order; their spelling and grammar are never
corrected. Sentence strings are returned in reading order and empty results are
discarded. This deliberately simple rule may miss punctuation-free boundaries
and can split after abbreviations.

## Metadata exclusion heuristic

Metadata detection inspects only the first five non-empty OCR lines that also
start within the top 20% of the processed page. A candidate is excluded when it
matches an explicit `Name:`, `Date:`, `Title:`, or `Subject:` label; is entirely
a recognized date; or is the first line and has a conservative short title-like
form (at most eight title-case/uppercase words and no sentence-ending mark).
Excluded lines remain present in `raw_text`; only `body_text` and `sentences`
omit them.

## Low-confidence handling

The default threshold is 60 and callers may configure any finite value from 0
through 100. Every non-empty token below the threshold remains in reconstructed
text and also becomes an `OCRRegion` containing its unchanged text, numeric
confidence, bounding box, and available page/block/paragraph/line identifiers.
No uncertain token is silently discarded.

## Limitations

Tesseract confidence is engine-specific and is not a calibrated probability.
Handwriting—especially cursive, faint, overlapping, or highly individual
writing—may be transcribed poorly even after preprocessing. Line order follows
Tesseract's hierarchy and can be wrong for columns or unusual layouts. The
metadata rules can miss novel labels or dates and may remove a genuine short
title-like first body line. Restricting inspection to both the first lines and
top page region reduces, but cannot eliminate, false removals. The deterministic
sentencizer also cannot reliably infer boundaries when OCR loses punctuation.
