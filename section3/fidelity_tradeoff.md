# The Fidelity vs. Correction Trade-off

## 1. Why CER and WER can hide silent normalization

CER and WER measure edit distance, not the type or direction of an edit.
Correcting an intentional misspelling is treated like introducing an OCR error.
Given ground truth “The
chlid went hom.” and prediction “The child went home.”, fluency improves but both
source irregularities disappear. Aggregate scores can remain good while rare
non-standard forms are repeatedly normalized because common words dominate
averages and fidelity-sensitive tokens may be scarce. Lower CER or WER does not
prove preservation of
spelling errors, invented words, unusual capitalization, punctuation, or
non-standard grammar. Fluent correctness is not the goal; faithful reproduction
is.

## 2. Preprocessing and post-processing safeguards

OCR preprocessing must remain visual: contrast enhancement, denoising, deskew,
thresholding, and conservative morphology are appropriate; spell-checking,
grammar correction, autocomplete, and language-model rewriting are not.
Aggressive morphology can merge characters or erase thin strokes and
punctuation. Versioned preprocessing profiles should compare raw and processed
images on a fixed regression set.

Raw OCR output must remain separate from body and downstream fields, retaining
token confidence and bounding boxes. Low-confidence text must stay verbatim and
flagged, never silently replaced. Preserve capitalization, punctuation, unusual
words, and a documented whitespace policy. Excluded metadata must remain in raw
text with its exclusion recorded. Explicit schema fields prevent
normalization from hiding inside transcription. Any correction layer must be
separate, opt-in, auditable, and retain source and corrected values.

## 3. Source-Fidelity / Non-Standard Preservation metric

A Source-Fidelity Score uses labelled ground-truth spans covering
misspellings, invented words, unusual capitalization, punctuation, and other
non-standard forms. After aligning ground-truth and predicted tokens or spans, a
sensitive span is preserved only if reproduced exactly. The score is preserved
fidelity-sensitive spans divided by total fidelity-sensitive spans. A companion
Silent Normalization Rate is normalized or “corrected” sensitive spans divided
by total sensitive spans.

Intentional irregularities may require human labels; exact matching can penalize
harmless segmentation or whitespace differences. This complements, rather than
replaces, CER and WER.

CI/CD should use a versioned golden set rich in misspellings, invented words,
punctuation anomalies, and mixed capitalization. Every preprocessing, OCR-engine,
prompt, model, or post-processing change should run CER, WER, Source-Fidelity
Score, and Silent Normalization Rate. Builds should fail if fidelity falls below
an agreed threshold or normalization exceeds its budget. Results should be
reported by document type, writer or source, and non-standard category; changed
fidelity-sensitive outputs require human review.
