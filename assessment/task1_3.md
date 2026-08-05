# Task 1.3 — Evaluation

The composite combines character fidelity (35%), word fidelity (35%), and
sentence preservation (30%). Equal CER/WER weights balance fine-grained copying
against token accuracy; Sentence F1 gives substantial—but smaller—credit to
structure. CER/WER quality is clamped at zero only during composition; reported
error rates may exceed one.

It misses semantic correctness, layout, reading order, confidence calibration,
and error severity. CER and WER alone ignore sentence correspondence and can hide
whether errors concentrate in important passages. A longest-common-subsequence
ratio is an additional metric for verbatim transcription.

A future Source-Fidelity / Normalisation Error metric should align source and
prediction spans, then count unsupported changes to misspellings, invented words,
unusual capitalization, and punctuation. It should report such normalisations per
source token (with category breakdowns), penalizing fluent “corrections” even when
meaning is preserved. This is recommended but not implemented here.
