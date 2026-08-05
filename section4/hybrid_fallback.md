# Task 4.2 — Hybrid OCR Fallback

## 1. Motivation

Traditional OCR should remain the default transcription path. Tesseract is
deterministic for a fixed image and configuration, inexpensive to operate, and
fast enough for routine pages. It also exposes word confidence and bounding
boxes, which support explainable fallback decisions. Its weaknesses are most
visible with handwriting, poor scans, residual skew, faded ink, unusual layout,
and crossed-out or overwritten text.

A Vision LLM can interpret handwriting and difficult layouts and can use visual
context when individual glyphs are ambiguous. However, it is slower, more
expensive, non-deterministic, constrained by image/token limits, and capable of
producing fluent text that is not present in the source. Vision should therefore
not replace traditional OCR entirely. It should be a bounded specialist used
only when measurable OCR evidence indicates that additional visual analysis is
worth its cost and risk.

## 2. Hybrid architecture

```text
                           +----------------------+
                           | PDF or image upload  |
                           +----------+-----------+
                                      |
                                      v
                           +----------+-----------+
                           | Page rasterization   |  (PDF only)
                           | and preprocessing    |
                           +----------+-----------+
                                      |
                                      v
                           +----------+-----------+
                           | Tesseract OCR        |
                           | text/confidence/boxes|
                           +----------+-----------+
                                      |
                                      v
                           +----------+-----------+
                           | Confidence analysis  |
                           +----------+-----------+
                                      |
                                      v
                                +-----+-----+
                                | Decision  |
                                +--+-----+--+
                     high confidence |     | low confidence
                                     |     |
                         +-----------v-+ +-v------------------+
                         | Accept OCR  | | Crop uncertain     |
                         | unchanged   | | regions            |
                         +-----------+-+ +---------+----------+
                                     |             |
                                     |             v
                                     |   +---------+----------+
                                     |   | Vision OCR         |
                                     |   | verbatim request   |
                                     |   +---------+----------+
                                     |             |
                                     +------+------+
                                            |
                                            v
                                      +-----+-----+
                                      |   Merge   |
                                      +-----+-----+
                                                |
                                                v
                                  +-------------+--------------+
                                  | Deterministic sentence     |
                                  | extraction                  |
                                  +-------------+--------------+
                                                |
                                                v
                                  +-------------+--------------+
                                  | Classification pipeline    |
                                  | primary -> Incomplete-only |
                                  | embedded-agent routing     |
                                  +----------------------------+
```

For a PDF, rasterization produces page images at a controlled resolution. The
existing visual preprocessing then improves contrast, removes noise, thresholds,
deskews, and performs conservative morphology. Tesseract remains the first OCR
engine and supplies the baseline text, confidence, and geometry.

Confidence analysis aggregates token- and region-level evidence without changing
the transcription. A deterministic decision policy accepts reliable pages
unchanged or identifies uncertain crops for Vision OCR. Vision is instructed to
transcribe verbatim, including misspellings, capitalization, punctuation,
cross-outs where representable, and non-standard grammar. A validated merge
updates only approved regions. Sentence extraction and the existing two-agent
classification pipeline operate on the resulting merged text; classification
does not decide whether Vision runs.

## 3. Fallback policy

Fallback decisions should use versioned thresholds calibrated on a held-out set,
not one confidence number in isolation. Practical triggers include:

- average word confidence below a configured page threshold;
- the proportion or consecutive count of non-empty words below the existing
  low-confidence threshold exceeding a limit;
- a spatial cluster of low-confidence boxes covering a substantial text region;
- Tesseract execution failure or an undecodable OCR response after preprocessing;
- blank OCR output when visual foreground analysis indicates that ink exists;
- excessive replacement characters or unknown-symbol patterns; and
- implausibly sparse text relative to detected text-line regions.

Trigger evaluation should distinguish a small uncertain crop from a page-wide
failure. A page-wide Vision request is justified only when region localization
is unreliable or uncertainty is widespread. Thresholds should include minimum
region size and padding so isolated punctuation is not cropped without context.

Fallback must **not** occur merely because the OCR output contains spelling
mistakes, invented words, unusual capitalization, or non-standard grammar. Those
may be faithful source forms, and using linguistic correctness as a trigger
would encourage silent normalization. The policy relies on OCR/visual evidence,
not a dictionary or fluency score.

## 4. Region-based Vision

The preferred unit of fallback is a padded crop containing one uncertain line
or connected group of low-confidence words. Sending only uncertain regions
reduces transferred pixels, Vision tokens, latency, and cost. It also increases
throughput by keeping clear text on the fast Tesseract path and narrows the area
in which a non-deterministic model can alter content.

Cropping has trade-offs. A crop may remove neighboring words needed to resolve
handwriting, split a character at its boundary, omit layout cues, or make reading
order ambiguous. Small regions can also generate disproportionate request
overhead. Regions should therefore be merged when boxes overlap or belong to the
same line, expanded with bounded context padding, and linked to page coordinates.
If low-confidence regions cover most of a page or cannot be localized reliably,
a controlled full-page fallback may be more accurate than many fragmented calls.

## 5. Merge strategy

The Tesseract result is the immutable baseline. Vision output is a proposed
replacement for a specific page-coordinate region, not a new authoritative page
transcription. The merge process should:

1. validate that the Vision response belongs to the requested document, page,
   and region;
2. retain Tesseract tokens outside the region unchanged;
3. replace only the reviewed region when the response passes schema, fidelity,
   and boundary checks;
4. reconstruct reading order from stored region anchors rather than model prose;
   and
5. mark every inserted Vision span with its source and decision outcome.

Persistence must retain three distinct artifacts:

- **original OCR:** complete Tesseract text, token confidence, and bounding boxes;
- **Vision output:** verbatim response, requested crop coordinates, model/version,
  request identifier, latency, and validation result; and
- **merged output:** final text plus a map from every replacement to both source
  variants.

Overwriting original OCR is unsafe because Vision can normalize, omit, reorder,
or hallucinate text. An append-only audit trail should record thresholds, trigger
reasons, crop checksum, timestamps, model and prompt versions, original and
replacement text, and whether the replacement was accepted, rejected, or sent
for human review. Confidence values and bounding boxes remain attached to the
Tesseract evidence; Vision-derived spans should use a distinct provenance marker
rather than pretending to have comparable Tesseract confidence.

## 6. Failure handling

Vision is an enhancement, not a requirement for completing OCR. On a timeout,
provider/API failure, malformed response, HTTP 429 exhaustion, or rejected merge,
the system returns the original Tesseract result. The job should complete with a
degraded or fallback-unavailable indicator and preserve low-confidence flags for
review rather than fail the entire document.

Rate limits may use bounded retry with exponential backoff, but retries must not
block OCR completion indefinitely. Region requests should be idempotent and
associated with stable identifiers so a retry cannot create duplicate merges.
If only some regions succeed, accept independently validated replacements and
retain Tesseract text for all failed regions. Errors should be recorded without
exposing document content in routine logs. A configurable circuit breaker at the
workflow level can temporarily bypass Vision after sustained provider failure;
OCR-only processing must remain available.

## 7. Cost analysis

| Strategy | Cost and quality profile |
|---|---|
| OCR only | Lowest variable cost and latency; deterministic and effective for clear text, but weaker on difficult handwriting and degraded scans |
| Vision only | Applies the most expensive and slowest path to every page, including easy pages; increases token/image usage and exposure to non-determinism |
| Hybrid | Pays the Vision cost only for evidence-based uncertain regions while preserving the predictable OCR path for the majority of clear content |

The hybrid design offers the best practical cost-quality trade-off because its
expensive capability is proportional to observed uncertainty rather than total
document volume. Region coalescing, per-document fallback budgets, maximum crop
counts, and provider token limits should bound worst-case spending. Cost review
must include unsuccessful requests and retries, not only accepted replacements.
No fixed savings or accuracy improvement should be assumed before measurement on
representative documents.

## 8. Monitoring

Production dashboards should track:

- document-, page-, and region-level fallback rate;
- reasons that triggered fallback and confidence distributions before fallback;
- Vision request latency, timeout rate, and queue time;
- Vision token/image usage and cost per region, page, and document;
- HTTP 429, provider error, retry, and retry-exhaustion rates;
- number, size, and page coverage of merged regions;
- accepted, rejected, OCR-retained, and manually reviewed replacements;
- manual-review rate and reviewer overturn rate;
- blank-output and OCR-only degraded-completion rate; and
- source-fidelity regressions comparing original, Vision, and merged text.

Metrics should be segmented by document type, image quality, language/script,
writer/source where permitted, preprocessing profile, and model/prompt version.
Sudden fallback-rate changes may indicate scan-quality drift, a preprocessing
regression, Tesseract change, or poorly calibrated thresholds.

## 9. Limitations

Vision models can hallucinate plausible text, silently correct source errors, or
vary across repeated requests. Their confidence is not directly comparable to
Tesseract confidence, and Tesseract scores themselves may be poorly calibrated
for handwriting. A bad threshold can therefore under-trigger on unreadable text
or over-trigger on faithful but unusual forms.

Sending document crops to an external model creates privacy, retention,
residency, and access-control concerns. Sensitive documents may require Vision
fallback to be disabled or restricted to an approved deployment. Cost and
latency remain variable, provider rate limits can reduce availability, and image
or token limits may truncate context. Region crops can lose layout context,
whereas full-page calls increase cost and the hallucination surface. Human review
remains necessary for high-impact documents and unresolved low-confidence areas.

## 10. Conclusion

Use **OCR first and Vision second**. Tesseract should always produce and preserve
the baseline transcription. Deterministic, visual-confidence rules should invoke
Vision primarily for padded low-confidence regions, with full-page fallback only
for widespread or unlocalizable failure. Every proposed replacement must be
validated, merged narrowly, and recorded with its original OCR evidence and
provenance. If Vision fails, return OCR plus uncertainty flags. This structure
improves difficult-handwriting coverage while bounding cost, latency,
non-determinism, and source-fidelity risk.
