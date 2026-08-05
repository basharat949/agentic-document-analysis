# Difficult Embedded-Agent Examples

Both inputs are already labelled `Incomplete`. All extracted text is copied
verbatim; no spelling, grammar, or punctuation is corrected. Every complete span
has a structural category, while an incomplete span would use `category: null`.
For multiple complete spans, source order is retained and the orchestrator later
selects the highest category using `Compound-Complex > Complex > Compound >
Simple`.

## 1. Embedded sentence exists

Input:

```json
{"sentence":"Although Nida whisperd the keys are under the mat","category":"Incomplete"}
```

Expected output:

```json
{"input_sentence":"Although Nida whisperd the keys are under the mat","status":"single","embedded_spans":[{"text":"the keys are under the mat","is_complete":true,"category":"Simple"}],"reason":"The incomplete outer concession clause contains one contiguous complete proposition."}
```

Why difficult: The whole input is an `Although` fragment, but the reported
content has its own subject and finite predicate and can stand alone. The agent
must extract only that inner span and retain `whisperd` elsewhere unchanged.

## 2. No embedded sentence exists

Input:

```json
{"sentence":"While near the brokn gate after midnite","category":"Incomplete"}
```

Expected output:

```json
{"input_sentence":"While near the brokn gate after midnite","status":"none","embedded_spans":[],"reason":"No contiguous span forms a complete proposition with its own subject and finite predicate."}
```

Why difficult: Several modifiers and a subordinating word make the fragment
sound like a larger construction, but no inner subject-predicate proposition is
present and nothing may be invented.
