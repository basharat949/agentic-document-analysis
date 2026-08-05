# Task 2.3 — Error Analysis

## 1. Error concentration

The supplied confusion matrix is:

| Gold category | Predicted Simple | Predicted Compound/Complex | Predicted Incomplete |
|---|---:|---:|---:|
| Simple | 81 | 12 | 7 |
| Compound/Complex | 18 | 62 | 3 |
| Incomplete | 3 | 4 | 10 |

The largest error cell is **Gold Compound/Complex → Predicted Simple (18)**.
Likely causes include omitted or OCR-lost punctuation, run-on clauses being read
as one clause, and confusion between coordinated predicates and coordinated
independent clauses. Missing subjects in later clauses and non-standard
subject–verb agreement can further obscure clause boundaries.

The next substantial structural error is **Gold Simple → Predicted
Compound/Complex (12)**. Coordinated verbs may be mistaken for coordinated
clauses, especially when punctuation is unreliable. Misspelled subordinate
markers can also make ordinary modifiers appear to introduce dependent clauses,
while OCR punctuation loss can create misleading clause groupings in either
direction.

There is an explicit source inconsistency: the prose states that 7 of 17 gold
Incomplete sentences were predicted Simple, but the displayed matrix assigns 3
to Simple and 4 to Compound/Complex. This analysis preserves the matrix and
interprets the intended point as **7 of 17 gold Incomplete sentences being
misclassified into complete categories**.

## 2. Reducing missed Incomplete classifications

Add a conservative completeness-verification step before accepting `Simple` or
another complete category. Acceptance should require affirmative evidence of a
complete independent clause: an identifiable subject (or a licensed implicit
subject in an imperative) and a finite predicate that forms a complete
proposition. Missing subjects, unresolved subordinate markers, or uncertain
clause boundaries should prevent an immediate complete classification.

Low-confidence or structurally ambiguous cases should be marked provisionally
Incomplete and passed through either a deterministic validation step or the
Embedded Sentence Agent. This should improve Incomplete recall and recover
nested complete material that would otherwise be missed. The trade-off is more
complete sentences falsely labelled Incomplete, lower Incomplete precision,
more specialist calls, and increased latency and cost. The verification should
therefore be conservative, measurable, and based on explicit structural signals
rather than general uncertainty.

## 3. Additional evaluation column

Add an **Embedded recovery outcome** column for every routed sentence, with one
of these values:

- `correctly promoted`
- `correctly remained Incomplete`
- `incorrectly promoted`
- `missed recoverable embedded sentence`

The evaluation record should also retain the original classifier label, whether
the embedded agent was invoked, whether a complete embedded sentence was found,
and the final label. Final-label confusion alone cannot distinguish a routing
failure from a specialist-agent failure. For example, an incorrect final
Incomplete label could mean that the primary classifier never routed a
recoverable sentence, or that routing occurred but the embedded agent missed the
span. Recording intermediate decisions makes those failure modes separately
measurable and actionable.

## 4. Production threshold

From the displayed matrix, 20 sentences were predicted Incomplete
(`7 + 3 + 10`), of which 10 were correct. There are 17 gold Incomplete sentences
(`3 + 4 + 10`), of which 10 were recovered. Therefore:

- **Incomplete precision:** `10 / 20 = 50%`
- **Incomplete recall:** `10 / 17 ≈ 58.8%`

These results are not production-ready. A release threshold should require at
least **90% precision and 90% recall** for Incomplete, with **95% recall
preferred** when missing an Incomplete sentence prevents specialist routing.
Recall is especially important because a false negative bypasses the Embedded
Sentence Agent entirely and removes any chance of recovery. Precision still
matters: excessive false positives create avoidable specialist calls, latency,
and cost.

These thresholds should be demonstrated on a representative held-out set that
includes OCR noise, misspellings, non-standard grammar, and varied sentence
structures. Precision, recall, routing rate, and recovery outcomes should then
be monitored after deployment for distribution shift and regressions.
