# Task 4.1 — Source-Fidelity Score

## Formal definition

Let:

- (G) be the ordered ground-truth token sequence;
- (P) be the ordered predicted token sequence;
- (S) be the ground-truth token **occurrences** marked fidelity-sensitive by
  the deterministic heuristic; and
- (M) be the number of sensitive ground-truth occurrences aligned to an
  exactly identical predicted token.

The Source-Fidelity Score is:

**SFS(G, P) = M / |S|**

When (|S| = 0), SFS is defined as **1.0**. Occurrences are counted by position,
not unique token value, so two identical sensitive source tokens contribute two
denominator entries. Tokens are split only on whitespace and aligned using
minimum Levenshtein edit distance. Exact case and attached punctuation are
required for preservation. Prediction-only insertions never increase (M).

The heuristic identifies surface forms such as punctuation-bearing, mixed-case,
alphanumeric, repeated-character, non-ASCII, and other non-lowercase-alphabetic
tokens. It uses no dictionary or human labels and cannot determine whether a
misspelling is intentional.

## Worked example A — poor CER but perfect SFS

| Field | Value |
|---|---|
| Ground truth | `hello!!! the cat sat on the mat` |
| Prediction | `hello!!! many dogs ran very far` |

`hello!!!` is the only fidelity-sensitive ground-truth token and is copied
exactly, so (M = 1), |S| = 1, and **SFS = 1.0**. Many ordinary characters and
words changed, so CER is qualitatively poor. SFS intentionally focuses on the
preservation question that aggregate edit distance does not isolate.

## Worked example B — low CER but poor SFS

| Field | Value |
|---|---|
| Ground truth | `please wait!!!` |
| Prediction | `please wait!` |

Only a small punctuation change occurred, so CER may still look relatively good.
However, `wait!!!` was not copied exactly. The sensitive occurrence is lost, so
(M = 0), |S| = 1, and **SFS = 0.0**.

## Misleading edge cases

For ground truth `chlid` and prediction `child`, both tokens are lowercase and
alphabetic. The heuristic does not recognize `chlid` as sensitive, the
denominator is zero, and SFS returns **1.0**, even if the misspelling was
intentional and was silently normalized. This is a known limitation, not desired
semantic behavior.

Conversely, ordinary sentence-initial capitalization such as `The` is marked
sensitive, even when it is not unusual. Any attached punctuation also makes a
token sensitive, including routine punctuation. Human-labelled sensitive spans
would distinguish intended irregularities from normal writing more faithfully.

## Limitations and interpretation

- There is no lexicon, spell checker, or human annotation.
- The heuristic cannot know whether a misspelling is intentional.
- Whitespace tokenization does not model punctuation as separate tokens or
  resolve harmless segmentation differences.
- Sentence-initial capitalization and ordinary punctuation can create false
  positives.
- Exact matching can penalize harmless whitespace or formatting differences.
- The zero-denominator rule can conceal changes to ordinary-looking forms.
- SFS complements CER and WER; it does not replace either metric.
