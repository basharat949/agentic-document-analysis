"""Evaluate OCR transcriptions with error, sentence, and composite metrics."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+")
_SENTENCE_OVERLAP_THRESHOLD = 0.50
_CER_WEIGHT = 0.35
_WER_WEIGHT = 0.35
_SENTENCE_F1_WEIGHT = 0.30


class EvaluationError(Exception):
    """Base exception for evaluation failures."""


class EvaluationInputError(EvaluationError):
    """Raised when an evaluation JSON file or sample is malformed."""


@dataclass(frozen=True, slots=True)
class EvaluationSample:
    """A ground-truth and predicted transcription pair.

    Attributes:
        sample_id: Caller-provided identifier displayed in reports.
        ground_truth: Reference transcription, preserved verbatim.
        predicted: OCR transcription, preserved verbatim.
    """

    sample_id: str
    ground_truth: str
    predicted: str


@dataclass(frozen=True, slots=True)
class SentenceMetrics:
    """Sentence-level precision, recall, F1, and match count."""

    precision: float
    recall: float
    f1: float
    matched_sentences: int


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """All evaluation metrics for one transcription sample."""

    sample_id: str
    cer: float
    wer: float
    sentence_precision: float
    sentence_recall: float
    sentence_f1: float
    composite_score: float


@dataclass(frozen=True, slots=True)
class AggregateResult:
    """Macro averages across evaluated samples."""

    sample_count: int
    cer: float
    wer: float
    sentence_precision: float
    sentence_recall: float
    sentence_f1: float
    composite_score: float


def levenshtein_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Compute Levenshtein distance using insertion, deletion, and substitution.

    Args:
        reference: Reference character or token sequence.
        hypothesis: Predicted character or token sequence.

    Returns:
        The minimum number of unit-cost edits needed to transform the reference
        into the hypothesis.
    """
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference

    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_item in enumerate(reference, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_item in enumerate(hypothesis, start=1):
            insertion = current[hypothesis_index - 1] + 1
            deletion = previous[hypothesis_index] + 1
            substitution = previous[hypothesis_index - 1] + (
                reference_item != hypothesis_item
            )
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def character_error_rate(ground_truth: str, predicted: str) -> float:
    """Calculate character error rate without clamping values above one.

    For an empty reference, an empty prediction has error ``0.0``. A non-empty
    prediction returns its character insertion count, equivalent to using a
    denominator of one because the conventional denominator would be zero.
    """
    distance = levenshtein_distance(ground_truth, predicted)
    if not ground_truth:
        return float(distance)
    return distance / len(ground_truth)


def word_error_rate(ground_truth: str, predicted: str) -> float:
    """Calculate whitespace-tokenized word error rate without clamping.

    For an empty reference, an empty prediction has error ``0.0``. Otherwise,
    the result is the predicted-word insertion count using a denominator of one.
    """
    reference_words = ground_truth.split()
    predicted_words = predicted.split()
    distance = levenshtein_distance(reference_words, predicted_words)
    if not reference_words:
        return float(distance)
    return distance / len(reference_words)


def split_sentences(text: str) -> tuple[str, ...]:
    """Split text after ``.``, ``!``, or ``?`` followed by whitespace."""
    if not text.strip():
        return ()
    return tuple(
        sentence.strip()
        for sentence in _SENTENCE_BOUNDARY_PATTERN.split(text)
        if sentence.strip()
    )


def sentence_overlap(ground_truth_sentence: str, predicted_sentence: str) -> float:
    """Return case-insensitive unique-token recall for a sentence pair.

    Tokens are split only on whitespace. Overlap is the size of the intersection
    of unique predicted and ground-truth token sets divided by the number of
    unique ground-truth tokens. Original strings are not modified.
    """
    ground_truth_tokens = {
        token.casefold() for token in ground_truth_sentence.split()
    }
    predicted_tokens = {token.casefold() for token in predicted_sentence.split()}
    if not ground_truth_tokens:
        return 1.0 if not predicted_tokens else 0.0
    return len(ground_truth_tokens & predicted_tokens) / len(ground_truth_tokens)


def sentence_metrics(ground_truth: str, predicted: str) -> SentenceMetrics:
    """Calculate one-to-one sentence precision, recall, and F1.

    A pair is eligible when its unique-token overlap is at least 50%. Maximum
    bipartite matching is used so no sentence on either side is counted twice.
    """
    ground_truth_sentences = split_sentences(ground_truth)
    predicted_sentences = split_sentences(predicted)

    if not ground_truth_sentences and not predicted_sentences:
        return SentenceMetrics(1.0, 1.0, 1.0, 0)
    if not ground_truth_sentences or not predicted_sentences:
        return SentenceMetrics(0.0, 0.0, 0.0, 0)

    eligible_matches = [
        sorted(
            (
                ground_truth_index
                for ground_truth_index, ground_truth_sentence in enumerate(
                    ground_truth_sentences
                )
                if sentence_overlap(ground_truth_sentence, predicted_sentence)
                >= _SENTENCE_OVERLAP_THRESHOLD
            ),
            key=lambda ground_truth_index: (
                -sentence_overlap(
                    ground_truth_sentences[ground_truth_index], predicted_sentence
                ),
                ground_truth_index,
            ),
        )
        for predicted_sentence in predicted_sentences
    ]

    ground_truth_to_prediction: dict[int, int] = {}

    def find_match(predicted_index: int, visited: set[int]) -> bool:
        for ground_truth_index in eligible_matches[predicted_index]:
            if ground_truth_index in visited:
                continue
            visited.add(ground_truth_index)
            previous_prediction = ground_truth_to_prediction.get(ground_truth_index)
            if previous_prediction is None or find_match(previous_prediction, visited):
                ground_truth_to_prediction[ground_truth_index] = predicted_index
                return True
        return False

    matched = sum(
        find_match(predicted_index, set())
        for predicted_index in range(len(predicted_sentences))
    )
    precision = matched / len(predicted_sentences)
    recall = matched / len(ground_truth_sentences)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return SentenceMetrics(precision, recall, f1, matched)


def composite_score(cer: float, wer: float, sentence_f1: float) -> float:
    """Combine character, word, and sentence quality into a score from 0 to 1."""
    cer_quality = max(0.0, 1.0 - cer)
    wer_quality = max(0.0, 1.0 - wer)
    score = (
        _CER_WEIGHT * cer_quality
        + _WER_WEIGHT * wer_quality
        + _SENTENCE_F1_WEIGHT * sentence_f1
    )
    return min(1.0, max(0.0, score))


def evaluate_sample(sample: EvaluationSample) -> EvaluationResult:
    """Calculate all metrics for one evaluation sample."""
    cer = character_error_rate(sample.ground_truth, sample.predicted)
    wer = word_error_rate(sample.ground_truth, sample.predicted)
    sentences = sentence_metrics(sample.ground_truth, sample.predicted)
    return EvaluationResult(
        sample_id=sample.sample_id,
        cer=cer,
        wer=wer,
        sentence_precision=sentences.precision,
        sentence_recall=sentences.recall,
        sentence_f1=sentences.f1,
        composite_score=composite_score(cer, wer, sentences.f1),
    )


def aggregate_results(results: Sequence[EvaluationResult]) -> AggregateResult:
    """Calculate unweighted macro averages across sample results.

    Raises:
        EvaluationInputError: If no results are supplied.
    """
    if not results:
        raise EvaluationInputError("At least one evaluation result is required")
    count = len(results)
    return AggregateResult(
        sample_count=count,
        cer=sum(result.cer for result in results) / count,
        wer=sum(result.wer for result in results) / count,
        sentence_precision=(
            sum(result.sentence_precision for result in results) / count
        ),
        sentence_recall=sum(result.sentence_recall for result in results) / count,
        sentence_f1=sum(result.sentence_f1 for result in results) / count,
        composite_score=sum(result.composite_score for result in results) / count,
    )


def load_samples(path: str | Path) -> tuple[EvaluationSample, ...]:
    """Load and validate evaluation samples from a JSON file.

    Args:
        path: JSON file containing a non-empty list of sample objects.

    Returns:
        Validated immutable evaluation samples.

    Raises:
        EvaluationInputError: If the file cannot be read, parsed, or validated.
    """
    input_path = Path(path)
    try:
        content = input_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvaluationInputError(f"Could not read input file {input_path}: {exc}") from exc

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise EvaluationInputError(
            f"Malformed JSON in {input_path} at line {exc.lineno}, column {exc.colno}"
        ) from exc

    if not isinstance(payload, list):
        raise EvaluationInputError("Input JSON must contain a list of sample objects")
    if not payload:
        raise EvaluationInputError("Input JSON must contain at least one sample")

    samples: list[EvaluationSample] = []
    seen_ids: set[str] = set()
    required_fields = ("sample_id", "ground_truth", "predicted")
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise EvaluationInputError(f"Sample at index {index} must be an object")
        missing = [field for field in required_fields if field not in item]
        if missing:
            raise EvaluationInputError(
                f"Sample at index {index} is missing required fields: "
                + ", ".join(missing)
            )
        invalid = [field for field in required_fields if not isinstance(item[field], str)]
        if invalid:
            raise EvaluationInputError(
                f"Sample at index {index} fields must be strings: " + ", ".join(invalid)
            )
        sample_id = item["sample_id"]
        if not sample_id:
            raise EvaluationInputError(f"Sample at index {index} has an empty sample_id")
        if sample_id in seen_ids:
            raise EvaluationInputError(f"Duplicate sample_id: {sample_id}")
        seen_ids.add(sample_id)
        samples.append(
            EvaluationSample(
                sample_id=sample_id,
                ground_truth=item["ground_truth"],
                predicted=item["predicted"],
            )
        )
    return tuple(samples)


def evaluate_file(path: str | Path) -> tuple[tuple[EvaluationResult, ...], AggregateResult]:
    """Load a JSON file and return per-sample and aggregate metrics."""
    results = tuple(evaluate_sample(sample) for sample in load_samples(path))
    return results, aggregate_results(results)


def format_report(
    results: Sequence[EvaluationResult], aggregate: AggregateResult
) -> str:
    """Format per-sample metrics and macro averages as a readable table."""
    headers = ("sample_id", "CER", "WER", "Sent P", "Sent R", "Sent F1", "Composite")
    rows = [
        (
            result.sample_id,
            f"{result.cer:.4f}",
            f"{result.wer:.4f}",
            f"{result.sentence_precision:.4f}",
            f"{result.sentence_recall:.4f}",
            f"{result.sentence_f1:.4f}",
            f"{result.composite_score:.4f}",
        )
        for result in results
    ]
    rows.append(
        (
            f"MACRO ({aggregate.sample_count})",
            f"{aggregate.cer:.4f}",
            f"{aggregate.wer:.4f}",
            f"{aggregate.sentence_precision:.4f}",
            f"{aggregate.sentence_recall:.4f}",
            f"{aggregate.sentence_f1:.4f}",
            f"{aggregate.composite_score:.4f}",
        )
    )
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def format_row(row: Sequence[str]) -> str:
        return " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)
    return "\n".join((format_row(headers), separator, *(format_row(row) for row in rows)))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line evaluator and return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json", type=Path, help="Path to evaluation JSON")
    arguments = parser.parse_args(argv)

    try:
        results, aggregate = evaluate_file(arguments.input_json)
    except EvaluationInputError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(format_report(results, aggregate))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
