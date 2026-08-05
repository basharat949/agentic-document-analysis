"""Tests for Task 1.3 transcription evaluation metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from section1.eval import (
    EvaluationInputError,
    EvaluationSample,
    aggregate_results,
    character_error_rate,
    evaluate_file,
    evaluate_sample,
    load_samples,
    main,
    sentence_metrics,
    word_error_rate,
)


def test_exact_match() -> None:
    """An exact transcription receives perfect metrics."""
    result = evaluate_sample(
        EvaluationSample("exact", "One sentence. Another one!", "One sentence. Another one!")
    )

    assert result.cer == 0.0
    assert result.wer == 0.0
    assert result.sentence_precision == 1.0
    assert result.sentence_recall == 1.0
    assert result.sentence_f1 == 1.0
    assert result.composite_score == 1.0


def test_character_substitution() -> None:
    """One character substitution is normalized by reference length."""
    assert character_error_rate("cat", "cut") == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    ("ground_truth", "predicted", "expected"),
    [
        ("one two", "one bright two", 0.5),
        ("one two three", "one three", 1 / 3),
    ],
)
def test_word_insertion_and_deletion(
    ground_truth: str, predicted: str, expected: float
) -> None:
    """Word insertions and deletions use whitespace token counts."""
    assert word_error_rate(ground_truth, predicted) == pytest.approx(expected)


def test_sentence_match_at_exactly_fifty_percent_overlap() -> None:
    """The overlap threshold is inclusive at exactly 50%."""
    metrics = sentence_metrics("red blue green yellow", "RED blue")

    assert metrics.matched_sentences == 1
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0


def test_sentence_matching_is_one_to_one() -> None:
    """One predicted sentence cannot satisfy two reference sentences."""
    metrics = sentence_metrics("alpha beta. alpha gamma.", "alpha beta gamma.")

    assert metrics.matched_sentences == 1
    assert metrics.precision == 1.0
    assert metrics.recall == 0.5
    assert metrics.f1 == pytest.approx(2 / 3)


def test_empty_ground_truth_is_explicit() -> None:
    """Empty references count prediction content as unnormalized insertions."""
    result = evaluate_sample(EvaluationSample("empty-reference", "", "two words"))

    assert result.cer == 9.0
    assert result.wer == 2.0
    assert result.sentence_precision == 0.0
    assert result.sentence_recall == 0.0
    assert result.composite_score == 0.0


def test_empty_prediction() -> None:
    """A missing prediction produces full character and word error."""
    result = evaluate_sample(EvaluationSample("empty-prediction", "two words", ""))

    assert result.cer == 1.0
    assert result.wer == 1.0
    assert result.sentence_f1 == 0.0
    assert result.composite_score == 0.0


def test_both_transcriptions_empty_are_a_match() -> None:
    """Two empty transcriptions have zero errors and perfect sentence agreement."""
    result = evaluate_sample(EvaluationSample("both-empty", "", ""))

    assert result.cer == 0.0
    assert result.wer == 0.0
    assert result.sentence_f1 == 1.0
    assert result.composite_score == 1.0


def test_malformed_json(tmp_path: Path) -> None:
    """Malformed JSON is translated to a clear input exception."""
    input_path = tmp_path / "broken.json"
    input_path.write_text("[{", encoding="utf-8")

    with pytest.raises(EvaluationInputError, match="Malformed JSON"):
        load_samples(input_path)


def test_missing_required_fields(tmp_path: Path) -> None:
    """Every input sample must contain all three required fields."""
    input_path = tmp_path / "missing.json"
    input_path.write_text(json.dumps([{"sample_id": "x", "ground_truth": "text"}]))

    with pytest.raises(EvaluationInputError, match="predicted"):
        load_samples(input_path)


def test_aggregate_calculations(tmp_path: Path) -> None:
    """Macro metrics are the unweighted arithmetic mean of sample metrics."""
    payload = [
        {"sample_id": "perfect", "ground_truth": "hello.", "predicted": "hello."},
        {"sample_id": "missing", "ground_truth": "hello.", "predicted": ""},
    ]
    input_path = tmp_path / "samples.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    results, aggregate = evaluate_file(input_path)

    assert aggregate.sample_count == 2
    assert aggregate.cer == 0.5
    assert aggregate.wer == 0.5
    assert aggregate.sentence_precision == 0.5
    assert aggregate.sentence_recall == 0.5
    assert aggregate.sentence_f1 == 0.5
    assert aggregate.composite_score == 0.5
    assert aggregate == aggregate_results(results)


def test_cli_returns_nonzero_for_malformed_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI reports malformed input on stderr and returns a non-zero code."""
    input_path = tmp_path / "bad.json"
    input_path.write_text("not-json", encoding="utf-8")

    exit_code = main([str(input_path)])

    assert exit_code != 0
    assert "Malformed JSON" in capsys.readouterr().err
