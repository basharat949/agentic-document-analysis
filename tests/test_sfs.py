"""Tests for the heuristic Source-Fidelity Score."""

from __future__ import annotations

from typing import Any

import pytest

from section4.sfs import _identify_fidelity_sensitive_occurrences, sfs


def test_exact_preservation_of_sensitive_tokens() -> None:
    """Exact punctuation and mixed-case forms receive full credit."""
    assert sfs("Error!!! iPhone", "Error!!! iPhone") == 1.0


def test_punctuation_normalization_loses_preservation() -> None:
    """Simplifying attached punctuation is not an exact sensitive match."""
    assert sfs("Wait!!!", "Wait!") == 0.0


def test_mixed_case_normalization_loses_preservation() -> None:
    """Case normalization changes the fidelity-sensitive token."""
    assert sfs("iPhone works", "iphone works") == 0.0


def test_duplicate_sensitive_tokens_are_counted_by_occurrence() -> None:
    """One predicted duplicate preserves only one of two source occurrences."""
    assert sfs("WOW WOW", "WOW") == 0.5


def test_sensitive_insertion_does_not_increase_preservation() -> None:
    """A sensitive prediction-only insertion cannot change a zero denominator."""
    assert sfs("plain text", "plain text WOW") == 1.0


def test_sensitive_token_deletion_loses_preservation() -> None:
    """Deleting the only sensitive source occurrence produces zero credit."""
    assert sfs("hello??? world", "world") == 0.0


def test_no_sensitive_source_tokens_returns_one_by_design() -> None:
    """The zero-denominator rule shows why SFS must complement CER and WER."""
    assert sfs("plain lowercase words", "different words") == 1.0


@pytest.mark.parametrize(
    ("ground_truth", "predicted"),
    [("", ""), ("", "extra")],
)
def test_empty_ground_truth_returns_one(ground_truth: str, predicted: str) -> None:
    """An empty reference has no sensitive occurrences."""
    assert sfs(ground_truth, predicted) == 1.0


@pytest.mark.parametrize("ground_truth", [None, 1, [], True])
def test_non_string_ground_truth_raises_type_error(ground_truth: Any) -> None:
    """The public metric rejects non-string reference values."""
    with pytest.raises(TypeError, match="ground_truth"):
        sfs(ground_truth, "prediction")


@pytest.mark.parametrize("predicted", [None, 1, [], True])
def test_non_string_prediction_raises_type_error(predicted: Any) -> None:
    """The public metric rejects non-string prediction values."""
    with pytest.raises(TypeError, match="predicted"):
        sfs("reference", predicted)


def test_non_ascii_token_requires_exact_preservation() -> None:
    """A non-ASCII token is sensitive and must retain its accents."""
    assert sfs("café", "café") == 1.0
    assert sfs("café", "cafe") == 0.0


def test_alphanumeric_token_requires_exact_preservation() -> None:
    """Mixed letters and digits identify a fidelity-sensitive form."""
    assert sfs("AB12", "AB12") == 1.0
    assert sfs("AB12", "AB13") == 0.0


def test_long_repeated_character_token_requires_exact_preservation() -> None:
    """Normalizing a character run removes preservation credit."""
    assert sfs("soooo", "soooo") == 1.0
    assert sfs("soooo", "so") == 0.0


def test_alignment_preserves_sensitive_token_after_ordinary_insertion() -> None:
    """An inserted ordinary token does not disrupt an exact sensitive match."""
    assert sfs("hello!!! world", "extra hello!!! world") == 1.0


def test_reordered_sensitive_token_uses_best_minimum_edit_alignment() -> None:
    """A tied minimum-edit path preserves the moved exact sensitive occurrence.

    Moving ``WOW`` is represented by insertion/deletion at the same minimum cost
    as substitutions. The documented fidelity-preserving tie-breaker selects the
    path that aligns ``WOW`` exactly, so the current behavior is SFS 1.0.
    """
    assert sfs("WOW plain", "plain WOW") == 1.0


def test_known_limitation_ordinary_looking_misspelling_is_not_detected() -> None:
    """The heuristic misses lowercase alphabetic ``chlid`` and returns 1.0."""
    assert sfs("chlid", "child") == 1.0


def test_sentence_initial_capitalization_is_a_false_positive_edge_case() -> None:
    """The current surface heuristic treats ordinary initial ``The`` as sensitive."""
    assert sfs("The cat", "A cat") == 0.0


@pytest.mark.parametrize("threshold", [0, -1, True, 1.5, "2", None])
def test_repetition_threshold_requires_positive_non_boolean_integer(
    threshold: Any,
) -> None:
    """Internal configuration rejects invalid repetition thresholds clearly."""
    with pytest.raises(ValueError, match="positive integer"):
        _identify_fidelity_sensitive_occurrences(
            ("soooo",),
            repetition_threshold=threshold,
        )
