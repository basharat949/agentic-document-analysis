"""Heuristic Source-Fidelity Score for verbatim transcription.

This module operationalizes source fidelity without a dictionary, spell checker,
or language model. Without human labels or a lexicon, it cannot reliably detect
ordinary-looking misspellings such as ``"chlid"``. The score is therefore a
heuristic that complements CER and WER rather than replacing them.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Collection, Sequence

__all__ = ("sfs",)

_DEFAULT_REPETITION_THRESHOLD = 2

type _AlignmentScore = tuple[int, int]


def sfs(ground_truth: str, predicted: str) -> float:
    """Estimate preservation of fidelity-sensitive source-token occurrences.

    Text is split only on whitespace. Token spelling, case, and attached
    punctuation are otherwise retained exactly. A deterministic heuristic marks
    unusual source tokens as fidelity-sensitive, then a Levenshtein-style token
    alignment counts sensitive occurrences copied exactly.

    This heuristic cannot reliably recognize ordinary-looking misspellings such
    as ``"chlid"`` without human annotations or a lexicon. It is intended to
    complement CER and WER, not replace them.

    Args:
        ground_truth: Verbatim reference transcription.
        predicted: Verbatim predicted transcription.

    Returns:
        The fraction of fidelity-sensitive reference-token occurrences preserved
        exactly. Returns ``1.0`` when the reference has no sensitive tokens.

    Raises:
        TypeError: If either argument is not a string.
    """
    if not isinstance(ground_truth, str):
        raise TypeError("ground_truth must be a string")
    if not isinstance(predicted, str):
        raise TypeError("predicted must be a string")

    reference_tokens = tuple(ground_truth.split())
    predicted_tokens = tuple(predicted.split())
    sensitive_occurrences = _identify_fidelity_sensitive_occurrences(
        reference_tokens
    )
    denominator = sum(sensitive_occurrences)
    if denominator == 0:
        return 1.0

    preserved = _count_preserved_sensitive_occurrences(
        reference_tokens,
        predicted_tokens,
        sensitive_occurrences,
    )
    return float(preserved / denominator)


def _identify_fidelity_sensitive_occurrences(
    tokens: Sequence[str],
    *,
    repetition_threshold: int = _DEFAULT_REPETITION_THRESHOLD,
    sensitive_tokens: Collection[str] = frozenset(),
) -> tuple[bool, ...]:
    """Identify sensitive token occurrences with deterministic surface rules.

    ``sensitive_tokens`` permits an internally supplied, exactly matched set of
    human-labelled forms without changing the public ``sfs`` signature.
    Repetition is flagged when a consecutive run is longer than
    ``repetition_threshold``.
    """
    if (
        isinstance(repetition_threshold, bool)
        or not isinstance(repetition_threshold, int)
        or repetition_threshold < 1
    ):
        raise ValueError("repetition_threshold must be a positive integer")
    return tuple(
        token in sensitive_tokens
        or _is_fidelity_sensitive(token, repetition_threshold)
        for token in tokens
    )


def _is_fidelity_sensitive(token: str, repetition_threshold: int) -> bool:
    """Return whether a token has a fidelity-sensitive surface form."""
    has_letter = any(character.isalpha() for character in token)
    has_digit = any(character.isdigit() for character in token)

    return any(
        (
            _has_unusual_case(token),
            any(unicodedata.category(character).startswith("P") for character in token),
            has_letter and has_digit,
            _has_long_character_run(token, repetition_threshold),
            any(not character.isascii() for character in token),
            not (token.isascii() and token.isalpha() and token.islower()),
        )
    )


def _has_unusual_case(token: str) -> bool:
    """Detect internal uppercase or mixed-case patterns such as ``iPhone``."""
    letters = tuple(character for character in token if character.isalpha())
    if not letters:
        return False
    has_upper = any(character.isupper() for character in letters)
    has_lower = any(character.islower() for character in letters)
    if not (has_upper and has_lower):
        return False
    return letters[0].islower() or any(character.isupper() for character in letters[1:])


def _has_long_character_run(token: str, threshold: int) -> bool:
    """Detect a consecutive repeated-character run longer than ``threshold``."""
    previous: str | None = None
    run_length = 0
    for character in token:
        if character == previous:
            run_length += 1
        else:
            previous = character
            run_length = 1
        if run_length > threshold:
            return True
    return False


def _count_preserved_sensitive_occurrences(
    reference: Sequence[str],
    predicted: Sequence[str],
    sensitive: Sequence[bool],
) -> int:
    """Align tokens and return exactly copied sensitive reference occurrences.

    Each dynamic-programming cell stores ``(edit_distance, preserved_count)``.
    The primary objective is minimum Levenshtein distance. When several minimum
    paths tie, the path preserving more sensitive occurrences wins. A final
    deterministic priority prefers diagonal, deletion, then insertion. Because
    alignment operates on indices, duplicate source tokens remain distinct
    occurrences rather than collapsing through set membership.
    """
    previous_row: list[_AlignmentScore] = [
        (predicted_count, 0) for predicted_count in range(len(predicted) + 1)
    ]

    for reference_index, reference_token in enumerate(reference, start=1):
        current_row: list[_AlignmentScore] = [(reference_index, 0)]
        is_sensitive = sensitive[reference_index - 1]

        for predicted_index, predicted_token in enumerate(predicted, start=1):
            exact_match = reference_token == predicted_token
            diagonal_cost, diagonal_preserved = previous_row[predicted_index - 1]
            candidates = (
                (
                    diagonal_cost + (not exact_match),
                    diagonal_preserved + (is_sensitive and exact_match),
                    0,
                ),
                (
                    previous_row[predicted_index][0] + 1,
                    previous_row[predicted_index][1],
                    1,
                ),
                (
                    current_row[predicted_index - 1][0] + 1,
                    current_row[predicted_index - 1][1],
                    2,
                ),
            )
            best_cost, best_preserved, _ = min(
                candidates,
                key=lambda candidate: (
                    candidate[0],
                    -candidate[1],
                    candidate[2],
                ),
            )
            current_row.append((best_cost, best_preserved))

        previous_row = current_row

    return previous_row[-1][1]
