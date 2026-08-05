"""Tests for Task 2.2 Part B primary batch classification."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from section2.classifier import (
    AgentResponseValidationError,
    ClassifierAgentResult,
    EmbeddedAgentResult,
    ModelValidationError,
    SentenceCategory,
    SentenceClassificationPipeline,
)


def _result(
    sentence: str,
    category: SentenceCategory = SentenceCategory.SIMPLE,
) -> ClassifierAgentResult:
    """Create a valid deterministic classifier response."""
    return ClassifierAgentResult(
        sentence=sentence,
        category=category,
        reason="Deterministic test response.",
    )


class RecordingClassifierClient:
    """Configurable classifier double with call-counting mocks."""

    def __init__(
        self,
        batch_response: object,
        individual_responses: dict[str, ClassifierAgentResult] | None = None,
    ) -> None:
        self.classify_batch = Mock(return_value=batch_response)
        responses = individual_responses or {}
        self.classify_one = Mock(side_effect=lambda sentence: responses[sentence])


class UnusedEmbeddedClient:
    """Embedded client that fails if Part B invokes it."""

    def analyze(self, sentence: str) -> EmbeddedAgentResult:
        raise AssertionError(f"Embedded agent must not run in Part B: {sentence!r}")


def _pipeline(classifier: RecordingClassifierClient) -> SentenceClassificationPipeline:
    """Build a pipeline with an embedded client that must remain unused."""
    return SentenceClassificationPipeline(classifier, UnusedEmbeddedClient())


def test_normal_batch_is_called_once_and_preserves_order() -> None:
    """A complete batch needs no individual recovery calls."""
    sentences = ("First.", "Second.", "Third.")
    classifier = RecordingClassifierClient(tuple(_result(value) for value in sentences))

    primary = _pipeline(classifier)._classify_primary(sentences)

    classifier.classify_batch.assert_called_once_with(sentences)
    classifier.classify_one.assert_not_called()
    assert tuple(item.input_index for item in primary) == (0, 1, 2)
    assert tuple(item.result.sentence for item in primary) == sentences


def test_missing_batch_item_is_recovered_in_original_order() -> None:
    """One omitted middle occurrence is recovered without changing order."""
    sentences = ("First.", "Second.", "Third.")
    classifier = RecordingClassifierClient(
        (_result("First."), _result("Third.")),
        {"Second.": _result("Second.", SentenceCategory.COMPOUND)},
    )

    primary = _pipeline(classifier)._classify_primary(sentences)

    classifier.classify_one.assert_called_once_with("Second.")
    assert tuple(item.input_index for item in primary) == (0, 1, 2)
    assert primary[1].result.category is SentenceCategory.COMPOUND


def test_several_missing_items_are_recovered_individually() -> None:
    """Each missing occurrence produces one individual classifier call."""
    sentences = ("One.", "Two.", "Three.", "Four.")
    classifier = RecordingClassifierClient(
        (_result("Three."),),
        {
            "One.": _result("One."),
            "Two.": _result("Two."),
            "Four.": _result("Four."),
        },
    )

    primary = _pipeline(classifier)._classify_primary(sentences)

    assert classifier.classify_one.call_args_list == [
        (("One.",),),
        (("Two.",),),
        (("Four.",),),
    ]
    assert tuple(item.input_index for item in primary) == (0, 1, 2, 3)


def test_duplicate_inputs_map_to_distinct_occurrences() -> None:
    """Two matching responses satisfy two duplicate input indices."""
    sentences = ("Same.", "Same.")
    classifier = RecordingClassifierClient((_result("Same."), _result("Same.")))

    primary = _pipeline(classifier)._classify_primary(sentences)

    classifier.classify_one.assert_not_called()
    assert tuple(item.input_index for item in primary) == (0, 1)


def test_missing_duplicate_occurrence_is_recovered_once() -> None:
    """One batch response for two duplicates leaves one indexed occurrence."""
    sentences = ("Same.", "Same.")
    classifier = RecordingClassifierClient(
        (_result("Same."),),
        {"Same.": _result("Same.")},
    )

    primary = _pipeline(classifier)._classify_primary(sentences)

    classifier.classify_one.assert_called_once_with("Same.")
    assert tuple(item.input_index for item in primary) == (0, 1)


def test_unknown_response_sentence_is_rejected() -> None:
    """A batch response cannot be assigned to an unknown input sentence."""
    classifier = RecordingClassifierClient((_result("Unknown."),))

    with pytest.raises(AgentResponseValidationError, match="remaining input"):
        _pipeline(classifier)._classify_primary(("Expected.",))


def test_too_many_duplicate_responses_are_rejected() -> None:
    """Responses beyond available matching occurrences fail validation."""
    classifier = RecordingClassifierClient((_result("Same."), _result("Same.")))

    with pytest.raises(AgentResponseValidationError, match="remaining input"):
        _pipeline(classifier)._classify_primary(("Same.",))


@pytest.mark.parametrize("mutated", ["recieved it", "I  recieved it", "I recieved it!"])
def test_mutated_response_sentence_is_rejected(mutated: str) -> None:
    """Spelling, whitespace, and punctuation mutations cannot be mapped."""
    classifier = RecordingClassifierClient((_result(mutated),))

    with pytest.raises(AgentResponseValidationError):
        _pipeline(classifier)._classify_primary(("I recieved it",))


@pytest.mark.parametrize("batch_response", ["not a result sequence", 42, None])
def test_invalid_batch_container_is_rejected(batch_response: object) -> None:
    """The batch client must return a non-string sequence."""
    classifier = RecordingClassifierClient(batch_response)

    with pytest.raises(ModelValidationError, match="non-string sequence"):
        _pipeline(classifier)._classify_primary(("Input.",))


def test_invalid_batch_item_is_rejected() -> None:
    """Every item in a valid response container must use the result model."""
    classifier = RecordingClassifierClient((object(),))

    with pytest.raises(ModelValidationError, match="response at index 0"):
        _pipeline(classifier)._classify_primary(("Input.",))


def test_invalid_individual_response_is_rejected() -> None:
    """Recovery calls must also return validated classifier result models."""
    classifier = RecordingClassifierClient(())
    classifier.classify_one.return_value = object()
    classifier.classify_one.side_effect = None

    with pytest.raises(ModelValidationError, match="classify_one response"):
        _pipeline(classifier)._classify_primary(("Input.",))


def test_empty_tuple_calls_neither_classifier_method() -> None:
    """Empty primary input returns immediately without touching the client."""
    classifier = RecordingClassifierClient(())

    assert _pipeline(classifier)._classify_primary(()) == ()
    classifier.classify_batch.assert_not_called()
    classifier.classify_one.assert_not_called()


def test_public_classify_executes_primary_then_stops_at_part_c() -> None:
    """The public skeleton completes Part B before raising its Part C marker."""
    classifier = RecordingClassifierClient((_result("Input."),))
    pipeline = _pipeline(classifier)

    with pytest.raises(NotImplementedError, match="Part C"):
        pipeline.classify(["Input."])

    classifier.classify_batch.assert_called_once_with(("Input.",))
    classifier.classify_one.assert_not_called()
