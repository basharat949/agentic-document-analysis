"""Tests for Task 2.2 Part C deterministic routing and finalization."""

from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import Mock, call

import pytest

from section2.classifier import (
    AgentName,
    AgentResponseValidationError,
    ClassifierAgentResult,
    EmbeddedAgentResult,
    EmbeddedSpan,
    EmbeddedStatus,
    ModelValidationError,
    SentenceCategory,
    SentenceClassificationPipeline,
)


def _classifier_result(
    sentence: str,
    category: SentenceCategory,
) -> ClassifierAgentResult:
    """Create a deterministic valid primary classifier result."""
    return ClassifierAgentResult(sentence, category, "Primary structural decision.")


def _embedded_result(
    sentence: str,
    status: EmbeddedStatus,
    spans: tuple[EmbeddedSpan, ...] = (),
) -> EmbeddedAgentResult:
    """Create a deterministic valid embedded-agent result."""
    return EmbeddedAgentResult(
        input_sentence=sentence,
        status=status,
        embedded_spans=spans,
        reason="Embedded structural decision.",
    )


class ClassifierClient:
    """Primary client double returning one configured batch."""

    def __init__(self, responses: Sequence[ClassifierAgentResult]) -> None:
        self.classify_batch = Mock(return_value=tuple(responses))
        self.classify_one = Mock()


class EmbeddedClient:
    """Embedded client double returning configured occurrence responses."""

    def __init__(self, responses: Sequence[object] = ()) -> None:
        self.analyze = Mock(side_effect=list(responses))


def _pipeline(
    classifier_results: Sequence[ClassifierAgentResult],
    embedded_results: Sequence[object] = (),
) -> tuple[SentenceClassificationPipeline, ClassifierClient, EmbeddedClient]:
    """Build a pipeline and expose its recording client doubles."""
    classifier = ClassifierClient(classifier_results)
    embedded = EmbeddedClient(embedded_results)
    return SentenceClassificationPipeline(classifier, embedded), classifier, embedded


def test_normal_batch_never_calls_embedded_agent() -> None:
    """Non-incomplete categories finish on the direct classifier path."""
    sentences = ("Birds sing.", "I called and she answered.")
    pipeline, classifier, embedded = _pipeline(
        (
            _classifier_result(sentences[0], SentenceCategory.SIMPLE),
            _classifier_result(sentences[1], SentenceCategory.COMPOUND),
        )
    )

    result = pipeline.classify(sentences)

    classifier.classify_batch.assert_called_once_with(sentences)
    embedded.analyze.assert_not_called()
    assert tuple(item.agent_path for item in result.results) == (
        (AgentName.CLASSIFIER,),
        (AgentName.CLASSIFIER,),
    )


def test_single_embedded_span_promotes_final_category() -> None:
    """A single complete span supplies the final structural category."""
    sentence = "Becaus she said I dont know"
    span = EmbeddedSpan("I dont know", True, SentenceCategory.SIMPLE)
    pipeline, _, embedded = _pipeline(
        (_classifier_result(sentence, SentenceCategory.INCOMPLETE),),
        (_embedded_result(sentence, EmbeddedStatus.SINGLE, (span,)),),
    )

    result = pipeline.classify((sentence,)).results[0]

    embedded.analyze.assert_called_once_with(sentence)
    assert result.final_category is SentenceCategory.SIMPLE
    assert result.original_classifier_category is SentenceCategory.INCOMPLETE
    assert result.embedded_sentence == "I dont know"
    assert result.agent_path == (
        AgentName.CLASSIFIER,
        AgentName.EMBEDDED_SENTENCE,
    )
    assert result.reason.startswith("Classifier: ")
    assert " Embedded agent: " in result.reason


def test_no_embedded_sentence_remains_incomplete() -> None:
    """Status none retains Incomplete and records both invoked agents."""
    sentence = "After the noisey bus"
    pipeline, _, _ = _pipeline(
        (_classifier_result(sentence, SentenceCategory.INCOMPLETE),),
        (_embedded_result(sentence, EmbeddedStatus.NONE),),
    )

    result = pipeline.classify((sentence,)).results[0]

    assert result.final_category is SentenceCategory.INCOMPLETE
    assert result.embedded_sentence is None
    assert result.agent_path == (
        AgentName.CLASSIFIER,
        AgentName.EMBEDDED_SENTENCE,
    )
    assert "No complete embedded sentence was found" in result.reason


def test_incomplete_only_remains_incomplete() -> None:
    """Nested content without a complete span cannot promote the category."""
    sentence = "She mutterd that old broken window"
    span = EmbeddedSpan("that old broken window", False, None)
    pipeline, _, _ = _pipeline(
        (_classifier_result(sentence, SentenceCategory.INCOMPLETE),),
        (_embedded_result(sentence, EmbeddedStatus.INCOMPLETE_ONLY, (span,)),),
    )

    result = pipeline.classify((sentence,)).results[0]

    assert result.final_category is SentenceCategory.INCOMPLETE
    assert result.embedded_sentence is None
    assert "Only incomplete nested content was found" in result.reason


def test_multiple_selects_highest_category() -> None:
    """Compound-Complex outranks every other complete embedded category."""
    sentence = "Simple part then compound part then complex part then top part"
    spans = (
        EmbeddedSpan("Simple part", True, SentenceCategory.SIMPLE),
        EmbeddedSpan("compound part", True, SentenceCategory.COMPOUND),
        EmbeddedSpan("complex part", True, SentenceCategory.COMPLEX),
        EmbeddedSpan("top part", True, SentenceCategory.COMPOUND_COMPLEX),
    )
    pipeline, _, _ = _pipeline(
        (_classifier_result(sentence, SentenceCategory.INCOMPLETE),),
        (_embedded_result(sentence, EmbeddedStatus.MULTIPLE, spans),),
    )

    result = pipeline.classify((sentence,)).results[0]

    assert result.final_category is SentenceCategory.COMPOUND_COMPLEX
    assert result.embedded_sentence == "top part"


def test_multiple_tie_selects_first_in_source_order() -> None:
    """The first span wins when highest-complexity categories tie."""
    sentence = "first complex then second complex"
    spans = (
        EmbeddedSpan("first complex", True, SentenceCategory.COMPLEX),
        EmbeddedSpan("second complex", True, SentenceCategory.COMPLEX),
    )
    pipeline, _, _ = _pipeline(
        (_classifier_result(sentence, SentenceCategory.INCOMPLETE),),
        (_embedded_result(sentence, EmbeddedStatus.MULTIPLE, spans),),
    )

    result = pipeline.classify((sentence,)).results[0]

    assert result.embedded_sentence == "first complex"


def test_multiple_ignores_incomplete_spans() -> None:
    """Incomplete entries never participate in final category ranking."""
    sentence = "broken bit then a complete thought"
    spans = (
        EmbeddedSpan("broken bit", False, None),
        EmbeddedSpan("a complete thought", True, SentenceCategory.COMPOUND),
    )
    pipeline, _, _ = _pipeline(
        (_classifier_result(sentence, SentenceCategory.INCOMPLETE),),
        (_embedded_result(sentence, EmbeddedStatus.MULTIPLE, spans),),
    )

    result = pipeline.classify((sentence,)).results[0]

    assert result.final_category is SentenceCategory.COMPOUND
    assert result.embedded_sentence == "a complete thought"


def test_invalid_category_is_contract_violation() -> None:
    """An impossible invalid-category response raises rather than finalizing."""
    sentence = "Incomplete fragment"
    pipeline, _, _ = _pipeline(
        (_classifier_result(sentence, SentenceCategory.INCOMPLETE),),
        (_embedded_result(sentence, EmbeddedStatus.INVALID_CATEGORY),),
    )

    with pytest.raises(AgentResponseValidationError, match="invalid_category"):
        pipeline.classify((sentence,))


def test_mutated_embedded_input_sentence_is_rejected() -> None:
    """The embedded result must copy the routed sentence exactly."""
    sentence = "I recieved it because"
    pipeline, _, _ = _pipeline(
        (_classifier_result(sentence, SentenceCategory.INCOMPLETE),),
        (_embedded_result("I received it because", EmbeddedStatus.NONE),),
    )

    with pytest.raises(AgentResponseValidationError, match="exactly match"):
        pipeline.classify((sentence,))


def test_wrong_embedded_agent_return_type_is_rejected() -> None:
    """Routing accepts only a validated EmbeddedAgentResult model."""
    sentence = "Incomplete fragment"
    pipeline, _, _ = _pipeline(
        (_classifier_result(sentence, SentenceCategory.INCOMPLETE),),
        (object(),),
    )

    with pytest.raises(ModelValidationError, match="EmbeddedAgentResult"):
        pipeline.classify((sentence,))


def test_mixed_batch_returns_original_order() -> None:
    """Direct and embedded routes merge back into original input order."""
    sentences = ("Direct one.", "Because inner works", "Direct two.")
    primary = (
        _classifier_result(sentences[2], SentenceCategory.COMPLEX),
        _classifier_result(sentences[1], SentenceCategory.INCOMPLETE),
        _classifier_result(sentences[0], SentenceCategory.SIMPLE),
    )
    embedded_result = _embedded_result(
        sentences[1],
        EmbeddedStatus.SINGLE,
        (EmbeddedSpan("inner works", True, SentenceCategory.SIMPLE),),
    )
    pipeline, _, _ = _pipeline(primary, (embedded_result,))

    result = pipeline.classify(sentences)

    assert tuple(item.original_sentence for item in result.results) == sentences
    assert tuple(item.final_category for item in result.results) == (
        SentenceCategory.SIMPLE,
        SentenceCategory.SIMPLE,
        SentenceCategory.COMPLEX,
    )


def test_duplicate_incomplete_inputs_each_call_embedded_agent() -> None:
    """Every duplicate occurrence receives its own embedded-agent call."""
    sentence = "After the noisey bus"
    pipeline, _, embedded = _pipeline(
        (
            _classifier_result(sentence, SentenceCategory.INCOMPLETE),
            _classifier_result(sentence, SentenceCategory.INCOMPLETE),
        ),
        (
            _embedded_result(sentence, EmbeddedStatus.NONE),
            _embedded_result(sentence, EmbeddedStatus.NONE),
        ),
    )

    result = pipeline.classify((sentence, sentence))

    assert embedded.analyze.call_args_list == [call(sentence), call(sentence)]
    assert len(result.results) == 2


def test_empty_input_calls_neither_client() -> None:
    """An empty batch returns an empty immutable pipeline result."""
    pipeline, classifier, embedded = _pipeline(())

    result = pipeline.classify([])

    assert result.results == ()
    classifier.classify_batch.assert_not_called()
    classifier.classify_one.assert_not_called()
    embedded.analyze.assert_not_called()
