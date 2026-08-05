"""Tests for Task 2.2 Part A models, interfaces, and pipeline skeleton."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from section2.classifier import (
    AgentName,
    AgentResponseValidationError,
    ClassificationPipelineResult,
    ClassifierAgentClient,
    ClassifierAgentResult,
    EmbeddedAgentResult,
    EmbeddedSentenceAgentClient,
    EmbeddedSpan,
    EmbeddedStatus,
    FinalClassification,
    InvalidSentenceError,
    ModelValidationError,
    SentenceCategory,
    SentenceClassificationPipeline,
)


class StubClassifierClient:
    """Non-network classifier double used to verify dependency injection."""

    def classify_batch(
        self, sentences: list[str] | tuple[str, ...]
    ) -> tuple[ClassifierAgentResult, ...]:
        return tuple(self.classify_one(sentence) for sentence in sentences)

    def classify_one(self, sentence: str) -> ClassifierAgentResult:
        return ClassifierAgentResult(
            sentence=sentence,
            category=SentenceCategory.SIMPLE,
            reason="One independent clause.",
        )


class StubEmbeddedClient:
    """Non-network embedded-agent double used to verify injection."""

    def analyze(self, sentence: str) -> EmbeddedAgentResult:
        return EmbeddedAgentResult(
            input_sentence=sentence,
            status=EmbeddedStatus.NONE,
            embedded_spans=(),
            reason="No embedded sentence.",
        )


def test_valid_model_construction() -> None:
    """Core models retain validated values and ordered immutable results."""
    classifier_result = ClassifierAgentResult(
        sentence="Birds sing.",
        category=SentenceCategory.SIMPLE,
        reason="One independent clause.",
    )
    final = FinalClassification(
        original_sentence="Birds sing.",
        final_category=SentenceCategory.SIMPLE,
        original_classifier_category=SentenceCategory.SIMPLE,
        reason="One independent clause.",
        embedded_sentence=None,
        agent_path=(AgentName.CLASSIFIER,),
    )
    pipeline_result = ClassificationPipelineResult(results=(final,))

    assert classifier_result.category is SentenceCategory.SIMPLE
    assert pipeline_result.results == (final,)


def test_invalid_category_is_rejected() -> None:
    """Raw or unknown category values cannot bypass the enum boundary."""
    with pytest.raises(ModelValidationError, match="SentenceCategory"):
        ClassifierAgentResult(
            sentence="Birds sing.",
            category=cast(SentenceCategory, "Unknown"),
            reason="Invalid category fixture.",
        )


@pytest.mark.parametrize("sentence", ["", 42, None])
def test_empty_or_non_string_sentence_is_rejected(sentence: object) -> None:
    """Models reject empty and non-text source sentences."""
    with pytest.raises(InvalidSentenceError):
        ClassifierAgentResult(
            sentence=cast(str, sentence),
            category=SentenceCategory.SIMPLE,
            reason="One clause.",
        )


def test_classifier_response_requires_exact_sentence_copy() -> None:
    """Even spelling correction makes the response fail copy validation."""
    result = ClassifierAgentResult(
        sentence="I recieved it.",
        category=SentenceCategory.SIMPLE,
        reason="One independent clause.",
    )

    result.validate_input_copy("I recieved it.")
    with pytest.raises(AgentResponseValidationError, match="exactly match"):
        result.validate_input_copy("I received it.")


def test_valid_embedded_span_is_contiguous() -> None:
    """A verbatim contiguous embedded span is accepted and retained."""
    result = EmbeddedAgentResult(
        input_sentence="Becaus she said I dont know",
        status=EmbeddedStatus.SINGLE,
        embedded_spans=(
            EmbeddedSpan("I dont know", True, SentenceCategory.SIMPLE),
        ),
        reason="One complete embedded proposition.",
    )

    result.validate_input_copy("Becaus she said I dont know")
    assert result.embedded_spans[0].text == "I dont know"


def test_valid_incomplete_span_has_no_category() -> None:
    """An incomplete-only response retains a span with a null category."""
    span = EmbeddedSpan("that old broken window", False, None)
    result = EmbeddedAgentResult(
        input_sentence="He mutterd that old broken window",
        status=EmbeddedStatus.INCOMPLETE_ONLY,
        embedded_spans=(span,),
        reason="The nested content lacks a finite predicate.",
    )

    assert result.embedded_spans == (span,)
    assert result.embedded_spans[0].category is None


def test_invalid_embedded_span_is_rejected() -> None:
    """A corrected or assembled span cannot pass substring validation."""
    with pytest.raises(AgentResponseValidationError, match="contiguous"):
        EmbeddedAgentResult(
            input_sentence="Becaus she said I dont know",
            status=EmbeddedStatus.SINGLE,
            embedded_spans=(
                EmbeddedSpan("I don't know", True, SentenceCategory.SIMPLE),
            ),
            reason="Invalid corrected span fixture.",
        )


@pytest.mark.parametrize(
    ("is_complete", "category"),
    [
        (True, None),
        (True, SentenceCategory.INCOMPLETE),
        (False, SentenceCategory.SIMPLE),
    ],
)
def test_embedded_span_category_consistency(
    is_complete: bool,
    category: SentenceCategory | None,
) -> None:
    """Complete and incomplete spans enforce their category contract."""
    with pytest.raises(ModelValidationError):
        EmbeddedSpan("source span", is_complete, category)


@pytest.mark.parametrize(
    ("status", "spans"),
    [
        (
            EmbeddedStatus.NONE,
            (EmbeddedSpan("I dont know", True, SentenceCategory.SIMPLE),),
        ),
        (EmbeddedStatus.SINGLE, ()),
        (
            EmbeddedStatus.MULTIPLE,
            (EmbeddedSpan("I dont know", True, SentenceCategory.SIMPLE),),
        ),
        (
            EmbeddedStatus.INCOMPLETE_ONLY,
            (EmbeddedSpan("I dont know", True, SentenceCategory.SIMPLE),),
        ),
    ],
)
def test_embedded_status_span_consistency(
    status: EmbeddedStatus,
    spans: tuple[EmbeddedSpan, ...],
) -> None:
    """Every embedded result status enforces its required span shape."""
    with pytest.raises(ModelValidationError):
        EmbeddedAgentResult(
            input_sentence="She said I dont know",
            status=status,
            embedded_spans=spans,
            reason="Invalid status fixture.",
        )


def test_multiple_spans_must_preserve_source_order() -> None:
    """Multiple spans are accepted only in their original left-to-right order."""
    first = EmbeddedSpan("I lost it", True, SentenceCategory.SIMPLE)
    second = EmbeddedSpan("the shop is shut", True, SentenceCategory.SIMPLE)

    valid = EmbeddedAgentResult(
        input_sentence="She said I lost it then the shop is shut",
        status=EmbeddedStatus.MULTIPLE,
        embedded_spans=(first, second),
        reason="Two complete spans.",
    )
    assert valid.embedded_spans == (first, second)

    with pytest.raises(AgentResponseValidationError, match="source order"):
        EmbeddedAgentResult(
            input_sentence="She said I lost it then the shop is shut",
            status=EmbeddedStatus.MULTIPLE,
            embedded_spans=(second, first),
            reason="Reversed spans.",
        )


def test_invalid_embedded_status_is_rejected() -> None:
    """Raw status strings cannot bypass the EmbeddedStatus enum."""
    with pytest.raises(ModelValidationError, match="EmbeddedStatus"):
        EmbeddedAgentResult(
            input_sentence="No inner sentence",
            status=cast(EmbeddedStatus, "none"),
            embedded_spans=(),
            reason="Invalid raw status fixture.",
        )


def test_agent_path_is_immutable_and_known() -> None:
    """Agent paths become tuples and reject names outside AgentName."""
    final = FinalClassification(
        original_sentence="Birds sing.",
        final_category=SentenceCategory.SIMPLE,
        original_classifier_category=SentenceCategory.SIMPLE,
        reason="One independent clause.",
        embedded_sentence=None,
        agent_path=cast(tuple[AgentName, ...], [AgentName.CLASSIFIER]),
    )

    assert final.agent_path == (AgentName.CLASSIFIER,)
    with pytest.raises(FrozenInstanceError):
        final.agent_path = ()  # type: ignore[misc]

    with pytest.raises(ModelValidationError, match="known AgentName"):
        FinalClassification(
            original_sentence="Birds sing.",
            final_category=SentenceCategory.SIMPLE,
            original_classifier_category=SentenceCategory.SIMPLE,
            reason="One independent clause.",
            embedded_sentence=None,
            agent_path=cast(tuple[AgentName, ...], ("other_agent",)),
        )


def test_pipeline_uses_dependency_injection() -> None:
    """The pipeline stores the exact clients supplied by its caller."""
    classifier = StubClassifierClient()
    embedded = StubEmbeddedClient()
    pipeline = SentenceClassificationPipeline(
        classifier_client=cast(ClassifierAgentClient, classifier),
        embedded_agent_client=cast(EmbeddedSentenceAgentClient, embedded),
    )

    assert pipeline.classifier_client is classifier
    assert pipeline.embedded_agent_client is embedded


def test_classify_rejects_empty_sentence_before_skeleton_error() -> None:
    """Public orchestration validates every input before stopping at the skeleton."""
    pipeline = SentenceClassificationPipeline(
        classifier_client=StubClassifierClient(),
        embedded_agent_client=StubEmbeddedClient(),
    )

    with pytest.raises(InvalidSentenceError, match=r"sentences\[1\]"):
        pipeline.classify(("Valid.", ""))


def test_classify_preserves_duplicate_inputs_by_index() -> None:
    """Duplicates produce distinct final results through index-based mapping."""
    pipeline = SentenceClassificationPipeline(
        classifier_client=StubClassifierClient(),
        embedded_agent_client=StubEmbeddedClient(),
    )

    result = pipeline.classify(("Same.", "Same."))

    assert tuple(item.original_sentence for item in result.results) == (
        "Same.",
        "Same.",
    )


def test_classify_returns_final_result() -> None:
    """The completed orchestrator returns a validated final classification."""
    pipeline = SentenceClassificationPipeline(
        classifier_client=StubClassifierClient(),
        embedded_agent_client=StubEmbeddedClient(),
    )

    result = pipeline.classify(("Birds sing.",))

    assert result.results[0].final_category is SentenceCategory.SIMPLE
    assert result.results[0].agent_path == (AgentName.CLASSIFIER,)
