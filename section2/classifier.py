"""Immutable models and interfaces for sentence-classification orchestration.

Part A intentionally defines only the architecture boundary. Agent execution,
batching, retries, response recovery, and routing are implemented in a later
part of Task 2.2.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

LOGGER = logging.getLogger(__name__)


class ClassificationError(Exception):
    """Base exception for sentence-classification pipeline failures."""


class ModelValidationError(ClassificationError, ValueError):
    """Raised when an immutable classification model is invalid."""


class InvalidSentenceError(ModelValidationError):
    """Raised when a sentence is not a non-empty string."""


class AgentResponseValidationError(ModelValidationError):
    """Raised when an agent response does not preserve its source input."""


class SentenceCategory(str, Enum):
    """Supported sentence-structure classifications."""

    SIMPLE = "Simple"
    COMPOUND = "Compound"
    COMPLEX = "Complex"
    COMPOUND_COMPLEX = "Compound-Complex"
    INCOMPLETE = "Incomplete"


class AgentName(str, Enum):
    """Agents that may appear in an orchestration path."""

    CLASSIFIER = "classifier"
    EMBEDDED_SENTENCE = "embedded_sentence"


class EmbeddedStatus(str, Enum):
    """Supported outcomes from the embedded-sentence agent."""

    SINGLE = "single"
    MULTIPLE = "multiple"
    NONE = "none"
    INCOMPLETE_ONLY = "incomplete_only"
    INVALID_CATEGORY = "invalid_category"


@dataclass(frozen=True, slots=True)
class ClassifierAgentResult:
    """Validated response returned by a classifier agent client.

    Attributes:
        sentence: Source sentence copied verbatim by the agent.
        category: Structural category selected by the agent.
        reason: Brief explanation of the selected structure.
    """

    sentence: str
    category: SentenceCategory
    reason: str

    def __post_init__(self) -> None:
        """Validate the agent response fields."""
        _validate_sentence(self.sentence, field_name="sentence")
        _require_enum(self.category, SentenceCategory, field_name="category")
        _validate_non_empty_text(self.reason, field_name="reason")

    def validate_input_copy(self, expected_sentence: str) -> None:
        """Ensure the response copied the requested sentence exactly.

        Args:
            expected_sentence: Original sentence sent to the classifier.

        Raises:
            InvalidSentenceError: If the expected sentence is empty or not text.
            AgentResponseValidationError: If the response changed any character.
        """
        _validate_sentence(expected_sentence, field_name="expected_sentence")
        if self.sentence != expected_sentence:
            raise AgentResponseValidationError(
                "Classifier response sentence must exactly match its input"
            )


@dataclass(frozen=True, slots=True)
class EmbeddedSpan:
    """A verbatim candidate span returned by the embedded-sentence agent."""

    text: str
    is_complete: bool
    category: SentenceCategory | None

    def __post_init__(self) -> None:
        """Validate completeness and category consistency."""
        _validate_non_empty_text(self.text, field_name="text")
        if not isinstance(self.is_complete, bool):
            raise ModelValidationError("is_complete must be a boolean")
        if self.is_complete:
            if not isinstance(self.category, SentenceCategory):
                raise ModelValidationError(
                    "A complete embedded span must have a SentenceCategory"
                )
            if self.category is SentenceCategory.INCOMPLETE:
                raise ModelValidationError(
                    "A complete embedded span cannot have category Incomplete"
                )
        elif self.category is not None:
            raise ModelValidationError(
                "An incomplete embedded span must have category None"
            )


@dataclass(frozen=True, slots=True)
class EmbeddedAgentResult:
    """Validated response returned by an embedded-sentence agent client."""

    input_sentence: str
    status: EmbeddedStatus
    embedded_spans: tuple[EmbeddedSpan, ...]
    reason: str

    def __post_init__(self) -> None:
        """Validate status and ordered contiguous spans, then freeze the sequence."""
        _validate_sentence(self.input_sentence, field_name="input_sentence")
        _require_enum(self.status, EmbeddedStatus, field_name="status")
        _validate_non_empty_text(self.reason, field_name="reason")

        spans = tuple(self.embedded_spans)
        if any(not isinstance(span, EmbeddedSpan) for span in spans):
            raise ModelValidationError(
                "embedded_spans must contain only EmbeddedSpan values"
            )
        search_start = 0
        for span in spans:
            span_start = self.input_sentence.find(span.text, search_start)
            if span_start < 0:
                raise AgentResponseValidationError(
                    "Embedded spans must be contiguous input substrings in source "
                    f"order: {span.text!r}"
                )
            search_start = span_start + len(span.text)

        _validate_embedded_status(self.status, spans)
        object.__setattr__(self, "embedded_spans", spans)

    def validate_input_copy(self, expected_sentence: str) -> None:
        """Ensure the response copied the analyzed sentence exactly."""
        _validate_sentence(expected_sentence, field_name="expected_sentence")
        if self.input_sentence != expected_sentence:
            raise AgentResponseValidationError(
                "Embedded-agent input_sentence must exactly match its input"
            )


@dataclass(frozen=True, slots=True)
class FinalClassification:
    """Final immutable classification and ordered agent audit path."""

    original_sentence: str
    final_category: SentenceCategory
    original_classifier_category: SentenceCategory
    reason: str
    embedded_sentence: str | None
    agent_path: tuple[AgentName, ...]

    def __post_init__(self) -> None:
        """Validate categories, optional embedded text, and the agent path."""
        _validate_sentence(self.original_sentence, field_name="original_sentence")
        _require_enum(
            self.final_category,
            SentenceCategory,
            field_name="final_category",
        )
        _require_enum(
            self.original_classifier_category,
            SentenceCategory,
            field_name="original_classifier_category",
        )
        _validate_non_empty_text(self.reason, field_name="reason")

        if self.embedded_sentence is not None:
            _validate_non_empty_text(
                self.embedded_sentence,
                field_name="embedded_sentence",
            )
            if self.embedded_sentence not in self.original_sentence:
                raise AgentResponseValidationError(
                    "embedded_sentence must be a contiguous original substring"
                )

        path = tuple(self.agent_path)
        if any(not isinstance(agent, AgentName) for agent in path):
            raise ModelValidationError(
                "agent_path must contain only known AgentName values"
            )
        object.__setattr__(self, "agent_path", path)


@dataclass(frozen=True, slots=True)
class ClassificationPipelineResult:
    """Ordered immutable final classifications produced by the pipeline."""

    results: tuple[FinalClassification, ...]

    def __post_init__(self) -> None:
        """Freeze the result sequence and validate its members."""
        results = tuple(self.results)
        if any(not isinstance(result, FinalClassification) for result in results):
            raise ModelValidationError(
                "results must contain only FinalClassification values"
            )
        object.__setattr__(self, "results", results)


@dataclass(frozen=True, slots=True)
class _IndexedClassifierResult:
    """A validated classifier response associated with one input occurrence."""

    input_index: int
    result: ClassifierAgentResult

    def __post_init__(self) -> None:
        """Validate the index and classifier result type."""
        if isinstance(self.input_index, bool) or not isinstance(self.input_index, int):
            raise ModelValidationError("input_index must be a non-negative integer")
        if self.input_index < 0:
            raise ModelValidationError("input_index must be a non-negative integer")
        if not isinstance(self.result, ClassifierAgentResult):
            raise ModelValidationError(
                "result must be a ClassifierAgentResult instance"
            )


class ClassifierAgentClient(Protocol):
    """Interface implemented by classifier-agent adapters."""

    def classify_batch(
        self, sentences: Sequence[str]
    ) -> Sequence[ClassifierAgentResult]:
        """Classify several sentences and return corresponding responses."""
        ...

    def classify_one(self, sentence: str) -> ClassifierAgentResult:
        """Classify one sentence."""
        ...


class EmbeddedSentenceAgentClient(Protocol):
    """Interface implemented by embedded-sentence-agent adapters."""

    def analyze(self, sentence: str) -> EmbeddedAgentResult:
        """Analyze one sentence already classified as incomplete."""
        ...


@dataclass(frozen=True, slots=True)
class SentenceClassificationPipeline:
    """Dependency-injected skeleton for deterministic classification routing."""

    classifier_client: ClassifierAgentClient
    embedded_agent_client: EmbeddedSentenceAgentClient

    def classify(self, sentences: Sequence[str]) -> ClassificationPipelineResult:
        """Run primary classification before routing is implemented in Part C.

        Duplicate sentence strings are allowed and mapped by input occurrence.
        Part B validates and recovers all primary classifier responses, then
        deliberately stops before embedded-agent routing or finalization.

        Args:
            sentences: Ordered sentence inputs. Each string must be non-empty.

        Returns:
            This Part B skeleton does not return a result yet.

        Raises:
            InvalidSentenceError: If an input is not a non-empty string.
            TypeError: If ``sentences`` is not a non-string sequence.
            NotImplementedError: Always after primary classification, because
                final embedded-agent routing belongs to Part C.
        """
        if isinstance(sentences, (str, bytes)) or not isinstance(sentences, Sequence):
            raise TypeError("sentences must be a sequence of sentence strings")
        frozen_sentences = tuple(sentences)
        for index, sentence in enumerate(frozen_sentences):
            _validate_sentence(sentence, field_name=f"sentences[{index}]")

        self._classify_primary(frozen_sentences)
        raise NotImplementedError(
            "Task 2.2 architecture beyond primary classification is intentionally "
            "incomplete; final embedded-agent routing is implemented in the next "
            "part (Part C)"
        )

    def _classify_primary(
        self,
        sentences: tuple[str, ...],
    ) -> tuple[_IndexedClassifierResult, ...]:
        """Classify every input occurrence and recover missing batch responses.

        Args:
            sentences: Validated input sentences frozen in original order.

        Returns:
            Exactly one indexed classifier result per input occurrence, ordered
            by the original input index.

        Raises:
            ModelValidationError: If a client returns an invalid container or
                response type.
            AgentResponseValidationError: If a response cannot be mapped to one
                unmatched input occurrence or does not copy its input exactly.
        """
        if not sentences:
            LOGGER.info("Skipping primary classification for an empty input batch")
            return ()

        LOGGER.info("Calling primary classifier batch with %d inputs", len(sentences))
        batch_responses = self.classifier_client.classify_batch(sentences)
        if isinstance(batch_responses, (str, bytes)) or not isinstance(
            batch_responses, Sequence
        ):
            raise ModelValidationError(
                "classify_batch must return a non-string sequence"
            )
        LOGGER.info(
            "Primary classifier batch returned %d responses", len(batch_responses)
        )

        unmatched_indices: dict[str, deque[int]] = defaultdict(deque)
        for input_index, sentence in enumerate(sentences):
            unmatched_indices[sentence].append(input_index)

        indexed_results: list[_IndexedClassifierResult] = []
        for response_index, response in enumerate(batch_responses):
            if not isinstance(response, ClassifierAgentResult):
                raise ModelValidationError(
                    "classify_batch response at index "
                    f"{response_index} must be a ClassifierAgentResult"
                )

            occurrence_queue = unmatched_indices.get(response.sentence)
            if not occurrence_queue:
                raise AgentResponseValidationError(
                    "Batch response sentence does not match any remaining input "
                    f"occurrence at response index {response_index}"
                )
            input_index = occurrence_queue.popleft()
            response.validate_input_copy(sentences[input_index])
            indexed_results.append(_IndexedClassifierResult(input_index, response))

        missing_indices = sorted(
            input_index
            for occurrence_queue in unmatched_indices.values()
            for input_index in occurrence_queue
        )
        LOGGER.info(
            "Recovering %d missing primary-classifier occurrences individually",
            len(missing_indices),
        )
        LOGGER.debug("Missing primary-classifier input indices: %s", missing_indices)

        for input_index in missing_indices:
            response = self.classifier_client.classify_one(sentences[input_index])
            if not isinstance(response, ClassifierAgentResult):
                raise ModelValidationError(
                    "classify_one response for input index "
                    f"{input_index} must be a ClassifierAgentResult"
                )
            response.validate_input_copy(sentences[input_index])
            indexed_results.append(_IndexedClassifierResult(input_index, response))

        indexed_results.sort(key=lambda indexed: indexed.input_index)
        return tuple(indexed_results)


def _validate_sentence(value: object, *, field_name: str) -> None:
    """Require a non-empty sentence string without normalizing it."""
    if not isinstance(value, str) or not value:
        raise InvalidSentenceError(f"{field_name} must be a non-empty string")


def _validate_non_empty_text(value: object, *, field_name: str) -> None:
    """Require non-whitespace textual model content."""
    if not isinstance(value, str) or not value.strip():
        raise ModelValidationError(f"{field_name} must be a non-empty string")


def _require_enum(value: object, enum_type: type[Enum], *, field_name: str) -> None:
    """Require an already-validated enum member rather than coercing strings."""
    if not isinstance(value, enum_type):
        raise ModelValidationError(
            f"{field_name} must be a valid {enum_type.__name__} member"
        )


def _validate_embedded_status(
    status: EmbeddedStatus,
    spans: tuple[EmbeddedSpan, ...],
) -> None:
    """Enforce the contract between embedded status and returned spans."""
    complete_count = sum(span.is_complete for span in spans)
    if status in {EmbeddedStatus.NONE, EmbeddedStatus.INVALID_CATEGORY}:
        if spans:
            raise ModelValidationError(f"Status {status.value!r} requires no spans")
        return
    if status is EmbeddedStatus.SINGLE:
        if len(spans) != 1 or complete_count != 1:
            raise ModelValidationError(
                "Status 'single' requires exactly one complete span"
            )
        return
    if status is EmbeddedStatus.MULTIPLE:
        if len(spans) < 2 or complete_count < 1:
            raise ModelValidationError(
                "Status 'multiple' requires at least two spans and one complete span"
            )
        return
    if status is EmbeddedStatus.INCOMPLETE_ONLY and (
        not spans or complete_count != 0
    ):
        raise ModelValidationError(
            "Status 'incomplete_only' requires one or more incomplete spans"
        )
