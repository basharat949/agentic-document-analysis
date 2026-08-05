"""Tests for Task 2.2 Part D rate-limit retry and exponential backoff."""

from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import Mock

import pytest

from section2.classifier import (
    ClassifierAgentResult,
    EmbeddedAgentResult,
    EmbeddedStatus,
    ModelValidationError,
    RateLimitError,
    RetryPolicy,
    SentenceCategory,
    SentenceClassificationPipeline,
)


def _classifier_result(
    sentence: str,
    category: SentenceCategory = SentenceCategory.SIMPLE,
) -> ClassifierAgentResult:
    """Build a valid deterministic classifier response."""
    return ClassifierAgentResult(sentence, category, "Classifier reason.")


class RetryClassifierClient:
    """Classifier test double whose methods expose configurable mocks."""

    def __init__(self, batch_effects: Sequence[object]) -> None:
        self.classify_batch = Mock(side_effect=list(batch_effects))
        self.classify_one = Mock()


class RetryEmbeddedClient:
    """Embedded-agent test double whose analyze call is configurable."""

    def __init__(self, effects: Sequence[object] = ()) -> None:
        self.analyze = Mock(side_effect=list(effects))


def _pipeline(
    classifier: RetryClassifierClient,
    *,
    embedded: RetryEmbeddedClient | None = None,
    policy: RetryPolicy | None = None,
    sleep: Mock | None = None,
) -> tuple[SentenceClassificationPipeline, RetryEmbeddedClient, Mock]:
    """Build a retry-configured pipeline and return recording dependencies."""
    embedded_client = embedded or RetryEmbeddedClient()
    sleep_mock = sleep or Mock()
    pipeline = SentenceClassificationPipeline(
        classifier_client=classifier,
        embedded_agent_client=embedded_client,
        retry_policy=policy or RetryPolicy(),
        sleep=sleep_mock,
    )
    return pipeline, embedded_client, sleep_mock


def test_batch_succeeds_first_attempt_without_sleep() -> None:
    """A successful initial batch call does not invoke backoff sleep."""
    classifier = RetryClassifierClient([(_classifier_result("Input."),)])
    pipeline, _, sleep = _pipeline(classifier)

    result = pipeline.classify(("Input.",))

    assert result.results[0].final_category is SentenceCategory.SIMPLE
    assert classifier.classify_batch.call_count == 1
    sleep.assert_not_called()


def test_batch_retries_once_after_rate_limit() -> None:
    """One HTTP 429 marker produces one initial-delay sleep and a retry."""
    classifier = RetryClassifierClient(
        [RateLimitError("429"), (_classifier_result("Input."),)]
    )
    pipeline, _, sleep = _pipeline(classifier)

    pipeline.classify(("Input.",))

    assert classifier.classify_batch.call_count == 2
    sleep.assert_called_once_with(0.25)


def test_exponential_delays_before_success() -> None:
    """Success on attempt four uses all three exponential delays."""
    classifier = RetryClassifierClient(
        [
            RateLimitError("429-1"),
            RateLimitError("429-2"),
            RateLimitError("429-3"),
            (_classifier_result("Input."),),
        ]
    )
    policy = RetryPolicy(
        max_attempts=4,
        initial_delay_seconds=0.25,
        multiplier=2.0,
        max_delay_seconds=4.0,
    )
    pipeline, _, sleep = _pipeline(classifier, policy=policy)

    pipeline.classify(("Input.",))

    assert [item.args[0] for item in sleep.call_args_list] == [0.25, 0.5, 1.0]


def test_delay_is_capped() -> None:
    """Calculated exponential delays never exceed the configured maximum."""
    classifier = RetryClassifierClient(
        [
            RateLimitError("429-1"),
            RateLimitError("429-2"),
            RateLimitError("429-3"),
            (_classifier_result("Input."),),
        ]
    )
    policy = RetryPolicy(
        max_attempts=4,
        initial_delay_seconds=0.75,
        multiplier=2.0,
        max_delay_seconds=1.0,
    )
    pipeline, _, sleep = _pipeline(classifier, policy=policy)

    pipeline.classify(("Input.",))

    assert [item.args[0] for item in sleep.call_args_list] == [0.75, 1.0, 1.0]


def test_retry_exhaustion_reraises_without_final_sleep() -> None:
    """The final rate-limit error propagates after exactly max_attempts calls."""
    errors = [RateLimitError("first"), RateLimitError("second"), RateLimitError("final")]
    classifier = RetryClassifierClient(errors)
    pipeline, _, sleep = _pipeline(classifier)

    with pytest.raises(RateLimitError, match="final"):
        pipeline.classify(("Input.",))

    assert classifier.classify_batch.call_count == 3
    assert [item.args[0] for item in sleep.call_args_list] == [0.25, 0.5]


def test_arbitrary_exception_is_not_retried() -> None:
    """Unexpected exceptions propagate immediately without sleeping."""
    classifier = RetryClassifierClient([RuntimeError("boom")])
    pipeline, _, sleep = _pipeline(classifier)

    with pytest.raises(RuntimeError, match="boom"):
        pipeline.classify(("Input.",))

    assert classifier.classify_batch.call_count == 1
    sleep.assert_not_called()


def test_validation_exception_is_not_retried() -> None:
    """Model validation failures are deterministic and fail immediately."""
    classifier = RetryClassifierClient([ModelValidationError("bad model")])
    pipeline, _, sleep = _pipeline(classifier)

    with pytest.raises(ModelValidationError, match="bad model"):
        pipeline.classify(("Input.",))

    assert classifier.classify_batch.call_count == 1
    sleep.assert_not_called()


def test_missing_response_recovery_retries_classify_one() -> None:
    """A missing batch occurrence receives retry around its individual call."""
    classifier = RetryClassifierClient([()])
    classifier.classify_one.side_effect = [
        RateLimitError("429"),
        _classifier_result("Missing."),
    ]
    pipeline, _, sleep = _pipeline(classifier)

    result = pipeline.classify(("Missing.",))

    assert result.results[0].original_sentence == "Missing."
    assert classifier.classify_one.call_count == 2
    sleep.assert_called_once_with(0.25)


def test_embedded_analysis_is_retried() -> None:
    """An incomplete primary result retries embedded analysis after HTTP 429."""
    sentence = "After the bus"
    classifier = RetryClassifierClient(
        [(_classifier_result(sentence, SentenceCategory.INCOMPLETE),)]
    )
    embedded_result = EmbeddedAgentResult(
        input_sentence=sentence,
        status=EmbeddedStatus.NONE,
        embedded_spans=(),
        reason="No complete embedded sentence.",
    )
    embedded = RetryEmbeddedClient([RateLimitError("429"), embedded_result])
    pipeline, _, sleep = _pipeline(classifier, embedded=embedded)

    result = pipeline.classify((sentence,))

    assert result.results[0].final_category is SentenceCategory.INCOMPLETE
    assert embedded.analyze.call_count == 2
    sleep.assert_called_once_with(0.25)


def test_retry_state_is_independent_across_pipeline_calls() -> None:
    """Each classify call starts again from the configured initial delay."""
    result = (_classifier_result("Input."),)
    classifier = RetryClassifierClient(
        [RateLimitError("first"), result, RateLimitError("second"), result]
    )
    pipeline, _, sleep = _pipeline(classifier)

    pipeline.classify(("Input.",))
    pipeline.classify(("Input.",))

    assert [item.args[0] for item in sleep.call_args_list] == [0.25, 0.25]


@pytest.mark.parametrize("max_attempts", [0, -1, 1.5, True])
def test_retry_policy_rejects_invalid_attempts(max_attempts: object) -> None:
    """Attempt counts must be positive non-boolean integers."""
    with pytest.raises(ModelValidationError, match="max_attempts"):
        RetryPolicy(max_attempts=max_attempts)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("initial_delay_seconds", -0.1),
        ("initial_delay_seconds", float("nan")),
        ("initial_delay_seconds", float("inf")),
        ("initial_delay_seconds", True),
        ("multiplier", 0.99),
        ("multiplier", float("nan")),
        ("multiplier", True),
        ("max_delay_seconds", -0.1),
        ("max_delay_seconds", float("inf")),
        ("max_delay_seconds", False),
    ],
)
def test_retry_policy_rejects_invalid_numeric_values(
    field: str,
    value: object,
) -> None:
    """Delay and multiplier values must be finite, bounded, and non-boolean."""
    arguments = {field: value}
    with pytest.raises(ModelValidationError, match=field):
        RetryPolicy(**arguments)  # type: ignore[arg-type]
