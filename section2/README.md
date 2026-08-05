# Task 2.2 Part A — Classification Architecture

This section defines immutable domain models, agent-client interfaces, primary
batch recovery, and deterministic embedded-agent routing. HTTP retry behavior is
intentionally deferred to a later part.

## Why a hand-written orchestrator

The workflow has a small, fixed decision tree, so a hand-written orchestrator is
easier to inspect and test than a graph framework. It avoids an unnecessary
runtime dependency and keeps every transition visible in ordinary Python. There
is no mutable global request state: models are frozen and a pipeline receives its
dependencies when constructed.

## Code-enforced routing

The next part will classify inputs and make routing decisions from validated
`SentenceCategory` values—not from free-form model prose. Normal categories will
finish after `AgentName.CLASSIFIER`; only `SentenceCategory.INCOMPLETE` may route
to `AgentName.EMBEDDED_SENTENCE`. Each final result will retain an ordered,
immutable `agent_path`. Input indices, rather than sentence text, will preserve
ordering and make duplicate sentences unambiguous. Agent responses must copy the
input exactly, and embedded spans must be contiguous source substrings before a
route can produce a final result.

## Injected Protocol clients

`ClassifierAgentClient` and `EmbeddedSentenceAgentClient` describe only the
operations the orchestrator needs. Injection separates orchestration from any
transport or provider, supports deterministic test doubles, and prevents the
domain layer from importing an LLM SDK. A concrete client can later call a local
or remote service without changing the pipeline's routing contract.

## Batch classification and missing-response recovery

The primary classifier receives all sentences in one batch, reducing API
round-trips and per-request overhead. Responses are mapped to unmatched input
occurrences and their original indices rather than through a plain
`dict[sentence]`; this preserves order when identical sentences appear more than
once. If the batch omits an occurrence, that exact input index is classified
individually instead of being silently dropped. Every batch and individual
response must copy its source sentence exactly before it is accepted. HTTP 429
retry behavior is intentionally reserved for a later part.

## Embedded-agent routing and finalization

Routing is enforced in Python from the validated
`SentenceCategory.INCOMPLETE` value; model output cannot choose its successor.
Other categories finish directly after the classifier. An incomplete result is
analyzed once by the embedded agent, whose input copy, status, and spans are
validated before finalization. For multiple complete spans, selection uses
`Compound-Complex > Complex > Compound > Simple`, with the first source-order
span winning a tie. The immutable `agent_path` records only `classifier` for
direct results and both agents for incomplete routes. An `invalid_category`
response is an orchestration contract violation because only incomplete inputs
can reach this agent, so it raises instead of becoming a final classification.

## Rate-limit retry and backoff

Retries are limited to the explicit `RateLimitError`, which agent adapters use
to represent HTTP 429 responses. `max_attempts` includes the first request, and
each subsequent delay grows exponentially from the configured initial delay
until reaching the configured cap. The sleep callable is injected into the
immutable pipeline, allowing tests to record delays without waiting. Model
validation errors, response-contract violations, and arbitrary exceptions fail
immediately: retrying them would hide deterministic defects rather than relieve
temporary rate limiting.

## Adding a third specialist

A future specialist can be introduced by adding its name to `AgentName`, defining
a focused result model and `Protocol`, then injecting that client into the
pipeline. A new explicit category/status guard would own the route, and the
specialist would be appended to `agent_path`. Existing routes remain unchanged,
which keeps expansion reviewable and prevents agents from selecting their own
successors.
