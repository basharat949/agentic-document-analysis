# Task 2.2 Part A — Classification Architecture

This section defines immutable domain models, agent-client interfaces, and the
public orchestration boundary. Part A intentionally does not execute either
agent and does not implement batching, retries, missing-response recovery, or
embedded-agent routing.

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
retry behavior and embedded-agent routing are intentionally reserved for later
parts.

## Adding a third specialist

A future specialist can be introduced by adding its name to `AgentName`, defining
a focused result model and `Protocol`, then injecting that client into the
pipeline. A new explicit category/status guard would own the route, and the
specialist would be appended to `agent_path`. Existing routes remain unchanged,
which keeps expansion reviewable and prevents agents from selecting their own
successors.
