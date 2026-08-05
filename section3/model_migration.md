# Task 3.2 — Classification Model Deprecation

## Assumptions and migration principles

The current model serves both agents but each agent is configured and evaluated
as an independent deployment unit. The Classifier Agent assigns one of five
sentence categories and controls whether Python routing invokes the Embedded
Sentence Agent. The Embedded Sentence Agent operates only on `Incomplete`
inputs, extracts verbatim spans, and supports deterministic final-category
selection. Model versions, prompts, schemas, and settings are configuration, not
code changes.

All three replacement candidates are unproven. No candidate should be selected
from vendor benchmark scores alone: those scores do not represent this system's
OCR noise, non-standard language, strict JSON contracts, or routing-dependent
failure costs. Selection requires like-for-like measurement on representative
data and production traffic observed safely in shadow mode.

## 1. Migration objectives and success criteria

The objective is to replace the deprecated model within 90 days without losing
source fidelity, structural classification quality, routing reliability, or
operational predictability.

### Classifier Agent

Success requires:

- no material regression in overall or per-category classification accuracy;
- `Incomplete` precision of at least 90% and recall of at least 90%, preferably
  95% because a false negative prevents specialist routing;
- a stable, explainable `Incomplete` routing rate on comparable traffic;
- exact compliance with the required JSON fields, types, and enum values;
- exact, character-for-character preservation of the input sentence;
- acceptable p50, p95, and p99 latency and timeout rate;
- cost per classified sentence within the approved budget;
- acceptable HTTP 429 frequency, retry count, and retry-exhaustion rate; and
- no regression in missing batch-response rate or individual recovery success.

### Embedded Sentence Agent

Success requires:

- high embedded-span detection and recovery accuracy;
- correct `single`, `multiple`, `none`, and `incomplete_only` decisions;
- accurate classification of complete spans and a low false-promotion rate;
- exact JSON/schema compliance, including nullable categories for incomplete
  spans;
- verbatim, contiguous span extraction in source order;
- correct behavior when no recoverable span exists;
- acceptable latency, cost, HTTP 429 behavior, and retry exhaustion; and
- no recursive agent invocation or unsupported routing decisions.

### End-to-end pipeline

Success requires:

- no material degradation in final-category accuracy;
- correct Python-enforced routing and accurate immutable `agent_path` values;
- stable routing volume and embedded recovery rate;
- every input occurrence producing exactly one result in original order,
  including duplicate sentences and individually recovered batch omissions;
- zero silent spelling, whitespace, capitalization, or punctuation mutation;
- end-to-end p95 latency within the agreed service-level objective (SLO);
- total cost per document within budget; and
- no unacceptable increase in schema failures, rate limits, missing responses,
  failed jobs, or manual-review volume.

Baseline and candidate metrics must use identical definitions, datasets, and
confidence intervals. Aggregate accuracy alone is insufficient because it can
hide a routing-critical `Incomplete` regression.

## 2. Step-by-step 90-day plan

| Period | Phase | Activities and exit criteria |
|---|---|---|
| Days 1–10 | Freeze and baseline | Version and freeze current prompts, schemas, model identifier, temperature, decoding settings, retry policy, and evaluation code. Capture current production classification distribution, per-category quality, routing rate, embedded recovery, schema failures, latency, cost, 429s, retries, and missing responses. Build a representative held-out set with misspellings, OCR noise, fragments, run-ons, complex clauses, coordinated predicates/clauses, non-standard agreement, and embedded-sentence cases. Remove duplicates and data leakage between tuning and held-out partitions. Exit when the baseline is reproducible and metric ownership is agreed. |
| Days 11–25 | Offline comparison | Run all three candidates on the same frozen inputs with identical prompts, temperature, schemas, retry policy, and test conditions. Record per-agent and end-to-end accuracy, `Incomplete` precision/recall, routing differences, embedded recovery, verbatim preservation, JSON failures, missing responses, latency, 429 behavior, token usage, and cost. Repeat enough runs to reveal non-deterministic contract failures. Exit with a comparable scorecard, not a vendor-score ranking. |
| Days 26–35 | Failure analysis and adaptation | Review disagreements by category and failure type. Determine whether failures arise from model capability, prompt interpretation, schema handling, or decoding configuration. If a candidate requires prompt/schema adaptation, version the change and re-run the entire frozen suite; do not compare an adapted candidate against an unretested baseline. Reject any candidate that fails mandatory routing, fidelity, or schema gates, regardless of price. Exit with one preferred candidate and one fallback. |
| Days 36–50 | Shadow deployment | Send sampled production requests to both the old and preferred candidate models. Only the old model's result reaches users or controls routing. Store access-controlled comparison records with model/prompt/schema versions. Analyze output disagreement, category distribution, `Incomplete` routing differences, schema validity, source fidelity, latency, 429s, and cost. Exit only after sufficient representative volume and all shadow gates pass. |
| Days 51–65 | Classifier Agent canary | Move only the Classifier Agent to the candidate for an initial 5% of eligible traffic. Keep the Embedded Sentence Agent on the old model. Increase gradually (for example, 5% → 15% → 30% → 50% → 100%) only after a defined observation window and passing gates. Monitor `Incomplete` recall and routing rate most closely. Pause or roll back automatically on a trigger. |
| Days 66–75 | Embedded Agent migration | With the classifier stable, shadow the candidate for the Embedded Sentence Agent, then canary it separately. Compare span detection, verbatim extraction, no-false-promotion rate, category selection inputs, and final outcomes. Increase traffic only after its independent gates pass. |
| Days 76–85 | Scale and operational validation | Increase candidate traffic toward 100% while retaining immediate per-agent rollback. Validate database model-version attribution, structured logs, dashboards, alerts, retry telemetry, and on-call procedures. Run concurrency, timeout, and sustained-load checks. Confirm budgets and SLOs using observed traffic. |
| Days 86–90 | Full cutover and sign-off | Complete cutover, monitor continuously through the highest-risk window, and obtain engineering/product sign-off. Keep the old model configuration available during the rollback window. Retire it only after the rollback period closes, all gates remain healthy, and deprecation timing requires retirement. Preserve evaluation and shadow evidence for audit. |

The timeline should compress only if evidence is available earlier; it should not
skip a phase. If a mandatory gate fails near day 90, use the pre-approved
fallback candidate or formally escalate the deprecation risk rather than forcing
an unsafe cutover.

## 3. Candidate-selection decision framework

### Mandatory quality gates

Weighted scoring begins only after a candidate passes schema, fidelity,
`Incomplete` precision/recall, false-promotion, latency, and reliability gates.
A cheaper or faster candidate cannot win by compensating for a routing-critical
failure with cost or latency points. Candidates should be compared using
category-specific confusion matrices and failure slices, not only overall
accuracy.

### Classifier Agent scorecard

| Criterion | Weight | Measurement |
|---|---:|---|
| Classification quality | 30% | Overall and macro accuracy plus per-category precision/recall |
| `Incomplete` recall | 25% | Recall on held-out `Incomplete` cases and critical slices |
| Schema compliance | 15% | Exact valid responses divided by total responses |
| Verbatim preservation | 10% | Exact sentence-copy success rate |
| Latency | 10% | p50/p95/p99 and timeout rate under comparable load |
| Cost | 10% | Total model cost per sentence and representative document |

### Embedded Sentence Agent scorecard

| Criterion | Weight | Measurement |
|---|---:|---|
| Embedded-sentence recovery | 30% | Correct recoverable spans and correct no-span outcomes |
| Final-category accuracy | 20% | Correct category supplied for the deterministic finalizer |
| No-false-promotion rate | 15% | Incomplete cases correctly retained as `Incomplete` |
| Schema compliance | 15% | Exact fields, enum/null rules, and valid ordered spans |
| Latency | 10% | p50/p95/p99 and timeout rate under comparable load |
| Cost | 10% | Total model cost per routed sentence |

Scores should be normalized against agreed targets, accompanied by raw values
and uncertainty, and reviewed alongside qualitative error samples. The final
decision records why the selected model passed each gate and why alternatives
were rejected.

## 4. Rollout strategy

The rollout order is **offline benchmark → shadow → canary → gradual traffic
increase**. Shadow mode precedes canary because it exercises the real production
input distribution without exposing users to candidate output. It enables
direct, request-level disagreement analysis and reveals OCR patterns, schema
failures, latency tails, and rate-limit behavior that a held-out set may miss.

The two agents should not migrate simultaneously. Moving both would confound a
classifier routing change with an embedded-recovery change: a different final
label could not be attributed confidently to the gatekeeper or specialist.
Separate migration permits per-agent rollback and cleaner metrics.

Recommended order:

1. Migrate the Classifier Agent first while the Embedded Sentence Agent remains
   on the old model.
2. After classifier stability is demonstrated, migrate the Embedded Sentence
   Agent through its own shadow and canary phases.

This approach is slower and temporarily operates a mixed-model system. It also
requires per-agent configuration, dashboards, and version attribution. Those
costs are justified by substantially better fault isolation, safer rollback,
and the ability to distinguish routing regressions from recovery regressions.

## 5. Minimum test suite before production traffic

### A. Contract and schema tests

- Require the exact JSON fields and types for each agent response.
- Accept only valid category/status enum values and reject extra keys.
- Verify exact sentence copying and verbatim contiguous embedded spans.
- Reject malformed JSON, missing fields, incorrect nullability, invalid span
  ordering, and category/completeness inconsistencies.

### B. Classifier quality tests

- Cover all five categories with balanced and naturally imbalanced reporting.
- Include difficult OCR/noisy text, misspellings, punctuation loss, run-ons,
  compound predicates versus compound clauses, subordinate-clause markers, and
  non-standard grammar.
- Report confusion matrices and category-specific precision and recall.

### C. Incomplete-routing tests

- Verify `Incomplete` always routes to the Embedded Sentence Agent.
- Verify non-`Incomplete` categories never invoke that agent.
- Recover missing batch items individually rather than dropping them.
- Preserve duplicate occurrences and original ordering by index.
- Verify `agent_path` contains exactly the agents actually invoked.

### D. Embedded Agent tests

- Cover one embedded sentence, no span, multiple spans, and incomplete nested
  content.
- Verify highest-complexity selection and first-in-source-order tie-breaking.
- Ensure incomplete nested content remains `Incomplete` and cannot be promoted.
- Prohibit recursion and extraction of non-contiguous or normalized text.

An explicit routing-critical regression case is:

| Expected field | Expected value |
|---|---|
| Input | `becaus I go Home and` |
| Classifier output | `Incomplete` |
| Embedded agent invoked | Yes |
| Embedded sentence | `I go Home` copied verbatim |
| Final category | `Simple` |
| Original classifier category | `Incomplete` |
| `agent_path` | `["classifier", "embedded_sentence"]` |

### E. Reliability tests

- Cover explicit HTTP 429 retry, exponential delays, delay caps, and exhaustion.
- Verify schema, validation, and arbitrary errors are not retried.
- Exercise concurrent requests to confirm retry state and routing remain
  thread-safe.
- Measure latency, timeout behavior, missing responses, and recovery under load.

### F. Regression tests

- Require no statistically significant degradation against the current model on
  the untouched held-out set.
- Compare category-specific and end-to-end routing metrics, not only averages.
- Confirm schema/fidelity gates and cost/latency limits under the same conditions.
- Review disagreement samples for systematic harms hidden by aggregate metrics.

## 6. Production gates and rollback

Example minimum release gates are:

| Gate | Minimum condition |
|---|---|
| Overall quality | No material accuracy regression; confidence interval within agreed non-inferiority margin |
| `Incomplete` recall | At least 90%; preferably 95% where a miss prevents routing |
| `Incomplete` precision | At least 90% |
| Schema compliance | At least 99.9% exact compliance |
| Verbatim preservation | Zero silent sentence mutation in the evaluation set |
| Latency | p95 within the agreed SLO at representative concurrency |
| Cost | Per-sentence and per-document cost within approved budget |
| Reliability | No unacceptable 429, exhaustion, timeout, or missing-response regression |

Rollback triggers include an `Incomplete` routing regression, spike in schema or
verbatim-preservation failures, latency/SLO breach, increased HTTP 429 or 5xx
rate, retry exhaustion, missing responses, or cost beyond the approved limit.
Alert thresholds and observation windows must be agreed before canary traffic.

The model version is controlled independently for each agent through deployment
configuration. Rollback changes only the affected agent to the last known-good
model; it does not require a code release or force the other agent to revert.
Prompts, schemas, decoding settings, and retry policy remain versioned and
reproducible. Shadow outputs, disagreement samples, request/version metadata, and
failure logs are retained under the applicable privacy policy for diagnosis.

## 7. Monitoring after cutover

Dashboards and alerts should segment metrics by agent, model version, prompt
version, language/noise slice, and final category. Monitor:

- classification distribution and category-specific precision/recall;
- `Incomplete` routing rate and false-negative review samples;
- embedded recovery, no-span, multiple-span, and false-promotion rates;
- old/candidate disagreement samples during the retained comparison period;
- JSON/schema validation and verbatim-copy failures;
- p50, p95, and p99 agent and end-to-end latency;
- token usage and cost per sentence/document;
- HTTP 429 and 5xx rates, retry counts, backoff delay, and retry exhaustion;
- missing batch responses and individual recovery success; and
- final job failure rate and manual-review volume.

Compare live values with both the pre-migration baseline and canary expectations.
Use alerting for sudden changes and scheduled human review for gradual drift or
category-specific degradation. The first post-cutover review should occur within
24 hours, followed by frequent reviews during the rollback window and a formal
stability sign-off before the old model is retired.
