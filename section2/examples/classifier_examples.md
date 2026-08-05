# Difficult Classifier Examples

These examples supplement the classifier prompt. Each preserves the supplied
spelling, grammar, capitalization, and punctuation exactly.

## 1. Simple

Input:

```json
{"sentence":"Me and Rafi packs and labels the dented boxs"}
```

Expected output:

```json
{"sentence":"Me and Rafi packs and labels the dented boxs","category":"Simple","reason":"The compound subject and compound predicate belong to one independent clause; no dependent clause is present."}
```

Why difficult: The non-standard agreement, compound subject, and two coordinated
verbs can be mistaken for multiple clauses, but there is only one clause.

## 2. Compound

Input:

```json
{"sentence":"The screen went blak I press restart nothing happen"}
```

Expected output:

```json
{"sentence":"The screen went blak I press restart nothing happen","category":"Compound","reason":"Three independent clauses are present as a run-on with omitted punctuation and conjunctions."}
```

Why difficult: Missing punctuation and tense/agreement errors obscure three
separate subject-predicate units; those errors must not be repaired.

## 3. Complex

Input:

```json
{"sentence":"I left the parcel were you said it shud go"}
```

Expected output:

```json
{"sentence":"I left the parcel were you said it shud go","category":"Complex","reason":"One independent clause contains a dependent location clause despite the misspellings."}
```

Why difficult: `were` is a misspelled relative location marker, and the nested
verb `said` may make the text look compound even though there is one main clause.

## 4. Compound-Complex

Input:

```json
{"sentence":"Even tho the map was rong we kept drivin but Sam called for help"}
```

Expected output:

```json
{"sentence":"Even tho the map was rong we kept drivin but Sam called for help","category":"Compound-Complex","reason":"One dependent concession clause accompanies two coordinated independent clauses."}
```

Why difficult: Informal spelling hides the dependent marker, while two main
clauses must both be recognized without normalizing the sentence.

## 5. Incomplete

Input:

```json
{"sentence":"Which the superviser left beside the mashine"}
```

Expected output:

```json
{"sentence":"Which the superviser left beside the mashine","category":"Incomplete","reason":"The relative clause is dependent and has no independent main clause."}
```

Why difficult: The span contains a subject and finite verb and sounds
propositional, but initial `Which` leaves it dependent on missing outer content.
