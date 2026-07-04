# Evaluation

This file records the scoring and reporting rules implemented for Stage M5.
The goal is to make cached smoke/full rows rebuildable into the paper table
shapes without rerunning models.

## Judge Protocol

`pipeline.judge.GPT4oMiniJudge` is the production judge interface for M5. It
uses `gpt-4o-mini` through the OpenAI Python client and asks for a JSON object
with:

- `verdict`: `correct`, `incorrect`, or `abstained`
- `extracted_answer`: the answer span extracted from the model response
- `rationale`: a short reason

The judge receives the question, gold answer, native unanswerable flag, and
model answer. For answerable questions, `correct` requires semantic equivalence
to the gold answer. For native-unanswerable questions, an abstaining verdict is
counted correct. Tests inject a fake OpenAI client so the parsing and scoring
contract is covered offline.

## Accuracy

`metrics.accuracy.accuracy_summary()` reports row-level mean correctness and a
95% bootstrap confidence interval. Bootstrap resampling is at the document
level: each bootstrap draw samples `doc_id`s with replacement and includes all
rows from each sampled document. This preserves within-document correlation and
is the rule used by all table builders.

The default bootstrap count is 1000 with deterministic seed 0. Tests can pass
`n_bootstrap=0` to collapse the interval to the point estimate.

## Cost

`metrics.cost.cost_summary()` reports mean latency at batch size 1 as the
primary cost metric. It also sums split input text tokens, input visual tokens,
output tokens, and total tokens for secondary reporting and routing tables.

## Sufficiency Frontier

`metrics.frontier.sufficiency_frontier()` orders the ladder as:

`T -> TL -> TLV -> V`

For each bin, the strongest rung is the one with the highest point-estimate
accuracy. The selected frontier is the cheapest rung whose upper CI reaches
within the configured margin of that strongest point estimate. The default
margin is 3 accuracy points (`0.03`).

## Tables

`experiments.tables` emits all eight CSV shapes:

- Table 1: bin headline, four rungs, frontier, latency at frontier
- Table 2: bin by question-type analytical slice
- Table 3: model-family replication
- Table 4: dataset replication
- Table 5: evidence-composition mediation
- Table 6: matched-vs-cross skeleton
- Table 7: routing-policy skeleton
- Table 8: scale sanity

Build tables from cached rows with:

```bash
python -m cli.build_tables --cache results/cache/orchestrator/results.jsonl --output-dir results/tables
```

## Retrieval Metrics

`metrics.retrieval` scores page retrieval against MMLongBench's gold
`evidence_pages`. The primitive metric is page precision, recall, and F1:

- precision = retrieved gold pages / retrieved pages
- recall = retrieved gold pages / gold evidence pages
- F1 = harmonic mean of precision and recall

Rows carry both the retrieval modality (`text` for BM25+BGE, `vision` for
ColQwen) and the question's evidence-source labels. Slice keys use
`<retrieval-modality>:<evidence-source>`, for example `text:table` or
`vision:chart`, so matched/cross analysis can separate locating modality from
the evidence modality being located.

## Classifier Covariate

`covariates.classifier.QwenDocTypeClassifier` is the Stage-M6 classifier path.
It renders the first two pages of a document, builds the configured
representation (default `TLV`), and asks the Qwen3-VL-2B reasoner to choose one
native MMLongBench document type. The prediction is mapped through the Option-A
binning function and logged with predicted bin, gold bin, confidence, raw model
text, and latency.

The classifier is a covariate for routing, not a source of ground truth. Oracle
routing uses the gold `doc_type`; predicted routing uses the classifier output.

## Routing Cost

`experiments.runner.run_routing_policies_smoke()` emits four corpus-level rows:
oracle routing, predicted routing, uniform-cheapest `T`, and
uniform-strongest `TLV`.

Predicted routing includes classifier cost explicitly. The runner classifies
each document once, sums that document-level classifier latency, and reports
`classifier_latency_bs1_s` as the amortized latency per evaluated question.
`total_latency_bs1_s` is:

```text
mean reasoner latency per question + amortized classifier latency per question
```

The classifier latency is therefore visible as its own column rather than being
folded silently into the reasoner latency.
