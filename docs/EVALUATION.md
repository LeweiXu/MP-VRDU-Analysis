# Evaluation

This file records the scoring and reporting rules implemented for Stage M5 and
extended in Section F4-F6. The goal is to make cached smoke/full rows
rebuildable into the paper table shapes without rerunning models.

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

## Judge-Human Agreement Gate

Section F2 uses `cli.gates agreement-sample` to create a 200-row sheet
stratified over `doc_type` x question type. The current MMLongBench loader uses
an explicit `question_type` field when present and otherwise falls back to the
derived evidence-hop label (`none`, `single`, `multi`).

Completed sheets are scored with `cli.gates agreement-score`, which computes
Cohen's kappa over `correct`, `incorrect`, and `abstained` labels. The
pre-registered gate is kappa >= 0.75. Until that JSON artifact is recorded, the
main numbers remain untrusted.

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

The Section-F1 frontier-divergence gate is implemented by
`cli.gates frontier`. It reads `results/tables/full/table1_headline.csv` and
returns Go only when at least two configured Option-A bins have different
frontier rungs.

## Tables

`experiments.tables` emits all eight CSV shapes:

- Table 1: bin headline, four rungs, frontier, latency at frontier
- Table 2: bin by question-type analytical slice
- Table 3: model-family replication
- Table 4: dataset replication
- Table 5: evidence-composition mediation by evidence-source mix
- Table 6: matched vision retrieval vs cross text-to-vision retrieval
- Table 7: corpus-level routing policies with classifier cost
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

Table 6 is restricted to bins whose Table-1 oracle frontier requires a visual
rung (`TLV` or `V`). It compares `matched_vision` (vision retrieval plus visual
reasoning) with `cross_text_to_vision` (text retrieval plus visual reasoning)
and reports accuracy, latency deltas relative to the matched row, and retrieval
precision/recall/F1.

## Composition Mediation

Table 5 decomposes each MMLongBench Option-A bin by normalized evidence-source
modality (`text`, `table`, `chart`, `figure`, `layout`). Per-question modality
shares are split evenly across all labeled sources for that question, so shares
sum to one within each bin. The predicted bin frontier is the strongest
frontier among modalities with at least 10% share, with the global modality
frontiers computed from all rows carrying that source label.

## Classifier Covariate

`covariates.classifier.QwenDocTypeClassifier` is the Stage-M6 classifier path.
It renders the first two pages of a document, builds the configured
representation (default `TLV`), and asks the Qwen3-VL-2B reasoner to choose one
native MMLongBench document type. The prediction is mapped through the Option-A
binning function and logged with predicted bin, gold bin, confidence, raw model
text, and latency.

The classifier is a covariate for routing, not a source of ground truth. Oracle
routing uses the gold `doc_type`; predicted routing uses the classifier output.

Section F3 uses `cli.gates classifier-pilot --full` to sample 100 distinct
documents and run this classifier once per document. The feasibility gate is
top-1 Option-A bin accuracy >= 0.70. The same CLI can score an existing
`classifier.jsonl` side artifact via `classifier-score`.

## Routing Cost

The `T7_routing` experiment (`experiments/T7_routing.py`) builds Table 7 with
four policies: oracle routing, predicted routing, uniform-cheapest `T`, and
uniform-strongest `TLV`. Each policy emits one corpus-level row. The policy rows
reuse T1's oracle-ladder rows; the doc-type classifier is the only new GPU work
and runs once per document as a side artifact (`classifier.jsonl`).

Predicted routing includes classifier cost explicitly. The classifier runs once
per document; the sum of classifier latency is divided by the number of
evaluated question rows selected by the predicted policy and reported as
`classifier_latency_bs1_s`, so
`total_latency_bs1_s` is:

```text
mean reasoner latency per question + amortized classifier latency per question
```

The classifier latency is therefore visible as its own column rather than being
folded silently into the reasoner latency.
