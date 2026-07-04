# Runbook

Operational commands for Section 2. All commands are root-relative and assume the
repo environment is active through `envs/mpvrdu/bin/python`.

## F1 Frontier Divergence

Generate Table 1 on Kaya, pull results, then judge/build locally:

```bash
envs/mpvrdu/bin/python -m kaya.kaya submit kaya/generate.py -- --experiment T1_headline --full
envs/mpvrdu/bin/python -m kaya.kaya pull
envs/mpvrdu/bin/python -m cli.experiments --phase judge --experiment T1_headline --full --judge gpt-4o-mini
```

Evaluate the gate:

```bash
envs/mpvrdu/bin/python -m cli.gates frontier \
  --table results/tables/full/table1_headline.csv \
  --json-output results/gates/F1_frontier_divergence.json
```

Go if at least two of `text_heavy`, `in_between`, and `visual_heavy` have
different `frontier` values. Record the result in `docs/DECISIONS.md` after the
human checkpoint.

## F2 Judge-Human Agreement

Create the 200-row human labelling sheet from judged T1 rows. The default
sampling frame is the full corpus, oracle condition, `TLV` representation:

```bash
envs/mpvrdu/bin/python -m cli.gates agreement-sample --full \
  --results results/cache/full/T1_headline/results.jsonl \
  --output results/gates/agreement_sample.csv
```

Fill `human_label` with one of `correct`, `incorrect`, or `abstained`, then score:

```bash
envs/mpvrdu/bin/python -m cli.gates agreement-score \
  --sheet results/gates/agreement_sample.csv \
  --json-output results/gates/F2_judge_human_agreement.json
```

Go if Cohen's kappa is at least `0.75`. If not, revise the judge prompt or adopt
a stricter judge before trusting main-run numbers.

## F3 Classifier Feasibility

Run the 100-document classifier pilot on the full corpus:

```bash
envs/mpvrdu/bin/python -m cli.gates classifier-pilot --full \
  --output results/gates/classifier_pilot.csv \
  --json-output results/gates/F3_classifier_feasibility.json
```

For dry-run review of the exact documents without loading the classifier:

```bash
envs/mpvrdu/bin/python -m cli.gates classifier-pilot --full --sample-only \
  --output results/gates/classifier_pilot_docs.csv
```

Go if top-1 Option-A bin accuracy is at least `0.70`. If the pilot fails, either
upgrade the classifier or scope RQ3 to oracle-routing upper bound only.

## F4 Exp 1 Replications

F4 is three table paths after the F1-F3 gates are recorded.

Table 2 is aggregation-only from judged T1 rows:

```bash
envs/mpvrdu/bin/python -m cli.experiments --phase judge --experiment T2_analytical --full --judge gpt-4o-mini
```

Table 3 adds the InternVL3-8B family replication on Kaya, then judges/builds
locally:

```bash
envs/mpvrdu/bin/python -m kaya.kaya submit kaya/generate.py -- --experiment T3_family --full
envs/mpvrdu/bin/python -m kaya.kaya pull
envs/mpvrdu/bin/python -m cli.experiments --phase judge --experiment T3_family --full --judge gpt-4o-mini
```

Table 4 runs the headline ladder on LongDocURL:

```bash
envs/mpvrdu/bin/python -m kaya.kaya submit kaya/generate.py -- --experiment T4_dataset --full
envs/mpvrdu/bin/python -m kaya.kaya pull
envs/mpvrdu/bin/python -m cli.experiments --phase judge --experiment T4_dataset --full --judge gpt-4o-mini
```

Before Table 4, stage `LongDocURL_public.jsonl` under `.data/longdocurl/` or
ensure the `dengchao/LongDocURL` Hugging Face snapshot is in `.cache`, and stage
PDFs under `.data/longdocurl/documents/<doc_no>.pdf`. The public annotation
snapshot includes source paths but not usable local PDFs.

## F5 Exp 2 Mechanism

Table 5 is aggregation-only from judged T1 rows:

```bash
envs/mpvrdu/bin/python -m cli.experiments --phase judge --experiment T5_composition --full --judge gpt-4o-mini
```

Table 6 adds matched/cross retrieval generation on Kaya, including retrieval
metrics in `results/cache/full/T6_matched_cross/side/retrieval.jsonl`:

```bash
envs/mpvrdu/bin/python -m kaya.kaya submit kaya/generate.py -- --experiment T6_matched_cross --full
envs/mpvrdu/bin/python -m kaya.kaya pull
envs/mpvrdu/bin/python -m cli.experiments --phase judge --experiment T6_matched_cross --full --judge gpt-4o-mini
```

Table 6 is only populated for bins whose Table-1 oracle frontier is `TLV` or
`V`; an empty CSV with the stable columns means no bin met that condition.

## F6 Exp 3 Routing

Table 7 reuses judged T1 rows and adds the classifier side artifact on Kaya:

```bash
envs/mpvrdu/bin/python -m kaya.kaya submit kaya/generate.py -- --experiment T7_routing --full
envs/mpvrdu/bin/python -m kaya.kaya pull
envs/mpvrdu/bin/python -m cli.experiments --phase judge --experiment T7_routing --full --judge gpt-4o-mini
```

Predicted routing counts classifier cost explicitly as total classifier latency
divided by the number of evaluated question rows in the predicted policy. The
checkpoint asks whether predicted routing beats both uniform baselines on the
accuracy-latency Pareto view.
