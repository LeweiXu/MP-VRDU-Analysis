# MP-VRDU Representation & Deployment Study

This repository implements the empirical pipeline for the MP-VRDU thesis:
measure which document representations are sufficient for multi-page visually
rich document understanding, and price those choices under deployable local
models.

The current codebase covers the MVP and Section-2 F1-F6 experiments. Stage F7
(appendix scale/sensitivity work) is intentionally left for a later pass.

## Pipeline Shape

The pipeline is built around frozen interfaces. Experiments choose questions,
input conditions, and representation rungs; the shared orchestrator handles
rendering, payload construction, model calls, judging, and caches.

```text
                   setup/stage assets
                 scripts/setup_env.py
                 scripts/prestage.py
                         |
                         v
 .data/ PDFs + rows   .cache/ HF/tool weights   results/cache/
         |                    |                       ^
         v                    v                       |
  data.loader           models.get_reasoner           |
  Question[] ----------> Reasoner backend             |
         |                    ^                       |
         v                    |                       |
 pipeline.conditioner   models.payload.ModelInput     |
 Oracle/Retrieved/etc.          ^                     |
         |                      |                     |
         v                      |                     |
 data.render.Page[] --> pipeline.representation ------+
         |              T / TL / TLV / V
         v
 pipeline.orchestrator
         |
         +--> PredictionCache: model output, no judge
         +--> ResultCache: judged ResultRow
                         |
                         v
                  experiments.tables
                  table CSVs in results/tables/
```

The important execution split is:

- **Generate phase:** runs on Kaya GPU compute nodes, offline, and fills
  prediction caches under `results/cache/<smoke|full>/<experiment>/`.
- **Judge/build phase:** runs locally with internet/API keys, reuses cached
  predictions, and writes CSVs under `results/tables/<smoke|full>/`.

## Repository Map

- `config.py`: root-relative paths and experiment knobs.
- `schema.py`: frozen dataclasses (`Question`, `Page`, `Payload`,
  `Prediction`, `Score`, `ResultRow` inputs).
- `data/`: MMLongBench and LongDocURL loaders, binning, PDF rendering.
- `tools/`: text, layout, OCR, and visual channel helpers.
- `pipeline/`: conditioners, representations, reasoner/judge ABCs, orchestrator.
- `models/`: backend registry plus Qwen3-VL and InternVL local backends.
- `covariates/`: retrieval and document-type classifier paths.
- `metrics/`: accuracy, cost, frontier, retrieval, abstention metrics.
- `experiments/`: one reusable experiment per table (`T1_headline` through
  `T8_scale`) plus the two-phase driver and table builders.
- `cli/`: local experiment, table, gate, and probe entry points.
- `kaya/`: remote setup, prestage, sync/submit runner, and GPU generation entry.
- `docs/`: fixed decisions, runbook, implementation plan, data/model/evaluation
  details.

Machine-local artifacts stay under `.cache/`, `.data/`, `envs/`, `results/`,
and `logs/`; these are intentionally ignored by git. The `data/` directory is
Python package code, not dataset storage.

## Local Setup

Create or refresh the local environment:

```bash
python3.11 -m venv envs/mpvrdu
envs/mpvrdu/bin/python -m pip install --upgrade pip wheel setuptools
envs/mpvrdu/bin/python -m pip install -r requirements.txt
```

Optional local API keys go in `.env`:

```bash
GEMINI_API_KEY=...
OPENAI_API_KEY=...
HF_TOKEN=...
```

`HF_TOKEN` may be forwarded to Kaya login-node setup commands. Judge keys are
not forwarded to Kaya; judging is local.

Run local unit coverage:

```bash
envs/mpvrdu/bin/python -m pytest
```

Useful focused checks:

```bash
envs/mpvrdu/bin/python -m pytest tests/test_data.py tests/test_reasoner.py
envs/mpvrdu/bin/python -m json.tool kaya/config.json
```

## Kaya Setup

All Kaya commands are run from the local repo root. The runner reads
`kaya/config.json`, rsyncs source to the configured remote root, loads modules,
activates the remote env, and submits SLURM jobs when needed.

Check resolved config:

```bash
envs/mpvrdu/bin/python -m kaya.kaya show-config
```

Push source:

```bash
envs/mpvrdu/bin/python -m kaya.kaya push
```

Create/update the remote conda env:

```bash
envs/mpvrdu/bin/python -m kaya.kaya run scripts/setup_env.py
```

Stage the full configured inventory on the Kaya login node:

```bash
envs/mpvrdu/bin/python -m kaya.kaya run scripts/prestage.py
```

For a smaller first setup pass:

```bash
envs/mpvrdu/bin/python -m kaya.kaya run scripts/prestage.py -- --smoke
envs/mpvrdu/bin/python -m kaya.kaya run scripts/prestage.py -- --skip-tool-caches --model-id Qwen/Qwen3-VL-2B-Instruct
```

The full prestage path downloads/stages MMLongBench, configured reasoner
weights, retrieval weights, and parser/OCR/tool caches. LongDocURL Table-4 runs
also require PDFs staged manually as
`.data/longdocurl/documents/<doc_no>.pdf`; the public annotation cache records
source paths but does not provide usable local PDFs.

## Experiment Selectors

Use `--experiment` with either a single experiment or a group:

```text
T1_headline        Table 1, RQ1 headline oracle ladder
T2_analytical      Table 2, analytical slices from T1
T3_family          Table 3, InternVL family replication
T4_dataset         Table 4, LongDocURL dataset replication
T5_composition     Table 5, evidence-composition mediation
T6_matched_cross   Table 6, matched/cross retrieval
T7_routing         Table 7, routing policies with classifier cost

rq1                T1-T4
rq2                T5-T6
rq3                T7
section2           T1-T7, excluding deferred F7/T8 appendix work
all                all registered experiments
```

Stage F7 / appendix sensitivity and scale-sanity work is not part of the
current runbook.

## Smoke Runs

Smoke mode is the default: frozen small corpus, Qwen3-VL-2B, lower output token
cap. It uses the same caches and table builders as full mode.

Generate one smoke experiment on Kaya:

```bash
envs/mpvrdu/bin/python -m kaya.kaya submit --time 00:10:00 --mem 16G --cpus-per-task 2 \
  cli/generate.py -- --experiment T1_headline
```

Pull remote artifacts:

```bash
envs/mpvrdu/bin/python -m kaya.kaya pull
```

Judge/build locally:

```bash
envs/mpvrdu/bin/python -m cli.experiments --phase judge --experiment T1_headline --judge gpt-4o-mini
```

Run all smoke experiments:

```bash
envs/mpvrdu/bin/python -m kaya.kaya submit --time 00:30:00 cli/generate.py -- --experiment all
envs/mpvrdu/bin/python -m kaya.kaya pull
envs/mpvrdu/bin/python -m cli.experiments --phase judge --experiment all --judge gpt-4o-mini
```

On a machine with both GPU and internet, generate and judge in one process:

```bash
envs/mpvrdu/bin/python -m cli.experiments --phase all --experiment T1_headline --judge gpt-4o-mini
```

## Full Runs

Add `--full` to use the full MMLongBench corpus and Qwen3-VL-8B primary model.
Use `--questions N` for capped pilots that still use full-mode model/config.

Generate a 10-question full-mode Table-1 pilot on Kaya:

```bash
envs/mpvrdu/bin/python -m kaya.kaya submit --time 00:20:00 --mem 32G --cpus-per-task 2 --gres gpu:v100:2 \
  cli/generate.py -- --experiment T1_headline --full --questions 10
```

Kaya's `gpu` partition is V100-based. Full-mode Qwen3-VL-8B can OOM on a
single 16 GB V100 even for capped pilots, so use two V100s for full 8B
generation. Smoke mode uses Qwen3-VL-2B and can use one GPU.

Generate full Table 1:

```bash
envs/mpvrdu/bin/python -m kaya.kaya submit cli/generate.py -- --experiment T1_headline --full
```

Judge/build full Table 1 locally:

```bash
envs/mpvrdu/bin/python -m kaya.kaya pull
envs/mpvrdu/bin/python -m cli.experiments --phase judge --experiment T1_headline --full --judge gpt-4o-mini
```

Section-2 gate commands:

```bash
envs/mpvrdu/bin/python -m cli.gates frontier \
  --table results/tables/full/table1_headline.csv \
  --json-output results/gates/F1_frontier_divergence.json

envs/mpvrdu/bin/python -m cli.gates agreement-sample --full \
  --results results/cache/full/T1_headline/results.jsonl \
  --output results/gates/agreement_sample.csv

envs/mpvrdu/bin/python -m cli.gates agreement-score \
  --sheet results/gates/agreement_sample.csv \
  --json-output results/gates/F2_judge_human_agreement.json

envs/mpvrdu/bin/python -m cli.gates classifier-pilot --full \
  --output results/gates/classifier_pilot.csv \
  --json-output results/gates/F3_classifier_feasibility.json
```

F4-F6 generation jobs:

```bash
envs/mpvrdu/bin/python -m kaya.kaya submit cli/generate.py -- --experiment T3_family --full
envs/mpvrdu/bin/python -m kaya.kaya submit cli/generate.py -- --experiment T4_dataset --full
envs/mpvrdu/bin/python -m kaya.kaya submit cli/generate.py -- --experiment T6_matched_cross --full
envs/mpvrdu/bin/python -m kaya.kaya submit cli/generate.py -- --experiment T7_routing --full
```

Run the Section-2 generation set (T1-T7) as one SLURM job, continuing past
per-experiment failures:

```bash
envs/mpvrdu/bin/python -m kaya.kaya submit \
  --job-name section2_generate \
  --gres gpu:v100:2 \
  --cpus-per-task 2 \
  --mem 32G \
  --time 1-00:00:00 \
  cli/generate.py -- --experiment section2 --full --continue-on-error
```

Each experiment writes its own cache directory under
`results/cache/full/<experiment>/`. The grouped generation command also writes
`generate_status.json` in each experiment directory with `success` or failure
details.

F4-F6 aggregation-only or local judge/build commands:

```bash
envs/mpvrdu/bin/python -m cli.experiments --phase judge --experiment T2_analytical --full --judge gpt-4o-mini
envs/mpvrdu/bin/python -m cli.experiments --phase judge --experiment T3_family --full --judge gpt-4o-mini
envs/mpvrdu/bin/python -m cli.experiments --phase judge --experiment T4_dataset --full --judge gpt-4o-mini
envs/mpvrdu/bin/python -m cli.experiments --phase judge --experiment T5_composition --full --judge gpt-4o-mini
envs/mpvrdu/bin/python -m cli.experiments --phase judge --experiment T6_matched_cross --full --judge gpt-4o-mini
envs/mpvrdu/bin/python -m cli.experiments --phase judge --experiment T7_routing --full --judge gpt-4o-mini
```

## Kaya Job Controls

Check current GPU, memory, queue, and scheduler start-estimate status:

```bash
envs/mpvrdu/bin/python scripts/kaya_status.py
envs/mpvrdu/bin/python scripts/kaya_status.py --json
```

`kaya.kaya submit` accepts SLURM overrides before the script path:

```bash
envs/mpvrdu/bin/python -m kaya.kaya submit \
  --job-name t1_10q \
  --partition gpu \
  --gres gpu:v100:2 \
  --cpus-per-task 2 \
  --mem 32G \
  --time 00:20:00 \
  cli/generate.py -- --experiment T1_headline --full --questions 10
```

Short walltimes usually matter most for queue backfill. Use `--no-wait` to
submit and return immediately, then watch later:

```bash
envs/mpvrdu/bin/python -m kaya.kaya submit --no-wait cli/generate.py -- --experiment T1_headline --full
envs/mpvrdu/bin/python -m kaya.kaya watch
```

Cancel a job if needed:

```bash
ssh kaya 'scancel <jobid>'
```

## Outputs

- Prediction cache: `results/cache/<smoke|full>/<experiment>/predictions.jsonl`
- Throwaway generate rows: `results/cache/<smoke|full>/<experiment>/generate_results.jsonl`
- Judged rows: `results/cache/<smoke|full>/<experiment>/results.jsonl`
- Side artifacts: same experiment cache directory, e.g. `classifier.jsonl` or
  `retrieval.jsonl`
- Tables: `results/tables/<smoke|full>/*.csv`
- Kaya logs: `logs/<job-name>_<jobid>.out` and `.err`

The caches are deterministic and resumable. Re-running an experiment fills
missing cells and reuses existing prediction rows when the question, condition,
representation, model spec, and rendering configuration match.
