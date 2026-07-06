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
  prediction caches under `results/cache/<smoke|full>/<task>/`.
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
- `experiments/`: the experiment library — one generation task per file
  (`G1_sufficiency` … `G6_classifier`) over `base.py` + `registry.py`, the
  generate+judge engine `driver.py`, the table builders `tables.py`, and the
  table routing `reporting.py`. Add a generation experiment by dropping in a new
  `G*_*.py` and registering it.
- `cli/`: the runnable entry points — `generate` (GPU), `judge`, `build`, plus
  `gates` and `run_probe`.
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

## Generation Task Selectors

Work is organized by **generation task** (the only GPU work), not by table.
Use `--generation` with a single task or a group:

```text
G1_sufficiency     oracle ladder, primary 8B      -> tables 1, 2, 5, 7
G2_family          InternVL3-8B ladder            -> table 3
G3_dataset         held-out MMLongBench ladder    -> table 4
G5_retrieval       matched/cross retrieval cells  -> table 6
G6_classifier      doc-type classifier (side)     -> table 7 routing price

reasoners          G1, G2, G3, G5 (the reasoner-cell tasks)
all                every task
```

Tables are then built locally from these tasks' judged rows (see below). A
scale-sanity task (2B/32B, table 8) is out of scope for now.

## Smoke Runs

Smoke mode is the default: frozen small corpus, Qwen3-VL-2B, lower output token
cap. It uses the same caches and table builders as full mode.

Generate one smoke task on Kaya:

```bash
envs/mpvrdu/bin/python -m kaya.kaya submit --time 00:10:00 --mem 16G --cpus-per-task 2 \
  cli/generate.py -- --generation G1_sufficiency
```

Pull remote artifacts:

```bash
envs/mpvrdu/bin/python -m kaya.kaya pull
```

Judge (score predictions) then build the tables locally:

```bash
envs/mpvrdu/bin/python -m cli.judge --generation G1_sufficiency --judge gpt-4o-mini
envs/mpvrdu/bin/python -m cli.build
```

Run all smoke tasks:

```bash
envs/mpvrdu/bin/python -m kaya.kaya submit --time 00:30:00 cli/generate.py -- --generation all
envs/mpvrdu/bin/python -m kaya.kaya pull
envs/mpvrdu/bin/python -m cli.judge --generation all --judge gpt-4o-mini
envs/mpvrdu/bin/python -m cli.build
```

## Full Runs

Add `--full` to use the full MMLongBench corpus and Qwen3-VL-8B primary model.
Use `--questions N` for capped pilots that still use full-mode model/config.

Generate a 10-question full-mode sufficiency pilot on Kaya:

```bash
envs/mpvrdu/bin/python -m kaya.kaya submit --time 00:20:00 --mem 32G --cpus-per-task 2 --gres gpu:v100:2 \
  cli/generate.py -- --generation G1_sufficiency --full --questions 10
```

Kaya's `gpu` partition is V100-based. Full-mode Qwen3-VL-8B can OOM on a
single 16 GB V100 even for capped pilots, so use two V100s for full 8B
generation. Smoke mode uses Qwen3-VL-2B and can use one GPU.

Generate the full sufficiency task (source of Table 1):

```bash
envs/mpvrdu/bin/python -m kaya.kaya submit cli/generate.py -- --generation G1_sufficiency --full
```

Judge then build locally:

```bash
envs/mpvrdu/bin/python -m kaya.kaya pull
envs/mpvrdu/bin/python -m cli.judge --generation G1_sufficiency --full --judge gpt-4o-mini
envs/mpvrdu/bin/python -m cli.build --full
```

Section-2 gate commands:

```bash
envs/mpvrdu/bin/python -m cli.gates frontier \
  --table results/tables/full/table1_headline.csv \
  --json-output results/gates/F1_frontier_divergence.json

envs/mpvrdu/bin/python -m cli.gates agreement-sample --full \
  --results results/cache/full/G1_sufficiency/results.jsonl \
  --output results/gates/agreement_sample.csv

envs/mpvrdu/bin/python -m cli.gates agreement-score \
  --sheet results/gates/agreement_sample.csv \
  --json-output results/gates/F2_judge_human_agreement.json

envs/mpvrdu/bin/python -m cli.gates classifier-pilot --full \
  --output results/gates/classifier_pilot.csv \
  --json-output results/gates/F3_classifier_feasibility.json
```

The other generation jobs (feed tables 3/4/6/7):

```bash
envs/mpvrdu/bin/python -m kaya.kaya submit cli/generate.py -- --generation G2_family --full
envs/mpvrdu/bin/python -m kaya.kaya submit cli/generate.py -- --generation G3_dataset --full
envs/mpvrdu/bin/python -m kaya.kaya submit cli/generate.py -- --generation G5_retrieval --full
envs/mpvrdu/bin/python -m kaya.kaya submit cli/generate.py -- --generation G6_classifier --full
```

Run every task as one SLURM job, continuing past per-task failures:

```bash
envs/mpvrdu/bin/python -m kaya.kaya submit \
  --job-name all_generate \
  --gres gpu:v100:2 \
  --cpus-per-task 2 \
  --mem 32G \
  --time 1-00:00:00 \
  cli/generate.py -- --generation all --full --continue-on-error
```

Each task writes its own cache directory under `results/cache/full/<task>/`,
including a `generate_status.json` with `success` or failure details.

Judge every task, then build all eight tables in one local step:

```bash
envs/mpvrdu/bin/python -m cli.judge --generation all --full --judge gpt-4o-mini
envs/mpvrdu/bin/python -m cli.build --full
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
  cli/generate.py -- --generation G1_sufficiency --full --questions 10
```

Short walltimes usually matter most for queue backfill. Use `--no-wait` to
submit and return immediately, then watch later:

```bash
envs/mpvrdu/bin/python -m kaya.kaya submit --no-wait cli/generate.py -- --generation G1_sufficiency --full
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
