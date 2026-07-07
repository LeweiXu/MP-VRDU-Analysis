# Session handoff - 2026-07-07

This session changed the generation pipeline to be YAML-first, added scanned-vs-
digital text routing, and launched the full G1/G2/G5 rerun on Kaya.

Current active cluster job:

- Job: `1012121`
- Name: `g1g2g5-full`
- Spec: `specs/g1_g5_rerun.yaml`
- Run tag: `yaml-g1-g2-g5-rerun`
- Resources: `2x V100`, `64G`, `24h`
- Last checked: running on `k036` at `52:19` elapsed, still in
  `G1_sufficiency` parse pre-pass.
- Status at last check: green. Logs show active Marker/OCR/layout work, with no
  `Traceback`, `ERROR`, `CUDA out of memory`, `Killed`, or Slurm failure text.

A local watcher is running for the transition into inference:

- Watcher session: `18442`
- It runs `kaya pull` once per minute.
- It exits and reports when
  `results/cache/yaml-g1-g2-g5-rerun/full/G1_sufficiency/predictions.jsonl`
  appears with at least one row.
- Watcher log: `/tmp/mpvrdu_inference_watch_1012121.log`

## Main changes made

### OCR routing while keeping paper rungs

The representation ladder remains the paper ladder:

- `T`
- `TL`
- `TLV`
- `V`

There is no OCR-specific rung. Instead, the text channel now routes by document
scan label:

- digital-born documents use Marker text
- scanned documents use PaddleOCR text

The scan/digital source is `annotations/doc_labels.csv`; human `scan_label` wins,
then `auto_scan`, with render-based classification as fallback. `TL` and `TLV`
continue using the existing layout channel.

### YAML-first experiment specs

Generation now supports YAML specs through:

```bash
python -m cli.generate --spec <spec.yaml>
```

The loader expands runs into the existing `GenerationTask` machinery. Specs can
define:

- `version`, `name`, `run_tag`, `mode`, `dataset`
- shared `config`
- multiple `runs`
- question selectors
- models
- conditions: `oracle`, `full`, `buried`, `retrieved`
- representations
- side artifacts such as retrieval diagnostics

Run-tag based artifact consumption is also wired:

```bash
python -m cli.judge --run-tag <tag> --judge stub
python -m cli.build --run-tag <tag> --bootstrap 0
```

### Representations

The code supports dynamic channel combinations such as `TV` and `LV`, but this
project's paper experiments should use the rung set only:

```yaml
representations: [T, TL, TLV, V]
```

For G5 retrieval, the rerun intentionally uses `TLV` only.

## Specs

Important YAML files:

- `specs/full_generation.yaml`: full generation template.
- `specs/smoke_generation.yaml`: smoke spec, currently minimal and includes G1,
  G2, and G5.
- `specs/smoke_g1_g5.yaml`: minimal one-GPU G1/G5 smoke spec.
- `specs/g1_g5_rerun.yaml`: active full rerun spec, despite the old filename.

The active rerun spec currently contains:

- `G1_sufficiency`
  - model: `qwen3vl-8b-local`
  - reps: `T, TL, TLV, V`
  - condition: `oracle`
- `G2_family`
  - model: `internvl3-8b-local`
  - reps: `T, TL, TLV, V`
  - condition: `oracle`
- `G5_retrieval`
  - model: `qwen3vl-8b-local`
  - rep: `TLV`
  - retrievers: `vision`, `text`
  - k sweep: `1, 3, 5, 7, 9`
  - writes `retrieval.jsonl`

The spec was validated with `experiments.yaml_spec.load_yaml_experiment`.

## Smoke tests and verification

Local focused tests passed:

```text
21 passed
```

Earlier broader local run passed:

```text
54 passed, 1 skipped, 1 deselected
```

The deselected test was the known network/Hugging Face related prestage test.

Kaya smoke job `1012073` completed successfully:

- run tag: `yaml-smoke-g1-g5`
- G1 expanded to `2 questions x 2 reps = 4 predictions`
- G5 expanded to `1 question x 2 k values x 2 retrievers = 4 predictions`
- G5 retrieval artifact had the expected text/vision rows at `k=1` and `k=3`
- run-tag judge/build paths worked with the stub judge

## Kaya jobs from this session

Cancelled:

- `1011922` - old `yaml-smoke`
- `1011928` - old 15 minute smoke
- `1011965` - old 15 minute smoke
- `1011968` - pending G2-inclusive smoke
- `1012106` - G1/G5-only full job, cancelled before replacing with G1/G2/G5

Completed:

- `1012073` - one-GPU G1/G5 smoke, successful

Active:

- `1012121` - full G1/G2/G5 rerun

## Dependency note

`PyYAML==6.0.2` was added to both requirements files and installed directly into
the Kaya environment:

```bash
envs/mpvrdu/bin/python -m pip install PyYAML==6.0.2
```

It was already present on Kaya when checked.

## Useful commands

Check active job state:

```bash
envs/mpvrdu/bin/python -m kaya.kaya watch 1012121 --tail-lines 160
```

Pull live logs/artifacts without waiting for job completion:

```bash
envs/mpvrdu/bin/python -m kaya.kaya pull
```

Inspect live logs:

```bash
tail -n 120 logs/g1g2g5-full_1012121.out
tail -n 120 logs/g1g2g5-full_1012121.err
```

Search logs for failures:

```bash
rg -n "Traceback|Exception|ERROR|CRITICAL|CUDA out of memory|OutOfMemory|OOM|FAILED|Killed|slurmstepd|RuntimeError" \
  logs/g1g2g5-full_1012121.out logs/g1g2g5-full_1012121.err
```

Check whether G1 inference has started:

```bash
test -s results/cache/yaml-g1-g2-g5-rerun/full/G1_sufficiency/predictions.jsonl && \
  wc -l results/cache/yaml-g1-g2-g5-rerun/full/G1_sufficiency/predictions.jsonl
```

After the job completes, score and build:

```bash
python -m cli.judge --run-tag yaml-g1-g2-g5-rerun --judge <judge>
python -m cli.build --run-tag yaml-g1-g2-g5-rerun --bootstrap 1000
```

Use `--judge stub` only for plumbing checks, not final experiment scoring.
