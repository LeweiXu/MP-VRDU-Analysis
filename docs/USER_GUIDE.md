# User guide: A Doc-Type Recipe for Multi-Page Document QA (v4)

> The user-facing guide: *what* the paper claims and *why* (below), plus *how to
> run* the experiments (the Runbook at the end). This guide reflects the **v4
> pivot** (`docs/pivot_v4.md`) for the thesis half. `README.md` is the ground
> truth for how the codebase is built *today*, and `docs/AGENT_GUIDE.md` holds the
> implementation decisions and reference. The v4 mechanism and task changes
> (PyMuPDF `T`, parser-derived `TL`, four generation tasks, the new telemetry
> schema) are the adopted plan, but the code migration is still pending, so the
> **Runbook below describes the current (pre-migration) commands and tasks**, which
> still work as written.

---

## 1. One-line thesis

**The representation an MP-VRDU system requires is a function of document type, not a single
property of the model.** We measure this function on a cost-ordered representation ladder, turn it
into a deployment recipe indexed by document type, and explain the recipe through evidence
composition.

Venue target: **EACL long paper** (8 pages). Everything not serving the thesis is moved to the
honours thesis or explicitly cut. The whole plan follows two principles: **few tasks, full
coverage** (the smallest number of runs collects all the data every table needs, so most
"experiments" are YAML runs of one task with a single field changed) and **collect all cheap data,
every run** (uniform telemetry on every run, so we select late what goes in the main paper vs the
appendix).

## 2. Motivation

Sensitive documents (medical records, contracts, financial filings, compliance reports) cannot be
sent to a cloud API; they must be processed on-premise with a self-hosted open MLLM, realistically
3B–32B. Under that constraint the expensive design choices are made without evidence:

- Do you need to feed pages as images at all? Vision encoding is the most expensive thing on the
  GPU; if cheap text suffices for your documents, the saving is large.
- Does the answer depend on what kind of document you have? A contract corpus and a slide-deck
  corpus are not the same problem; one recipe cannot serve both without waste.
- Without doc-type labels, is it worth classifying each document? Routing adds a classifier's own
  cost; if a uniform policy is nearly as good, routing is dead weight.

The contribution is a **recipe indexed by document type**, plus the mechanism that explains it,
plus honest cost-frontier readings across representation, parser, retriever, resolution,
quantization, and model size.

## 3. Central construct: the cost-ordered representation ladder

The cost knob is *how evidence is represented*. Holding gold pages fixed, page content is fed from
cheapest to most expensive; we find the cheapest form that still works, separately per document
type.

| Rung | Content | Role |
|---|---|---|
| `T` | cheap embedded-text extraction (PyMuPDF); digital-born only | cheapest reference |
| `TL` | parser-derived layout-rich text (markdown from a PDF parser) | mid |
| `TLV` | parser text + native-resolution page image | expensive |
| `V` | page image only | parser-independent reference |

Two v4 changes to state honestly: **bounding-box JSON is dropped everywhere** (it drove truncation
and did not help the reasoner), so at `TL` the parser's markdown text *replaces* `T`'s cheap text
rather than adding a layout channel; and the ladder is therefore **cost-ordered, not cumulative**
(`T ⊄ TL`). The rungs go `T` (cheap extract) < `TL` (parser) < `TLV` (parser + image), with `V` the
parser-independent image-only reference. The `T/TL/TLV/V` names stay for continuity, but "L" is now
vestigial: "TL" means "parser-derived text."

The parser at `TL`/`TLV` is itself under comparison (PyMuPDF floor, PaddleOCR-VL, MinerU 2.5,
Unlimited OCR), scored by **downstream QA accuracy** rather than edit-distance. Marker is dropped
from the comparison set.

## 4. Deployable vs analytical axes

- **Document type is deployable.** A lightweight classifier can predict it from the first pages
  (RQ3), so recipes may be indexed by it.
- **Question type is analytical only.** It is used to *explain* the recipe (mechanism), never in a
  deployment recommendation, because a practitioner does not know a question's type in advance.

## 5. Research questions

The RQ framing is not set in stone; what matters is that all data is collected correctly and
uniformly. The answerable / unanswerable split (§9) governs which questions feed which RQ.

**RQ1 - Recipe by doc type (what to build).** Given the correct pages, what is the cheapest
representation that lets an 8B MLLM reason to an answer, and does that frontier depend on document
type? *Deliverables:* the cost-ordered headline table (three bins × four representations, frontier
marked, answerable-only); a **parser comparison** (swap the `TL`/`TLV` parser); an **image-
resolution sweep** (supervisor hardware); plus family (InternVL) and dataset (held-out MMLongBench)
replications.

**RQ2 - Retrieval (mechanism + retrieval-side modality).** Does retrieval modality have to match
reasoning modality? *Deliverables:* **matched vs cross across all three bins** at `TLV` (with `V` via
YAML); a **top-k sweep** with retrieval methods framed as cost rungs, k ∈ {1,3,5,7,10} on supervisor
hardware; and a **retrieval-accuracy benchmark** (page P/R/F1 per bin, every method), which is a
by-product of the same retrieval pass. RQ2 inference runs at `TLV`/`V` only, since `TLV` is already
the strongest reasoning rung and matches what you would deploy.

**RQ3 - Deployment (routing + hallucination + prompting + cost sweeps).** *Deliverables:* **routing**,
four policies (oracle, predicted, uniform-cheapest, uniform-strongest), predicted-routing counting
the classifier's own cost; **hallucination × prompting** on the ~250 unanswerable questions
(retrieval-fed 2–3 pages, three prompt conditions, correct = abstention, no oracle arm); and **cost
sweeps** over quantization (4/8/16-bit) and model size (2B/4B/8B/32B), framed as cost-frontier
studies on supervisor hardware.

## 6. Retrieval methods as cost rungs

Both retrieval axes are laid out cheapest → most expensive, mirroring the representation ladder.

| Axis | Cheap | Mid | Expensive |
|---|---|---|---|
| **Text** | BM25 | BGE-M3 | Qwen3-Embedding-4B |
| **Vision** | ColModernVBERT | ColQwen2.5-v0.2 | ColQwen3-4B |

**Joint retrieval** is a free post-hoc deduplicated union of two already-computed page sets
(matched-tier pairs only, k ∈ {1,3,5} per method so the union stays < 10 pages). It is its own
condition family, not a point on the single-method k-axis.

## 7. Pre-registered setup

Every choice below is fixed before the main runs.

- **Primary cost metric:** latency per question at batch=1, with **prefill and decode split out**
  (prefill isolates the cost of ingesting the representation). Tokens, peak VRAM, and truncation are
  reported alongside. The full telemetry schema (`docs/pivot_v4.md` §6) is collected uniformly on
  every run.
- **Sufficiency margin:** accuracy drop ≤ 3 points relative to the strongest representation.
  Sensitivity for margin ∈ {2, 3, 5} in the Appendix.
- **Doc-type binning by manual annotation.** MMLongBench's native `doc_type` encodes *domain*, not
  *dominant modality*, so we label the bin axis directly. A fast (<1 hr) manual pass over all **135
  documents** produces `bin_label` (**text-dominant / mixed-modality / visual-dominant**),
  `scan_label` (`digital` / `scanned`), and `dominant_visual` (exploratory). Bins describe which
  modality dominates the document's information content, not scan status and not page count. A
  second annotator labels a 20–30 doc subset and we report Cohen's κ (recommended, not blocking).
- **Ladder implementation:** `T` = PyMuPDF embedded text (digital-born only); `TL` = parser-derived
  markdown text (parser under comparison); `TLV` = that text + native-resolution page image; `V` =
  page image only. No bounding-box channel.
- **Reasoner:** Qwen3-VL-8B primary. InternVL3-8B replicates the RQ1 headline only. Quantization
  (4/8/16-bit) and model size (2B/4B/8B/32B) are cost-frontier sweeps on supervisor hardware.
- **Retrieval:** the cost rungs in §6, plus free joint unions. RQ2 inference runs at `TLV`/`V`; the
  k-sweep runs on the supervisor.
- **Judge:** a different family from the reasoner. Gemini 2.5 Flash is the default (free tier);
  GPT-4o-mini is the paid alternative. Judge-human agreement on 200 hand-labelled questions;
  **Cohen's κ ≥ 0.75 required** on the answerable-only set before any main-run number is trusted.
- **Confidence:** every headline number carries a **document-level** bootstrap 95% CI (1000 resamples
  over documents). A frontier claim requires the cheaper representation's CI upper bound to reach
  within 3 points of the strongest representation's point estimate.
- **Hardware split, marked per run.** We run on **Kaya** (2×V100 16 GB, sm_70, no FlashAttention-2)
  and the supervisor's **A100 / H100**; every YAML run carries `machine: kaya | supervisor`. The
  size/quant/resolution sweeps and the RQ2 k-sweep are supervisor-only.

## 8. Experiments and the hardware / evidence-page partition

Four generation tasks, per "few tasks, full coverage." The parser, resolution, family, dataset,
quantization, and model-size "experiments" are **YAML runs over G1**, not separate tasks.

- **G1 - Oracle ladder.** Oracle pages × {T, TL, TLV, V}, answerable-only, 8B. Feeds the headline
  and the per-bin frontier, plus the parser/resolution/family/dataset/quant/size runs.
- **G2 - Retrieval.** One pass, two scorers: the retrieval-accuracy side-artifact (every method) and
  the reasoner at `TLV`/`V` for matched-vs-cross and the k-sweep. Supervisor.
- **G3 - Hallucination / prompting.** Unanswerable-only × retrieved 2–3 pages × three prompt
  conditions, correct = abstention.
- **G4 - Routing / classifier.** Classifier pricing + routing-policy assembly; routing accuracy
  reuses G1's rows.

**Hardware / evidence-page partition.** Most MMLongBench questions have 0–2 evidence pages; only
~50 have >2. **Kaya** runs the ≤ 2 evidence-page questions (short, so the input cap can be raised);
the **supervisor** runs the > 2 remainder at the same agreed cap. The cap is a fixed constant
identical on both machines; machine is the executor, not a condition. Cells record `machine` /
`cap_used`, runs record `evidence_page_filter`, and full-corpus tables pool the two slices at build
time.

> Note: the current code still ships six tasks (G1/G2/G3/G5/G6, with G4_scale a stub) and the v3
> mechanisms. The four-task consolidation above is the v4 plan; the Runbook below documents the
> commands and tasks as they exist today.

## 9. The answerable / unanswerable split

The ~250 unanswerable questions (of 1091) are **removed from RQ1 and RQ2** (so those accuracies are
"accuracy on answerable questions") and used **only** in the RQ3 hallucination study, which is
necessarily retrieval-fed (no oracle arm exists for zero-gold-page questions). The judge's
abstention scoring and the κ ≥ 0.75 gate are confirmed on the answerable-only set.

## 10. Known risks fixed by data

- **Visual-dominant bin may be thin.** Most likely Gate-1 casualty, widest CIs. Fallbacks, in order:
  (i) collapse to a two-bin contrast (text-dominant vs rest); (ii) recruit a visual-heavy dataset
  (SlideVQA). The manual `bin_label` annotation is the insurance against a bad bin axis.
- **Sampling correlation.** Questions cluster within documents (135 docs, 1091 Q). Any subsetting and
  all CIs are handled at the **document level**.
- **V100 hardware limits.** Kaya's V100s are Volta (sm_70): no FlashAttention-2, so attention can
  fall back to an O(seq²) kernel that OOMs long multi-page sequences, and the 8B does not fit one
  V100 in bf16. Mitigations are baked in (efficient kernel, per-size caps, per-cell skip on OOM,
  2×V100 sharding or 4-bit). The size/quant/resolution sweeps and the k-sweep run on the supervisor.
- **Parser environments.** The comparison parsers are heavy, separately-pinned VLM stacks that will
  not co-exist with the reasoner env; each runs as an isolated env pre-warmed to the parser cache.

## 11. What was cut (and where it went)

Cut from the paper, retained for the honours thesis / future work: the full distractor-burying sweep
beyond the RQ2 k-depth study; scaling as a *story* (kept only as a cost-frontier sanity check); the
multi-dataset robustness suite beyond one replication. These are real but do not serve the single
thesis at 8 pages.

---

# Runbook: running the experiments locally

This is the local guide: the pipeline runs on your own machine here. To dispatch
the GPU-heavy generation to a SLURM/HPC cluster instead, see
`kaya/KAYA_USER_GUIDE.md` (the same commands, wrapped in push / submit / pull).

> The commands, flags, and task names below describe the **current** implementation
> (still the v3 mechanisms and the six-task layout). The v4 pivot's task
> consolidation and mechanism changes are the adopted plan but not yet in the code,
> so run against what is documented here until the migration lands.

Commands assume the project environment is active (or prefix each with
`envs/<your-env>/bin/`). Everything is root-relative and self-contained under the
repo root.

## The three roles

The study is organized by **generation task**, not by paper table (many tables
are pure aggregations of the same generated predictions). Work splits into three
role modules, so the GPU half runs on a cluster and everything else stays local:

1. **generate** (`cli.generate`, GPU): runs a task's cells (conditioner
   -> render -> representation -> reasoner) plus any GPU side work (retrieval
   diagnostics, the doc-type classifier), caching predictions per task. No
   internet.
2. **judge** (`cli.judge`, internet, no GPU): reads a task's cached
   predictions, scores each with an LLM judge, writes `results.jsonl`. Builds no
   tables. Loads no models.
3. **build** (`cli.build`, local): routes each table's source-task judged
   rows into the eight table CSVs plus a combined `all_tables.md`. Pure pandas.

```bash
python -m cli.generate --spec specs/full_generation.yaml   # GPU: cache predictions
python -m cli.judge --run-tag yaml-full                    # score cached predictions
python -m cli.build --run-tag yaml-full                    # build CSVs + .md from artifacts
```

A cluster submits `cli/generate.py --spec <file.yaml>`; see `kaya/KAYA_USER_GUIDE.md`.

### Generation tasks (what runs on the GPU)

| Task | Generates | Feeds tables |
|---|---|---|
| `G1_sufficiency` | oracle pages x the T/TL/TLV/V ladder, primary 8B | 1, 2, 5, 7 |
| `G2_family` | the same ladder on InternVL3-8B | 3 (with G1) |
| `G3_dataset` | the ladder on a held-out MMLongBench subset (text_heavy + in_between) | 4 |
| `G5_retrieval` | matched/cross retrieval cells swept over k=(1,3,5,7,9) + retrieval R/P/F1 per k | 6 |
| `G6_classifier` | the doc-type classifier per document (side only) | 7 (routing price) |

(A scale-sanity task for 2B/32B, feeding Table 8, is out of scope for now, so
Table 8 shows the single primary size.)

## YAML generation specs

Generation is YAML-first. A spec declares the cache namespace, smoke/full mode,
config overrides, and one or more explicit cell grids. `specs/full_generation.yaml`
is the complete template for G1/G2/G3/G5/G6; `specs/smoke_generation.yaml` is a
small 2B smoke template. Representations may be any ordered combination of text
(`T`), layout (`L`), and vision (`V`): `T`, `L`, `V`, `TL`, `TV`, `LV`, `TLV`.

## `cli.generate` / `cli.judge` arguments

| Flag | Default | Meaning |
|---|---|---|
| `--spec PATH` | required for generate | YAML generation spec; it carries all run config. |
| `--full` | off (smoke) | Full corpus + 8B reasoner. Without it: the frozen ~7-doc smoke corpus + 2B. |
| `--judge SPEC` | `gemini` | (judge only) `gemini` (free tier), `gpt-4o-mini` (paid), or `stub` (offline). |
| `--questions N` | none | Global cap: first N questions. Overrides `--per-bin-questions`. |
| `--per-bin-questions N` | 100 | Full mmlongbench only: ~N questions per Option-A bin, whole documents. `0` = whole corpus. |
| `--sample-seed N` | 0 | Which documents fill the per-bin subset. |
| `--quantization {4bit,8bit}` | off (bf16) | Quantized reasoner (bitsandbytes); appends `-4bit`/`-8bit` to the spec, so quantized rows get their own cache. |
| `--visual-resolution {full,high,med,low,min}` | off (size-aware) | Fix the per-page vision-token budget for every reasoner. `full`≈1280, `high`≈768 (current 8B default), `med`≈512, `low`≈320, `min`≈224 tokens/page. Lower = more downscaling. Not in the cache key, so clear/`--run-tag` when changing it for one spec. |
| `--run-tag TAG` | off | Namespace this run's cache tree (`results/cache/<TAG>/`) so parallel full runs don't share files. |
| `--continue-on-error` | off | Generate: record a failing task's status and continue. Judge: skip cells with no cached prediction (partial cache) so a partial table still builds. |
| `--verbose` / `--quiet` | smoke=verbose | DEBUG per-cell/per-stage logging / force INFO. |

`cli.build` takes `--full` / `--run-tag` (to locate the cache),
`--output-dir`, `--markdown`, `--bootstrap`, `--seed`.

**Judge/build rule:** judge and build are artifact-driven. They read manifests,
`predictions.jsonl`, `results.jsonl`, and side artifacts under the selected
`--run-tag`; they do not require the original generate flags to be repeated.
Run-tagged builds write to `results/tables/<run-tag>/`.

## Running individual vs all tasks

```bash
python -m cli.generate --spec specs/smoke_generation.yaml
python -m cli.generate --spec specs/full_generation.yaml
```

To run a subset, copy a template and remove or edit entries under `runs:`.

## Generation cache: what the GPU phase writes

Everything lands under `results/cache/<run-tag>/<mode>/<run-name>/`, where
`run-name` is the YAML `runs[].name`. For `G1_sufficiency` in full mode:

```text
results/cache/yaml-full/full/G1_sufficiency/
  experiment_manifest.json
  predictions.jsonl       # durable reasoner outputs (the real GPU artifact)
  generate_results.jsonl  # predictions scored by a throwaway STUB judge (ignore its scores)
  generate_status.json    # {status: success|failed, error, traceback, ...} for this run
  <side artifacts>        # e.g. retrieval.jsonl (T6), classifier.jsonl (T7)
```

- **`predictions.jsonl`** is one JSON object per cell (append-only, resumable).
  Fields: `prediction_key` (SHA-256 over question_id + doc_id + condition +
  representation + **model_spec** + dpi), `question_id`, `doc_id`, `condition`,
  `representation`, `model_spec`, `text` (the model's answer), `page_indices`,
  `input_text_tokens`, `input_visual_tokens`, `output_tokens`, `latency_s`,
  `provenance`, `note`. Because `model_spec` is in the key, a bf16 run and a
  `-4bit` run (or 2B vs 8B) write distinct rows into the same file and never
  collide, so caches merge cleanly.
- **`generate_results.jsonl`** mirrors predictions but scored by a throwaway stub
  judge (`judge_spec: generate-throwaway`) just so the generate phase can print
  rough counts. Do not read accuracy from it.
- **`generate_status.json`** records whether the task finished
  (`status: success`) or its error + traceback. `--continue-on-error` writes one
  per task in a grouped run.
- The cache is **append-only and resumable**: re-running skips cells already in
  `predictions.jsonl` (keyed as above) and only generates missing/failed ones.
  Delete a task's dir (`rm -rf results/cache/full/<task>`) to force a clean
  regenerate.

Two shared, reproducible caches sit alongside (not per-experiment) and are worth
keeping between runs: `results/cache/renders/` (rasterized PDF pages) and
`results/cache/marker/` (Marker parse output).

## Judge phase: how it reads the cache and what it writes

`cli.judge` re-resolves the same cells but, instead of calling the
reasoner, **reads the cached prediction** for each cell (a prediction-cache hit
keyed without the judge), sends (question, gold answer, model answer) to the
judge, and writes the scored row. It loads no models and builds no tables; a
missing prediction raises (that cell was never generated) unless
`--continue-on-error`. It produces:

```text
results/cache/full/G1_sufficiency/results.jsonl   # predictions + REAL judge scores
```

`cli.build` then reads those `results.jsonl` files and writes the tables.

- **`results.jsonl`** is one row per cell with the real verdict. Fields:
  `cache_key`, `question_id`, `doc_id`, `doc_type`, `condition`, `representation`,
  `model_spec`, `judge_spec`, `answer` (judge-extracted answer), `correct`
  (bool), `abstained` (bool), `score`, `is_unanswerable`, `hop`,
  `evidence_sources`, the token/latency fields, and `metadata`. This is the file
  every table builder and gate reads.
- **`results/tables/<mode>/tableN_*.csv`** are the final tables (e.g.
  `table1_headline.csv`, `table7_routing.csv`): one row per bin/policy with
  accuracy, document-level bootstrap CIs, latency, token splits, and the marked
  frontier. `python -m cli.build` (re)builds these, plus a combined
  `all_tables.md`, from cached judged rows without re-judging.

Judge API keys live only in the local `.env` (`GEMINI_API_KEY` /
`OPENAI_API_KEY`), read from the environment, so export them (e.g.
`set -a; . ./.env; set +a`) before the judge phase.

## Gates

```bash
# F1: Go if >=2 bins differ. --table defaults to the Table-1 CSV for the mode/run-tag.
python -m gates frontier --run-tag bf16-lowres \
    --json-output results/gates/F1_frontier_divergence.json
# F2: 200-row sheet + a viewing packet (page images) under results/gates/agreement_view/.
# --results defaults to G1's results.jsonl for the mode/run-tag.
python -m gates agreement-sample --full --run-tag bf16-lowres \
    --output results/gates/agreement_sample.csv
# hand-label the human_label column in the CSV (open agreement_view.md alongside), then:
python -m gates agreement-score --sheet results/gates/agreement_sample.csv \
    --json-output results/gates/F2_judge_human_agreement.json    # F2: Cohen's kappa, gate 0.75
python -m gates classifier-pilot --full \
    --output results/gates/classifier_pilot.csv \
    --json-output results/gates/F3_classifier_feasibility.json   # F3: gate top-1 bin accuracy 0.70
```

Table notes: **Table 4** replicates on a held-out subset of MMLongBench documents
(disjoint docs for text_heavy/in_between; visual_heavy is out of scope for now, so
that bin is blank). **Table 6** is only populated for bins where vision materially
helps, and reports matched-vs-cross **per k** (a `k` column) across the G5 sweep
`(1, 3, 5, 7, 9)`. **Table 7** predicted routing reports the classifier's amortized
latency as its own column. **Table 8** (scale) shows the single primary size until
a scale generation task exists.

## Inspecting results and annotating documents

```bash
# Look at cached inference cells: copies the PDF + fed pages + an info.md (every
# generate & judge field) into ./inspect/ so you can open them in VSCode.
python -m scripts.inspect_results --run-tag bf16-lowres --full \
    --generation G1_sufficiency --incorrect-only --limit 20

# Manually label the 135 documents (text/visual bin, scanned vs digital, dominant
# visual element, multi-column). Interactive + resumable; writes annotations/doc_labels.csv.
python -m scripts.annotate_docs annotate
python -m scripts.annotate_docs score        # human bin vs the doc_type-derived bin

# Group the 135 PDFs into per-doc_type folders under .data/mmlongbench_docs_split/.
python scripts/split_docs_by_type.py
```
