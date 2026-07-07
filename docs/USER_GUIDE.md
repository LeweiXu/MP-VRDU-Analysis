# User guide: A Doc-Type Recipe for Multi-Page Document QA (v3)

> The user-facing guide: *what* the paper claims and *why* (below), plus *how to
> run* the experiments (the Runbook at the end). The companion
> `implementation_plan.md` governs how the codebase is built, and
> `docs/AGENT_GUIDE.md` holds the implementation decisions and reference. This
> guide mirrors the v3 experimental plan and supersedes all earlier
> multi-topic / nine-RQ / three-topic specs. Where an older doc disagrees, this
> file is current.

---

## 1. One-line thesis

**The representation an MP-VRDU system requires is a function of document type, not a single
property of the model.** We measure this function on a controlled representation ladder, turn it
into a deployment recipe indexed by document type, and explain the recipe through evidence
composition.

Venue target: **EACL long paper** (8 pages). Everything not serving the thesis is moved to the
honours thesis or explicitly cut.

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

The contribution is a **recipe indexed by document type**, plus the mechanism that explains it.

## 3. Central construct: the representation ladder

The cost knob is *how evidence is represented*. Holding gold pages fixed, page content is fed from
cheapest to most expensive; we find the cheapest form that still works, separately per document
type.

| Rung | Content | Role |
|---|---|---|
| `T` | raw text (Marker-extracted) | reference (cheap) |
| `T+L` | text + serialized bbox layout (JSON) | cumulative |
| `T+L+V` | text + layout + native-resolution page image | cumulative |
| `V` | page image only | parser-independent reference |

The ladder is cumulative on `T`/`T+L`/`T+L+V` (marginal value of each added modality); `V` is the
parser-independent reference.

## 4. Deployable vs analytical axes

- **Document type is deployable.** A lightweight classifier can predict it from the first pages
  (RQ3), so recipes may be indexed by it.
- **Question type is analytical only.** It is used to *explain* the recipe (mechanism), never in a
  deployment recommendation, because a practitioner does not know a question's type in advance.

## 5. Research questions

**RQ1 — Recipe by doc type (what to build).** Given the correct pages, what is the cheapest
representation that lets an 8B MLLM reason to an answer, and does that frontier depend on document
type? *Deliverable:* a 3-row headline table (text-heavy / in-between / visual-heavy) × four
representations, sufficiency frontier marked; replicated on a second model family and a second
dataset.

**RQ2 — Mechanism behind the recipe (why it looks that way).** What explains the doc-type effect,
and does retrieval require the same modality as reasoning? *Deliverable:* (a) the doc-type effect
re-expressed as an evidence-composition effect (the recipe is what it is because a text-heavy
corpus is X% pure-text evidence); (b) matched vs cross (text-retrieval + vision-reasoning)
pipelines, with cross wins explained via a locate–reason modality divergence — a page can be
text-locatable but vision-reasoned.

**RQ3 — Routing under uncertainty (what to do without labels).** Without gold doc-type labels,
does running a lightweight classifier and dispatching to the RQ1 recipe beat a uniform policy,
once the classifier's own cost is counted? *Deliverable:* corpus-level accuracy and total cost of
four policies — oracle routing, predicted routing, uniform-cheapest, uniform-strongest; classifier
latency is added into predicted-routing cost, not hidden.

## 6. Pre-registered setup

Every choice below is fixed before the main runs.

- **Primary cost metric:** latency per question at batch=1 on a single A100 80GB. Text and vision
  tokens reported separately as secondary. (Local deployment cares about response time; token
  counts across modalities are not FLOPs-equivalent.)
- **Sufficiency margin:** accuracy drop ≤ 3 points relative to the strongest representation.
  Sensitivity for margin ∈ {2, 3, 5} in the Appendix.
- **Doc-type binning (Option A, fixed):** MMLongBench-Doc native `doc_type` categories aggregated
  by semantic domain into three bins:
  - **Text-heavy** = Administration/Industry file + Academic paper + Research report/Introduction
    (**578 Q / 54 docs**).
  - **In-between** = Financial report + Guidebook + Tutorial/Workshop (**412 Q / 50 docs**).
  - **Visual-heavy** = Brochure (**101 Q / 15 docs**).
  Data-driven clustering by evidence-modality distribution is reported in the Appendix as
  robustness (and is the fallback if the visual-heavy bin proves too thin; see §9). Semantic
  aggregation is practitioner-interpretable; data-driven grouping would leak the effect being
  studied, so it is validator, not primary.
- **Ladder implementation:** `T` = Marker raw text; `T+L` = Marker text + serialized bbox JSON;
  `T+L+V` = Marker text + native-resolution page image; `V` = page image only. Parser swap
  (Marker vs PyMuPDF) in the Appendix.
- **Reasoner:** Qwen3-VL-8B primary. InternVL3-8B replicates the RQ1 headline table only.
  Qwen3-VL-2B / 32B for scale sanity in the Appendix.
- **Retrieval:** BM25 + BGE-large (text), ColQwen (vision). RQ2 compares *matched* (retrieval
  modality = reasoning modality) vs *cross* (text retrieval + vision reasoning). Vision-retrieval +
  text-reasoning is not tested (no practical rationale; inflates the comparison surface).
- **Judge:** GPT-4o-mini (different family from Qwen and InternVL). Judge–human agreement on 200
  hand-labelled questions; **Cohen's κ ≥ 0.75 required** before any main-run number is trusted.
- **Confidence:** every headline number carries a bootstrap 95% CI (1000 resamples over
  questions). A frontier claim requires the cheaper representation's CI upper bound to reach within
  3 points of the strongest representation's point estimate.

## 7. Experiments

- **Exp 1 · RQ1 — Recipe by document type.** Sweep the ladder on oracle pages with Qwen3-VL-8B;
  fill the 3×4 headline table (Table 1), mark the frontier; re-slice by question type into the
  analytical 3×4 (Table 2, not for deployment); replicate the headline on InternVL3-8B (Table 3)
  and LongDocURL (Table 4).
- **Exp 2 · RQ2 — Mechanism.** (a) Evidence-composition mediation: decompose each doc-type bin
  into shares of text/table/chart/figure/layout evidence; show per-modality frontier + composition
  predicts the doc-type frontier (Table 5). (b) Retrieval-side modality: on cells where RQ1 says
  vision is needed, compare matched vs cross pipelines under real retrieval on accuracy and latency
  (Table 6); cross wins explained by locate–reason divergence in one paragraph + one qualitative
  figure.
- **Exp 3 · RQ3 — Routing.** Four policies on the full corpus: oracle routing, predicted routing
  (Qwen3-VL-2B few-shot classifies the first pages, then recipe), uniform-cheapest (`T`
  everywhere), uniform-strongest (`T+L+V` everywhere). Predicted-routing total latency includes the
  classifier's own latency (Table 7).
- **Exp 4 · Appendix — Scale sanity.** Re-run the RQ1 headline on Qwen3-VL-2B and 32B (Table 8).
  Main text cites one sentence: "the recipe is qualitatively stable across 2B–32B," or names the
  bins where the frontier moves. No scaling headline is claimed.

## 8. Go / no-go gates (Weeks 1–2)

- **Gate 1 · RQ1 frontier divergence.** Run Exp 1's headline table on 8B, oracle pages, full
  MMLongBench-Doc. **Go** if ≥2 of 3 doc-type rows have different sufficiency frontiers. **No-go**
  if all three land on the same rung → doc-type is not a useful axis; reframe around evidence
  composition alone.
- **Gate 2 · Judge–human agreement.** Hand-label 200 questions across doc-type × question-type
  strata. **Go** if GPT-4o-mini reaches κ ≥ 0.75. **No-go** → iterate the judge prompt or fall back
  to GPT-4o full before any main run.
- **Gate 3 · Classifier feasibility.** On a 100-doc pilot, run Qwen3-VL-2B few-shot doc-type
  classification from the first two pages. **Go** if top-1 ≥ 70%. **No-go** → upgrade the
  classifier or scope RQ3 to the oracle-routing upper bound only.

## 9. Known risks fixed by data

- **Visual-heavy bin is thin (101 Q / 15 docs).** It is the most likely Gate-1 casualty and will
  carry the widest CIs. If it cannot be separated from the other bins at the 3-point margin, the
  fallbacks, in order, are: (i) adopt the Appendix evidence-composition (data-driven) binning as
  primary; (ii) collapse to a two-bin contrast (text-heavy vs rest); (iii) recruit a visual-heavy
  dataset (SlideVQA) as the visual anchor. Recorded so the choice is pre-committed, not improvised.
- **Sampling correlation.** Questions cluster within documents (135 docs, 1091 Q). Any subsetting
  and all CIs are handled at the **document level** (draw documents, take their questions) so
  precision is not overstated.
- **Qwen3-VL API availability.** `transformers==4.53.2` did not expose the Qwen3-VL model class at
  Stage 1; resolving the load path (transformers upgrade within the vLLM/colpali window, or a
  confirmed vLLM path) is on the critical path before Gate 1, since every number needs a working
  8B reasoner.

## 10. What was cut (and where it went)

Cut from the paper, retained for the honours thesis / future work: the full retrieval-sufficiency
and distractor-burying sweep; scaling as a *story* (kept only as an Appendix sanity check);
fail-safe abstention; the multi-dataset robustness suite beyond one replication. These are real but
do not serve the single thesis at 8 pages.

---

# Runbook: running the experiments locally

This is the local guide: the pipeline runs on your own machine here. To dispatch
the GPU-heavy generation to a SLURM/HPC cluster instead, see
`kaya/KAYA_USER_GUIDE.md` (the same commands, wrapped in push / submit / pull).

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
python -m cli.generate --generation G1_sufficiency --full   # GPU: cache predictions
python -m cli.judge      --generation all --full              # score cached predictions
python -m cli.build      --full                               # build the 8 CSVs + .md
```

A cluster submits `cli/generate.py`; see `kaya/KAYA_USER_GUIDE.md`.

### Generation tasks (what runs on the GPU)

| Task | Generates | Feeds tables |
|---|---|---|
| `G1_sufficiency` | oracle pages x the T/TL/TLV/V ladder, primary 8B | 1, 2, 5, 7 |
| `G2_family` | the same ladder on InternVL3-8B | 3 (with G1) |
| `G3_dataset` | the ladder on a held-out MMLongBench subset (text_heavy + in_between) | 4 |
| `G5_retrieval` | matched/cross retrieval cells + retrieval R/P/F1 | 6 |
| `G6_classifier` | the doc-type classifier per document (side only) | 7 (routing price) |

(A scale-sanity task for 2B/32B, feeding Table 8, is out of scope for now, so
Table 8 shows the single primary size.)

## `cli.generate` / `cli.judge` arguments

| Flag | Default | Meaning |
|---|---|---|
| `--generation SEL` | `all` | A task (`G1_sufficiency`), a group (`all`, `reasoners`), or a comma list. |
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

**Flag-matching rule:** judge re-resolves the same cells as generate, so the
corpus/model flags (`--generation`, `--full`, `--per-bin-questions`,
`--sample-seed`, `--quantization`, `--run-tag`) MUST match, or judge looks for
predictions that were never generated and errors.

## Running individual vs all tasks

```bash
python -m cli.generate --generation G1_sufficiency --full             # one task
python -m cli.generate --generation G1_sufficiency,G5_retrieval --full # a subset
python -m cli.generate --generation reasoners --full                  # the reasoner-cell tasks
python -m cli.generate --generation all --full                        # every task
```

Groups: `all` = G1,G2,G3,G5,G6; `reasoners` = the four tasks with reasoner cells
(skips the classifier side task). Tables 2 and 5 are pure aggregations of
`G1_sufficiency`, so building them just needs G1 generated + judged.

## Generation cache: what the GPU phase writes

Everything lands under `results/cache/<mode>/<task>/`, where `<mode>` is
`smoke` or `full` (and `<mode>` gains a `/<run-tag>` prefix under
`results/cache/` when `--run-tag` is set). For `G1_sufficiency` in full mode:

```text
results/cache/full/G1_sufficiency/
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
  frontier. `python -m cli.build` (re)builds these — plus a combined
  `all_tables.md` — from cached judged rows without re-judging.

Judge API keys live only in the local `.env` (`GEMINI_API_KEY` /
`OPENAI_API_KEY`), read from the environment, so export them (e.g.
`set -a; . ./.env; set +a`) before the judge phase.

## Gates

```bash
# F1: Go if >=2 bins differ. --table defaults to the Table-1 CSV for the mode/run-tag.
python -m scripts.gates frontier --run-tag bf16-lowres \
    --json-output results/gates/F1_frontier_divergence.json
# F2: 200-row sheet + a viewing packet (page images) under results/gates/agreement_view/.
# --results defaults to G1's results.jsonl for the mode/run-tag.
python -m scripts.gates agreement-sample --full --run-tag bf16-lowres \
    --output results/gates/agreement_sample.csv
# hand-label the human_label column in the CSV (open agreement_view.md alongside), then:
python -m scripts.gates agreement-score --sheet results/gates/agreement_sample.csv \
    --json-output results/gates/F2_judge_human_agreement.json    # F2: Cohen's kappa, gate 0.75
python -m scripts.gates classifier-pilot --full \
    --output results/gates/classifier_pilot.csv \
    --json-output results/gates/F3_classifier_feasibility.json   # F3: gate top-1 bin accuracy 0.70
```

Table notes: **Table 4** replicates on a held-out subset of MMLongBench documents
(disjoint docs for text_heavy/in_between; visual_heavy is out of scope for now, so
that bin is blank). **Table 6** is only populated for bins whose Table-1 frontier
is `TLV`/`V`. **Table 7** predicted routing reports the classifier's amortized
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

