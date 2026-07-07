# Session handoff — 2026-07-07

This session was pipeline tooling + two experiment changes, not a cluster run.
Everything below is **committed on `main`** (commits `02bffb6 large refactor +
README` and `cd85fad add top k sweep`); the tree is clean. Tests: **94 passed, 1
skipped**.

The one big open thread is the **OCR-as-a-rung** change: it's fully designed and
documented but deliberately **not implemented yet** (it's a frozen-interface
checkpoint). See "The main open thread" below.

## What shipped this session

- **`cli/` now holds only the three experiment roles** (`generate`, `judge`,
  `build`). Everything else moved to `scripts/`: `gates.py`, `run_probe.py`, plus
  the new tools. Invoke as `python -m scripts.<name>`. Target tree in
  `docs/implementation_plan.md` reflects it.
- **Item-1 cleanup (the "temp/cache/full looks table-organized" thing).** That was
  a stale gitignored `temp/` from before the generation-task refactor; deleted. The
  live cache is task-keyed `results/cache/[<run-tag>/]<smoke|full>/<task>/`. Also
  repointed `scripts/gates.py`'s old hardcoded `T1_headline` defaults to run-tag-
  aware resolution (`default_results_path`/`default_table1_path`, driven by
  `--run-tag`/`--full`).
- **Inspection tool.** `experiments/inspect.py` (library) + `scripts/inspect_results.py`.
  Joins a task's `predictions.jsonl` + `results.jsonl` back to the question + PDF,
  renders the fed pages, and dumps each cell into `./inspect/<slug>/` (copied PDF,
  page PNGs, `info.md` listing **every** generate + judge field). Filters:
  `--question/--doc/--representation/--condition/--incorrect-only/--abstained-only/--limit`.
  `inspect/` is gitignored.
- **Gate-F2 human-judging viewer.** `scripts.gates agreement-sample --render` now
  renders the sampled cells' pages into a scrollable `agreement_view.md`
  (`experiments/gates.py::render_agreement_packet`) so a human can label from
  VSCode; `agreement-score` (Cohen's kappa) is unchanged.
- **Document annotation tool.** `scripts/annotate_docs.py`: interactive, resumable
  `annotate` (opens each PDF, menu per field, writes after every doc);
  `dominant_visual` accepts **multiple** picks (stored `;`-joined). Fields:
  bin (text/in-between/visual-heavy), scanned/digital, dominant_visual, multi_column,
  notes; seeded with `doc_type_bin` + `data.render.classify_scanned` priors.
  `score` reports human-bin-vs-doc_type-bin agreement (tests the 3-domain
  assumption) + scanned fraction. Writes `annotations/doc_labels.csv` (committed,
  version-controlled). Companion `scripts/split_docs_by_type.py` groups the 135
  PDFs into per-doc_type folders under `.data/mmlongbench_docs_split/`.
- **G5 top-k sweep (implemented).** `config.k_values = (1, 3, 5, 7, 9)`. G5 now runs
  the full sweep: `matched_cross_sweep_cells` emits 10 cells/question
  (`retrieved_{vision,text}_k{k}`, question-major/k-minor), `run_side` logs
  `retrieval.jsonl` per (question, modality, k), and `build_table6_matched_vs_cross`
  reports **each k separately** (new `k` column, `_condition_k` parses k from the
  conditioner name). Retrievers memoize by `(question, page_count, k)`.

## The main open thread: OCR as its own rung (designed, NOT implemented)

Agreed plan, deferred on purpose. Full writeup lives in **README → "Planned: OCR as
its own rung"** and **AGENT_GUIDE → "G5 top-k sweep; OCR-rung plan (deferred)"**.
Summary:

- Add OCR as a fourth channel and its own cumulative rung → ladder becomes
  **`T / TO / TOL / TOLV / V`**. `T` stays pure Marker text; `TO` adds an `[ocr]`
  block; `V` stays vision-only. Payload order `[text]→[ocr]→[layout]→images`.
- OCR runs on **every** page (digital or scanned), cached to disk per (page, dpi)
  like Marker, warmed in the parse pre-pass, GPU freed before the reasoner. Use the
  existing `tools/text.py::ocr` (PaddleOCR), currently wired but unused.
- **Not an OOM risk** (analysis in README): the `max_input_tokens` cap truncates
  the combined text tail before attention; OCR adds text tokens, not vision tokens.
- **Why it's a checkpoint, not a silent edit:** it renames `schema.Modality`
  `TL`→`TOL`, `TLV`→`TOLV`, which touches the frontier order (`metrics/frontier.py`
  `RUNG_ORDER`), the table builders (`experiments/tables.py`: `{TLV,V}` sets, rung
  tuples, composition special-cases), `matched_cross_*` default rung, and
  `scripts/gates.py`'s agreement default rung. It needs a **fresh `--run-tag`**
  because the `bf16-lowres` cache/tables use the old rung names. When you implement
  it, work through those references and re-run.

## Other open items

1. **Re-run G5 to populate the sweep.** The existing `bf16-lowres` G5 cache only has
   `retrieved_*_k1` (the pre-sweep single k). Table 6 there is k=1 only. To get the
   full k=(1,3,5,7,9) table, re-generate + re-judge G5 (same run-tag adds the new
   cells since each k is its own condition/prediction row), then rebuild. The
   oracle-ladder tasks (G1/G2/G3) are unaffected by the sweep.
2. **`annotations/doc_labels.csv` is seeded, not annotated.** It has the 135 rows
   with auto priors but only 1 stray `bin_label` from a smoke test. To start real
   annotation, either `python -m scripts.annotate_docs annotate` (it skips
   already-done rows) or rebuild clean with `annotate_docs sheet --force`.
3. **The `inspect/` folder** (gitignored) has the G5 cells that were being viewed;
   safe to delete anytime with `rm -rf inspect/`.
4. **Deprioritized from the prior handoff:** the G2 InternVL replication (said to be
   not important right now) and the separate 4-bit run. Those Kaya job IDs from the
   earlier handoff are stale.

## Tooling quick reference

```bash
# inspect cached inference cells (writes ./inspect/, open in VSCode)
python -m scripts.inspect_results --run-tag bf16-lowres --full \
    --generation G1_sufficiency --incorrect-only --limit 20

# annotate the 135 docs (interactive, resumable)
python -m scripts.annotate_docs annotate
python -m scripts.annotate_docs score

# gate F2 with a viewing packet
python -m scripts.gates agreement-sample --full --run-tag bf16-lowres --render
python -m scripts.gates agreement-score --sheet results/gates/agreement_sample.csv

# group the 135 PDFs by doc_type for browsing
python scripts/split_docs_by_type.py
```

## Dataset note (came up at the end)

The **246 zero-evidence-page questions are the *unanswerable* ones** (gold answer
`"Not answerable"`, `hop="none"`, `is_unanswerable=True`; ~22% of the corpus). They
are not trick/general-knowledge questions; the correct behavior is to **abstain**,
and the judge scores them correct only on abstention. For G1 (oracle) there are no
gold pages, so `OracleConditioner`/`render_question_pages` feed **page 0 only** as a
sanity anchor. So for these, G1 is really an abstention/hallucination probe (does
the model say "not answerable" instead of inventing an answer from page 0), not a
reading-comprehension test.
