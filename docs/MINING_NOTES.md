# Mining notes: what the current data answers, and what it does not

Written for the collaborator who plans experiments and drafts the paper without
running the code. For each RQ: the tables in `results/tables/all_tables.md` that now
answer it, and an explicit list of what it still **cannot** show on the data we have,
with the spec that would fix each gap.

Mechanics live elsewhere: `CODEBASE_GUIDE.md` Part A for the row schema and table
build, Part B for models/prompts/judge/dataset. `RQ_BRIEF.md` for what each RQ means.
Regenerate everything below with `python -m ops.build`.

Read the **per-column `n` footer on every table**. It is how you spot a cell measured
on fewer questions than its neighbours, which on this data is common: 3269 of 37429
cells (8.7%) OOMed on the 16 GB V100 and carry no accuracy. OOM is **not** missing at
random. It concentrates in long documents, the image-bearing rungs, and multi-hop
questions: on the headline run the TLV rung loses 130 of 847 questions, and multi-hop
attrition (343 → 236 across the ladder) is 15x the single-hop attrition (479 → 472).
So an unpaired comparison across rungs compares progressively easier question mixes.
That is exactly why the fidelity table below is paired.

---

## RQ1 — Error attribution (where the loss is)

**Tables:** `headline`, `fidelity_transition`, `composition`, `source_stratification`,
`attribution`, `parser`, `integration`, `hallucination`, `abstention_by_doctype`.

What they establish now:

- **The ladder is not monotonic.** TLV (parser text + page images) is the best rung at
  56.8%, above V at 45.9%, TL at 39.4% and T at 31.9% (`headline`). So the image
  channel and the text channel each carry something the other does not.
- **The lead fidelity number.** `fidelity_transition` pairs each question with itself
  across two rungs and reports how often adding a channel flips the verdict. Adding
  page images on top of parser text (TL→TLV) turns **18.7%** of paired questions from
  wrong to right against only 2.4% right-to-wrong, on 717 paired questions. Over the
  raw embedded text layer (T→TLV) it is **27.9%** wrong-to-right on 716 pairs.
- **And it is concentrated exactly where you would predict.** Split by evidence
  source, TL→TLV wrong-to-right is **32.1% for Chart** and 23.9% for Figure, but only
  **6.7% for Table** and 10.6% for Pure-text. The parser recovers tabular and textual
  content adequately and drops chart/figure content that the page image preserves.
  This is the "selected but unusable" statistic.
- **Integration degrades with hop, and worsens up the ladder.** `integration` shows
  the single-minus-multi gap widening from +12.1 points at T to +20.3 at TLV and +20.8
  at V. Multi-page evidence costs more, not less, on the richer representations.
- **Attribution, provisionally.** `attribution` charges +24.9 points to representation
  at T and +17.3 at TL (against the best rung TLV), +9.4 to retrieval at TLV, and
  leaves a +43.2 point reasoning residual.

### RQ1 gaps

| Gap | Why it is a gap | What fixes it |
| --- | --- | --- |
| **The reasoning residual is not a reasoning number.** | The +43.2 in `attribution` is the raw shortfall of the best oracle rung from 100%. It still contains judge false negatives and answers that are correct without matching the gold span. Neither correction exists in the data and neither has been netted out. | **No spec.** Two external analyses: a hand-check of judge false negatives on a sample, and a correct-without-gold partition. Until both, quote it only as a labelled upper bound. |
| **Retrieval loss is on a partial pool.** | `attribution`'s retrieval column reads the g2-retrieval-full generation pool: 4311 ok rows over **305 of 847 questions**, only the TLV and V rungs, with judging in flight when the cluster was wiped. TLV survives at k1 (paired n=254) and k3; joint k5 at TLV is 5 cells; all k10 OOMed. | `ops/specs/g2_retrieval_full_rerun.yaml` |
| **Retrieval loss on T and TL is unmeasured, not zero.** | G2 never ran the text-only rungs, so those cells are blank by construction. The blank is not evidence that retrieval costs nothing there. | `ops/specs/g2_retrieval_full_rerun.yaml` (add T/TL to `reasoner_representations` if that comparison is wanted; the current spec mirrors the original TLV/V grid). |
| **The memorisation channel is unmeasurable, not measured-and-null.** | Questions inherited from public datasets (ChartQA, DocVQA) may be memorised rather than read. `metadata.source_dataset` looked like the way to split this, but it is the loader's dataset id, hardcoded `"mmlongbench"` at `data/loader.py:170`. The staged parquet ships 7 columns and none names an upstream dataset; no annotation file carries it either. `source_stratification` reports the single stratum and says so. | **No spec can fix this.** It needs hand-labelling of document origin, or upstream annotations MMLongBench-Doc does not publish. |
| **Two evidence sources have thin paired n.** | In `fidelity_transition`: `(none)` is 17 paired questions (the 19 questions carrying an empty `evidence_sources` list), and `Generalized-text (Layout)` is 98. Chart (159), Table (150), Pure-text (255) and Figure (264) are sound. Do not quote the `(none)` row. | Partly `ops/specs/g1_failed_only_a100.yaml` (recovering OOM cells raises every paired n); the `(none)` bucket is an upstream annotation gap, not a compute gap. |
| **Faithfulness is measured on unanswerable questions only.** | `hallucination` and `abstention_by_doctype` read the G3 unanswerable pool. They show what an abstention instruction buys, never what it costs on questions that do have an answer. The false-abstention tax is invisible. | `ops/specs/g1_prompting_answerable.yaml` |

---

## RQ2 — Deployment feasibility (what can be run)

**Tables:** `scale`, `quantization`, `vram_headroom`, `oom_frontier`, `prefill_cost`.

These read complete, judged caches and are the most trustworthy block in the report.
`oom_frontier` reads `predictions.jsonl` rather than results, so it counts the failed
cells directly instead of inferring them from absence.

### RQ2 gaps

| Gap | Why it is a gap | What fixes it |
| --- | --- | --- |
| **The frontier is read through 3269 OOM cells.** | Those cells are the frontier in one sense (they are what did not fit on 16 GB), but they also mean the *accuracy* a representation buys is measured on the questions that survived, which are the shorter documents. Feasibility and accuracy are entangled in the current numbers. | `ops/specs/g1_failed_only_a100.yaml` — same axis values, larger card, so the accuracy side can be read without the survivorship filter. |
| **32B never ran to completion.** | The largest reasoner needs a card the cluster did not have (see the GPU-scope constraint), so the size sweep tops out below the interesting end. | `ops/specs/g1_failed_only_a100.yaml` on an A100/H100. |
| **Latency is not comparable end-to-end.** | Decode latency is inflated roughly 20x by the 256-token verbosity setting, and InternVL records only end-to-end (no prefill/decode split). Use `prefill_cost`, not `latency_s`. | Nothing to run; a reporting caveat. Already noted in `CODEBASE_GUIDE.md` Part B section 9. |

---

## RQ3 — Recoverable loss (which levers help)

**Tables:** `resolution`, `matched_cross` (provisional), `kdepth` (provisional),
`routing`.

`matched_cross` and `kdepth` both read the partial G2 generation pool and carry the
**PROVISIONAL (partial G2 pool)** stamp in their captions.

### RQ3 gaps

| Gap | Why it is a gap | What fixes it |
| --- | --- | --- |
| **There is no prompt-lever table.** | The prompt is one of the levers RQ3 is about, but the answerable pool has only ever run `prompt_modes: [none]`. The abstention-instruction effect can currently be read only on unanswerable questions, which cannot show the cost side. Until those cells exist, the RQ1 abstention tables stand in. | `ops/specs/g1_prompting_answerable.yaml` |
| **`generic` and `targeted` confound two instructions.** | `targeted` is `generic` plus the abstention sentence, so a difference between them mixes brevity with permission-to-abstain. RQ3 cannot say which lever moved the number. | `ops/specs/_target_prompt_concise_control.yaml` — **not runnable**, assumes a new `config.PROMPT_MODES` entry. Proposal only. |
| **Both retrieval levers are provisional.** | `matched_cross` (modality) and `kdepth` (depth) are the two retrieval levers, and both read the 36% pool. `kdepth` in particular is where OOM attrition is worst, so the deep-k cells are a survivorship-biased easy subset that would bias retrieved accuracy *upward* exactly where it should drop. | `ops/specs/g2_retrieval_full_rerun.yaml` |
| **The G2 inference arm is ambiguous in writing.** | The spec ran `bge-m3` as the text arm; `config.BASELINE` captions it `bm25` (the default the spec overrode). Both captions now say so rather than picking one. Cite the spec. | Nothing to run; resolve in the paper text. |
| **Integration is measured observationally, not causally.** | `integration` conditions on hop but never sets it, so the single-minus-multi gap cannot separate a genuine integration failure from multi-hop questions being drawn from a harder pool (and OOMing more). | `ops/specs/_target_hop_conditioned_selection.yaml` — **not runnable**, assumes a new `InputConditioner`. Proposal only. |

---

## Appendix tables (retained, not mapped to an RQ)

`retrieval_accuracy`, `retrieval_accuracy_overall`, `retrieval_dpi`, `scan_vs_digital`.

The retrieval benchmark side-artifact (page P/R/F1 per method) is **complete** — it is
only the retrieval-fed *generation* that is partial. So these three are trustworthy
even though the G2 generation tables are not. `scan_vs_digital` is complete too; it
sits here because the scan split is a corpus property rather than an RQ lever.

---

## The five specs, in the order worth running

1. **`g2_retrieval_full_rerun.yaml`** — unblocks the most: `attribution`'s retrieval
   column, `matched_cross`, and `kdepth`. Removes three PROVISIONAL stamps.
2. **`g1_prompting_answerable.yaml`** — cheapest real gain (2541 cells, fits a V100).
   Creates an RQ3 prompt-lever table that does not exist at all today.
3. **`g1_failed_only_a100.yaml`** — needs the larger card. Improves every table at
   once by removing the survivorship filter.
4. **`_target_prompt_concise_control.yaml`** and
   **`_target_hop_conditioned_selection.yaml`** — proposals, not runs. Each names the
   pipeline change and the frozen interface it would touch. Neither is implemented and
   neither should be run; they need approval first.
