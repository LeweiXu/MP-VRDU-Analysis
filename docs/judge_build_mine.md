# Task: judge everything, build the planned tables, mine for additional tables

Status: instruction for the coding agent. Three goals, in order: (1) run the judge
phase over all completed cells, (2) verify/implement the builders for the planned
result tables, (3) data-mine the collected telemetry for additional useful tables.

Authoritative context: `README.md` (experiment), `docs/AGENT_GUIDE.md` (code
structure, frozen contracts), `docs/DECISIONS.md` (history), `HANDOFF.md` (current
run state). Read `HANDOFF.md` first — it defines what is complete, what is pending,
and the known caveats this task must respect.

---

## 0. Ground truth about the data (read before doing anything)

The run is **~85% of cells complete on Kaya**, which is interpretable as-is. But it
is a **moving cache**: some reruns are still pending (per `HANDOFF.md`). This shapes
everything below.

Three facts that must condition the work:

1. **The cache is incomplete and will change.** InternVL3-8B was fixed (einops) but
   **not yet rerun** — the whole family arm may still be empty. An OOM/`--failed-only`
   sweep and the G2 H100 inference remainder are pending. Therefore judge+build is
   **not a one-shot**: it must be **idempotent and re-runnable**, and tables that
   depend on pending cells must be **marked blocked**, not silently built from
   partial data.

2. **Answers are ~35× longer than v3** (`max_tokens` 64→256; G1 uses the empty
   `none` prompt). Accuracy is mostly unaffected (gold is embedded in verbose text),
   but **decode latency and output-token counts are inflated ~20×**. Every
   cost/latency table — planned or mined — inherits this. Do **not** present decode
   latency, tokens/answer, or any cost-frontier derived from them as a clean
   finding without flagging the inflation. Prefill latency, VRAM, and input-token
   counts are unaffected and are the trustworthy cost signals.

3. **Binning shipped as native `doc_type`**, not the designed modality bins. The
   `doc_type` column is the 7 MMLongBench classes. `evidence_source` survives as the
   composition lens. Any "per bin" table means **per `doc_type`** unless it
   explicitly uses `evidence_source`.

---

## 1. Judge phase — run over everything that exists

Goal: every completed prediction cell gets a judge verdict, cheaply and
re-runnably.

- **Judge is GPU-free and prediction-cached.** Predictions are keyed without the
  judge; `ops.judge` scores `predictions.jsonl` directly and writes `results.jsonl`.
  Re-running the judge never re-runs the model. Use this: judge is safe to run now
  and re-run after each pending rerun lands.
- **Scope: judge all run_tags that have predictions**, one run at a time (Gemini
  Tier-1 rate limits; use the two-key fallback `GEMINI_API_KEY_SECONDARY`). Judge
  the scanned `*-scanned` run_tags as well as the digital ones — scan:digital ∪
  scan:scanned = scan:any is merged at build.
- **Only judge `status == ok` cells.** OOM/error rows carry no prediction; they must
  pass through to `results.jsonl` retaining their `status`/`skipped_reason` (a failed
  cell still writes one row), never be scored as wrong. Verify the judge path
  preserves failed rows rather than dropping or mis-scoring them.
- **Idempotence check.** Re-running judge on an already-judged run must be a no-op
  for unchanged cells (result cache keyed with `judge_spec`). Confirm this holds so
  the post-rerun re-judge only scores the newly completed cells.
- **Record coverage.** After judging, emit a short per-run coverage line: cells
  judged / ok / oom / err, and the answerable/unanswerable split. This is how you
  and the user see what is real before building tables on it.

Deliverable: `results.jsonl` for every run_tag with predictions; a coverage summary
printed (not a new doc).

---

## 2. Build the planned tables — verify first, implement the gaps

The planned tables are enumerated in the two results writeups already in the repo
(`G1_Representation.md`, `G2_Retrieval.md`) and the reporting builders in
`reporting/tables/`. Goal: every planned table builds correctly from the current
cache, or is explicitly marked blocked.

### 2a. Verify the existing builders against the shipped writeups
For each table in `G1_Representation.md` and `G2_Retrieval.md`, confirm a builder
exists and reproduces that table's shape from the cache:

- **G1:** headline (cost-ordered ladder × doc_type, oracle), parser comparison,
  resolution sweep, scale (accuracy vs VRAM/latency across specs), composition
  (accuracy × evidence_source × rung), routing (build-time over G1 + G3 classifier
  price).
- **G2:** retrieval accuracy overall (P/R/F1 × method × k), retrieval accuracy ×
  doc_type, retrieval DPI, and — when the H100 inference lands — matched-vs-cross
  and the k-depth inference tables.

For each: does the builder exist, does it read the right source run_tags, does it
group by `doc_type` (not a stale `bin`), and does it handle missing cells by
**omitting with an n** rather than counting them wrong?

### 2b. Table-by-table status, not a single build
Produce a **build-status manifest** — one row per planned table: `built` /
`blocked (reason)` / `partial (which cells missing)`. Specifically:

- **InternVL family-replication table → blocked** until the InternVL rerun lands.
  Do not build it from the currently-empty arm and present a blank/garbage row.
- **G2 matched-vs-cross + k-depth inference → blocked** on the H100 remainder;
  the retrieval **accuracy** side-artifact tables are **buildable now** (the
  benchmark is complete).
- **Resolution table** — check the 488 CUDA-fault cells; build with the survivors
  and report the reduced n, or mark partial.
- Everything with a complete cache → **build now**.

### 2c. Correctness rules for every builder
- **Group by `doc_type`** (7 classes) unless the table is explicitly
  evidence_source-based.
- **Reuse, never recompute, cached accuracy/cost.** Routing selects existing G1
  rows; it must not re-derive correctness.
- **Doc-level bootstrap CIs** stay doc-coherent (resample `doc_id`s).
- **A table is only correct when handed exactly its source run_tags' rows** — verify
  the routing registry hands each builder its intended sources (a builder that
  doesn't filter by `model_spec` will silently pool a sweep if given extra tags).
- **Cost columns carry a caveat flag** where they include decode latency / output
  tokens (§0.2).
- Missing cells → **omit and report n**, never impute or count as wrong.

Deliverable: all buildable tables built under their run tags; a build-status
manifest listing built / blocked / partial with reasons.

---

## 3. Data-mine for additional tables — hypothesis-driven, not fishing

The telemetry was deliberately over-collected. Mine it, but every candidate table
must clear a bar (§3c) or be discarded. Do **not** emit dozens of marginal
cross-tabs.

### 3a. The seams worth mining (these have a specific question behind them)
Each is a question the collected data can answer that isn't already a planned table:

- **Truncation incidence** — `tokens_dropped` / truncation flags should be ~zero
  (the cap was removed for the reasoner). Mine where it *isn't* zero (the
  qwen3-embedding retrieval cap, per `HANDOFF.md`). Table: where does truncation
  still occur, and does it correlate with a retrieval-accuracy drop? This validates
  (or complicates) the cap-removal story.
- **Scan vs digital** — the scanned companion runs exist. Table: ladder accuracy
  by rung for scanned vs digital, per doc_type. Hypothesis: `T` collapses on scans
  (embedded text is empty by design) so the frontier shifts toward `TLV`/`V`. This
  is a genuine deployment finding and is on-thesis.
- **Prefill cost vs representation** — prefill latency and input tokens are
  *uncontaminated* by the verbose-answer issue. Table: prefill cost per rung per
  doc_type — the honest "what does each representation cost to ingest." This is a
  cleaner cost story than the decode-inflated latency and may be a better headline
  cost axis.
- **VRAM headroom** — `peak_vram_bytes` per rung / resolution / model spec. Table:
  how close each config runs to the 16 GB V100 ceiling → predicts what fits on
  accessible hardware. Directly serves the deployment lens.
- **Quantization sensitivity** — 4/8/16-bit rows exist. Table: accuracy delta and
  VRAM delta per quant level per doc_type. Cost-frontier framing (accuracy-per-VRAM).
- **Retrieval evidence-survival vs downstream accuracy** — join the retrieval
  side-artifact (did the gold page get retrieved) to G2 inference accuracy (did the
  answer come out right). Table: conditional accuracy given the gold page was / was
  not in the top-k. Separates retrieval failure from reasoning failure. (Blocked on
  H100 inference, but define it now.)
- **Abstention behaviour (G3)** — the three prompt modes over the unanswerable set.
  Table: abstention rate × prompt mode × doc_type. The targeted-prompt effect is
  already visible in smoke; quantify it at full scale.
- **OOM frontier** — where cells OOM'd on V100 (status rows). Table: OOM rate by
  rung × resolution × pages-fed. This is the empirical "what doesn't fit" map, and
  it's a deployment finding hiding in the failure rows.

### 3b. Explicitly de-prioritise (contaminated or low-value)
- Any table whose primary axis is **decode latency, total latency, or
  output-tokens/answer** — contaminated by the verbose-answer issue (§0.2). If mined
  at all, flag heavily; do not headline.
- Marginal cross-tabs with **tiny per-cell n** (the doc_type × rung × resolution ×
  quant cube gets thin fast) — report only where n supports a CI.

### 3c. The bar a mined table must clear
Keep a candidate only if all hold: (i) it answers a question a reader of a
deployment-framed MP-VRDU paper would ask; (ii) the n per cell supports a bootstrap
CI (don't ship a 3-sample cell); (iii) it isn't primarily driven by the
verbose-answer contamination; (iv) it isn't already a planned table. Everything else
is discarded, not shipped "just in case."

### 3d. Output
For surviving candidates: build the table under a clearly-named `mined_*` builder,
and add a one-line entry to a **candidates summary** (`docs/generated/mined_tables.md`
or similar) stating the question it answers, the n, and any caveat. The user selects
which graduate to the paper; the agent does not decide inclusion.

Deliverable: mined tables built as `mined_*`, plus a candidates summary with
question / n / caveat per table.

---

## 4. Order of operations

1. **Judge everything with predictions now** (§1) — cheap, unblocks all accuracy
   tables, re-runnable after reruns land.
2. **Verify + build the planned tables that are complete** (§2), emit the
   build-status manifest marking blocked/partial ones.
3. **Mine** (§3) over the same judged cache; build surviving `mined_*` tables +
   candidates summary.
4. **After the pending reruns land** (InternVL, OOM sweep, H100 G2 inference):
   re-run judge (scores only the new cells), then rebuild the previously-blocked
   tables. The idempotence from §1/§2 makes this a clean top-up, not a redo.

---

## 5. Acceptance checks

- Every run_tag with predictions has a `results.jsonl`; failed rows preserved with
  status, not mis-scored.
- Re-running judge on an already-judged run is a no-op (idempotent).
- A build-status manifest exists: every planned table is built / blocked / partial
  with a reason; no blocked table shipped from empty/partial data as if complete.
- All tables group by `doc_type` (or explicitly by `evidence_source`); routing
  reuses G1 rows without recomputation; CIs are doc-level.
- Cost tables that include decode latency / output tokens carry the verbose-answer
  caveat.
- Mined tables are `mined_*`, each justified in the candidates summary with n and
  caveat; contaminated/thin candidates discarded, not shipped.
- Nothing in this task requires the GPU except the already-planned reruns
  (judge/build/mine are all cache-side).

---

## 6. What NOT to do

- Do not build the InternVL family table (or any blocked table) from partial data
  and present it as complete.
- Do not headline any decode-latency / total-latency / tokens-per-answer finding
  without the verbose-answer caveat.
- Do not impute, drop, or count-as-wrong the OOM/error cells — omit with an n.
- Do not ship marginal mined tables to hit a count; the bar (§3c) is the filter.
- Do not change frozen contracts, cache keys, or re-run the reasoner to "fix" data
  — this task is judge + build + mine over the cache as it stands.