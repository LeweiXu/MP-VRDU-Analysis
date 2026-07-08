# Pivot v4 — implementation plan (clean-slate do-over)

Status: instructions for the coding agent building v4. This is a **do-over**, not
an incremental refactor: v3 is parked as read-only reference, the target tree is
scaffolded fresh, tests are rewritten, and code is built bottom-up. The plan fixes
the **target structure**, the **organising paradigm**, and **rules**; it delegates
line-level decisions to the agent, which can see the actual code. When the plan and
the code disagree on something unforeseen, follow the paradigm (§1) and record the
deviation (§10).

Read alongside `PIVOT_v4.md` (the science decision record — authoritative on
what/why) and the v3 `README.md` / `AGENT_GUIDE.md` (now reference-only, parked in
`old/`). Where a v3 mechanism already does the right thing (loading discipline,
per-cell skip, doc-coherent sampling, caching), v4 **lifts and adapts** it — the
plan says so rather than pretending to build from nothing.

---

## 0. Why a do-over (and the one risk it carries)

The v3→v4 delta is deep: the ladder mechanics change, binning changes source, the
task set is reshaped, the input cap is removed, the machine split dissolves, the
env strategy changes, and the tree is substantially different. An incremental move
would spend most of its effort holding v3 scaffolding up while rewiring
underneath — which is exactly how stale v3 assumptions survive unnoticed into v4. A
clean build with v3 parked as reference removes that.

**The risk a do-over carries:** there is no "behaviour-identical checkpoint." The
repo goes from "v3 works" to "v4 built from scratch" with a valley where nothing
runs end to end. The mitigations are (a) v3 lives untouched in `old/` for reference
throughout, (b) v3 output files are preserved as **structural fixtures** for
plumbing tests (§ Phase 0), and (c) v4 is built **bottom-up, one green layer at a
time** (§ Phase 4) so breakage is always localised to the layer in hand.

Note on the fixtures: v4's *values* will not match v3's (different ladder, binning,
parser, no cap), so the old results are **not** a correctness oracle. They are a
**format/shape reference** — real jsonl rows to test v4's readers, mergers,
`--failed-only` selection, and build-step cardinality against. Plumbing correctness,
not science correctness.

---

## 1. Organising paradigm (the rule the structure encodes)

The repo implements *a representation-controlled document-understanding pipeline,
run under many configurations.* Two placement rules:

1. **Name by what a thing IS**, not by what it does for you or when it was added.
   `covariates/` → `retrievers/` + `models/classifier.py`; `local_vlm.py` →
   `qwen3vl.py` (symmetric with `internvl.py`); `tools/layout.py` → `tools/parser.py`.
   Generation tasks keep **`G[num]_[name]`** where the number is a stable handle and
   the name states the **mechanism** (`G1_oracle_ladder`, `G2_retrieval`,
   `G3_hallucination`, `G4_classifier_pricing`) — never an RQ or table number.

2. **Separate the science spine from the plumbing.** Spine stays flat at root and
   prominent: `data/ tools/ retrievers/ models/ pipeline/ scoring/ reporting/`.
   Operational tooling groups under `ops/`. `pipeline/` keeps its **narrow** meaning
   — the five frozen stages of one cell — and never becomes a parent for the whole
   codebase.

A file's home is decided by **what it is**, not **who imports it**.

---

## 2. Target file structure

Authoritative on *shape*; the agent may add/split leaf files where the code
demands, obeying §1. Comments double as the generic docstring target.

```
mpvrdu/
├── config.py                     run knobs: paths, resolution presets (NO input-token cap)
├── schema.py                     frozen data contracts + telemetry fields
│
├── data/                         dataset layer
│   ├── loader.py                 dataset rows → Question; answerable/unanswerable split
│   ├── binning.py                bin_label lookup from manual annotations
│   ├── annotations.py            read/validate the human doc-label table
│   └── render.py                 PDF → per-page PNG + embedded-text spans
│
├── tools/                        per-page channel builders
│   ├── text.py                   cheap embedded-text extraction (T channel)
│   ├── parser.py                 PDF-parser layout-rich text for TL/TLV
│   └── visual.py                 page-image channel + vision-token estimation (resolution)
│
├── retrievers/                   page retrievers (cost rungs)
│   ├── text.py                   BM25 / BGE-M3 / Qwen3-Embedding
│   ├── vision.py                 ColModernVBERT / ColQwen2.5 / ColQwen3
│   └── joint.py                  deduplicated union of a text + a vision page set
│
├── models/                       reasoner backends + classifier, behind registries
│   ├── __init__.py               get_reasoner registry (name → backend)
│   ├── qwen3vl.py                Qwen3-VL backend (was local_vlm.py)
│   ├── internvl.py               InternVL backend
│   ├── classifier.py             doc-type/bin classifier
│   └── payload.py                backend-neutral prompt/image container
│
├── pipeline/                     the five frozen stages of ONE cell (narrow meaning)
│   ├── conditioner.py            page selection (oracle / retrieved / similarity)
│   ├── representation.py         T/TL/TLV/V composer (cost-ordered; parser text; no bbox)
│   ├── reasoner.py               Reasoner ABC
│   ├── judge.py                  scoring interface + API judges
│   └── orchestrator.py           composes stages; owns cache layers + telemetry capture
│
├── scoring/                      cached cells → numbers (was metrics/ + live gate math)
│   ├── accuracy.py               doc-level accuracy + bootstrap CIs
│   ├── cost.py                   token / latency / VRAM aggregation
│   ├── frontier.py               sufficiency-frontier rule
│   ├── retrieval.py              page precision / recall / F1
│   ├── abstention.py             abstention detection
│   └── agreement.py              judge–human κ (surviving computation from retired F2)
│
├── experiments/                  what runs / how / on what
│   ├── tasks/                    G[num]_[name] generation tasks
│   │   ├── base.py               GenerationTask ABC + shared cell factories
│   │   ├── G1_oracle_ladder.py   oracle × {T,TL,TLV,V}; base grid for RQ1 sweeps
│   │   ├── G2_retrieval.py       retrieved × TLV/V × method × k
│   │   ├── G3_hallucination.py   unanswerable × similarity pages × prompt
│   │   └── G4_classifier_pricing.py  side-only classifier pricing (no reasoner cells)
│   ├── engine/                   run machinery
│   │   ├── driver.py             generate+judge loop; telemetry; retry/--failed-only
│   │   ├── side_artifacts.py     shared side-artifact writers
│   │   ├── artifacts.py          artifact-driven judge/build helpers
│   │   └── paths.py              cache/table path layout
│   ├── corpus/                   what-to-run-on resolution
│   │   ├── resolve.py            question-set resolver + sampling (was corpus.py)
│   │   ├── smoke.py              reproducible --smoke subset
│   │   └── yaml_spec.py          YAML spec → dynamic tasks (incl. corpus scope)
│   └── registry.py               task-name → task collection
│
├── reporting/                    judged rows → tables
│   ├── build.py                  task→table routing + build-time routing assembly + CSV/md
│   └── tables/                   content-named builders (no T#)
│       ├── _common.py            shared helpers (bin order, rep order, telemetry cols, filters)
│       ├── _markdown.py          markdown rendering
│       ├── headline.py           cost-ordered ladder × bin
│       ├── parser.py             parser comparison
│       ├── resolution.py         resolution sweep (scientific)
│       ├── matched_cross.py      retrieval matched-vs-cross
│       ├── kdepth.py             top-k sweep (+ joint condition)
│       ├── retrieval_accuracy.py page-F1 per bin/method
│       ├── hallucination.py      abstention × prompt
│       ├── routing.py            routing policies (build-time; reuses G1 + G4 price)
│       ├── scale.py              size / quantization frontier
│       └── composition.py        evidence-source composition (appendix)
│
├── ops/                          entry points + operational tooling (flat)
│   ├── generate.py               GPU generation from YAML (was cli/generate.py)
│   ├── judge.py                  judging entry point (was cli/judge.py)
│   ├── build.py                  table-build entry point (was cli/build.py)
│   ├── kaya/                     SLURM sync/submit runner + config + guides
│   ├── specs/                    YAML specs (template + saved run configs)
│   └── scripts/                  standalone utilities (see Phase 2 copy-list)
│
├── docs/                         authored prose only
│   ├── PROJECT_SPEC.md  USER_GUIDE.md  AGENT_GUIDE.md
│   ├── DECISIONS.md              pivot changelog + Phase-1 probe/env verdicts
│   ├── REPO_STRUCTURE.md         tree + auto-generated per-file map
│   └── generated/                script outputs (dataset_stats, label dists, all_tables, questions)
│
├── tests/                        v4 pytest suite (rewritten from scratch)
├── old/            [reference]   untouched v3 snapshot; DELETED when v4 is green
└── README.md  CLAUDE.md  requirements*.txt  __init__.py
```

Gone vs v3: `covariates/` (dissolved), `gates/` (retired — §Phase 4 note), `cli/`
(→ `ops/` root), `metrics/` (→ `scoring/`).

---

## PHASE 0 — Capture (before touching anything)

Cheap, first, non-destructive.

- **Preserve v3 outputs as structural fixtures.** Copy the existing G1/G2/G3/G5/G6
  results (the 300-question-subset predictions/results/side-artifacts) into a
  reference location (`tests/fixtures/v3_results/` or under `old/`). Mark them
  clearly: **v3-shaped, values NOT comparable to v4** — usable only to test row
  shape, field names, cache-key layout, jsonl parsing, `--failed-only` selection,
  and build-step cardinality.
- **Record the current git commit** of v3 in `DECISIONS.md` as the reference point.

Deliverable: fixtures in place + a one-line `DECISIONS.md` entry. No code changes.

---

## PHASE 1 — Probes & decisions (these constrain everything downstream)

Both outputs are written to `docs/DECISIONS.md` before any building.

### 1a. Resolution probe
- Write `ops/scripts/resolution_probe.py` and submit it to Kaya.
- Probe on the **V rung** (image-only, **parser-independent**) at the worst case
  (~10 pages): vision tokens are the binding VRAM constraint now that text is
  uncapped-but-bounded, so a V-based probe is the honest floor and doesn't depend
  on the not-yet-built v4 parser path.
- Report the **highest resolution preset that fits 16 GB without OOM**. That preset
  becomes the single fixed **deployment resolution** used by every table except the
  scientific resolution sweep.
- Mark it **re-runnable**: if the Phase-4 TLV path (parser markdown) shifts the
  sequence profile materially, re-run to confirm.

### 1b. Environment / dependency decision
- **Evaluate dropping vLLM.** It is the dominant v3 constraint (exact torch pin,
  `openai<=1.90` cap). v4 reasoning is batch-1, latency-measured — if plain HF
  `transformers` generate suffices, **drop vLLM** and relax the whole core env.
  Verify nothing in v4 still needs vLLM serving.
- **Decide the env partition.** Because the parser↔reasoner boundary is disk
  (Phase 4), parsers need not share the reasoner's env. Target: a **core reasoning
  env** (Qwen3-VL + InternVL + retrievers + judge plumbing) + **one isolated env per
  parser that will not co-exist** (PaddleOCR-VL / MinerU 2.5 / Unlimited OCR);
  parsers that happen to co-exist share one `parse` env. Keep the local-Blackwell
  env.
- Strip the dropped stacks (Marker/Surya/Docling) and any pins that existed *only*
  for them; re-examine whether `pillow<11`, the exact torch pin, and the
  transformers 4.57 ceiling can now relax.
- **`pip check` clean per env** is the bar. Decide empirically by attempting
  installs — cannot be determined statically.

Deliverable: chosen deployment resolution + env partition + vLLM verdict, all in
`DECISIONS.md`. Requirements files may be drafted here but are finalised in Phase 4.

---

## PHASE 2 — Park & scaffold (no deletion of logic yet)

Order matters: **park first, scaffold second, delete nothing applicable until v4 is
green.** Deciding what's "no longer applicable" before v4 exists is guessing.

1. **Move the entire v3 tree into `old/`, untouched.** It stays importable-for-
   reference and wired into nothing. `old/` is reference-only and is **deleted in a
   final commit once v4 is fully green**.
2. **Direct-copy the genuinely-unchanged files** straight to their v4 homes
   (bypassing `old/`), each confirmed by the agent to still apply:
   - **Clean copies (expected):** `ops/kaya/kaya.py` + Kaya guides;
     `ops/scripts/`: `download_hf.py`, `gpu_test.py`, `kaya_status.py`,
     `setup_env.py`, `dataset_stats.py`, `profile_datasets.py`, `dump_docstrings.py`;
     most of `docs/` (`PROJECT_SPEC.md`, `USER_GUIDE.md`, `AGENT_GUIDE.md` — then
     reconciled to v4 in Phase 4), `scripts/split_docs_by_type.py` if still used for
     annotation browsing.
   - **Looks copyable but needs rework (do NOT blind-copy):** `annotate_docs.py`
     (now writes the three-label schema: `bin_label`/`scan_label`/`dominant_visual`);
     `prestage.py` (now stages the core env + isolated parser envs); `inspect_results.py`
     (absorbs the retired `gates/viewer.py`); `run_probe.py` (v4 feasibility set).
   Mark each copy in `DECISIONS.md` as clean-copy or copied-pending-rework.
3. **Scaffold the new empty tree** (§2): create the dirs and empty/stub modules with
   their generic docstrings, so imports resolve and tests can target real paths.

Deliverable: `old/` = full v3 snapshot; direct-copy set in place; empty v4 tree
scaffolded; nothing deleted.

---

## PHASE 3 — Tests first (code correctness, not science)

Tests here verify **importability and plumbing correctness** — never science
outputs (you can't assert a frontier before the pipeline exists). Delete all v3
tests (they encode v3 structure) and write v4 tests against the scaffolded stubs;
they run red until Phase 4 fills them in.

Write these as executable specs of the invariants already fixed in `PIVOT_v4.md`:

- **Cell robustness:** a task over N cells emits **exactly N rows** regardless of
  outcome; a failed cell writes a row with `status ∈ {oom,error}` + `skipped_reason`,
  never omitted.
- **`--failed-only`:** selects exactly the `status != ok` rows from a fixture jsonl
  and upgrades them in place; leaves `ok` rows untouched.
- **Machine-independent keying:** identical inputs produce identical SHA-256 cell
  keys and identical resolved cell lists under simulated different environments
  (no device-count / hostname / `torch.cuda` leak).
- **Truncation canary:** with no cap, `text_tokens_fed == total_text_tokens` and
  `tokens_dropped == 0` on every cell.
- **Representation:** the composer builds the four rungs, TL/TLV use parser text,
  and **no bbox JSON** is emitted; parser and reasoner are never both resident
  (mockable via load/free call-order assertions).
- **Corpus scope (new, §Phase 4):** each sampling mode resolves the expected
  question set; sampling is **document-coherent**; RQ1/RQ2 specs never draw from the
  unanswerable pool and RQ3 only from it.
- **I/O plumbing on the Phase-0 fixtures:** the jsonl reader parses real v3-shaped
  rows; the build step groups them at the right cardinality; side-artifact readers
  parse the real retrieval/classifier artifacts. (Shape only — values not asserted.)
- **YAML spec loading:** a spec resolves to the correct task + cell grid + corpus
  scope; a `supervisor`-only concept does **not** exist (there is no machine field).
- **Registry/import:** every task, backend, retriever, parser, and table builder is
  importable and discoverable via its registry.

Deliverable: v3 tests deleted; v4 contract/invariant/import/plumbing tests written,
red against stubs.

---

## PHASE 4 — Build bottom-up (one green layer at a time)

Port-and-adapt from `old/` in **dependency order**, turning each layer's tests green
before the next. Each lift from `old/` is a conscious "copy this, adapt to v4,"
never a bulk move. Order:

`schema` → `data` → `tools` / `retrievers` → `models` → `pipeline` stages →
`orchestrator` / `engine` → `experiments/tasks` → `scoring` → `reporting` → `ops`
entry points.

Per-layer science + mechanism requirements (authority is `PIVOT_v4.md`):

- **schema:** telemetry fields — per-cell (`status`, `skipped_reason`,
  `total_text_tokens`, `total_visual_tokens`, `text_tokens_fed`, `output_tokens`,
  prefill/decode latency split, `peak_vram_bytes`, `bin_label`, `scan_label`,
  `machine` as provenance), per-run manifest, retrieval side-artifact. Truncation
  fields kept as a **canary** (must read zero; do not delete).
- **data:** manual-annotation binning (retire `doc_type` bins); answerable/
  unanswerable split in the loader; three-label annotation schema.
- **tools:** `text.py` = cheap embedded text; `parser.py` = PDF-parser markdown for
  TL/TLV, **no bbox**; `visual.py` carries resolution (fixed deployment preset +
  sweep presets).
- **retrievers:** three text + three vision cost rungs + joint union; each emits a
  page set to cache; the retrieval side-artifact scores page-F1 per bin/method for
  all methods incl. non-inference ones.
- **models:** `qwen3vl.py` (renamed), `internvl.py`, `classifier.py`; `get_reasoner`
  registry; quantization as a spec suffix; **evaluate/land the vLLM-drop** from
  Phase 1.
- **pipeline:** cost-ordered (non-cumulative) ladder; conditioner adds the
  similarity page-selection for hallucination; orchestrator captures telemetry and
  owns the two cache layers.
- **engine (model lifecycle — generalise v3, don't rebuild):** parser boundary is
  the **disk cache**, not a Python import (in-process or subprocess into an isolated
  parser env); **one engine load per run**, unloaded **before** any reasoner loads;
  **strict non-co-residence** of parser/retriever/reasoner; lazy reasoner load/free
  between specs.
- **engine (robustness = the machine split):** one row per cell always; cell-failure
  (row with status) vs task-failure (missing **completion marker**); a
  **systemic-failure abort** (agent sets threshold — guideline: trip on ~15–20
  consecutive failures or >50% early, configurable; the point is only to distinguish
  sporadic OOM from a broken run); **`--failed-only`** re-run upgrades failed rows in
  place — this *is* the Kaya→supervisor handoff (code on GitHub, cache handed over
  manually, completed cache returns for local judge/build; **no sync tooling**).
- **corpus / YAML scope (new requirement):** a spec declares its corpus
  declaratively via a `corpus:` block:
  - `sampling: full` — all 1091 questions (the stretch goal; also the real
    stress-test of cap-removal + retry + parser-env isolation at scale).
  - `sampling: {per_bin: N, seed: S}` — draw **whole documents** per bin to N
    (default research subset; preserves doc-coherent sampling for the doc-level
    bootstrap CIs — do **not** simplify to per-question sampling).
  - `sampling: {limit: N}` or an explicit id list — smoke/debug fast path.
  - The **answerable pool is bound by the task**, not the spec: G1/G2 sample from
    answerable (~841), G3 from unanswerable (~250); sampling happens within that
    pool so a spec can't cross-contaminate.
- **scoring:** doc-level accuracy + bootstrap CI, cost/frontier, retrieval P/R/F1,
  abstention; **relocate the surviving gate math** here (`frontier.py`,
  `agreement.py` for κ) and drop the F1/F2/F3 CLI/threshold scaffolding; scoring
  **skips `status != ok` rows** (so cells that OOM even on the H100 need no
  exclusion list).
- **reporting:** content-named table builders; **explicit task→table routing** in
  `build.py` (one task → many tables); routing table assembled at **build time**
  reusing G1 rows + the G4 classifier price.
- **ops:** the three role entry points at `ops/` root; rework the copied-pending
  scripts (`annotate_docs`, `prestage`, `inspect_results`, `run_probe`).

**Cache invalidation happens once, here.** Bump the cache namespace/version at the
start of Phase 4 so v3 and v4 cells never co-mingle.

**Final cleanup (only when v4 is fully green):** delete `old/`; remove any genuinely
dead code; regenerate `REPO_STRUCTURE.md`'s file map; reconcile `PROJECT_SPEC.md` /
`README.md` / `AGENT_GUIDE.md` to v4 and fold `PIVOT_v4.md` into `DECISIONS.md`.

---

## 5. Docstring rules (applied as each module is written in Phase 4)

- Generic description of the file's **current function** — 1–3 sentences.
- **No plan-talk:** no "v4 should", "legacy", "will change to", RQ/table numbers, or
  pivot/roadmap references. Described behaviour is current behaviour.
- `REPO_STRUCTURE.md`'s per-file section is auto-generated by `dump_docstrings.py` —
  fix the `.py` docstring and regenerate; never hand-edit the generated section.
- Add to `CLAUDE.md`: *"Module docstrings describe current function only — no
  roadmap, pivot, or RQ/table references."*

---

## 6. Frozen contracts to honour (from v3, still binding)

Even in a do-over these stay stable so the two-layer cache and role split keep
working: the `schema.py` data contracts (extended additively for v4 telemetry, not
reshaped); `ModelInput` + its conversions; `Reasoner.answer`; `Judge.score`;
`Retriever.retrieve`; the orchestrator cell-key composition; the two-cache design
(prediction cache without judge_spec; result cache with it). Machine-independence
(§Phase 4) is an **additional** v4 constraint on the key. Changing a frozen contract
is a recorded decision in `DECISIONS.md`, not a silent edit.

---

## 7. Model lifecycle invariants (restated for emphasis)

1. Parser output crosses to the reasoner **only via the disk cache**.
2. A parser/retriever engine loads **at most once per run**, reused across all
   pages/docs, unloaded **before** any reasoner loads.
3. Parser, retriever, reasoner **never share VRAM**.
4. Reasoner weights load per model-spec and are **freed between specs**.

---

## 8. Robustness invariants (restated for emphasis)

1. **Exactly one row per cell**, always; failures recorded with `status` +
   `skipped_reason`, never omitted.
2. **Cell failure ≠ task failure**: task completion is proven by a completion
   marker; a manifest without one is a task-level failure.
3. **Systemic failure aborts loudly** (agent-set, configurable threshold).
4. **`--failed-only`** re-runs only failed rows and upgrades them in place — this is
   the entire machine-split implementation.
5. **Machine-independent keying** so a supervisor re-run completes the *same* file.
6. Failed rows are **retried** on a later run; `ok` rows are **never** re-run; the
   final jsonl converges to complete. Scoring skips rows that never succeed.

---

## 9. Cap, resolution, env (restated for emphasis)

- **Input-token cap removed entirely.** Full-sequence runs; overflow handled by the
  retry. Truncation telemetry kept as a zero-should-hold **canary**, not dead code.
- **Resolution** is the one cross-machine invariant: a single fixed deployment preset
  (from the Phase-1 probe) everywhere, except the scientific resolution sweep which
  deliberately varies it.
- **Envs:** as few as `pip check` cleanly allows; parsers isolated behind the disk
  boundary; vLLM dropped if v4 doesn't need it.

---

## 10. Deviation & decision log

The agent will hit unforeseen calls (a file fitting two homes, an env partition that
won't `pip check`, a copied script needing more rework than expected). Rule: follow
the paradigm (§1) and the invariants (§6–§9); when a real judgement is made, record
one line in `docs/DECISIONS.md` (what, why, what it affected). Do not delete anything
whose caller you have not confirmed, and do not delete `old/` until v4 is fully
green. When blocked between two paradigm-consistent options, prefer the one that
keeps the suite green, and flag for human review rather than guessing on the science.

---

## 11. Acceptance checks per phase

- **Phase 0:** v3 fixtures preserved and labelled non-comparable; reference commit
  recorded.
- **Phase 1:** deployment resolution chosen; env partition + vLLM verdict in
  `DECISIONS.md`; probe script submitted and returned a preset.
- **Phase 2:** `old/` holds an untouched v3 snapshot; direct-copy set placed and
  each marked clean vs pending-rework; empty v4 tree scaffolded; imports resolve.
- **Phase 3:** v3 tests deleted; v4 invariant/plumbing tests written and red against
  stubs; fixtures parsed by the I/O tests.
- **Phase 4:** every layer's tests green bottom-up; cap gone and canary reads zero;
  one row per cell with `status`; `--failed-only` upgrades in place; keys verified
  machine-independent; YAML `corpus:` scope supports `full` / `per_bin` / `limit`
  with doc-coherent sampling and task-bound answerable pool; four `G[num]_[name]`
  tasks discoverable; sweeps run as YAML variants; routing assembled at build time;
  `pip check` clean per env; cache namespace bumped.
- **Final:** `old/` deleted; dead code removed; file map regenerated; v4 docs
  reconciled; `PIVOT_v4.md` folded into `DECISIONS.md`.

---

## 12. Explicitly out of scope

- No incremental "move and adapt in place" — this is a scaffold-fresh do-over.
- No new numbering scheme for tables (content names); tasks keep `G[num]_[name]`.
- No `pipeline/`-as-parent nesting of the science spine.
- No hand-editing of generated docs.
- No separate machine-split / sync implementation — it is `--failed-only` + a manual
  folder handoff.
- No deletion of `old/` or dead code until v4 is fully green.
- No science-metric changes beyond `PIVOT_v4.md` (CIs, margins, κ bar unchanged).