# Multi-Page VRDU Error Attribution — Paper Working Repo
 
This repo is for **writing the paper**, not running the pipeline. Most core data is
collected; some experiments are planned or in consideration. This README is the single
source of truth for what the paper argues, how the argument is structured, what data
backs each claim, what is still to run, and which prior work we sit against. **Read this
first in any new chat** — it is written to be the only context needed to continue
planning.
 
**Target:** 8-page ACL-style venue (two-column). Space is tight. Page budget as agreed:
~1 page intro + related work, ~1 page failure modes (§3), ~1 page methodology (§4),
leaving ~5 pages for results and analysis. **~600 words is a full page of body text.**
 
Companion files (do not duplicate their content here):
- `CODEBASE_GUIDE.md` — the pipeline, result-row schema, cache layout, table-build
  system, and current data state. Authoritative on all implementation/operational detail.
- `AGENT_GUIDE.md` — frozen interfaces, swap points, caching contract for coding agents.
- `all_tables.md` — the full result tables (source of every number).
- `template.yaml` — the spec format: how any new experiment is declared in `ops/specs/`.
- `custom.bib` — bibliography, and the source of truth for keys.
- `RW.txt` — related-work reference file: which papers are genuine threats, what each
  owns, what differentiates us, and what was deliberately cut. Read before touching §2.
- `papers/` — markdown of the papers in `RW.txt`.
- `tables/` — the per-experiment LaTeX table files, `\input`-ed by the draft.
- `draft/` — the paper draft (`acl_latex.tex`, single file).
---
 
## 1. Thesis (the spine)
 
**MP-VRDU error is distributed across the evidence-management pipeline rather than
localised to any single stage the field optimises.** Standard evaluation reports one
answer-accuracy number that collapses a chain of distinct, separable failures. We hold a
retrieve-then-generate pipeline fixed one stage at a time, attribute error to the stage
where information is lost, and then ask which of those losses move when a practitioner
intervenes.
 
**The paper is two movements, not five parallel questions.**
 
1. **Locate the loss.** Where in the pipeline does MP-VRDU discard the answer, and why.
2. **Move the loss.** What settings a practitioner can vary, and what each one does.
Movement 1 earns the right to make movement 2; movement 2 is why movement 1 was worth
doing. Movement 1 is partly confirmatory in outline but corrective in detail: the field
assumes a single ordering and a single bottleneck, and we find the ordering holds only
in aggregate while the bottleneck relocates rather than shrinking. Movement 2 is where
the contribution concentrates.
 
**Generalisation guard (state in §4.1 and the intro):** we instrument a
retriever-generator pipeline because its stages *expose* the evidence-management
decisions every MP-VRDU system makes. A real retriever's rankings enter only as an
instrument to source realistic evidence drops and distractors, not as the object of
study. So the losses characterise evidence management **in general**, not retrieval in
particular — this is what stops a reviewer scoping the paper to "RAG-VRDU only".
 
---
 
## 2. The failure-mode taxonomy (E1–E5)
 
**Superseded: the five-RQ structure is gone.** Per supervisor feedback, five co-equal
RQs over-partitioned the paper and left it with a map but no argument. The RQs survive
as subsections of §5 and §6, not as research questions.
 
§3 defines five failure modes by **what happens to the evidence**, not by which
component was at fault. Each mode presumes the ones before it were satisfied, so the
modes are mutually exclusive and form a **chain**. E5 branches off the chain rather than
extending it.
 
| Mode | Name | Condition |
|---|---|---|
| E1 | Acquisition | Answer-bearing evidence never enters the pipeline in usable form |
| E2 | Selection | Evidence entered but does not reach the reasoner, or is diluted by non-evidence |
| E3 | Fidelity | The right evidence reaches the reasoner but its encoding does not carry the answer |
| E4 | Reasoning | Evidence is present, sufficient, and legible; the answer is still wrong |
| E5 | Faithfulness | The response is not calibrated to the evidence, in **either** direction |
 
**Design decisions locked this cycle:**
 
- **Boundary A for E1/E3.** E1 is *absence only* (empty text layer, skipped region).
  Everything about how faithfully present content is rendered is E3. This keeps the
  chain strictly preconditional and puts the fidelity transitions — the paper's most
  novel measurement — unambiguously in E3. Cost: E1 is thin, and the parser sweep sits
  in deployment rather than in E1.
- **E4 is "Reasoning", not "Integration".** E1–E3 are defined by the *state of the
  evidence*; naming E4 by a capability mixed two kinds of thing. E4 is now the residual
  after E1–E3 are excluded, with **integration as its dominant mechanism** and
  positional/adjacency effects folded in. Integration must stay distinct from E3 or the
  "bottleneck relocates" finding has nowhere to relocate to.
- **E5 is bidirectional.** Both fabricating on unanswerable questions *and* falsely
  abstaining when evidence is present and sufficient. The false-abstention half shares
  E4's preconditions and differs in the reasoner's disposition toward them.
- **Evidence is not assumed page-level.** §3 is written system-agnostically; page-level
  evidence is an artifact of MMLongBench-Doc's annotation, not of MP-VRDU. E4 keeps
  "pages" deliberately, since cross-page combination is what makes MP-VRDU distinct.
- **E1 and E2 need the reduction premise.** Neither mode exists for a system that can
  read the whole document, so §3's opener establishes that context length and attention
  dilution force reduction. Without it E1/E2 read as implementation choices.
---
 
## 3. Section plan and naming
 
| § | Title | Status |
|---|---|---|
| 1 | Introduction | **Not written.** Write last. |
| 2 | Related Work | **Drafted this cycle**, ~194 words. |
| 3 | Failure Modes in MP-VRDU | **Drafted this cycle**, ~570 words. Opener + E1–E5. |
| 4 | Methodology | **Drafted this cycle**, ~570 words, three subsections. |
| 5 | *(naming open)* | Movement 1. Attribution results. |
| 6 | *(naming open)* | Movement 2. Intervention results. |
| 7 | Discussion | Not written. |
| 8 | Limitations | Not written. |
 
**§4 subsections (locked):** 4.1 Attribution by construction · 4.2 Pipeline
configuration · 4.3 Data and evaluation. §4 must not forward-reference §5/§6.
 
**§5/§6 naming — open.** Candidates, in preference order:
1. **Where the Loss Occurs** / **What Moves It** — plain, parallel, avoids "analysis".
2. **Attribution** / **Intervention** — compact, matches §4.1 vocabulary, but collides
   with the §4.1 title unless that is renamed.
3. **Locating the Loss** / **Moving the Loss**.
Avoid "Analysis of Failure Modes": *analysis* is the word the paper spends its length
deflecting.
 
**Also open:** whether §6 splits forced levers (deployment) from chosen levers
(inference-time) as two subsections, or interleaves them by which loss they target.
 
**Where the intervention framing lives.** §3.2 (the lever surface) was **cut** — §3 is
now pure problem statement. The funnel from movement 1 to movement 2 currently appears
only in §4.1's opener. It needs a home in the **introduction's contributions**, or §6
opens by establishing its own frame. Flagged as unresolved.
 
---
 
## 4. Experiment & data status — master table
 
Legend: ✅ done + judged · ◧ partial/stranded · ⏳ planned (spec ready or trivial) ·
💡 in consideration (ranked in §5) · ❌ not started.
 
Mode column maps each experiment to the E1–E5 taxonomy; movement column to §5 or §6.
 
| Mode | Mv | Experiment | Table file | Status | Notes |
|---|---|---|---|---|---|
| E3 | 5 | Representation ladder (overall + doc_type) | `RQ1_Representation` | ✅ | T31.9/TL39.4/TLV56.8/V45.9 |
| E3 | 5 | Fidelity transitions **T→TL + TL→TLV**, by source | `RQ1_Representation` | ✅ | T→TLV dropped (composition of the two) |
| E3 | 5 | Fidelity transitions by **doc_type** | `RQ1_Representation` | ✅ | disjoint rows, unlike source |
| E1 | 5 | Scanned vs digital ladder | — | 💡 | **high**: motivates vision; data exists |
| E2 | 5 | Retrieval P/R/F1 (6 retrievers × k) | `Retrievers` | ✅ | backdrop |
| E2 | 5 | LOPO (sufficiency), 4 conds × 2 rankers | `RQ2_Selection` (a) | ⏳ | drop/keep × best/worst; ColQwen3+**BM25** |
| E2 | 5 | Distractor (robustness) — **+k design** | `RQ2_Selection` (b) | ⏳ | **REDESIGNED, see §4.1 below** |
| E2 | 5 | Real retriever-fed generation (G2 complete) | `RQ2_Attribution` | ◧ | ~36% pulled, judging in flight; H100 |
| E2 | 5 | Recall@k → accuracy@k curve | — | 💡 | **high**, depends on G2 completion |
| E5 | 5 | Distractor-only (gold absent) → fabricate? | — | 💡 | bridge E2→E5; cheap |
| E4 | 5 | Integration gap (S/M/M−S × domain × rung) | `RQ3_Integration` | ✅ | sign = M−S; 2 domains positive |
| E4 | 5 | Hop cross-tab (domain × rung × 1/2/3+) | — | ✅ built, ❌ **not used** | 3+ too thin (23/28 cells n<20) |
| E5 | 5 | Faithfulness × pool × **4 rungs** | `RQ3_Faithfulness` | ◧ | unanswerable TLV done; **rest pending** |
| E5 | 6 | Abstention prompt on answerable pool | reuse `RQ3_Faithfulness` | ⏳ | **false-abstention cost — now E5's second half** |
| E4 | 5 | Integration by source heterogeneity | — | 💡 | **high IF** evidence_source is per-page |
| E4 | 6 | Positional sensitivity of gold pages | — | 💡 | folded into E4; interleaving partly covers it |
| — | 6 | Quantization sweep (acc + weights) | `RQ4_Reasoner` | ✅ | quant free; weights col (peak-VRAM dropped) |
| — | 6 | Reasoner scale | `RQ4_Reasoner` | ◧ | Qwen 2/4/8B done; **32B pending** |
| — | 6 | **Matched budget: 8B-16bit vs 32B-4bit** | `RQ4_Reasoner` | ⏳ | **high**; needs 32B weights + 4-bit pass |
| — | 6 | Family: InternVL3-8B | `RQ4_Reasoner` | ✅ | trails Qwen3-VL-8B throughout |
| — | 6 | Family: **Llama-3.2-11B-Vision** | `RQ4_Reasoner` | ⏳ | cross-attn, not in-sequence vision tokens |
| E4 | 6 | **Qwen3-VL-8B-Thinking** | `RQ4_Reasoner` + `RQ5_Levers` | ⏳ | report **M−S**, not pooled acc |
| E1 | 6 | Parser/acquisition sweep | `RQ4_Reasoner` | ✅ | paddle/mineru/unlimited; deployment choice |
| — | 6 | Prefill + OOM frontier per rung | `RQ4_Cost` | ✅ | ~18× prefill on image rungs |
| E3 | 6 | Domain minimum-viable-representation | — | 💡 | existing data; links E3↔deployment |
| — | 6 | Max-runnable-length per budget | — | 💡 | reframe of OOM frontier |
| E3 | 6 | Resolution sweep | `RQ5_Levers` | ✅ | **pooled rows in**: TLV +7.6, V +13.7 |
| E4 | 6 | Interleaving TLVi (per-page) | `RQ5_Levers` | ⏳ | **hinge**; S=TLV by construction, run M only |
| E2 | 6 | Retrieval depth (from Distractor) | reuse `RQ2_Selection` | ⏳ | re-read k-sweep as a failing lever |
| E4 | 6 | CoT as a lever on integration | `RQ5_Levers` | ⏳ | same run as the E5 prompt mode |
| — | — | Human error annotation (~150–200 failures) | — | 💡 | validates auto-attribution; no GPU |
 
### 4.1 Distractor redesign — **+k, not pad-to-k** (decided this cycle)
 
**New design:** add *k* distractor pages **on top of** the gold set, for k = 1, 2, 3,
conditioned on gold count 1, 2, 3. A question with 1 gold page at k=1 sees 2 pages
total; 3 gold at k=3 sees 6.
 
**Old design (superseded):** pad the gold set with top-ranked non-evidence up to a
**constant** total k.
 
**Consequences to handle:**
- **The length control is gone.** The old design held total context constant so a drop
  was attributable to dilution rather than length. Under +k, total pages vary with both
  gold count and k, so a drop confounds dilution with length.
- **§4.1 of the draft must change.** It currently reads "padding the context with
  high-ranking non-evidence at a fixed page count, which holds context length constant
  so that a drop reflects dilution rather than length." That justification no longer
  holds. Rewrite before the run lands.
- **What it buys:** dilution *ratio* at fixed absolute distractor count, and the effect
  conditioned on gold count, which pad-to-k partly obscured.
- **No data exists for this design.** The existing k=0 cells survive; everything else
  needs the rerun. Cell count needs re-estimating (old pad-to-k estimate: ~18,528).
**Cell-count estimates for other planned runs** (H100, all 4 rungs): LOPO ≈ 2,988 cells
under one ranker, roughly double for the ColQwen3+BM25 pair. Inference-only; all six
retrievers' rankings already exist.
 
---
 
## 5. In-consideration experiments, ranked by promise
 
Ranked by (argument value ÷ cost). **Movement 2 is now the paper's centre of gravity, so
levers outrank additional attribution.**
 
1. **Interleaving TLVi (E4, movement 2)** — the hinge. Movement 2 currently has **one
   populated lever** (resolution) and **zero populated levers that resist**. The entire
   "some losses resist inference-time repair" half rests on this run plus CoT.
2. **G2 completion + recall@k→accuracy@k (E2)** — the actual selection-loss headline;
   without it E2 leans entirely on LOPO/Distractor proxies. Expensive (H100), scoped.
3. **CoT as a lever (E4, movement 2)** — second resisting-lever candidate; same run as
   the E5 prompt mode, so cheap relative to value.
4. **Scanned-vs-digital ladder (E1)** — cheap, existing data, distinct result: "no text
   layer → vision is not optional." Also the only thing that thickens E1. **NB confound:**
   scanned n concentrated in ~2 doc_types; must compare within-doc_type.
5. **Integration by source-heterogeneity (E4)** — sharpest integration claim. **Gated:**
   needs `evidence_source` resolvable per gold-page — UNVERIFIED, check first.
6. **Human error annotation** — validates the automatic attribution end-to-end; no GPU;
   highest-leverage for defending "is your attribution even correct?".
7. **Domain minimum-viable-representation** — existing data; "what you can afford to cut."
8. **Distractor-only / gold-absent (E2→E5 bridge)** — cheap.
9. **Max-runnable-length table** — reframe of OOM data.
**CoT scope decision (locked).** CoT enters as a **prompt preamble only** — a fourth
entry in `config.PROMPT_MODES`, a small code change, not a pipeline change. Multi-turn
CoT and page-by-page decomposition would break the one-cell-one-generation contract and
the cache key, and are **out of scope**. **ToT is out of scope** entirely. Two run-level
consequences:
- **Raise the 256-token decode cap for CoT cells.** CoT emits reasoning *then* answer; at
  256 tokens the answer may be truncated away, reading as an accuracy drop that is really
  a truncation artifact. `truncation_occurred`/`tokens_dropped` are input-side canaries
  and will **not** catch this.
- **Judge confound.** A CoT answer is long and reasoning-laden, so the judge sees a
  different artifact than the terse baseline. Either instruct a delimited final answer and
  extract before judging, or hand-check a CoT subsample. Folds into the §6 judge-FN check.
---
 
## 6. Cheap data-inspection checks that gate real decisions (no GPU)
 
These block claims we are about to build on. Resolve in one pass:
- [ ] **Is `evidence_source` per-page or per-question?** Gates the E4 heterogeneity split.
      If per-question, that experiment is impossible as designed.
- [ ] **Hop-count arithmetic.** `CODEBASE_GUIDE` gives 487 single / 358 multi; the dataset
      table gives 485 questions citing one page. 485+358 = 843 against 847 answerable.
      **Reconcile before any hop count appears in prose.** Currently avoided in §4.3 by
      omitting the numbers.
- [ ] **VRAM measurement**: peak_vram was single-device (device-0 only) on a 2×V100
      (32GB total, ~16GB/card) — this is why peak-VRAM columns were dropped for
      **model-weight footprint**. Confirm the weights numbers and whether the vision
      tower/embeddings stayed fp16 under 8/4-bit (changes the 4-bit figure). Belongs in
      the movement-2 deployment write-up, not §4.
- [x] **G3 retriever wording** — **RESOLVED**: the spec ran **bge-m3**; the BM25 caption
      was the `config.BASELINE` default the spec overrode. The `\flag{}` can be cleared.
- [ ] **Judge FN rate unmeasured** — needed for any "reasoning residual" (E4) claim and as
      the shield vs Graph-RAG "reasoning dominates". Hand-check a stratified subsample.
      **Also gates CoT**, whose verbose answers may be judged differently.
- [ ] **Answerable-side page selection for the faithfulness run** — if it uses **oracle**
      pages, the None row's 4 accuracy cells are free (the existing G1 ladder); if it
      matches G3's retrieved setup, they must be re-run. **Now doubly important:** E5's
      false-abstention half only has E4's preconditions if the pages are oracle. Pin this
      in the spec before launching.
- [ ] **Does LOPO's "drop worst" measure the reasoner or the annotation?** If accuracy
      barely moves, that gold page was carrying nothing — a claim about MMLongBench's gold
      set as much as about the model. Decide the framing *before* the run.
---
 
## 7. Paper goals & the reviewer threat model
 
**Goal:** a stage-resolved account of where MP-VRDU loses information **and what moves
those losses**, defensible as a contribution rather than an analysis.
 
**The central attack: "this is just an analysis paper / the contribution isn't enough."**
Not the "dataset analysis" charge (we propose no dataset) but the broader
"analysis-only, insufficient novelty" charge. Defences, in order:
1. **Movement 2 is the contribution.** It converts a map into guidance: these losses you
   can move cheaply, this one you cannot. This is the "so what" a reviewer wants.
   *(Superseded: the old defence was "five co-equal RQs distribute risk". That shield is
   gone with the RQ structure. The paper now has a thesis instead, which is stronger but
   less hedged.)*
2. **The genre has precedent.** Component-wise empirical studies that hold a pipeline
   fixed and propose nothing are published (`systemslevelrag2026`, `notallrags2026`,
   `dissectingagenticrag2026`). Cite them in §2 rather than defending the paper type
   from first principles.
3. **Novel measurements no neighbour can produce**: within-question fidelity transitions
   (selected-but-unusable), the LOPO/Distractor sufficiency-vs-robustness pair, the
   integration-gap-widens-with-representation result.
4. **Generalisation guard** (§1): retriever is instrument, not object → not RAG-only.
**Secondary attacks & where they're handled:**
- *"Multi-hop is just harder, not an integration failure"* → the **precondition chain**
  is the structural rebuttal (E4 is only reached once E1–E3 are excluded); LOPO is the
  empirical one.
- *"Your 'integration' is an input-format artifact"* → interleaving/TLVi is the test.
- *"Vision>text is known"* → demoted to confirmatory/instrument-validation.
- *"Reasoning dominates (Graph-RAG)"* → different setting (text multi-hop, easy
  retrieval, no acquisition or fidelity nodes); needs the judge-FN check.
- *"DocScope already does oracle stage-attribution on long documents"* → **new, see §8.**
- *"Only one dataset (MMLongBench)"* → state the DUDE/SlideVQA/ChartQA contamination
  caveat; scope claims accordingly.
**Movement-2 hedging rule (important).** Only **one** lever is populated (resolution,
+7.6 TLV / +13.7 V) and it *works*. Zero populated levers *resist*. Until interleaving
and CoT land, write movement 2 as **"which levers reach which losses"** — directional
findings — never as "we show integration resists repair". A flat lever is a finding, and
saying so is what makes the hedge structural rather than tentative.
 
---
 
## 8. Prior work & positioning
 
**`RW.txt` is the authoritative positioning file.** It states what each paper owns, the
differentiator, and lists nine papers deliberately cut with reasons. Do not re-add ColPali,
VisRAG, VisDoM, M3DocRAG, or the agent-failure cluster to the positioning argument;
they are furniture or out of scope.
 
**Threats, in order:**
 
- **DocScope** (`feng2026docscope`) — **upgraded to nearest neighbour this cycle.** Earlier
  notes wrongly described it as post-hoc annotation. It is not: it runs a **cumulative
  oracle study** supplying gold pages → gold regions → gold facts to four models, finding
  fact extraction (not region grounding) the dominant bottleneck. So "we use an oracle to
  isolate stages on long VRDU documents" is **not** a clean novelty claim.
  **Surviving differentiators:** (i) the document is never re-encoded — what varies is how
  much *annotation* is supplied, so E1 and E3 fall outside their surface entirely;
  (ii) it requires trajectory-emitting models, making it a model benchmark rather than a
  pipeline analysis; (iii) **no intervention half** — it locates and stops;
  (iv) cumulative, not factorial. One-sentence version: *they ask which stage of a model's
  reasoning trajectory is weakest; we ask which stage of an evidence pipeline discards the
  answer and what recovers it.*
- **OHRBench** (`zhang2025ohrbench`) — cascading OCR loss. **Corrected differentiator:**
  it is *not* "synthetic vs real". Their controlled noise injection along semantic and
  formatting axes is arguably better-controlled than a real-parser swap. Our actual
  advantage is the **image channel**: a text-only pipeline has nothing that can compensate
  for parser loss, so the compensation effect our TL→TLV transitions measure cannot be
  observed in their setting at all.
- **UniDoc-Bench** (`peng2025unidocbench`) — owns "which modality wins" (fusion beats
  unimodal and joint, unified protocol). **Concede it outright:** our TLV > V agrees with
  their fusion finding. Conceding costs nothing and disarms them. Their questions cite few
  pages, so cross-page combination (E4) is absent by construction.
- **Graph-RAG** (`zarrinkia2026reasoningbottleneck`) — reasoning dominates once coverage is
  high, opposite our likely direction. Explain by setting; gate on judge-FN.
**Partial overlaps — do NOT claim as novel:** Facet-RAG (`elchafei2026facetrag`) and
TaSR-RAG (`sun2026tasrrag`) establish that diagnostic-intervention attribution exists.
Cut from §2 for space; restore if a reviewer raises method novelty.
 
**The predecessor survey** (`xu2026managing`, first-authored) frames MP-VRDU as evidence
management and is the direct parent. Cites the reduction-is-forced claim in §3.
 
**Load-bearing intro sentence:** *Prior work attributes error at individual nodes in
adjacent settings — OCR loss in text RAG, retrieval-vs-reasoning in text multi-hop,
trajectory stages in long-document QA — but none measures the complete loss surface of a
multi-page visual pipeline where these nodes coexist, and none asks which losses move.*
 
---
 
## 9. Draft status
 
**Drafted this cycle (all complete rewrites, all pending paste into `acl_latex.tex`):**
- **§2 Related Work** — ~194 words, three paragraphs: genre precedent, four direct
  overlaps (OHRBench, UniDoc-Bench, Graph-RAG, DocScope), the gap. Opens on the finding,
  not a definition of empirical analysis.
- **§3 Failure Modes in MP-VRDU** — ~570 words. One opening paragraph (sequential
  conditions → accuracy localises nothing → chain → reduction premise), then five
  `\paragraph{}`-labelled modes. Citations resolved against `custom.bib`.
- **§4 Methodology** — ~570 words, three subsections. 4.1 carries all justification
  (chain dictates measurement, oracle as the instrument, fixed-stage principle,
  generalisation guard); 4.2 is compressed configuration; 4.3 is data and evaluation
  pointing at the dataset table.
**Not written:** §1 Introduction (last), §5, §6, §7 Discussion, §8 Limitations, Abstract.
 
**Superseded and to be deleted from the draft:** the old §3 (`Problem Statement?
(Analysis Framework or Attribution Framework)`) and its five RQ statements; the old
Methodology; the old one-paragraph Related Work.
 
### Blocking issues in `acl_latex.tex`
 
1. **`\flag{}` is used but never defined** — the draft does not compile. Add a definition
   or strip the uses.
2. **§1 renders empty.** The intro's second half exists only as a commented-out block.
3. **`RQ2_Retrieval.tex` is commented out** but `\ref{tab:RQ2_retrieval}` is still cited
   → compiles to `??`.
4. **`RQ3_Integration.tex`'s `\footnotesize` note is commented out**, so the concession
   about Academic paper and Brochure running positive is invisible in the PDF. One-character
   fix.
5. **Labels that do not exist yet:** `app:config` (referenced by §4.2), and whatever §5/§6
   are eventually labelled.
6. **Bibkeys missing from `custom.bib`:** `systemslevelrag2026`, `notallrags2026`,
   `dissectingagenticrag2026`, `peng2025unidocbench`, `feng2026docscope`.
7. **Bibkeys resolved this cycle** — add these entries (BibTeX verified against arXiv):
   `cui2025paddleocrvl` (2510.14528), `niu2025mineru25` (2509.22186, **note: first author
   is Niu, not Wang, and it is 2025 not 2024** — the old `wang2024mineru` key was wrong on
   both), `bai2025qwen3vl` (2511.21631). Beware collision with the existing `bai2025qwen2`
   (Qwen2.5-VL) — different model, and its title field is malformed
   (`Qwen2. 5-VL Technical Report (No. arXiv: 2502.13923). arXiv`).
8. **Cross-reference sweep needed.** With the RQ structure gone, every "(RQ1, acquisition)"
   parenthetical, "the integration deficit of RQ1" (which was also wrong — it was RQ3), and
   `\ref{sec:rqs}` needs rewriting.
### Prose fixes still outstanding in the surviving §5 text
 
- **§5.1 claims the ladder ordering holds**; Academic paper inverts it (39.2→37.7→42.0→31.4).
  Say it holds in aggregate and inverts on Academic paper — the T→TL regression data now
  *explains* why (9.4% R→W vs 4.7% W→R).
- **§5.3 claims multi < single at every rung**; Academic paper and Brochure run positive.
- **§5.5 asserts a retrieval-depth null result** ("raises recall, does not close the gap")
  citing `tab:RQ2_distractor`, but retrieval depth is ⏳ and the distractor arm is now being
  **redesigned**. This is currently over-claiming on data that will not exist in that form.
### Conventions (locked)
 
M−S for integration gaps (negative = multi worse); `booktabs` + `\setlength{\tabcolsep}` +
bold headers + right-aligned numerics; captions are single short sentences with detail in a
`\footnotesize` note **above** the caption; per-cell n shown where OOM attrition thins cells
(dagger the unreliable); weight footprint replaces peak-VRAM everywhere; no vertical rules;
**no em-dashes**; `/academic-humanizer` standards (see below).
 
**`/academic-humanizer` skill** — created this cycle, combining the humanizer AI-pattern
detection with academic register (subordination over short declaratives, hedging matched to
evidential strength, no elegant variation on technical terms, claim-strength discipline).
Supersedes bare `/humanizer` for all paper prose.
 
---
 
## 10. Brief history (how we got here)
 
- **Started** as a 3-node then 5-node error taxonomy, one big "error attribution" RQ
  carrying half the paper, framed as "the bottleneck relocates."
- **Reframed** to dodge the "just dataset analysis" charge: confirmatory results demoted
  to instrument-validation; surprising results re-centered.
- **Added** LOPO and Distractor, which bracket the selection stage and defend the
  integration finding against "multi-hop is just harder."
- **Split** into five co-equal RQs as a defensive structure (no single finding could sink
  the paper).
- **Consolidated the tables**: nine files down from eleven. Moved the parser sweep to
  deployment. Added T→TL and per-doc_type fidelity transitions, turning the Academic-paper
  ladder inversion from an anomaly into a **mechanism**.
- **Collapsed the five RQs into two movements** (supervisor, this cycle). Five co-equal RQs
  were a shield that cost the paper a thesis. Movement 1 locates, movement 2 moves. The RQs
  survive as §5/§6 subsections, not as research questions.
- **Rebuilt §3 as a failure-mode taxonomy** (E1–E5, precondition chain, Boundary A,
  E4 renamed from Integration to Reasoning, E5 made bidirectional, evidence generalised
  off page-level). Cut §3.2 (the lever surface) entirely.
- **Rewrote §4** so justification leads and configuration follows, with the oracle promoted
  from a clause to the instrument the framework rests on.
- **Rewrote §2** around genre precedent plus four direct threats, after discovering the
  empirical-pipeline-analysis genre is established and that DocScope is closer than
  previously assessed.
- **Redesigned the distractor arm** from pad-to-k to +k (this cycle), trading the
  context-length control for dilution-ratio-at-fixed-count conditioned on gold count.
---
 
## 11. What a new chat should do first
 
1. Read this README, then `RW.txt` if touching related work, then `CODEBASE_GUIDE.md` for
   any implementation question.
2. If planning experiments: consult §4 (status) and §5 (ranked candidates); don't re-derive
   configs — they're in `CODEBASE_GUIDE.md`.
3. If writing: match §9 conventions and use `/academic-humanizer`. Tables live in `tables/`,
   `\input`-ed; captions one sentence; M−S sign; no em-dashes.
4. **Never assert a number from memory.** Trace it to `all_tables.md` or flag it.
5. If a claim depends on a §4-◧/⏳ experiment or a §6 check, write it as a conditional and
   flag it rather than asserting it.
6. **Respect the movement-2 hedging rule** (§7): one populated lever, and it works. Do not
   write "integration resists repair" until interleaving or CoT lands.
7. Bibkeys: `custom.bib` is the source of truth; use a plausible key if one is missing and
   flag it for manual resolution.
